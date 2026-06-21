"""GPU-batched Orbit Wars simulator (notebook cells 13/15/17/19 + settle/settle_n from 23).

Bit-exact to the official engine including official comets (see memory `gpuenv-kaggle-parity`).
All state lives as ``(B, *)`` tensors on ``cfg.device``; one :meth:`GpuEnv.step` advances every
parallel env one tick. The free functions ``env_encode`` / ``env_step`` / ``opponent_action`` etc.
operate on a :class:`GpuEnv`; functions that read run-tunable hyperparameters take ``cfg`` (the
pure geometry/feature helpers do not -- they read only :mod:`orbit_wars_v12.constants`).
"""
import math

import numpy as np
import torch

from .constants import (BIG, BOARD_SIZE, CENTER, COMET_PRODUCTION, COMET_RADIUS, DIAG_HALF, DTYPE,
                        N_THREAT_FEATS, N_THREAT_FLEETS, PI, PLANET_CAP,
                        ROTATION_RADIUS_LIMIT, SHIP_SPEED, SUN_RADIUS, THREAT_ETA_SCALE,
                        THREAT_MAX_SPEED, fleet_speed_t, ship_log_t)

# opponent_action codes by league "kind" (the two notebook names are the same mapping).
_OPP_BY_KIND = {"random": 0, "starter": 1, "noop": 2, "medium": 3, "greedy": 4, "intermediate": 5}
_KIND2OPP = _OPP_BY_KIND

# seat-swap board rotations (k*90deg) mapping ego -> each seat; see _apply_rot_aug.
_SEAT_ROT = {2: (0, 2), 4: (0, 1, 2, 3)}
ROTATE_AUGMENT = True   # legacy per-env RANDOM rotation fallback (used if _rot_aug set w/o _rot_k)


class GpuEnv:
    def __init__(self, cfg, planet_cap=None, fleet_cap=None, episode_steps=None, ship_speed=None):
        self.cfg = cfg
        self.Ec = PLANET_CAP if planet_cap is None else planet_cap
        self.Fc = cfg.FLEET_CAP if fleet_cap is None else fleet_cap
        self.T = cfg.EPISODE_STEPS if episode_steps is None else episode_steps
        self.vmax = SHIP_SPEED if ship_speed is None else ship_speed
        self.dev = cfg.device
        self.B = 0
        self.n_players = 2

    def reset(self, worlds, n_players=2):
        cfg = self.cfg
        self.n_players = int(n_players)
        B = len(worlds)
        self.B = B
        Ec, Fc = self.Ec, self.Fc
        dev = self.dev
        z = lambda *s: torch.zeros(s, dtype=DTYPE, device=dev)
        # planets (B, Ec)
        self.p_alive = z(B, Ec)
        self.p_owner = torch.full((B, Ec), -1.0, dtype=DTYPE, device=dev)
        self.p_x = z(B, Ec); self.p_y = z(B, Ec); self.p_radius = z(B, Ec)
        self.p_ships = z(B, Ec); self.p_prod = z(B, Ec); self.p_is_comet = z(B, Ec)
        self.p_init_x = z(B, Ec); self.p_init_y = z(B, Ec); self.p_rotates = z(B, Ec)
        self.p_comet_vx = z(B, Ec); self.p_comet_vy = z(B, Ec)   # comet straight-line velocity
        # fill from worlds (CPU numpy then upload once)
        pa = np.zeros((B, Ec), np.float32); po = np.full((B, Ec), -1.0, np.float32)
        px = np.zeros((B, Ec), np.float32); py = np.zeros((B, Ec), np.float32)
        pr = np.zeros((B, Ec), np.float32); ps = np.zeros((B, Ec), np.float32)
        pp = np.zeros((B, Ec), np.float32); pix = np.zeros((B, Ec), np.float32)
        piy = np.zeros((B, Ec), np.float32); prot = np.zeros((B, Ec), np.float32)
        angv = np.zeros((B,), np.float32)
        for b, w in enumerate(worlds):
            pls = w["planets"][:Ec]
            for i, pl in enumerate(pls):
                pid, owner, x, y, rad, sh, prod = pl
                pa[b, i] = 1.0; po[b, i] = float(owner)
                px[b, i] = x; py[b, i] = y; pr[b, i] = rad
                ps[b, i] = sh; pp[b, i] = prod
                pix[b, i] = x; piy[b, i] = y
                rr = math.hypot(x - CENTER, y - CENTER)
                prot[b, i] = 1.0 if (rr + rad < ROTATION_RADIUS_LIMIT) else 0.0
            angv[b] = w["angular_velocity"]
            if self.n_players >= 4:               # official 4p seating: owners 0..3, 10 ships each
                hb = int(w.get("home_base", -1))
                if 0 <= hb and hb + 3 < Ec:
                    for j in range(4):
                        po[b, hb + j] = float(j); ps[b, hb + j] = 10.0
        t = lambda a: torch.from_numpy(a).to(dev)
        self.p_alive = t(pa); self.p_owner = t(po); self.p_x = t(px); self.p_y = t(py)
        self.p_radius = t(pr); self.p_ships = t(ps); self.p_prod = t(pp)
        self.p_init_x = t(pix); self.p_init_y = t(piy); self.p_rotates = t(prot)
        self.p_is_comet = z(B, Ec)
        # fleets (B, Fc)
        self.f_alive = z(B, Fc); self.f_owner = z(B, Fc); self.f_x = z(B, Fc)
        self.f_y = z(B, Fc); self.f_angle = z(B, Fc); self.f_ships = z(B, Fc)
        self.f_seq = z(B, Fc)
        # official comet schedule (padded waypoint playback)
        NS = len(cfg.COMET_SPAWN_STEPS)
        if cfg.COMET_OFFICIAL and worlds and ('comet_paths' in worlds[0]):
            cp = np.stack([w['comet_paths'] for w in worlds])           # (B,NS,4,L,2)
            cl = np.stack([w['comet_len'] for w in worlds])             # (B,NS)
            cs = np.stack([w['comet_ships'] for w in worlds])           # (B,NS)
            self.c_paths = torch.from_numpy(cp).to(dev)
            self.c_len = torch.from_numpy(cl).to(dev)
            self.c_ships = torch.from_numpy(cs).to(dev)
            self.c_slot = torch.full((B, NS, 4), -1, dtype=torch.long, device=dev)
        else:
            self.c_paths = None
        # per-env
        self.ang_vel = t(angv)
        self.step_ct = torch.zeros(B, dtype=DTYPE, device=dev)
        self.done = torch.zeros(B, dtype=DTYPE, device=dev)
        if getattr(self, '_rot_aug', False):       # seat-swap aug (training env only); set by collect_ppo
            _apply_rot_aug(self)
        self.p_x_prev = self.p_x.clone(); self.p_y_prev = self.p_y.clone()   # one-step prev pos -> finite-diff body velocity


# ---- seat-swap augmentation helpers (training env only) --------------------------------------
# collect_ppo tiles the worlds P-fold and stamps a per-env rotation index env._rot_k (seat-block
# b -> _SEAT_ROT[P][b]); GpuEnv.reset's hook (above) rotates ALL absolute-coord state by k*90deg
# about CENTER. The board is 4-fold rotationally symmetric, so this exactly moves the ego (pid 0)
# onto each seat's geometry -- the absolute-coord seat overfit is averaged out. eval/recal/gauntlet
# build their own canonical envs (no _rot_aug), so ratings stay comparable.
def _rot90_about_center(x, y, k):
    """Rotate (x,y) about CENTER by k*90deg (k int tensor broadcastable to x). +90: (x,y)->(-y,x)."""
    dx = x - CENTER; dy = y - CENTER
    nx = torch.where(k == 0, dx, torch.where(k == 1, -dy, torch.where(k == 2, -dx, dy)))
    ny = torch.where(k == 0, dy, torch.where(k == 1, dx, torch.where(k == 2, -dy, -dx)))
    return CENTER + nx, CENTER + ny


def _apply_rot_aug(env):
    """Rotate ALL absolute-coord state by env._rot_k (per-env k*90deg about CENTER): planet
    positions + orbit anchors + comet waypoints. radius/ownership/ang_vel are rotation-invariant;
    fleets are empty at reset. If _rot_k is unset, fall back to a per-env RANDOM k (legacy aug)."""
    B = env.B
    k = getattr(env, "_rot_k", None)
    if k is None:
        k = torch.randint(0, 4, (B, 1), device=env.dev)
    env.p_x, env.p_y = _rot90_about_center(env.p_x, env.p_y, k)
    env.p_init_x, env.p_init_y = _rot90_about_center(env.p_init_x, env.p_init_y, k)
    if getattr(env, "c_paths", None) is not None:          # comet waypoints (B,NS,4,L,2)
        kk = k.view(B, 1, 1, 1)
        rx, ry = _rot90_about_center(env.c_paths[..., 0], env.c_paths[..., 1], kk)
        env.c_paths = torch.stack([rx, ry], dim=-1)


def _fleet_target_core(fx, fy, fang, fships, vmax, p_x, p_y, p_radius, p_alive, p_x_prev, p_y_prev):
    '''Pure-tensor core of fleet_target_batch (NO env object) -> torch.compile-friendly. Identical
    math; the planet state + previous positions (for the finite-diff planet velocity) arrive as
    tensor args instead of being read off the env (which dynamo cannot trace through cleanly).'''
    spd = fleet_speed_t(fships, vmax)                            # (B,M) fleet speed
    fvx = (torch.cos(fang) * spd).unsqueeze(2)                   # (B,M,1) fleet velocity vector
    fvy = (torch.sin(fang) * spd).unsqueeze(2)
    pxr = p_x.unsqueeze(1); pyr = p_y.unsqueeze(1)               # (B,1,Ec) planet position
    pvx = (p_x - p_x_prev).unsqueeze(1); pvy = (p_y - p_y_prev).unsqueeze(1)   # (B,1,Ec) planet velocity (finite diff)
    rr = (p_radius * p_radius).unsqueeze(1)
    alive = (p_alive > 0.5).unsqueeze(1)
    rx = fx.unsqueeze(2) - pxr                  # (B,M,Ec) relative position (fleet - planet)
    ry = fy.unsqueeze(2) - pyr
    wx = fvx - pvx; wy = fvy - pvy              # (B,M,Ec) relative velocity (fleet - planet)
    a = wx * wx + wy * wy
    b = 2.0 * (rx * wx + ry * wy)
    c = rx * rx + ry * ry - rr
    disc = b * b - 4.0 * a * c
    sq = torch.sqrt(disc.clamp_min(0.0))
    t_hit = (-b - sq) / (2.0 * a).clamp_min(1e-9)               # first contact = smaller root (TICKS)
    t_hit = torch.where(c <= 0.0, torch.zeros_like(t_hit), t_hit)   # fleet already inside the disk -> now
    hit = (disc >= 0.0) & (t_hit >= 0.0) & alive & (a > 1e-9)
    tvals = torch.where(hit, t_hit, torch.full_like(t_hit, BIG))
    best_t, best_e = tvals.min(2)               # (B,M) earliest contact TIME + planet
    any_hit = best_t < BIG
    # sun occlusion: is the (static) sun reached BEFORE the target? compare in TICKS
    sx = fx - CENTER; sy = fy - CENTER
    dxx = torch.cos(fang); dyy = torch.sin(fang)
    tcs = -(sx * dxx + sy * dyy)
    sperp2 = sx * sx + sy * sy - tcs * tcs
    sr = SUN_RADIUS * SUN_RADIUS
    t_sun = (tcs - torch.sqrt((sr - sperp2).clamp_min(0.0))) / spd.clamp_min(1e-6)   # ticks to sun entry
    sun_block = (tcs >= 0.0) & (sperp2 <= sr) & (t_sun >= 0.0) & (t_sun < best_t)
    valid = any_hit & (~sun_block)
    tgt = torch.where(valid, best_e, torch.full_like(best_e, -1))             # SUN-AWARE (sun-blocked -> -1)
    eta = torch.where(valid, best_t, torch.zeros_like(best_t))                # best_t already in TICKS
    app_tgt = torch.where(any_hit, best_e, torch.full_like(best_e, -1))       # APPARENT target (ignores the sun)
    app_eta = torch.where(any_hit, best_t, torch.zeros_like(best_t))          # -> sun-bound fleets still attributed here
    return tgt, eta, app_tgt, app_eta


def fleet_target_batch(env, fx, fy, fang, fships, vmax):
    '''fx,fy,fang,fships: (B,M). Returns (tgt, eta, app_tgt, app_eta), each (B,M): tgt/eta = the
    SUN-AWARE target (a fleet whose path is absorbed by the sun -> -1); app_tgt/app_eta = the APPARENT
    target ignoring the sun (the planet the ray points at) -- the threat encoder uses the apparent
    target so a sun-bound fleet still SHOWS UP in the threat list of the planet it is aimed at.
    Thin env wrapper around _fleet_target_core (resolves the previous-position finite-diff fallback).'''
    p_x_prev = getattr(env, "p_x_prev", env.p_x); p_y_prev = getattr(env, "p_y_prev", env.p_y)
    return _fleet_target_core(fx, fy, fang, fships, vmax, env.p_x, env.p_y, env.p_radius, env.p_alive,
                              p_x_prev, p_y_prev)


def _encode_core(ego, na, T, vmax, p_alive, p_owner, p_x, p_y, p_radius, p_is_comet, ang_vel,
                 p_prod, p_ships, f_x, f_y, f_angle, f_ships, f_alive, f_owner, f_seq, step_ct,
                 p_x_prev, p_y_prev):
    """Pure-tensor core of env_encode (NO env object) -> torch.compile-friendly. ``ego``/``na``/``T``/
    ``vmax`` are python scalars (compile specializes one graph per distinct value -- a tiny, fixed set:
    ego in 0..3, na in {2,4}, T in {200,500}); everything else is a tensor. Identical math to the
    wrapper below; shapes + device are read off the tensors instead of the env object."""
    B, Ec = p_owner.shape
    Fc = f_owner.shape[1]
    dev = p_owner.device
    alive = p_alive > 0.5
    av = p_alive
    owner = p_owner
    dxc = p_x - CENTER; dyc = p_y - CENTER
    dist = torch.sqrt(dxc * dxc + dyc * dyc)
    comet = p_is_comet > 0.5
    rotating = (~comet) & ((dist + p_radius) < ROTATION_RADIUS_LIMIT)
    angv = ang_vel.unsqueeze(1)
    vmag = torch.where(rotating, angv.abs() * dist, torch.zeros_like(dist))
    cw = torch.where(rotating,
                     torch.where(angv >= 0.0, torch.ones_like(dist), -torch.ones_like(dist)),
                     torch.zeros_like(dist))
    # ego-relative seat ONE-HOT: rel = (owner - ego) mod n_players -> channel rel fires;
    # neutral = all-zero. Channel 0 = self; channels 1..3 = the other seats in seat order
    # (rotation-equivariant: the same relative seat always lands in the same channel).
    rel = owner - float(ego)
    rel = rel - na * torch.floor(rel / na)
    _owned = owner >= 0.0
    own_oh = [(_owned & (rel == float(k))).to(DTYPE) for k in range(4)]
    b0 = p_x / BOARD_SIZE
    b1 = p_y / BOARD_SIZE
    b2 = vmag / THREAT_MAX_SPEED
    b3 = cw
    b4 = p_radius / 3.0
    b5 = p_prod / 5.0
    b6 = p_ships  # RAW garrison (per-planet ship-log compression removed; un-normalized integer count)
    b7a, b7b, b7c, b7d = own_oh   # [self, enemy+1, enemy+2, enemy+3]; neutral all-zero
    b8 = dist / DIAG_HALF
    b9 = comet.to(DTYPE)
    actable = (owner == float(ego)) & alive & (p_ships > 0.0)
    b10 = actable.to(DTYPE)
    body = torch.stack([b0, b1, b2, b3, b4, b5, b6, b7a, b7b, b7c, b7d, b8, b9, b10], 2)  # (B,Ec,14)

    # threat: per planet, the TOP N_THREAT_FLEETS inbound fleets BY SIZE. Attribution uses the APPARENT
    # target (the planet the ray points at, IGNORING the sun), so a fleet that will be absorbed by the sun
    # still SHOWS UP in its target's threat list (it is genuinely aimed here).
    _t, _e, app_tgt, app_eta = _fleet_target_core(f_x, f_y, f_angle, f_ships, vmax, p_x, p_y, p_radius,
                                                  p_alive, p_x_prev, p_y_prev)
    falive = f_alive > 0.5
    # fleet ownership = 4-channel EGO-RELATIVE seat ONE-HOT [self, enemy+1, +2, +3] (rotation-equivariant;
    # 2p only lights channels 0..1; empty/dead slots all-zero, masked by `valid` below).
    frel = f_owner - float(ego)
    frel = frel - na * torch.floor(frel / na)               # (B,Fc) ego-relative seat
    fraw = f_ships.abs()                                    # RAW inbound-fleet size (log compression is legacy)
    slot = torch.arange(Ec, dtype=DTYPE, device=dev).view(1, Ec, 1)
    tgtf = app_tgt.to(DTYPE).unsqueeze(1)                    # (B,1,Fc) APPARENT target (sun-bound fleets included)
    targeting = (tgtf == slot) & (app_tgt >= 0).unsqueeze(1) & falive.unsqueeze(1)  # (B,Ec,Fc)
    W = float(1 << 20)
    absf = f_ships.abs().unsqueeze(1)
    seqf = f_seq.unsqueeze(1)
    key_big = torch.where(targeting, absf * W - seqf, torch.full((1,), -BIG, device=dev))
    big_top_v, idx = torch.topk(key_big, N_THREAT_FLEETS, 2, largest=True)   # TOP 30 BY SIZE (seq tie-break)
    valid = big_top_v > -BIG * 0.5
    def pick(src):
        return src.unsqueeze(1).expand(B, Ec, Fc).gather(2, idx)
    zt = torch.zeros(B, Ec, N_THREAT_FLEETS, device=dev)
    own_ch = [torch.where(valid, pick((frel == float(k)).to(DTYPE)), zt) for k in range(4)]  # [self,e+1,e+2,e+3]
    etf = torch.where(valid, pick(app_eta) / THREAT_ETA_SCALE, zt)
    slg = torch.where(valid, pick(fraw), zt)
    threat = torch.stack(own_ch + [etf, slg], 3).reshape(B, Ec, N_THREAT_FEATS * N_THREAT_FLEETS)  # (B,Ec,180)

    entities = torch.cat([body, threat], 2) * av.unsqueeze(2)   # zero dead slots
    entity_mask = av
    action_mask = b10

    # globals (B,10)
    def psum(mask):
        return (p_ships * mask.to(DTYPE)).sum(1)
    mine = (owner == float(ego)) & alive
    en = (owner >= 0.0) & (owner != float(ego)) & alive    # ALL opponents pooled
    neu = (owner < 0.0) & alive
    my_ships = psum(mine); en_ships = psum(en)
    my_pl = mine.to(DTYPE).sum(1); en_pl = en.to(DTYPE).sum(1); neu_pl = neu.to(DTYPE).sum(1)
    npl = av.sum(1); total = npl.clamp_min(1.0)
    my_fleet = (f_ships * (f_owner == float(ego)).to(DTYPE) * falive.to(DTYPE)).sum(1)
    en_fleet = (f_ships * (f_owner != float(ego)).to(DTYPE) * falive.to(DTYPE)).sum(1)
    ME = 40.0
    g0 = step_ct / max(1, T)
    g1 = ang_vel * 10.0
    g2 = ship_log_t(my_ships); g3 = ship_log_t(en_ships)
    g4 = my_pl / total; g5 = en_pl / total; g6 = neu_pl / total
    g7 = ship_log_t(my_fleet); g8 = ship_log_t(en_fleet)
    g9 = torch.minimum(npl, torch.full_like(npl, ME)) / ME
    globals_ = torch.stack([g0, g1, g2, g3, g4, g5, g6, g7, g8, g9], 1)
    return entities, entity_mask, action_mask, globals_


# The compiled handle is installed by maybe_compile_encode(cfg) at train start (USE_COMPILE); until
# then env_encode runs _encode_core EAGERLY -- so eval/league/MCTS paths are unaffected unless compiled.
_ENCODE_CORE = _encode_core


def env_encode(env, ego=0):
    """v9: ownership is a 4-channel EGO-RELATIVE seat ONE-HOT [self, enemy seat+1, seat+2,
    seat+3] (neutral = all-zero), so the policy can tell WHICH opponent owns a body with no
    ordinal fake-magnitude. In 2p only the first two channels ever fire, so 4p generalizes the
    same input space; a dead seat's channel simply stops firing. The GLOBAL enemy aggregates
    still pool ALL opponents (everyone-vs-me totals).

    Thin env wrapper: unpacks the env tensors + scalars and calls _encode_core (compiled in place by
    maybe_compile_encode, else eager). Keeping the heavy math in a pure-tensor function is what lets
    torch.compile trace it -- dynamo cannot follow the env object's getattr-with-default accesses."""
    na = float(getattr(env, "n_players", 2))
    p_x_prev = getattr(env, "p_x_prev", env.p_x); p_y_prev = getattr(env, "p_y_prev", env.p_y)
    # The encoder never needs grad (its output is stored into the rollout buffers as constants; the
    # update re-runs net(...) on those, not on this output). Forcing no_grad at the dispatch boundary
    # keeps grad_mode CONSTANT across every caller (rollout ego/opp, league recal, eval, MCTS) so the
    # torch.compile'd _encode_core never recompiles on a grad_mode guard flip (was thrashing the
    # recompile_limit when the ungated ego encode ran grad-on between no_grad opponent encodes).
    with torch.no_grad():
        return _ENCODE_CORE(int(ego), na, int(env.T), env.vmax, env.p_alive, env.p_owner, env.p_x, env.p_y,
                            env.p_radius, env.p_is_comet, env.ang_vel, env.p_prod, env.p_ships, env.f_x,
                            env.f_y, env.f_angle, env.f_ships, env.f_alive, env.f_owner, env.f_seq,
                            env.step_ct, p_x_prev, p_y_prev)


def maybe_compile_encode(cfg):
    """Install a torch.compile'd _encode_core as the handle env_encode dispatches to (idempotent,
    called once at train start when cfg.USE_COMPILE). The pure core has only tensor args + a few
    python scalars (ego/na/T/vmax), so it traces with NO graph breaks -- fusing the ~50-op encoder the
    launch-bound rollout calls 2-4x/step. Robust: torch.compile is LAZY, so a backend failure surfaces
    on the FIRST call, not at compile() time -- a one-shot guard catches it and permanently reverts to
    eager, so an absent/broken compiler can never kill a training run."""
    global _ENCODE_CORE
    if not getattr(cfg, "USE_COMPILE", False) or getattr(_ENCODE_CORE, "_ow_compiled", False):
        return _ENCODE_CORE
    try:
        compiled = torch.compile(_encode_core, mode=getattr(cfg, "COMPILE_MODE", "default"))
    except Exception as e:
        print("  [compile] env_encode not compiled (%r) -> eager" % e)
        return _ENCODE_CORE

    def _guard(*a, **k):                  # first call triggers the (lazy) backend; fall back on failure
        global _ENCODE_CORE
        try:
            out = compiled(*a, **k)
            _ENCODE_CORE = compiled       # success -> dispatch straight to the bare compiled core thereafter
            return out
        except Exception as e:
            print("  [compile] env_encode failed at runtime (%r) -> reverting to eager" % e)
            _ENCODE_CORE = _encode_core
            return _encode_core(*a, **k)
    _guard._ow_compiled = True
    _ENCODE_CORE = _guard
    return _ENCODE_CORE


def launch_fleets(env, owner, from_slot, angle, ships, commit, seq):
    '''Append committed launches (all (B,L)) into the fleet pool, distinct free slots.'''
    B, Fc = env.B, env.Fc
    dev = env.dev
    L = from_slot.shape[1]
    free = env.f_alive < 0.5
    freef = free.to(DTYPE)
    fr = torch.cumsum(freef, 1) - 1.0
    n_free = freef.sum(1)
    idx = torch.where(free, fr.long(), torch.full_like(fr.long(), Fc))
    rank_to_slot = torch.full((B, Fc + 1), Fc, dtype=torch.long, device=dev)
    slot_src = torch.arange(Fc, dtype=torch.long, device=dev).unsqueeze(0).expand(B, Fc)
    rank_to_slot.scatter_(1, idx, slot_src)
    rank_to_slot = rank_to_slot[:, :Fc]
    commitb = commit > 0.5
    crank = (torch.cumsum(commit.to(DTYPE), 1) - 1.0).long()
    place = commitb & (crank < n_free.unsqueeze(1).long())
    gslot = rank_to_slot.gather(1, crank.clamp(0, Fc - 1))
    dump = torch.full_like(gslot, Fc)
    wslot = torch.where(place, gslot, dump)
    def scatter_into(field, vals):
        aug = torch.cat([field, torch.zeros(B, 1, dtype=DTYPE, device=dev)], 1)
        aug.scatter_(1, wslot, vals)
        return aug[:, :Fc]
    fs = from_slot.clamp(0, env.p_x.shape[1] - 1)
    opx = env.p_x.gather(1, fs); opy = env.p_y.gather(1, fs); orad = env.p_radius.gather(1, fs)
    sx = opx + torch.cos(angle) * (orad + 0.1)
    sy = opy + torch.sin(angle) * (orad + 0.1)
    onesL = torch.ones(B, L, dtype=DTYPE, device=dev)
    env.f_alive = scatter_into(env.f_alive, onesL)
    env.f_owner = scatter_into(env.f_owner, owner)
    env.f_x = scatter_into(env.f_x, sx)
    env.f_y = scatter_into(env.f_y, sy)
    env.f_angle = scatter_into(env.f_angle, angle)
    env.f_ships = scatter_into(env.f_ships, ships)
    env.f_seq = scatter_into(env.f_seq, seq)


def opponent_action(cfg, env, opponent, pid=1):
    '''Scripted launches for seat `pid` (v8: any seat in a 2p/4p game): one per owned planet, half garrison (>=20).
    0=random heading, 1=starter (nearest static non-owned), 2=noop, 3=medium (starter++: nearest incl. rotating, lead-aim),
    4=greedy (nearest beatable non-owned planet within CAPTURE_RADIUS, just-enough ships),
    5=intermediate (in-flight-aware capture from nearest capable source + counter + rebalance + comet-escape).
    -> angle,ships,commit (B,Ec).'''
    B, Ec = env.B, env.Ec
    dev = env.dev
    min_ships = 20.0
    angle = torch.zeros(B, Ec, dtype=DTYPE, device=dev)
    ships = torch.zeros(B, Ec, dtype=DTYPE, device=dev)
    commit = torch.zeros(B, Ec, dtype=DTYPE, device=dev)
    if opponent == 2:
        return angle, ships, commit
    half = torch.floor(env.p_ships / 2.0)
    base = (env.p_owner == float(pid)) & (env.p_alive > 0.5) & (half >= min_ships)
    if opponent == 0:  # random heading
        angle = torch.rand(B, Ec, dtype=DTYPE, device=dev) * (2.0 * PI)
        ships = torch.where(base, half, torch.zeros_like(half))
        commit = base.to(DTYPE)
        return angle, ships, commit
    if opponent == 3:  # medium (starter++): nearest non-owned planet INCL. rotating, lead-aimed at its intercept
        sx = env.p_x.unsqueeze(2); sy = env.p_y.unsqueeze(2)             # (B,Ec,1) source
        tx = env.p_x.unsqueeze(1); ty = env.p_y.unsqueeze(1)            # (B,1,Ec) dest
        vt = (env.p_alive > 0.5) & (env.p_owner != float(pid))                 # (B,Ec) valid targets (rotating allowed)
        d = torch.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)
        dmask = torch.where(vt.unsqueeze(1), d, torch.full_like(d, BIG))
        bestd, tgt = dmask.min(2)                                       # (B,Ec) nearest valid target per source
        has_tgt = bestd < BIG * 0.5
        dpx = env.p_x.gather(1, tgt); dpy = env.p_y.gather(1, tgt)      # (B,Ec) target position
        if cfg.LEAD_TARGET:                                            # lead the (possibly orbiting) target
            rdx = dpx - CENTER; rdy = dpy - CENTER
            r_d = torch.sqrt(rdx * rdx + rdy * rdy)
            phi0 = torch.atan2(rdy, rdx)
            drot = env.p_rotates.gather(1, tgt) > 0.5
            w = torch.where(drot, env.ang_vel.view(B, 1), torch.zeros(B, 1, device=dev, dtype=DTYPE))
            off = env.p_radius + 0.1
            v = fleet_speed_t(half.clamp_min(1.0), env.vmax)
            t = (torch.sqrt((dpx - env.p_x) ** 2 + (dpy - env.p_y) ** 2) - off).clamp_min(0.0) / v.clamp_min(1e-6)
            for _ in range(16):
                phi = phi0 + w * t
                ix = CENTER + r_d * torch.cos(phi); iy = CENTER + r_d * torch.sin(phi)
                t = (torch.sqrt((ix - env.p_x) ** 2 + (iy - env.p_y) ** 2) - off).clamp_min(0.0) / v.clamp_min(1e-6)
            phi = phi0 + w * t
            ix = CENTER + r_d * torch.cos(phi); iy = CENTER + r_d * torch.sin(phi)
            angle = torch.atan2(iy - env.p_y, ix - env.p_x)
        else:
            angle = torch.atan2(dpy - env.p_y, dpx - env.p_x)
        go = base & has_tgt
        ships = torch.where(go, half, torch.zeros_like(half))
        commit = go.to(DTYPE)
        return angle, ships, commit
    if opponent == 4:  # greedy: capture the NEAREST non-owned planet within CAPTURE_RADIUS we can already beat
        sx = env.p_x.unsqueeze(2); sy = env.p_y.unsqueeze(2)            # (B,Ec,1) source
        tx = env.p_x.unsqueeze(1); ty = env.p_y.unsqueeze(1)           # (B,1,Ec) dest
        d = torch.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)                # (B,Ec,Ec)
        vt = (env.p_alive > 0.5) & (env.p_owner != float(pid))                # (B,Ec) non-owned alive
        can_beat = env.p_ships.unsqueeze(2) > env.p_ships.unsqueeze(1) # (B,src,dst) src ships > dst ships
        cand = vt.unsqueeze(1) & can_beat & (d <= cfg.CAPTURE_RADIUS)  # capturable & in range
        dmask = torch.where(cand, d, torch.full_like(d, BIG))
        bestd, tgt = dmask.min(2)                                       # (B,Ec) nearest capturable target
        has_tgt = bestd < BIG * 0.5
        tgt_ships = env.p_ships.gather(1, tgt)                          # (B,Ec)
        n = torch.minimum(env.p_ships, torch.floor(tgt_ships) + cfg.CAPTURE_MARGIN)  # just enough to capture (+ buffer)
        dpx = env.p_x.gather(1, tgt); dpy = env.p_y.gather(1, tgt)      # (B,Ec) target position
        if cfg.LEAD_TARGET:                                            # lead the (possibly orbiting) target
            rdx = dpx - CENTER; rdy = dpy - CENTER
            r_d = torch.sqrt(rdx * rdx + rdy * rdy)
            phi0 = torch.atan2(rdy, rdx)
            drot = env.p_rotates.gather(1, tgt) > 0.5
            w = torch.where(drot, env.ang_vel.view(B, 1), torch.zeros(B, 1, device=dev, dtype=DTYPE))
            off = env.p_radius + 0.1
            v = fleet_speed_t(n.clamp_min(1.0), env.vmax)
            t = (torch.sqrt((dpx - env.p_x) ** 2 + (dpy - env.p_y) ** 2) - off).clamp_min(0.0) / v.clamp_min(1e-6)
            for _ in range(16):
                phi = phi0 + w * t
                ix = CENTER + r_d * torch.cos(phi); iy = CENTER + r_d * torch.sin(phi)
                t = (torch.sqrt((ix - env.p_x) ** 2 + (iy - env.p_y) ** 2) - off).clamp_min(0.0) / v.clamp_min(1e-6)
            phi = phi0 + w * t
            ix = CENTER + r_d * torch.cos(phi); iy = CENTER + r_d * torch.sin(phi)
            angle = torch.atan2(iy - env.p_y, ix - env.p_x)
        else:
            angle = torch.atan2(dpy - env.p_y, dpx - env.p_x)
        own_ok = (env.p_owner == float(pid)) & (env.p_alive > 0.5) & (env.p_ships > 0.0)
        go = own_ok & has_tgt & (n >= 1.0)
        ships = torch.where(go, n, torch.zeros_like(n))
        commit = go.to(DTYPE)
        return angle, ships, commit
    if opponent == 5:  # intermediate: in-flight-aware capture from nearest capable source (cascade) + rebalance
        owned = (env.p_owner == float(pid)) & (env.p_alive > 0.5) & (env.p_ships > 0.0)   # (B,Ec) sources that can act
        sx = env.p_x.unsqueeze(2); sy = env.p_y.unsqueeze(2)
        tx = env.p_x.unsqueeze(1); ty = env.p_y.unsqueeze(1)
        d = torch.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)                # (B,src,dst)
        srcrange = torch.arange(Ec, device=dev).view(1, Ec, 1)
        # capture (in-flight aware): per target, needed = garrison + enemy_inbound - friendly_inbound + margin
        ftgt, _feta, *_ = fleet_target_batch(env, env.f_x, env.f_y, env.f_angle, env.f_ships, env.vmax)
        f_ok = (env.f_alive > 0.5) & (ftgt >= 0)            # SUN-AWARE target: sun-bound fleets don't arrive
        tslot_f = ftgt.clamp(0, Ec - 1)
        in_fr = torch.zeros(B, Ec, device=dev, dtype=DTYPE)            # friendly ships already inbound per planet
        in_en = torch.zeros(B, Ec, device=dev, dtype=DTYPE)            # enemy ships inbound per planet
        in_fr.scatter_add_(1, tslot_f, env.f_ships * ((env.f_owner == float(pid)) & f_ok).to(DTYPE))
        in_en.scatter_add_(1, tslot_f, env.f_ships * ((env.f_owner != float(pid)) & f_ok).to(DTYPE))
        vt = (env.p_alive > 0.5) & (env.p_owner != float(pid))
        needed_t = torch.floor(env.p_ships) + torch.ceil(in_en) - torch.floor(in_fr) + cfg.CAPTURE_MARGIN   # (B,Ec)
        open_t = vt & (needed_t > 0.0)                                 # not already covered by friendly inbound
        can_supply = env.p_ships.unsqueeze(2) >= needed_t.unsqueeze(1) # (B,src,dst) source can cover the requirement
        capable = owned.unsqueeze(2) & open_t.unsqueeze(1) & can_supply
        d_cap = torch.where(capable, d, torch.full_like(d, BIG))
        nearest_cap = d_cap.argmin(1)                                  # (B,dst) nearest CAPABLE source (cascade)
        target_has = d_cap.min(1).values < BIG * 0.5
        assigned = (nearest_cap.unsqueeze(1) == srcrange) & target_has.unsqueeze(1)
        d_asg = torch.where(assigned, d, torch.full_like(d, BIG))
        cap_d, cap_tgt = d_asg.min(2)                                  # (B,Ec) source's nearest assigned target
        cap_has = cap_d < BIG * 0.5
        cap_n = torch.minimum(env.p_ships, needed_t.gather(1, cap_tgt))   # send exactly the requirement
        # rebalance: feed the poorest owned planet WITHIN REBALANCE_RADIUS (local) when surplus exceeds a percentage
        eyeE = (torch.eye(Ec, device=dev, dtype=DTYPE).unsqueeze(0) > 0.5)
        reb_cand = owned.unsqueeze(1) & (d <= cfg.REBALANCE_RADIUS) & (~eyeE)   # (B,src,dst) owned, in-radius, not self
        reb_dst_ships = torch.where(reb_cand, env.p_ships.unsqueeze(1), torch.full_like(d, BIG))
        poor_ships, poor_idx = reb_dst_ships.min(2)                    # (B,Ec) poorest owned-in-radius per source
        has_poor = poor_ships < BIG * 0.5
        gap = env.p_ships - poor_ships
        reb_ok = owned & (~cap_has) & has_poor & (gap > cfg.REBALANCE_PCT * env.p_ships.clamp_min(1.0))
        reb_n = torch.floor(gap * 0.5)
        reb_tgt = poor_idx
        # comet escape (top priority): owned comets evacuate ALL ships to the nearest owned non-comet planet
        safe_dst = (owned & (env.p_is_comet < 0.5)).unsqueeze(1) & (~eyeE)   # (B,src,dst) dest owned non-comet, not self
        d_evac = torch.where(safe_dst, d, torch.full_like(d, BIG))
        evac_d, evac_tgt = d_evac.min(2)
        evac_ok = owned & (env.p_is_comet > 0.5) & (evac_d < BIG * 0.5) & (env.p_ships >= 1.0)
        # priority: comet-escape > capture > rebalance
        use_cap = (~evac_ok) & cap_has
        use_reb = (~evac_ok) & (~cap_has) & reb_ok
        tgt = torch.where(evac_ok, evac_tgt, torch.where(use_cap, cap_tgt, reb_tgt))
        nn = torch.where(evac_ok, env.p_ships, torch.where(use_cap, cap_n, torch.where(use_reb, reb_n, torch.zeros_like(reb_n))))
        go = evac_ok | use_cap | use_reb
        dpx = env.p_x.gather(1, tgt); dpy = env.p_y.gather(1, tgt)
        if cfg.LEAD_TARGET:
            rdx = dpx - CENTER; rdy = dpy - CENTER
            r_d = torch.sqrt(rdx * rdx + rdy * rdy)
            phi0 = torch.atan2(rdy, rdx)
            drot = env.p_rotates.gather(1, tgt) > 0.5
            w = torch.where(drot, env.ang_vel.view(B, 1), torch.zeros(B, 1, device=dev, dtype=DTYPE))
            off = env.p_radius + 0.1
            v = fleet_speed_t(nn.clamp_min(1.0), env.vmax)
            t = (torch.sqrt((dpx - env.p_x) ** 2 + (dpy - env.p_y) ** 2) - off).clamp_min(0.0) / v.clamp_min(1e-6)
            for _ in range(16):
                phi = phi0 + w * t
                ix = CENTER + r_d * torch.cos(phi); iy = CENTER + r_d * torch.sin(phi)
                t = (torch.sqrt((ix - env.p_x) ** 2 + (iy - env.p_y) ** 2) - off).clamp_min(0.0) / v.clamp_min(1e-6)
            phi = phi0 + w * t
            ix = CENTER + r_d * torch.cos(phi); iy = CENTER + r_d * torch.sin(phi)
            angle = torch.atan2(iy - env.p_y, ix - env.p_x)
        else:
            angle = torch.atan2(dpy - env.p_y, dpx - env.p_x)
        ships = torch.where(go, nn, torch.zeros_like(nn))
        commit = go.to(DTYPE)
        return angle, ships, commit
    # opponent == 1: starter -- fire at nearest static non-owned planet
    distc = torch.sqrt((env.p_x - CENTER) ** 2 + (env.p_y - CENTER) ** 2)
    is_static = (distc + env.p_radius) >= ROTATION_RADIUS_LIMIT
    valid_tgt = is_static & (env.p_alive > 0.5) & (env.p_owner != float(pid))
    sx = env.p_x.unsqueeze(2); sy = env.p_y.unsqueeze(2)
    tx = env.p_x.unsqueeze(1); ty = env.p_y.unsqueeze(1)
    d = torch.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)
    dmask = torch.where(valid_tgt.unsqueeze(1), d, torch.full_like(d, BIG))
    bestd, bestidx = dmask.min(2)
    has_tgt = bestd < BIG * 0.5
    btx = env.p_x.gather(1, bestidx); bty = env.p_y.gather(1, bestidx)
    angle = torch.atan2(bty - env.p_y, btx - env.p_x)
    go = base & has_tgt
    ships = torch.where(go, half, torch.zeros_like(half))
    commit = go.to(DTYPE)
    return angle, ships, commit


class StepOut:
    pass


def _decode_target(cfg, env, action, legal):
    '''Gated-allocation decode. action (B,Ec,Ec+1): [...,:Ec]=WHERE allocation rows, [...,Ec]=fire{0,1}.
    A planet FIRES (full garrison, routed by its renormalised off-diagonal WHERE row) or HOLDS. ships[s,d]
    = floor(Ahat[s,d]*S_s) for d!=s, to ALIVE dests, >= MIN_LAUNCH_SHIPS, only if fire[s]=1; the rest stay
    home. Each launch aims at the orbiting intercept. Returns angle/ships/can (B,Ec,Ec), valid/invalid/launches (B,).'''
    B, Ec, dev = env.B, env.Ec, env.dev
    A = action[..., :Ec]                                        # (B,Ec,Ec) WHERE allocation
    fire = action[..., Ec]                                      # (B,Ec) launch gate in {0,1}
    S = env.p_ships                                             # (B,Ec)
    eye = torch.eye(Ec, dtype=DTYPE, device=dev).unsqueeze(0)   # (1,Ec,Ec)
    A_send = A * (1.0 - eye)                                    # drop self-diagonal (s->s meaningless)
    A_send = A_send / A_send.sum(2, keepdim=True).clamp_min(1e-6)   # renormalise -> fire = FULL garrison out
    raw = A_send * (S * fire).unsqueeze(2)                      # (B,Ec,Ec) ships src->dst, zero if held
    dest_alive = (env.p_alive > 0.5).to(DTYPE).unsqueeze(1)     # (B,1,Ec)
    gate = legal.to(DTYPE).unsqueeze(2) * dest_alive            # (B,Ec,Ec) legal source & alive dest
    n = torch.floor(raw) * gate
    n = torch.where(n >= float(cfg.MIN_LAUNCH_SHIPS), n, torch.zeros_like(n))   # drop dribbles -> stay home
    can = (n >= 1.0).to(DTYPE)
    # intercept heading for every (src,dst): src = row planet, dst = col planet
    spx = env.p_x.unsqueeze(2); spy = env.p_y.unsqueeze(2)      # (B,Ec,1) source
    dpx = env.p_x.unsqueeze(1); dpy = env.p_y.unsqueeze(1)      # (B,1,Ec) dest
    off = env.p_radius.unsqueeze(2) + 0.1                       # (B,Ec,1) source surface
    if cfg.LEAD_TARGET:
        rdx = dpx - CENTER; rdy = dpy - CENTER
        r_d = torch.sqrt(rdx * rdx + rdy * rdy)                 # (B,1,Ec)
        phi0 = torch.atan2(rdy, rdx)
        drot = (env.p_rotates > 0.5).unsqueeze(1)               # (B,1,Ec)
        w = torch.where(drot, env.ang_vel.view(B, 1, 1), torch.zeros(B, 1, 1, device=dev, dtype=DTYPE))
        v = fleet_speed_t(n.clamp_min(1.0), env.vmax)           # (B,Ec,Ec)
        t = (torch.sqrt((dpx - spx) ** 2 + (dpy - spy) ** 2) - off).clamp_min(0.0) / v.clamp_min(1e-6)
        for _ in range(16):
            phi = phi0 + w * t
            ix = CENTER + r_d * torch.cos(phi); iy = CENTER + r_d * torch.sin(phi)
            t = (torch.sqrt((ix - spx) ** 2 + (iy - spy) ** 2) - off).clamp_min(0.0) / v.clamp_min(1e-6)
        phi = phi0 + w * t
        ix = CENTER + r_d * torch.cos(phi); iy = CENTER + r_d * torch.sin(phi)
        angle = torch.atan2(iy - spy, ix - spx)                 # (B,Ec,Ec)
    else:
        angle = torch.atan2(dpy - spy, dpx - spx).expand(B, Ec, Ec).contiguous()
    valid = can.sum((1, 2))                                     # launches all go to alive dests
    invalid = torch.zeros(B, dtype=DTYPE, device=dev)           # below-min launches just stay home (no penalty); budget is structural
    launches = can.sum((1, 2))
    return angle, n, can, valid, invalid, launches


def spawn_comets(cfg, env):
    '''Spawn COMET_PAIRS point-symmetric comet pairs into free planet slots: neutral objects that
    sweep the inner region at COMET_SPEED and expire past COMET_EXPIRE_RADIUS. Capturable; produce
    COMET_PRODUCTION once owned. Velocity aims past the center at a random perihelion (avoids the sun).'''
    B, dev = env.B, env.dev
    for _pair in range(cfg.COMET_PAIRS):
        theta = torch.rand(B, device=dev) * (2.0 * PI)
        d_peri = cfg.COMET_PERI_MIN + torch.rand(B, device=dev) * (cfg.COMET_PERI_MAX - cfg.COMET_PERI_MIN)
        sgn = torch.where(torch.rand(B, device=dev) < 0.5, torch.ones(B, device=dev), -torch.ones(B, device=dev))
        ships = torch.randint(1, 100, (B, 4), device=dev).min(1).values.to(DTYPE)   # min-of-4 -> skewed low
        for _c in range(2):
            th = theta + (0.0 if _c == 0 else PI)                       # point-symmetric pair
            r0 = float(cfg.COMET_SPAWN_RADIUS)
            px = CENTER + r0 * torch.cos(th); py = CENTER + r0 * torch.sin(th)
            toward = torch.atan2(CENTER - py, CENTER - px)
            alpha = torch.asin((d_peri / r0).clamp(max=0.99)) * (sgn if _c == 0 else -sgn)
            vdir = toward + alpha
            vx = cfg.COMET_SPEED * torch.cos(vdir); vy = cfg.COMET_SPEED * torch.sin(vdir)
            free = env.p_alive < 0.5
            slot = torch.argmax(free.to(torch.int8), 1, keepdim=True)    # (B,1) first free slot
            has = free.any(1, keepdim=True)
            def _set(field, val):
                v = val.unsqueeze(1) if val.dim() == 1 else val
                field.scatter_(1, slot, torch.where(has, v, field.gather(1, slot)))
            _set(env.p_alive, torch.ones(B, device=dev))
            _set(env.p_owner, torch.full((B,), -1.0, device=dev))
            _set(env.p_x, px); _set(env.p_y, py); _set(env.p_init_x, px); _set(env.p_init_y, py)
            _set(env.p_comet_vx, vx); _set(env.p_comet_vy, vy)
            _set(env.p_ships, ships)
            _set(env.p_prod, torch.full((B,), float(COMET_PRODUCTION), device=dev))
            _set(env.p_radius, torch.full((B,), float(COMET_RADIUS), device=dev))
            _set(env.p_is_comet, torch.ones(B, device=dev))
            _set(env.p_rotates, torch.zeros(B, device=dev))


def spawn_comets_official(env, e):
    """Place the 4 precomputed symmetric comets of spawn event e at waypoint 0 (free planet slots)."""
    B, dev = env.B, env.dev
    has = env.c_len[:, e] > 0                                    # (B,) event exists for this world
    one = torch.ones(B, 1, dtype=DTYPE, device=dev)
    for m in range(4):
        free = env.p_alive < 0.5
        slot = torch.argmax(free.to(torch.int8), 1, keepdim=True)            # (B,1) first free slot
        ok = (has & free.any(1)).unsqueeze(1)                                # (B,1)
        px = env.c_paths[:, e, m, 0, 0:1]; py = env.c_paths[:, e, m, 0, 1:2]
        def _set(field, val):
            field.scatter_(1, slot, torch.where(ok, val, field.gather(1, slot)))
        _set(env.p_alive, one)
        _set(env.p_owner, -one)
        _set(env.p_x, px); _set(env.p_y, py)
        _set(env.p_init_x, px); _set(env.p_init_y, py)
        _set(env.p_ships, env.c_ships[:, e].unsqueeze(1))
        _set(env.p_prod, one * float(COMET_PRODUCTION))
        _set(env.p_radius, one * float(COMET_RADIUS))
        _set(env.p_is_comet, one)
        _set(env.p_rotates, torch.zeros_like(one))
        env.c_slot[:, e, m] = torch.where(ok.squeeze(1), slot.squeeze(1),
                                          torch.full_like(slot.squeeze(1), -1))


def env_step(cfg, env, ego_action, opponent=None, opp_action=None, step_idx=None, seats=None):
    """One tick, N-player (v8). ego_action (B,Ec,Ec+1)=[alloc | fire] for player 0.
    seats = list of opponent seat dicts, one per pid 1..n_players-1:
        {"pid": p, "script": code}    scripted bot (opponent_action codes), or
        {"pid": p, "action": tensor}  neural (B,Ec,Ec+1) gated-alloc action.
    v7 back-compat: seats=None -> a single pid-1 seat built from (opponent, opp_action)."""
    B, Ec, Fc = env.B, env.Ec, env.Fc
    N = int(getattr(env, "n_players", 2))
    if seats is None:
        seats = [{"pid": 1, "action": opp_action}] if opp_action is not None else [{"pid": 1, "script": opponent}]
    _sc = int(env.step_ct[0].item()) if step_idx is None else int(step_idx)   # avoid a per-step CUDA sync
    if cfg.COMETS_ENABLED and _sc in cfg.COMET_SPAWN_STEPS:   # hidden-schedule comet spawn
        if cfg.COMET_OFFICIAL and getattr(env, 'c_paths', None) is not None:
            spawn_comets_official(env, cfg.COMET_SPAWN_STEPS.index(_sc))
        else:
            spawn_comets(cfg, env)
    vmax = env.vmax
    dev = env.dev
    ego = 0

    ego_owned0 = (env.p_owner == float(ego)) & (env.p_alive > 0.5)
    comet0 = env.p_is_comet > 0.5   # comet status at step start (mask comet expiry out of capture/lost)

    # --- decode ego action (gated allocation: fire -> full garrison, routed) ---
    legal = (env.p_owner == float(ego)) & (env.p_alive > 0.5) & (env.p_ships > 0.0)
    e_ang, e_shp, e_can, valid, invalid, launches = _decode_target(cfg, env, ego_action, legal)

    # --- opponent seat launches (scripted: one launch/planet; neural: full alloc decode) ---
    slot_idx = torch.arange(Ec, dtype=torch.long, device=dev).unsqueeze(0).expand(B, Ec)
    src_mat = torch.arange(Ec, dtype=torch.long, device=dev).view(1, Ec, 1).expand(B, Ec, Ec).reshape(B, Ec * Ec)
    flatM = lambda x: x.reshape(B, Ec * Ec)                                   # (B,Ec,Ec) -> (B,Ec*Ec)
    own_blk = [torch.full((B, Ec * Ec), float(ego), dtype=DTYPE, device=dev)]
    slot_blk = [src_mat]
    ang_blk = [flatM(e_ang)]; shp_blk = [flatM(e_shp)]; can_blk = [flatM(e_can)]
    ded = e_shp.sum(2)                                                        # (B,Ec) ships deducted/source
    for st in seats:
        pid = int(st["pid"])
        if st.get("action") is not None:          # neural seat: decode exactly like the ego
            legal_p = (env.p_owner == float(pid)) & (env.p_alive > 0.5) & (env.p_ships > 0.0)
            o_ang, o_shp, o_can, _, _, _ = _decode_target(cfg, env, st["action"], legal_p)
            ded = ded + o_shp.sum(2)
            own_blk.append(torch.full((B, Ec * Ec), float(pid), dtype=DTYPE, device=dev))
            slot_blk.append(src_mat)
            ang_blk.append(flatM(o_ang)); shp_blk.append(flatM(o_shp)); can_blk.append(flatM(o_can))
        else:                                     # scripted seat: one launch per owned planet
            o_ang, o_shp, o_can = opponent_action(cfg, env, int(st["script"]), pid)
            ded = ded + o_shp
            own_blk.append(torch.full((B, Ec), float(pid), dtype=DTYPE, device=dev))
            slot_blk.append(slot_idx)
            ang_blk.append(o_ang); shp_blk.append(o_shp); can_blk.append(o_can)

    # --- deduct ships from origin planets ---
    env.p_ships = env.p_ships - ded

    # --- place fleets (ego first, then seats in pid order) ---
    owner = torch.cat(own_blk, 1)
    from_slot = torch.cat(slot_blk, 1)
    angle = torch.cat(ang_blk, 1)
    ships = torch.cat(shp_blk, 1)
    commit = torch.cat(can_blk, 1)
    Ltot = owner.shape[1]
    seq = (env.step_ct * float(Ltot + 1)).unsqueeze(1) + \
          torch.arange(Ltot, dtype=DTYPE, device=dev).unsqueeze(0)
    launch_fleets(env, owner, from_slot, angle, ships, commit, seq)

    # --- production ---
    env.p_ships = env.p_ships + env.p_prod * (env.p_owner != -1.0).to(DTYPE) * (env.p_alive > 0.5).to(DTYPE)

    # --- planet new positions (orbit) ---
    stepf = env.step_ct
    dxc = env.p_init_x - CENTER; dyc = env.p_init_y - CENTER
    r = torch.sqrt(dxc * dxc + dyc * dyc)
    ia = torch.atan2(dyc, dxc)
    ca = ia + env.ang_vel.unsqueeze(1) * stepf.unsqueeze(1)
    rot = env.p_rotates > 0.5
    cmt = env.p_is_comet > 0.5
    nx = torch.where(rot, CENTER + r * torch.cos(ca), torch.where(cmt, env.p_x + env.p_comet_vx, env.p_x))
    ny = torch.where(rot, CENTER + r * torch.sin(ca), torch.where(cmt, env.p_y + env.p_comet_vy, env.p_y))
    old_px, old_py = env.p_x, env.p_y
    _comet_expired = None
    if cfg.COMETS_ENABLED and cfg.COMET_OFFICIAL and getattr(env, 'c_paths', None) is not None:
        # waypoint playback: comet of event e at tick u sweeps path[u-s] -> path[u-s+1]
        _comet_expired = torch.zeros(B, Ec, dtype=torch.bool, device=dev)
        _ar = torch.arange(B, device=dev)
        for _e, _s_e in enumerate(cfg.COMET_SPAWN_STEPS):
            if not (_s_e <= _sc <= _s_e + cfg.COMET_MAX_LEN):   # at most one event is ever live
                continue
            k = int(_sc - _s_e + 1)                          # waypoint index this tick (same for all envs)
            for _m in range(4):
                sl = env.c_slot[:, _e, _m]                   # (B,) planet slot, -1 = none
                live = sl >= 0
                if not bool(live.any()):
                    continue
                slc = sl.clamp_min(0).unsqueeze(1)
                adv = live & (k < env.c_len[:, _e])          # still has waypoints -> advance
                exp = live & (k >= env.c_len[:, _e])         # path ended -> expire after combat
                wp = env.c_paths[_ar, _e, _m, min(k, cfg.COMET_MAX_LEN - 1)]   # (B,2)
                nx.scatter_(1, slc, torch.where(adv.unsqueeze(1), wp[:, 0:1], nx.gather(1, slc)))
                ny.scatter_(1, slc, torch.where(adv.unsqueeze(1), wp[:, 1:2], ny.gather(1, slc)))
                _comet_expired.scatter_(1, slc, exp.unsqueeze(1) | _comet_expired.gather(1, slc))
                env.c_slot[:, _e, _m] = torch.where(exp, torch.full_like(sl, -1), sl)

    # --- fleet movement + swept collision against planet paths ---
    falive = env.f_alive > 0.5
    speed = fleet_speed_t(env.f_ships, vmax)
    fox, foy = env.f_x, env.f_y
    fnx = fox + torch.cos(env.f_angle) * speed
    fny = foy + torch.sin(env.f_angle) * speed
    Ax = fox.unsqueeze(2); Ay = foy.unsqueeze(2)
    Bx = fnx.unsqueeze(2); By = fny.unsqueeze(2)
    P0x = old_px.unsqueeze(1); P0y = old_py.unsqueeze(1)
    P1x = nx.unsqueeze(1); P1y = ny.unsqueeze(1)
    rad = env.p_radius.unsqueeze(1)
    palive = (env.p_alive.unsqueeze(1) > 0.5)
    d0x = Ax - P0x; d0y = Ay - P0y
    dvx = (Bx - Ax) - (P1x - P0x); dvy = (By - Ay) - (P1y - P0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - rad * rad
    disc = b * b - 4.0 * a * c
    sq = torch.sqrt(disc.clamp_min(0.0))
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    hit_quad = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    hit_lin = (a < 1e-12) & (c <= 0.0)
    hit = torch.where(a < 1e-12, hit_lin, hit_quad)
    hit = hit & palive & falive.unsqueeze(2)
    slotf = torch.arange(Ec, dtype=DTYPE, device=dev).view(1, 1, Ec)
    order = torch.where(hit, slotf, torch.full_like(slotf, float(Ec)))
    fh_v, tgt_slot = order.min(2)
    has_hit = fh_v < float(Ec)

    # OOB / sun removal (point-to-segment to sun center)
    oob = (fnx < 0.0) | (fnx > BOARD_SIZE) | (fny < 0.0) | (fny > BOARD_SIZE)
    vx, vy, wx, wy = fox, foy, fnx, fny
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    tt = ((CENTER - vx) * (wx - vx) + (CENTER - vy) * (wy - vy)) / l2.clamp_min(1e-12)
    tt = tt.clamp(0.0, 1.0)
    prx = vx + tt * (wx - vx); pry = vy + tt * (wy - vy)
    sundist = torch.sqrt((CENTER - prx) ** 2 + (CENTER - pry) ** 2)
    sun_hit = (l2 > 0.0) & (sundist < SUN_RADIUS)
    sun_pt = torch.sqrt((CENTER - vx) ** 2 + (CENTER - vy) ** 2) < SUN_RADIUS
    sun_hit = torch.where(l2 > 0.0, sun_hit, sun_pt)

    remove_fleet = falive & (has_hit | oob | sun_hit)
    contributes = falive & has_hit

    # --- combat: arrivals per PLAYER, official top-vs-second resolve (N-player) ---
    cf = contributes.to(DTYPE)
    tslot = tgt_slot.clamp(0, Ec - 1)
    arr_p = []
    for p in range(N):
        sp = env.f_ships * cf * (env.f_owner == float(p)).to(DTYPE)
        ap = torch.zeros(B, Ec, dtype=DTYPE, device=dev)
        ap.scatter_add_(1, tslot, sp)
        arr_p.append(ap)
    arr = torch.stack(arr_p, 2)                                  # (B,Ec,N)
    top2v, top2i = arr.topk(2, dim=2)                            # N >= 2 always
    surv_ships = top2v[..., 0] - top2v[..., 1]                   # exact tie -> 0 survivors
    surv_owner = torch.where(surv_ships > 0.0, top2i[..., 0].to(DTYPE),
                             torch.full((B, Ec), -1.0, dtype=DTYPE, device=dev))
    any_arr = top2v[..., 0] > 0.0
    apply = any_arr & (surv_ships > 0.0) & (env.p_alive > 0.5)
    same = (env.p_owner == surv_owner)
    reinforce = apply & same
    attack = apply & (~same)
    env.p_ships = torch.where(reinforce, env.p_ships + surv_ships, env.p_ships)
    after = env.p_ships - surv_ships
    flips = attack & (after < 0.0)
    env.p_ships = torch.where(attack, torch.where(after < 0.0, -after, after), env.p_ships)
    env.p_owner = torch.where(flips, surv_owner, env.p_owner)

    # --- apply planet positions; clear removed fleets ---
    env.p_x_prev = old_px; env.p_y_prev = old_py   # one-step prev pos for finite-diff body velocity (threat lead)
    env.p_x = nx; env.p_y = ny
    if cfg.COMETS_ENABLED:                               # comet expiry: official = path end; legacy = swept past the inner region
        if _comet_expired is not None:
            gone = _comet_expired & (env.p_alive > 0.5)
        else:
            cdist = torch.sqrt((env.p_x - CENTER) ** 2 + (env.p_y - CENTER) ** 2)
            gone = (env.p_is_comet > 0.5) & (env.p_alive > 0.5) & (cdist > cfg.COMET_EXPIRE_RADIUS)
        keepc = (~gone).to(DTYPE)
        env.p_alive = env.p_alive * keepc; env.p_is_comet = env.p_is_comet * keepc
        env.p_comet_vx = env.p_comet_vx * keepc; env.p_comet_vy = env.p_comet_vy * keepc
        env.p_ships = env.p_ships * keepc
        env.p_owner = torch.where(gone, torch.full_like(env.p_owner, -1.0), env.p_owner)
    keep = falive & (~remove_fleet)
    keepf = keep.to(DTYPE)
    env.f_alive = keepf
    env.f_owner = env.f_owner * keepf
    env.f_x = fnx * keepf; env.f_y = fny * keepf
    env.f_angle = env.f_angle * keepf
    env.f_ships = env.f_ships * keepf

    env.step_ct = env.step_ct + 1.0

    ego_owned1 = (env.p_owner == float(ego)) & (env.p_alive > 0.5)

    out = StepOut()
    out.invalid = invalid
    out.valid = valid
    out.launches = launches
    out.launched_ships = e_shp.sum((1, 2))   # total ships launched this step (ship-weighted launch reward)
    out.owned_launch_ships = (e_shp * ego_owned0.unsqueeze(1).to(DTYPE)).sum((1, 2))   # ships launched to ALREADY-OWNED planets
    noncomet1 = env.p_is_comet < 0.5
    out.captured = (ego_owned1 & (~ego_owned0) & noncomet1).to(DTYPE).sum(1)
    out.lost = (ego_owned0 & (~ego_owned1) & (~comet0)).to(DTYPE).sum(1)
    out.captured_prod = (env.p_prod * (ego_owned1 & (~ego_owned0) & noncomet1).to(DTYPE)).sum(1)
    out.lost_prod = (env.p_prod * (ego_owned0 & (~ego_owned1) & (~comet0)).to(DTYPE)).sum(1)
    return out


def settle_n(env):
    """Per-player settle: ships (B,N) = planets+fleets per player; alive (B,N) bool.

    Vectorized over players (one-hot owner broadcast on a trailing player axis) -- a few large kernels
    instead of the per-player Python loop's ~6N small launches, which matters because this runs every
    rollout step. Bit-identical to the loop (same masks, same per-player sums)."""
    N = int(getattr(env, "n_players", 2))
    pid = torch.arange(N, device=env.p_owner.device, dtype=env.p_owner.dtype)    # (N,)
    op = (env.p_owner.unsqueeze(-1) == pid) & (env.p_alive > 0.5).unsqueeze(-1)  # (B,E,N)
    gp = (env.f_owner.unsqueeze(-1) == pid) & (env.f_alive > 0.5).unsqueeze(-1)  # (B,Fc,N)
    s = (env.p_ships.unsqueeze(-1) * op.to(DTYPE)).sum(1) + (env.f_ships.unsqueeze(-1) * gp.to(DTYPE)).sum(1)  # (B,N)
    alive = op.any(1) | gp.any(1)                                               # (B,N)
    return s, alive


def settle(env):
    """v7-compat 2p view: (s0, s1, side0_alive, side1_alive). BC/eval/probe cells use this."""
    s, alive = settle_n(env)
    return s[:, 0], s[:, 1], alive[:, 0], alive[:, 1]
