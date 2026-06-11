"""v7 parity: official Kaggle engine vs the setup1_v7 GpuEnv, full games incl. OFFICIAL comets.

Three claims, all seed-matched (configuration.seed -> generate_world(seed)):
  A. WORLD-GEN: the notebook's CPU generate_world(seed) == the official engine's initial state.
  B. DYNAMICS: replaying the official per-turn moves through a v7 explicit-launch stepper
     (launch -> production -> orbit + comet waypoint playback -> swept collision -> combat ->
     comet expiry) reproduces the official obs EVERY tick of full 500-step games.
  C. HEURISTIC LAUNCHES: at every tick, re-running opponent_action(code) on the free-running
     GpuEnv state emits the SAME moves the same heuristic produced inside the official engine
     (via the eval_kaggle_v7 obs adapter). This is the "launch machine" closed-loop check.

Games are scripted-vs-scripted (starter/medium/greedy/intermediate codes), both seats our v7
heuristics, so every launch in the official game came from the v7 launch pipeline.

    .venv/Scripts/python.exe scripts/parity_check_v7.py --seeds 0,1,2 --worldgen-seeds 40
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import argparse, math, sys, types
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import eval_kaggle_v7 as EV   # noqa: E402  (load_nb_defs, make_scripted_agent, OPP_CODE)

DT = torch.float32
MATCHUPS = [("starter", "medium"), ("greedy", "intermediate"), ("medium", "greedy")]


def _g(obs, key, default=None):
    try:
        return obs[key]
    except Exception:
        return getattr(obs, key, default)


# ------------------------------------------------------------------ A. world-gen parity
def test_worldgen(g, n_seeds):
    from kaggle_environments import make
    bad_seeds = []
    max_d = 0.0
    for sd in range(n_seeds):
        k = make("orbit_wars", configuration={"episodeSteps": 500, "seed": sd}, debug=False)
        st = k.reset(2)
        obs = st[0]["observation"]
        w = g["generate_world"](sd)
        op = _g(obs, "planets")
        wp = w["planets"]
        errs = []
        if abs(float(_g(obs, "angular_velocity")) - w["angular_velocity"]) != 0.0:
            errs.append("ang_vel %r != %r" % (_g(obs, "angular_velocity"), w["angular_velocity"]))
        if len(op) != len(wp):
            errs.append("n_planets off=%d ours=%d" % (len(op), len(wp)))
        else:
            for a, b in zip(op, wp):
                for j in range(7):
                    d = abs(float(a[j]) - float(b[j]))
                    max_d = max(max_d, d)
                    if d != 0.0:
                        errs.append("p%d field%d off=%r ours=%r" % (a[0], j, a[j], b[j]))
        if errs:
            bad_seeds.append((sd, errs[:4]))
    print("[A] world-gen: %d/%d seeds EXACT (max |diff| = %g)" % (n_seeds - len(bad_seeds), n_seeds, max_d))
    for sd, errs in bad_seeds[:5]:
        print("    seed %d: %s" % (sd, "; ".join(errs)))
    return len(bad_seeds) == 0


# ------------------------------------------------------------------ B. explicit v7 stepper
def spawn_phase(env, g):
    """Comet spawn for this tick (split out of the step so the heuristic recompute sees it,
    exactly like env_step runs spawn before opponent_action)."""
    ST = list(g["COMET_SPAWN_STEPS"])
    _sc = int(env.step_ct[0].item())
    if g["COMETS_ENABLED"] and g["COMET_OFFICIAL"] and _sc in ST and getattr(env, "c_paths", None) is not None:
        g["spawn_comets_official"](env, ST.index(_sc))


def step_explicit_v7(env, g, moves_by_player):
    """One v7 tick with EXPLICIT official moves [pid, angle, ships]. Mirrors the notebook's
    env_step dynamics (launch/production/orbit+playback/collision/combat/expiry); the comet
    spawn is done by spawn_phase() beforehand."""
    CENTER, BOARD, SUN = g["CENTER"], g["BOARD_SIZE"], g["SUN_RADIUS"]
    ST = list(g["COMET_SPAWN_STEPS"]); CML = g["COMET_MAX_LEN"]
    fleet_speed_t, launch_fleets = g["fleet_speed_t"], g["launch_fleets"]
    B, Ec, dev = env.B, env.Ec, env.dev
    _sc = int(env.step_ct[0].item())
    # ---- launches: official process_moves semantics (player 0 then 1, sequential validation) ----
    ow, fs, an, sh = [], [], [], []
    for pl, moves in enumerate(moves_by_player):
        if not moves or not isinstance(moves, list):
            continue
        for mv in moves:
            if not isinstance(mv, (list, tuple)) or len(mv) != 3:
                continue
            fid, a, n = int(mv[0]), float(mv[1]), int(mv[2])
            if 0 <= fid < Ec and env.p_alive[0, fid].item() > 0.5 \
                    and int(env.p_owner[0, fid].item()) == pl \
                    and 0 < n <= env.p_ships[0, fid].item():
                env.p_ships[0, fid] -= n
                ow.append(float(pl)); fs.append(fid); an.append(a); sh.append(float(n))
    if ow:
        L = len(ow)
        seq = (env.step_ct * float(L + 1)).view(1, 1) + torch.arange(L, dtype=DT).view(1, L)
        launch_fleets(env, torch.tensor(ow, dtype=DT).view(1, L),
                      torch.tensor(fs, dtype=torch.long).view(1, L),
                      torch.tensor(an, dtype=DT).view(1, L),
                      torch.tensor(sh, dtype=DT).view(1, L),
                      torch.ones(1, L, dtype=DT), seq)
    # ---- production ----
    env.p_ships = env.p_ships + env.p_prod * (env.p_owner != -1.0).to(DT) * (env.p_alive > 0.5).to(DT)
    # ---- planet new positions: orbit; comets straight-line (official: vx=0, playback overrides) ----
    stepf = env.step_ct
    dxc = env.p_init_x - CENTER; dyc = env.p_init_y - CENTER
    r = torch.sqrt(dxc * dxc + dyc * dyc); ia = torch.atan2(dyc, dxc)
    ca = ia + env.ang_vel.unsqueeze(1) * stepf.unsqueeze(1)
    rot = env.p_rotates > 0.5
    cmt = env.p_is_comet > 0.5
    nx = torch.where(rot, CENTER + r * torch.cos(ca), torch.where(cmt, env.p_x + env.p_comet_vx, env.p_x))
    ny = torch.where(rot, CENTER + r * torch.sin(ca), torch.where(cmt, env.p_y + env.p_comet_vy, env.p_y))
    old_px, old_py = env.p_x, env.p_y
    _comet_expired = None
    if g["COMETS_ENABLED"] and g["COMET_OFFICIAL"] and getattr(env, "c_paths", None) is not None:
        _comet_expired = torch.zeros(B, Ec, dtype=torch.bool, device=dev)
        _ar = torch.arange(B, device=dev)
        for _e, _s_e in enumerate(ST):
            if not (_s_e <= _sc <= _s_e + CML):
                continue
            k = int(_sc - _s_e + 1)
            for _m in range(4):
                sl = env.c_slot[:, _e, _m]
                live = sl >= 0
                if not bool(live.any()):
                    continue
                slc = sl.clamp_min(0).unsqueeze(1)
                adv = live & (k < env.c_len[:, _e])
                exp = live & (k >= env.c_len[:, _e])
                wp = env.c_paths[_ar, _e, _m, min(k, CML - 1)]
                nx.scatter_(1, slc, torch.where(adv.unsqueeze(1), wp[:, 0:1], nx.gather(1, slc)))
                ny.scatter_(1, slc, torch.where(adv.unsqueeze(1), wp[:, 1:2], ny.gather(1, slc)))
                _comet_expired.scatter_(1, slc, exp.unsqueeze(1) | _comet_expired.gather(1, slc))
                env.c_slot[:, _e, _m] = torch.where(exp, torch.full_like(sl, -1), sl)
    # ---- fleet movement + swept-pair collision ----
    falive = env.f_alive > 0.5
    speed = fleet_speed_t(env.f_ships, env.vmax)
    fox, foy = env.f_x, env.f_y
    fnx = fox + torch.cos(env.f_angle) * speed
    fny = foy + torch.sin(env.f_angle) * speed
    Ax = fox.unsqueeze(2); Ay = foy.unsqueeze(2); Bx = fnx.unsqueeze(2); By = fny.unsqueeze(2)
    P0x = old_px.unsqueeze(1); P0y = old_py.unsqueeze(1); P1x = nx.unsqueeze(1); P1y = ny.unsqueeze(1)
    rad = env.p_radius.unsqueeze(1); palive = (env.p_alive.unsqueeze(1) > 0.5)
    d0x = Ax - P0x; d0y = Ay - P0y
    dvx = (Bx - Ax) - (P1x - P0x); dvy = (By - Ay) - (P1y - P0y)
    a = dvx * dvx + dvy * dvy; b = 2.0 * (d0x * dvx + d0y * dvy); c = d0x * d0x + d0y * d0y - rad * rad
    disc = b * b - 4.0 * a * c; sq = torch.sqrt(disc.clamp_min(0.0))
    t1 = (-b - sq) / (2.0 * a); t2 = (-b + sq) / (2.0 * a)
    hit = torch.where(a < 1e-12, (a < 1e-12) & (c <= 0.0), (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0))
    hit = hit & palive & falive.unsqueeze(2)
    slotf = torch.arange(Ec, dtype=DT, device=dev).view(1, 1, Ec)
    fh_v, tgt_slot = torch.where(hit, slotf, torch.full_like(slotf, float(Ec))).min(2)
    has_hit = fh_v < float(Ec)
    oob = (fnx < 0.0) | (fnx > BOARD) | (fny < 0.0) | (fny > BOARD)
    l2 = (fox - fnx) ** 2 + (foy - fny) ** 2
    tt = (((CENTER - fox) * (fnx - fox) + (CENTER - foy) * (fny - foy)) / l2.clamp_min(1e-12)).clamp(0, 1)
    prx = fox + tt * (fnx - fox); pry = foy + tt * (fny - foy)
    sun_hit = torch.where(l2 > 0.0, (l2 > 0.0) & (torch.sqrt((CENTER - prx) ** 2 + (CENTER - pry) ** 2) < SUN),
                          torch.sqrt((CENTER - fox) ** 2 + (CENTER - foy) ** 2) < SUN)
    remove_fleet = falive & (has_hit | oob | sun_hit)
    contributes = falive & has_hit
    # ---- combat ----
    arr0 = torch.zeros(B, Ec, dtype=DT, device=dev); arr1 = torch.zeros(B, Ec, dtype=DT, device=dev)
    cf = contributes.to(DT)
    tslot = tgt_slot.clamp(0, Ec - 1)
    arr0.scatter_add_(1, tslot, env.f_ships * cf * (env.f_owner == 0.0).to(DT))
    arr1.scatter_add_(1, tslot, env.f_ships * cf * (env.f_owner == 1.0).to(DT))
    has0 = arr0 > 0.0; has1 = arr1 > 0.0
    top = torch.maximum(arr0, arr1); second = torch.minimum(arr0, arr1)
    both = has0 & has1
    surv = torch.where(both, top - second, top)
    surv = torch.where(both & (arr0 == arr1), torch.zeros_like(surv), surv)
    surv_owner = torch.full((B, Ec), -1.0, dtype=DT, device=dev)
    surv_owner = torch.where(arr0 > arr1, torch.zeros_like(surv_owner), surv_owner)
    surv_owner = torch.where(arr1 > arr0, torch.ones_like(surv_owner), surv_owner)
    apply = (has0 | has1) & (surv > 0.0) & (env.p_alive > 0.5)
    same = (env.p_owner == surv_owner)
    reinforce = apply & same; attack = apply & (~same)
    env.p_ships = torch.where(reinforce, env.p_ships + surv, env.p_ships)
    after = env.p_ships - surv
    flips = attack & (after < 0.0)
    env.p_ships = torch.where(attack, torch.where(after < 0.0, -after, after), env.p_ships)
    env.p_owner = torch.where(flips, surv_owner, env.p_owner)
    # ---- apply positions; comet expiry; clear removed fleets ----
    env.p_x = nx; env.p_y = ny
    if g["COMETS_ENABLED"] and _comet_expired is not None:
        gone = _comet_expired & (env.p_alive > 0.5)
        keepc = (~gone).to(DT)
        env.p_alive = env.p_alive * keepc; env.p_is_comet = env.p_is_comet * keepc
        env.p_comet_vx = env.p_comet_vx * keepc; env.p_comet_vy = env.p_comet_vy * keepc
        env.p_ships = env.p_ships * keepc
        env.p_owner = torch.where(gone, torch.full_like(env.p_owner, -1.0), env.p_owner)
    keep = (falive & (~remove_fleet)).to(DT)
    env.f_alive = keep; env.f_owner = env.f_owner * keep
    env.f_x = fnx * keep; env.f_y = fny * keep; env.f_angle = env.f_angle * keep
    env.f_ships = env.f_ships * keep
    env.step_ct = env.step_ct + 1.0


# ------------------------------------------------------------------ state comparison
def compare_state(env, g, obs, t_next, world, tol_pos=2e-2, tol_ships=0.5):
    """Mismatches between the replay state (after tick t_next-1) and official obs at t_next.
    Official shows freshly-spawned comets at path[0] one obs EARLIER than our env materialises
    them (env_step spawns at the start of the next tick) -> verify them against path[0] and
    exclude from the alive-set check. Returns (bad list, max_pos_diff, max_ship_diff)."""
    ST = list(g["COMET_SPAWN_STEPS"])
    PC = g["PLANET_CAP"]
    off = {int(p[0]): p for p in _g(obs, "planets")}
    bad = []
    pending = set()
    if t_next in ST:
        e = ST.index(t_next)
        L = int(world["comet_len"][e])
        if L > 0:
            cids = sorted(set(int(c) for c in (_g(obs, "comet_planet_ids") or [])) & set(off.keys()))
            new4 = cids[-4:]
            for m, pid in enumerate(new4):
                p = off[pid]
                ex = float(world["comet_paths"][e, m, 0, 0]); ey = float(world["comet_paths"][e, m, 0, 1])
                if abs(p[2] - ex) > tol_pos or abs(p[3] - ey) > tol_pos:
                    bad.append("comet-spawn p%d off=(%.4f,%.4f) path0=(%.4f,%.4f)" % (pid, p[2], p[3], ex, ey))
                if abs(p[5] - float(world["comet_ships"][e])) > tol_ships:
                    bad.append("comet-spawn p%d ships off=%s ours=%s" % (pid, p[5], world["comet_ships"][e]))
                pending.add(pid)
    ours_alive = set(s for s in range(PC) if env.p_alive[0, s].item() > 0.5)
    off_ids = set(off.keys()) - pending
    if ours_alive != off_ids:
        bad.append("alive-set off-only=%s ours-only=%s" % (sorted(off_ids - ours_alive), sorted(ours_alive - off_ids)))
    mpd = msd = 0.0
    for pid in sorted(off_ids & ours_alive):
        p = off[pid]
        o_owner = int(env.p_owner[0, pid].item()); o_ships = env.p_ships[0, pid].item()
        o_x = env.p_x[0, pid].item(); o_y = env.p_y[0, pid].item()
        if int(p[1]) != o_owner:
            bad.append("p%d owner off=%d ours=%d" % (pid, int(p[1]), o_owner))
        sd = abs(p[5] - o_ships); msd = max(msd, sd)
        if sd > tol_ships:
            bad.append("p%d ships off=%s ours=%.1f" % (pid, p[5], o_ships))
        pd = max(abs(p[2] - o_x), abs(p[3] - o_y)); mpd = max(mpd, pd)
        if pd > tol_pos:
            bad.append("p%d pos off=(%.4f,%.4f) ours=(%.4f,%.4f)" % (pid, p[2], p[3], o_x, o_y))
    off_fleets = _g(obs, "fleets") or []
    n_ours = int((env.f_alive[0] > 0.5).sum().item())
    if len(off_fleets) != n_ours:
        bad.append("fleet-count off=%d ours=%d" % (len(off_fleets), n_ours))
    else:
        # per-owner ship totals (order-free)
        for ownr in (0, 1):
            so = sum(f[6] for f in off_fleets if int(f[1]) == ownr)
            su = (env.f_ships[0] * (env.f_owner[0] == float(ownr)) * (env.f_alive[0] > 0.5)).sum().item()
            if abs(so - su) > tol_ships:
                bad.append("fleet-ships owner%d off=%s ours=%.1f" % (ownr, so, su))
    return bad, mpd, msd


# ------------------------------------------------------------------ C. heuristic recompute
def recompute_moves(env, g, code, player):
    """Re-run opponent_action(code) on the free-running replay state for `player`
    (opponent_action acts as owner 1 -> swap owners for player 0)."""
    v = types.SimpleNamespace()
    for k in ("B", "Ec", "Fc", "dev", "vmax", "T",
              "p_alive", "p_x", "p_y", "p_radius", "p_ships", "p_prod", "p_is_comet",
              "p_rotates", "p_init_x", "p_init_y", "ang_vel", "step_ct",
              "f_alive", "f_x", "f_y", "f_angle", "f_ships", "f_seq"):
        setattr(v, k, getattr(env, k))
    if player == 1:
        v.p_owner, v.f_owner = env.p_owner, env.f_owner
    else:
        po = env.p_owner
        v.p_owner = torch.where(po == 0.0, torch.ones_like(po),
                                torch.where(po == 1.0, torch.zeros_like(po), po))
        v.f_owner = torch.where(env.f_alive > 0.5, 1.0 - env.f_owner, env.f_owner)
    with torch.no_grad():
        ang, shp, com = g["opponent_action"](v, code)
    moves = []
    for s in range(env.Ec):
        if com[0, s].item() > 0.5:
            n = int(math.floor(shp[0, s].item()))
            if n > 0:
                moves.append([s, float(ang[0, s].item()), n])
    return moves


def moves_match(rec, rc, ang_tol=1e-4):
    """Structural: same (pid, ships) sequence; angles within ang_tol. Returns (ok, max_dang)."""
    rec = rec or []
    if len(rec) != len(rc):
        return False, float("inf")
    mx = 0.0
    for a, b in zip(rec, rc):
        if int(a[0]) != int(b[0]) or int(a[2]) != int(b[2]):
            return False, float("inf")
        d = abs(float(a[1]) - float(b[1]))
        d = min(d, abs(d - 2.0 * math.pi))
        mx = max(mx, d)
    return mx <= ang_tol, mx


# ------------------------------------------------------------------ B+C driver
def run_game_parity(g, sd, kindA, kindB, steps):
    from kaggle_environments import make
    codeA, codeB = EV.OPP_CODE[kindA], EV.OPP_CODE[kindB]
    agA = EV.make_scripted_agent(g, codeA)
    agB = EV.make_scripted_agent(g, codeB)
    kenv = make("orbit_wars", configuration={"episodeSteps": steps, "seed": sd}, debug=False)
    out = kenv.run([agA, agB])

    world = g["generate_world"](sd)
    env = g["GpuEnv"](g["PLANET_CAP"], g["FLEET_CAP"], steps, g["SHIP_SPEED"], torch.device("cpu"))
    env.reset([world])

    n_steps = len(out) - 1
    h_total = h_bad = 0
    h_max_dang = 0.0
    mpd = msd = 0.0
    first_div = None
    for t in range(n_steps):
        spawn_phase(env, g)
        for pl, code in ((0, codeA), (1, codeB)):
            rec = out[t + 1][pl]["action"]
            rc = recompute_moves(env, g, code, pl)
            ok, dang = moves_match(rec, rc)
            h_total += 1
            if not ok:
                h_bad += 1
            elif dang < float("inf"):
                h_max_dang = max(h_max_dang, dang)
        step_explicit_v7(env, g, [out[t + 1][0]["action"], out[t + 1][1]["action"]])
        bad, pd, sdd = compare_state(env, g, out[t + 1][0]["observation"], t + 1, world)
        mpd = max(mpd, pd); msd = max(msd, sdd)
        if bad:
            first_div = (t, bad)
            break
    tag = "%s-vs-%s seed %d" % (kindA, kindB, sd)
    if first_div is None:
        print("  %-34s MATCH all %3d steps | max|pos|=%.2e max|ships|=%.2e | heuristic %d/%d steps exact (max dAngle %.1e)"
              % (tag, n_steps, mpd, msd, h_total - h_bad, h_total, h_max_dang))
        return True, h_bad, h_total
    t, bad = first_div
    print("  %-34s DIVERGED at step %d -> %s" % (tag, t, "; ".join(bad[:5])))
    return False, h_bad, h_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", default=EV.NB_DEFAULT)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--worldgen-seeds", type=int, default=40)
    args = ap.parse_args()
    os.chdir(REPO)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    print("loading v7 defs from %s ..." % args.nb)
    g = EV.load_nb_defs(args.nb)
    g["DEVICE"] = torch.device("cpu")
    assert g["COMET_OFFICIAL"] and g["COMETS_ENABLED"], "v7 nb must have official comets on"

    ok_a = test_worldgen(g, args.worldgen_seeds)

    seeds = [int(s) for s in args.seeds.split(",")]
    print("\n[B+C] full-game replay parity (official engine, scripted-vs-scripted, comets ON):")
    n_ok = n_run = 0
    hb = ht = 0
    for sd in seeds:
        for kindA, kindB in MATCHUPS:
            ok, b, t = run_game_parity(g, sd, kindA, kindB, args.steps)
            n_ok += int(ok); n_run += 1
            hb += b; ht += t
    print("\nSUMMARY: world-gen %s | dynamics %d/%d games tick-exact | heuristic launches %d/%d agent-steps exact"
          % ("EXACT" if ok_a else "MISMATCH", n_ok, n_run, ht - hb, ht))


if __name__ == "__main__":
    main()
