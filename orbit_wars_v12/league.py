"""Dual-Elo self-play league + scripted/neural evaluation helpers (notebook cells 28 + 29).

Every member carries a 2p (``elo``) and a 4p (``elo4``) rating; the learner is grounded on a
ROLLING self-anchor (its own frozen high-water snapshot) so the rating never clamps on saturated
scripts. Scripted/invited members are rating-anchored; learner snapshots are PFSP-sampled. Recals
BLEND into the live ratings (first=raw) and run a MEASURED eviction of mastered scripts at start
and at every recalibration (v12). All hyperparameters come from ``self.cfg``.
"""
import copy
import math
import os

import torch

from .checkpoint import (build_from_cfg, build_policy, discover_invited, load_snapshot,
                         save_ckpt, _freeze_snapshot, _unwrap, _wrap_if_legacy)
from .constants import DTYPE, PLANET_CAP
from .env import GpuEnv, env_encode, env_step, settle, settle_n, _OPP_BY_KIND
from .policy import act
from .worldgen import make_world_pool

# ---- fixed FFA trio / calibration kinds ----------------------------------------------------
GAUNTLET4_SEATS = ("starter", "greedy", "intermediate")   # fixed FFA trio for the 4p gauntlet
_CAL4_KINDS = ("random", "starter", "intermediate", "greedy", "medium")

# ---- rolling self-anchor recal constants + primitives --------------------------------------
SELF_KIND = "selfanchor"
PROMOTE_S2 = 0.80    # 2p seat-avg score vs the top anchor required to promote
PROMOTE_WR4 = 0.40   # 4p 1st-place win-rate vs the top anchor required to promote
ANCHOR_KEEP = 8      # max self-anchors kept resident (top rungs)
GREEDY_WATCHDOG = 0.85   # alarm if learner 2p score vs greedy falls below this
RECAL_CLAMP = (0.02, 0.98)


def eval_gauntlet(cfg, net, worlds, anchors=("starter", "intermediate", "greedy"), n_envs=None):
    """Fixed-opponent 2p score for best-ckpt selection (greedy, deployment-faithful op set)."""
    n = n_envs or cfg.ELO_RECAL_ENVS
    return sum(_eval_score(cfg, net, a, worlds, n) for a in anchors) / len(anchors)


def _eval_score(cfg, net, anchor_kind, worlds, n_envs):
    """No-grad mean learner-score (win + 0.5*draw) of `net` vs a scripted anchor on `worlds` (greedy, 2p)."""
    dev = cfg.device
    env = GpuEnv(cfg)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)])
    active = torch.ones(n_envs, device=dev); score = torch.zeros(n_envs, device=dev)
    def _sc(s0, s1):
        return torch.where(s0 > s1, torch.ones_like(s0), torch.where(s0 < s1, torch.zeros_like(s0), torch.full_like(s0, 0.5)))
    with torch.no_grad():
        for _t in range(env.T):
            ent, em, am, gl = env_encode(env, 0)
            a_t, _, _ = act(cfg, net, ent, em, am, gl, greedy=True)
            env_step(cfg, env, a_t, _OPP_BY_KIND[anchor_kind], None, step_idx=_t)
            s0, s1, a0, a1 = settle(env)
            term = (env.step_ct >= float(env.T - 2)) | (~(a0 & a1))
            newly = (active > 0.5) & term
            score = torch.where(newly, _sc(s0, s1), score)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s0, s1, _, _ = settle(env)
        score = torch.where(active > 0.5, _sc(s0, s1), score)
    return float(score.mean().item())


def _eval_score4(cfg, net, trio, worlds, n_envs):
    """No-grad mean PAIRWISE score of greedy `net` (seat 0) vs 3 scripted anchors in a 4p FFA."""
    dev = cfg.device
    env = GpuEnv(cfg)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)], n_players=4)
    seats = [{"pid": p + 1, "script": _OPP_BY_KIND[k]} for p, k in enumerate(trio)]
    active = torch.ones(n_envs, device=dev); score = torch.zeros(n_envs, device=dev)
    def _pw(s_all):
        s0 = s_all[:, 0:1].expand_as(s_all[:, 1:]); rest = s_all[:, 1:]
        one = torch.ones_like(rest)
        return torch.where(s0 > rest, one, torch.where(s0 < rest, 0.0 * one, 0.5 * one)).mean(1)
    with torch.no_grad():
        for _t in range(env.T):
            ent, em, am, gl = env_encode(env, 0)
            a_t, _, _ = act(cfg, net, ent, em, am, gl, greedy=True)
            env_step(cfg, env, a_t, seats=seats, step_idx=_t)
            s_all, alive_all = settle_n(env)
            n_alive = alive_all.to(DTYPE).sum(1)
            term = (env.step_ct >= float(env.T - 2)) | (n_alive <= 1.0) | (~alive_all[:, 0])
            newly = (active > 0.5) & term
            score = torch.where(newly, _pw(s_all), score)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s_all, _ = settle_n(env)
        score = torch.where(active > 0.5, _pw(s_all), score)
    return float(score.mean().item())


def _eval_winrate4(cfg, net, kind, worlds, n_envs):
    """4p FFA 1st-place (outright-win) rate of greedy `net` (seat 0) vs THREE copies of
    scripted `kind` (seats 1-3). Baseline 0.25 when all four are equal."""
    dev = cfg.device
    env = GpuEnv(cfg)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)], n_players=4)
    seats = [{"pid": p + 1, "script": _OPP_BY_KIND[kind]} for p in range(3)]
    active = torch.ones(n_envs, device=dev); win = torch.zeros(n_envs, device=dev)
    def _win(s_all):
        s0 = s_all[:, 0:1]; rest = s_all[:, 1:]
        return (s0 > rest).all(1).to(s_all.dtype)
    with torch.no_grad():
        for _t in range(env.T):
            ent, em, am, gl = env_encode(env, 0)
            a_t, _, _ = act(cfg, net, ent, em, am, gl, greedy=True)
            env_step(cfg, env, a_t, seats=seats, step_idx=_t)
            s_all, alive_all = settle_n(env)
            n_alive = alive_all.to(DTYPE).sum(1)
            term = (env.step_ct >= float(env.T - 2)) | (n_alive <= 1.0) | (~alive_all[:, 0])
            newly = (active > 0.5) & term
            win = torch.where(newly, _win(s_all), win)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s_all, _ = settle_n(env)
        win = torch.where(active > 0.5, _win(s_all), win)
    return float(win.mean().item())


def _eval_winrate4_net(cfg, net, opp_net, worlds, n_envs):
    """4p FFA 1st-place (outright-win) rate of greedy `net` (seat 0) vs THREE copies of the NEURAL
    `opp_net` (seats 1-3) -- the neural twin of `_eval_winrate4`, so league players are eviction-tested
    on the identical standard as scripted anchors. Baseline 0.25 when all four are equal."""
    dev = cfg.device
    env = GpuEnv(cfg)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)], n_players=4)
    active = torch.ones(n_envs, device=dev); win = torch.zeros(n_envs, device=dev)
    def _win(s_all):
        s0 = s_all[:, 0:1]; rest = s_all[:, 1:]
        return (s0 > rest).all(1).to(s_all.dtype)
    with torch.no_grad():
        for _t in range(env.T):
            ent, em, am, gl = env_encode(env, 0)
            a_t, _, _ = act(cfg, net, ent, em, am, gl, greedy=True)
            step_seats = []
            for pid in (1, 2, 3):
                oe, om, oa, og = env_encode(env, pid)
                o_act, _, _ = act(cfg, opp_net, oe, om, oa, og, greedy=True)
                step_seats.append({"pid": pid, "action": o_act})
            env_step(cfg, env, a_t, seats=step_seats, step_idx=_t)
            s_all, alive_all = settle_n(env)
            n_alive = alive_all.to(DTYPE).sum(1)
            term = (env.step_ct >= float(env.T - 2)) | (n_alive <= 1.0) | (~alive_all[:, 0])
            newly = (active > 0.5) & term
            win = torch.where(newly, _win(s_all), win)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s_all, _ = settle_n(env)
        win = torch.where(active > 0.5, _win(s_all), win)
    return float(win.mean().item())


def _eval_pair_scores4(cfg, net, trio, worlds, n_envs):
    """4p FFA with greedy `net` at seat 0 and the scripted `trio` at seats 1..3. Returns
    {(i, j): mean pairwise score of seat i vs seat j} for the pairs among seats 1..3 only
    (the ego is just part of the arena), frozen per env at termination."""
    dev = cfg.device
    env = GpuEnv(cfg)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)], n_players=4)
    seats = [{"pid": p + 1, "script": _OPP_BY_KIND[k]} for p, k in enumerate(trio)]
    pairs = ((1, 2), (1, 3), (2, 3))
    active = torch.ones(n_envs, device=dev)
    psc = torch.zeros(len(pairs), n_envs, device=dev)
    def _ps(s_all):
        out = []
        for (i, j) in pairs:
            si, sj = s_all[:, i], s_all[:, j]
            out.append(torch.where(si > sj, torch.ones_like(si),
                                   torch.where(si < sj, torch.zeros_like(si),
                                               torch.full_like(si, 0.5))))
        return out
    with torch.no_grad():
        for _t in range(env.T):
            ent, em, am, gl = env_encode(env, 0)
            a_t, _, _ = act(cfg, net, ent, em, am, gl, greedy=True)
            env_step(cfg, env, a_t, seats=seats, step_idx=_t)
            s_all, alive_all = settle_n(env)
            n_alive = alive_all.to(DTYPE).sum(1)
            term = (env.step_ct >= float(env.T - 2)) | (n_alive <= 1.0) | (~alive_all[:, 0])
            newly = (active > 0.5) & term
            for k, p in enumerate(_ps(s_all)):
                psc[k] = torch.where(newly, p, psc[k])
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s_all, _ = settle_n(env)
        still = active > 0.5
        for k, p in enumerate(_ps(s_all)):
            psc[k] = torch.where(still, p, psc[k])
    return {pairs[k]: float(psc[k].mean().item()) for k in range(len(pairs))}


def _js_div(p, q):
    m = 0.5 * (p + q); eps = 1e-9
    kl = lambda a, b: (a * (torch.log(a + eps) - torch.log(b + eps))).sum()
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def _recal_link(cfg, base, s):
    """Elo of an agent scoring `s` (in [0,1], pairwise-style 0.5=even) vs a reference at `base`."""
    s = min(max(s, RECAL_CLAMP[0]), RECAL_CLAMP[1])
    return base + cfg.ELO_SCALE * math.log10(s / (1.0 - s))


def _score2_vs(cfg, net, opp_net, worlds, n):
    """Seat-AVERAGED greedy 2p score of `net` vs frozen `opp_net` (kills the seat-0 bias)."""
    dev = cfg.device
    def _one(a_net, b_net):
        env = GpuEnv(cfg)
        env.reset([worlds[i % len(worlds)] for i in range(n)], n_players=2)
        active = torch.ones(n, device=dev); score = torch.zeros(n, device=dev)
        def _sc(s0, s1):
            return torch.where(s0 > s1, torch.ones_like(s0),
                               torch.where(s0 < s1, torch.zeros_like(s0), 0.5 * torch.ones_like(s0)))
        with torch.no_grad():
            for t in range(env.T):
                e, m, a, g = env_encode(env, 0); a0, _, _ = act(cfg, a_net, e, m, a, g, greedy=True)
                oe, om, oa, og = env_encode(env, 1); a1, _, _ = act(cfg, b_net, oe, om, oa, og, greedy=True)
                env_step(cfg, env, a0, seats=[{"pid": 1, "action": a1}], step_idx=t)
                s0, s1, al0, al1 = settle(env)
                term = (env.step_ct >= float(env.T - 2)) | (~(al0 & al1))
                newly = (active > 0.5) & term
                score = torch.where(newly, _sc(s0, s1), score)
                active = torch.where(term, torch.zeros_like(active), active)
                if active.sum().item() == 0:
                    break
            s0, s1, _, _ = settle(env)
            score = torch.where(active > 0.5, _sc(s0, s1), score)
        return float(score.mean().item())
    ab = _one(net, opp_net)
    ba = _one(opp_net, net)
    return 0.5 * (ab + (1.0 - ba))


def _eval4_vs(cfg, net, anc_net, worlds, n):
    """4p FFA: seat0=net, seat1=anchor, seats 2,3 = greedy/intermediate scripts.
    Returns (winrate_1st, pairwise_vs_anchor) -- gate on the win-rate, rate the chain on pairwise."""
    dev = cfg.device
    env = GpuEnv(cfg)
    env.reset([worlds[i % len(worlds)] for i in range(n)], n_players=4)
    fill = (_OPP_BY_KIND[GAUNTLET4_SEATS[1]], _OPP_BY_KIND[GAUNTLET4_SEATS[2]])   # greedy, intermediate
    active = torch.ones(n, device=dev)
    wr = torch.zeros(n, device=dev); pw = torch.zeros(n, device=dev)
    def _eval(s_all):
        s0 = s_all[:, 0]; a1 = s_all[:, 1]
        first = (s0 > s_all[:, 1:].max(1).values).to(DTYPE)                       # strict 1st place
        pwa = torch.where(s0 > a1, torch.ones_like(s0),
                          torch.where(s0 < a1, torch.zeros_like(s0), 0.5 * torch.ones_like(s0)))
        return first, pwa
    with torch.no_grad():
        for t in range(env.T):
            e, m, a, g = env_encode(env, 0); a0, _, _ = act(cfg, net, e, m, a, g, greedy=True)
            oe, om, oa, og = env_encode(env, 1); a1, _, _ = act(cfg, anc_net, oe, om, oa, og, greedy=True)
            seats = [{"pid": 1, "action": a1}, {"pid": 2, "script": fill[0]}, {"pid": 3, "script": fill[1]}]
            env_step(cfg, env, a0, seats=seats, step_idx=t)
            s_all, alive_all = settle_n(env)
            n_alive = alive_all.to(DTYPE).sum(1)
            term = (env.step_ct >= float(env.T - 2)) | (n_alive <= 1.0) | (~alive_all[:, 0])
            newly = (active > 0.5) & term
            f_, p_ = _eval(s_all)
            wr = torch.where(newly, f_, wr); pw = torch.where(newly, p_, pw)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s_all, _ = settle_n(env); f_, p_ = _eval(s_all)
        wr = torch.where(active > 0.5, f_, wr); pw = torch.where(active > 0.5, p_, pw)
    return float(wr.mean().item()), float(pw.mean().item())


_NEURAL_G4 = {"nets": None}


def _neural_gauntlet_nets(cfg):
    """Top invited ckpts as a FIXED neural opponent trio (lazy-loaded, cached for the run)."""
    if _NEURAL_G4["nets"] is None:
        nets = []
        for p, _aelo in discover_invited(cfg)[:3]:
            try:
                n_, _cfg = load_snapshot(cfg, p, fallback_cfg=cfg.INVITE_CFG)
                nets.append(n_)
            except Exception as e:
                print("    [gauntlet4n] skip %s -> %r" % (os.path.basename(p), e))
        _NEURAL_G4["nets"] = nets
    return _NEURAL_G4["nets"]


def _eval_score4_vs(cfg, net, opp_nets, worlds, n_envs):
    """Mean pairwise score of greedy `net` (seat 0) in a 4p FFA where seats 1..3 are the
    given neural nets, padded with the GAUNTLET4_SEATS scripts when fewer than 3."""
    dev = cfg.device
    env = GpuEnv(cfg)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)], n_players=4)
    seats = []
    for i in range(3):
        if i < len(opp_nets):
            seats.append({"pid": i + 1, "net": opp_nets[i]})
        else:
            seats.append({"pid": i + 1, "script": _OPP_BY_KIND[GAUNTLET4_SEATS[i]]})
    active = torch.ones(n_envs, device=dev); score = torch.zeros(n_envs, device=dev)
    def _pw(s_all):
        s0 = s_all[:, 0:1].expand_as(s_all[:, 1:]); rest = s_all[:, 1:]
        one = torch.ones_like(rest)
        return torch.where(s0 > rest, one, torch.where(s0 < rest, 0.0 * one, 0.5 * one)).mean(1)
    with torch.no_grad():
        for _t in range(env.T):
            ent, em, am, gl = env_encode(env, 0)
            a_t, _, _ = act(cfg, net, ent, em, am, gl, greedy=True)
            step_seats = []
            for sp in seats:
                if "net" in sp:
                    oe, om, oa, og = env_encode(env, sp["pid"])
                    o_act, _, _ = act(cfg, sp["net"], oe, om, oa, og, greedy=True)
                    step_seats.append({"pid": sp["pid"], "action": o_act})
                else:
                    step_seats.append(sp)
            env_step(cfg, env, a_t, seats=step_seats, step_idx=_t)
            s_all, alive_all = settle_n(env)
            n_alive = alive_all.to(DTYPE).sum(1)
            term = (env.step_ct >= float(env.T - 2)) | (n_alive <= 1.0) | (~alive_all[:, 0])
            newly = (active > 0.5) & term
            score = torch.where(newly, _pw(s_all), score)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s_all, _ = settle_n(env)
        score = torch.where(active > 0.5, _pw(s_all), score)
    return float(score.mean().item())


def eval_gauntlet4(cfg, net, worlds, n_envs=None):
    """Mixed 4p best-ckpt gauntlet: scripted trio FFA blended with an invited-neural FFA
    (falls back to scripted-only when no invited members are present)."""
    n = n_envs or cfg.ELO_RECAL_ENVS
    g_s = _eval_score4(cfg, net, GAUNTLET4_SEATS, worlds, n)
    nets = _neural_gauntlet_nets(cfg)
    if not nets:
        return g_s
    for n_ in nets:
        n_.to(cfg.device)
    g_n = _eval_score4_vs(cfg, net, nets, worlds, n)
    for n_ in nets:
        n_.to('cpu')
    return (1.0 - cfg.GAUNTLET_NEURAL_W) * g_s + cfg.GAUNTLET_NEURAL_W * g_n


class League:
    """Dual-Elo self-play league: every member carries a 2p (elo) AND a 4p (elo4)
    rating; the learner is grounded on a ROLLING self-anchor (its own frozen high-water
    snapshot) so the rating never clamps on saturated scripts. Scripted/invited members
    are rating-anchored; snapshots are PFSP-sampled. Recals blend (first=raw) and run a
    MEASURED eviction of mastered scripts at start and every recalibration."""

    def __init__(self, cfg, rng, invite=True):
        self.cfg = cfg
        self.rng = rng
        self.learner_elo = cfg.ELO_INIT
        self.learner_elo4 = cfg.ELO_INIT
        self.advanced = False                                # curriculum: greedy/medium locked until the advance gate
        def anc(label, kind, elo):
            return {"label": label, "net": None, "kind": kind, "elo": elo, "elo4": elo,
                    "anchor": True, "pinned": True, "n": 0, "n4": 0}
        self.members = [
            anc("random", "random", cfg.ELO_RANDOM),
            anc("starter", "starter", cfg.ELO_STARTER),
            anc("greedy", "greedy", cfg.ELO_GREEDY),
            anc("medium", "medium", cfg.ELO_MEDIUM),
            anc("intermediate", "intermediate", cfg.ELO_INTERMEDIATE),
        ]
        if invite:
            for path, aelo in discover_invited(cfg):
                try:
                    self.add_invited(path, aelo)
                except Exception as e:
                    print("  !! invite failed %s -> %r" % (path, e))

    def add_invited(self, path, aelo):
        net, member_cfg = load_snapshot(self.cfg, path, fallback_cfg=self.cfg.INVITE_CFG)
        lab = os.path.splitext(os.path.basename(path))[0]
        m = {"label": "inv:" + lab, "net": net, "kind": "invited", "cfg": member_cfg,
             "elo": float(aelo), "elo4": float(aelo) + self.cfg.INVITE_ELO4_OFFSET,
             "anchor": True, "pinned": True, "n": 0, "n4": 0}
        self.members.append(m)
        print("    invited %-26s provisional elo2 %6.0f / elo4 %6.0f (measured at train start)  (%s)"
              % (m["label"], m["elo"], m["elo4"], os.path.basename(path)))
        return m

    def snapshots(self):
        return [m for m in self.members if m["kind"] == "snapshot"]

    def anchor(self, kind):
        return next(m for m in self.members if m["kind"] == kind)

    def add_checkpoint(self, path, label=None, elo=None):
        net, member_cfg = load_snapshot(self.cfg, path)
        m = {"label": label or os.path.basename(path), "net": net, "kind": "snapshot", "cfg": member_cfg,
             "elo": self.cfg.ELO_INIT if elo is None else elo, "elo4": self.cfg.ELO_INIT if elo is None else elo,
             "anchor": False, "pinned": True, "n": 0, "n4": 0}
        self.members.append(m)
        return m

    def add_learner_snapshot(self, net, it):
        snap = _freeze_snapshot(copy.deepcopy(net))
        m = {"label": "it%d" % it, "net": snap, "kind": "snapshot",
             "elo": self.learner_elo, "elo4": self.learner_elo4,
             "anchor": False, "pinned": False, "n": 0, "n4": 0}
        self.members.append(m)
        self._prune()
        try:  # persist the weights as soon as the snapshot is generated (survives pruning)
            d = os.path.join(self.cfg.CKPT_DIR, "league_agents"); os.makedirs(d, exist_ok=True)
            save_ckpt(self.cfg, snap, os.path.join(d, m["label"] + ".pt"),
                      meta={"iter": it, "elo": float(m["elo"]), "elo4": float(m["elo4"])})
        except Exception as e:
            print("  !! league snapshot save failed: %r" % e)

    def _prune(self):
        """Keep LEAGUE_MAX_SNAPSHOTS most distinct auto-snapshots: repeatedly drop the one whose
        NEAREST neighbor (dest+gate Jensen-Shannon divergence) is closest = most redundant.
        Anchors / pinned ckpts and the NEWEST snapshot are always kept. FIFO fallback on error."""
        autos = [m for m in self.members if m['kind'] == 'snapshot' and not m['pinned']]
        if len(autos) <= self.cfg.LEAGUE_MAX_SNAPSHOTS:
            return
        try:
            fps = [self._fingerprint(m['net']) for m in autos]
        except Exception as e:
            print('  !! league diversity-prune -> FIFO fallback: %r' % e)
            for m in autos[:len(autos) - self.cfg.LEAGUE_MAX_SNAPSHOTS]:
                self.members.remove(m)
            return
        def d(i, j):
            return _js_div(fps[i][0], fps[j][0]) + _js_div(fps[i][1], fps[j][1])
        newest = len(autos) - 1                      # most recently appended -> always keep
        keep = list(range(len(autos)))
        while len(keep) > self.cfg.LEAGUE_MAX_SNAPSHOTS:
            drop, drop_nn = None, float('inf')
            for i in keep:
                if i == newest:
                    continue
                nn = min(d(i, j) for j in keep if j != i)
                if nn < drop_nn:
                    drop_nn, drop = nn, i
            if drop is None:
                break
            keep.remove(drop)
        kept = set(keep)
        for idx, m in enumerate(autos):
            if idx not in kept:
                self.members.remove(m)
        print('  league diversity-prune: kept %d/%d snapshots (max behavioral distinctness)'
              % (len(kept), len(autos)))

    def _p_beat(self, m, fmt):
        me = self.learner_elo if fmt == 2 else self.learner_elo4
        oe = m["elo"] if fmt == 2 else m["elo4"]
        return 1.0 / (1.0 + 10.0 ** ((oe - me) / self.cfg.ELO_SCALE))

    def _weights(self, fmt, net=None):
        """Per-member sampling weight for one format: PFSP(expected score) x footprint mix
        (2p only) x invited/scripted boosts x advance lock, then the scripted-share cap and the
        watchdog / 2p-mastered gates. (Weak-point boost + dominated-bench removed in v12.)"""
        cfg = self.cfg
        members = self.members
        elo_w = [self._pfsp_weight(self._p_beat(m, fmt)) for m in members]
        ws = list(elo_w)
        if fmt == 2 and net is not None and cfg.MATCH_FP_W > 0.0:
            try:
                lfp = self._fingerprint(net)
                fd = []
                for m in members:
                    if m["net"] is None:
                        fd.append(1.0)                                  # scripted anchor: maximally distinct
                    else:
                        if m.get("fp") is None:
                            m["fp"] = self._fingerprint(m["net"])
                        fd.append(_js_div(lfp[0], m["fp"][0]) + _js_div(lfp[1], m["fp"][1]))
                mxe = max(elo_w) or 1.0; mxf = max(fd) or 1.0
                ws = [cfg.MATCH_ELO_W * (elo_w[i] / mxe) + cfg.MATCH_FP_W * (fd[i] / mxf) + cfg.PFSP_FLOOR
                      for i in range(len(members))]
            except Exception:
                ws = list(elo_w)
        for i, m in enumerate(members):
            if m["kind"] == "invited":
                ws[i] *= cfg.INVITE_MATCH_BOOST if not self._mastered(m, fmt, cfg.INVITE_MASTER_WR) else cfg.INVITE_MASTERED_W
            elif m["net"] is None and cfg.SCRIPT_MATCH_BOOST != 1.0:
                if not self._mastered(m, fmt, cfg.SCRIPT_BOOST_DROP_WR):
                    ws[i] *= cfg.SCRIPT_MATCH_BOOST              # spend compute on UNMASTERED scripted anchors
            if (not self.advanced) and m["kind"] in ("greedy", "medium"):
                ws[i] = 0.0                                  # curriculum lock until the advance gate
        # scripted-share cap: scripts collectively <= SCRIPT_SHARE_CAP while any neural carries weight
        scr = sum(w for w, m in zip(ws, members) if m["net"] is None)
        neu = sum(w for w, m in zip(ws, members) if m["net"] is not None)
        if neu > 0.0 and scr > cfg.SCRIPT_SHARE_CAP * (scr + neu):
            f = (cfg.SCRIPT_SHARE_CAP / (1.0 - cfg.SCRIPT_SHARE_CAP)) * (neu / scr)
            ws = [w * f if m["net"] is None else w for w, m in zip(ws, members)]
        # watchdog / mastery gates: dormant watchdogs out (calibration-only); a 2p-mastered
        # script leaves 2p matchmaking (still eligible in 4p until removed by eviction)
        for i, m in enumerate(members):
            if m["net"] is not None:
                continue
            if m["kind"] in cfg.WATCHDOG_KINDS and not m.get("wd_active", False):
                ws[i] = 0.0
            elif fmt == 2 and self._mastered(m, 2, cfg.SCRIPT_BOOST_DROP_WR):
                ws[i] = 0.0
        return ws

    def sample_seats(self, net=None, k=3, fmt=4):
        """k seats for an FFA: weighted draws WITHOUT replacement; at most MAX_SCRIPT_SEATS_4P
        scripted seats (the rest go to neural members), and a neural member fills any seat the
        pool is too thin to fill distinctly."""
        ws = self._weights(fmt, net)
        members = self.members
        neural = [m for m in members if m["net"] is not None]
        out = []; n_script = 0
        for _ in range(k):
            if n_script >= self.cfg.MAX_SCRIPT_SEATS_4P and any(w > 0.0 for w, m in zip(ws, members) if m["net"] is not None):
                ws = [0.0 if m["net"] is None else w for w, m in zip(ws, members)]
            tot = sum(ws)
            if tot <= 0.0:                                   # not enough distinct players -> a network takes the seat
                out.append(self.rng.choice(neural) if neural else self.rng.choice(members))
                continue
            r = self.rng.random() * tot; c = 0.0; pick = len(members) - 1
            for i, w in enumerate(ws):
                c += w
                if r <= c:
                    pick = i; break
            m = members[pick]; out.append(m)
            if m["net"] is None:
                n_script += 1
            if sum(1 for w in ws if w > 0.0) > 1:
                ws[pick] = 0.0                                # without replacement while distinct players remain
        return out

    def share_4p(self):
        """Share of 4p iterations from an EMA of the (elo2-elo4) gap, clipped to [S4_MIN, S4_MAX]."""
        cfg = self.cfg
        gap = (self.learner_elo - self.learner_elo4) / 400.0
        prev = getattr(self, "_gap_ema", None)
        self._gap_ema = gap if prev is None else (1.0 - cfg.S4_EMA_BETA) * prev + cfg.S4_EMA_BETA * gap
        return min(cfg.S4_MAX, max(cfg.S4_MIN, cfg.S4_BASE + cfg.S4_GAIN * self._gap_ema))

    # ---------------- rolling self-anchor recal (the learner is its own moving reference) ----------------
    def _self_anchors(self):
        return [m for m in self.members if m.get("kind") == SELF_KIND]

    def _top2(self):
        a = self._self_anchors()
        return max(a, key=lambda m: m["elo"]) if a else None

    def _top4(self):
        a = self._self_anchors()
        return max(a, key=lambda m: m["elo4"]) if a else None

    def _freeze_anchor(self, net, R2, R4):
        snap = _freeze_snapshot(copy.deepcopy(_unwrap(net)))
        k = 1 + max([int(x["label"][5:]) for x in self._self_anchors()
                     if x["label"].startswith("selfA") and x["label"][5:].isdigit()] + [-1])
        m = {"label": "selfA%d" % k, "net": snap, "kind": SELF_KIND,
             "elo": float(R2), "elo4": float(R4), "anchor": True, "pinned": True, "n": 0, "n4": 0}
        self.members.append(m)
        top4 = self._top4()                          # never trim the current 4p reference
        while len(self._self_anchors()) > ANCHOR_KEEP:
            victim = next((x for x in sorted(self._self_anchors(), key=lambda z: z["elo"]) if x is not top4), None)
            if victim is None:
                break
            self.members.remove(victim)
        return m

    def _ground_ratings(self, net, worlds, n, ground_members, first):
        """Ground the learner (and, on a full recal, every snapshot) on the top self-anchor.
        Blends into the live ratings (RECAL_BLEND) unless `first` (raw). NO promotion / eviction
        here -- those live in _reground so a promotion re-ground can reuse this safely.
        Returns (s2, wr4, pw4, top2, top4, g2)."""
        cfg = self.cfg
        def blend(old, meas):
            return meas if first else (1.0 - cfg.RECAL_BLEND) * old + cfg.RECAL_BLEND * meas
        # lazy chain init: freeze the current learner as the origin anchor selfA0
        if not self._self_anchors():
            R2 = self.learner_elo
            if (not math.isfinite(R2)) or (R2 < cfg.ELO_GREEDY):    # uncalibrated -> tie origin to the script floor
                gm = [self.anchor(k) for k in ("greedy", "medium")]
                R2 = sum(_recal_link(cfg, a["elo"], _eval_score(cfg, net, a["kind"], worlds, n)) for a in gm) / len(gm)
            R4 = self.learner_elo4 if (math.isfinite(self.learner_elo4) and self.learner_elo4 >= cfg.ELO_GREEDY) else R2
            m0 = self._freeze_anchor(net, R2, R4)
            print("    [chain init] froze learner as %s (R2 %.0f / R4 %.0f)" % (m0["label"], R2, R4))
        top2 = self._top2(); top2["net"].to(cfg.device)
        s2 = _score2_vs(cfg, net, top2["net"], worlds, n)
        self.learner_elo = blend(self.learner_elo, _recal_link(cfg, top2["elo"], s2))
        g2 = _eval_score(cfg, net, "greedy", worlds, n)             # collapse watchdog (NOT in the estimate)
        wr4, pw4, top4 = 1.0, 1.0, None
        if cfg.FOURP_ENABLED:
            top4 = self._top4(); top4["net"].to(cfg.device)
            wr4, pw4 = _eval4_vs(cfg, net, top4["net"], worlds, n)
            self.learner_elo4 = blend(self.learner_elo4, _recal_link(cfg, top4["elo4"], pw4))
        if ground_members:                                          # de-compress the snapshot ladder (full recals)
            for mm in self.members:
                if mm["anchor"] or mm["net"] is None or mm.get("kind") == SELF_KIND:
                    continue
                mm["net"].to(cfg.device)
                mm["elo"] = blend(mm["elo"], _recal_link(cfg, top2["elo"], _score2_vs(cfg, mm["net"], top2["net"], worlds, n)))
                if cfg.FOURP_ENABLED:
                    _, mpw = _eval4_vs(cfg, mm["net"], top4["net"], worlds, n)
                    mm["elo4"] = blend(mm["elo4"], _recal_link(cfg, top4["elo4"], mpw))
                mm["net"].to("cpu"); mm["n"] = 0; mm["n4"] = 0
        for a in self._self_anchors():
            if a["net"] is not None:
                a["net"].to("cpu")
        return s2, wr4, pw4, top2, top4, g2

    def _reground(self, net, worlds, n=None, ground_members=False):
        """One recalibration: ground ratings (blended; first=raw), maybe promote (DUAL-format
        dominance), then VALIDATE watchdogs and run the MEASURED eviction. Scripted-anchor eviction
        fires at the very first recal (start/resume) and at every recal thereafter; the neural
        league-player pass runs only on the first/full recal (``first or ground_members``)."""
        cfg = self.cfg
        n = n or cfg.ELO_RECAL_ENVS
        first = not getattr(self, "_grounded", False)
        self._grounded = True
        s2, wr4, pw4, top2, top4, g2 = self._ground_ratings(net, worlds, n, ground_members, first)
        promoted = False
        if (s2 >= PROMOTE_S2) and (wr4 >= PROMOTE_WR4):            # DUAL-format dominance gate
            R2 = _recal_link(cfg, top2["elo"], s2)
            R4 = _recal_link(cfg, top4["elo4"], pw4) if cfg.FOURP_ENABLED else R2
            mp = self._freeze_anchor(net, R2, R4)
            promoted = True
            d4 = (R4 - top4["elo4"]) if cfg.FOURP_ENABLED else 0.0
            print("    [PROMOTE] %s: s2 %.2f wr4 %.2f pw4 %.2f -> R2 %.0f (+%.0f) R4 %.0f (+%.0f)"
                  % (mp["label"], s2, wr4, pw4, R2, R2 - top2["elo"], R4, d4))
        wd = "" if g2 >= GREEDY_WATCHDOG else "  [WATCHDOG] vs greedy %.2f < %.2f -- COLLAPSE?" % (g2, GREEDY_WATCHDOG)
        print("    [recal] elo2 %.0f (s2 %.2f vs %s) elo4 %.0f (pw4 %.2f) | greedy %.2f | anchors %d%s%s"
              % (self.learner_elo, s2, top2["label"], self.learner_elo4, pw4, g2,
                 len(self._self_anchors()), " [first=raw]" if first else "", wd))
        if promoted:                                               # fresh worlds + full league recal per anchor
            self._promo_ctr = getattr(self, "_promo_ctr", 0) + 1
            fresh = make_world_pool(cfg, n, base_seed=cfg.SEED + 80_000_000 + 7919 * self._promo_ctr)
            self._ground_ratings(net, fresh, n, True, first=False)
            self._force_resample = True                            # ask the training loop for fresh training worlds
            print("    [promotion] new anchor #%d -> fresh worlds + full league recal" % self._promo_ctr)
        self.validate_watchdogs(net, worlds, n)                    # rating check -> activate failing watchdogs
        # measured eviction: scripted anchors every recal; the heavier neural-player pass only on the
        # full/first recal (where the league is already being fully re-grounded).
        self.evict_mastered(net, worlds, n, neural=(first or ground_members))
        return promoted

    def recalibrate_elo(self, net, worlds, n_envs=None, all_anchors=False):
        return self._reground(net, worlds, n_envs, ground_members=True)

    def recalibrate_elo4(self, net, worlds, n_envs=None):
        return None                                               # 4p is grounded inside _reground

    def recalibrate_learner(self, net, worlds, n_envs=None):
        return self._reground(net, worlds, n_envs, ground_members=False)

    # ---------------- watchdog calibration validators + measured eviction ----------------
    def validate_watchdogs(self, net, worlds, n=None):
        """Each DORMANT watchdog (random/greedy) scores the learner vs its own Elo prediction;
        if the learner underperforms (actual < expected - WATCHDOG_TOL) the rating is inflated /
        the policy regressed, so the watchdog ENTERS the league as a live training opponent."""
        cfg = self.cfg
        n = n or cfg.ELO_RECAL_ENVS
        for m in self.members:
            if m["net"] is not None or m["kind"] not in cfg.WATCHDOG_KINDS or m.get("wd_active", False):
                continue
            actual = _eval_score(cfg, net, m["kind"], worlds, n)
            expected = self._p_beat(m, 2)
            if actual < expected - cfg.WATCHDOG_TOL:
                m["wd_active"] = True
                print("    [watchdog] %-8s FAILED validation: 2p %.2f < expected %.2f - %.2f -> ENTERS league"
                      % (m["kind"], actual, expected, cfg.WATCHDOG_TOL))

    def _newest_snapshot(self):
        """The most recently frozen learner snapshot (the current self) -- never evicted, so the
        league always keeps the freshest self-play opponent to train against."""
        autos = [m for m in self.members if m["kind"] == "snapshot" and not m["pinned"]]
        return autos[-1] if autos else None

    def _measure_mastery(self, net, m, worlds, n):
        """(s2, w4) of the learner vs member `m` on ONE standard for scripts and league players:
        s2 = seat-avg 2p score, w4 = 4p 1st-place rate vs THREE copies (0.0 when 4p is disabled)."""
        cfg = self.cfg
        if m["net"] is None:                                        # scripted anchor (on-GPU)
            s2 = _eval_score(cfg, net, m["kind"], worlds, n)
            w4 = _eval_winrate4(cfg, net, m["kind"], worlds, n) if cfg.FOURP_ENABLED else 0.0
        else:                                                       # neural league player (snapshot/invited)
            m["net"].to(cfg.device)
            s2 = _score2_vs(cfg, net, m["net"], worlds, n)
            w4 = _eval_winrate4_net(cfg, net, m["net"], worlds, n) if cfg.FOURP_ENABLED else 0.0
            m["net"].to("cpu")
        return s2, w4

    def evict_mastered(self, net, worlds, n=None, neural=True):
        """MEASURED eviction (start + every recal): remove every league member the learner has
        MASTERED under ONE standard -- 2p seat-avg score s2 and 4p 1st-place rate w4 vs three copies,
        mastered when ``s2 > MASTER_EVICT_2P_WR`` OR ``w4 > MASTER_EVICT_4P_WR`` (per-format; the 4p
        clause is skipped when 4p is disabled). Covers scripted anchors AND neural league players
        (snapshots + invited) -- the latter only when ``neural`` (gated by the caller to the heavier
        full-recal/start path). NEVER touched: the rolling self-anchors (the Elo reference) and the
        newest snapshot (the current self); keep-kinds random/greedy are never removed -- a re-mastered
        ACTIVE watchdog retires to DORMANT instead."""
        cfg = self.cfg
        n = n or cfg.ELO_RECAL_ENVS
        keep_newest = self._newest_snapshot()
        for m in list(self.members):
            is_script = (m["net"] is None and m["kind"] in _OPP_BY_KIND)
            is_player = (m["net"] is not None and m["kind"] in ("snapshot", "invited"))
            if not (is_script or (neural and is_player)) or m is keep_newest:
                continue                                            # self-anchors / freshest self: never
            s2, w4 = self._measure_mastery(net, m, worlds, n)
            mastered = (s2 > cfg.MASTER_EVICT_2P_WR) or (cfg.FOURP_ENABLED and w4 > cfg.MASTER_EVICT_4P_WR)
            m["mastered"] = bool(mastered)
            if not mastered:
                continue
            if m["kind"] in cfg.MASTER_KEEP_KINDS:                  # random/greedy: retire watchdog, never remove
                if m["kind"] in cfg.WATCHDOG_KINDS and m.get("wd_active", False):
                    m["wd_active"] = False                          # re-mastered -> calibration-only
                    print("    [watchdog] %-8s re-mastered (2p=%.2f 4p_win=%.2f) -> back to DORMANT"
                          % (m["kind"], s2, w4))
            else:
                self.members.remove(m)
                who = "script" if is_script else m["kind"]
                print("    [evict] %-14s (%-7s) mastered (2p=%.2f 4p_win=%.2f) -> removed from league"
                      % (m["label"], who, s2, w4))

    def _pfsp_weight(self, p):
        cfg = self.cfg
        if cfg.PFSP_MODE == "hard":
            return (1.0 - p) ** cfg.PFSP_POWER + cfg.PFSP_FLOOR   # focus on opponents you can't yet beat
        if cfg.PFSP_MODE == "even":
            return p * (1.0 - p) + cfg.PFSP_FLOOR             # focus on evenly-matched opponents
        return 1.0                                        # uniform

    def _mastered(self, m, fmt, thr):
        n = m["n"] if fmt == 2 else m.get("n4", 0)
        wr = m.get("wr") if fmt == 2 else m.get("wr4")
        return (n >= self.cfg.SCRIPT_BOOST_MIN_GAMES) and ((wr if wr is not None else 0.0) >= thr)

    def sample(self, net=None, fmt=2):
        """PFSP-sample ONE opponent using the per-format ratings + boosts."""
        ws = self._weights(fmt, net)
        tot = sum(ws)
        if tot <= 0.0:
            return self.rng.choice(self.members)
        r = self.rng.random() * tot
        c = 0.0
        for m, w in zip(self.members, ws):
            c += w
            if r <= c:
                return m
        return self.members[-1]

    def update_elo(self, m, learner_score):
        """2p result: learner_score in [0,1] = win + 0.5*draw over the rollout. Anchors stay
        fixed. The 2p delta also drags BOTH 4p ratings by ELO_COUPLING (correlated ratings)."""
        cfg = self.cfg
        exp = self._p_beat(m, 2)
        delta = cfg.ELO_K * (learner_score - exp)
        self.learner_elo += delta
        self.learner_elo4 += cfg.ELO_COUPLING * delta
        m["n"] += 1
        m["wr"] = learner_score if m.get("wr") is None else (1.0 - cfg.WR_EMA_BETA) * m["wr"] + cfg.WR_EMA_BETA * learner_score
        if not m["anchor"]:
            m["elo"] -= delta
            m["elo4"] -= cfg.ELO_COUPLING * delta

    def update_elo_4p(self, seats, seat_scores):
        """4p FFA result as len(seats) pairwise updates at ELO_K4/len each (standard multiplayer
        decomposition). Coupled back into the 2p ratings by ELO_COUPLING."""
        cfg = self.cfg
        kf = cfg.ELO_K4 / max(1, len(seats))
        for m, sc in zip(seats, seat_scores):
            exp = self._p_beat(m, 4)
            delta = kf * (sc - exp)
            self.learner_elo4 += delta
            self.learner_elo += cfg.ELO_COUPLING * delta
            m["n4"] = m.get("n4", 0) + 1
            m["wr4"] = sc if m.get("wr4") is None else (1.0 - cfg.WR_EMA_BETA) * m["wr4"] + cfg.WR_EMA_BETA * sc
            if not m["anchor"]:
                m["elo4"] -= delta
                m["elo"] -= cfg.ELO_COUPLING * delta

    def leaderboard(self):
        cfg = self.cfg
        rows = sorted(self.members, key=lambda m: -m["elo"])
        s = "  league | learner elo2 %.0f elo4 %.0f | share4p %.2f | %d members\n" % (
            self.learner_elo, self.learner_elo4, self.share_4p(), len(self.members))
        for m in rows:
            tag = "[anchor]" if m["anchor"] else ("[pinned]" if m["pinned"] else "")
            if m["kind"] == "invited":
                tag = "[invited %s%s]" % ("M" if self._mastered(m, 2, cfg.INVITE_MASTER_WR) else "-",
                                          "M" if self._mastered(m, 4, cfg.INVITE_MASTER_WR) else "-")
            elif m["net"] is None:
                boosted = not self._mastered(m, 2, cfg.SCRIPT_BOOST_DROP_WR)
                tag += " x%.1f" % cfg.SCRIPT_MATCH_BOOST if boosted else " (mastered)"
            wr = m.get("wr"); wr4 = m.get("wr4")
            s += "    %-22s elo2 %6.0f (n=%-3d wr %s)  elo4 %6.0f (n4=%-3d wr %s) %s\n" % (
                m["label"], m["elo"], m["n"], ("%.2f" % wr) if wr is not None else " -- ",
                m["elo4"], m.get("n4", 0), ("%.2f" % wr4) if wr4 is not None else " -- ", tag)
        return s

    def _probe_obs(self):
        """Fingerprint probe = REAL encoded game states (on-manifold), cached once."""
        if getattr(self, '_probe', None) is None:
            env = GpuEnv(self.cfg)
            env.reset(make_world_pool(self.cfg, 16, base_seed=0xC0FFEE % 100000))
            with torch.no_grad():
                for _t in range(20):
                    e, m, a, g = env_encode(env, 0)
                    rnd = torch.softmax(torch.randn(env.B, PLANET_CAP, PLANET_CAP + 1, device=self.cfg.device), -1)
                    env_step(self.cfg, env, rnd, 1, None, step_idx=_t)
            e, m, a, g = env_encode(env, 0)
            self._probe = (e.cpu(), m.cpu(), a.cpu(), g.cpu())
        return self._probe

    def _fingerprint(self, net):
        """Behavioral fingerprint = mean (WHERE allocation, gate fire/hold) on the fixed probe batch."""
        ent, em, am, gl = self._probe_obs()
        dev = next(net.parameters()).device
        with torch.no_grad():
            dest_logits, gate_logits, _ = net(ent.to(dev), em.to(dev), am.to(dev), gl.to(dev))
            dest = torch.softmax(dest_logits, -1).reshape(-1, dest_logits.shape[-1]).mean(0)  # (E,)
            gp = torch.sigmoid(gate_logits).reshape(-1).float().mean()                         # mean fire-prob
            gate = torch.stack([gp, 1.0 - gp])                                                 # (2,) fire/hold
        return dest.cpu(), gate.cpu()

    def calibrate_anchor_elo4(self, net, worlds, n_envs=None):
        """Measure the scripted anchors' TRUE 4p pairwise strength (2 lineups x 3 seat
        rotations, ego = current learner) and re-pin every scripted elo4 on one scale with
        greedy fixed at ELO_GREEDY. recalibrate_elo4/recalibrate_learner then ground the 4p
        scale on the measured GAUNTLET4_SEATS base instead of the 2p-copied constants."""
        cfg = self.cfg
        n = n_envs or cfg.ELO_RECAL_ENVS
        acc, cnt = {}, {}
        def run(trio):
            for r in range(3):
                tr = list(trio[r:]) + list(trio[:r])
                ps = _eval_pair_scores4(cfg, net, tr, worlds, n)
                for (a, b), s in ps.items():
                    ka, kb = tr[a - 1], tr[b - 1]
                    acc[(ka, kb)] = acc.get((ka, kb), 0.0) + s;         cnt[(ka, kb)] = cnt.get((ka, kb), 0) + 1
                    acc[(kb, ka)] = acc.get((kb, ka), 0.0) + (1.0 - s); cnt[(kb, ka)] = cnt.get((kb, ka), 0) + 1
        run(list(GAUNTLET4_SEATS))                 # starter / greedy / intermediate
        run(["medium", "greedy", "random"])        # greedy shared -> one common scale
        def rel(k):
            ss = [acc[(k, o)] / cnt[(k, o)] for o in _CAL4_KINDS if (k, o) in acc]
            s = min(max(sum(ss) / len(ss), 0.05), 0.95)
            return cfg.ELO_SCALE * math.log10(s / (1.0 - s))
        r = {k: rel(k) for k in _CAL4_KINDS if any((k, o) in acc for o in _CAL4_KINDS)}
        out = {k: cfg.ELO_GREEDY + r[k] - r["greedy"] for k in r}
        for k, e in out.items():
            try:
                self.anchor(k)["elo4"] = float(e)
            except StopIteration:
                pass                               # anchor already evicted (resume) -- skip
        self._anc4_base = sum(out[k] for k in GAUNTLET4_SEATS) / float(len(GAUNTLET4_SEATS))
        print("    [anc4] measured 4p anchors: " + " ".join("%s=%.0f" % (k, out[k]) for k in sorted(out))
              + " | trio base %.0f" % self._anc4_base)

    def calibrate_invited(self, worlds, n_envs=None):
        """Measure every invited member's elo2 (vs the ground anchors) and elo4 (vs the 4p trio,
        on the measured base). Filename/meta stamps carry old run-internal scales -- anchoring to
        them poisons the expected-score model AND matchmaking. Frozen agents: measuring once per
        train() start is cheap and exact."""
        cfg = self.cfg
        n = n_envs or cfg.ELO_RECAL_ENVS
        inv = [m for m in self.members if m["kind"] == "invited"]
        if not inv:
            return
        ground = [m for m in self.members if m["anchor"] and m["net"] is None
                  and m["kind"] in cfg.ELO_GROUND_KINDS]
        if not ground:
            ground = [m for m in self.members if m["anchor"] and m["net"] is None]
        base4 = getattr(self, "_anc4_base", None)
        if base4 is None:
            base4 = (cfg.ELO_STARTER + cfg.ELO_GREEDY + cfg.ELO_INTERMEDIATE) / 3.0
        def _est(base, sc):
            sc = min(max(sc, 0.02), 0.98)
            return base + cfg.ELO_SCALE * math.log10(sc / (1.0 - sc))
        for m in inv:
            old2, old4 = m["elo"], m["elo4"]
            m["net"].to(cfg.device)
            m["elo"] = sum(_est(a["elo"], _eval_score(cfg, m["net"], a["kind"], worlds, n))
                           for a in ground) / len(ground)
            m["elo4"] = (_est(base4, _eval_score4(cfg, m["net"], GAUNTLET4_SEATS, worlds, n))
                         if cfg.FOURP_ENABLED else m["elo"])
            m["net"].to("cpu")
            print("    [inv cal] %-26s elo2 %6.0f -> %6.0f | elo4 %6.0f -> %6.0f"
                  % (m["label"], old2, m["elo"], old4, m["elo4"]))

    def state_dict(self):
        members = []
        for m in self.members:
            members.append({"label": m["label"], "kind": m["kind"], "elo": m["elo"],
                            "elo4": m.get("elo4", m["elo"]), "anchor": m["anchor"], "pinned": m["pinned"],
                            "n": m["n"], "n4": m.get("n4", 0), "wr": m.get("wr"), "wr4": m.get("wr4"),
                            "cfg": m.get("cfg"),
                            "model": (_unwrap(m["net"]).state_dict() if m["net"] is not None else None)})
        return {"learner_elo": self.learner_elo, "learner_elo4": self.learner_elo4,
                "advanced": self.advanced, "members": members}

    def load_state_dict(self, sd):
        self.learner_elo = sd["learner_elo"]
        self.learner_elo4 = sd.get("learner_elo4", sd["learner_elo"])
        self.advanced = sd.get("advanced", self.advanced)
        self.members = []
        for e in sd["members"]:
            net = None
            if e["model"] is not None:
                net = build_from_cfg(self.cfg, e["cfg"]) if e.get("cfg") else build_policy(self.cfg)
                net.load_state_dict(e["model"])
                net = _freeze_snapshot(_wrap_if_legacy(net, e.get("cfg")))
            m = {"label": e["label"], "kind": e["kind"], "net": net, "elo": e["elo"],
                 "elo4": e.get("elo4", e["elo"]), "anchor": e["anchor"], "pinned": e["pinned"],
                 "n": e["n"], "n4": e.get("n4", 0)}
            if e.get("cfg") is not None:
                m["cfg"] = e["cfg"]
            if e.get("wr") is not None:
                m["wr"] = e["wr"]
            if e.get("wr4") is not None:
                m["wr4"] = e["wr4"]
            self.members.append(m)
