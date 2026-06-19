"""AlphaZero-style MCTS for Orbit Wars -- proof of concept (NOT wired into training).

The v12 stack already gives us everything AlphaZero needs except the search itself: a
bit-exact, fast forward model (:class:`~orbit_wars_v12.env.GpuEnv`) and a policy net that
emits a prior + a value head (:mod:`orbit_wars_v12.policy`). This module adds the missing
piece -- a tree search that uses the net as a **prior over sampled candidate actions** and a
bounded **leaf value** -- so a (small) net can spend inference compute to play better, and so
search visit distributions can later serve as policy-improvement targets (BC warm-start ->
self-play distillation; that loop is the *next* step, not this file).

Scope / simplifications for the PoC:

* **Opponent = a fixed deterministic script** (``opponent_action`` codes). With the opponent
  fixed, the game is a *deterministic single-agent MDP* from the ego's view, so plain PUCT is
  sound -- we deliberately dodge the simultaneous-move subtlety here. The real fix for
  self-play (decoupled-UCT / regret-matching at each node, 2p zero-sum first) comes once we
  have a BC-warm-started net to self-play.
* **Single env (B=1).** A node stores a full state snapshot; expansion restores it into one
  shared work env and steps once. Batched-across-games search is a later optimization.
* **Leaf value = bounded ship/production margin heuristic** by default (``value="net"`` uses
  the value head). SUBMISSION.md found the PPO critic is policy-biased and lost to exactly this
  heuristic for leaf eval; the AlphaZero loop is what later de-biases the head.

Determinism (required for correct snapshot/restore + reproducible search) holds when comets are
official (``COMET_OFFICIAL=True``, waypoint playback -- no ``torch.rand``) and the scripted
opponent is deterministic (codes 1/3/4/5; avoid 0=random). Candidate ego actions are sampled
once and stored on the node, so the transitions themselves never re-sample.
"""
from __future__ import annotations

import math
import time

import torch

from .env import GpuEnv, env_encode, env_step, settle, _OPP_BY_KIND
from .policy import act

# Mutable per-env state cloned by a snapshot. The static official-comet arrays
# (c_paths / c_len / c_ships) are read-only during stepping, so they are shared by
# reference (bound once on the work env) rather than cloned; only c_slot mutates.
_SNAP_ATTRS = (
    "p_alive", "p_owner", "p_x", "p_y", "p_radius", "p_ships", "p_prod", "p_is_comet",
    "p_init_x", "p_init_y", "p_rotates", "p_comet_vx", "p_comet_vy",
    "f_alive", "f_owner", "f_x", "f_y", "f_angle", "f_ships", "f_seq",
    "ang_vel", "step_ct", "done", "c_slot",
)

PROD_WEIGHT = 20.0       # heuristic: 1 production ~ PROD_WEIGHT ships of long-run value
VALUE_SCALE = 200.0      # logistic scale for the strength-margin leaf value


def snapshot_env(env) -> dict:
    """Clone the mutable state of a GpuEnv into a plain dict (B and n_players included)."""
    snap = {a: (getattr(env, a).clone() if torch.is_tensor(getattr(env, a, None)) else None)
            for a in _SNAP_ATTRS}
    snap["_B"] = env.B
    snap["_n_players"] = int(getattr(env, "n_players", 2))
    return snap


def restore_env(env, snap: dict) -> None:
    """Overwrite ``env``'s mutable state from a snapshot. Clones on the way in so later in-place
    env_step writes (e.g. ``c_slot[...] =``) never corrupt the stored snapshot."""
    for a in _SNAP_ATTRS:
        v = snap.get(a)
        if torch.is_tensor(v):
            setattr(env, a, v.clone())
    env.B = snap["_B"]
    env.n_players = snap["_n_players"]


def heuristic_value(env, ego: int = 0) -> float:
    """Bounded position value in [0,1] for ``ego`` (B=1): logistic of the ship + fleet +
    PROD_WEIGHT*production margin against all opponents pooled. Matches settle_n's accounting."""
    alive = env.p_alive > 0.5
    owner = env.p_owner
    fa = env.f_alive > 0.5
    mine = (owner == float(ego)) & alive
    opp = (owner >= 0.0) & (owner != float(ego)) & alive
    fo = env.f_owner
    my = ((env.p_ships * mine).sum() + (env.f_ships * ((fo == float(ego)) & fa)).sum()
          + PROD_WEIGHT * (env.p_prod * mine).sum())
    op = ((env.p_ships * opp).sum() + (env.f_ships * ((fo != float(ego)) & fa)).sum()
          + PROD_WEIGHT * (env.p_prod * opp).sum())
    return float(torch.sigmoid((my - op) / VALUE_SCALE).item())


def criticality(env, ego: int = 0) -> float:
    """How pivotal this turn is, in [0,1] (B=1). High when the game is materially CLOSE -- a
    swing turn worth spending the overage bank on. Cheap (one strength-margin closeness)."""
    alive = env.p_alive > 0.5
    owner = env.p_owner
    fa = env.f_alive > 0.5
    mine = (owner == float(ego)) & alive
    opp = (owner >= 0.0) & (owner != float(ego)) & alive
    fo = env.f_owner
    my = float(((env.p_ships * mine).sum() + (env.f_ships * ((fo == float(ego)) & fa)).sum()).item())
    op = float(((env.p_ships * opp).sum() + (env.f_ships * ((fo != float(ego)) & fa)).sum()).item())
    tot = my + op
    if tot <= 0.0:
        return 1.0
    return max(0.0, 1.0 - abs(my - op) / tot)            # 1 = dead even -> most critical


class _Node:
    """One tree node: a state snapshot + K candidate edges (prior P, visits N, value sum W)."""
    __slots__ = ("snap", "cands", "P", "N", "W", "children", "terminal", "tv")

    def __init__(self, snap, cands, P, terminal=False, tv=0.0):
        K = len(cands)
        self.snap = snap
        self.cands = cands          # list of K action bundles (1, E, E+1)
        self.P = P                  # (K,) prior over candidates
        self.N = torch.zeros(K)
        self.W = torch.zeros(K)
        self.children = [None] * K
        self.terminal = terminal
        self.tv = tv                # terminal value (ego perspective) when terminal


def _candidates(cfg, net, env, ego, K):
    """K candidate full-turn actions for ``ego`` = the policy-greedy move (index 0, a safe
    fallback) + K-1 samples, with prior P = softmax of their policy log-probs."""
    ent, em, am, gl = env_encode(env, ego)
    cands, logps = [], []
    a, lp, _ = act(cfg, net, ent, em, am, gl, greedy=True)
    cands.append(a); logps.append(lp.reshape(()))
    for _ in range(max(0, K - 1)):
        a, lp, _ = act(cfg, net, ent, em, am, gl, greedy=False)
        cands.append(a); logps.append(lp.reshape(()))
    # Tree bookkeeping (P/N/W/Q/U) lives on CPU: it is tiny (K,) and per-sim GPU kernels +
    # .item() syncs would dominate. Candidate ACTION tensors stay on the net's device for stepping.
    P = torch.softmax(torch.stack(logps), 0).cpu()
    return cands, P


def _leaf_value(cfg, net, env, value):
    if value == "net":
        ent, em, am, gl = env_encode(env, 0)
        _, _, v = act(cfg, net, ent, em, am, gl, greedy=True)
        return float(torch.sigmoid(v.reshape(()) / VALUE_SCALE).item())
    return heuristic_value(env, 0)


def _terminal_value(s0, s1, al0, al1):
    if al0 and not al1:
        return 1.0
    if al1 and not al0:
        return 0.0
    if (not al0) and (not al1):
        return 0.5
    return 1.0 if float(s0) > float(s1) else 0.0 if float(s0) < float(s1) else 0.5


def _select(node, c_puct):
    """PUCT: argmax Q + c*P*sqrt(sum N)/(1+N); unvisited edges get a neutral FPU of 0.5."""
    N = node.N
    ntot = float(N.sum().item())
    Q = torch.where(N > 0, node.W / N.clamp_min(1.0), torch.full_like(N, 0.5))
    U = c_puct * node.P * math.sqrt(ntot + 1e-8) / (1.0 + N)
    return int(torch.argmax(Q + U).item())


def _expand_child(cfg, net, work, node, a, opp_code, K, value):
    """Apply edge ``a`` (ego candidate + scripted opponent) from ``node``'s state; return
    (backup_value, child_node). The child is terminal (game decided / horizon) or a fresh
    leaf with its own snapshot + candidates."""
    restore_env(work, node.snap)
    step_idx = int(work.step_ct[0].item())
    env_step(cfg, work, node.cands[a], opp_code, None, step_idx=step_idx)
    s0, s1, al0, al1 = settle(work)
    al0, al1 = bool(al0[0].item()), bool(al1[0].item())
    horizon = float(work.step_ct[0].item()) >= float(work.T)
    if (not al0) or (not al1) or horizon:
        v = _terminal_value(s0[0].item(), s1[0].item(), al0, al1)
        return v, _Node(None, [], None, terminal=True, tv=v)
    v = _leaf_value(cfg, net, work, value)
    child = _Node(snapshot_env(work), *_candidates(cfg, net, work, 0, K))
    return v, child


@torch.no_grad()
def mcts_search(cfg, net, work, root_snap, opp_code, n_sims=32, c_puct=1.5, K=6,
                value="heuristic", time_budget=None):
    """Run up to ``n_sims`` PUCT simulations from ``root_snap`` (ego = seat 0, opponent = scripted
    ``opp_code``) using ``work`` as the single restore-and-step env. Stops early if ``time_budget``
    (seconds) is exceeded -- the deploy wall-clock guard; the most-visited edge so far is always a
    safe answer (greedy is edge 0). Returns (best_action, root_node, stats)."""
    t0 = time.perf_counter()
    restore_env(work, root_snap)
    root = _Node(root_snap, *_candidates(cfg, net, work, 0, K))
    ran = 0
    for _ in range(n_sims):
        ran += 1
        path, node = [], root
        while True:
            a = _select(node, c_puct)
            path.append((node, a))
            ch = node.children[a]
            if ch is None:
                v, child = _expand_child(cfg, net, work, node, a, opp_code, K, value)
                node.children[a] = child
                break
            if ch.terminal:
                v = ch.tv
                break
            node = ch
        for n, a in path:
            n.N[a] += 1.0
            n.W[a] += v
        if time_budget is not None and (time.perf_counter() - t0) > time_budget:
            break
    best = int(torch.argmax(root.N).item())
    visits = root.N
    nz = visits[visits > 0]
    probs = nz / nz.sum() if nz.numel() else nz
    stats = {
        "visits": visits.tolist(),
        "sims_run": ran,
        "best_idx": best,
        "switched": best != 0,                                   # search preferred non-greedy
        "root_q": float((root.W.sum() / root.N.sum().clamp_min(1.0)).item()),
        "visit_entropy": float(-(probs * probs.clamp_min(1e-9).log()).sum().item()) if nz.numel() else 0.0,
        "elapsed": time.perf_counter() - t0,
    }
    return root.cands[best], root, stats


class MctsAgent:
    """Seat-0 agent that plays the most-visited MCTS move each turn, spending the 60 s overage
    bank SMARTLY: ordinary turns get ``n_sims`` under a ``turn_budget_s`` wall-clock guard;
    CRITICAL turns (close game, :func:`criticality` > ``crit_thresh``) scale up toward
    ``crit_sims`` under the larger ``crit_budget_s``, drawing the extra time from ``bank_s`` until
    it runs out. Reuses one work env; binds the static official-comet arrays on first use."""

    def __init__(self, cfg, net, opp_kind="greedy", n_sims=32, c_puct=1.5, K=6, value="heuristic",
                 crit_sims=None, crit_thresh=0.6, turn_budget_s=0.9, crit_budget_s=4.0,
                 bank_s=60.0):
        self.cfg = cfg
        self.net = net.eval()
        self.opp_code = _OPP_BY_KIND[opp_kind]
        self.n_sims, self.c_puct, self.K, self.value = n_sims, c_puct, K, value
        self.crit_sims = crit_sims if crit_sims is not None else 4 * n_sims
        self.crit_thresh = crit_thresh
        self.turn_budget_s, self.crit_budget_s = turn_budget_s, crit_budget_s
        self.bank_s = bank_s
        self.work = GpuEnv(cfg)
        self.work.c_paths = None
        self.last_stats = None

    def act(self, game_env, step_idx=None):
        if getattr(self.work, "c_paths", None) is None and getattr(game_env, "c_paths", None) is not None:
            self.work.c_paths, self.work.c_len, self.work.c_ships = (
                game_env.c_paths, game_env.c_len, game_env.c_ships)
        crit = criticality(game_env, 0)
        # critical AND bank can fund the overspend -> scale sims + widen the wall-clock guard.
        use_bank = crit > self.crit_thresh and self.bank_s > (self.crit_budget_s - self.turn_budget_s)
        sims = int(round(self.n_sims + crit * (self.crit_sims - self.n_sims))) if use_bank else self.n_sims
        budget = self.crit_budget_s if use_bank else self.turn_budget_s
        snap = snapshot_env(game_env)
        a, _root, stats = mcts_search(self.cfg, self.net, self.work, snap, self.opp_code,
                                      n_sims=sims, c_puct=self.c_puct, K=self.K, value=self.value,
                                      time_budget=budget)
        self.bank_s = max(0.0, self.bank_s - max(0.0, stats["elapsed"] - self.turn_budget_s))
        stats.update(crit=crit, used_bank=use_bank, bank_s=self.bank_s)
        self.last_stats = stats
        return a, stats


@torch.no_grad()
def play_2p_vs_script(cfg, act_fn, world, opp_kind="greedy", max_steps=None):
    """Play seat 0 (driven by ``act_fn(game_env, t) -> action``) vs a scripted seat-1 bot on one
    world. Returns (score, (s0, s1), n_steps); score = 1 win / 0.5 draw / 0 loss for seat 0."""
    opp_code = _OPP_BY_KIND[opp_kind]
    env = GpuEnv(cfg)
    env.reset([world], n_players=2)
    T = env.T if max_steps is None else min(max_steps, env.T)
    steps = 0
    for t in range(T):
        a0 = act_fn(env, t)
        env_step(cfg, env, a0, opp_code, None, step_idx=t)
        steps = t + 1
        _, _, al0, al1 = settle(env)
        if not (bool(al0[0].item()) and bool(al1[0].item())):
            break
    s0, s1, _, _ = settle(env)
    s0, s1 = float(s0[0].item()), float(s1[0].item())
    score = 1.0 if s0 > s1 else 0.0 if s0 < s1 else 0.5
    return score, (s0, s1), steps


def greedy_act_fn(cfg, net):
    """An ``act_fn`` for play_2p_vs_script that plays the raw policy-greedy move (no search)."""
    net = net.eval()

    @torch.no_grad()
    def _fn(game_env, t):
        ent, em, am, gl = env_encode(game_env, 0)
        a, _, _ = act(cfg, net, ent, em, am, gl, greedy=True)
        return a
    return _fn
