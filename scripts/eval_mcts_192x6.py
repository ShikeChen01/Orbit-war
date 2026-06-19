"""Fair eval of the 192x6 MCTS agent (it276) vs the three trunk MLPs, WITH search at run-time.

it276 is a small net (h192/6-blk) TRAINED TO BE DEPLOYED UNDER MCTS, so scoring it greedily would
undersell it. orbit_wars_v12.mcts.MctsAgent plays seat 0 by spending inference compute on a PUCT
tree that plans against a fixed SCRIPTED opponent model (it can't know the real opponent's policy --
the honest deployment), while the real seat-1 move is supplied here by the NEURAL opponent. So this
is "MCTS-192x6 (seat 0, search) vs <neural> (seat 1, greedy)".

Structural limits of the PoC search (orbit_wars_v12/mcts.py): it is **2p-only and seat-0-only**
(ego is hard-coded to seat 0; _expand_child steps a single scripted opponent; _terminal_value is
2p). So:
  * 2p H2H is run WITH real search, MCTS always at seat 0. The seat-0 positional edge therefore
    applies to MCTS; to isolate the SEARCH contribution we run the SAME net GREEDILY at seat 0 on
    the identical worlds as a baseline. Search uplift = mcts_score - greedy_score at fixed seat.
  * A seat-balanced all-neural 4p arena (every agent visits every seat) cannot include a
    seat-0-only/2p-only searcher; see the note this script prints + the README in the output.

    .venv/Scripts/python scripts/eval_mcts_192x6.py --calibrate     # 1 game, measure per-turn ms
    .venv/Scripts/python scripts/eval_mcts_192x6.py --n-worlds 16 --sims 32 --K 6 --turn-budget 0.5
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("MPLBACKEND", "Agg")

import torch

from orbit_wars_v12 import Config, load_snapshot, make_world_pool
from orbit_wars_v12.env import GpuEnv, env_encode, env_step, settle
from orbit_wars_v12.policy import act
from orbit_wars_v12 import mcts

MCTS_CKPT = "notebooks/weight/MLPs/192x6-MCTS-30d-100b/it276.pt"
OPPONENTS = [
    ("mlp-512x86", "notebooks/weight/MLPs/512x86/it951.pt"),
    ("mlp-512x32", "notebooks/weight/MLPs/512x32/run3/it1426.pt"),
    ("mlp-256x32", "notebooks/weight/MLPs/256x32/it351.pt"),
]


def _e(v):
    return "%.0f" % v if isinstance(v, (int, float)) else "--"


def _greedy_act(cfg, net, env, seat):
    return act(cfg, net, *env_encode(env, seat), greedy=True)[0]


def play_mcts_vs_neural(cfg, agent, opp_net, world, max_steps=None):
    """B=1 game: seat0 = MctsAgent (real search, plans vs its scripted model), seat1 = opp_net
    greedy. Returns (score_seat0, (s0,s1), n_steps, mean_turn_ms, switch_rate)."""
    env = GpuEnv(cfg)
    env.reset([world], n_players=2)
    T = env.T if max_steps is None else min(max_steps, env.T)
    turn_ms, switched, steps = [], [], 0
    for t in range(T):
        t0 = time.perf_counter()
        a0, st = agent.act(env, t)
        turn_ms.append(1000.0 * (time.perf_counter() - t0))
        switched.append(1.0 if st["switched"] else 0.0)
        a1 = _greedy_act(cfg, opp_net, env, 1)
        env_step(cfg, env, a0, seats=[{"pid": 1, "action": a1}], step_idx=t)
        steps = t + 1
        _, _, al0, al1 = settle(env)
        if not (bool(al0[0].item()) and bool(al1[0].item())):
            break
    s0, s1, _, _ = settle(env)
    s0, s1 = float(s0[0].item()), float(s1[0].item())
    score = 1.0 if s0 > s1 else 0.0 if s0 < s1 else 0.5
    mean_ms = sum(turn_ms) / max(1, len(turn_ms))
    sr = sum(switched) / max(1, len(switched))
    return score, (s0, s1), steps, mean_ms, sr


def play_greedy_vs_neural(cfg, net, opp_net, world, max_steps=None):
    """B=1 game, seat0 = `net` GREEDY (no search), seat1 = opp_net greedy. Same harness as the
    MCTS game so the only difference is the search. Returns (score_seat0, (s0,s1), n_steps)."""
    env = GpuEnv(cfg)
    env.reset([world], n_players=2)
    T = env.T if max_steps is None else min(max_steps, env.T)
    steps = 0
    with torch.no_grad():
        for t in range(T):
            a0 = _greedy_act(cfg, net, env, 0)
            a1 = _greedy_act(cfg, opp_net, env, 1)
            env_step(cfg, env, a0, seats=[{"pid": 1, "action": a1}], step_idx=t)
            steps = t + 1
            _, _, al0, al1 = settle(env)
            if not (bool(al0[0].item()) and bool(al1[0].item())):
                break
    s0, s1, _, _ = settle(env)
    s0, s1 = float(s0[0].item()), float(s1[0].item())
    return (1.0 if s0 > s1 else 0.0 if s0 < s1 else 0.5), (s0, s1), steps


def make_agent(cfg, net, args):
    # flat per-turn budget (schedule=None) -> measures search lift at a FIXED budget; max_sims caps it.
    return mcts.MctsAgent(cfg, net, opp_kind=args.opp_model, K=args.K, c_puct=args.c_puct,
                          value=args.value, schedule=None, turn_budget_s=args.turn_budget,
                          max_sims=args.sims, bank_s=args.bank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=16, help="worlds per opponent (MCTS is B=1, slow)")
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--value", default="heuristic", choices=["heuristic", "net"])
    ap.add_argument("--opp-model", default="greedy", choices=["starter", "greedy", "intermediate", "medium"],
                    help="scripted opponent the SEARCH plans against (real opp is still neural)")
    ap.add_argument("--turn-budget", type=float, default=0.5, help="per-turn wall-clock guard (s)")
    ap.add_argument("--crit-budget", type=float, default=2.0, help="critical-turn wall-clock guard (s)")
    ap.add_argument("--bank", type=float, default=60.0, help="overage bank (s) for critical turns")
    ap.add_argument("--max-steps", type=int, default=None, help="cap game length (ticks)")
    ap.add_argument("--calibrate", action="store_true", help="1 game vs first opp, print timing, exit")
    args = ap.parse_args()

    cfg = Config.create(smoke=False, CKPT_DIR=os.path.join("runs", "mcts_eval"))
    dev = cfg.device
    mnet, mc = load_snapshot(cfg, MCTS_CKPT); mnet = mnet.to(dev).eval()
    mck = torch.load(MCTS_CKPT, map_location="cpu", weights_only=False)
    print("==== device %s | EPISODE_STEPS=%d | MCTS sims=%d K=%d budget=%.2fs/crit%.2fs opp-model=%s value=%s ===="
          % (dev, cfg.EPISODE_STEPS, args.sims, args.K, args.turn_budget, args.crit_budget,
             args.opp_model, args.value))
    print("  MCTS net: %s  h%s x %sblk  saved elo=%s elo4=%s"
          % (os.path.basename(MCTS_CKPT), mc["HIDDEN"], mc["N_RES_BLOCKS"], _e(mck.get("elo")), _e(mck.get("elo4"))))

    opps = []
    for name, path in OPPONENTS:
        net, _ = load_snapshot(cfg, path); net = net.to(dev).eval()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        print("  opponent %-12s saved elo=%s elo4=%s" % (name, _e(ck.get("elo")), _e(ck.get("elo4"))))
        opps.append((name, net))

    worlds = make_world_pool(cfg, max(args.n_worlds, 8), base_seed=cfg.SEED)

    if args.calibrate:
        print("\n---- CALIBRATION: 1 MCTS game vs %s (max_steps=%s) ----" % (opps[0][0], args.max_steps))
        agent = make_agent(cfg, mnet, args)
        t0 = time.perf_counter()
        sc, (s0, s1), n, ms, sr = play_mcts_vs_neural(cfg, agent, opps[0][1], worlds[0], args.max_steps)
        wall = time.perf_counter() - t0
        print("  result: MCTS %.1f (%.0f vs %.0f) in %d steps | %.1fs wall | mean %.0f ms/turn | switch %.0f%%"
              % (sc, s0, s1, n, wall, ms, 100 * sr))
        print("  => est. full 500-step game ~%.0fs; %d worlds x 3 opp ~%.1f min (MCTS only)"
              % (ms * n / 1000.0 * (500.0 / max(1, n)), args.n_worlds, 3 * args.n_worlds * (ms * 250 / 1000.0) / 60.0))
        return

    # ---- 2p H2H: MCTS(seat0, search) vs each neural opp(seat1); greedy(seat0) baseline same worlds ----
    print("\n==== 2p H2H: MCTS-192x6 (seat0, search) vs each opponent (seat1) ====")
    print("  [greedy = same 192x6 net at seat0 WITHOUT search, identical worlds -> isolates search uplift]")
    rows = []
    for name, onet in opps:
        m_scores, g_scores, tturn, tsw = [], [], [], []
        for i in range(args.n_worlds):
            w = worlds[i % len(worlds)]
            agent = make_agent(cfg, mnet, args)               # fresh per game (resets overage bank)
            sc, (s0, s1), n, ms, sr = play_mcts_vs_neural(cfg, agent, onet, w, args.max_steps)
            gsc, _, _ = play_greedy_vs_neural(cfg, mnet, onet, w, args.max_steps)
            m_scores.append(sc); g_scores.append(gsc); tturn.append(ms); tsw.append(sr)
            print("    [%s] world %2d: MCTS %.1f (%4.0f vs %4.0f, %3d steps, %.0fms/turn, sw %2.0f%%) | greedy %.1f"
                  % (name, i, sc, s0, s1, n, ms, 100 * sr, gsc), flush=True)
        mm = sum(m_scores) / len(m_scores); gg = sum(g_scores) / len(g_scores)
        rows.append((name, mm, gg, sum(tturn) / len(tturn), sum(tsw) / len(tsw)))
        print("  ---- %-12s : MCTS seat0 win+0.5draw = %.3f | greedy seat0 = %.3f | search uplift = %+.3f"
              % (name, mm, gg, mm - gg))

    print("\n==== SUMMARY (seat0 only; win+0.5*draw vs each opponent over %d worlds) ====" % args.n_worlds)
    print("    %-12s | %-10s | %-10s | %-12s | %-9s | %-8s" %
          ("opponent", "MCTS s0", "greedy s0", "search lift", "ms/turn", "switch%"))
    for name, mm, gg, ms, sr in rows:
        print("    %-12s | %10.3f | %10.3f | %+12.3f | %9.0f | %7.0f%%" % (name, mm, gg, mm - gg, ms, 100 * sr))
    mean_m = sum(r[1] for r in rows) / len(rows); mean_g = sum(r[2] for r in rows) / len(rows)
    print("    %-12s | %10.3f | %10.3f | %+12.3f |" % ("(mean)", mean_m, mean_g, mean_m - mean_g))

    print("\n  NOTE on 4p arena: the v12 MCTS PoC is seat-0-only and 2p-only (ego hard-coded to seat 0,")
    print("  single scripted opponent in the tree, 2p terminal value). A seat-balanced all-neural 4p")
    print("  arena (every agent visits every seat) therefore cannot include this searcher as-is; a fair")
    print("  4p number needs a 4p/decoupled-UCT extension of mcts.py (the PoC explicitly defers this).")


if __name__ == "__main__":
    main()
