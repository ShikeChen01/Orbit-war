"""MCTS depth/breadth planner for the step-scheduled deploy budget (orbit_wars_v12.mcts.MctsAgent).

Front-loaded bank (default DEFAULT_SCHEDULE): 30 s over steps 0-20, 20 s over 20-60, 5 s over 60-80,
then greedy; 55 s of the 60 s bank spent, 5 s held as a turbulence reserve. Per-turn budget in a
phase = phase_seconds / phase_length -> 1.5 / 0.5 / 0.25 s.

Each PUCT simulation costs ~1 net forward (after the _candidates dedup) + 1 env_step. For a sim
budget S, a K-ary tree FULLY resolved to depth d needs S ~ K^d, so d ~ log_K(S) -- the conservative
"solidly searched" depth (PUCT's principal variation reaches deeper on forcing lines). This prints
the S and the breadth<->depth envelope per phase, for both nets, at the probe's peak and derated
fp32 rates. Plug a MEASURED forward time in with --fwd-ms to replace the FLOP-model estimate.

    python scripts/mcts_budget_plan.py
    python scripts/mcts_budget_plan.py --fwd-ms 24 --step-ms 4     # measured 256x6 on the target box
"""
import argparse
import math

SCHEDULE = [("0-20  opening", 20, 30.0), ("20-60 midgame", 40, 20.0), ("60-80 late ", 20, 5.0)]
KS = (2, 3, 4, 6, 8)
# (label, params_M, GFLOP/forward) from the validated cost model (B=1, E=48 planets)
NETS = [("tx 256x6 (narrow)", 5.8, 0.536), ("tx 512x12 (current GRPO large)", 44.0, 4.05)]


def plan(label, fwd_ms, step_ms):
    per_sim = fwd_ms + step_ms
    print("\n  %s | forward ~%.0f ms | per-sim ~%.0f ms (1 fwd + %g ms step)" % (label, fwd_ms, per_sim, step_ms))
    print("    %-14s %7s %6s |  depth d=log_K(S) at breadth K = %s" % ("phase (steps)", "s/turn", "sims", " ".join("%4d" % k for k in KS)))
    for lab, n, secs in SCHEDULE:
        per_turn = secs / n
        S = per_turn * 1000.0 / per_sim
        cells = []
        for K in KS:
            cells.append("%4.1f" % (math.log(S) / math.log(K)) if S >= K else " <1 ")
        print("    %-14s %6.2fs %6.1f |                                   %s" % (lab, per_turn, S, " ".join(cells)))
    print("    steps 80+: greedy (1 fwd ~%.0f ms), no search" % fwd_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-ms", type=float, default=4.0, help="per-sim env_step cost on the deploy cpu")
    ap.add_argument("--eff-gflops", type=float, nargs="+", default=[42.0, 20.0],
                    help="effective fp32 GFLOP/s (probe peak ~42; derated ~20 for small B=1 GEMM)")
    ap.add_argument("--fwd-ms", type=float, default=None, help="override with a MEASURED forward time (ms)")
    args = ap.parse_args()
    print("MCTS step-scheduled budget plan (bank 60 s: 30+20+5 = 55 s front-loaded, 5 s turbulence reserve)")
    if args.fwd_ms is not None:
        plan("measured net", args.fwd_ms, args.step_ms)
        return
    for label, pM, gflop in NETS:
        print("\n######## %s  (%.1fM params, %.2f GFLOP/forward) ########" % (label, pM, gflop))
        for eff in args.eff_gflops:
            tag = "PEAK" if eff >= 40 else "derated"
            plan("%s @ %.0f GFLOP/s (%s)" % (label, eff, tag), gflop / eff * 1000.0, args.step_ms)


if __name__ == "__main__":
    main()
