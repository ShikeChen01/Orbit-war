"""Single-env (batch=1) greedy win-rate of a v5 target-actor .pt vs scripted opponents.
The competition runs single-env; batched eval is GPU-fp non-deterministic near the win/loss
boundary (see memory), so this is the deployment-true number. Reuses viz_target_ckpt.play.

    python scripts/eval_target_single.py --ckpt <ckpt.pt> --opponent starter --games 64 --seed 1000
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_target_ckpt import load_notebook_defs, build_actor, play


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--opponent", default="starter")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    g = load_notebook_defs()
    net, cfg, blob = build_actor(g, args.ckpt)
    print("== %s  iter=%s  (HIDDEN=%s blocks=%s attn=%s)  vs %s, %d single-env games ==" % (
        os.path.basename(args.ckpt), blob.get("iter"),
        cfg.get("HIDDEN"), cfg.get("N_RES_BLOCKS"), cfg.get("USE_ATTENTION"),
        args.opponent, args.games))
    w = d = l = 0
    margins = []
    for i in range(args.games):
        _, (s0, s1) = play(g, net, args.opponent, args.steps, args.seed + i)
        margins.append(s0 - s1)
        if s0 > s1: w += 1
        elif s0 < s1: l += 1
        else: d += 1
    n = args.games
    margins.sort()
    print("  win %.3f (%d)  draw %.3f (%d)  loss %.3f (%d)" % (w/n, w, d/n, d, l/n, l))
    print("  mean ship margin %+.0f | median %+.0f | min %+.0f | max %+.0f"
          % (sum(margins)/n, margins[n//2], margins[0], margins[-1]))


if __name__ == "__main__":
    main()
