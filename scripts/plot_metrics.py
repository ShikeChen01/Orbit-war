"""Plot training curves from a native run's metrics.csv into one multi-panel PNG.

    python scripts/plot_metrics.py runs/grpo/ppo_curric/metrics.csv --stage1 100 --stage2 100
    python scripts/plot_metrics.py a/metrics.csv b/metrics.csv --labels PPO GRPO   # overlay compare
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np


def load(path):
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k, v in r.items():
                try: rows.setdefault(k, []).append(float(v))
                except (ValueError, TypeError): pass
    return {k: np.array(v) for k, v in rows.items()}


PANELS = [
    ("win rate %", [("wr_starter", "vs starter"), ("wr_random", "vs random")]),
    ("launches / step", [("valid_per_step", "valid"), ("launch_per_step", "launched"), ("invalid_per_step", "invalid")]),
    ("episode return", [("ep_return", "ret")]),
    ("episode length", [("ep_len", "len")]),
    ("sigma (exploration)", [("sigma", "sigma")]),
    ("value loss / approx_kl", [("loss_vf", "vf"), ("approx_kl", "approx_kl")]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stage1", type=int, default=0, help="iter where stage 2 begins (draws a divider)")
    ap.add_argument("--stage2", type=int, default=0, help="#iters of stage 2 (divider at stage1+stage2)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [load(p) for p in args.csv]
    labels = args.labels or [os.path.basename(os.path.dirname(p)) or os.path.basename(p) for p in args.csv]
    dividers = []
    if args.stage1: dividers.append(args.stage1)
    if args.stage1 and args.stage2: dividers.append(args.stage1 + args.stage2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (title, series) in zip(axes.flat, PANELS):
        for ri, d in enumerate(runs):
            x = d.get("iter")
            if x is None: continue
            for col, lab in series:
                if col in d:
                    tag = lab if len(runs) == 1 else f"{labels[ri]}:{lab}"
                    ax.plot(x, d[col], label=tag, lw=1.4)
        for dv in dividers:
            ax.axvline(dv, color="#888", ls="--", lw=0.8)
        ax.set_title(title); ax.set_xlabel("iteration"); ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("  vs  ".join(labels), fontsize=13)
    fig.tight_layout()
    out = args.out or (os.path.join(os.path.dirname(args.csv[0]), "metrics.png") if len(runs) == 1
                       else "runs/metrics_compare.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
