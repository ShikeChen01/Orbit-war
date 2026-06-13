"""Replot Orbit Wars v5 training-health curves from a saved metrics.json (dumped by train()).

Each metric gets its OWN panel + y-scale (so sigma / clipfrac / approx_kl are all readable, unlike the
old single shared axis); faint raw line + bold EMA. approx_kl uses symlog.

    python scripts/plot_training_health.py path/to/metrics.json
    python scripts/plot_training_health.py metrics.json --out health.png --ema 0.15
"""
from __future__ import annotations
import argparse, json, os


def _ema(y, a):
    out, m = [], None
    for v in y:
        m = float(v) if m is None else (1 - a) * m + a * float(v)
        out.append(m)
    return out


def plot_health(hist, out_png, ema=0.15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    it = hist["iter"]

    def panel(ax, key, title, pct=False, symlog=False):
        if key not in hist:
            ax.set_visible(False); return
        y = hist[key]
        ax.plot(it, y, color="tab:blue", alpha=0.25, lw=0.8)
        ax.plot(it, _ema(y, ema), color="tab:blue", lw=1.8)
        ax.set_title(title, fontsize=9); ax.grid(alpha=0.3, lw=0.5)
        if pct: ax.set_ylim(-0.02, 1.02)
        if symlog: ax.set_yscale("symlog", linthresh=1e-3)

    fig, ax = plt.subplots(3, 3, figsize=(15, 10))
    panel(ax[0, 0], "return", "episode return (real units)")
    panel(ax[0, 1], "win_rate", "win rate (vs sampled opp)", pct=True)
    panel(ax[0, 2], "learner_elo", "learner Elo")
    panel(ax[1, 0], "sigma", "exploration (sigma / mean entropy)")
    panel(ax[1, 1], "clipfrac", "PPO clip fraction")
    panel(ax[1, 2], "approx_kl", "approx KL (symlog)", symlog=True)
    panel(ax[2, 0], "lnch_per_step", "launches / step")
    panel(ax[2, 1], "r_outcome", "reward: outcome channel")
    a = ax[2, 2]
    any_dense = False
    for k, c in (("r_capture", "tab:green"), ("r_dispatch", "tab:orange"), ("r_milestone", "tab:purple")):
        if k in hist:
            a.plot(it, hist[k], color=c, alpha=0.20, lw=0.8)
            a.plot(it, _ema(hist[k], ema), color=c, lw=1.6, label=k.replace("r_", "")); any_dense = True
    a.set_title("reward: dense channels"); a.grid(alpha=0.3, lw=0.5)
    if any_dense: a.legend(fontsize=7)
    for col in ax[-1]:
        col.set_xlabel("iter")
    fig.suptitle("Orbit Wars v5 -- training health (faint = raw, bold = EMA)", fontsize=12)
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close(fig)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", help="metrics.json saved by train()")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ema", type=float, default=0.15)
    args = ap.parse_args()
    hist = json.load(open(args.metrics))
    out = args.out or os.path.splitext(args.metrics)[0] + "_health.png"
    plot_health(hist, out, ema=args.ema)


if __name__ == "__main__":
    main()
