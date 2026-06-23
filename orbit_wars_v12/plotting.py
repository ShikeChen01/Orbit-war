"""Training-health curves (notebook cell 37).

``plot_training_health(hist)`` draws each metric on its own panel (raw faint + EMA bold) from the
``hist`` dict returned by :meth:`orbit_wars_v12.train.Trainer.train`.
"""


def _ema(y, a=0.15):
    out, m = [], None
    for v in y:
        m = float(v) if m is None else (1 - a) * m + a * float(v)
        out.append(m)
    return out


def plot_training_health(hist, show=True, save_path=None):
    """Render the 3x4 training-health grid. Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt

    it = hist["iter"]

    def _panel(ax, key, title, pct=False, symlog=False):
        y = hist[key]
        ax.plot(it, y, color="tab:blue", alpha=0.25, lw=0.8)
        ax.plot(it, _ema(y), color="tab:blue", lw=1.8)
        ax.set_title(title, fontsize=9); ax.grid(alpha=0.3, lw=0.5)
        if pct:
            ax.set_ylim(-0.02, 1.02)
        if symlog:
            ax.set_yscale("symlog", linthresh=1e-3)

    fig, ax = plt.subplots(3, 4, figsize=(20, 10))
    _panel(ax[0, 0], "return", "episode return (real units)")
    _panel(ax[0, 1], "win_rate", "win rate (vs sampled opp)", pct=True)
    _panel(ax[0, 2], "learner_elo", "learner Elo (2p)")
    _panel(ax[1, 0], "sigma", "exploration (sigma / mean entropy)")
    _panel(ax[1, 1], "clipfrac", "PPO clip fraction")
    _panel(ax[1, 2], "approx_kl", "approx KL (symlog)", symlog=True)
    _panel(ax[0, 3], "grad_norm", "grad norm (pre-clip, symlog)", symlog=True)
    _panel(ax[1, 3], "learner_elo4", "learner Elo (4p)")
    _panel(ax[2, 3], "share_4p", "share of 4p iterations", pct=True)
    _panel(ax[2, 0], "lnch_per_step", "launches / step")
    _panel(ax[2, 1], "r_outcome", "reward: outcome channel")
    for k, c in (("r_capture", "tab:green"), ("r_launch", "tab:orange"), ("r_milestone", "tab:purple"),
                 ("r_alive", "tab:blue"), ("r_win_bet", "tab:red")):
        if not hist.get(k) or not any(hist[k]):   # skip channels that are absent (old hist) or all-zero (feature off)
            continue
        ax[2, 2].plot(it, hist[k], color=c, alpha=0.20, lw=0.8)
        ax[2, 2].plot(it, _ema(hist[k]), color=c, lw=1.6, label=k.replace("r_", ""))
    ax[2, 2].set_title("reward: dense channels"); ax[2, 2].grid(alpha=0.3, lw=0.5); ax[2, 2].legend(fontsize=7)
    for a in ax[-1]:
        a.set_xlabel("iter")
    fig.suptitle("Orbit Wars v12 -- training health (faint = raw, bold = EMA)", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    if show:
        plt.show()
    return fig
