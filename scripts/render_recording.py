"""Render a recorded Orbit Wars game (ow_reward_trace --record-json) to a GIF + keyframe PNG.

Decoupled from the (archived) Python policy twin: it only DRAWS the per-tick board geometry the
native recorder dumped, so it always matches the current C++ model. One JSON in -> game.gif +
game_keyframes.png out (next to the JSON unless --out given).

    python scripts/render_recording.py runs/viz/ppo_it0100.json
    python scripts/render_recording.py rec.json --out runs/viz/ppo_it0100 --stride 3 --fps 12
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

OWNER_COLORS = {
    -1: ("neutral", "#888888"),
    0: ("YOUR AGENT (p0)", "#3a8bff"),  # blue
    1: ("opponent (p1)", "#ff4040"),    # red
    2: ("p2", "#3ec46d"), 3: ("p3", "#ff9a2e"),
}


def _draw(ax, frame, tick, total, board, sun_r, comet_ids):
    import matplotlib.patches as mpatches
    ax.clear()
    ax.set_xlim(0, board); ax.set_ylim(0, board); ax.set_aspect("equal")
    ax.set_facecolor("#0b0f1a"); ax.set_xticks([]); ax.set_yticks([])
    c = board / 2.0
    ax.add_patch(mpatches.Circle((c, c), sun_r, color="#ffd23f", zorder=1, alpha=0.95))
    s0 = s1 = 0
    for p in frame["planets"]:
        pid, owner, x, y, radius, ships, prod = p
        if owner == 0: s0 += ships
        elif owner == 1: s1 += ships
        if pid in comet_ids:
            ax.add_patch(mpatches.Circle((x, y), max(radius, 0.9), facecolor="#9af6ff",
                                         edgecolor="#e0ffff", lw=0.8, zorder=4, alpha=0.95))
        else:
            ax.add_patch(mpatches.Circle((x, y), radius, facecolor=OWNER_COLORS.get(owner, OWNER_COLORS[-1])[1],
                                         edgecolor="white", lw=0.6, zorder=3, alpha=0.95))
        fs = 6 + min(prod, 6) * 0.7
        ax.text(x, y, str(int(ships)), color="white", ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=5)
    for f in frame["fleets"]:
        owner, x, y, angle, ships = f
        col = OWNER_COLORS.get(owner, OWNER_COLORS[-1])[1]
        ax.scatter([x], [y], marker="^", s=26, color=col, edgecolors="white", linewidths=0.4, zorder=6)
        ax.plot([x, x + 2.6 * np.cos(angle)], [y, y + 2.6 * np.sin(angle)], color=col, lw=1.0, zorder=6, alpha=0.9)
    ax.set_title(f"tick {tick}/{total}   p0(blue) {s0}  vs  p1(red) {s1}",
                 color="white", fontsize=10, fontweight="bold")
    handles = [mpatches.Patch(color=OWNER_COLORS[0][1], label="p0 (agent)"),
               mpatches.Patch(color=OWNER_COLORS[1][1], label="p1 (opp)"),
               mpatches.Patch(color=OWNER_COLORS[-1][1], label="neutral"),
               mpatches.Patch(color="#9af6ff", label="comet")]
    ax.legend(handles=handles, loc="upper right", fontsize=6, facecolor="#0b0f1a",
              edgecolor="#444", labelcolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--out", default=None, help="output prefix (default: alongside the json)")
    ap.add_argument("--stride", type=int, default=3, help="render every Nth tick (keeps GIF small)")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    with open(args.json) as fh:
        doc = json.load(fh)
    board = doc.get("board_size", 100.0); sun_r = doc.get("sun_radius", 10.0)
    comet_ids = set()  # comets flagged per-frame; union is fine for coloring
    ticks = doc["ticks"]
    for fr in ticks: comet_ids |= set(fr.get("comets", []) or [])
    sel = ticks[::max(1, args.stride)]
    if sel[-1] is not ticks[-1]: sel.append(ticks[-1])
    total = ticks[-1]["t"]
    out = args.out or os.path.splitext(args.json)[0]
    label = f"p0={doc.get('ego','?')} vs p1={doc.get('opp','?')}  outcome={doc.get('outcome',0):+.0f}"

    # keyframe montage (always viewable, even if GIF writer hiccups)
    n_keys = min(6, len(sel))
    idxs = np.linspace(0, len(sel) - 1, n_keys, dtype=int)
    figm, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#0b0f1a")
    for ax, i in zip(axes.flat, idxs):
        _draw(ax, sel[i], sel[i]["t"], total, board, sun_r, comet_ids)
    for ax in axes.flat[len(idxs):]: ax.axis("off")
    figm.suptitle(f"Orbit Wars keyframes  --  {label}", color="white", fontsize=12)
    figm.savefig(out + "_keyframes.png", dpi=90, facecolor="#0b0f1a", bbox_inches="tight")
    plt.close(figm)
    print("wrote", out + "_keyframes.png")

    fig, ax = plt.subplots(figsize=(8, 8.5), facecolor="#0b0f1a")
    def upd(i): _draw(ax, sel[i], sel[i]["t"], total, board, sun_r, comet_ids); return ax.patches
    anim = animation.FuncAnimation(fig, upd, frames=len(sel), interval=1000 / args.fps, blit=False)
    anim.save(out + ".gif", writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    print("wrote", out + ".gif", f"({len(sel)} frames)")


if __name__ == "__main__":
    main()
