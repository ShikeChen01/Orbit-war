"""Watch your trained agent play: render one full Orbit Wars game to a GIF/MP4.

Runs ONE game of the deployed search agent (player 0, blue) vs the built-in
StarterAgent (player 1, red) in the real kaggle simulation, records every tick,
and renders an animated GIF (plus an MP4 if ffmpeg is on PATH and a static
keyframe-montage PNG as a robustness fallback).

Player 0 uses the *same* agent as ``submission.py`` (the deployed SearchAgent) so
the replay is representative -- only ``time_budget`` is lowered to 0.3s so a game
finishes in ~1-2 min instead of ~5. Use ``--fast`` to swap in the raw greedy
PolicyAgent (near-instant) for quick iteration.

    python scripts/visualize_game.py            # deployed SearchAgent (default)
    python scripts/visualize_game.py --fast     # raw policy, quick
    python scripts/visualize_game.py --seed 7   # different game

Outputs land in runs/viz/: game.gif, game.mp4 (if ffmpeg), game_keyframes.png.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# Board constants come straight from the engine so the plot extent / sun match.
from orbit_wars_rl.env.game import (
    BOARD_SIZE,
    CENTER,
    SUN_RADIUS,
    make_kaggle_env,
    scores_by_player,
)
from orbit_wars_rl.agents import StarterAgent

_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs", "grpo", "win1", "best.pt",
)

# Owner -> (display name, fill colour). -1 is neutral. 0/1 are the two seats;
# 2/3 are supported in case the engine ever runs 4 players.
OWNER_COLORS = {
    -1: ("neutral", "#888888"),
    0: ("YOUR AGENT (p0)", "#3a8bff"),  # blue
    1: ("starter (p1)", "#ff4040"),      # red
    2: ("p2", "#3ec46d"),                # green
    3: ("p3", "#ff9a2e"),                # orange
}


def build_player0(fast: bool):
    """Player 0: deployed SearchAgent by default, raw PolicyAgent with --fast.

    Falls back to PolicyAgent if SearchAgent fails to load for any reason.
    Returns (agent, label).
    """
    if fast:
        from orbit_wars_rl.agents.ppo_policy import PolicyAgent
        return PolicyAgent.load(_CKPT, device="cpu"), "PolicyAgent (fast/greedy)"

    try:
        from scripts.search_agent import SearchAgent
        # Same config as submission.py, but a smaller per-turn budget so the
        # whole game records quickly.
        agent = SearchAgent(
            _CKPT, device="cpu", K=8, H=8, value="heuristic",
            opponent_model="starter", mode="per_planet", temperature=1.5,
            backend="py", fast_encode=False, time_budget=0.3,
        )
        return agent, "SearchAgent (deployed, time_budget=0.3)"
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[warn] SearchAgent failed to load ({exc!r}); falling back to PolicyAgent")
        from orbit_wars_rl.agents.ppo_policy import PolicyAgent
        return PolicyAgent.load(_CKPT, device="cpu"), "PolicyAgent (fallback)"


def record_game(p0_fast: bool, seed: int, num_players: int = 2):
    """Run one full game and return (list of per-tick observations, p0 label)."""
    a0, a0_label = build_player0(p0_fast)
    a1 = StarterAgent()
    env = make_kaggle_env(configuration={"seed": seed})
    agents = [a0.to_kaggle_agent(), a1.to_kaggle_agent()]
    a0.reset(); a1.reset()
    print(f"running one game: p0={a0_label}  vs  p1=StarterAgent  (seed={seed}) ...")
    env.run(agents)
    # The board state at tick t lives in steps[t][0]["observation"].
    frames = [step[0]["observation"] for step in env.steps]
    return frames, a0_label, num_players


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _draw_frame(ax, obs, tick, total, num_players):
    """Draw a single tick onto a (cleared) matplotlib Axes."""
    import matplotlib.patches as mpatches

    ax.clear()
    ax.set_xlim(0, BOARD_SIZE)
    ax.set_ylim(0, BOARD_SIZE)
    ax.set_aspect("equal")
    ax.set_facecolor("#0b0f1a")
    ax.set_xticks([]); ax.set_yticks([])

    # Central sun.
    ax.add_patch(mpatches.Circle((CENTER, CENTER), SUN_RADIUS,
                                 color="#ffd23f", zorder=1, alpha=0.95))

    comet_ids = set(obs.get("comet_planet_ids", []) or [])

    # Planets (filled circles); comets are planets whose id is flagged.
    for p in obs.get("planets", []):
        pid, owner, x, y, radius, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        is_comet = pid in comet_ids
        color = OWNER_COLORS.get(owner, OWNER_COLORS[-1])[1]
        if is_comet:
            # Comet bodies: cyan ring so they read as transient hazards.
            ax.add_patch(mpatches.Circle((x, y), max(radius, 0.9),
                                         facecolor="#9af6ff", edgecolor="#e0ffff",
                                         lw=0.8, zorder=4, alpha=0.95))
        else:
            ax.add_patch(mpatches.Circle((x, y), radius, facecolor=color,
                                         edgecolor="white", lw=0.6, zorder=3,
                                         alpha=0.95))
        # Ship count label; size nudged up a touch by production.
        fs = 6 + min(prod, 6) * 0.7
        ax.text(x, y, str(int(ships)), color="white", ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=5)

    # Fleets (small triangles), with a heading line and a tiny ship-count label.
    for f in obs.get("fleets", []):
        _fid, owner, x, y, angle, _from, ships = (
            f[0], f[1], f[2], f[3], f[4], f[5], f[6])
        color = OWNER_COLORS.get(owner, OWNER_COLORS[-1])[1]
        ax.scatter([x], [y], marker="^", s=28, color=color,
                   edgecolors="white", linewidths=0.4, zorder=6)
        # Short line showing heading so motion direction is visible.
        ax.plot([x, x + 2.6 * np.cos(angle)], [y, y + 2.6 * np.sin(angle)],
                color=color, lw=1.0, zorder=6, alpha=0.9)
        ax.text(x + 1.2, y + 1.2, str(int(ships)), color=color, ha="left",
                va="bottom", fontsize=5, zorder=7)

    # Title: tick + per-player score (total ships owned).
    scores = scores_by_player(obs, num_players)
    score_txt = "   ".join(
        f"{OWNER_COLORS.get(i, ('p%d' % i, '#fff'))[0]}: {scores[i]}"
        for i in range(num_players)
    )
    ax.set_title(f"Orbit Wars  |  tick {tick}/{total}\n{score_txt}",
                 color="white", fontsize=10, fontweight="bold")

    # Persistent legend so colours are unambiguous.
    handles = [
        mpatches.Patch(color=OWNER_COLORS[0][1], label=OWNER_COLORS[0][0]),
        mpatches.Patch(color=OWNER_COLORS[1][1], label=OWNER_COLORS[1][0]),
        mpatches.Patch(color=OWNER_COLORS[-1][1], label="neutral"),
        mpatches.Patch(color="#9af6ff", label="comet"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6,
              facecolor="#0b0f1a", edgecolor="#444", labelcolor="white")


def render(frames, p0_label, num_players, outdir, fps=10):
    """Render frames to GIF (+ MP4 if ffmpeg) and a static keyframe PNG."""
    import matplotlib
    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt
    from matplotlib import animation

    os.makedirs(outdir, exist_ok=True)
    total = len(frames) - 1

    # --- static keyframe montage first: always a viewable artifact ---------- #
    png_path = os.path.join(outdir, "game_keyframes.png")
    n_keys = min(6, len(frames))
    idxs = np.linspace(0, len(frames) - 1, n_keys, dtype=int)
    fig_m, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#0b0f1a")
    for ax, t in zip(axes.flat, idxs):
        _draw_frame(ax, frames[t], t, total, num_players)
    for ax in axes.flat[len(idxs):]:
        ax.axis("off")
    fig_m.suptitle(f"Orbit Wars keyframes  --  p0 = {p0_label}",
                   color="white", fontsize=12)
    fig_m.savefig(png_path, dpi=90, facecolor="#0b0f1a", bbox_inches="tight")
    plt.close(fig_m)
    print(f"wrote {png_path}")

    # --- animation ---------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 8.5), facecolor="#0b0f1a")

    def update(i):
        _draw_frame(ax, frames[i], i, total, num_players)
        return ax.patches

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=1000 / fps, blit=False)

    gif_path = os.path.join(outdir, "game.gif")
    anim.save(gif_path, writer=animation.PillowWriter(fps=fps))
    print(f"wrote {gif_path}")

    mp4_path = os.path.join(outdir, "game.mp4")
    try:
        anim.save(mp4_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
        print(f"wrote {mp4_path}")
    except Exception as exc:
        print(f"[info] skipping MP4 (ffmpeg unavailable or failed: {exc!r})")

    plt.close(fig)
    return gif_path, png_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true",
                    help="use the raw greedy PolicyAgent for p0 (near-instant)")
    ap.add_argument("--seed", type=int, default=42, help="game seed (default 42)")
    ap.add_argument("--fps", type=int, default=10, help="animation frames/sec")
    ap.add_argument("--outdir", default=os.path.join("runs", "viz"))
    args = ap.parse_args()

    frames, p0_label, num_players = record_game(args.fast, args.seed)

    final = frames[-1]
    scores = scores_by_player(final, num_players)
    winner = max(range(num_players), key=lambda i: scores[i]) if max(scores) > 0 else -1
    win_txt = "draw" if winner == -1 else OWNER_COLORS.get(winner, (f"p{winner}", ""))[0]
    print("-" * 60)
    print(f"game finished: {len(frames)} ticks recorded")
    print(f"final scores (total ships owned): {scores}")
    print(f"winner: p{winner} -> {win_txt}")
    print("-" * 60)

    gif_path, png_path = render(frames, p0_label, num_players, args.outdir, args.fps)
    print(f"\nDone. Frames: {len(frames)}.  p0 agent: {p0_label}")
    print(f"  GIF: {os.path.abspath(gif_path)}")
    print(f"  PNG: {os.path.abspath(png_path)}")


if __name__ == "__main__":
    main()
