"""Render the STARTING positions of N training-env games (notebook GpuEnv) so we can see whether
seat 0 (the learner's ego seat in collect_ppo) starts in a better spot than seat 1.

Execs the real v9 notebook cells (FULL config) until GpuEnv + make_world_pool exist, builds the
ACTUAL training world pool (default base_seed 14336 = pool round 7 @ it726), resets a 2p env, and
plots each board's t=0 layout (planets colored by owner, sized by radius, ship counts labeled).
Also prints per-seat positional heuristics + the seat0-seat1 gap averaged over the games, so we can
tell if the seat-0 advantage is positional (asymmetric homes) or just tempo (symmetric homes).

    python scripts/render_train_starts.py                 # 10 games, 2p, seed 14336
    python scripts/render_train_starts.py --n 12 --base-seed 14336 --players 2
"""
import argparse, json, math, os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.pop("SSLKEYLOGFILE", None)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
os.environ.setdefault("OW_CKPT_DIR", os.path.join("runs", "v9_eval"))
os.makedirs(os.environ["OW_CKPT_DIR"], exist_ok=True)

import numpy as np

NB = os.path.join("notebooks", "setup1_v9_a100.ipynb")

OWNER_COLORS = {-1: "#888888", 0: "#3a8bff", 1: "#ff4040", 2: "#3ec46d", 3: "#ff9a2e"}


def load_ns():
    """Exec notebook code cells (FULL config) until GpuEnv AND make_world_pool are defined."""
    doc = json.load(open(NB, encoding="utf-8"))
    ns = {"__name__": "nb"}
    for i, c in enumerate(doc["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        exec(compile(src, "cell%d" % i, "exec"), ns)
        if callable(ns.get("GpuEnv")) and callable(ns.get("make_world_pool")):
            print("[render] env + world-gen ready after cell %d" % i, flush=True)
            break
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=14336)   # pool round 7 @ it726
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--out", default=os.path.join("runs", "viz", "train_starts.png"))
    args = ap.parse_args()

    ns = load_ns()
    GpuEnv, make_world_pool = ns["GpuEnv"], ns["make_world_pool"]
    CENTER = float(ns.get("CENTER", 50.0))
    BOARD = float(ns.get("BOARD_SIZE", 2 * CENTER))
    SUN_R = float(ns.get("SUN_RADIUS", 0.0))

    worlds = make_world_pool(args.n, base_seed=args.base_seed)
    env = GpuEnv(ns["PLANET_CAP"], ns["FLEET_CAP"], ns["EPISODE_STEPS"], ns["SHIP_SPEED"], ns["DEVICE"])
    env.reset(worlds, n_players=args.players)

    al = env.p_alive.cpu().numpy(); ow = env.p_owner.cpu().numpy()
    px = env.p_x.cpu().numpy(); py = env.p_y.cpu().numpy(); pr = env.p_radius.cpu().numpy()
    psh = env.p_ships.cpu().numpy(); ppr = env.p_prod.cpu().numpy()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ncol = 5
    nrow = int(math.ceil(args.n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), facecolor="#0b0f1a")
    axes = np.atleast_1d(axes).flatten()

    R_NEAR = 0.30 * BOARD          # "nearby" radius for the supply heuristic
    rows = []   # per-game metrics
    for b in range(args.n):
        ax = axes[b]
        ax.set_xlim(0, BOARD); ax.set_ylim(0, BOARD); ax.set_aspect("equal")
        ax.set_facecolor("#0b0f1a"); ax.set_xticks([]); ax.set_yticks([])
        if SUN_R > 0:
            ax.add_patch(mpatches.Circle((CENTER, CENTER), SUN_R, color="#ffd23f", zorder=1, alpha=0.9))
        idx = np.where(al[b] > 0.5)[0]
        for i in idx:
            c = OWNER_COLORS.get(int(ow[b, i]), OWNER_COLORS[-1])
            ax.add_patch(mpatches.Circle((px[b, i], py[b, i]), max(pr[b, i], 0.6), facecolor=c,
                                         edgecolor="white", lw=0.5, zorder=3, alpha=0.95))
            ax.text(px[b, i], py[b, i], str(int(psh[b, i])), color="white", ha="center",
                    va="center", fontsize=6, fontweight="bold", zorder=5)

        # ---- positional heuristics per seat ----
        def seat_stats(seat):
            mine = idx[ow[b, idx] == seat]
            neut = idx[ow[b, idx] < 0]
            if len(mine) == 0:
                return None
            hx, hy = px[b, mine[0]], py[b, mine[0]]        # home = first owned planet
            d_neut = np.hypot(px[b, neut] - hx, py[b, neut] - hy)
            near = neut[d_neut <= R_NEAR]
            supply = float(np.sum(ppr[b, neut] / (1.0 + d_neut)))   # prod weighted by 1/dist
            return dict(home_prod=float(ppr[b, mine[0]]), own_prod=float(ppr[b, mine].sum()),
                        d_center=float(np.hypot(hx - CENTER, hy - CENTER)),
                        n_near=int(len(near)), near_prod=float(ppr[b, near].sum()),
                        supply=supply, nearest=float(d_neut.min()) if len(neut) else float("nan"))
        s0, s1 = seat_stats(0), seat_stats(1)
        if s0 and s1:
            rows.append((b, s0, s1))
            ttl = "g%d  own_prod %.0f/%.0f  near_prod %.0f/%.0f  supply %.1f/%.1f" % (
                b, s0["own_prod"], s1["own_prod"], s0["near_prod"], s1["near_prod"], s0["supply"], s1["supply"])
        else:
            ttl = "g%d" % b
        ax.set_title(ttl, color="white", fontsize=7)

    for ax in axes[args.n:]:
        ax.axis("off")
    handles = [mpatches.Patch(color=OWNER_COLORS[0], label="seat0 (learner)"),
               mpatches.Patch(color=OWNER_COLORS[1], label="seat1 (opp)"),
               mpatches.Patch(color=OWNER_COLORS[-1], label="neutral")]
    fig.legend(handles=handles, loc="upper right", fontsize=9, facecolor="#0b0f1a",
               edgecolor="#444", labelcolor="white")
    fig.suptitle("Training-env starting positions (base_seed %d, %dp) — numbers = ship counts"
                 % (args.base_seed, args.players), color="white", fontsize=13)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=95, facecolor="#0b0f1a", bbox_inches="tight")
    print("wrote", os.path.abspath(args.out))

    # ---- aggregate seat0 - seat1 gap (positive => seat0 advantaged) ----
    if rows:
        keys = ["home_prod", "own_prod", "d_center", "n_near", "near_prod", "supply", "nearest"]
        print("\n  per-seat starting heuristics, seat0 - seat1 gap (avg over %d games):" % len(rows))
        print("  %-10s | %8s %8s | %8s" % ("metric", "seat0", "seat1", "gap(0-1)"))
        for k in keys:
            a = np.mean([r[1][k] for r in rows]); b_ = np.mean([r[2][k] for r in rows])
            print("  %-10s | %8.2f %8.2f | %+8.2f" % (k, a, b_, a - b_))
        # fraction of games where seat0's supply/near_prod exceeds seat1's
        f_sup = np.mean([r[1]["supply"] > r[2]["supply"] for r in rows])
        f_np = np.mean([r[1]["near_prod"] > r[2]["near_prod"] for r in rows])
        print("  seat0 supply > seat1 in %.0f%% of games; near_prod > in %.0f%%" % (100 * f_sup, 100 * f_np))


if __name__ == "__main__":
    main()
