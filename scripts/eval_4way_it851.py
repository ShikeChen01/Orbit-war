"""Evaluate four v9 MLP checkpoints against each other on the local GPU. Neural-only.

Hero is it851 (512x86); field is run3/it1426, 256x32/it351 and run3/it751. All are v9
(F_DIM=104), so the v9 notebook's load_snapshot rebuilds each from its own config -- the
512x86 trunk is just a deeper res stack, same obs/heads. Two seat-balanced views over one
shared pool of randomized worlds:

  * 2p H2H     -- full round robin. Every unordered pair is played at BOTH seatings
                  (each agent as P0 and as P1) and seat-averaged so position is neutral.
                  Prints the win-rate matrix + each agent's mean vs the field.
  * 4p arena   -- all four nets in one game, one per seat, over all 4 cyclic seat rotations
                  so every agent occupies every seat exactly once. Metric = mean pairwise
                  win+0.5*draw vs the field (0.5 = par), seat-balanced.

    .venv/Scripts/python scripts/eval_4way_it851.py
    .venv/Scripts/python scripts/eval_4way_it851.py --n-envs 512
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_mlp_v9 import load_ns, load_net, h2h_2p   # noqa: E402  (sets up env on import)
from eval_it51_run3 import ffa_neural               # noqa: E402  (N-player FFA harness)

CKPTS = [
    "notebooks/weight/MLPs/512x86/it851.pt",          # hero
    "notebooks/weight/MLPs/512x32/run3/it1426.pt",
    "notebooks/weight/MLPs/256x32/it351.pt",
    "notebooks/weight/MLPs/512x32/run3/it751.pt",
]


def _e(v):
    return "%.0f" % v if isinstance(v, (int, float)) else "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=256,
                    help="worlds per seating (H2H) / per seat-rotation (arena)")
    ap.add_argument("--ckpts", nargs="+", default=CKPTS)
    args = ap.parse_args()

    ns = load_ns()
    DEVICE = ns["DEVICE"]
    worlds = ns["make_world_pool"](args.n_envs, base_seed=ns["SEED"])
    print("\n==== device %s | n_envs=%d | EPISODE_STEPS=%d | neural-only ===="
          % (DEVICE, args.n_envs, ns["EPISODE_STEPS"]))

    # ---- load the nets ----
    names, nets = [], []
    for p in args.ckpts:
        name = os.path.splitext(os.path.basename(p))[0]
        net, ck = load_net(ns, p)
        cfg = ck["config"]
        arch = "h%s x %sblk  F_DIM=%s" % (cfg.get("HIDDEN"), cfg.get("N_RES_BLOCKS"), cfg.get("F_DIM"))
        print("  loaded %-8s  iter=%-5s  [%s]  saved elo=%s elo4=%s"
              % (name, ck.get("iter"), arch, _e(ck.get("elo")), _e(ck.get("elo4"))))
        names.append(name)
        nets.append(net)
    P = len(nets)

    # ---- 2p H2H: full round robin, seat-averaged ----
    print("\n==== 2p H2H (full round robin, seat-averaged; cell = row's score vs col) ====")
    M = [[None] * P for _ in range(P)]   # M[i][j] = row i's seat-averaged score vs col j
    for i, j in itertools.combinations(range(P), 2):
        ab = h2h_2p(ns, nets[i], nets[j], worlds, args.n_envs)   # i @ P0
        ba = h2h_2p(ns, nets[j], nets[i], worlds, args.n_envs)   # j @ P0
        s_i = 0.5 * (ab + (1.0 - ba))                            # i's seat-averaged score
        M[i][j] = s_i
        M[j][i] = 1.0 - s_i
        print("    %-8s vs %-8s : %-8s=%.3f  %-8s=%.3f   (raw: %s@P0=%.3f, %s@P0=%.3f)"
              % (names[i], names[j], names[i], s_i, names[j], 1.0 - s_i,
                 names[i], ab, names[j], ba))

    print("\n  H2H win-rate matrix (row vs col; 0.5 = par):")
    print("    %-9s | %s | %s" % ("", " ".join("%8s" % n for n in names), "  mean"))
    h2h_mean = [0.0] * P
    for i in range(P):
        cells, acc = [], 0.0
        for j in range(P):
            if i == j:
                cells.append("%8s" % "--")
            else:
                cells.append("%8.3f" % M[i][j]); acc += M[i][j]
        h2h_mean[i] = acc / (P - 1)
        print("    %-9s | %s | %6.3f" % (names[i], " ".join(cells), h2h_mean[i]))

    # ---- 4p arena, all cyclic seat rotations (each agent visits every seat once) ----
    print("\n==== 4p arena (all %d neural, %d cyclic seat rotations) ====" % (P, P))
    per_seat = [[None] * P for _ in range(P)]   # per_seat[agent][seat] = mean pairwise
    for r in range(P):
        seat_agent = [(k + r) % P for k in range(P)]            # agent index at each seat
        nets_by_seat = [nets[ai] for ai in seat_agent]
        seat_scores = ffa_neural(ns, nets_by_seat, worlds, args.n_envs)
        print("  rotation %d  seats=[%s] -> pairwise %s"
              % (r, ", ".join(names[ai] for ai in seat_agent),
                 ", ".join("%.3f" % s for s in seat_scores.tolist())))
        for k in range(P):
            per_seat[seat_agent[k]][k] = float(seat_scores[k].item())

    ffa_overall = [sum(per_seat[a]) / P for a in range(P)]
    order = sorted(range(P), key=lambda a: -ffa_overall[a])
    print("\n  4p arena ranking (mean pairwise vs field, seat-balanced; 0.5 = par):")
    print("    %-9s | %-8s | %s" % ("agent", "overall", "  ".join("seat%d" % k for k in range(P))))
    for a in order:
        print("    %-9s | %8.3f | %s"
              % (names[a], ffa_overall[a], "  ".join("%.3f" % per_seat[a][k] for k in range(P))))

    # ---- combined leaderboard ----
    print("\n==== leaderboard ====")
    print("    %-9s | %-9s | %-9s" % ("agent", "2p H2H", "4p arena"))
    for a in sorted(range(P), key=lambda a: -(h2h_mean[a] + ffa_overall[a])):
        print("    %-9s | %9.3f | %9.3f" % (names[a], h2h_mean[a], ffa_overall[a]))


if __name__ == "__main__":
    main()
