"""Evaluate the 512x32 run3 pair against run2/it726, on the local GPU. Neural-only.

Two views over a shared pool of randomized worlds:

  * 2p H2H     -- run3/it101 vs run3/it51, played at BOTH seat assignments
                  (it101 as P0 and as P1) and seat-averaged so position is balanced.
  * 4-way FFA  -- run3/it101, run3/it51, run2/it726 and a fixed league yardstick
                  (256x32/it351) in one game, one per seat. Run over all 4 cyclic seat
                  rotations so every agent occupies every seat exactly once. Metric =
                  mean pairwise win+0.5*draw vs the field, seat-balanced.

run2/it726 is the "64-layers-by-mistake" run (blk=64); its checkpoint config carries the
real arch, so load_net rebuilds it correctly. All four are v9 (F_DIM=104).

    .venv/Scripts/python scripts/eval_run3_vs_run2.py
    .venv/Scripts/python scripts/eval_run3_vs_run2.py --n-envs 512
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_mlp_v9 import load_ns, load_net, h2h_2p   # noqa: E402  (sets up env on import)
from eval_it51_run3 import ffa_neural               # noqa: E402  (N-player FFA harness)

IT101 = "notebooks/weight/MLPs/512x32/run3/it101.pt"
IT51 = "notebooks/weight/MLPs/512x32/run3/it51.pt"
IT726 = "notebooks/weight/MLPs/512x32/run2(64-layers-by-mistake)/it726.pt"
REF = "notebooks/weight/MLPs/256x32/it351.pt"   # fixed yardstick (4th seat in FFA)


def _e(v):
    return "%.0f" % v if isinstance(v, (int, float)) else "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=256, help="worlds per seating / per seat-rotation")
    ap.add_argument("--it101", default=IT101)
    ap.add_argument("--it51", default=IT51)
    ap.add_argument("--it726", default=IT726)
    ap.add_argument("--ref", default=REF, help="4th agent in the FFA (fixed yardstick)")
    args = ap.parse_args()

    ns = load_ns()
    DEVICE = ns["DEVICE"]
    worlds = ns["make_world_pool"](args.n_envs, base_seed=ns["SEED"])
    print("\n==== device %s | n_envs=%d | EPISODE_STEPS=%d | neural-only ===="
          % (DEVICE, args.n_envs, ns["EPISODE_STEPS"]))

    # ---- load the nets ----
    paths = [args.it101, args.it51, args.it726, args.ref]
    names, nets = [], []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        net, ck = load_net(ns, p)
        cfg = ck["config"]
        arch = "h%s x %sblk  F_DIM=%s" % (cfg.get("HIDDEN"), cfg.get("N_RES_BLOCKS"), cfg.get("F_DIM"))
        print("  loaded %-7s  iter=%-4s  [%s]  saved elo=%s elo4=%s"
              % (name, ck.get("iter"), arch, _e(ck.get("elo")), _e(ck.get("elo4"))))
        names.append(name)
        nets.append(net)
    it101, it51, it726, ref = nets
    n101, n51, n726, nref = names

    # ---- 2p H2H: it101 vs it51, both seatings, seat-averaged ----
    print("\n==== 2p H2H: %s vs %s (seat-averaged) ====" % (n101, n51))
    ab = h2h_2p(ns, it101, it51, worlds, args.n_envs)    # it101 @ P0
    ba = h2h_2p(ns, it51, it101, worlds, args.n_envs)    # it51  @ P0
    s101 = 0.5 * (ab + (1.0 - ba))
    print("    %-18s | %-7s | %-7s | %-9s | %-9s"
          % ("matchup", n101, n51, "%s@P0" % n101, "%s@P0" % n51))
    print("    %-18s | %7.3f | %7.3f | %9.3f | %9.3f"
          % ("%s vs %s" % (n101, n51), s101, 1.0 - s101, ab, ba))

    # ---- 4-way FFA, all cyclic seat rotations (each agent visits every seat once) ----
    P = len(nets)
    print("\n==== 4-way FFA (%s, %d cyclic seat rotations) ===="
          % (" / ".join(names), P))
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
    print("\n  FFA ranking (mean pairwise vs field, seat-balanced; 0.5 = par):")
    print("    %-8s | %-8s | %s" % ("agent", "overall", "  ".join("seat%d" % k for k in range(P))))
    for a in order:
        print("    %-8s | %8.3f | %s"
              % (names[a], ffa_overall[a], "  ".join("%.3f" % per_seat[a][k] for k in range(P))))


if __name__ == "__main__":
    main()
