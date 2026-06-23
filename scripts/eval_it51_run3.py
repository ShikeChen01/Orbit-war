"""Evaluate run3/it51 (512x32) against the three 256x32 league agents (it201/it351/it401).

Per the request: NEURAL-ONLY (no scripted anchors). Two views, both over a pool of
randomized worlds ("different world settings") with seats swapped so position is balanced:

  * 4-way FFA   -- all four nets in one game, one per seat. Run over all 4 cyclic seat
                   rotations so every agent occupies every seat exactly once
                   ("switch positions"). Metric = mean pairwise win+0.5*draw vs the field.
  * 2p H2H      -- it51 vs each of the three, played at BOTH seat assignments
                   (it51 as P0 and as P1) and seat-averaged.

    python scripts/eval_it51_run3.py
    python scripts/eval_it51_run3.py --n-envs 256
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_mlp_v9 import load_ns, load_net, h2h_2p   # noqa: E402  (sets up env on import)
import torch                                          # noqa: E402

IT51 = "notebooks/weight/MLPs/512x32/run3/it51.pt"
TRIO = [
    "notebooks/weight/MLPs/256x32/it201.pt",
    "notebooks/weight/MLPs/256x32/it351.pt",
    "notebooks/weight/MLPs/256x32/it401.pt",
]


def ffa_neural(ns, nets_by_seat, worlds, n_envs):
    """N-player FFA, seat k driven by neural net nets_by_seat[k]. Game ends when <=1 seat
    is alive or the clock runs out; final standings are scored pairwise (win=1, draw=.5,
    loss=0). Returns a length-P tensor: each SEAT's mean pairwise score across `worlds`."""
    GpuEnv, env_encode, act, env_step, settle_n = (
        ns["GpuEnv"], ns["env_encode"], ns["act"], ns["env_step"], ns["settle_n"])
    DEVICE, T, DTYPE = ns["DEVICE"], ns["EPISODE_STEPS"], ns["DTYPE"]
    P = len(nets_by_seat)
    env = GpuEnv(ns["PLANET_CAP"], ns["FLEET_CAP"], T, ns["SHIP_SPEED"], DEVICE)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)], n_players=P)
    active = torch.ones(n_envs, device=DEVICE)
    pw = torch.zeros(P, n_envs, device=DEVICE)

    def _pwise(a, b):
        return torch.where(a > b, torch.ones_like(a),
                           torch.where(a < b, torch.zeros_like(a), 0.5 * torch.ones_like(a)))

    def snapshot(s_all):                       # P x n_envs : each seat's mean pairwise vs others
        rows = []
        for k in range(P):
            others = [_pwise(s_all[:, k], s_all[:, j]) for j in range(P) if j != k]
            rows.append(torch.stack(others, 0).mean(0))
        return torch.stack(rows, 0)

    with torch.no_grad():
        for t in range(T):
            e, m, a, g = env_encode(env, 0)
            a0, _, _ = act(nets_by_seat[0], e, m, a, g, greedy=True)
            seats = []
            for k in range(1, P):
                ek, mk, ak, gk = env_encode(env, k)
                akt, _, _ = act(nets_by_seat[k], ek, mk, ak, gk, greedy=True)
                seats.append({"pid": k, "action": akt})
            env_step(env, a0, seats=seats, step_idx=t)
            s_all, alive_all = settle_n(env)
            n_alive = alive_all.to(DTYPE).sum(1)
            term = (env.step_ct >= float(env.T - 2)) | (n_alive <= 1.0)
            newly = (active > 0.5) & term
            cur = snapshot(s_all)
            pw = torch.where(newly.unsqueeze(0), cur, pw)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s_all, _ = settle_n(env)
        cur = snapshot(s_all)
        pw = torch.where((active > 0.5).unsqueeze(0), cur, pw)
    return pw.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=256, help="worlds per seat-rotation / per H2H seating")
    ap.add_argument("--it51", default=IT51)
    ap.add_argument("--trio", nargs="+", default=TRIO)
    args = ap.parse_args()

    ns = load_ns()
    DEVICE = ns["DEVICE"]
    worlds = ns["make_world_pool"](args.n_envs, base_seed=ns["SEED"])

    print("\n==== device %s | n_envs=%d | EPISODE_STEPS=%d | neural-only ===="
          % (DEVICE, args.n_envs, ns["EPISODE_STEPS"]))

    # ---- load the four nets ----
    def _e(v):
        return "%.0f" % v if isinstance(v, (int, float)) else "--"

    paths = [args.it51] + list(args.trio)
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
    P = len(nets)
    hero = names[0]

    # ---- 4-way FFA, all cyclic seat rotations (each agent visits every seat once) ----
    print("\n==== 4-way FFA (all %d neural, %d cyclic seat rotations) ====" % (P, P))
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

    # ---- 2p H2H: it51 vs each of the trio, both seatings, seat-averaged ----
    print("\n==== 2p H2H: %s vs each (seat-averaged; raw = score at P0) ====" % hero)
    print("    %-18s | %-7s | %-7s | %-9s | %-9s"
          % ("matchup", hero, "opp", "%s@P0" % hero, "opp@P0"))
    h2h_overall = {}
    for b in range(1, P):
        ab = h2h_2p(ns, nets[0], nets[b], worlds, args.n_envs)   # it51 @ P0
        ba = h2h_2p(ns, nets[b], nets[0], worlds, args.n_envs)   # opp  @ P0
        hero_score = 0.5 * (ab + (1.0 - ba))
        h2h_overall[names[b]] = hero_score
        print("    %-18s | %7.3f | %7.3f | %9.3f | %9.3f"
              % ("%s vs %s" % (hero, names[b]), hero_score, 1.0 - hero_score, ab, ba))
    mean_h2h = sum(h2h_overall.values()) / len(h2h_overall)
    print("    %-18s | %7.3f |" % ("(mean over trio)", mean_h2h))


if __name__ == "__main__":
    main()
