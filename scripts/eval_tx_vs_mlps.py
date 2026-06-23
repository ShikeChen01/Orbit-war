"""Cross-architecture neural eval: the 512h/12-layer TRANSFORMER vs three ResNet-trunk MLPs.

All four checkpoints are v9-lineage (F_DIM=104, PLANET_CAP=48, gated-alloc action space), so the
v12 package's load_snapshot rebuilds each at its OWN arch from its checkpoint config -- the
transformer (ARCH=transformer, 12 tx-layers, 16 heads) and the trunks (ARCH=trunk, 32/86 res
blocks) all consume the same obs and emit the same action, so they play in one shared GpuEnv.

The v9/v10 notebook scripts can't build the transformer (build_from_cfg there has no transformer
branch), so this uses the orbit_wars_v12 package directly (its PolicyNet handles both archs).

Two seat-balanced views over one shared pool of randomized worlds:

  * 2p H2H     -- full round robin. Every unordered pair is played at BOTH seatings (each agent as
                  P0 and as P1) and seat-averaged so position is neutral. Prints the win-rate
                  matrix + each agent's mean vs the field.
  * 4p arena   -- all four nets in one game, one per seat, over all 4 cyclic seat rotations so every
                  agent occupies every seat exactly once. Metric = mean pairwise win+0.5*draw vs
                  the field, seat-balanced (0.5 = par).

    .venv/Scripts/python scripts/eval_tx_vs_mlps.py
    .venv/Scripts/python scripts/eval_tx_vs_mlps.py --n-envs 512
"""
import argparse
import itertools
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch

from orbit_wars_v12 import Config, load_snapshot, make_world_pool
from orbit_wars_v12.env import GpuEnv, env_encode, env_step, settle_n
from orbit_wars_v12.policy import act
from orbit_wars_v12.league import _score2_vs   # seat-averaged greedy 2p H2H

CKPTS = [
    "notebooks/weight/Transformers/512h-12n-16hd-4v/it426.pt",   # hero: transformer
    "notebooks/weight/MLPs/512x86/it951.pt",
    "notebooks/weight/MLPs/512x32/run3/it1426.pt",
    "notebooks/weight/MLPs/256x32/it351.pt",
]

# short, readable names (parent-dir tag + iter) so the trunks don't all collapse to "itNNNN"
NAMES = ["tx-512h12L", "mlp-512x86", "mlp-512x32", "mlp-256x32"]


def _e(v):
    return "%.0f" % v if isinstance(v, (int, float)) else "--"


def ffa_neural(cfg, nets_by_seat, worlds, n_envs):
    """N-player all-neural FFA, seat k driven by nets_by_seat[k]. Game ends when <=1 seat is alive
    or the clock runs out; standings are scored pairwise (win=1, draw=.5, loss=0). Returns a
    length-P tensor: each SEAT's mean pairwise score across `worlds`."""
    dev, P = cfg.device, len(nets_by_seat)
    env = GpuEnv(cfg)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)], n_players=P)
    active = torch.ones(n_envs, device=dev)
    pw = torch.zeros(P, n_envs, device=dev)

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
        for t in range(env.T):
            a0 = act(cfg, nets_by_seat[0], *env_encode(env, 0), greedy=True)[0]
            seats = []
            for k in range(1, P):
                ak = act(cfg, nets_by_seat[k], *env_encode(env, k), greedy=True)[0]
                seats.append({"pid": k, "action": ak})
            env_step(cfg, env, a0, seats=seats, step_idx=t)
            s_all, alive_all = settle_n(env)
            n_alive = alive_all.to(s_all.dtype).sum(1)
            term = (env.step_ct >= float(env.T - 2)) | (n_alive <= 1.0)
            newly = (active > 0.5) & term
            pw = torch.where(newly.unsqueeze(0), snapshot(s_all), pw)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s_all, _ = settle_n(env)
        pw = torch.where((active > 0.5).unsqueeze(0), snapshot(s_all), pw)
    return pw.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=256,
                    help="worlds per seating (H2H) / per seat-rotation (arena)")
    ap.add_argument("--ckpts", nargs="+", default=CKPTS)
    ap.add_argument("--names", nargs="+", default=NAMES)
    args = ap.parse_args()

    cfg = Config.create(smoke=False, CKPT_DIR=os.path.join("runs", "tx_eval"))
    dev = cfg.device
    worlds = make_world_pool(cfg, args.n_envs, base_seed=cfg.SEED)
    print("\n==== device %s | n_envs=%d | EPISODE_STEPS=%d | FLEET_CAP=%d | neural-only ===="
          % (dev, args.n_envs, cfg.EPISODE_STEPS, cfg.FLEET_CAP))

    # ---- load the nets (each rebuilds at its own arch from its ckpt config) ----
    names, nets = [], []
    for p, nm in itertools.zip_longest(args.ckpts, args.names):
        name = nm or os.path.splitext(os.path.basename(p))[0]
        net, mcfg = load_snapshot(cfg, p)
        net = net.to(dev).eval()
        ck = torch.load(p, map_location="cpu", weights_only=False)
        arch = mcfg.get("ARCH")
        if arch == "transformer":
            desc = "transformer h%s x %dL/%dh" % (mcfg["HIDDEN"], mcfg["N_TX_LAYERS"], mcfg["N_HEADS"])
        else:
            desc = "trunk h%s x %dblk" % (mcfg["HIDDEN"], mcfg["N_RES_BLOCKS"])
        print("  loaded %-12s  iter=%-5s  [%s  F_DIM=%s]  saved elo=%s elo4=%s"
              % (name, ck.get("iter"), desc, mcfg["F_DIM"], _e(ck.get("elo")), _e(ck.get("elo4"))))
        names.append(name)
        nets.append(net)
    P = len(nets)

    # ---- 2p H2H: full round robin, seat-averaged ----
    print("\n==== 2p H2H (full round robin, seat-averaged; cell = row's score vs col) ====")
    M = [[None] * P for _ in range(P)]
    for i, j in itertools.combinations(range(P), 2):
        s_i = _score2_vs(cfg, nets[i], nets[j], worlds, args.n_envs)   # i's seat-averaged score vs j
        M[i][j] = s_i
        M[j][i] = 1.0 - s_i
        print("    %-12s vs %-12s : %.3f  /  %.3f" % (names[i], names[j], s_i, 1.0 - s_i))

    print("\n  H2H win-rate matrix (row vs col; 0.5 = par):")
    print("    %-12s | %s | %s" % ("", " ".join("%11s" % n for n in names), "   mean"))
    h2h_mean = [0.0] * P
    for i in range(P):
        cells, acc = [], 0.0
        for j in range(P):
            if i == j:
                cells.append("%11s" % "--")
            else:
                cells.append("%11.3f" % M[i][j]); acc += M[i][j]
        h2h_mean[i] = acc / (P - 1)
        print("    %-12s | %s | %7.3f" % (names[i], " ".join(cells), h2h_mean[i]))

    # ---- 4p arena, all cyclic seat rotations (each agent visits every seat once) ----
    print("\n==== 4p arena (all %d neural, %d cyclic seat rotations) ====" % (P, P))
    per_seat = [[None] * P for _ in range(P)]
    for r in range(P):
        seat_agent = [(k + r) % P for k in range(P)]
        seat_scores = ffa_neural(cfg, [nets[ai] for ai in seat_agent], worlds, args.n_envs)
        print("  rotation %d  seats=[%s] -> pairwise %s"
              % (r, ", ".join(names[ai] for ai in seat_agent),
                 ", ".join("%.3f" % s for s in seat_scores.tolist())))
        for k in range(P):
            per_seat[seat_agent[k]][k] = float(seat_scores[k].item())

    ffa_overall = [sum(per_seat[a]) / P for a in range(P)]
    order = sorted(range(P), key=lambda a: -ffa_overall[a])
    print("\n  4p arena ranking (mean pairwise vs field, seat-balanced; 0.5 = par):")
    print("    %-12s | %-8s | %s" % ("agent", "overall", "  ".join("seat%d" % k for k in range(P))))
    for a in order:
        print("    %-12s | %8.3f | %s"
              % (names[a], ffa_overall[a], "  ".join("%.3f" % per_seat[a][k] for k in range(P))))

    # ---- combined leaderboard ----
    print("\n==== leaderboard (sorted by 2p H2H + 4p arena) ====")
    print("    %-12s | %-9s | %-9s" % ("agent", "2p H2H", "4p arena"))
    for a in sorted(range(P), key=lambda a: -(h2h_mean[a] + ffa_overall[a])):
        print("    %-12s | %9.3f | %9.3f" % (names[a], h2h_mean[a], ffa_overall[a]))


if __name__ == "__main__":
    main()
