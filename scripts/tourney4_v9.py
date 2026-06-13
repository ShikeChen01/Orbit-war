"""True 4-WAY FFA tournament between four v9 MLP-trunk checkpoints (F_DIM 104).

Unlike eval_mlp_v9.py's 4p tests (1 net + 3 scripts, or 2 nets + 2 greedy bots), here all
FOUR checkpoints occupy the four seats of ONE shared game. We run all 24 seat permutations
over a FIXED world pool, so seat position is perfectly balanced (each net sits in each seat
in exactly 6 of the 24 perms; every unordered pair meets in all 24). Then we report:

  * standings   -- each net's mean pairwise score (= expected score vs a random one of the
                   other three; win=1 draw=.5 loss=0) + 1st/2nd/3rd/4th placement rates
  * H2H matrix  -- row's score vs col, aggregated over every shared game

Execs the REAL cells of notebooks/setup1_v9_a100.ipynb (FULL config) up to the gauntlet cell,
exactly like eval_mlp_v9.py, so the canonical env + model loader are in scope.

    python scripts/tourney4_v9.py --ckpts a.pt b.pt c.pt d.pt [--n-envs 256]
"""
import argparse
import itertools
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
os.environ.setdefault("OW_CKPT_DIR", os.path.join("runs", "v9_eval"))
os.makedirs(os.environ["OW_CKPT_DIR"], exist_ok=True)

import torch

NB = os.path.join("notebooks", "setup1_v9_a100.ipynb")
STOP_AFTER = "League._prune = _league_prune"   # last line of the gauntlet/league cell


def load_ns():
    """Exec the notebook's code cells (FULL config) until the gauntlet cell is defined."""
    doc = json.load(open(NB, encoding="utf-8"))
    ns = {"__name__": "nb"}
    for i, c in enumerate(doc["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        print("[tourney] exec cell %2d: %s" % (i, src.splitlines()[0][:64]), flush=True)
        exec(compile(src, "cell%d" % i, "exec"), ns)
        if STOP_AFTER in src:
            break
    assert ns["F_DIM"] == 104, "v9 must run the one-hot obs (F_DIM 104), got %r" % ns["F_DIM"]
    return ns


def load_net(ns, path):
    net, _cfg = ns["load_snapshot"](path)        # build_from_cfg + load + wrap_if_legacy
    return net.to(ns["DEVICE"]).eval(), torch.load(path, map_location="cpu", weights_only=False)


def play4(ns, seat_nets, worlds, n_envs):
    """One batched 4p FFA: seat k driven by seat_nets[k] (greedy). Returns the FROZEN
    final per-seat score tensor (n_envs, 4)."""
    GpuEnv, env_encode, act, env_step, settle_n = (
        ns["GpuEnv"], ns["env_encode"], ns["act"], ns["env_step"], ns["settle_n"])
    DEVICE, T, DTYPE = ns["DEVICE"], ns["EPISODE_STEPS"], ns["DTYPE"]
    env = GpuEnv(ns["PLANET_CAP"], ns["FLEET_CAP"], T, ns["SHIP_SPEED"], DEVICE)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)], n_players=4)
    s_final = torch.zeros(n_envs, 4, device=DEVICE)
    done = torch.zeros(n_envs, device=DEVICE)
    with torch.no_grad():
        for t in range(T):
            e, m, a, g = env_encode(env, 0); a0, _, _ = act(seat_nets[0], e, m, a, g, greedy=True)
            seats = []
            for sd in (1, 2, 3):
                oe, om, oa, og = env_encode(env, sd)
                a_sd, _, _ = act(seat_nets[sd], oe, om, oa, og, greedy=True)
                seats.append({"pid": sd, "action": a_sd})
            env_step(env, a0, seats=seats, step_idx=t)
            s_all, alive_all = settle_n(env)
            n_alive = alive_all.to(DTYPE).sum(1)
            term = (env.step_ct >= float(env.T - 2)) | (n_alive <= 1.0)
            newly = (done < 0.5) & term
            s_final = torch.where(newly[:, None], s_all, s_final)
            done = torch.where(term, torch.ones_like(done), done)
            if done.sum().item() == n_envs:
                break
        s_all, _ = settle_n(env)
        s_final = torch.where(done[:, None] < 0.5, s_all, s_final)
    return s_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="exactly 4 checkpoints (the 4 seats)")
    ap.add_argument("--n-envs", type=int, default=256, help="games per seat permutation (x24 perms)")
    args = ap.parse_args()
    assert len(args.ckpts) == 4, "need exactly 4 checkpoints for a 4-way tournament"

    ns = load_ns()
    DEVICE = ns["DEVICE"]
    worlds = ns["make_world_pool"](args.n_envs, base_seed=ns["SEED"])
    names = [os.path.splitext(os.path.basename(p))[0] for p in args.ckpts]
    nets, cks = [], []
    for p in args.ckpts:
        net, ck = load_net(ns, p)
        nets.append(net); cks.append(ck)
    N = 4

    def _e(v):
        return "%.0f" % v if isinstance(v, (int, float)) else "--"

    print("\n==== 4-WAY FFA TOURNAMENT | device %s | n_envs=%d | EPISODE_STEPS=%d | 24 seat perms ===="
          % (DEVICE, args.n_envs, ns["EPISODE_STEPS"]))
    for i in range(N):
        print("  seat-pool %d = %-8s (iter=%s saved elo=%s elo4=%s)"
              % (i, names[i], cks[i].get("iter"), _e(cks[i].get("elo")), _e(cks[i].get("elo4"))))

    pair_win = torch.zeros(N, N, device=DEVICE)   # pair_win[i,j] = i's score vs j (win+0.5draw), summed
    pair_n = torch.zeros(N, N, device=DEVICE)
    place = torch.zeros(N, 4, device=DEVICE)      # place[i,r] = # games net i finished rank r (0 = 1st)

    perms = list(itertools.permutations(range(N)))   # 24
    for pi, perm in enumerate(perms):
        s_final = play4(ns, [nets[perm[s]] for s in range(N)], worlds, args.n_envs)   # (n_envs, 4)
        # pairwise score per ordered seat pair -> aggregate by NET identity
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                ni, nj = perm[i], perm[j]
                wij = ((s_final[:, i] > s_final[:, j]).float().sum()
                       + 0.5 * (s_final[:, i] == s_final[:, j]).float().sum())
                pair_win[ni, nj] += wij
                pair_n[ni, nj] += s_final.shape[0]
        # placement rank per seat: G[e,s] = # seats scoring strictly higher than s (0 = 1st)
        G = (s_final.unsqueeze(1) > s_final.unsqueeze(2)).sum(2)   # (n_envs, 4)
        for s in range(N):
            ns_i = perm[s]
            for r in range(4):
                place[ns_i, r] += (G[:, s] == r).float().sum()
        print("  [perm %2d/24] seats=(%s)" % (pi + 1, ",".join(names[perm[s]] for s in range(N))), flush=True)

    games_per_net = place.sum(1)                       # = 24 * n_envs
    mean_pw = (pair_win.sum(1) / pair_n.sum(1))        # expected score vs a random opponent
    prate = place / games_per_net[:, None]             # placement-rate rows

    order = sorted(range(N), key=lambda k: -mean_pw[k].item())
    print("\n==== STANDINGS (mean pairwise = win+0.5*draw vs a random one of the other 3) ====")
    print("%-2s | %-8s | %-9s | %-6s | %-6s | %-6s | %-6s | %s"
          % ("#", "ckpt", "mean pw", "1st", "2nd", "3rd", "4th", "saved elo/4"))
    for rank, k in enumerate(order, 1):
        print("%-2d | %-8s | %9.3f | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%% | %s/%s"
              % (rank, names[k], mean_pw[k].item(),
                 100 * prate[k, 0].item(), 100 * prate[k, 1].item(),
                 100 * prate[k, 2].item(), 100 * prate[k, 3].item(),
                 _e(cks[k].get("elo")), _e(cks[k].get("elo4"))))

    print("\n==== HEAD-TO-HEAD matrix (row's score vs col, win+0.5*draw, over all 24x%d games) ====" % args.n_envs)
    print("%-8s | %s" % ("", " | ".join("%-7s" % names[j] for j in range(N))))
    for i in range(N):
        cells = []
        for j in range(N):
            cells.append("   -   " if i == j else "%7.3f" % (pair_win[i, j] / pair_n[i, j]).item())
        print("%-8s | %s" % (names[i], " | ".join(cells)))


if __name__ == "__main__":
    main()
