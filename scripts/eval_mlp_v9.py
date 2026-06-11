"""Evaluate v9 MLP-trunk checkpoints (F_DIM 104) in 2p AND 4p on the local GPU.

Execs the REAL cells of notebooks/setup1_v9_a100.ipynb (FULL config, not SMOKE) up to & incl.
the gauntlet cell, so the canonical env + model + scripted-anchor gauntlets are available, then:

  * 2p gauntlet  -- greedy learner vs each fixed scripted anchor (the metric `elo` is grounded on)
  * 4p gauntlet  -- greedy learner (seat 0) FFA vs the fixed scripted trio (grounds `elo4`)
  * neural H2H   -- it201 vs it351 head-to-head, both seat assignments (2p) + a 4p FFA duel

    python scripts/eval_mlp_v9.py
    python scripts/eval_mlp_v9.py --n-envs 256 --ckpts notebooks/weight/MLPs/it201.pt ...
"""
import argparse
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
        print("[eval] exec cell %2d: %s" % (i, src.splitlines()[0][:72]), flush=True)
        exec(compile(src, "cell%d" % i, "exec"), ns)
        if STOP_AFTER in src:
            break
    assert ns["F_DIM"] == 104, "v9 must run the one-hot obs (F_DIM 104), got %r" % ns["F_DIM"]
    return ns


def load_net(ns, path):
    """Load any save_ckpt .pt onto DEVICE. F_DIM-101 (v7/v8) ckpts are auto-wrapped by
    load_snapshot's LegacyObsAdapter so they consume the v9 one-hot obs (collapsing internally
    to their native scalar-ownership view -- bit-exact to how they actually run)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net, cfg = ns["load_snapshot"](path)              # build_from_cfg + load + wrap_if_legacy
    net = net.to(ns["DEVICE"]).eval()
    return net, ck


# ---------------------------------------------------------------------------
# neural head-to-head (mirrors collect_ppo's opponent path, greedy, no PPO buffer)
# ---------------------------------------------------------------------------
def h2h_2p(ns, netA, netB, worlds, n_envs):
    """Greedy mean score of netA (seat 0) vs netB (seat 1) on `worlds`. win=1 draw=.5 loss=0."""
    GpuEnv, env_encode, act, env_step, settle = (
        ns["GpuEnv"], ns["env_encode"], ns["act"], ns["env_step"], ns["settle"])
    DEVICE, T = ns["DEVICE"], ns["EPISODE_STEPS"]
    env = GpuEnv(ns["PLANET_CAP"], ns["FLEET_CAP"], T, ns["SHIP_SPEED"], DEVICE)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)], n_players=2)
    active = torch.ones(n_envs, device=DEVICE); score = torch.zeros(n_envs, device=DEVICE)
    def _sc(s0, s1):
        return torch.where(s0 > s1, torch.ones_like(s0),
                           torch.where(s0 < s1, torch.zeros_like(s0), torch.full_like(s0, 0.5)))
    with torch.no_grad():
        for t in range(T):
            e, m, a, g = env_encode(env, 0); a0, _, _ = act(netA, e, m, a, g, greedy=True)
            oe, om, oa, og = env_encode(env, 1); a1, _, _ = act(netB, oe, om, oa, og, greedy=True)
            env_step(env, a0, seats=[{"pid": 1, "action": a1}], step_idx=t)
            s0, s1, al0, al1 = settle(env)
            term = (env.step_ct >= float(env.T - 2)) | (~(al0 & al1))
            newly = (active > 0.5) & term
            score = torch.where(newly, _sc(s0, s1), score)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s0, s1, _, _ = settle(env)
        score = torch.where(active > 0.5, _sc(s0, s1), score)
    return float(score.mean().item())


def h2h_4p(ns, netA, netB, fill_kind, worlds, n_envs):
    """4p FFA: seat0=netA, seat1=netB, seats 2&3 = scripted `fill_kind`. Returns netA's mean
    pairwise score AND netA-vs-netB pairwise score (seat 0 vs seat 1)."""
    GpuEnv, env_encode, act, env_step, settle_n = (
        ns["GpuEnv"], ns["env_encode"], ns["act"], ns["env_step"], ns["settle_n"])
    DEVICE, T, DTYPE = ns["DEVICE"], ns["EPISODE_STEPS"], ns["DTYPE"]
    opp = ns["_OPP_BY_KIND"][fill_kind]
    env = GpuEnv(ns["PLANET_CAP"], ns["FLEET_CAP"], T, ns["SHIP_SPEED"], DEVICE)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)], n_players=4)
    active = torch.ones(n_envs, device=DEVICE)
    pw_all = torch.zeros(n_envs, device=DEVICE)   # seat0 mean pairwise vs seats 1,2,3
    pw_AB = torch.zeros(n_envs, device=DEVICE)    # seat0 vs seat1 only
    def _pwise(a, b):
        return torch.where(a > b, torch.ones_like(a), torch.where(a < b, torch.zeros_like(a), 0.5 * torch.ones_like(a)))
    with torch.no_grad():
        for t in range(T):
            e, m, a, g = env_encode(env, 0); a0, _, _ = act(netA, e, m, a, g, greedy=True)
            oe, om, oa, og = env_encode(env, 1); a1, _, _ = act(netB, oe, om, oa, og, greedy=True)
            seats = [{"pid": 1, "action": a1},
                     {"pid": 2, "script": opp}, {"pid": 3, "script": opp}]
            env_step(env, a0, seats=seats, step_idx=t)
            s_all, alive_all = settle_n(env)
            n_alive = alive_all.to(DTYPE).sum(1)
            term = (env.step_ct >= float(env.T - 2)) | (n_alive <= 1.0) | (~alive_all[:, 0])
            newly = (active > 0.5) & term
            s0 = s_all[:, 0]
            mean_pw = torch.stack([_pwise(s0, s_all[:, k]) for k in (1, 2, 3)], 0).mean(0)
            ab = _pwise(s0, s_all[:, 1])
            pw_all = torch.where(newly, mean_pw, pw_all)
            pw_AB = torch.where(newly, ab, pw_AB)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s_all, _ = settle_n(env)
        s0 = s_all[:, 0]
        mean_pw = torch.stack([_pwise(s0, s_all[:, k]) for k in (1, 2, 3)], 0).mean(0)
        ab = _pwise(s0, s_all[:, 1])
        pw_all = torch.where(active > 0.5, mean_pw, pw_all)
        pw_AB = torch.where(active > 0.5, ab, pw_AB)
    return float(pw_all.mean().item()), float(pw_AB.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+",
                    default=["notebooks/weight/MLPs/it201.pt", "notebooks/weight/MLPs/it351.pt"])
    ap.add_argument("--n-envs", type=int, default=256, help="games per anchor / per H2H assignment")
    args = ap.parse_args()

    ns = load_ns()
    DEVICE = ns["DEVICE"]
    worlds = ns["make_world_pool"](args.n_envs, base_seed=ns["SEED"])
    anchors2 = ["starter", "intermediate", "greedy", "medium"]   # 2p scripted opponents
    trio4 = ns["GAUNTLET4_SEATS"]                                # ("starter","greedy","intermediate")

    print("\n==== device %s | n_envs=%d | EPISODE_STEPS=%d ====" % (DEVICE, args.n_envs, ns["EPISODE_STEPS"]))

    def _e(v):
        return "%.0f" % v if isinstance(v, (int, float)) else "--"

    nets = {}
    rows = []
    for path in args.ckpts:
        name = os.path.splitext(os.path.basename(path))[0]
        net, ck = load_net(ns, path)
        nets[name] = net
        fd = int(ck["config"].get("F_DIM"))
        tag = "v7/legacy F_DIM=%d (wrapped)" % fd if fd != ns["F_DIM"] else "v9 F_DIM=%d" % fd
        print("\n---- %s [%s] (iter=%s saved elo=%s elo4=%s) ----"
              % (name, tag, ck.get("iter"), _e(ck.get("elo")), _e(ck.get("elo4"))))

        # 2p: per-anchor + mean
        per2 = {a: ns["_eval_score"](net, a, worlds, args.n_envs) for a in anchors2}
        mean2 = sum(per2.values()) / len(per2)
        print("  2p vs anchors:  " + "  ".join("%s=%.3f" % (a, per2[a]) for a in anchors2)
              + "   | mean=%.3f" % mean2)

        # 4p: per-seat trio breakdown via the trio gauntlet + the headline number
        g4 = ns["eval_gauntlet4"](net, worlds, n_envs=args.n_envs)
        print("  4p FFA vs trio %s:  mean pairwise=%.3f" % (str(trio4), g4))
        rows.append((name, per2, mean2, g4, ck))

    # ---- neural head-to-head (all unordered pairs, seat-averaged) ----
    import itertools
    names = list(nets)
    if len(names) >= 2:
        print("\n==== neural head-to-head (2p, seat-averaged score of row vs col) ====")
        for A, B in itertools.combinations(names, 2):
            ab = h2h_2p(ns, nets[A], nets[B], worlds, args.n_envs)   # A seat0
            ba = h2h_2p(ns, nets[B], nets[A], worlds, args.n_envs)   # B seat0
            a_overall = 0.5 * (ab + (1.0 - ba))                      # A's seat-averaged score
            print("  %-8s vs %-8s : %s=%.3f  %s=%.3f   (raw: %s@s0=%.3f, %s@s0=%.3f)"
                  % (A, B, A, a_overall, B, 1.0 - a_overall, A, ab, B, ba))
        print("\n==== neural head-to-head (4p FFA: A=seat0, B=seat1, seats 2&3 = greedy bots) ====")
        for A, B in itertools.combinations(names, 2):
            _, pwab = h2h_4p(ns, nets[A], nets[B], "greedy", worlds, args.n_envs)  # A vs B, A@s0
            _, pwba = h2h_4p(ns, nets[B], nets[A], "greedy", worlds, args.n_envs)  # B vs A, B@s0
            a_pw = 0.5 * (pwab + (1.0 - pwba))                       # A's seat-averaged vs B
            print("  %-8s vs %-8s : %s=%.3f  %s=%.3f   (raw pairwise: %s@s0=%.3f, %s@s0=%.3f)"
                  % (A, B, A, a_pw, B, 1.0 - a_pw, A, pwab, B, pwba))

    # ---- summary table ----
    print("\n==== SUMMARY (greedy, win+0.5*draw) ====")
    print("%-8s | %-7s | %-7s | %-7s | %-7s | %-8s | %-9s | %-9s"
          % ("ckpt", "starter", "interm", "greedy", "medium", "2p mean", "4p FFA", "saved elo/4"))
    for name, per2, mean2, g4, ck in rows:
        print("%-8s | %7.3f | %7.3f | %7.3f | %7.3f | %8.3f | %9.3f | %s/%s"
              % (name, per2["starter"], per2["intermediate"], per2["greedy"], per2["medium"],
                 mean2, g4, _e(ck.get("elo")), _e(ck.get("elo4"))))


if __name__ == "__main__":
    main()
