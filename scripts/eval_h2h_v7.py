"""Head-to-head between two v7 TRANSFORMER target-actor checkpoints in the batched GpuEnv.

Plays checkpoint A vs checkpoint B directly (neural vs neural) on random worlds, the exact
self-play plumbing the trainer uses: A acts as owner 0 (env_encode ego=0), B acts as owner 1
(env_encode ego=1 -> opp_action). Greedy/deterministic by default (matches eval_gauntlet).

Seats are balanced: each world is played twice with A and B swapped between owner0/owner1, so a
side advantage cannot inflate the result. Comets are OFF by default (the recalibration/Elo concern
is about anchor-only grounding, and the head-to-head line is set assuming no comets in the batch).

Reuses the notebook-def loader + arch-from-ckpt builder from eval_kaggle_v7.

    python scripts/eval_h2h_v7.py --a it1041.pt --b it101.pt --games 256
"""
from __future__ import annotations
import os, sys, math, argparse
# claim a GPU BEFORE importing eval_kaggle_v7 (it pins CUDA_VISIBLE_DEVICES=-1 for the CPU kaggle engine)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from eval_kaggle_v7 import load_nb_defs, ARCH_KEYS, NB_DEFAULT  # reuse the v7 notebook loader

CKPT_DIR = os.path.join("notebooks", "weight", "Transformers")


def resolve_ckpt(p):
    """Accept a full path or a bare label (it1041 / it1041.pt) under notebooks/weight/Transformers."""
    for cand in (p, os.path.join(CKPT_DIR, p), os.path.join(CKPT_DIR, p + ".pt")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("checkpoint not found: %s" % p)


def build_actor(g, ckpt, device, fp16_roundtrip=False):
    """Build PolicyNet with the ARCH from the ckpt's own config; strict-load; move to `device`.
    fp16_roundtrip: round float weights to fp16 then back to fp32 (simulates an fp16-stored submission)."""
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = blob.get("config") or {}
    g["DEVICE"] = device
    for k in ARCH_KEYS:
        if k in cfg:
            g[k] = cfg[k]
    net = g["build_policy"]()
    sd = blob["model"]
    if fp16_roundtrip:
        sd = {k: (v.half().float() if torch.is_floating_point(v) else v) for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval(), blob


def play_batch(g, net0, net1, worlds, device, greedy=True):
    """net0 = owner0, net1 = owner1. Returns owner0's per-game score (win=1, draw=.5, loss=0)."""
    GpuEnv, env_encode, act, env_step, settle = (
        g["GpuEnv"], g["env_encode"], g["act"], g["env_step"], g["settle"])
    env = GpuEnv(g["PLANET_CAP"], g["FLEET_CAP"], g["EPISODE_STEPS"], g["SHIP_SPEED"], device)
    env.reset(worlds)
    n = len(worlds)
    active = torch.ones(n, device=device)
    score = torch.zeros(n, device=device)

    def _sc(s0, s1):
        return torch.where(s0 > s1, torch.ones_like(s0),
                           torch.where(s0 < s1, torch.zeros_like(s0), torch.full_like(s0, 0.5)))

    with torch.no_grad():
        for t in range(g["EPISODE_STEPS"]):
            e0, m0, a0, gl0 = env_encode(env, 0)
            act0, _, _ = act(net0, e0, m0, a0, gl0, greedy=greedy)
            e1, m1, a1, gl1 = env_encode(env, 1)
            act1, _, _ = act(net1, e1, m1, a1, gl1, greedy=greedy)
            env_step(env, act0, 1, act1, step_idx=t)          # opp code unused (opp_action given)
            s0, s1, al0, al1 = settle(env)
            term = (env.step_ct >= float(env.T - 2)) | (~(al0 & al1))
            newly = (active > 0.5) & term
            score = torch.where(newly, _sc(s0, s1), score)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s0, s1, _, _ = settle(env)
        score = torch.where(active > 0.5, _sc(s0, s1), score)
    return score


def _tally(scores):
    w = float((scores == 1.0).sum().item())
    d = float((scores == 0.5).sum().item())
    l = float((scores == 0.0).sum().item())
    return w, d, l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="it1041", help="checkpoint A (the one under test)")
    ap.add_argument("--b", default="it101", help="checkpoint B (the reference opponent)")
    ap.add_argument("--games", type=int, default=256, help="distinct worlds; each played twice (seat-swapped)")
    ap.add_argument("--steps", type=int, default=None, help="override episode length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--comets", action="store_true", help="enable comets (default OFF)")
    ap.add_argument("--sample", action="store_true", help="stochastic actions (default greedy)")
    ap.add_argument("--nb", default=NB_DEFAULT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--line", type=float, default=0.80, help="A win-rate pass line")
    ap.add_argument("--fp16-a", action="store_true", help="round checkpoint A weights fp16->fp32 (submission sim)")
    args = ap.parse_args()
    os.chdir(REPO)

    a_ckpt, b_ckpt = resolve_ckpt(args.a), resolve_ckpt(args.b)
    device = torch.device(args.device)
    g = load_nb_defs(args.nb)
    g["COMETS_ENABLED"] = bool(args.comets)
    if args.steps:
        g["EPISODE_STEPS"] = int(args.steps)

    net_a, blob_a = build_actor(g, a_ckpt, device, fp16_roundtrip=args.fp16_a)
    net_b, blob_b = build_actor(g, b_ckpt, device)
    elo_a = (blob_a.get("elo"), blob_a.get("iter"))
    elo_b = (blob_b.get("elo"), blob_b.get("iter"))

    greedy = not args.sample
    print("H2H  A=%s (it%s, Elo %.0f)  vs  B=%s (it%s, Elo %.0f)"
          % (os.path.basename(a_ckpt), elo_a[1], elo_a[0] or 0,
             os.path.basename(b_ckpt), elo_b[1], elo_b[0] or 0))
    print("worlds=%d (x2 seats = %d games) | comets=%s | %s | device=%s | steps=%d\n"
          % (args.games, 2 * args.games, g["COMETS_ENABLED"],
             "greedy" if greedy else "sample", device, g["EPISODE_STEPS"]), flush=True)

    pool = g["make_world_pool"](args.games, base_seed=args.seed)

    # seat 0: A=owner0, B=owner1  -> A score = score_s0
    sc_s0 = play_batch(g, net_a, net_b, pool, device, greedy=greedy)
    # seat 1: B=owner0, A=owner1  -> A score = 1 - score_s1
    sc_s1 = play_batch(g, net_b, net_a, pool, device, greedy=greedy)
    a_scores = torch.cat([sc_s0, 1.0 - sc_s1])

    w, d, l = _tally(a_scores)
    n = w + d + l
    wr = (w + 0.5 * d) / max(1.0, n)
    z = 1.96
    ci = z * math.sqrt(max(wr * (1 - wr), 1e-9) / max(1.0, n))

    # per-seat sanity (detect a seat bias)
    wr_s0 = float(sc_s0.mean().item())             # A as owner0
    wr_s1 = float((1.0 - sc_s1).mean().item())     # A as owner1

    label = os.path.splitext(os.path.basename(a_ckpt))[0]
    opp = os.path.splitext(os.path.basename(b_ckpt))[0]
    print("A (%s) vs B (%s):" % (label, opp))
    print("  W %d  L %d  D %d   (n=%d)" % (int(w), int(l), int(d), int(n)))
    print("  A win-rate = %.3f  +/- %.3f  (95%% CI)" % (wr, ci))
    print("  per-seat:  A-as-owner0 %.3f | A-as-owner1 %.3f" % (wr_s0, wr_s1))
    verdict = "PASS" if wr >= args.line else "FAIL"
    print("\n  line = %.2f  ->  %s" % (args.line, verdict))

    od = os.path.join("runs", "viz", label)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "h2h_%s_vs_%s.txt" % (label, opp)), "w") as f:
        f.write("A=%s vs B=%s | worlds=%d x2 | comets=%s | %s\n"
                % (a_ckpt, b_ckpt, args.games, g["COMETS_ENABLED"], "greedy" if greedy else "sample"))
        f.write("W %d L %d D %d  wr=%.3f +/-%.3f  (owner0 %.3f / owner1 %.3f)  -> %s\n"
                % (int(w), int(l), int(d), wr, ci, wr_s0, wr_s1, verdict))
    return 0 if wr >= args.line else 1


if __name__ == "__main__":
    raise SystemExit(main())
