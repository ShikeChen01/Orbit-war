"""Render a v5 target-actor checkpoint's LOSING games vs an opponent to GIF + keyframe PNG.

Motivation: a checkpoint can show a high *training* win-rate yet lose under greedy eval. This tool
finds the worst losses over a seed range (batched, fast), then replays the top-K single-env and
renders each to GIF so the failure mode is visible (e.g. passivity / never launching / bad aim).

    python scripts/render_losses.py --ckpt notebooks/1400iter-20hidden-MLPs/weights/512x8-MLPs-E750.pt \
        --opponent starter --scan 64 --base-seed 1000 --top 3 --stride 3
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_target_ckpt import load_notebook_defs, build_actor, play  # reuse validated loader + single-game rollout

OPP = {"random": 0, "starter": 1, "noop": 2}


def scan_outcomes(g, net, opp, envs, steps, base_seed):
    """Batched greedy rollout over `envs` seeds -> per-seed (outcome sign, ship margin s0-s1)."""
    GpuEnv, env_encode, act, env_step, settle = g["GpuEnv"], g["env_encode"], g["act"], g["env_step"], g["settle"]
    dev = g["DEVICE"]
    env = GpuEnv(g["PLANET_CAP"], g["FLEET_CAP"], steps, g["SHIP_SPEED"], dev)
    env.reset([g["generate_world"](base_seed + i) for i in range(envs)])
    active = torch.ones(envs, device=dev)
    outcome = torch.zeros(envs, device=dev)
    margin = torch.zeros(envs, device=dev)
    for _ in range(steps):
        ent, em, am, gl = env_encode(env, 0)
        a_t, _, _ = act(net, ent, em, am, gl, greedy=True)
        env_step(env, a_t, OPP[opp], g["ACT_THRESHOLD"], None)
        s0, s1, a0, a1 = settle(env)
        term = (env.step_ct >= float(env.T - 2)) | (~(a0 & a1))
        newly = (active > 0.5) & term
        outcome = torch.where(newly, torch.sign(s0 - s1), outcome)
        margin = torch.where(newly, s0 - s1, margin)
        active = torch.where(term, torch.zeros_like(active), active)
        if active.sum().item() == 0:
            break
    s0, s1, _, _ = settle(env)
    still = active > 0.5
    outcome = torch.where(still, torch.sign(s0 - s1), outcome)
    margin = torch.where(still, s0 - s1, margin)
    return outcome.cpu().tolist(), margin.cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--opponent", default="starter", choices=["starter", "random", "noop"])
    ap.add_argument("--scan", type=int, default=64, help="how many seeds to scan for losses")
    ap.add_argument("--base-seed", type=int, default=1000, help="first seed (matches eval_target_ckpt default)")
    ap.add_argument("--top", type=int, default=3, help="render this many losses")
    ap.add_argument("--pick", default="worst", choices=["worst", "narrow"],
                    help="worst = biggest-margin blowouts; narrow = closest losses (nearly won)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    g = load_notebook_defs()
    net, cfg, blob = build_actor(g, args.ckpt)
    name = os.path.splitext(os.path.basename(args.ckpt))[0]
    print("loaded %s  iter=%s train_win_rate=%s | HIDDEN=%s blocks=%s attn=%s glu=%s"
          % (name, blob.get("iter"), blob.get("win_rate"), cfg.get("HIDDEN"),
             cfg.get("N_RES_BLOCKS"), cfg.get("USE_ATTENTION"), cfg.get("USE_GLU")))

    outcome, margin = scan_outcomes(g, net, args.opponent, args.scan, args.steps, args.base_seed)
    losses = sorted([(args.base_seed + i, margin[i]) for i in range(args.scan) if outcome[i] < 0],
                    key=lambda sm: sm[1], reverse=(args.pick == "narrow"))  # worst=most-negative first; narrow=closest-to-0 first
    n_win = sum(o > 0 for o in outcome); n_draw = sum(o == 0 for o in outcome); n_loss = len(losses)
    print("scan vs %s: %d seeds -> win %d  draw %d  loss %d  | mean margin %+.0f"
          % (args.opponent, args.scan, n_win, n_draw, n_loss, sum(margin) / len(margin)))
    if not losses:
        print("no losses in this seed range -- nothing to render."); return
    print("worst losses (seed, ship margin):", ", ".join("(%d, %+.0f)" % sm for sm in losses[:args.top]))

    outdir = args.outdir or os.path.join("runs", "viz", "losses", "%s_vs_%s" % (name, args.opponent))
    os.makedirs(outdir, exist_ok=True)
    renv = dict(os.environ); renv.pop("SSLKEYLOGFILE", None)
    flips = 0
    for rank, (seed, mg) in enumerate(losses[:args.top], 1):
        ticks, (s0, s1) = play(g, net, args.opponent, args.steps, seed)
        rep = 1.0 if s0 > s1 else (-1.0 if s1 > s0 else 0.0)  # outcome from the SINGLE-ENV replay
        tag = "loss" if rep < 0 else ("WIN" if rep > 0 else "draw")
        if rep >= 0:  # scan said loss, replay disagrees -> batched/single eval non-determinism
            flips += 1
            print("  [!] seed %d scanned as LOSS but single-env replay -> %s (p0=%.0f vs p1=%.0f)" % (seed, tag, s0, s1))
        out = os.path.join(outdir, "%s%d_seed%d_margin%+d" % (tag, rank, seed, int(mg)))
        js = out + ".json"
        json.dump({"board_size": g["BOARD_SIZE"], "sun_radius": g["SUN_RADIUS"],
                   "ego": name, "opp": args.opponent, "outcome": rep, "ticks": ticks}, open(js, "w"))
        print("  [%d] seed %d  %d ticks  p0=%.0f vs p1=%.0f  (%s)  -> %s" % (rank, seed, len(ticks), s0, s1, tag, js))
        subprocess.run([sys.executable, "scripts/render_recording.py", js, "--out", out, "--stride", str(args.stride)],
                       check=True, env=renv)
    if flips:
        print("WARNING: %d/%d rendered games flipped between batched scan and single-env replay "
              "(eval is batch-dependent; trust the single-env label)." % (flips, min(args.top, len(losses))))
    print("-> GIFs + keyframes in", outdir)


if __name__ == "__main__":
    main()
