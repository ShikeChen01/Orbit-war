"""Batched greedy win-rate eval of a v5 target-actor .pt checkpoint vs scripted opponents.
Reuses the notebook defs + the loader from viz_target_ckpt.py.

    python scripts/eval_target_ckpt.py --ckpt <ckpt.pt> --opponents starter random --envs 64 --steps 500
"""
from __future__ import annotations
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_target_ckpt import load_notebook_defs, build_actor

OPP = {"random": 0, "starter": 1, "noop": 2}


def eval_vs(g, net, opp, envs, steps, base_seed):
    GpuEnv, env_encode, act, env_step, settle = g["GpuEnv"], g["env_encode"], g["act"], g["env_step"], g["settle"]
    env = GpuEnv(g["PLANET_CAP"], g["FLEET_CAP"], steps, g["SHIP_SPEED"], g["DEVICE"])
    env.reset([g["generate_world"](base_seed + i) for i in range(envs)])
    dev = g["DEVICE"]
    active = torch.ones(envs, device=dev); outcome = torch.zeros(envs, device=dev)
    for t in range(steps):
        ent, em, am, gl = env_encode(env, 0)
        a_t, _, _ = act(net, ent, em, am, gl, greedy=True)
        env_step(env, a_t, OPP[opp], g["ACT_THRESHOLD"], None)
        s0, s1, a0, a1 = settle(env)
        dead = ~(a0 & a1); term = (env.step_ct >= float(env.T - 2)) | dead
        newly = (active > 0.5) & term
        outcome = torch.where(newly, torch.sign(s0 - s1), outcome)
        active = torch.where(term, torch.zeros_like(active), active)
        if active.sum().item() == 0:
            break
    s0, s1, _, _ = settle(env)
    outcome = torch.where(active > 0.5, torch.sign(s0 - s1), outcome)
    win = (outcome > 0).float().mean().item()
    draw = (outcome == 0).float().mean().item()
    loss = (outcome < 0).float().mean().item()
    margin = (s0 - s1).mean().item()
    return win, draw, loss, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--opponents", nargs="+", default=["starter", "random"])
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    g = load_notebook_defs()
    net, cfg, blob = build_actor(g, args.ckpt)
    print("== %s  iter=%s train_win_rate=%s  (HIDDEN=%s blocks=%s attn=%s) ==" % (
        os.path.basename(args.ckpt), blob.get("iter"), blob.get("win_rate"),
        cfg.get("HIDDEN"), cfg.get("N_RES_BLOCKS"), cfg.get("USE_ATTENTION")))
    for opp in args.opponents:
        w, d, l, m = eval_vs(g, net, opp, args.envs, args.steps, args.seed)
        print("  vs %-8s (%d envs): win %.2f  draw %.2f  loss %.2f  | mean ship margin %+.0f"
              % (opp, args.envs, w, d, l, m))


if __name__ == "__main__":
    main()
