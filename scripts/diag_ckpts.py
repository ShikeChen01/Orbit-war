"""Diagnostic: for every weight in the E750 run dir, print metadata and eval vs starter WITH
launch instrumentation -- so we can tell 'never launches' (passive) apart from 'launches but loses'.

    python scripts/diag_ckpts.py
"""
from __future__ import annotations
import os, sys, glob
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_target_ckpt import load_notebook_defs, build_actor

OPP = {"random": 0, "starter": 1, "noop": 2}


def eval_vs(g, net, opp, envs, steps, base_seed):
    GpuEnv, env_encode, act, env_step, settle = g["GpuEnv"], g["env_encode"], g["act"], g["env_step"], g["settle"]
    dev = g["DEVICE"]
    env = GpuEnv(g["PLANET_CAP"], g["FLEET_CAP"], steps, g["SHIP_SPEED"], dev)
    env.reset([g["generate_world"](base_seed + i) for i in range(envs)])
    active = torch.ones(envs, device=dev)
    outcome = torch.zeros(envs, device=dev); margin = torch.zeros(envs, device=dev)
    lnch = torch.zeros(envs, device=dev); valid = torch.zeros(envs, device=dev); steps_live = torch.zeros(envs, device=dev)
    for _ in range(steps):
        ent, em, am, gl = env_encode(env, 0)
        a_t, _, _ = act(net, ent, em, am, gl, greedy=True)
        out = env_step(env, a_t, OPP[opp], g["ACT_THRESHOLD"], None)
        lnch += active * out.launches; valid += active * out.valid; steps_live += active
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
    win = (outcome > 0).float().mean().item(); loss = (outcome < 0).float().mean().item()
    lps = (lnch.sum() / steps_live.sum().clamp_min(1)).item()
    vps = (valid.sum() / steps_live.sum().clamp_min(1)).item()
    return win, loss, margin.mean().item(), lps, vps


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    g = load_notebook_defs()
    paths = sorted(glob.glob("notebooks/1400iter-20hidden-MLPs/weights/*.pt"))
    print("ACT_THRESHOLD =", g["ACT_THRESHOLD"], "| greedy eval, 32 envs vs starter, base_seed 1000\n")
    for p in paths:
        try:
            net, cfg, blob = build_actor(g, p)
        except Exception as e:
            print("%-30s  FAILED to load: %s" % (os.path.basename(p), repr(e))); continue
        w, l, m, lps, vps = eval_vs(g, net, "starter", 32, 500, 1000)
        print("%-30s iter=%-5s wr_meta=%-4s blk=%s attn=%s glu=%s | vs starter win %.2f loss %.2f margin %+.0f | launch/step %.3f valid/step %.3f"
              % (os.path.basename(p), blob.get("iter"), blob.get("win_rate"),
                 cfg.get("N_RES_BLOCKS"), cfg.get("USE_ATTENTION"), cfg.get("USE_GLU"),
                 w, l, m, lps, vps))


if __name__ == "__main__":
    main()
