"""Train Orbit Wars PPO with the native C++/LibTorch trainer.

The env, policy, rollout, GAE, and PPO update all run in C++ (one `trainer.train()`
call). Python only generates the world pool and, at the end, copies the learned weights
into a Python EntityPolicy and saves a checkpoint that `PolicyAgent.load` / the arena
can use.

    python scripts/train_native.py --total-steps 2000000 --num-envs 256 --out runs/native/final.pt
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from orbit_wars_rl import _native as native
from orbit_wars_rl.agents.ppo_policy import EntityPolicy
from orbit_wars_rl.native_worldgen import generate_pool
from orbit_wars_rl.processors.observation import N_ENTITY_FEATURES, N_GLOBAL_FEATURES


def export_checkpoint(trainer, angle_bins, fractions, hidden, path):
    """Copy native weights into a Python EntityPolicy and save a PolicyAgent checkpoint."""
    policy_config = dict(
        n_entity_features=N_ENTITY_FEATURES,
        n_global_features=N_GLOBAL_FEATURES,
        actions_per_entity=1 + angle_bins * len(fractions),
        hidden=hidden,
    )
    policy = EntityPolicy(**policy_config)
    sd = policy.state_dict()
    native_w = trainer.get_weights()  # dict[str, np.ndarray] with matching keys
    missing = set(sd) - set(native_w)
    assert not missing, f"native weights missing keys: {missing}"
    new_sd = {k: torch.from_numpy(np.array(native_w[k], copy=True)).reshape(sd[k].shape) for k in sd}
    policy.load_state_dict(new_sd)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"policy_state": policy.state_dict(), "policy_config": policy_config}, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--rollout-steps", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--angle-bins", type=int, default=16)
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--pool-size", type=int, default=2000)
    ap.add_argument("--selfplay-start-step", type=int, default=300_000)
    ap.add_argument("--snapshot-every", type=int, default=300_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/native/final.pt")
    args = ap.parse_args()

    fractions = [0.25, 0.5, 0.75, 1.0]
    cfg = native.TrainerConfig()
    cfg.num_envs = args.num_envs
    cfg.rollout_steps = args.rollout_steps
    cfg.total_steps = args.total_steps
    cfg.hidden = args.hidden
    cfg.angle_bins = args.angle_bins
    cfg.episode_steps = args.episode_steps
    cfg.selfplay_start_step = args.selfplay_start_step
    cfg.snapshot_every = args.snapshot_every
    cfg.device = args.device
    cfg.seed = args.seed

    print(f"generating {args.pool_size} worlds...")
    pool = generate_pool(args.pool_size, base_seed=args.seed)
    trainer = native.Trainer(cfg)
    trainer.set_world_pool(pool)

    print(f"training {args.total_steps} steps on {args.device} ...")
    trainer.train(args.total_steps)
    print(f"final winrate(recent)={trainer.recent_winrate():.3f} return={trainer.recent_return():.3f}")

    path = export_checkpoint(trainer, args.angle_bins, fractions, args.hidden, args.out)
    print(f"saved checkpoint -> {path}")


if __name__ == "__main__":
    main()
