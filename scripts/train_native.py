"""Train Orbit Wars PPO with the native C++/LibTorch trainer.

The env, policy, rollout, GAE, and PPO update all run in C++. Python drives a *chunked*
loop: train a chunk of steps, then evaluate the current weights in the fast native arena
(true competition win rate vs the scripted baselines, comet-aware) -- a real learning
curve, not the reward-sign proxy. The best checkpoint by win-rate-vs-starter is saved.

    python scripts/train_native.py --total-steps 8000000 --num-envs 512 \
        --eval-every 400000 --prod-weight 20 --ent-coef 0.02 --out runs/native/run1.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

# Stream progress live even through a pipe (default is block buffering off a TTY).
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from orbit_wars_rl import _native as native
from orbit_wars_rl.agents.ppo_policy import EntityPolicy
from orbit_wars_rl.native_worldgen import generate_pool_cached, generate_world
from orbit_wars_rl.processors.observation import N_ENTITY_FEATURES, N_GLOBAL_FEATURES

FRACTIONS = [0.25, 0.5, 0.75, 1.0]


MAX_ENTITIES = 64


def policy_config(angle_bins, hidden, target_mode=False):
    n_choices = MAX_ENTITIES if target_mode else angle_bins
    return dict(
        n_entity_features=N_ENTITY_FEATURES,
        n_global_features=N_GLOBAL_FEATURES,
        actions_per_entity=1 + n_choices * len(FRACTIONS),
        hidden=hidden,
        target_mode=target_mode,
        num_fracs=len(FRACTIONS),
    )


def save_checkpoint(weights, cfg, path):
    policy = EntityPolicy(**cfg)
    sd = policy.state_dict()
    new_sd = {k: torch.from_numpy(np.array(weights[k], copy=True)).reshape(sd[k].shape) for k in sd}
    policy.load_state_dict(new_sd)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"policy_state": policy.state_dict(), "policy_config": cfg}, path)


def export_checkpoint(trainer, angle_bins, fractions, hidden, path, target_mode=False):
    """Back-compat helper (used by tests): export current trainer weights to a checkpoint."""
    save_checkpoint(trainer.get_weights(), policy_config(angle_bins, hidden, target_mode), path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=8_000_000)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--rollout-steps", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--angle-bins", type=int, default=16)
    ap.add_argument("--target-mode", action="store_true",
                    help="per-planet action picks a target entity (precise aim) instead of an angle bin")
    ap.add_argument("--episode-steps", type=int, default=500)
    ap.add_argument("--pool-size", type=int, default=2000)
    ap.add_argument("--no-comets", action="store_true",
                    help="train on comet-free worlds (instant world-gen); eval stays comet-aware")
    ap.add_argument("--selfplay-start-step", type=int, default=2_000_000)
    ap.add_argument("--snapshot-every", type=int, default=500_000)
    ap.add_argument("--prod-weight", type=float, default=20.0)
    ap.add_argument("--value-warmup-updates", type=int, default=0,
                    help="critic-only PPO updates before the policy moves (stabilizes BC finetune)")
    ap.add_argument("--reward-scale", type=float, default=50.0)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--opp-random", type=float, default=1.0)
    ap.add_argument("--opp-starter", type=float, default=1.0)
    ap.add_argument("--opp-self", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=400_000)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None, help="warm-start policy weights (e.g. a BC checkpoint)")
    ap.add_argument("--out", default="runs/native/run.pt")
    args = ap.parse_args()

    cfg = native.TrainerConfig()
    cfg.num_envs = args.num_envs
    cfg.rollout_steps = args.rollout_steps
    cfg.total_steps = args.total_steps
    cfg.hidden = args.hidden
    cfg.angle_bins = args.angle_bins
    cfg.target_mode = args.target_mode
    cfg.episode_steps = args.episode_steps
    cfg.selfplay_start_step = args.selfplay_start_step
    cfg.snapshot_every = args.snapshot_every
    cfg.prod_weight = args.prod_weight
    cfg.value_warmup_updates = args.value_warmup_updates
    cfg.reward_scale = args.reward_scale
    cfg.ent_coef = args.ent_coef
    cfg.lr = args.lr
    cfg.gamma = args.gamma
    cfg.clip = args.clip
    cfg.opp_random = args.opp_random
    cfg.opp_starter = args.opp_starter
    cfg.opp_self = args.opp_self
    cfg.device = args.device
    cfg.seed = args.seed

    pcfg = policy_config(args.angle_bins, args.hidden, args.target_mode)

    print(f"generating {args.pool_size} training worlds (comets={not args.no_comets}) "
          f"+ {args.eval_games} eval worlds...")
    pool = generate_pool_cached(args.pool_size, base_seed=args.seed, comets=not args.no_comets)
    eval_worlds = generate_pool_cached(args.eval_games, base_seed=900_000)  # eval always comet-aware

    trainer = native.Trainer(cfg)
    trainer.set_world_pool(pool)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu")
        w = {k: np.ascontiguousarray(v.detach().cpu().numpy().astype(np.float32))
             for k, v in ck["policy_state"].items()}
        trainer.load_weights(w)
        print(f"warm-started policy from {args.init_from}")
    arena = native.Arena(64, args.angle_bins, FRACTIONS, args.hidden, args.episode_steps,
                         args.device, args.target_mode)

    print(f"training {args.total_steps} steps | prod_w={args.prod_weight} ent={args.ent_coef} "
          f"lr={args.lr} selfplay@{args.selfplay_start_step} | eval every {args.eval_every}\n")
    print(f"{'step':>10} {'ret':>8} {'vsRAND':>7} {'vsSTART':>8} {'margin':>9} {'len':>5} "
          f"{'sps':>7} {'best':>6}")

    best_start = -1.0
    done = 0
    t_start = time.perf_counter()
    while done < args.total_steps:
        chunk = min(args.eval_every, args.total_steps - done)
        t0 = time.perf_counter()
        trainer.train(chunk)
        done += chunk
        sps = chunk / (time.perf_counter() - t0)

        w = trainer.get_weights()
        arena.load_p0(w)
        r_rand = arena.play(eval_worlds, 0, True, 12345)
        r_start = arena.play(eval_worlds, 1, True, 12345)
        n = args.eval_games
        wr_rand, wr_start = r_rand.p0_wins / n, r_start.p0_wins / n

        save_checkpoint(w, pcfg, args.out.replace(".pt", ".last.pt"))
        flag = ""
        if wr_start > best_start:
            best_start = wr_start
            save_checkpoint(w, pcfg, args.out)
            flag = "*"
        print(f"{done:>10} {trainer.recent_return():>8.2f} {wr_rand:>6.1%} {wr_start:>7.1%} "
              f"{r_start.mean_margin:>+9.1f} {r_start.mean_len:>5.0f} {sps:>7.0f} {flag:>6}")

    dt = time.perf_counter() - t_start
    print(f"\ndone in {dt/60:.1f} min | best vs starter={best_start:.1%} "
          f"-> {args.out} (last -> {args.out.replace('.pt', '.last.pt')})")


if __name__ == "__main__":
    main()
