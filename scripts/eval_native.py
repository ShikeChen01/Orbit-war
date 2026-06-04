"""Fast, faithful evaluation of a trained policy in the native C++ engine (with comets).

The native arena steps games with the bit-exact ported engine and scores them exactly
like the competition, so it is a ~50x-faster stand-in for `scripts/play_episode.py`.

    # win rate of a checkpoint vs the starter / random baselines (256 games each)
    python scripts/eval_native.py runs/native/final.pt --opponent starter --games 256
    python scripts/eval_native.py runs/native/final.pt --opponent random  --games 256

    # prove faithfulness: same seeds in the native arena vs the real kaggle engine
    python scripts/eval_native.py runs/native/final.pt --opponent starter --crosscheck 40
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from orbit_wars_rl import _native as native
from orbit_wars_rl.native_worldgen import generate_world
from orbit_wars_rl.processors.observation import N_ENTITY_FEATURES, N_GLOBAL_FEATURES

FRACTIONS = [0.25, 0.5, 0.75, 1.0]
OPP = {"random": 0, "starter": 1, "policy": 2}


def load_weights(path: str) -> tuple[dict, dict]:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["policy_config"]
    w = {k: np.ascontiguousarray(v.detach().cpu().numpy().astype(np.float32))
         for k, v in ckpt["policy_state"].items()}
    return cfg, w


def make_arena(cfg: dict, device: str, episode_steps: int = 500) -> native.Arena:
    target_mode = bool(cfg.get("target_mode", False))
    angle_bins = 16 if target_mode else (cfg["actions_per_entity"] - 1) // len(FRACTIONS)
    return native.Arena(
        max_entities=64, angle_bins=angle_bins, fractions=FRACTIONS,
        hidden=cfg["hidden"], episode_steps=episode_steps, device=device, target_mode=target_mode,
    )


def evaluate(path, opponent, games, seed_base, device, deterministic, p1_path=None):
    cfg, w = load_weights(path)
    arena = make_arena(cfg, device)
    arena.load_p0(w)
    if opponent == "policy":
        p1cfg, p1w = load_weights(p1_path or path)
        arena.load_p1(p1w)
    worlds = [generate_world(seed_base + i) for i in range(games)]
    return arena.play(worlds, OPP[opponent], deterministic, seed_base)


def crosscheck(path, opponent, n, seed_base, device):
    """Play identical seeds in the native arena and the real kaggle engine; compare."""
    from orbit_wars_rl.agents import RandomAgent, StarterAgent
    from orbit_wars_rl.agents.ppo_policy import PolicyAgent
    from orbit_wars_rl.env.game import make_kaggle_env, scores_by_player

    cfg, w = load_weights(path)
    arena = make_arena(cfg, device)
    arena.load_p0(w)

    agree = 0
    rows = []
    for i in range(n):
        seed = seed_base + i
        # native: one world, one game
        r = arena.play([generate_world(seed)], OPP[opponent], True, seed)
        nat = 0 if r.p0_wins else (1 if r.p1_wins else -1)
        nat_margin = r.mean_margin
        # real engine, same seed
        a0 = PolicyAgent.load(path, device="cpu")
        a1 = StarterAgent() if opponent == "starter" else RandomAgent(seed=seed)
        env = make_kaggle_env(configuration={"seed": seed})
        a0.reset(); a1.reset()
        env.run([a0.to_kaggle_agent(), a1.to_kaggle_agent()])
        obs = env.steps[-1][0]["observation"]
        sc = scores_by_player(obs, 2)
        real = 0 if sc[0] > sc[1] else (1 if sc[1] > sc[0] else -1)
        real_margin = sc[0] - sc[1]
        ok = nat == real
        agree += ok
        rows.append((seed, nat, real, nat_margin, real_margin, ok))

    print(f"\ncrosscheck vs {opponent}: {agree}/{n} outcomes agree "
          f"({agree/n:.0%})  [native vs real kaggle engine, same seeds]")
    print(f"{'seed':>6} {'nat':>4} {'real':>5} {'nat_marg':>9} {'real_marg':>10}  ok")
    for seed, nat, real, nm, rm, ok in rows:
        print(f"{seed:>6} {nat:>4} {real:>5} {nm:>9.1f} {rm:>10.0f}  {'' if ok else 'MISMATCH'}")
    return agree, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--opponent", default="starter", choices=list(OPP))
    ap.add_argument("--p1", default=None, help="checkpoint for the policy opponent")
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--seed-base", type=int, default=10_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of greedy")
    ap.add_argument("--crosscheck", type=int, default=0, metavar="N",
                    help="also play N identical seeds in the real engine and compare")
    args = ap.parse_args()

    if args.crosscheck:
        crosscheck(args.checkpoint, args.opponent, args.crosscheck, args.seed_base, args.device)
        return

    import time
    t0 = time.perf_counter()
    r = evaluate(args.checkpoint, args.opponent, args.games, args.seed_base, args.device,
                 not args.stochastic, args.p1)
    dt = time.perf_counter() - t0
    n = r.p0_wins + r.p1_wins + r.draws
    print(f"{args.checkpoint} vs {args.opponent}: {n} games in {dt:.2f}s")
    print(f"  winrate={r.p0_wins/n:.3f}  (p0={r.p0_wins} p1={r.p1_wins} draws={r.draws})  "
          f"mean_margin={r.mean_margin:+.1f}  mean_len={r.mean_len:.0f}")


if __name__ == "__main__":
    main()
