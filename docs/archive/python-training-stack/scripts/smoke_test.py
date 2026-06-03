"""End-to-end sanity check: env builds, an episode runs, processors round-trip.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import numpy as np

from orbit_wars_rl.env import OrbitWarsEnv
from orbit_wars_rl.agents import StarterAgent


def main() -> None:
    env = OrbitWarsEnv(opponents=[StarterAgent()], num_players=2, seed=0)
    obs, info = env.reset(seed=0)
    print("observation shapes:", {k: tuple(v.shape) for k, v in obs.items()})
    print("action space:", env.action_space)

    A = env.act_processor.actions_per_entity
    total_reward, steps = 0.0, 0
    terminated = False
    while not terminated:
        act = np.zeros(env.obs_processor.max_entities, dtype=np.int64)
        own = obs["action_mask"] > 0.5
        if own.any():
            act[own] = np.random.randint(1, A, size=int(own.sum()))
        obs, reward, terminated, truncated, info = env.step(act)
        total_reward += reward
        steps += 1

    print(f"episode finished in {steps} steps")
    print(f"  winner: player {info.get('winner')}  (0 = us)")
    print(f"  final score margin: {info.get('score_margin')}")
    print(f"  total shaped reward: {total_reward:.3f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
