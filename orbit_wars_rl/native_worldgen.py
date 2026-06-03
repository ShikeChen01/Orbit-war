"""Generate Orbit Wars worlds for the native trainer.

Reuses the OFFICIAL map generator (`generate_planets`) and home-assignment logic so the
maps the native trainer sees are drawn from the exact same distribution as the real
competition env. Comets are omitted (the native trainer trains on the no-comet variant
for now). Each world is a plain dict the C++ Trainer consumes via `world_from_dict`.
"""
from __future__ import annotations

import random

from orbit_wars_rl import _bootstrap  # noqa: F401
from kaggle_environments.envs.orbit_wars.orbit_wars import generate_planets


def generate_world(seed: int) -> dict:
    rng = random.Random(seed)
    angular_velocity = rng.uniform(0.025, 0.05)
    planets = generate_planets(rng)
    num_groups = len(planets) // 4
    if num_groups > 0:
        base = rng.randint(0, num_groups - 1) * 4
        planets[base][1] = 0       # player 0 home (Q1)
        planets[base][5] = 10
        planets[base + 3][1] = 1   # player 1 home (Q4)
        planets[base + 3][5] = 10
    return {"planets": planets, "angular_velocity": angular_velocity}


def generate_pool(n: int, base_seed: int = 0) -> list[dict]:
    return [generate_world(base_seed + i) for i in range(n)]
