"""Generate Orbit Wars worlds (with comet schedules) for the native trainer/arena.

Reuses the OFFICIAL map generator (`generate_planets`) and home-assignment logic so the
maps the native code sees are drawn from the exact same distribution as the real
competition env. The comet *schedule* is pre-generated here too: comets spawn at fixed
steps from a seed-derived RNG, so the whole episode's comets are reproducible and can be
injected deterministically inside the (RNG-free) C++ step. Each world is a plain dict the
C++ side consumes via `world_from_dict`.

Because the comet RNG is seeded `f"orbit_wars-comet-{seed}-{spawn_step}"` exactly like the
real interpreter, a native game built from seed S has the *identical* comets the real
kaggle env produces for `configuration={"seed": S}` -- which is what makes the native
arena a faithful, fast proxy for the competition.
"""
from __future__ import annotations

import os
import pickle
import random

from orbit_wars_rl import _bootstrap  # noqa: F401
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    COMET_SPAWN_STEPS,
    generate_comet_paths,
    generate_planets,
)

COMET_SPEED = 4.0


def build_comet_schedule(seed: int, initial_planets: list, angular_velocity: float) -> list[dict]:
    """Pre-resolve every comet spawn for an episode (mirrors interpreter lines 431-474).

    For each scheduled spawn step we derive the same per-spawn RNG the engine uses, call
    the official `generate_comet_paths`, and (only if it succeeds) draw the comet ship
    count in the same order. Comets never coexist across spawn steps (path length <= 40,
    spawns are 100 apart), so the running `comet_planet_ids` is always empty here.
    """
    schedule = []
    for spawn_step in COMET_SPAWN_STEPS:
        comet_rng = random.Random(f"orbit_wars-comet-{seed}-{spawn_step}")
        paths = generate_comet_paths(
            initial_planets, angular_velocity, spawn_step, set(), COMET_SPEED, rng=comet_rng
        )
        if paths:
            ships = min(
                comet_rng.randint(1, 99),
                comet_rng.randint(1, 99),
                comet_rng.randint(1, 99),
                comet_rng.randint(1, 99),
            )
            schedule.append({"spawn_step": spawn_step, "paths": paths, "ships": ships})
    return schedule


def generate_world(seed: int, comets: bool = True) -> dict:
    rng = random.Random(seed)
    angular_velocity = rng.uniform(0.025, 0.05)
    planets = generate_planets(rng)
    pre_home = [p[:] for p in planets]  # == obs0.initial_planets (copied before home assignment)
    num_groups = len(planets) // 4
    if num_groups > 0:
        base = rng.randint(0, num_groups - 1) * 4
        planets[base][1] = 0       # player 0 home (Q1)
        planets[base][5] = 10
        planets[base + 3][1] = 1   # player 1 home (Q4)
        planets[base + 3][5] = 10
    schedule = build_comet_schedule(seed, pre_home, angular_velocity) if comets else []
    return {"planets": planets, "angular_velocity": angular_velocity, "comet_schedule": schedule}


def generate_pool(n: int, base_seed: int = 0, comets: bool = True) -> list[dict]:
    return [generate_world(base_seed + i, comets=comets) for i in range(n)]


def generate_pool_cached(n: int, base_seed: int = 0, comets: bool = True,
                         cache_dir: str = "runs/_worldcache") -> list[dict]:
    """Same as generate_pool but memoized to disk. Comet-path generation is the slow part
    of every run's startup (~minute for a few thousand worlds); caching makes re-runs and
    sweeps start instantly. Cache key = (n, base_seed, comets)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"pool_n{n}_s{base_seed}_c{int(comets)}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    pool = generate_pool(n, base_seed=base_seed, comets=comets)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return pool
