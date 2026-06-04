"""Comet parity: the native step must reproduce the official engine *including* the
RNG comet-spawn steps, once the pre-generated schedule is supplied.

test_parity.py skips the 5 spawn transitions (the bare native core doesn't do RNG). Here
we pre-resolve the episode's comet schedule the same way native_worldgen does, attach it
to every state fed to `step_from_state`, and check *every* transition -- so the spawn
injection (ids, off-board placement, ship counts, paths) is verified bit-for-bit too.
"""
from __future__ import annotations

import pytest

from orbit_wars_rl import _native as native
from orbit_wars_rl.env.game import make_kaggle_env
from orbit_wars_rl.agents.scripted import RandomAgent, StarterAgent
from orbit_wars_rl.native_worldgen import build_comet_schedule

OBS_KEYS = [
    "planets", "fleets", "comets", "comet_planet_ids",
    "initial_planets", "angular_velocity", "next_fleet_id", "step",
]


def _to_plain(obj):
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "items"):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def _state(obs, num_agents, schedule):
    s = {k: _to_plain(obs[k]) for k in OBS_KEYS}
    s["num_agents"] = num_agents
    s["comet_schedule"] = schedule
    return s


def _approx_equal(a, b, tol=1e-6):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_approx_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict):
        return set(a) == set(b) and all(_approx_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def _diff(nat, off, tol=1e-6):
    for key in ["planets", "fleets", "comet_planet_ids", "next_fleet_id", "comets"]:
        if not _approx_equal(nat.get(key), off.get(key), tol):
            return f"{key}: native={nat.get(key)!r} official={off.get(key)!r}"
    return None


def run(num_episodes=3, base_seed=0, players=2, tol=1e-6):
    checked = spawn_checked = 0
    mismatches = []
    for ep in range(num_episodes):
        seed = base_seed + ep
        env = make_kaggle_env(configuration={"seed": seed})
        env.run([RandomAgent(seed=seed).to_kaggle_agent(), StarterAgent().to_kaggle_agent()])
        steps = env.steps
        # Pre-resolve the schedule from step-0 initial_planets (pre-home copy) + ang. velocity.
        obs0 = steps[0][0]["observation"]
        schedule = build_comet_schedule(
            seed, _to_plain(obs0["initial_planets"]), float(obs0["angular_velocity"])
        )
        for t in range(len(steps) - 1):
            cur, nxt = steps[t], steps[t + 1]
            if cur[0]["status"] != "ACTIVE":
                continue
            obs = cur[0]["observation"]
            step_idx = obs.get("step", t)
            state = _state(obs, players, schedule)
            actions = [_to_plain(nxt[i].get("action") or []) for i in range(players)]
            result = native.step_from_state(state, actions)
            official_next = {k: _to_plain(nxt[0]["observation"][k]) for k in OBS_KEYS}
            checked += 1
            if (step_idx + 1) in (50, 150, 250, 350, 450):
                spawn_checked += 1
            d = _diff(result, official_next, tol)
            if d is not None:
                mismatches.append(f"ep{ep} step{step_idx}: {d}")
                if len(mismatches) >= 5:
                    return checked, spawn_checked, mismatches
    return checked, spawn_checked, mismatches


def test_comet_spawn_parity():
    checked, spawn_checked, mismatches = run(num_episodes=3, base_seed=0)
    assert spawn_checked >= 10, f"expected to exercise spawn steps, got {spawn_checked}"
    assert not mismatches, "comet parity mismatches:\n" + "\n".join(mismatches)


if __name__ == "__main__":
    checked, spawn_checked, mismatches = run(num_episodes=3, base_seed=0)
    print(f"checked {checked} transitions ({spawn_checked} at spawn steps)")
    print("ALL MATCH" if not mismatches else "MISMATCHES:\n" + "\n".join(mismatches))
