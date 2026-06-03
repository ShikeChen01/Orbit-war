"""Parity: the native C++ step must reproduce the official engine's transitions.

Strategy: play real episodes with the official kaggle engine, capturing each
(state_t, actions_t) -> state_{t+1}. Feed (state_t, actions_t) to the native
`step_from_state` and assert the result equals state_{t+1}.

Comet-spawn transitions (t+1 in {50,150,250,350,450}) inject RNG comets inside the
step, which the native core deliberately does not do (RNG stays in Python / native
worldgen). Those few transitions are skipped — everything else (launch, production,
movement, continuous collision, rotation, comet movement, expiration, combat) is checked.
"""
from __future__ import annotations

import math

import pytest

from orbit_wars_rl import _native as native
from orbit_wars_rl.env.game import make_kaggle_env
from orbit_wars_rl.agents.scripted import RandomAgent, StarterAgent

COMET_SPAWN_STEPS = {50, 150, 250, 350, 450}
OBS_KEYS = [
    "planets", "fleets", "comets", "comet_planet_ids",
    "initial_planets", "angular_velocity", "next_fleet_id", "step",
]


def _to_plain(obj):
    """Recursively convert kaggle Struct/namespace into plain dict/list/scalars."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "items"):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def _obs_to_state(obs, num_agents):
    s = {k: _to_plain(obs[k]) for k in OBS_KEYS}
    s["num_agents"] = num_agents
    return s


def _approx_equal(a, b, tol=1e-6):
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_approx_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict):
        if set(a) != set(b):
            return False
        return all(_approx_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def _diff(native_state, official_state, tol=1e-6):
    """Return a short description of the first field that mismatches, or None."""
    for key in ["planets", "fleets", "comet_planet_ids", "next_fleet_id", "comets"]:
        a, b = native_state.get(key), official_state.get(key)
        if not _approx_equal(a, b, tol):
            return f"{key}: native={a!r} official={b!r}"
    return None


def run_parity(num_episodes=2, base_seed=0, players=2, tol=1e-6):
    """Returns (checked, mismatches: list[str])."""
    checked = 0
    mismatches = []
    for ep in range(num_episodes):
        env = make_kaggle_env(configuration={"seed": base_seed + ep})
        a0 = RandomAgent(seed=base_seed + ep)
        a1 = StarterAgent()
        env.run([a0.to_kaggle_agent(), a1.to_kaggle_agent()])
        steps = env.steps
        for t in range(len(steps) - 1):
            cur, nxt = steps[t], steps[t + 1]
            if cur[0]["status"] != "ACTIVE":
                continue
            obs = cur[0]["observation"]
            step_idx = obs.get("step", t)
            if (step_idx + 1) in COMET_SPAWN_STEPS:
                continue  # RNG spawn; not the native core's job
            state = _obs_to_state(obs, players)
            # In kaggle_environments, state[i].action holds the action that PRODUCED that
            # state, so the t -> t+1 transition uses the action stored on step t+1.
            actions = [_to_plain(nxt[i].get("action") or []) for i in range(players)]
            result = native.step_from_state(state, actions)
            official_next = _obs_to_state(nxt[0]["observation"], players)
            checked += 1
            d = _diff(result, official_next, tol)
            if d is not None:
                mismatches.append(f"ep{ep} step{step_idx}: {d}")
                if len(mismatches) >= 5:
                    return checked, mismatches
    return checked, mismatches


def test_native_step_matches_official():
    checked, mismatches = run_parity(num_episodes=3, base_seed=0)
    assert checked > 100, f"too few transitions checked ({checked})"
    assert not mismatches, "parity mismatches:\n" + "\n".join(mismatches)


if __name__ == "__main__":
    checked, mismatches = run_parity(num_episodes=3, base_seed=0)
    print(f"checked {checked} transitions")
    if mismatches:
        print(f"{len(mismatches)} mismatch(es):")
        for m in mismatches:
            print(" ", m)
    else:
        print("ALL MATCH")
