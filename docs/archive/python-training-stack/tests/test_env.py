"""Sanity tests for the env, processors, and policy. Run: pytest"""
from __future__ import annotations

import numpy as np

from orbit_wars_rl.agents import RandomAgent, StarterAgent
from orbit_wars_rl.env import OrbitWarsEnv
from orbit_wars_rl.processors import EntityObservation, PerPlanetAction


def _random_masked_action(env, obs):
    act = np.zeros(env.obs_processor.max_entities, dtype=np.int64)
    own = obs["action_mask"] > 0.5
    if own.any():
        act[own] = np.random.randint(1, env.act_processor.actions_per_entity, size=int(own.sum()))
    return act


def test_env_reset_and_step():
    env = OrbitWarsEnv(opponents=[RandomAgent(seed=1)], num_players=2, seed=1)
    obs, info = env.reset(seed=1)
    assert set(obs) == {"entities", "entity_mask", "action_mask", "globals"}
    assert obs["entities"].shape == (env.obs_processor.max_entities, 15)
    assert obs["action_mask"].sum() >= 1  # we always start owning a home planet

    obs, reward, term, trunc, info = env.step(_random_masked_action(env, obs))
    assert np.isfinite(reward)
    assert "score_margin" in info


def test_episode_terminates_and_scores():
    env = OrbitWarsEnv(opponents=[StarterAgent()], num_players=2, seed=2)
    obs, _ = env.reset(seed=2)
    term = False
    steps = 0
    while not term and steps < env.episode_steps + 5:
        obs, reward, term, trunc, info = env.step(_random_masked_action(env, obs))
        steps += 1
    assert term
    assert info["winner"] in (-1, 0, 1)


def test_action_processor_only_launches_from_owned():
    obs_proc = EntityObservation(max_entities=32)
    act_proc = PerPlanetAction(max_entities=32)
    env = OrbitWarsEnv(obs_processor=obs_proc, act_processor=act_proc, opponents=[RandomAgent()], seed=3)
    obs, _ = env.reset(seed=3)
    # Force every row to a launch class; decode must still only emit owned planets.
    act = np.full(32, 5, dtype=np.int64)
    moves = act_proc.decode(act, env._context)
    owned_ids = {int(env._context["planet_ids"][i]) for i in range(32) if obs["action_mask"][i] > 0.5}
    for from_id, angle, ships in moves:
        assert from_id in owned_ids
        assert ships >= 1


def test_policy_forward_and_masking():
    import torch

    from orbit_wars_rl.agents.ppo_policy import EntityPolicy, obs_to_tensors
    from orbit_wars_rl.processors.observation import N_ENTITY_FEATURES, N_GLOBAL_FEATURES

    env = OrbitWarsEnv(opponents=[RandomAgent()], seed=4)
    obs, _ = env.reset(seed=4)
    act_proc = env.act_processor
    policy = EntityPolicy(N_ENTITY_FEATURES, N_GLOBAL_FEATURES, act_proc.actions_per_entity, hidden=32)
    tens = obs_to_tensors(obs, torch.device("cpu"))
    action, logp, value = policy.act(tens)
    assert action.shape == (1, env.obs_processor.max_entities)
    # Non-actionable rows must be forced to the no-op class (0).
    non_actionable = obs["action_mask"] < 0.5
    assert (action.squeeze(0).numpy()[non_actionable] == 0).all()
    assert np.isfinite(logp.item()) and np.isfinite(value.item())
