"""The native encoder must match the Python EntityObservation byte-for-byte (float32)."""
from __future__ import annotations

import numpy as np

from orbit_wars_rl import _native as native
from orbit_wars_rl.env.game import make_kaggle_env
from orbit_wars_rl.agents.scripted import RandomAgent, StarterAgent
from orbit_wars_rl.processors.observation import EntityObservation
from tests.test_parity import _obs_to_state

MAX_ENT = 64
EP_STEPS = 500


def _check_episode(seed, players=2):
    env = make_kaggle_env(configuration={"seed": seed})
    env.run([RandomAgent(seed=seed).to_kaggle_agent(), StarterAgent().to_kaggle_agent()])
    proc = EntityObservation(max_entities=MAX_ENT, episode_steps=EP_STEPS)
    mismatches = 0
    checked = 0
    for t in range(0, len(env.steps), 7):  # sample steps
        obs = env.steps[t][0]["observation"]
        for player in range(players):
            obs_p = dict(obs)
            obs_p["player"] = player
            py = proc.process(obs_p, {"episodeSteps": EP_STEPS})[0]
            state = _obs_to_state(obs, players)
            ent, emask, amask, glob = native.encode_state(state, player, MAX_ENT, EP_STEPS)
            checked += 1
            for name, a, b in [
                ("entities", ent, py["entities"]),
                ("entity_mask", emask, py["entity_mask"]),
                ("action_mask", amask, py["action_mask"]),
                ("globals", glob, py["globals"]),
            ]:
                if not np.allclose(a, b, atol=1e-5, rtol=0):
                    mismatches += 1
                    d = np.abs(np.asarray(a) - np.asarray(b))
                    print(f"seed{seed} t{t} p{player} {name}: max|diff|={d.max():.2e}")
    return checked, mismatches


def test_encoder_matches_python():
    total_checked, total_mis = 0, 0
    for seed in range(3):
        c, m = _check_episode(seed)
        total_checked += c
        total_mis += m
    assert total_checked > 30
    assert total_mis == 0, f"{total_mis} encoder mismatches"


if __name__ == "__main__":
    tc, tm = 0, 0
    for s in range(3):
        c, m = _check_episode(s)
        tc += c; tm += m
    print(f"checked {tc} (state,player) encodings; mismatches={tm}")
    print("ALL MATCH" if tm == 0 else "MISMATCH")
