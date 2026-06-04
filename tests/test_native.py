"""Native trainer smoke + weight-export/forward parity.

Verifies the C++ PPO trainer runs, exports weights that load into the Python EntityPolicy,
and that the C++ policy's greedy output matches the exported Python policy on real states
(i.e. the weight export is correct). Strength/training-budget is out of scope here.
"""
from __future__ import annotations

import numpy as np
import torch

from orbit_wars_rl import _native as native
from orbit_wars_rl.agents.ppo_policy import EntityPolicy, obs_to_tensors
from orbit_wars_rl.agents.scripted import RandomAgent
from orbit_wars_rl.env.game import make_kaggle_env
from orbit_wars_rl.native_worldgen import generate_pool
from orbit_wars_rl.processors.observation import EntityObservation
from scripts.train_native import export_checkpoint
from tests.test_parity import _obs_to_state


def test_native_train_and_export_parity(tmp_path):
    cfg = native.TrainerConfig()
    cfg.num_envs = 32
    cfg.episode_steps = 120
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = native.Trainer(cfg)
    trainer.set_world_pool(generate_pool(100, 0))
    trainer.train(4000)  # just enough to move the weights off init

    ckpt = str(tmp_path / "n.pt")
    export_checkpoint(trainer, 16, [0.25, 0.5, 0.75, 1.0], 128, ckpt)
    state = torch.load(ckpt, map_location="cpu")
    py = EntityPolicy(**state["policy_config"])
    py.load_state_dict(state["policy_state"])
    py.eval()

    env = make_kaggle_env(configuration={"seed": 11})
    env.run([RandomAgent(seed=11).to_kaggle_agent(), RandomAgent(seed=12).to_kaggle_agent()])
    proc = EntityObservation()
    checked = mismatches = 0
    for k in range(0, len(env.steps), 25):
        obs = env.steps[k][0]["observation"]
        obs0 = dict(obs)
        obs0["player"] = 0
        cpp = np.array(trainer.greedy_classes(_obs_to_state(obs, 2)))
        arr, _ = proc.process(obs0, {"episodeSteps": 120})
        with torch.no_grad():
            logits, _ = py.forward(obs_to_tensors(arr, torch.device("cpu")))
        pyc = logits.argmax(-1).squeeze(0).numpy()
        checked += 1
        mismatches += int(not np.array_equal(cpp, pyc))
    assert checked >= 3
    assert mismatches == 0, "C++ policy vs exported Python policy greedy classes differ"


def test_native_target_mode_parity(tmp_path):
    """Same forward/export parity but with the target-based action head + masking."""
    cfg = native.TrainerConfig()
    cfg.num_envs = 32
    cfg.episode_steps = 120
    cfg.target_mode = True
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = native.Trainer(cfg)
    trainer.set_world_pool(generate_pool(100, 0))
    trainer.train(4000)

    ckpt = str(tmp_path / "t.pt")
    export_checkpoint(trainer, 16, [0.25, 0.5, 0.75, 1.0], 128, ckpt, target_mode=True)
    state = torch.load(ckpt, map_location="cpu")
    assert state["policy_config"]["target_mode"] is True
    py = EntityPolicy(**state["policy_config"])
    py.load_state_dict(state["policy_state"])
    py.eval()

    env = make_kaggle_env(configuration={"seed": 11})
    env.run([RandomAgent(seed=11).to_kaggle_agent(), RandomAgent(seed=12).to_kaggle_agent()])
    proc = EntityObservation()
    checked = mismatches = 0
    for k in range(0, len(env.steps), 25):
        obs = env.steps[k][0]["observation"]
        obs0 = dict(obs)
        obs0["player"] = 0
        cpp = np.array(trainer.greedy_classes(_obs_to_state(obs, 2)))
        arr, _ = proc.process(obs0, {"episodeSteps": 120})
        with torch.no_grad():
            logits, _ = py.forward(obs_to_tensors(arr, torch.device("cpu")))
        pyc = logits.argmax(-1).squeeze(0).numpy()
        checked += 1
        mismatches += int(not np.array_equal(cpp, pyc))
    assert checked >= 3
    assert mismatches == 0, "target-mode C++ vs Python greedy classes differ"
