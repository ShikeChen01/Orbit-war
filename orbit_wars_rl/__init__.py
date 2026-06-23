"""Orbit Wars RL: environment, agent abstraction, processors, and the native trainer.

Public entry points:
    from orbit_wars_rl.env import make_kaggle_env          # official game engine wrapper
    from orbit_wars_rl.agents import Agent, RandomAgent, StarterAgent
    from orbit_wars_rl.agents.ppo_policy import PolicyAgent  # serve a trained policy
    from orbit_wars_rl.processors import EntityObservation, PerPlanetAction
    from orbit_wars_rl import _native                       # compiled C++/LibTorch core

The native C++/LibTorch PPO trainer is the primary training path (see scripts/train_native.py
and docs/NATIVE_CPP.md). The original pure-Python gym env + PPO loop is archived under
docs/archive/python-training-stack/.
"""
from orbit_wars_rl import _bootstrap  # noqa: F401  (side-effect: SSL/env fixups)

__all__ = ["_bootstrap"]
