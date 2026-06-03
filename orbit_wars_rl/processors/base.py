"""Processor abstractions: the swap points for state/action *representation*.

These decouple "how the game looks to the network" and "how network outputs become
game moves" from both the env and the policy. Want to try a different observation
encoding or action parameterization? Implement a new processor and pass it to
``OrbitWarsEnv`` and the policy -- nothing else changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym

Observation = dict
Config = dict
Context = dict  # per-step bookkeeping the action processor needs to decode moves


class ObservationProcessor(ABC):
    """Raw kaggle observation  ->  network-ready arrays (+ decode context)."""

    @abstractmethod
    def build_space(self, config: Config) -> gym.Space:
        ...

    @abstractmethod
    def process(self, obs: Observation, config: Config) -> tuple[dict[str, Any], Context]:
        """Return ``(obs_arrays, context)``.

        ``obs_arrays`` matches :meth:`build_space`. ``context`` carries anything the
        action processor needs to turn a network action back into kaggle moves
        (e.g. the planet id and current ship count for each entity row).
        """
        ...


class ActionProcessor(ABC):
    """Network action  ->  kaggle moves ``[[from_planet_id, angle, num_ships], ...]``."""

    @abstractmethod
    def build_space(self, config: Config) -> gym.Space:
        ...

    @abstractmethod
    def decode(self, action: Any, context: Context) -> list[list]:
        ...

    @property
    @abstractmethod
    def actions_per_entity(self) -> int:
        """Size of the per-entity categorical the policy head must produce."""
        ...
