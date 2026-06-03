"""The agent abstraction.

An :class:`Agent` maps a raw kaggle Orbit Wars observation to a list of moves
``[[from_planet_id, angle, num_ships], ...]``. This is the single interface the
whole system swaps on:

* **Opponents** in the training env are Agents (random, starter, or a frozen policy).
* **Evaluation / the arena** pits any two Agents against each other.
* A trained PPO policy is wrapped as an Agent (see ``agents.ppo_policy.PolicyAgent``).
* ``Agent.to_kaggle_agent()`` produces the ``def agent(obs, config)`` callable that
  kaggle_environments (and the competition submission) expects.

Keeping this dependency-free (no torch) means scripted agents and the env work
without a deep-learning install.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

Move = list  # [from_planet_id: int, angle: float, num_ships: int]
Observation = dict
Config = dict


class Agent(ABC):
    """Base class for anything that can play Orbit Wars."""

    def reset(self) -> None:
        """Called at the start of an episode. Stateless agents can ignore this."""

    @abstractmethod
    def act(self, obs: Observation, config: Config) -> list[Move]:
        """Return a list of moves for the current turn."""
        raise NotImplementedError

    def to_kaggle_agent(self) -> Callable[[Observation, Config], list[Move]]:
        """Adapt to the ``agent(obs, config)`` signature kaggle_environments calls.

        kaggle passes a ``Struct`` (attribute-access) observation; we normalize to a
        plain dict so :meth:`act` always sees a dict.
        """

        def _kaggle_agent(obs, config):
            return self.act(_as_dict(obs), _as_dict(config))

        return _kaggle_agent


class FunctionAgent(Agent):
    """Wrap a plain ``fn(obs, config) -> moves`` callable as an :class:`Agent`."""

    def __init__(self, fn: Callable[[Observation, Config], list[Move]], name: str = "fn"):
        self._fn = fn
        self.name = name

    def act(self, obs: Observation, config: Config) -> list[Move]:
        return self._fn(obs, config)


def _as_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    # kaggle_environments Struct exposes the underlying dict via .__dict__ or items.
    if hasattr(obj, "items"):
        return dict(obj.items())
    return dict(getattr(obj, "__dict__", {}))
