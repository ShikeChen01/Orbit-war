"""Non-learning agents: useful as opponents, baselines, and sanity checks."""
from __future__ import annotations

import math
import random

from orbit_wars_rl.agents.base import Agent, Config, Move, Observation
from orbit_wars_rl.env.game import CENTER, ROTATION_RADIUS_LIMIT, Planet


class NoopAgent(Agent):
    """Does nothing. The weakest possible opponent / a control baseline."""

    name = "noop"

    def act(self, obs: Observation, config: Config) -> list[Move]:
        return []


class RandomAgent(Agent):
    """Send half the garrison from each owned planet in a random direction.

    Mirrors the env's built-in ``random`` agent but with its own RNG so episodes
    are reproducible when seeded.
    """

    name = "random"

    def __init__(self, seed: int | None = None, min_ships: int = 20):
        self._rng = random.Random(seed)
        self.min_ships = min_ships

    def reset(self) -> None:
        pass

    def act(self, obs: Observation, config: Config) -> list[Move]:
        player = obs.get("player", 0)
        moves: list[Move] = []
        for p in (Planet(*row) for row in obs.get("planets", [])):
            if p.owner == player and p.ships > 0:
                ships = p.ships // 2
                if ships >= self.min_ships:
                    moves.append([p.id, self._rng.uniform(0, 2 * math.pi), ships])
        return moves


class StarterAgent(Agent):
    """Greedy: each owned planet sends half its ships at the nearest static,
    non-owned planet. Mirrors the env's built-in ``starter`` agent.
    """

    name = "starter"

    def __init__(self, min_ships: int = 20):
        self.min_ships = min_ships

    def act(self, obs: Observation, config: Config) -> list[Move]:
        player = obs.get("player", 0)
        planets = [Planet(*row) for row in obs.get("planets", [])]

        static_targets = [
            p
            for p in planets
            if p.owner != player
            and math.hypot(p.x - CENTER, p.y - CENTER) + p.radius >= ROTATION_RADIUS_LIMIT
        ]

        moves: list[Move] = []
        for mp in planets:
            if mp.owner != player or mp.ships <= 0:
                continue
            closest, best = None, float("inf")
            for t in static_targets:
                d = math.hypot(mp.x - t.x, mp.y - t.y)
                if d < best:
                    best, closest = d, t
            if closest is not None:
                ships = mp.ships // 2
                if ships >= self.min_ships:
                    angle = math.atan2(closest.y - mp.y, closest.x - mp.x)
                    moves.append([mp.id, angle, ships])
        return moves
