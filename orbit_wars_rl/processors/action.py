"""Per-planet action parameterization.

Orbit Wars actions are a *variable-length* list of ``[from_planet_id, angle, num_ships]``.
We make this learnable by giving every entity row a single categorical decision:

    class 0                      -> no-op (don't launch from this planet)
    class 1 .. angle_bins*fracs  -> launch: (angle bin, ship-fraction bin)

The policy emits one categorical per entity (masked to owned planets with ships).
The full turn's action is the concatenation of all per-planet decisions, so its
log-prob is the sum of the per-entity log-probs. Clean to train, and the number of
classes is fixed regardless of how many planets are on the board.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from orbit_wars_rl.processors.base import ActionProcessor

DEFAULT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


class PerPlanetAction(ActionProcessor):
    def __init__(
        self,
        max_entities: int = 64,
        angle_bins: int = 16,
        fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
        min_ships: int = 1,
    ):
        self.max_entities = max_entities
        self.angle_bins = angle_bins
        self.fractions = tuple(fractions)
        self.min_ships = min_ships
        # Precompute the (angle, fraction) for each launch class index (1-based).
        self._launch_table = [
            (2.0 * math.pi * k / angle_bins, frac)
            for frac in self.fractions
            for k in range(angle_bins)
        ]

    @property
    def actions_per_entity(self) -> int:
        return 1 + self.angle_bins * len(self.fractions)

    def build_space(self, config: dict) -> gym.spaces.MultiDiscrete:
        return gym.spaces.MultiDiscrete([self.actions_per_entity] * self.max_entities)

    def decode(self, action, context: dict) -> list[list]:
        """action: int array (max_entities,) of per-entity class indices."""
        action = np.asarray(action).reshape(-1)
        planet_ids = context["planet_ids"]
        planet_ships = context["planet_ships"]
        actionable = context.get("actionable")  # only launch from owned planets w/ ships
        moves: list[list] = []
        for row, cls in enumerate(action):
            cls = int(cls)
            if cls <= 0:
                continue
            if actionable is not None and actionable[row] < 0.5:
                continue  # padding, enemy, or neutral planet -- never launch from it
            pid = int(planet_ids[row])
            if pid < 0:
                continue  # padding row
            ships_available = int(planet_ships[row])
            if ships_available <= 0:
                continue
            angle, frac = self._launch_table[cls - 1]
            ships = int(frac * ships_available)
            ships = max(self.min_ships, min(ships, ships_available))
            if ships > 0:
                moves.append([pid, float(angle), ships])
        return moves
