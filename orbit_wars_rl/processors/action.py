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
        target_mode: bool = False,
    ):
        self.max_entities = max_entities
        self.angle_bins = angle_bins
        self.fractions = tuple(fractions)
        self.min_ships = min_ships
        self.target_mode = target_mode
        # Precompute the (angle, fraction) for each launch class index (1-based).
        self._launch_table = [
            (2.0 * math.pi * k / angle_bins, frac)
            for frac in self.fractions
            for k in range(angle_bins)
        ]

    @property
    def actions_per_entity(self) -> int:
        if self.target_mode:
            return 1 + self.max_entities * len(self.fractions)
        return 1 + self.angle_bins * len(self.fractions)

    def build_space(self, config: dict) -> gym.spaces.MultiDiscrete:
        return gym.spaces.MultiDiscrete([self.actions_per_entity] * self.max_entities)

    def decode(self, action, context: dict) -> list[list]:
        """action: int array (max_entities,) of per-entity class indices."""
        if self.target_mode:
            return self._decode_target(action, context)
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

    def _decode_target(self, action, context: dict) -> list[list]:
        """Target-based decode (mirrors C++ decode_action_target): class-1 splits into
        (target_row, frac); the angle is computed to aim at that target's position."""
        action = np.asarray(action).reshape(-1)
        planet_ids = context["planet_ids"]
        planet_ships = context["planet_ships"]
        px, py = context["planet_x"], context["planet_y"]
        actionable = context.get("actionable")
        nf = len(self.fractions)
        moves: list[list] = []
        for row, cls in enumerate(action):
            cls = int(cls)
            if cls <= 0:
                continue
            if actionable is not None and actionable[row] < 0.5:
                continue
            pid = int(planet_ids[row])
            if pid < 0:
                continue
            avail = int(planet_ships[row])
            if avail <= 0:
                continue
            launch = cls - 1
            target_row = launch // nf
            fi = launch % nf
            if target_row == row or target_row >= self.max_entities:
                continue
            if int(planet_ids[target_row]) < 0:
                continue  # padding / invalid target
            angle = math.atan2(py[target_row] - py[row], px[target_row] - px[row])
            ships = int(self.fractions[fi] * avail)
            ships = max(self.min_ships, min(ships, avail))
            if ships > 0:
                moves.append([pid, float(angle), ships])
        return moves
