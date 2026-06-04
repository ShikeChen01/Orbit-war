"""Entity-based observation encoding.

The board has a *variable* number of planets/comets (20-40, plus transient comets),
so we encode each planet as a fixed feature vector, pad to ``max_entities`` rows, and
expose a mask. This is the natural representation for a per-entity (set/graph) policy
and keeps the tensor shape fixed for batching.

Output dict (matches :meth:`build_space`):
    entities     float32 (max_entities, F)  -- per-planet features
    entity_mask  float32 (max_entities,)     -- 1 for real planets, 0 for padding
    action_mask  float32 (max_entities,)     -- 1 where this player may launch
    globals      float32 (G,)                -- whole-board summary

Everything is expressed *relative to the acting player* so the same policy works
from any seat (0-3).
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from orbit_wars_rl.env.game import BOARD_SIZE, CENTER, ROTATION_RADIUS_LIMIT, Fleet, Planet
from orbit_wars_rl.processors.base import ObservationProcessor

# Per-entity feature layout (keep in sync with _entity_features).
ENTITY_FEATURE_NAMES = (
    "is_mine",
    "is_enemy",
    "is_neutral",
    "x_norm",
    "y_norm",
    "dx_center",
    "dy_center",
    "dist_center",
    "radius_norm",
    "ships_log",
    "production_norm",
    "is_comet",
    "is_orbiting",
    "enemy_fleet_pressure",
    "ally_fleet_pressure",
)
N_ENTITY_FEATURES = len(ENTITY_FEATURE_NAMES)

GLOBAL_FEATURE_NAMES = (
    "step_frac",
    "angular_velocity",
    "my_ships_log",
    "enemy_ships_log",
    "my_planet_frac",
    "enemy_planet_frac",
    "neutral_planet_frac",
    "my_fleet_ships_log",
    "enemy_fleet_ships_log",
    "entity_fill_frac",
)
N_GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)

_DIAG = math.hypot(BOARD_SIZE, BOARD_SIZE)
_SHIP_LOG_DENOM = math.log(1000.0)
_PRESSURE_RADIUS = 25.0


def _ship_log(n: float) -> float:
    return math.log1p(max(0.0, n)) / _SHIP_LOG_DENOM


class EntityObservation(ObservationProcessor):
    def __init__(self, max_entities: int = 64, episode_steps: int = 500, num_players: int = 4):
        self.max_entities = max_entities
        self.episode_steps = episode_steps
        self.num_players = num_players

    def build_space(self, config: dict) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "entities": gym.spaces.Box(
                    -np.inf, np.inf, (self.max_entities, N_ENTITY_FEATURES), np.float32
                ),
                "entity_mask": gym.spaces.Box(0, 1, (self.max_entities,), np.float32),
                "action_mask": gym.spaces.Box(0, 1, (self.max_entities,), np.float32),
                "globals": gym.spaces.Box(-np.inf, np.inf, (N_GLOBAL_FEATURES,), np.float32),
            }
        )

    def process(self, obs: dict, config: dict) -> tuple[dict, dict]:
        player = int(obs.get("player", 0))
        planets = [Planet(*row) for row in obs.get("planets", [])]
        fleets = [Fleet(*row) for row in obs.get("fleets", [])]
        episode_steps = int(getattr(config, "episodeSteps", None) or config.get("episodeSteps", self.episode_steps))
        ang_vel = float(obs.get("angular_velocity", 0.0))
        step = int(obs.get("step", 0))

        # Precompute per-planet fleet pressure (enemy / ally ships near the planet).
        enemy_press = [0.0] * len(planets)
        ally_press = [0.0] * len(planets)
        for f in fleets:
            for i, p in enumerate(planets):
                if math.hypot(f.x - p.x, f.y - p.y) <= _PRESSURE_RADIUS:
                    if f.owner == player:
                        ally_press[i] += f.ships
                    else:
                        enemy_press[i] += f.ships

        n = min(len(planets), self.max_entities)
        entities = np.zeros((self.max_entities, N_ENTITY_FEATURES), np.float32)
        entity_mask = np.zeros((self.max_entities,), np.float32)
        action_mask = np.zeros((self.max_entities,), np.float32)
        planet_ids = np.full((self.max_entities,), -1, np.int64)
        planet_ships = np.zeros((self.max_entities,), np.int64)
        planet_x = np.zeros((self.max_entities,), np.float64)
        planet_y = np.zeros((self.max_entities,), np.float64)

        # Stable ordering: own planets first, then by id -- keeps actionable rows dense.
        order = sorted(
            range(len(planets)),
            key=lambda i: (0 if planets[i].owner == player else 1, planets[i].id),
        )[: self.max_entities]
        comet_ids = set(obs.get("comet_planet_ids", []))

        for row, i in enumerate(order):
            p = planets[i]
            entities[row] = self._entity_features(p, player, comet_ids, enemy_press[i], ally_press[i])
            entity_mask[row] = 1.0
            planet_ids[row] = p.id
            planet_ships[row] = p.ships
            planet_x[row] = p.x
            planet_y[row] = p.y
            if p.owner == player and p.ships > 0:
                action_mask[row] = 1.0

        globals_vec = self._global_features(planets, fleets, player, step, episode_steps, ang_vel, n)

        obs_arrays = {
            "entities": entities,
            "entity_mask": entity_mask,
            "action_mask": action_mask,
            "globals": globals_vec,
        }
        context = {
            "planet_ids": planet_ids,
            "planet_ships": planet_ships,
            "planet_x": planet_x,
            "planet_y": planet_y,
            "actionable": action_mask,  # 1 where this player owns the planet and it has ships
            "player": player,
        }
        return obs_arrays, context

    def _entity_features(self, p: Planet, player: int, comet_ids: set, enemy_press: float, ally_press: float) -> np.ndarray:
        dx = (p.x - CENTER) / CENTER
        dy = (p.y - CENTER) / CENTER
        dist_center = math.hypot(p.x - CENTER, p.y - CENTER)
        is_orbiting = 1.0 if (dist_center + p.radius) < ROTATION_RADIUS_LIMIT else 0.0
        return np.array(
            [
                1.0 if p.owner == player else 0.0,
                1.0 if (p.owner != player and p.owner != -1) else 0.0,
                1.0 if p.owner == -1 else 0.0,
                p.x / BOARD_SIZE,
                p.y / BOARD_SIZE,
                dx,
                dy,
                dist_center / (_DIAG / 2),
                p.radius / 3.0,
                _ship_log(p.ships),
                p.production / 5.0,
                1.0 if p.id in comet_ids else 0.0,
                is_orbiting,
                _ship_log(enemy_press),
                _ship_log(ally_press),
            ],
            dtype=np.float32,
        )

    def _global_features(self, planets, fleets, player, step, episode_steps, ang_vel, n_entities) -> np.ndarray:
        my_ships = sum(p.ships for p in planets if p.owner == player)
        enemy_ships = sum(p.ships for p in planets if p.owner != player and p.owner != -1)
        my_planets = sum(1 for p in planets if p.owner == player)
        enemy_planets = sum(1 for p in planets if p.owner != player and p.owner != -1)
        neutral_planets = sum(1 for p in planets if p.owner == -1)
        total_planets = max(1, len(planets))
        my_fleet_ships = sum(f.ships for f in fleets if f.owner == player)
        enemy_fleet_ships = sum(f.ships for f in fleets if f.owner != player)
        return np.array(
            [
                step / max(1, episode_steps),
                ang_vel * 10.0,
                _ship_log(my_ships),
                _ship_log(enemy_ships),
                my_planets / total_planets,
                enemy_planets / total_planets,
                neutral_planets / total_planets,
                _ship_log(my_fleet_ships),
                _ship_log(enemy_fleet_ships),
                n_entities / self.max_entities,
            ],
            dtype=np.float32,
        )
