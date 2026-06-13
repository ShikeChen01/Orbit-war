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
    "incoming_enemy_ships",      # projected enemy fleet ships that will reach this planet
    "incoming_enemy_imminence",  # exp(-eta/tau): ->1 if the attack lands soon, 0 if none
    "incoming_ally_ships",       # projected reinforcements inbound
    "incoming_ally_imminence",
    "hold_margin",               # tanh((ships + prod*eta + help - attack)/scale): >0 hold, <0 fall
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
_SUN_RADIUS = 10.0            # mirrors native SUN_RADIUS
_THREAT_MAX_SPEED = 6.0       # mirrors Config.shipSpeed / THREAT_MAX_SPEED
_THREAT_ETA_TAU = 8.0         # imminence = exp(-eta/tau)
_THREAT_MARGIN_SCALE = 100.0  # hold-margin tanh divisor


def _ship_log(n: float) -> float:
    return math.log1p(max(0.0, n)) / _SHIP_LOG_DENOM


def _fleet_speed(ships: float) -> float:
    """Larger fleets move faster, capped -- mirrors sim fleet movement / native fleet_speed."""
    n = max(1.0, float(ships))
    v = 1.0 + (_THREAT_MAX_SPEED - 1.0) * (math.log(n) / math.log(1000.0)) ** 1.5
    return min(v, _THREAT_MAX_SPEED)


def _fleet_target(f, planets):
    """Closed-form straight-line projection: (planet_index, eta_ticks) the fleet first reaches,
    or (-1, 0.0) if it crosses the sun or hits nothing. Identical to native fleet_target() so the
    observation stays parity-exact. Threat features only -- never game mechanics."""
    dx = math.cos(f.angle)
    dy = math.sin(f.angle)
    best = -1
    best_t = 1e18
    for i, p in enumerate(planets):
        ox = f.x - p.x
        oy = f.y - p.y
        tca = -(ox * dx + oy * dy)
        if tca < 0.0:
            continue
        perp2 = ox * ox + oy * oy - tca * tca
        rr = p.radius * p.radius
        if perp2 > rr:
            continue
        t = tca - math.sqrt(rr - perp2)
        if t < 0.0:
            t = 0.0
        if t < best_t:
            best_t = t
            best = i
    if best < 0:
        return -1, 0.0
    sx = f.x - CENTER
    sy = f.y - CENTER
    tcs = -(sx * dx + sy * dy)
    if tcs >= 0.0:
        perp2 = sx * sx + sy * sy - tcs * tcs
        sr = _SUN_RADIUS * _SUN_RADIUS
        if perp2 <= sr:
            t_sun = tcs - math.sqrt(sr - perp2)
            if 0.0 <= t_sun < best_t:
                return -1, 0.0
    return best, best_t / _fleet_speed(f.ships)


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

        # Per-planet fleet pressure (raw proximity) + projected incoming threat (which planet a
        # fleet will actually reach, and when). Threat is the defensive signal: a fleet aimed
        # away is harmless; one aimed in from afar is the danger. _fleet_target filters misses.
        n_pl = len(planets)
        enemy_press = [0.0] * n_pl
        ally_press = [0.0] * n_pl
        inc_enemy = [0.0] * n_pl
        inc_ally = [0.0] * n_pl
        eta_enemy = [1e18] * n_pl
        eta_ally = [1e18] * n_pl
        for f in fleets:
            for i, p in enumerate(planets):
                if math.hypot(f.x - p.x, f.y - p.y) <= _PRESSURE_RADIUS:
                    if f.owner == player:
                        ally_press[i] += f.ships
                    else:
                        enemy_press[i] += f.ships
            ti, eta = _fleet_target(f, planets)
            if ti < 0:
                continue
            if f.owner == player:
                inc_ally[ti] += float(f.ships)
                if eta < eta_ally[ti]:
                    eta_ally[ti] = eta
            else:
                inc_enemy[ti] += float(f.ships)
                if eta < eta_enemy[ti]:
                    eta_enemy[ti] = eta

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
            entities[row] = self._entity_features(
                p, player, comet_ids, enemy_press[i], ally_press[i],
                inc_enemy[i], eta_enemy[i], inc_ally[i], eta_ally[i],
            )
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

    def _entity_features(self, p: Planet, player: int, comet_ids: set, enemy_press: float,
                         ally_press: float, inc_enemy: float, eta_enemy: float,
                         inc_ally: float, eta_ally: float) -> np.ndarray:
        dx = (p.x - CENTER) / CENTER
        dy = (p.y - CENTER) / CENTER
        dist_center = math.hypot(p.x - CENTER, p.y - CENTER)
        is_orbiting = 1.0 if (dist_center + p.radius) < ROTATION_RADIUS_LIMIT else 0.0
        enemy_imm = math.exp(-eta_enemy / _THREAT_ETA_TAU) if inc_enemy > 0.0 else 0.0
        ally_imm = math.exp(-eta_ally / _THREAT_ETA_TAU) if inc_ally > 0.0 else 0.0
        margin = (p.ships + p.production * eta_enemy + inc_ally - inc_enemy) if inc_enemy > 0.0 else 0.0
        hold = math.tanh(margin / _THREAT_MARGIN_SCALE) if inc_enemy > 0.0 else 0.0
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
                _ship_log(inc_enemy),
                enemy_imm,
                _ship_log(inc_ally),
                ally_imm,
                hold,
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
