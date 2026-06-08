# Orbit Wars submission: the scripted GREEDY local-capture bot (notebook opponent==4).
#
# Self-contained, pure-stdlib (no torch, no checkpoint, no package) -> zero timeout risk.
# Rule, per owned planet with ships > 0:
#   * target = NEAREST non-owned planet within CAPTURE_RADIUS that we strictly outnumber,
#   * send n = min(my_ships, floor(target_ships) + CAPTURE_MARGIN) ships (just enough + buffer),
#   * aim at the target's INTERCEPT if it orbits (lead-aim), else straight at it.
# Mirrors `opponent_action(env, 4)` in notebooks/setup1_transformer_ppo.ipynb.
from __future__ import annotations

import math
from collections import namedtuple

# --- env constants (kaggle_environments.envs.orbit_wars) -----------------------
CENTER = 50.0                  # board is 100x100, sun at the center
ROTATION_RADIUS_LIMIT = 50.0   # a planet orbits iff orbital_radius + radius < this
DEFAULT_SHIP_SPEED = 6.0       # config.shipSpeed default; speed scales with fleet size

# --- greedy knobs (match the notebook) -----------------------------------------
CAPTURE_RADIUS = 25.0          # only consider targets within this distance
CAPTURE_MARGIN = 2.0           # ships sent = floor(target_ships) + this (transit buffer)

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])


def _as_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "items"):
        return dict(obj.items())
    return dict(getattr(obj, "__dict__", {}))


def _fleet_speed(ships: float, max_speed: float) -> float:
    """Fleet speed scales with size: 1 ship -> 1/turn, capped at shipSpeed."""
    ships = max(float(ships), 1.0)
    s = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5
    return min(s, max_speed)


def _aim(src: Planet, tgt: Planet, n: int, ang_vel: float, ship_speed: float) -> float:
    """Heading from src to tgt, leading the target's orbit if it rotates."""
    rdx = tgt.x - CENTER
    rdy = tgt.y - CENTER
    r_d = math.hypot(rdx, rdy)                       # target orbital radius
    rotates = (r_d + tgt.radius) < ROTATION_RADIUS_LIMIT
    if not rotates or ang_vel == 0.0:
        return math.atan2(tgt.y - src.y, tgt.x - src.x)
    phi0 = math.atan2(rdy, rdx)                      # target's current orbital angle
    off = tgt.radius + 0.1                           # fly to the target's surface
    v = max(_fleet_speed(n, ship_speed), 1e-6)
    # Fixed-point: estimate flight time, advance the target along its orbit, repeat.
    t = max(math.hypot(tgt.x - src.x, tgt.y - src.y) - off, 0.0) / v
    ix, iy = tgt.x, tgt.y
    for _ in range(16):
        phi = phi0 + ang_vel * t
        ix = CENTER + r_d * math.cos(phi)
        iy = CENTER + r_d * math.sin(phi)
        t = max(math.hypot(ix - src.x, iy - src.y) - off, 0.0) / v
    return math.atan2(iy - src.y, ix - src.x)


def _moves(obs: dict, config: dict) -> list:
    player = obs.get("player", 0)
    planets = [Planet(*row) for row in obs.get("planets", [])]
    ang_vel = float(obs.get("angular_velocity", 0.0) or 0.0)
    ship_speed = float(config.get("shipSpeed", DEFAULT_SHIP_SPEED) or DEFAULT_SHIP_SPEED)

    targets = [p for p in planets if p.owner != player]  # neutral + enemy planets
    moves = []
    for mp in planets:
        if mp.owner != player or mp.ships <= 0:
            continue
        best, best_d = None, float("inf")
        for t in targets:
            if mp.ships <= t.ships:            # only attack planets we strictly outnumber
                continue
            d = math.hypot(mp.x - t.x, mp.y - t.y)
            if d <= CAPTURE_RADIUS and d < best_d:
                best_d, best = d, t
        if best is None:
            continue
        n = int(min(mp.ships, math.floor(best.ships) + CAPTURE_MARGIN))
        if n < 1:
            continue
        angle = _aim(mp, best, n, ang_vel, ship_speed)
        moves.append([mp.id, angle, n])
    return moves


def agent(obs, config):
    # Never raise on the eval box: a crash forfeits the episode; an empty list is a safe noop.
    try:
        return _moves(_as_dict(obs), _as_dict(config))
    except Exception:
        return []
