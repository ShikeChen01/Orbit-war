"""Pure-Python Orbit Wars forward model (no numpy/torch) for inference-time search.

A 1:1 port of the native `ow::step` (native/core/sim.hpp), which itself is a 1:1 port of
the official interpreter's deterministic mechanics. The native `.pyd` isn't available on
the Kaggle eval box, so a search-based submission needs a self-contained step it can call
to roll candidate moves forward. Comet *spawning* is intentionally omitted (agents can't
see the schedule); existing comets on the board move along their known paths.

State is the kaggle obs dict: planets [[id,owner,x,y,radius,ships,production],...],
fleets [[id,owner,x,y,angle,from_id,ships],...], comets [{planet_ids,paths,path_index}],
comet_planet_ids, initial_planets, angular_velocity, next_fleet_id, step.
Actions: [player0_moves, player1_moves], each a list of [from_id, angle, num_ships].

Parity-checked against native.step_from_state in tests/test_py_engine_parity.py.
"""
from __future__ import annotations

import copy
import math

BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0


def _point_to_segment_distance(px, py, vx, vy, wx, wy):
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    if l2 == 0.0:
        return math.hypot(px - vx, py - vy)
    t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (vx + t * (wx - vx)), py - (vy + t * (wy - vy)))


def _swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, r):
    d0x, d0y = ax - p0x, ay - p0y
    dvx, dvy = (bx - ax) - (p1x - p0x), (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def _remove_planets(state, ids):
    state["planets"] = [p for p in state["planets"] if p[0] not in ids]
    state["initial_planets"] = [p for p in state["initial_planets"] if p[0] not in ids]
    state["comet_planet_ids"] = [pid for pid in state["comet_planet_ids"] if pid not in ids]
    for g in state["comets"]:
        g["planet_ids"] = [pid for pid in g["planet_ids"] if pid not in ids]
    state["comets"] = [g for g in state["comets"] if g["planet_ids"]]


def step(state: dict, actions: list, ship_speed: float = 6.0, comet_speed: float = 4.0) -> dict:
    """Advance one tick (no comet spawn). Returns a NEW state dict; does not mutate input."""
    s = copy.deepcopy(state)
    planets, fleets = s["planets"], s["fleets"]
    step_idx = s.get("step", 0)
    by_id = {p[0]: p for p in planets}

    # --- Comet expiration before launch ---
    expired = set()
    for g in s["comets"]:
        idx = g["path_index"]
        for i, pid in enumerate(g["planet_ids"]):
            if idx >= len(g["paths"][i]):
                expired.add(pid)
    if expired:
        _remove_planets(s, expired)
        planets, fleets = s["planets"], s["fleets"]
        by_id = {p[0]: p for p in planets}

    # --- 0. Fleet launch ---
    num_agents = s.get("num_agents", len(actions))
    for pid in range(min(num_agents, len(actions))):
        for mv in actions[pid] or []:
            if len(mv) != 3:
                continue
            from_id, angle, ships = mv[0], mv[1], int(mv[2])
            frm = by_id.get(from_id)
            if frm is not None and frm[1] == pid and frm[5] >= ships and ships > 0:
                frm[5] -= ships
                sx = frm[2] + math.cos(angle) * (frm[4] + 0.1)
                sy = frm[3] + math.sin(angle) * (frm[4] + 0.1)
                fleets.append([s["next_fleet_id"], pid, sx, sy, angle, from_id, ships])
                s["next_fleet_id"] += 1

    # --- 1. Production ---
    for p in planets:
        if p[1] != -1:
            p[5] += p[6]

    # --- 2. Planet end-of-tick positions ---
    comet_pid_set = set(s["comet_planet_ids"])
    initial_by_id = {p[0]: p for p in s["initial_planets"]}
    planet_paths = {}  # id -> (ox, oy, nx, ny, check)
    for p in planets:
        if p[0] in comet_pid_set:
            continue
        nx, ny = p[2], p[3]
        ip = initial_by_id.get(p[0])
        if ip is not None:
            dx, dy = ip[2] - CENTER, ip[3] - CENTER
            r = math.hypot(dx, dy)
            if r + p[4] < ROTATION_RADIUS_LIMIT:
                ang = math.atan2(dy, dx) + s["angular_velocity"] * step_idx
                nx = CENTER + r * math.cos(ang)
                ny = CENTER + r * math.sin(ang)
        planet_paths[p[0]] = (p[2], p[3], nx, ny, True)

    expired_after_move = []
    for g in s["comets"]:
        g["path_index"] += 1
        idx = g["path_index"]
        for i, pid in enumerate(g["planet_ids"]):
            p = by_id.get(pid)
            if p is None:
                continue
            ox, oy = p[2], p[3]
            if idx >= len(g["paths"][i]):
                expired_after_move.append(pid)
                planet_paths[pid] = (ox, oy, ox, oy, True)
            else:
                nxp, nyp = g["paths"][i][idx]
                planet_paths[pid] = (ox, oy, nxp, nyp, ox >= 0)

    # --- 3. Fleet movement + continuous collision ---
    fleets_to_remove = set()
    combat_lists = {p[0]: [] for p in planets}
    for f in fleets:
        speed = 1.0 + (ship_speed - 1.0) * (math.log(f[6]) / math.log(1000.0)) ** 1.5
        speed = min(speed, ship_speed)
        ox, oy = f[2], f[3]
        f[2] += math.cos(f[4]) * speed
        f[3] += math.sin(f[4]) * speed
        nx, ny = f[2], f[3]
        hit = False
        for p in planets:
            pp = planet_paths.get(p[0])
            if pp is None or not pp[4]:
                continue
            if _swept_pair_hit(ox, oy, nx, ny, pp[0], pp[1], pp[2], pp[3], p[4]):
                combat_lists[p[0]].append((f[1], f[6]))
                fleets_to_remove.add(f[0])
                hit = True
                break
        if hit:
            continue
        if not (0 <= f[2] <= BOARD_SIZE and 0 <= f[3] <= BOARD_SIZE):
            fleets_to_remove.add(f[0])
            continue
        if _point_to_segment_distance(CENTER, CENTER, ox, oy, nx, ny) < SUN_RADIUS:
            fleets_to_remove.add(f[0])

    # --- 4. Apply planet movement ---
    for p in planets:
        pp = planet_paths.get(p[0])
        if pp is not None:
            p[2], p[3] = pp[2], pp[3]

    if expired_after_move:
        _remove_planets(s, set(expired_after_move))
        planets, fleets = s["planets"], s["fleets"]

    s["fleets"] = [f for f in s["fleets"] if f[0] not in fleets_to_remove]

    # --- 5. Combat resolution (planet order) ---
    for p in s["planets"]:
        arriving = combat_lists.get(p[0])
        if not arriving:
            continue
        player_ships = []  # (owner, ships) first-seen order
        for owner, ships in arriving:
            for ps in player_ships:
                if ps[0] == owner:
                    ps[1] += ships
                    break
            else:
                player_ships.append([owner, ships])
        player_ships.sort(key=lambda x: -x[1])  # stable: Python sort is stable
        top_player, top_ships = player_ships[0]
        if len(player_ships) > 1:
            second = player_ships[1][1]
            survivor_ships = top_ships - second
            if player_ships[0][1] == player_ships[1][1]:
                survivor_ships = 0
            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner, survivor_ships = top_player, top_ships
        if survivor_ships > 0:
            if p[1] == survivor_owner:
                p[5] += survivor_ships
            else:
                p[5] -= survivor_ships
                if p[5] < 0:
                    p[1] = survivor_owner
                    p[5] = -p[5]

    s["step"] = step_idx  # caller increments
    return s
