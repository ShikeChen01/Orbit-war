"""Pure-Python world generator (notebook cell 11).

Faithful mirror of the official ``REFERENCE_orbit_wars.py::generate_planets`` /
``generate_comet_paths`` (bit-exact-verified by ``scripts/test_official_comets.py``). World dicts
hold ``planets`` rows ``[id, owner, x, y, radius, ships, production]``, ``angular_velocity``,
``home_base`` and (when ``cfg.COMET_OFFICIAL``) padded comet-path arrays.
"""
import math
import random

import numpy as np

from .constants import (BOARD_SIZE, CENTER, COMET_RADIUS, MAX_PLANET_GROUPS, MIN_PLANET_GROUPS,
                        MIN_STATIC_GROUPS, PLANET_CLEARANCE, ROTATION_RADIUS_LIMIT, SUN_RADIUS)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def generate_planets(rng):
    '''Faithful mirror of REFERENCE_orbit_wars.py::generate_planets.
    Returns rows [id, owner, x, y, radius, ships, production].'''
    planets = []
    num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    idc = 0
    # Phase 1: guaranteed static groups (polar sampling).
    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        angle = rng.uniform(0, math.pi / 2)
        min_orbital = ROTATION_RADIUS_LIMIT - r
        max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital:
            continue
        orbital_r = rng.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)
        if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
            continue
        if (BOARD_SIZE - x) - r < 0 or (BOARD_SIZE - y) - r < 0:
            continue
        if (x - CENTER) < r + 5 or (y - CENTER) < r + 5:
            continue
        ships = min(rng.randint(5, 99), rng.randint(5, 99))
        # NOTE: the reference stores rows as [id, owner, y, x, r, ...] (x/y swapped naming),
        # which is just a relabel of the symmetric copies; we keep it identical.
        tps = [
            [idc, -1, y, x, r, ships, prod],
            [idc + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [idc + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [idc + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]
        valid = True
        for tp in tps:
            for p in planets:
                if _dist((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False; break
            if not valid:
                break
        if valid:
            planets.extend(tps); idc += 4; static_groups += 1
    # Phase 2: fill remaining groups (normal random loop).
    attempts = 0
    max_attempts = 5000
    has_orbiting = False
    while len(planets) < num_q1 * 4 or (not has_orbiting and attempts < max_attempts):
        attempts += 1
        if attempts >= max_attempts:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        x = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        y = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        orbital_radius = _dist((x, y), (CENTER, CENTER))
        if orbital_radius < SUN_RADIUS + r + 10:
            continue
        if orbital_radius + r >= ROTATION_RADIUS_LIMIT:
            if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
                continue
        valid = True
        ships = rng.randint(5, 30)
        tps = [
            [idc, -1, y, x, r, ships, prod],
            [idc + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [idc + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [idc + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]
        for tp in tps:
            tp_orb = _dist((tp[2], tp[3]), (CENTER, CENTER))
            tp_rot = tp_orb + tp[4] < ROTATION_RADIUS_LIMIT
            for p in planets:
                p_orb = _dist((p[2], p[3]), (CENTER, CENTER))
                p_rot = p_orb + p[4] < ROTATION_RADIUS_LIMIT
                if _dist((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False; break
                if tp_rot != p_rot:
                    if abs(tp_orb - p_orb) < tp[4] + p[4] + PLANET_CLEARANCE:
                        valid = False; break
            if not valid:
                break
        if valid:
            if orbital_radius + r < ROTATION_RADIUS_LIMIT:
                has_orbiting = True
            planets.extend(tps); idc += 4
    return planets


def generate_world(cfg, seed):
    '''Mirror of native_worldgen.generate_world (comets dropped unless cfg.COMET_OFFICIAL).'''
    rng = random.Random(seed)
    angular_velocity = rng.uniform(0.025, 0.05)
    planets = generate_planets(rng)
    num_groups = len(planets) // 4
    base = -1
    if num_groups > 0:
        base = rng.randint(0, num_groups - 1) * 4
        planets[base][1] = 0;      planets[base][5] = 10       # player 0 home
        planets[base + 3][1] = 1;  planets[base + 3][5] = 10   # player 1 home
    # v8: remember the home group; env.reset(n_players=4) re-seats owners 0..3 on base..base+3
    # exactly like the official engine's 4-player branch (same group => fair under 4-fold symmetry).
    w = {"planets": planets, "angular_velocity": angular_velocity, "home_base": base}
    if cfg.COMET_OFFICIAL:
        attach_official_comets(cfg, w, seed)
    return w


# ---- OFFICIAL comet paths (numpy-vectorized port of REFERENCE_orbit_wars.generate_comet_paths;
# ---- same math/draw-order/reject-semantics, bit-exact-verified by scripts/test_official_comets.py)
def _comet_paths_official(initial_planets, angular_velocity, spawn_step, comet_speed, rng):
    """Returns 4 symmetric waypoint paths (list of [x,y], one waypoint per tick) or None."""
    stat, orb = [], []
    for p in initial_planets:
        pr = math.sqrt((p[2] - CENTER) ** 2 + (p[3] - CENTER) ** 2)
        (orb if pr + p[4] < ROTATION_RADIUS_LIMIT else stat).append(p)
    stat_xy = np.array([[p[2], p[3]] for p in stat], np.float64).reshape(-1, 2)
    stat_rad = np.array([p[4] for p in stat], np.float64)
    orb_r = np.array([math.sqrt((p[2] - CENTER) ** 2 + (p[3] - CENTER) ** 2) for p in orb], np.float64)
    orb_a0 = np.array([math.atan2(p[3] - CENTER, p[2] - CENTER) for p in orb], np.float64)
    orb_rad = np.array([p[4] for p in orb], np.float64)
    num = 5000
    t_arr = 0.3 * math.pi + 1.4 * math.pi * np.arange(num) / (num - 1)
    cos_t, sin_t = np.cos(t_arr), np.sin(t_arr)
    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        if a * (1 - e) < SUN_RADIUS + COMET_RADIUS:
            continue
        b = a * math.sqrt(1 - e ** 2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)
        ex = c_val + a * cos_t; ey = b * sin_t
        cp, sp = math.cos(phi), math.sin(phi)
        x = CENTER + ex * cp - ey * sp
        y = CENTER + ex * sp + ey * cp
        dx = x[1:] - x[:-1]; dy = y[1:] - y[:-1]
        cum = np.cumsum(np.sqrt(dx * dx + dy * dy))            # cum[i-1] = arc length at dense[i]
        nk = int(cum[-1] // comet_speed) + 1
        sel = np.searchsorted(cum, comet_speed * np.arange(1, nk + 1), side="left") + 1
        sel = sel[sel < num]
        px = np.concatenate(([x[0]], x[sel])); py = np.concatenate(([y[0]], y[sel]))
        on = (px >= 0) & (px <= BOARD_SIZE) & (py >= 0) & (py <= BOARD_SIZE)
        if not on.any():
            continue
        i0 = int(np.argmax(on)); i1 = int(len(on) - 1 - np.argmax(on[::-1]))
        vx = px[i0:i1 + 1]; vy = py[i0:i1 + 1]                  # contiguous on-board SPAN (incl. interior)
        K = len(vx)
        if not (5 <= K <= 40):
            continue
        if (np.sqrt((vx - CENTER) ** 2 + (vy - CENTER) ** 2) < SUN_RADIUS + COMET_RADIUS).any():
            continue
        sym_x = np.stack([vy, BOARD_SIZE - vx, vx, BOARD_SIZE - vy])          # (4,K)
        sym_y = np.stack([vx, vy, BOARD_SIZE - vy, BOARD_SIZE - vx])
        if len(stat_xy):
            d = np.sqrt((sym_x[:, :, None] - stat_xy[None, None, :, 0]) ** 2
                        + (sym_y[:, :, None] - stat_xy[None, None, :, 1]) ** 2)
            if (d < stat_rad[None, None, :] + (COMET_RADIUS + 0.5)).any():
                continue
        if len(orb_r):
            gs = spawn_step - 1 + np.arange(K)                                # (K,)
            ang = orb_a0[:, None] + angular_velocity * gs[None, :]            # (P,K)
            ox = CENTER + orb_r[:, None] * np.cos(ang)
            oy = CENTER + orb_r[:, None] * np.sin(ang)
            d = np.sqrt((sym_x[:, :, None] - ox.T[None, :, :]) ** 2
                        + (sym_y[:, :, None] - oy.T[None, :, :]) ** 2)        # (4,K,P)
            if (d < orb_rad[None, None, :] + COMET_RADIUS).any():
                continue
        return [
            [[float(vy[k]), float(vx[k])] for k in range(K)],
            [[float(BOARD_SIZE - vx[k]), float(vy[k])] for k in range(K)],
            [[float(vx[k]), float(BOARD_SIZE - vy[k])] for k in range(K)],
            [[float(BOARD_SIZE - vy[k]), float(BOARD_SIZE - vx[k])] for k in range(K)],
        ]
    return None


def attach_official_comets(cfg, world, seed):
    """Precompute the official comet schedule for one world (CPU). Stores padded arrays:
    comet_paths (NS,4,COMET_MAX_LEN,2) f32 / comet_len (NS,) / comet_ships (NS,) where
    NS = len(COMET_SPAWN_STEPS). Draw order matches the engine exactly (paths, then ships)."""
    NS = len(cfg.COMET_SPAWN_STEPS)
    cp = np.zeros((NS, 4, cfg.COMET_MAX_LEN, 2), np.float32)
    cl = np.zeros((NS,), np.int64)
    cs = np.zeros((NS,), np.float32)
    for e, s in enumerate(cfg.COMET_SPAWN_STEPS):
        rng = random.Random(f"orbit_wars-comet-{seed}-{s}")
        paths = _comet_paths_official(world["planets"], world["angular_velocity"], s, cfg.COMET_SPEED, rng)
        if not paths:
            continue
        ships = min(rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 99))
        L = len(paths[0])
        for m in range(4):
            for k, (x, y) in enumerate(paths[m][:cfg.COMET_MAX_LEN]):
                cp[e, m, k, 0] = x; cp[e, m, k, 1] = y
        cl[e] = min(L, cfg.COMET_MAX_LEN)
        cs[e] = float(ships)
    world["comet_paths"] = cp; world["comet_len"] = cl; world["comet_ships"] = cs
    return world


def make_world_pool(cfg, n, base_seed=0):
    return [generate_world(cfg, base_seed + i) for i in range(n)]


# alias matching the plan's public-API name
build_world_pool = make_world_pool
