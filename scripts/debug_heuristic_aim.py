"""Debug: does the v5 atan2-straight launch heuristic actually hit its destination in the real
(rotating) environment? Mirrors the env dynamics exactly: a launched fleet flies STRAIGHT at fixed
speed, planets within ROTATION_RADIUS_LIMIT ORBIT the sun at the board angular velocity, and a hit
is fleet-within-planet-radius. The heuristic aims at the destination's position AT LAUNCH, so it does
not lead a moving target.
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_target_ckpt import load_notebook_defs

g = load_notebook_defs()
C = g["CENTER"]; SUN = g["SUN_RADIUS"]; BOARD = g["BOARD_SIZE"]; RLIM = g["ROTATION_RADIUS_LIMIT"]
import torch
fleet_speed = lambda n: float(g["fleet_speed_t"](torch.tensor([float(n)])).item())


def is_rotating(pl):
    x, y, rad = pl[2], pl[3], pl[4]
    return (math.hypot(x - C, y - C) + rad) < RLIM


def simulate(S, D, omega, ships=50, steps=600):
    """Fly a straight fleet from S aimed at D's launch position; rotate D each tick if it orbits."""
    sx, sy = S[2], S[3]
    dx0, dy0, rad = D[2], D[3], D[4]
    ang = math.atan2(dy0 - sy, dx0 - sx)
    v = fleet_speed(ships)
    vx, vy = v * math.cos(ang), v * math.sin(ang)
    Rd = math.hypot(dx0 - C, dy0 - C); ph0 = math.atan2(dy0 - C, dx0 - C)
    rot = is_rotating(D)
    fx, fy = sx, sy; mind = float("inf")
    for t in range(1, steps + 1):
        fx += vx; fy += vy
        if rot:
            ph = ph0 + omega * t
            Dx, Dy = C + Rd * math.cos(ph), C + Rd * math.sin(ph)
        else:
            Dx, Dy = dx0, dy0
        mind = min(mind, math.hypot(fx - Dx, fy - Dy))
        if math.hypot(fx - Dx, fy - Dy) <= max(rad, 0.9):
            return "HIT", t, mind, rad, rot
        if math.hypot(fx - C, fy - C) <= SUN:
            return "SUN", t, mind, rad, rot
        if not (0 <= fx <= BOARD and 0 <= fy <= BOARD):
            return "OFFBOARD", t, mind, rad, rot
    return "EXPIRE", steps, mind, rad, rot


for seed in (0, 1, 2):
    w = g["generate_world"](seed)
    omega = w["angular_velocity"]
    planets = w["planets"]
    # use one outer/static planet as the launch source (stable vantage)
    src = max(planets, key=lambda p: math.hypot(p[2] - C, p[3] - C))
    rot_hits = rot_tot = sta_hits = sta_tot = 0
    examples = []
    for D in planets:
        if D is src:
            continue
        res, t, mind, rad, rot = simulate(src, D, omega)
        hit = (res == "HIT")
        if rot:
            rot_tot += 1; rot_hits += hit
        else:
            sta_tot += 1; sta_hits += hit
        if rot and len(examples) < 3:
            examples.append((D[0], rot, res, round(mind, 1), round(rad, 1)))
    print("seed %d | omega=%.4f rad/tick | %d planets (%d rotating / %d static)"
          % (seed, omega, len(planets), rot_tot + (1 if is_rotating(src) else 0), sta_tot))
    print("   straight-heuristic HIT rate:  rotating dests %d/%d=%.0f%%   static dests %d/%d=%.0f%%"
          % (rot_hits, rot_tot, 100 * rot_hits / max(1, rot_tot),
             sta_hits, sta_tot, 100 * sta_hits / max(1, sta_tot)))
    for pid, rot, res, mind, rad in examples:
        print("     rotating dest pid=%d -> %-8s closest approach %.1f vs radius %.1f (miss by %.1fx)"
              % (pid, res, mind, rad, mind / max(rad, 0.9)))
