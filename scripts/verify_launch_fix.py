"""Verify the lead-the-target fix: call the REAL _decode_target to get launch headings, then fly each
fleet through a faithful sim (straight fleet at fleet_speed_t(n); orbiting dest at ang_vel; sub-tick
sampled for swept-collision fidelity) and report the hit rate on rotating vs static destinations, with
LEAD_TARGET on vs off.
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_target_ckpt import load_notebook_defs
import torch

g = load_notebook_defs()
C, SUN, BOARD, RLIM = g["CENTER"], g["SUN_RADIUS"], g["BOARD_SIZE"], g["ROTATION_RADIUS_LIMIT"]
fspeed = lambda n: float(g["fleet_speed_t"](torch.tensor([float(n)]), g["SHIP_SPEED"]).item())


def fly(sx, sy, ang, v, dx0, dy0, rad, rot, omega, steps=600, sub=10):
    Rd = math.hypot(dx0 - C, dy0 - C); ph0 = math.atan2(dy0 - C, dx0 - C)
    fx, fy = sx, sy; vx, vy = v * math.cos(ang), v * math.sin(ang)
    for t in range(1, steps + 1):
        for s in range(1, sub + 1):
            f = (t - 1 + s / sub)
            cx, cy = sx + f * vx, sy + f * vy
            if rot:
                ph = ph0 + omega * f; Dx, Dy = C + Rd * math.cos(ph), C + Rd * math.sin(ph)
            else:
                Dx, Dy = dx0, dy0
            if math.hypot(cx - Dx, cy - Dy) <= max(rad, 0.9):
                return True
            if math.hypot(cx - C, cy - C) <= SUN:
                return False
            if not (0 <= cx <= BOARD and 0 <= cy <= BOARD):
                return False
    return False


def decode_angle(env, src, dst):
    """Action launching 100% from src to dst; return the decoded heading for src + ships launched."""
    B, Ec = env.B, env.Ec
    a = torch.zeros(B, Ec, 2, device=env.dev)
    a[0, src, 0] = float(dst); a[0, src, 1] = float(g["N_PHI"] - 1)   # dest, max phi bucket (100%)
    legal = (env.p_owner == 0.0) & (env.p_alive > 0.5) & (env.p_ships > 0.0)
    angle, ships, can, valid, invalid, launches = g["_decode_target"](env, a, legal)
    return float(angle[0, src].item()), float(ships[0, src].item())


def run():
    GpuEnv = g["GpuEnv"]
    for lead in (False, True):
        g["LEAD_TARGET"] = lead
        rot_hit = rot_tot = sta_hit = sta_tot = 0
        for seed in (0, 1, 2, 3, 4):
            w = g["generate_world"](seed); omega = w["angular_velocity"]
            env = GpuEnv(g["PLANET_CAP"], g["FLEET_CAP"], 500, g["SHIP_SPEED"], g["DEVICE"])
            env.reset([w])
            po = env.p_owner[0].cpu().numpy(); pa = env.p_alive[0].cpu().numpy() > 0.5
            px = env.p_x[0].cpu().numpy(); py = env.p_y[0].cpu().numpy()
            pr = env.p_radius[0].cpu().numpy(); prot = env.p_rotates[0].cpu().numpy() > 0.5
            srcs = [i for i in range(len(pa)) if pa[i] and po[i] == 0.0 and env.p_ships[0, i].item() > 0]
            if not srcs:
                continue
            src = srcs[0]
            for dst in range(len(pa)):
                if not pa[dst] or dst == src:
                    continue
                ang, ships = decode_angle(env, src, dst)
                if ships < 1:
                    continue
                hit = fly(px[src], py[src], ang, fspeed(ships), px[dst], py[dst], float(pr[dst]),
                          bool(prot[dst]), omega)
                if prot[dst]:
                    rot_tot += 1; rot_hit += hit
                else:
                    sta_tot += 1; sta_hit += hit
        print("LEAD_TARGET=%-5s | rotating dests %d/%d=%3.0f%%   static dests %d/%d=%3.0f%%"
              % (lead, rot_hit, rot_tot, 100 * rot_hit / max(1, rot_tot),
                 sta_hit, sta_tot, 100 * sta_hit / max(1, sta_tot)))


if __name__ == "__main__":
    run()
