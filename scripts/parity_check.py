"""Dynamics parity: official Kaggle engine vs our GpuEnv, step-for-step.

Runs starter-vs-starter in the OFFICIAL engine (comet-free window: <50 steps), then replays the
SAME per-turn moves through an explicit-launch GpuEnv stepper and compares planet ships/owner/pos
+ fleet count every tick. First divergence => the vectorization bug. No policy involved.

    python scripts/parity_check.py --steps 45 --seeds 0,1,2,3,4
"""
from __future__ import annotations
import argparse, os, sys, math
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import viz_target_ckpt as V   # noqa: E402

DT = torch.float32


def load_defs(defs_nb):
    V.NB = defs_nb
    g = V.load_notebook_defs()
    g["DEVICE"] = "cpu"
    return g


def sync_env(env, g, obs):
    """Load an official obs (absolute owners 0/1/-1) into the GpuEnv. p_init/p_rotates from
    initial_planets so the orbit reproduces the engine; step_ct = obs.step."""
    CENTER, RLIM = g["CENTER"], g["ROTATION_RADIUS_LIMIT"]
    PC = g["PLANET_CAP"]
    for t in (env.p_alive, env.p_x, env.p_y, env.p_radius, env.p_ships, env.p_prod,
              env.p_is_comet, env.p_rotates, env.p_init_x, env.p_init_y):
        t.zero_()
    env.p_owner.fill_(-1.0)
    init_by_id = {int(p[0]): p for p in obs["initial_planets"]}
    for p in obs["planets"]:
        pid, owner, x, y, rad, ships, prod = p
        s = int(pid)
        if s >= PC:
            continue
        env.p_alive[0, s] = 1.0
        env.p_owner[0, s] = float(owner if owner is not None else -1)
        env.p_x[0, s] = x; env.p_y[0, s] = y; env.p_radius[0, s] = rad
        env.p_ships[0, s] = ships; env.p_prod[0, s] = prod
        ip = init_by_id.get(s, p)
        env.p_init_x[0, s] = ip[2]; env.p_init_y[0, s] = ip[3]
        rr = math.hypot(ip[2] - CENTER, ip[3] - CENTER)
        env.p_rotates[0, s] = 1.0 if (rr + rad < RLIM) else 0.0
    env.ang_vel[0] = float(obs.get("angular_velocity", 0.0) or 0.0)
    # fleets
    for t in (env.f_alive, env.f_owner, env.f_x, env.f_y, env.f_angle, env.f_ships, env.f_seq):
        t.zero_()
    for j, f in enumerate(obs.get("fleets", []) or []):
        if j >= g["FLEET_CAP"]:
            break
        fid, owner, x, y, angle, from_pid, ships = f
        env.f_alive[0, j] = 1.0; env.f_owner[0, j] = float(owner)
        env.f_x[0, j] = x; env.f_y[0, j] = y; env.f_angle[0, j] = angle
        env.f_ships[0, j] = ships; env.f_seq[0, j] = float(fid)
    env.step_ct[0] = float(obs.get("step", 0) or 0)


def step_explicit(env, g, moves0, moves1):
    """One engine tick with EXPLICIT launches (no policy). Mirrors env_step dynamics exactly."""
    CENTER, BOARD, SUN = g["CENTER"], g["BOARD_SIZE"], g["SUN_RADIUS"]
    fleet_speed_t, launch_fleets = g["fleet_speed_t"], g["launch_fleets"]
    B, Ec, dev, vmax = env.B, env.Ec, env.dev, env.vmax
    # ---- launches: deduct + place (official: process_moves then production) ----
    rows = [(0, moves0), (1, moves1)]
    ow, fs, an, sh = [], [], [], []
    for pl, moves in rows:
        for mv in moves:
            fid, a, n = int(mv[0]), float(mv[1]), int(mv[2])
            if fid < Ec and 0 < n and env.p_ships[0, fid].item() >= n and int(env.p_owner[0, fid].item()) == pl:
                env.p_ships[0, fid] -= n
                ow.append(float(pl)); fs.append(fid); an.append(a); sh.append(float(n))
    if ow:
        L = len(ow)
        owner = torch.tensor(ow, dtype=DT).view(1, L)
        from_slot = torch.tensor(fs, dtype=torch.long).view(1, L)
        angle = torch.tensor(an, dtype=DT).view(1, L)
        ships = torch.tensor(sh, dtype=DT).view(1, L)
        commit = torch.ones(1, L, dtype=DT)
        seq = (env.step_ct * float(L + 1)).view(1, 1) + torch.arange(L, dtype=DT).view(1, L)
        launch_fleets(env, owner, from_slot, angle, ships, commit, seq)
    # ---- production ----
    env.p_ships = env.p_ships + env.p_prod * (env.p_owner != -1.0).to(DT) * (env.p_alive > 0.5).to(DT)
    # ---- planet orbit ----
    stepf = env.step_ct
    dxc = env.p_init_x - CENTER; dyc = env.p_init_y - CENTER
    r = torch.sqrt(dxc * dxc + dyc * dyc); ia = torch.atan2(dyc, dxc)
    ca = ia + env.ang_vel.unsqueeze(1) * stepf.unsqueeze(1)
    rot = env.p_rotates > 0.5
    nx = torch.where(rot, CENTER + r * torch.cos(ca), env.p_x)
    ny = torch.where(rot, CENTER + r * torch.sin(ca), env.p_y)
    old_px, old_py = env.p_x, env.p_y
    # ---- fleet movement + swept collision ----
    falive = env.f_alive > 0.5
    speed = fleet_speed_t(env.f_ships, vmax)
    fox, foy = env.f_x, env.f_y
    fnx = fox + torch.cos(env.f_angle) * speed
    fny = foy + torch.sin(env.f_angle) * speed
    Ax = fox.unsqueeze(2); Ay = foy.unsqueeze(2); Bx = fnx.unsqueeze(2); By = fny.unsqueeze(2)
    P0x = old_px.unsqueeze(1); P0y = old_py.unsqueeze(1); P1x = nx.unsqueeze(1); P1y = ny.unsqueeze(1)
    rad = env.p_radius.unsqueeze(1); palive = (env.p_alive.unsqueeze(1) > 0.5)
    d0x = Ax - P0x; d0y = Ay - P0y
    dvx = (Bx - Ax) - (P1x - P0x); dvy = (By - Ay) - (P1y - P0y)
    a = dvx * dvx + dvy * dvy; b = 2.0 * (d0x * dvx + d0y * dvy); c = d0x * d0x + d0y * d0y - rad * rad
    disc = b * b - 4.0 * a * c; sq = torch.sqrt(disc.clamp_min(0.0))
    t1 = (-b - sq) / (2.0 * a); t2 = (-b + sq) / (2.0 * a)
    hit = torch.where(a < 1e-12, (a < 1e-12) & (c <= 0.0), (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0))
    hit = hit & palive & falive.unsqueeze(2)
    slotf = torch.arange(Ec, dtype=DT).view(1, 1, Ec)
    fh_v, tgt_slot = torch.where(hit, slotf, torch.full_like(slotf, float(Ec))).min(2)
    has_hit = fh_v < float(Ec)
    oob = (fnx < 0.0) | (fnx > BOARD) | (fny < 0.0) | (fny > BOARD)
    l2 = (fox - fnx) ** 2 + (foy - fny) ** 2
    tt = (((CENTER - fox) * (fnx - fox) + (CENTER - foy) * (fny - foy)) / l2.clamp_min(1e-12)).clamp(0, 1)
    prx = fox + tt * (fnx - fox); pry = foy + tt * (fny - foy)
    sun_hit = torch.where(l2 > 0.0, (l2 > 0.0) & (torch.sqrt((CENTER - prx) ** 2 + (CENTER - pry) ** 2) < SUN),
                          torch.sqrt((CENTER - fox) ** 2 + (CENTER - foy) ** 2) < SUN)
    remove_fleet = falive & (has_hit | oob | sun_hit)
    contributes = falive & has_hit
    # ---- combat ----
    arr0 = torch.zeros(B, Ec, dtype=DT); arr1 = torch.zeros(B, Ec, dtype=DT)
    cf = contributes.to(DT)
    tslot = tgt_slot.clamp(0, Ec - 1)
    arr0.scatter_add_(1, tslot, env.f_ships * cf * (env.f_owner == 0.0).to(DT))
    arr1.scatter_add_(1, tslot, env.f_ships * cf * (env.f_owner == 1.0).to(DT))
    has0 = arr0 > 0.0; has1 = arr1 > 0.0
    top = torch.maximum(arr0, arr1); second = torch.minimum(arr0, arr1)
    both = has0 & has1
    surv = torch.where(both, top - second, top)
    surv = torch.where(both & (arr0 == arr1), torch.zeros_like(surv), surv)
    surv_owner = torch.full((B, Ec), -1.0, dtype=DT)
    surv_owner = torch.where(arr0 > arr1, torch.zeros_like(surv_owner), surv_owner)
    surv_owner = torch.where(arr1 > arr0, torch.ones_like(surv_owner), surv_owner)
    apply = (has0 | has1) & (surv > 0.0) & (env.p_alive > 0.5)
    same = (env.p_owner == surv_owner)
    reinforce = apply & same; attack = apply & (~same)
    env.p_ships = torch.where(reinforce, env.p_ships + surv, env.p_ships)
    after = env.p_ships - surv
    flips = attack & (after < 0.0)
    env.p_ships = torch.where(attack, torch.where(after < 0.0, -after, after), env.p_ships)
    env.p_owner = torch.where(flips, surv_owner, env.p_owner)
    # ---- apply positions + clear fleets ----
    env.p_x = nx; env.p_y = ny
    keep = (falive & (~remove_fleet)).to(DT)
    env.f_alive = keep; env.f_owner = env.f_owner * keep
    env.f_x = fnx * keep; env.f_y = fny * keep; env.f_angle = env.f_angle * keep
    env.f_ships = env.f_ships * keep; env.f_seq = env.f_seq * keep
    env.step_ct = env.step_ct + 1.0


def compare(env, g, obs_post, tol=0.5):
    """Return list of mismatch strings vs the official post-step obs."""
    off = {int(p[0]): p for p in obs_post["planets"]}
    bad = []
    PC = g["PLANET_CAP"]
    ours_alive = set(s for s in range(PC) if env.p_alive[0, s].item() > 0.5)
    if ours_alive != set(off.keys()):
        bad.append("alive-set: ours=%s off=%s" % (sorted(ours_alive), sorted(off.keys())))
    for pid, p in off.items():
        if pid >= PC:
            continue
        o_owner = int(env.p_owner[0, pid].item()); o_ships = env.p_ships[0, pid].item()
        o_x = env.p_x[0, pid].item(); o_y = env.p_y[0, pid].item()
        if int(p[1]) != o_owner:
            bad.append("p%d owner off=%d ours=%d" % (pid, int(p[1]), o_owner))
        if abs(p[5] - o_ships) > tol:
            bad.append("p%d ships off=%d ours=%.1f" % (pid, p[5], o_ships))
        if abs(p[2] - o_x) > 1e-2 or abs(p[3] - o_y) > 1e-2:
            bad.append("p%d pos off=(%.3f,%.3f) ours=(%.3f,%.3f)" % (pid, p[2], p[3], o_x, o_y))
    n_off_fleets = len(obs_post.get("fleets", []) or [])
    n_our_fleets = int((env.f_alive[0] > 0.5).sum().item())
    if n_off_fleets != n_our_fleets:
        bad.append("fleet-count off=%d ours=%d" % (n_off_fleets, n_our_fleets))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defs-nb", default="notebooks/render/v5_legacy_defs.ipynb")
    ap.add_argument("--steps", type=int, default=45)   # < 50 keeps it comet-free
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    os.chdir(REPO)
    g = load_defs(args.defs_nb)
    GpuEnv = g["GpuEnv"]
    from kaggle_environments import make
    from kaggle_environments.envs.orbit_wars import orbit_wars as OW

    env = GpuEnv(g["PLANET_CAP"], g["FLEET_CAP"], 500, g["SHIP_SPEED"], "cpu")
    env.reset([g["generate_world"](0)])

    n_seeds = [int(s) for s in args.seeds.split(",")]
    print("parity: official starter-vs-starter vs GpuEnv replay | %d steps | %d games\n" % (args.steps, len(n_seeds)))
    total_div = 0
    for sd in n_seeds:
        kenv = make("orbit_wars", configuration={"episodeSteps": args.steps + 2}, debug=False)
        out = kenv.run(["starter", "starter"])
        first_div = None
        for t in range(min(args.steps, len(out) - 1)):
            obs_pre = out[t][0]["observation"]
            m0 = OW.starter_agent(out[t][0]["observation"])
            m1 = OW.starter_agent(out[t][1]["observation"])
            sync_env(env, g, obs_pre)
            step_explicit(env, g, m0, m1)
            bad = compare(env, g, out[t + 1][0]["observation"])
            if bad:
                first_div = (t, bad)
                break
        if first_div is None:
            print("seed %d: MATCH for all %d steps" % (sd, min(args.steps, len(out) - 1)))
        else:
            total_div += 1
            t, bad = first_div
            print("seed %d: DIVERGE at step %d -> %s" % (sd, t, "; ".join(bad[:6])))
    print("\n%d/%d games diverged" % (total_div, len(n_seeds)))


if __name__ == "__main__":
    main()
