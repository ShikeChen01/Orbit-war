"""Evaluate a TARGET-actor checkpoint in the STRICTLY-OFFICIAL Kaggle engine
(`kaggle_environments` `orbit_wars`), vs random & starter, over many games.

Transition code (`make_agent`): Kaggle `obs` -> our GpuEnv state (planets by id->slot, fleets,
angular_velocity, comets) -> `env_encode` (matching-generation defs) -> greedy policy ->
`_decode_target` -> Kaggle moves `[planet_id, angle, ships]`.

Stock official engine only (comets spawn at random steps the agent can't foresee -- true deployment).

    python scripts/eval_kaggle_engine.py \
        --ckpt notebooks/weight/weights/256-32-MLPs-E610-Iter950.pt \
        --defs-nb notebooks/render/v5_legacy_defs.ipynb --games 100
"""
from __future__ import annotations
import argparse, os, sys
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import viz_target_ckpt as V   # noqa: E402


def _g(obs, key, default=None):
    try:
        return obs[key]
    except Exception:
        return getattr(obs, key, default)


def make_agent(g, net):
    """Return a Kaggle agent(obs, config) closure. Persistent B=1 CPU env, overwritten each turn."""
    GpuEnv, env_encode, act, _decode_target = g["GpuEnv"], g["env_encode"], g["act"], g["_decode_target"]
    PLANET_CAP, FLEET_CAP = g["PLANET_CAP"], g["FLEET_CAP"]
    CENTER, RLIM, SHIP_SPEED = g["CENTER"], g["ROTATION_RADIUS_LIMIT"], g["SHIP_SPEED"]
    EP = g["EPISODE_STEPS"]
    DEV = g["DEVICE"]                            # match the module's device (its constant tensors live there)
    env = GpuEnv(PLANET_CAP, FLEET_CAP, EP, SHIP_SPEED, DEV)
    env.reset([g["generate_world"](0)])          # allocate B=1 tensors once

    def agent(obs, config):
        me = int(_g(obs, "player", 0))
        planets = _g(obs, "planets", []) or []
        fleets = _g(obs, "fleets", []) or []
        angv = float(_g(obs, "angular_velocity", 0.0) or 0.0)
        comet_ids = set(int(c) for c in (_g(obs, "comet_planet_ids", []) or []))
        env.T = int(_g(config, "episodeSteps", EP) or EP)
        # ---- planets: place by id -> slot (matches training slot=list-order, ids 0..N-1) ----
        env.p_alive.zero_(); env.p_owner.fill_(-1.0)
        for t in (env.p_x, env.p_y, env.p_radius, env.p_ships, env.p_prod, env.p_is_comet, env.p_rotates):
            t.zero_()
        for p in planets:
            pid, owner, x, y, rad, ships, prod = p
            s = int(pid)
            if s >= PLANET_CAP:
                continue
            env.p_alive[0, s] = 1.0
            env.p_owner[0, s] = 0.0 if owner == me else (-1.0 if (owner is None or owner < 0) else 1.0)
            env.p_x[0, s] = x; env.p_y[0, s] = y; env.p_radius[0, s] = rad
            env.p_ships[0, s] = ships; env.p_prod[0, s] = prod
            is_comet = s in comet_ids
            env.p_is_comet[0, s] = 1.0 if is_comet else 0.0
            rr = ((x - CENTER) ** 2 + (y - CENTER) ** 2) ** 0.5
            env.p_rotates[0, s] = 1.0 if ((not is_comet) and (rr + rad < RLIM)) else 0.0
        env.ang_vel[0] = angv
        # ---- fleets ----
        env.f_alive.zero_(); env.f_owner.zero_()
        for t in (env.f_x, env.f_y, env.f_angle, env.f_ships, env.f_seq):
            t.zero_()
        for j, f in enumerate(fleets):
            if j >= FLEET_CAP:
                break
            fid, owner, x, y, angle, from_pid, ships = f
            env.f_alive[0, j] = 1.0
            env.f_owner[0, j] = 0.0 if owner == me else 1.0
            env.f_x[0, j] = x; env.f_y[0, j] = y; env.f_angle[0, j] = angle
            env.f_ships[0, j] = ships; env.f_seq[0, j] = float(fid)
        env.step_ct[0] = float(_g(obs, "step", 0) or 0)
        # ---- encode -> greedy act -> decode -> moves ----
        with torch.no_grad():
            ent, em, am, gl = env_encode(env, 0)
            a_t, _, _ = act(net, ent, em, am, gl, greedy=True)
            legal = (env.p_owner == 0.0) & (env.p_alive > 0.5) & (env.p_ships > 0.0)
            angle, ships, can, valid = _decode_target(env, a_t, legal)[:4]
        moves = []
        for s in range(PLANET_CAP):
            if bool(can[0, s] > 0.5) and bool(legal[0, s]):
                n = int(ships[0, s].item())
                if n > 0:
                    moves.append([s, float(angle[0, s].item()), n])
        return moves

    return agent


# ---------------- parallel workers (process pool; kaggle engine is pure-Python CPU) -------------
_W = {}


def _init_worker(ckpt, defs_nb, device, steps):
    import torch as _t
    _t.set_num_threads(1)                                   # 1 thread/process; the pool fills the cores
    os.chdir(REPO)
    V.NB = defs_nb
    g = V.load_notebook_defs()
    g["DEVICE"] = device
    if "PHI_BINS_T" in g:
        g["PHI_BINS_T"] = g["PHI_BINS_T"].to(device)
    net, cfg, blob = V.build_actor(g, ckpt)
    net = net.to(device).eval()
    _W.update(agent=make_agent(g, net), steps=steps)


def _play_game(task):
    opp, idx = task
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"episodeSteps": _W["steps"]}, debug=False)  # stock engine (comets on)
    seat = idx % 2                                          # alternate seats for fairness
    roster = [_W["agent"], opp] if seat == 0 else [opp, _W["agent"]]
    out = env.run(roster)
    r = out[-1][seat]["reward"]
    res = "d" if (r is None or r == 0) else ("w" if r > 0 else "l")
    return (opp, res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--defs-nb", default="notebooks/render/v5_legacy_defs.ipynb")
    ap.add_argument("--games", type=int, default=40, help="games per opponent")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--opponents", default="random,starter")
    ap.add_argument("--device", default="cpu", help="cpu (fast for B=1) or cuda")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    os.chdir(REPO)
    import math as _m
    from concurrent.futures import ProcessPoolExecutor

    opps = [o.strip() for o in args.opponents.split(",")]
    tasks = [(op, i) for op in opps for i in range(args.games)]
    print("ckpt %s | %d games/opp x %d opp = %d games | %d workers | stock Kaggle engine (comets on)\n"
          % (os.path.basename(args.ckpt), args.games, len(opps), len(tasks), args.workers))

    tally = {op: {"w": 0, "l": 0, "d": 0} for op in opps}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.ckpt, args.defs_nb, args.device, args.steps)) as ex:
        for op, res in ex.map(_play_game, tasks, chunksize=2):
            tally[op][res] += 1

    z = 1.96
    lines = ["%-8s | %-4s %-4s %-4s | win-rate (95%% CI)" % ("opp", "W", "L", "D"), "-" * 48]
    for op in opps:
        t = tally[op]; n = t["w"] + t["l"] + t["d"]
        wr = (t["w"] + 0.5 * t["d"]) / max(1, n)
        ci = z * _m.sqrt(max(wr * (1 - wr), 1e-9) / max(1, n))
        lines.append("%-8s | %-4d %-4d %-4d | %.3f +/- %.3f" % (op, t["w"], t["l"], t["d"], wr, ci))
    out = "\n".join(lines)
    print(out)
    name = os.path.splitext(os.path.basename(args.ckpt))[0]
    od = os.path.join("runs", "viz", name); os.makedirs(od, exist_ok=True)
    open(os.path.join(od, "kaggle_engine_eval.txt"), "w").write(
        "ckpt %s | %d games/opp\n\n%s\n" % (args.ckpt, args.games, out))


if __name__ == "__main__":
    main()
