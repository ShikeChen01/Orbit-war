"""Round-robin all scripted agents in the STRICTLY-OFFICIAL Kaggle engine to justify their anchor ELOs.

Wraps each scripted agent (opponent_action codes: random/starter/medium/greedy) as a Kaggle
agent (obs -> GpuEnv with the scripted side as owner 1 -> opponent_action -> moves), plays every pair
both seats, and fits implied ELOs (anchored random=0) from the pairwise win-rate matrix. Compare the
implied gaps to the assigned anchors (random 0, starter 450, greedy 500, medium 500 -- the
Kaggle-leaderboard-calibrated values; in-house implied gaps are expected to run WIDER than
these compressed anchors).

    python scripts/scripted_tournament.py --games 40
"""
from __future__ import annotations
import argparse, os, sys, math
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import viz_target_ckpt as V   # noqa: E402

KIND2CODE = {"random": 0, "starter": 1, "medium": 3, "greedy": 4}
ASSIGNED = {"random": 0, "starter": 450, "greedy": 500, "medium": 500}


def _g(obs, key, default=None):
    try:
        return obs[key]
    except Exception:
        return getattr(obs, key, default)


def make_scripted_agent(g, code):
    """Kaggle agent driving opponent_action(code) for the acting side (synced as owner 1)."""
    GpuEnv, opponent_action = g["GpuEnv"], g["opponent_action"]
    PC, FC = g["PLANET_CAP"], g["FLEET_CAP"]
    CENTER, RLIM = g["CENTER"], g["ROTATION_RADIUS_LIMIT"]
    env = GpuEnv(PC, FC, g["EPISODE_STEPS"], g["SHIP_SPEED"], g["DEVICE"])
    env.reset([g["generate_world"](0)])

    def agent(obs, config):
        me = int(_g(obs, "player", 0))
        env.p_alive.zero_(); env.p_owner.fill_(-1.0)
        for t in (env.p_x, env.p_y, env.p_radius, env.p_ships, env.p_prod, env.p_is_comet, env.p_rotates):
            t.zero_()
        for p in _g(obs, "planets", []) or []:
            pid, owner, x, y, rad, sh, prod = p
            s = int(pid)
            if s >= PC:
                continue
            env.p_alive[0, s] = 1.0
            env.p_owner[0, s] = 1.0 if owner == me else (-1.0 if (owner is None or owner < 0) else 0.0)
            env.p_x[0, s] = x; env.p_y[0, s] = y; env.p_radius[0, s] = rad
            env.p_ships[0, s] = sh; env.p_prod[0, s] = prod
            rr = ((x - CENTER) ** 2 + (y - CENTER) ** 2) ** 0.5
            env.p_rotates[0, s] = 1.0 if (rr + rad < RLIM) else 0.0
        env.ang_vel[0] = float(_g(obs, "angular_velocity", 0.0) or 0.0)
        for t in (env.f_alive, env.f_owner, env.f_x, env.f_y, env.f_angle, env.f_ships):
            t.zero_()
        for j, f in enumerate(_g(obs, "fleets", []) or []):
            if j >= FC:
                break
            fid, owner, x, y, angle, frm, sh = f
            env.f_alive[0, j] = 1.0; env.f_owner[0, j] = 1.0 if owner == me else 0.0
            env.f_x[0, j] = x; env.f_y[0, j] = y; env.f_angle[0, j] = angle; env.f_ships[0, j] = sh
        env.step_ct[0] = float(_g(obs, "step", 0) or 0)
        with torch.no_grad():
            ang, shp, com = opponent_action(env, code)
        moves = []
        for s in range(PC):
            if bool(com[0, s] > 0.5):
                n = int(shp[0, s].item())
                if n > 0:
                    moves.append([s, float(ang[0, s].item()), n])
        return moves

    return agent


# ---------------- parallel pool ----------------
_W = {}


def _init_worker(defs_nb, device, steps):
    import torch as _t
    _t.set_num_threads(1)
    os.chdir(REPO)
    V.NB = defs_nb
    g = V.load_notebook_defs()
    g["DEVICE"] = device
    _W["agents"] = {k: make_scripted_agent(g, c) for k, c in KIND2CODE.items()}
    _W["steps"] = steps


def _play(task):
    a, b, idx = task
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"episodeSteps": _W["steps"]}, debug=False)
    seat = idx % 2
    roster = [_W["agents"][a], _W["agents"][b]] if seat == 0 else [_W["agents"][b], _W["agents"][a]]
    out = env.run(roster)
    r = out[-1][seat]["reward"]      # reward for agent a (in its seat)
    res = "d" if (r is None or r == 0) else ("w" if r > 0 else "l")
    return (a, b, res)


def fit_elos(wr, kinds, anchor="random", iters=30000, lr=2.0):
    """Unbiased iterative Elo fit: each rating moves so its expected total score matches the actual
    total score over the win matrix. Init at 0 (NOT the assigned values), anchor pinned at 0."""
    elo = {k: 0.0 for k in kinds}
    for _ in range(iters):
        for a in kinds:
            if a == anchor:
                continue
            exp = act = 0.0
            for b in kinds:
                if b == a or (a, b) not in wr:
                    continue
                exp += 1.0 / (1.0 + 10.0 ** ((elo[b] - elo[a]) / 400.0))
                act += wr[(a, b)]
            elo[a] += lr * (act - exp)
    shift = -elo[anchor]
    return {k: elo[k] + shift for k in kinds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defs-nb", default="notebooks/render/v5_legacy_defs.ipynb")
    ap.add_argument("--games", type=int, default=40, help="games per ordered pair")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    os.chdir(REPO)
    from concurrent.futures import ProcessPoolExecutor

    kinds = ["random", "starter", "greedy", "medium"]
    upairs = [(kinds[i], kinds[j]) for i in range(len(kinds)) for j in range(i + 1, len(kinds))]
    tasks = [(a, b, i) for (a, b) in upairs for i in range(args.games)]
    print("scripted round-robin | %d agents | %d pairs x %d games = %d games (seats alternate; directions complementary) | %d workers\n"
          % (len(kinds), len(upairs), args.games, len(tasks), args.workers))

    tally = {(a, b): {"w": 0, "l": 0, "d": 0} for (a, b) in upairs}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.defs_nb, args.device, args.steps)) as ex:
        for a, b, res in ex.map(_play, tasks, chunksize=2):
            tally[(a, b)][res] += 1

    wr = {}
    for (a, b), t in tally.items():
        n = t["w"] + t["l"] + t["d"]
        wr[(a, b)] = (t["w"] + 0.5 * t["d"]) / max(1, n)   # a's win-rate over the shared games
        wr[(b, a)] = (t["l"] + 0.5 * t["d"]) / max(1, n)   # complementary by construction

    # win-rate matrix (row beats column)
    print("win-rate matrix (row vs column):")
    hdr = "%-9s" % "" + "".join("%-9s" % k[:8] for k in kinds)
    print(hdr)
    for a in kinds:
        row = "%-9s" % a[:8]
        for b in kinds:
            row += "%-9s" % ("--" if a == b else "%.2f" % wr[(a, b)])
        print(row)
    overall = {a: sum(wr[(a, b)] for b in kinds if b != a) / (len(kinds) - 1) for a in kinds}

    fitted = fit_elos(wr, kinds)
    print("\n%-10s %-10s %-12s %-10s" % ("agent", "overall-wr", "implied-ELO", "assigned"))
    print("-" * 44)
    for k in sorted(kinds, key=lambda x: ASSIGNED[x]):
        print("%-10s %-10.2f %-12.0f %-10d" % (k, overall[k], fitted[k], ASSIGNED[k]))
    print("\n(implied ELO = Bradley-Terry fit to the win matrix, random anchored at 0)")


if __name__ == "__main__":
    main()
