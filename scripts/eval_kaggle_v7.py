"""Evaluate a v7 TRANSFORMER target-actor checkpoint in the STRICTLY-OFFICIAL Kaggle engine
(`kaggle_environments` `orbit_wars`) against the training GAUNTLET (starter, intermediate, greedy).

The in-training gauntlet/Elo (e.g. it101.pt: Elo 1157) is measured in our vectorized GpuEnv. This
script re-measures the SAME opponents in the real engine, so the only differences from the GpuEnv
score are the engine itself (its own comets, float order). Both the learner AND each scripted anchor
play through the bit-exact obs->GpuEnv adapter, so the anchors are identical to the gauntlet ones
(opponent_action codes), not approximations.

The v5 `eval_kaggle_engine.py` builds the legacy MLP actor and is incompatible with the v7
transformer + GatedCat (bilinear dest_q/dest_k) heads. This script:
  * execs the v7 notebook defs (setup1_v7_a100.ipynb) up to (not incl.) the BC/train run,
  * builds PolicyNet with the ARCH read from the CHECKPOINT's own config (it101 trained N_HEADS=12,
    notebook default is 8) and strict-loads the weights,
  * learner agent: kaggle obs -> GpuEnv (self=owner0) -> env_encode -> act(greedy) -> _decode_target
    -> moves [planet_id, angle, ships] over every committed (src,dst),
  * scripted anchor agent: kaggle obs -> GpuEnv (self=owner1) -> opponent_action(code) -> moves.

    python scripts/eval_kaggle_v7.py --ckpt notebooks/weight/Transformers/it101.pt --games 60
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # CPU eval: official engine is pure-Python CPU, B=1
import argparse, json, math, sys
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DEFAULT = os.path.join("notebooks", "legacy", "setup1_v7_a100.ipynb")

# scripted-anchor name -> opponent_action code (mirrors _eval_score's OPP map in the notebook)
OPP_CODE = {"random": 0, "starter": 1, "noop": 2, "medium": 3, "greedy": 4, "intermediate": 5}
# arch keys read from the checkpoint config (override notebook globals before build_policy)
ARCH_KEYS = ("HIDDEN", "N_RES_BLOCKS", "D_G", "N_TX_LAYERS", "N_HEADS", "TX_MLP_RATIO",
             "N_STEM_RES", "N_HEAD_RES", "USE_GLU", "USE_ATTENTION", "VALUE_RES_BLOCKS", "ARCH",
             "PLANET_CAP", "F_DIM", "G_DIM", "DEST_HEAD")


def load_nb_defs(nb):
    """Exec the v7 notebook code cells up to (but not incl.) the BC/train RUN cells -> a namespace."""
    doc = json.load(open(os.path.join(REPO, nb), encoding="utf-8"))

    def has_run(s):                                   # a top-level bc_pretrain()/train() call (ignore comments)
        for ln in s.splitlines():
            code = ln.split("#", 1)[0]
            if ("= train(" in code and "def train(" not in code) or "bc_pretrain()" in code:
                return True
        return False

    parts = []
    for c in doc["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if has_run(s):
            break
        parts.append(s)
    g = {"__name__": "__v7eval__"}
    exec(compile("\n\n".join(parts), "nb_defs", "exec"), g)
    return g


def build_actor_v7(g, ckpt):
    """Build PolicyNet with the ARCH from the ckpt's own config; strict-load the transformer weights."""
    blob = torch.load(os.path.join(REPO, ckpt), map_location="cpu", weights_only=False)
    cfg = blob.get("config") or {}
    g["DEVICE"] = torch.device("cpu")
    for k in ARCH_KEYS:
        if k in cfg:
            g[k] = cfg[k]
    net = g["build_policy"]()
    net.load_state_dict(blob["model"], strict=True)   # exact match or raise
    return net.to("cpu").eval(), cfg, blob


def _g(obs, key, default=None):
    try:
        return obs[key]
    except Exception:
        return getattr(obs, key, default)


def _load_obs_into_env(g, env, obs, config, self_code):
    """Map a Kaggle obs onto the persistent B=1 GpuEnv. self_code=0.0 -> acting player is owner 0
    (learner, env_encode ego=0); self_code=1.0 -> acting player is owner 1 (scripted opponent_action)."""
    PLANET_CAP, FLEET_CAP = g["PLANET_CAP"], g["FLEET_CAP"]
    CENTER, RLIM = g["CENTER"], g["ROTATION_RADIUS_LIMIT"]
    EP = g["EPISODE_STEPS"]
    other = 1.0 - self_code
    me = int(_g(obs, "player", 0))
    planets = _g(obs, "planets", []) or []
    fleets = _g(obs, "fleets", []) or []
    angv = float(_g(obs, "angular_velocity", 0.0) or 0.0)
    comet_ids = set(int(c) for c in (_g(obs, "comet_planet_ids", []) or []))
    env.T = int(_g(config, "episodeSteps", EP) or EP)
    env.p_alive.zero_(); env.p_owner.fill_(-1.0)
    for t in (env.p_x, env.p_y, env.p_radius, env.p_ships, env.p_prod, env.p_is_comet, env.p_rotates):
        t.zero_()
    for p in planets:
        pid, owner, x, y, rad, ships, prod = p
        s = int(pid)
        if s >= PLANET_CAP:
            continue
        env.p_alive[0, s] = 1.0
        env.p_owner[0, s] = self_code if owner == me else (-1.0 if (owner is None or owner < 0) else other)
        env.p_x[0, s] = x; env.p_y[0, s] = y; env.p_radius[0, s] = rad
        env.p_ships[0, s] = ships; env.p_prod[0, s] = prod
        is_comet = s in comet_ids
        env.p_is_comet[0, s] = 1.0 if is_comet else 0.0
        rr = ((x - CENTER) ** 2 + (y - CENTER) ** 2) ** 0.5
        env.p_rotates[0, s] = 1.0 if ((not is_comet) and (rr + rad < RLIM)) else 0.0
    env.ang_vel[0] = angv
    env.f_alive.zero_(); env.f_owner.zero_()
    for t in (env.f_x, env.f_y, env.f_angle, env.f_ships, env.f_seq):
        t.zero_()
    for j, f in enumerate(fleets):
        if j >= FLEET_CAP:
            break
        fid, owner, x, y, angle, from_pid, ships = f
        env.f_alive[0, j] = 1.0
        env.f_owner[0, j] = self_code if owner == me else other
        env.f_x[0, j] = x; env.f_y[0, j] = y; env.f_angle[0, j] = angle
        env.f_ships[0, j] = ships; env.f_seq[0, j] = float(fid)
    env.step_ct[0] = float(_g(obs, "step", 0) or 0)


def _new_env(g):
    env = g["GpuEnv"](g["PLANET_CAP"], g["FLEET_CAP"], g["EPISODE_STEPS"], g["SHIP_SPEED"], torch.device("cpu"))
    env.reset([g["generate_world"](0)])               # allocate B=1 tensors once
    return env


def make_learner_agent(g, net):
    env_encode, act, _decode_target = g["env_encode"], g["act"], g["_decode_target"]
    env = _new_env(g)

    def agent(obs, config):
        _load_obs_into_env(g, env, obs, config, self_code=0.0)
        with torch.no_grad():
            ent, em, am, gl = env_encode(env, 0)
            a_t, _, _ = act(net, ent, em, am, gl, greedy=True)
            legal = (env.p_owner == 0.0) & (env.p_alive > 0.5) & (env.p_ships > 0.0)
            angle, ships, can, *_ = _decode_target(env, a_t, legal)
        idx = (can[0] > 0.5).nonzero(as_tuple=False)
        moves = []
        for s, d in idx.tolist():
            n = int(ships[0, s, d].item())
            if n > 0:
                moves.append([s, float(angle[0, s, d].item()), n])
        return moves

    return agent


def make_scripted_agent(g, code):
    opponent_action = g["opponent_action"]
    env = _new_env(g)

    def agent(obs, config):
        _load_obs_into_env(g, env, obs, config, self_code=1.0)   # scripted bot acts as owner 1
        with torch.no_grad():
            angle, ships, commit = opponent_action(env, code)    # (B,Ec) one launch / owned planet
        idx = (commit[0] > 0.5).nonzero(as_tuple=False).flatten()
        moves = []
        for s in idx.tolist():
            n = int(math.floor(float(ships[0, s].item())))
            if n > 0:
                moves.append([s, float(angle[0, s].item()), n])
        return moves

    return agent


# ---------------- parallel workers (process pool; kaggle engine is pure-Python CPU) -------------
_W = {}


def _init_worker(ckpt, nb, steps, opps):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch as _t
    _t.set_num_threads(1)                              # 1 thread/process; the pool fills the cores
    os.chdir(REPO)
    g = load_nb_defs(nb)
    net, _, _ = build_actor_v7(g, ckpt)
    scripted = {op: make_scripted_agent(g, OPP_CODE[op]) for op in opps if op in OPP_CODE}
    _W.update(g=g, learner=make_learner_agent(g, net), scripted=scripted, steps=steps)


def _play_game(task):
    opp, idx = task
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"episodeSteps": _W["steps"]}, debug=False)  # stock engine (comets on)
    oppa = _W["scripted"][opp]
    seat = idx % 2                                     # alternate seats for fairness
    roster = [_W["learner"], oppa] if seat == 0 else [oppa, _W["learner"]]
    out = env.run(roster)
    r = out[-1][seat]["reward"]
    res = "d" if (r is None or r == 0) else ("w" if r > 0 else "l")
    return (opp, res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="notebooks/weight/Transformers/it101.pt")
    ap.add_argument("--nb", default=NB_DEFAULT, help="v7 notebook with the matching defs")
    ap.add_argument("--games", type=int, default=60, help="games per opponent")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--opponents", default="starter,intermediate,greedy")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()
    os.chdir(REPO)
    from concurrent.futures import ProcessPoolExecutor

    opps = [o.strip() for o in args.opponents.split(",") if o.strip()]
    tasks = [(op, i) for op in opps for i in range(args.games)]
    print("ckpt %s | %d games/opp x %d = %d games | %d workers | STOCK kaggle engine (comets on)\n"
          % (os.path.basename(args.ckpt), args.games, len(opps), len(tasks), args.workers), flush=True)

    tally = {op: {"w": 0, "l": 0, "d": 0} for op in opps}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.ckpt, args.nb, args.steps, opps)) as ex:
        done = 0
        for op, res in ex.map(_play_game, tasks, chunksize=1):
            tally[op][res] += 1
            done += 1
            if done % 10 == 0:
                print("  ... %d/%d games" % (done, len(tasks)), flush=True)

    z = 1.96
    lines = ["\n%-12s | %-4s %-4s %-4s | win-rate (95%% CI)" % ("opp", "W", "L", "D"), "-" * 52]
    scores = []
    for op in opps:
        t = tally[op]; n = t["w"] + t["l"] + t["d"]
        wr = (t["w"] + 0.5 * t["d"]) / max(1, n)
        scores.append(wr)
        ci = z * math.sqrt(max(wr * (1 - wr), 1e-9) / max(1, n))
        lines.append("%-12s | %-4d %-4d %-4d | %.3f +/- %.3f" % (op, t["w"], t["l"], t["d"], wr, ci))
    gaunt = sum(scores) / len(scores) if scores else 0.0
    lines.append("-" * 52)
    lines.append("GAUNTLET AVG (official engine) = %.3f" % gaunt)
    out = "\n".join(lines)
    print(out)
    name = os.path.splitext(os.path.basename(args.ckpt))[0]
    od = os.path.join("runs", "viz", name); os.makedirs(od, exist_ok=True)
    open(os.path.join(od, "kaggle_v7_gauntlet.txt"), "w").write(
        "ckpt %s | %d games/opp | opponents=%s\n%s\n" % (args.ckpt, args.games, args.opponents, out))
    print("\nsaved ->", os.path.join(od, "kaggle_v7_gauntlet.txt"))


if __name__ == "__main__":
    main()
