"""Package a v9 MLP-trunk target-actor checkpoint (F_DIM 104, per-seat ONE-HOT ownership) as a
Kaggle orbit_wars submission.

Same proven structure as scripts/package_submission_v7.py (embed the notebook def cells + an
obs->GpuEnv adapter + a greedy agent(); tarball with main.py + sibling weights.pt), with TWO v9
differences:

  1. SEAT-PRESERVING adapter. v9 ownership is an ego-relative seat one-hot, so the adapter must
     keep DISTINCT opponent seats (p_owner = the ABSOLUTE player index, neutral = -1) and call
     env_encode(env, ego=me) -- NOT collapse every enemy to one id like v7. It also recovers the
     original player count N (monotonic running-max of owner indices, snapped to {2,4}; the engine
     assigns home planets to owners 0..N-1 at t=0 so all N are visible on turn 1) and sets
     env.n_players, because env_encode computes the seat channel as (owner-ego) mod n_players.

  2. FP32 weights. The v9 net is only ~4.9M params (19.8 MB fp32), far under Kaggle's ~100 MB cap,
     so we store the checkpoint's fp32 weights verbatim -- exact parity with the local eval, no
     fp16 round-trip, and fp32 is the native CPU path on the eval box anyway.

    python scripts/package_submission_v9.py --ckpt notebooks/weight/MLPs/it351.pt --self-test
    kaggle competitions submit -c orbit-wars -f submission_it351.tar.gz -m "v9 it351 MLP seat-aware"
"""
from __future__ import annotations
import argparse, io, json, os, sys, tarfile, tempfile
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DEFAULT = os.path.join("notebooks", "setup1_v9_a100.ipynb")
ARCH_KEYS = ("HIDDEN", "N_RES_BLOCKS", "D_G", "N_TX_LAYERS", "N_HEADS", "TX_MLP_RATIO",
             "N_STEM_RES", "N_HEAD_RES", "USE_GLU", "USE_ATTENTION", "VALUE_RES_BLOCKS", "ARCH",
             "PLANET_CAP", "F_DIM", "G_DIM")


def extract_defs_src(nb):
    """Concatenated notebook code cells up to (not incl.) the BC/train RUN cell."""
    doc = json.load(open(os.path.join(REPO, nb), encoding="utf-8"))

    def has_run(s):
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
    return "\n\n".join(parts)


# ---- runtime agent template (lands verbatim in main.py; {DEFS}/{CONFIG} filled in) --------------
MAIN_TEMPLATE = '''\
# AUTO-GENERATED Orbit Wars v9 submission (MLP trunk, per-seat one-hot, greedy). Do not edit by hand.
# Built by scripts/package_submission_v9.py. Loads sibling weights.pt (FP32).
import os, sys, math, types, contextlib, io
import torch
torch.set_num_threads(max(1, (os.cpu_count() or 2)))

# The notebook defs import matplotlib.pyplot (plotting cells); never used at act() time. ALWAYS
# stub it: the real matplotlib on the eval box is built vs NumPy 1.x and crashes under the box's
# NumPy 2.x (_ARRAY_API not found), so we must not import the real one.
if "matplotlib" not in sys.modules:
    _mpl = types.ModuleType("matplotlib"); _plt = types.ModuleType("matplotlib.pyplot")
    _mpl.pyplot = _plt; _mpl.use = lambda *a, **k: None
    sys.modules["matplotlib"] = _mpl; sys.modules["matplotlib.pyplot"] = _plt

_DEFS_SRC = {DEFS!r}
_CONFIG = {CONFIG!r}
_ARCH_KEYS = {ARCH_KEYS!r}

# exec the notebook defs into an isolated namespace (suppress its import-time prints)
g = {{"__name__": "__owsub__"}}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(_DEFS_SRC, "ow_v9_defs", "exec"), g)

DEVICE = torch.device("cpu")
g["DEVICE"] = DEVICE
for _k in _ARCH_KEYS:
    if _k in _CONFIG:
        g[_k] = _CONFIG[_k]

_net = g["build_policy"]()

def _find_weights():
    # Kaggle execs the agent via bare exec() with NO __file__, cwd != agent dir, so probe known
    # locations. The agent + weights.pt extract to /kaggle_simulations/agent on the box.
    cands = []
    try:
        cands.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    cands += ["/kaggle_simulations/agent", os.getcwd(),
              os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else "."]
    for _d in cands:
        _p = os.path.join(_d, "weights.pt")
        if os.path.exists(_p):
            return _p
    raise FileNotFoundError("weights.pt not found; searched %r" % cands)

_sd = torch.load(_find_weights(), map_location="cpu", weights_only=False)
_sd = _sd["model"] if isinstance(_sd, dict) and "model" in _sd else _sd
_sd = {{k: (v.float() if torch.is_floating_point(v) else v) for k, v in _sd.items()}}
_net.load_state_dict(_sd, strict=True)
_net = _net.to(DEVICE).eval()

PLANET_CAP, FLEET_CAP = g["PLANET_CAP"], g["FLEET_CAP"]
CENTER, RLIM, EP = g["CENTER"], g["ROTATION_RADIUS_LIMIT"], g["EPISODE_STEPS"]
env_encode, act, _decode_target = g["env_encode"], g["act"], g["_decode_target"]
_env = g["GpuEnv"](PLANET_CAP, FLEET_CAP, EP, g["SHIP_SPEED"], DEVICE)
_env.reset([g["generate_world"](0)])   # allocate B=1 tensors once

# original player count N, recovered online: monotonic running-max of owner indices snapped to
# {{2,4}}. The engine gives every player a home planet (owner 0..N-1) at t=0, so all N are present
# on turn 1; the max is monotone so a late-game eliminated seat never lowers N. Reset across
# episodes when the step counter goes backwards.
_NP = {{"n": 0, "last_step": 1 << 30}}

def _infer_n_players(step, me, owners):
    if step < _NP["last_step"]:
        _NP["n"] = 0                       # new episode
    _NP["last_step"] = step
    mx = me
    for o in owners:
        if o is not None and o >= 0 and o > mx:
            mx = o
    if mx + 1 > _NP["n"]:
        _NP["n"] = mx + 1
    return 4 if _NP["n"] > 2 else 2        # engine only runs 2 or 4 players


def _g(obs, key, default=None):
    try:
        return obs[key]
    except Exception:
        return getattr(obs, key, default)


def _load_obs(env, obs, config):
    """Map a Kaggle obs onto the persistent B=1 GpuEnv preserving ABSOLUTE seats: p_owner = the
    player index (neutral = -1), ego = obs.player. env_encode(env, ego) then builds the per-seat
    one-hot (owner-ego) mod n_players, so distinct opponents keep distinct channels."""
    me = int(_g(obs, "player", 0))
    planets = _g(obs, "planets", []) or []
    fleets = _g(obs, "fleets", []) or []
    angv = float(_g(obs, "angular_velocity", 0.0) or 0.0)
    comet_ids = set(int(c) for c in (_g(obs, "comet_planet_ids", []) or []))
    step = int(_g(obs, "step", 0) or 0)
    owners = [int(p[1]) for p in planets if p[1] is not None]
    owners += [int(f[1]) for f in fleets if f[1] is not None]
    env.n_players = _infer_n_players(step, me, owners)
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
        env.p_owner[0, s] = -1.0 if (owner is None or owner < 0) else float(int(owner))   # ABSOLUTE seat
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
        env.f_owner[0, j] = -1.0 if (owner is None or owner < 0) else float(int(owner))   # ABSOLUTE seat
        env.f_x[0, j] = x; env.f_y[0, j] = y; env.f_angle[0, j] = angle
        env.f_ships[0, j] = ships; env.f_seq[0, j] = float(fid)
    env.step_ct[0] = float(step)
    return me


def agent(obs, config):
    me = _load_obs(_env, obs, config)
    with torch.no_grad():
        ent, em, am, gl = env_encode(_env, me)
        a_t, _, _ = act(_net, ent, em, am, gl, greedy=True)
        legal = (_env.p_owner == float(me)) & (_env.p_alive > 0.5) & (_env.p_ships > 0.0)
        angle, ships, can, *_ = _decode_target(_env, a_t, legal)
    idx = (can[0] > 0.5).nonzero(as_tuple=False)
    moves = []
    for s, d in idx.tolist():
        n = int(ships[0, s, d].item())
        if n > 0:
            moves.append([s, float(angle[0, s, d].item()), n])
    return moves
'''


def build(ckpt, out, nb):
    defs = extract_defs_src(nb)
    blob = torch.load(os.path.join(REPO, ckpt), map_location="cpu", weights_only=False)
    cfg_full = blob.get("config") or {}
    cfg = {k: cfg_full[k] for k in ARCH_KEYS if k in cfg_full}
    main_py = MAIN_TEMPLATE.format(DEFS=defs, CONFIG=cfg, ARCH_KEYS=ARCH_KEYS)
    # fp32 weights verbatim (small model -> well under the size cap; exact parity with the eval)
    wbuf = io.BytesIO(); torch.save({"model": blob["model"]}, wbuf); wbytes = wbuf.getvalue()

    out_path = os.path.join(REPO, out)
    with tarfile.open(out_path, "w:gz", compresslevel=9) as tf:
        mi = tarfile.TarInfo("main.py"); mb = main_py.encode("utf-8")
        mi.size = len(mb); tf.addfile(mi, io.BytesIO(mb))
        wi = tarfile.TarInfo("weights.pt"); wi.size = len(wbytes); tf.addfile(wi, io.BytesIO(wbytes))
    return out_path, len(main_py), len(wbytes), cfg


# ---- seat-parity self-test: deploy-adapter encoding must bit-match the training encoding -------
def _emit_obs(g, env, b, me):
    """Build the Kaggle obs the engine would hand player `me` from GpuEnv batch row b (the board is
    shared & absolute across players, so this is just the env's own state with player=me)."""
    planets, fleets = [], []
    for s in range(env.Ec):
        if env.p_alive[b, s] <= 0.5:
            continue
        o = int(env.p_owner[b, s].item())
        planets.append([s, (o if o >= 0 else -1), float(env.p_x[b, s]), float(env.p_y[b, s]),
                        float(env.p_radius[b, s]), int(round(float(env.p_ships[b, s].item()))),
                        float(env.p_prod[b, s])])
    comet_ids = [s for s in range(env.Ec) if env.p_alive[b, s] > 0.5 and env.p_is_comet[b, s] > 0.5]
    for j in range(env.Fc):
        if env.f_alive[b, j] <= 0.5:
            continue
        o = int(env.f_owner[b, j].item())
        fleets.append([int(env.f_seq[b, j].item()), (o if o >= 0 else -1), float(env.f_x[b, j]),
                       float(env.f_y[b, j]), float(env.f_angle[b, j]),
                       0, int(round(float(env.f_ships[b, j].item())))])
    return {"player": me, "planets": planets, "fleets": fleets,
            "angular_velocity": float(env.ang_vel[b].item()), "comet_planet_ids": comet_ids,
            "step": int(env.step_ct[b].item())}


def parity_test(nb, ckpt):
    """Build a varied 4p board with the notebook engine, then for EACH ego seat 0..3: emit the
    Kaggle obs, push it through the deploy adapter, and assert env_encode matches the training
    encoding bit-for-bit (proves the seat one-hot maps every opponent to the right channel)."""
    import contextlib as _c
    defs = extract_defs_src(nb)
    g = {"__name__": "__owparity__"}
    with _c.redirect_stdout(io.StringIO()):
        exec(compile(defs, "ow_v9_defs", "exec"), g)
    blob = torch.load(os.path.join(REPO, ckpt), map_location="cpu", weights_only=False)
    for k in ARCH_KEYS:
        if k in (blob.get("config") or {}):
            g[k] = blob["config"][k]
    g["DEVICE"] = torch.device("cpu")
    GpuEnv, env_encode, generate_world = g["GpuEnv"], g["env_encode"], g["generate_world"]
    PLANET_CAP, FLEET_CAP, EP = g["PLANET_CAP"], g["FLEET_CAP"], g["EPISODE_STEPS"]

    torch.manual_seed(0)
    ref = GpuEnv(PLANET_CAP, FLEET_CAP, EP, g["SHIP_SPEED"], g["DEVICE"])
    ref.reset([generate_world(7)], n_players=4)
    # diversify ownership: random ego launches + 3 scripted seats for ~60 ticks
    seats = [{"pid": 1, "script": g["_OPP_BY_KIND"]["greedy"]},
             {"pid": 2, "script": g["_OPP_BY_KIND"]["starter"]},
             {"pid": 3, "script": g["_OPP_BY_KIND"]["intermediate"]}]
    with torch.no_grad():
        for t in range(60):
            rnd = torch.softmax(torch.randn(ref.B, PLANET_CAP, PLANET_CAP + 1), -1)
            g["env_step"](ref, rnd, seats=seats, step_idx=t)
    owners_present = sorted(set(int(o) for o in ref.p_owner[0].tolist() if o >= 0)
                            | set(int(o) for o in ref.f_owner[0].tolist() if o >= 0))
    n_alive = int((ref.p_alive[0] > 0.5).sum())
    print("[parity] built 4p board: %d alive planets, seats present=%s" % (n_alive, owners_present))

    # deploy adapter (exec main.py defs portion in-process; reuse the same g for the agent fns)
    deploy = dict(g)
    dl_env = GpuEnv(PLANET_CAP, FLEET_CAP, EP, g["SHIP_SPEED"], g["DEVICE"])
    dl_env.reset([generate_world(0)])
    # rebuild the adapter closures bound to dl_env via the template's _load_obs logic
    exec(compile(_ADAPTER_FN, "ow_v9_adapter", "exec"), deploy)
    _load = deploy["_load_obs_into"]

    ok = True
    for me in range(4):
        eref = [x.clone() for x in env_encode(ref, me)]      # training view for ego=me
        obs = _emit_obs(g, ref, 0, me)
        used_me, na = _load(dl_env, obs, {"episodeSteps": EP}, PLANET_CAP, FLEET_CAP,
                            g["CENTER"], g["ROTATION_RADIUS_LIMIT"], EP)
        edep = env_encode(dl_env, used_me)
        if na != 4:
            print("  ego=%d: FAIL n_players inferred=%d (want 4)" % (me, na)); ok = False; continue
        diffs = [float((a - b).abs().max()) for a, b in zip(edep, eref)]
        match = all(d < 1e-4 for d in diffs)
        # explicit check on the 4 ownership one-hot channels (body cols 7..10)
        oh_ref = eref[0][..., 7:11]; oh_dep = edep[0][..., 7:11]
        oh_ok = bool(torch.equal((oh_ref > 0.5), (oh_dep > 0.5)))
        print("  ego=%d: encode max|d|=%s  one-hot=%s  -> %s"
              % (me, ["%.1e" % d for d in diffs], "exact" if oh_ok else "MISMATCH",
                 "OK" if (match and oh_ok) else "FAIL"))
        ok = ok and match and oh_ok
    assert ok, "seat-parity FAILED -- deploy encoding diverges from training; DO NOT submit"
    print("[parity] PASS -- deploy seat one-hot matches training for all egos 0..3")


# standalone copy of the adapter body for the in-process parity test (mirrors main.py _load_obs)
_ADAPTER_FN = '''
_NP = {"n": 0, "last_step": 1 << 30}
def _infer_n_players(step, me, owners):
    if step < _NP["last_step"]:
        _NP["n"] = 0
    _NP["last_step"] = step
    mx = me
    for o in owners:
        if o is not None and o >= 0 and o > mx:
            mx = o
    if mx + 1 > _NP["n"]:
        _NP["n"] = mx + 1
    return 4 if _NP["n"] > 2 else 2

def _load_obs_into(env, obs, config, PLANET_CAP, FLEET_CAP, CENTER, RLIM, EP):
    me = int(obs["player"])
    planets = obs.get("planets", []) or []
    fleets = obs.get("fleets", []) or []
    comet_ids = set(int(c) for c in (obs.get("comet_planet_ids", []) or []))
    step = int(obs.get("step", 0) or 0)
    owners = [int(p[1]) for p in planets if p[1] is not None] + [int(f[1]) for f in fleets if f[1] is not None]
    na = _infer_n_players(step, me, owners)
    env.n_players = na
    env.T = int(config.get("episodeSteps", EP) or EP)
    env.p_alive.zero_(); env.p_owner.fill_(-1.0)
    for t in (env.p_x, env.p_y, env.p_radius, env.p_ships, env.p_prod, env.p_is_comet, env.p_rotates):
        t.zero_()
    for p in planets:
        pid, owner, x, y, rad, ships, prod = p
        s = int(pid)
        if s >= PLANET_CAP:
            continue
        env.p_alive[0, s] = 1.0
        env.p_owner[0, s] = -1.0 if (owner is None or owner < 0) else float(int(owner))
        env.p_x[0, s] = x; env.p_y[0, s] = y; env.p_radius[0, s] = rad
        env.p_ships[0, s] = ships; env.p_prod[0, s] = prod
        is_comet = s in comet_ids
        env.p_is_comet[0, s] = 1.0 if is_comet else 0.0
        rr = ((x - CENTER) ** 2 + (y - CENTER) ** 2) ** 0.5
        env.p_rotates[0, s] = 1.0 if ((not is_comet) and (rr + rad < RLIM)) else 0.0
    env.ang_vel[0] = float(obs.get("angular_velocity", 0.0) or 0.0)
    env.f_alive.zero_(); env.f_owner.zero_()
    for t in (env.f_x, env.f_y, env.f_angle, env.f_ships, env.f_seq):
        t.zero_()
    for j, f in enumerate(fleets):
        if j >= FLEET_CAP:
            break
        fid, owner, x, y, angle, from_pid, ships = f
        env.f_alive[0, j] = 1.0
        env.f_owner[0, j] = -1.0 if (owner is None or owner < 0) else float(int(owner))
        env.f_x[0, j] = x; env.f_y[0, j] = y; env.f_angle[0, j] = angle
        env.f_ships[0, j] = ships; env.f_seq[0, j] = float(fid)
    env.step_ct[0] = float(step)
    return me, na
'''


def self_test(out):
    """Extract the tarball and exec main.py EXACTLY as Kaggle does: bare exec(code, {}) with NO
    __file__ and cwd = agent dir. Run 2p AND 4p turns, report timing."""
    import time, statistics
    from kaggle_environments import make

    tmp = tempfile.mkdtemp(prefix="ow_v9_isol_")
    with tarfile.open(os.path.join(REPO, out), "r:gz") as tf:
        tf.extractall(tmp)
    saved = list(sys.path)
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") not in (REPO, os.getcwd())]
    cwd = os.getcwd(); os.chdir(tmp)
    try:
        src = open(os.path.join(tmp, "main.py"), encoding="utf-8").read()
        ns = {}
        t0 = time.perf_counter()
        exec(compile(src, "main.py", "exec"), ns)
        load_ms = (time.perf_counter() - t0) * 1000
        agent = ns["agent"]
        for n in (2, 4):
            env = make("orbit_wars", debug=False); env.reset(num_agents=n)
            obs = env.state[0]["observation"]; cfg = env.configuration
            worst = 0.0; nmoves = 0; durs = []
            for _ in range(40):
                t0 = time.perf_counter()
                mv = agent(dict(obs) if not isinstance(obs, dict) else obs, dict(cfg))
                dt = (time.perf_counter() - t0) * 1000; worst = max(worst, dt); nmoves = len(mv); durs.append(dt)
                env.step([mv] + [[] for _ in range(n - 1)])
                if env.done:
                    break
                obs = env.state[0]["observation"]
            p95 = statistics.quantiles(durs, n=20)[-1] if len(durs) >= 20 else max(durs)
            print("  [%d-player] %d turns | median %.0f ms | p95 %.0f ms | worst %.0f ms | last moves=%d"
                  % (n, len(durs), statistics.median(durs), p95, worst, nmoves))
            assert isinstance(mv, list), "agent() must return a list"
    finally:
        sys.path = saved; os.chdir(cwd)
    print("  import+build (cold) = %.0f ms | overage bank = 60 s" % load_ms)
    print("[self-test] PASS (bare-exec, no __file__ -- matches Kaggle)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="notebooks/weight/MLPs/it351.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nb", default=NB_DEFAULT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    os.chdir(REPO)
    if args.out is None:
        args.out = "submission_%s.tar.gz" % os.path.splitext(os.path.basename(args.ckpt))[0]
    path, main_sz, w_sz, cfg = build(args.ckpt, args.out, args.nb)
    total = os.path.getsize(path) / 1e6
    print("wrote %s (%.1f MB)  [main.py %.0f KB | weights fp32 %.1f MB raw]  cfg F_DIM=%s ARCH=%s"
          % (args.out, total, main_sz / 1e3, w_sz / 1e6, cfg.get("F_DIM"), cfg.get("ARCH")))
    if args.self_test:
        parity_test(args.nb, args.ckpt)
        self_test(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
