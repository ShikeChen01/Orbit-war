"""Package a v7 TRANSFORMER target-actor checkpoint as a Kaggle orbit_wars submission.

The v7 model (PolicyNet + transformer + GatedCat heads) lives in the notebook, not in the
`orbit_wars_rl` package, so the legacy `package_submission.py` cannot serve it. This packager
embeds the notebook's def cells (the exact source `eval_kaggle_v7.load_nb_defs` execs) + the
obs->GpuEnv adapter + a greedy `agent(obs, config)`, and stores the weights as FP16 (loaded back
to FP32 at runtime for CPU inference -- fp16 round-trip costs ~nothing for greedy argmax, verified
with eval_h2h_v7 --fp16-a, and roughly halves the file so it fits Kaggle's ~100 MB limit).

Output is a `.tar.gz` (entrypoint `main.py` + sibling `weights.pt`); fp32 it1041 is 196 MB which
no single base64 .py can hold under the limit.

    python scripts/package_submission_v7.py --ckpt notebooks/weight/Transformers/it1041.pt --self-test
    kaggle competitions submit -c orbit-wars -f submission_it1041.tar.gz -m "v7 it1041 transformer"
"""
from __future__ import annotations
import argparse, io, json, os, sys, tarfile, tempfile, textwrap
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DEFAULT = os.path.join("notebooks", "legacy", "setup1_v7_a100.ipynb")
ARCH_KEYS = ("HIDDEN", "N_RES_BLOCKS", "D_G", "N_TX_LAYERS", "N_HEADS", "TX_MLP_RATIO",
             "N_STEM_RES", "N_HEAD_RES", "USE_GLU", "USE_ATTENTION", "VALUE_RES_BLOCKS", "ARCH",
             "PLANET_CAP", "F_DIM", "G_DIM", "DEST_HEAD")


def extract_defs_src(nb):
    """The concatenated notebook code cells up to (not incl.) the BC/train RUN -- identical to
    eval_kaggle_v7.load_nb_defs, but returned as source text to embed."""
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


# ---- the runtime agent: exec the embedded defs, build the net, serve greedy moves ----------------
# (kept as a string template so it lands verbatim in main.py; {DEFS} / {CONFIG} are filled in)
MAIN_TEMPLATE = '''\
# AUTO-GENERATED Orbit Wars v7 submission (transformer target-actor, greedy). Do not edit by hand.
# Built by scripts/package_submission_v7.py. Loads sibling weights.pt (FP16 -> FP32 at runtime).
import os, sys, math, types, contextlib, io
import torch
torch.set_num_threads(max(1, (os.cpu_count() or 2)))

# The notebook defs import matplotlib.pyplot (plotting cells); never used at act() time. ALWAYS
# stub it: the real matplotlib on the eval box is compiled against NumPy 1.x and crashes under the
# box's NumPy 2.x (_ARRAY_API not found), so we must not import the real one.
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
    exec(compile(_DEFS_SRC, "ow_v7_defs", "exec"), g)

DEVICE = torch.device("cpu")
g["DEVICE"] = DEVICE
for _k in _ARCH_KEYS:
    if _k in _CONFIG:
        g[_k] = _CONFIG[_k]

_net = g["build_policy"]()

def _find_weights():
    # Kaggle execs the agent via bare exec() with NO __file__, and the cwd is not the agent dir,
    # so probe the known locations. The agent + weights.pt live in /kaggle_simulations/agent on the box.
    cands = []
    try:
        cands.append(os.path.dirname(os.path.abspath(__file__)))   # local importlib / direct run
    except NameError:
        pass
    cands += ["/kaggle_simulations/agent", os.getcwd(), os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else "."]
    for _d in cands:
        _p = os.path.join(_d, "weights.pt")
        if os.path.exists(_p):
            return _p
    raise FileNotFoundError("weights.pt not found; searched %r" % cands)

_WP = _find_weights()
_sd = torch.load(_WP, map_location="cpu", weights_only=False)
_sd = _sd["model"] if isinstance(_sd, dict) and "model" in _sd else _sd
_sd = {{k: (v.float() if torch.is_floating_point(v) else v) for k, v in _sd.items()}}
_net.load_state_dict(_sd, strict=True)
_net = _net.to(DEVICE).eval()

PLANET_CAP, FLEET_CAP = g["PLANET_CAP"], g["FLEET_CAP"]
CENTER, RLIM, EP = g["CENTER"], g["ROTATION_RADIUS_LIMIT"], g["EPISODE_STEPS"]
env_encode, act, _decode_target = g["env_encode"], g["act"], g["_decode_target"]
_env = g["GpuEnv"](PLANET_CAP, FLEET_CAP, EP, g["SHIP_SPEED"], DEVICE)
_env.reset([g["generate_world"](0)])   # allocate B=1 tensors once


def _g(obs, key, default=None):
    try:
        return obs[key]
    except Exception:
        return getattr(obs, key, default)


def _load_obs(env, obs, config):
    """Map a Kaggle obs onto the persistent B=1 GpuEnv as owner 0 (learner ego=0); every other
    owner (incl. the other players in a 4-player game) collapses to owner 1 (enemy)."""
    other = 1.0
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
        env.p_owner[0, s] = 0.0 if owner == me else (-1.0 if (owner is None or owner < 0) else other)
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
        env.f_owner[0, j] = 0.0 if owner == me else other
        env.f_x[0, j] = x; env.f_y[0, j] = y; env.f_angle[0, j] = angle
        env.f_ships[0, j] = ships; env.f_seq[0, j] = float(fid)
    env.step_ct[0] = float(_g(obs, "step", 0) or 0)


def agent(obs, config):
    _load_obs(_env, obs, config)
    with torch.no_grad():
        ent, em, am, gl = env_encode(_env, 0)
        a_t, _, _ = act(_net, ent, em, am, gl, greedy=True)
        legal = (_env.p_owner == 0.0) & (_env.p_alive > 0.5) & (_env.p_ships > 0.0)
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
    # fp16 weights (float tensors only; ints/bools kept)
    sd16 = {k: (v.half() if torch.is_floating_point(v) else v) for k, v in blob["model"].items()}

    main_py = MAIN_TEMPLATE.format(DEFS=defs, CONFIG=cfg, ARCH_KEYS=ARCH_KEYS)
    wbuf = io.BytesIO(); torch.save({"model": sd16}, wbuf); wbytes = wbuf.getvalue()

    out_path = os.path.join(REPO, out)
    with tarfile.open(out_path, "w:gz", compresslevel=9) as tf:
        mi = tarfile.TarInfo("main.py"); mb = main_py.encode("utf-8")
        mi.size = len(mb); tf.addfile(mi, io.BytesIO(mb))
        wi = tarfile.TarInfo("weights.pt"); wi.size = len(wbytes); tf.addfile(wi, io.BytesIO(wbytes))
    return out_path, len(main_py), len(wbytes)


def self_test(out):
    """Extract the tarball into an isolated temp dir and exec main.py EXACTLY as Kaggle does:
    bare `exec(code, env)` with NO __file__ defined and the cwd set to the agent dir (mirrors
    kaggle_environments.agent.get_last_callable). Then run full turns and report timing."""
    import time, statistics
    from kaggle_environments import make

    tmp = tempfile.mkdtemp(prefix="ow_v7_isol_")
    with tarfile.open(os.path.join(REPO, out), "r:gz") as tf:
        tf.extractall(tmp)
    saved = list(sys.path)
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") not in (REPO, os.getcwd())]
    cwd = os.getcwd(); os.chdir(tmp)
    try:
        src = open(os.path.join(tmp, "main.py"), encoding="utf-8").read()
        ns = {}                                            # NO __file__, NO __name__ -- like Kaggle
        t0 = time.perf_counter()
        exec(compile(src, "main.py", "exec"), ns)          # bare exec (the failure mode we hit)
        load_ms = (time.perf_counter() - t0) * 1000
        agent = ns["agent"]

        for n in (2, 4):
            env = make("orbit_wars", debug=False); env.reset(num_agents=n)
            obs = env.state[0]["observation"]; cfg = env.configuration
            worst = 0.0; nmoves = 0; durs = []
            for _ in range(40):                            # 40 turns -> steady-state p95
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
    print("  import+build (cold, lazy on 1st act on Kaggle) = %.0f ms | overage bank = 60 s" % load_ms)
    print("[self-test] PASS (bare-exec, no __file__ -- matches Kaggle)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="notebooks/weight/Transformers/it1041.pt")
    ap.add_argument("--out", default="submission_it1041.tar.gz")
    ap.add_argument("--nb", default=NB_DEFAULT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    os.chdir(REPO)
    path, main_sz, w_sz = build(args.ckpt, args.out, args.nb)
    total = os.path.getsize(path) / 1e6
    print("wrote %s (%.1f MB)  [main.py %.0f KB | weights fp16 %.1f MB raw]"
          % (args.out, total, main_sz / 1e3, w_sz / 1e6))
    if args.self_test:
        self_test(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
