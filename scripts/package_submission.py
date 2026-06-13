"""Build a single self-contained Kaggle submission file.

The Kaggle Orbit Wars eval box is CPU-only, has no compiler, and cannot load the native
`.pyd` -- so the submission must be pure Python with its dependencies bundled. This packager
embeds the (pure-Python) `orbit_wars_rl` package and a trained checkpoint as base64+gzip,
reconstructs them to a temp dir at import time, and exposes `agent(obs, config)` via the raw
`PolicyAgent` (greedy). No `_native`, no search -> no timeout risk; ~55% vs starter, 99.5% vs
random for `runs/grpo/win1/best.pt`.

    python scripts/package_submission.py --ckpt runs/grpo/win1/best.pt --out submission_kaggle.py

Validate the output in isolation (no repo on sys.path) before submitting -- see the
`--self-test` flag, which imports the generated file from a temp cwd and runs one turn.
"""
from __future__ import annotations

import argparse
import base64
import glob
import gzip
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _b64gz(data: bytes) -> str:
    return base64.b64encode(gzip.compress(data, 9)).decode("ascii")


_HEADER = '''\
# AUTO-GENERATED self-contained Orbit Wars submission -- do not edit by hand.
# Built by scripts/package_submission.py. Pure Python: embeds the orbit_wars_rl package +
# a trained checkpoint (base64+gzip), reconstructs them to a temp dir, and serves the raw
# greedy PolicyAgent. No native .pyd, no search (no timeout risk).
import base64
import gzip
import os
import sys
import tempfile
import types

# gymnasium is only referenced by EntityObservation.build_space (never at act() time); on a
# box without it, install a minimal stub so the import succeeds.
try:
    import gymnasium  # noqa: F401
except Exception:
    _g = types.ModuleType("gymnasium")
    _sp = types.ModuleType("gymnasium.spaces")

    class _Box:  # noqa: D401
        def __init__(self, *a, **k):
            pass

    class _Dict:
        def __init__(self, *a, **k):
            pass

    _sp.Box = _Box
    _sp.Dict = _Dict
    _g.spaces = _sp
    sys.modules["gymnasium"] = _g
    sys.modules["gymnasium.spaces"] = _sp
'''

_FOOTER = '''\
_DIR = tempfile.mkdtemp(prefix="ow_submission_")
for _rel, _b in _FILES.items():
    _p = os.path.join(_DIR, _rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(_p), exist_ok=True)
    with open(_p, "wb") as _f:
        _f.write(gzip.decompress(base64.b64decode(_b)))
_WP = os.path.join(_DIR, "weights.pt")
with open(_WP, "wb") as _f:
    _f.write(gzip.decompress(base64.b64decode(_WEIGHTS)))
sys.path.insert(0, _DIR)

from orbit_wars_rl.agents.base import _as_dict  # noqa: E402
from orbit_wars_rl.agents.ppo_policy import PolicyAgent  # noqa: E402

_AGENT = PolicyAgent.load(_WP, device="cpu")


def agent(obs, config):
    return _AGENT.act(_as_dict(obs), _as_dict(config))
'''


def build(ckpt: str, out: str) -> str:
    files = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "orbit_wars_rl", "**", "*.py"), recursive=True)):
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        with open(p, "rb") as f:
            files[rel] = _b64gz(f.read())
    with open(os.path.join(ROOT, ckpt), "rb") as f:
        weights = _b64gz(f.read())

    parts = [_HEADER, "\n_FILES = {\n"]
    for rel, b in files.items():
        parts.append(f'    {rel!r}: {b!r},\n')
    parts.append("}\n\n")
    parts.append(f"_WEIGHTS = {weights!r}\n\n")
    parts.append(_FOOTER)
    text = "".join(parts)
    out_path = os.path.join(ROOT, out)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return out_path


def self_test(out: str) -> None:
    """Import the generated file from an isolated cwd (repo NOT on sys.path) and run a turn."""
    import importlib.util
    import shutil
    import sys
    import tempfile
    import time

    # Generate a real board via the official engine (available on the eval box too).
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    env.reset(num_agents=2)
    obs = env.state[0]["observation"]
    cfg = env.configuration

    tmp = tempfile.mkdtemp(prefix="ow_isol_")
    shutil.copy(os.path.join(ROOT, out), os.path.join(tmp, "sub.py"))
    saved = list(sys.path)
    # Isolate: drop repo root + cwd so the embedded package is the only orbit_wars_rl.
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") not in (ROOT, os.getcwd())]
    sys.path.insert(0, tmp)
    for m in list(sys.modules):
        if m == "orbit_wars_rl" or m.startswith("orbit_wars_rl."):
            del sys.modules[m]
    try:
        spec = importlib.util.spec_from_file_location("sub", os.path.join(tmp, "sub.py"))
        mod = importlib.util.module_from_spec(spec)
        t0 = time.perf_counter()
        spec.loader.exec_module(mod)
        load_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        mv = mod.agent(dict(obs), dict(cfg))
        turn_ms = (time.perf_counter() - t0) * 1000
        from orbit_wars_rl import _native as _n  # noqa  -- only to report if it leaked in
        leaked = "orbit_wars_rl._native" in sys.modules
    except ImportError:
        leaked = False
    finally:
        pass
    sys.path = saved
    print(f"[self-test] import OK in {load_ms:.0f} ms; agent() returned {len(mv)} moves "
          f"in {turn_ms:.0f} ms; _native loaded during run = {leaked}")
    assert isinstance(mv, list), "agent() must return a list of moves"
    assert turn_ms < 1000, "turn exceeded the 1s budget"
    print("[self-test] PASS")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/grpo/win1/best.pt")
    ap.add_argument("--out", default="submission_kaggle.py")
    ap.add_argument("--self-test", action="store_true", help="import in isolation + run one turn")
    args = ap.parse_args()
    path = build(args.ckpt, args.out)
    sz = os.path.getsize(path) / 1e6
    print(f"wrote {args.out} ({sz:.2f} MB) from {args.ckpt}")
    if args.self_test:
        self_test(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
