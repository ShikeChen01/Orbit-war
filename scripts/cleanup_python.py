"""Delete superseded Python now that training/eval run natively (ow_train_grpo / ow_eval / ow_train).

DRY-RUN BY DEFAULT: prints the plan and an import-safety check, deletes nothing. Pass --yes to
remove the SAFE set. REVIEW items are listed but never auto-deleted -- remove them by hand after
checking the noted caveat. Run from the repo root:

    python scripts/cleanup_python.py            # show the plan (safe)
    python scripts/cleanup_python.py --yes       # delete the SAFE set

What stays and why (the native loop demoted Python to: submission, world-gen, parity oracle):
  submission.py, scripts/search_agent.py ............ the Kaggle submission (pure Python)
  orbit_wars_rl/{py_engine,native_ckpt}.py .......... submission forward-model + .owc reader
  orbit_wars_rl/{agents,processors,env}/ ............ serving + obs/action + engine wrapper
  orbit_wars_rl/native_worldgen.py, gen_world_pool.py  one-time .owp pool generation
  scripts/bc_pretrain.py ............................ BC warm-start (GRPO init + KL reference)
  tests/, REFERENCE_orbit_wars.py ................... parity oracle
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (relative path, reason). Files/dirs that are dead now that the native exes exist.
SAFE_DELETE: list[tuple[str, str]] = [
    ("docs/archive/python-training-stack",
     "archived pure-Python gym + PPO stack; fully superseded by the native C++ trainer"),
    ("scripts/eval_native.py",
     "superseded by the ow_eval executable (native arena eval)"),
    ("scripts/sweep_native.py",
     "superseded; hyperparameter sweeps now re-run ow_train_grpo with different flags"),
]

# Listed for the user to remove manually -- each has a caveat, so NOT auto-deleted.
REVIEW: list[tuple[str, str]] = [
    ("scripts/train_native.py",
     "old Python->C++ training driver, superseded by ow_train_grpo -- BUT tests/test_native.py "
     "imports its export_checkpoint(); move that helper or drop the test first"),
    ("scripts/inspect_agent.py",
     "Python behavior-trace tool; keep only if you still debug agents in Python"),
    ("scripts/kaggle_probe_agent.py",
     "one-off hardware-probe submission; keep for reference or delete"),
    ("scripts/play_episode.py",
     "plays a policy in the REAL kaggle engine -- handy for final submission validation"),
]


def _module_names(rel: str) -> list[str]:
    """Importable module name(s) a path could be referenced by (best-effort)."""
    p = Path(rel)
    if p.suffix == ".py":
        return [p.stem]  # e.g. scripts/eval_native.py -> "eval_native"
    return []


def _iter_py_files(skip: set[Path]):
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dp = Path(dirpath)
        if any(part in {".venv", ".git", "_bld", "build", "__pycache__"} for part in dp.parts):
            continue
        if any(dp == s or s in dp.parents for s in skip):
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                fp = dp / fn
                if not any(fp == s for s in skip):
                    yield fp


def _find_importers(targets: list[str]) -> dict[str, list[str]]:
    """Map each module name -> list of files (outside the delete set) that import it."""
    skip = {ROOT / rel for rel, _ in SAFE_DELETE}
    mods = {m for rel, _ in SAFE_DELETE for m in _module_names(rel)}
    hits: dict[str, list[str]] = {m: [] for m in mods}
    if not mods:
        return hits
    for fp in _iter_py_files(skip):
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in mods:
            if f"import {m}" in text or f"from {m}" in text or f".{m} import" in text:
                hits[m].append(str(fp.relative_to(ROOT)))
    return {m: v for m, v in hits.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yes", action="store_true", help="actually delete the SAFE set")
    args = ap.parse_args()

    print("=" * 78)
    print("SAFE TO DELETE (superseded by the native exes):")
    existing = []
    for rel, why in SAFE_DELETE:
        path = ROOT / rel
        mark = "" if path.exists() else "   [already gone]"
        print(f"  - {rel}{mark}\n      {why}")
        if path.exists():
            existing.append((rel, path))

    print("\nREVIEW MANUALLY (a caveat -- not auto-deleted):")
    for rel, why in REVIEW:
        path = ROOT / rel
        mark = "" if path.exists() else "   [already gone]"
        print(f"  - {rel}{mark}\n      {why}")

    # Import-safety: never delete something the rest of the tree still imports.
    importers = _find_importers([])
    print("\nIMPORT-SAFETY CHECK:")
    blocked = set()
    if importers:
        for mod, files in importers.items():
            print(f"  !! '{mod}' is imported by: {', '.join(files)} -> will SKIP its file")
            blocked.add(mod)
    else:
        print("  ok -- no kept file imports a safe-delete target")

    if not args.yes:
        print("\nDRY RUN. Re-run with --yes to delete the SAFE set above.")
        return 0

    print("\nDELETING:")
    for rel, path in existing:
        if any(m in blocked for m in _module_names(rel)):
            print(f"  SKIP {rel} (still imported)")
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"  removed {rel}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
