"""Package the ``orbit_wars_v12`` Python package into a single self-contained notebook.

The PACKAGE is the source of truth. This script FLATTENS ``orbit_wars_v12/*.py`` into one
notebook so the notebook never has to be hand-edited (and so Claude can keep reasoning about the
small package files instead of one giant notebook):

  * external imports (torch/numpy/stdlib) are de-duplicated into ONE shared imports cell;
  * intra-package relative imports (``from .env import ...``) are DROPPED -- the flattened modules
    all share a single global namespace, so the names resolve directly (verified: no top-level
    name collisions across modules);
  * each module body is emitted verbatim as its own code cell, in topological order (a module only
    ever imports from earlier ones), preceded by a markdown cell carrying its docstring;
  * ``from __future__`` imports are re-prepended to the cell that needs them (cells compile
    independently, so this must be the first line of that cell);
  * finally the run-driver cells (settings -> optional BC warm-start -> train -> save league ->
    plot) call the package's PUBLIC API (``Config`` / ``train`` / ``save_league_agents`` /
    ``plot_training_health``), mirroring the notebook workflow.

Workflow: edit ``orbit_wars_v12/*.py``, then regenerate::

    python scripts/make_v12_nb.py            # regenerate notebooks/setup1_v12_a100.ipynb
    python scripts/make_v12_nb.py --check    # ...then exec it on CPU (SMOKE) to prove it runs
    python scripts/make_v12_nb.py -o x.ipynb # write somewhere else
"""
import argparse
import ast
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.join(REPO, "orbit_wars_v12")
DEFAULT_OUT = os.path.join(REPO, "notebooks", "setup1_v12_a100.ipynb")

# Topological order: every module imports only from modules earlier in this list (verified against
# the import graph). Flattening in this order means every top-level name a module references is
# already defined when its cell runs.
MODULE_ORDER = ["config", "constants", "distributions", "worldgen", "policy", "env",
                "checkpoint", "ppo", "rollout", "bc", "league", "train", "plotting"]

HEADER_MD = """# Orbit Wars -- v12 A100 run  *(GENERATED -- do not edit cells here)*

This notebook is **generated from the `orbit_wars_v12/` package** by `scripts/make_v12_nb.py`.
The package is the single source of truth: **edit the modules under `orbit_wars_v12/`, then run
`python scripts/make_v12_nb.py` to regenerate this notebook.** Hand-edits to the cells below are
overwritten on the next regen.

Layout: one shared **imports** cell, then **one cell per module** (relative imports stripped --
every definition shares one global namespace, exactly as the package does once imported), then the
**run** cells (settings -> optional BC warm-start -> train -> save league -> plot) that call the
package's public API."""

SETTINGS_CELL = """# ============================= RUN SETTINGS =============================
# The ONLY knobs to edit by hand. Everything above is generated from the package --
# to change model/env/training behaviour, edit orbit_wars_v12/*.py and regenerate.
SMOKE       = False     # True -> tiny net + few iters (fast CPU/GPU sanity run)
RESUME_FROM = None      # path to a *_train_state.pt to resume from, else None
OVERRIDES   = dict()    # any Config field, e.g. dict(TOTAL_ITERS=800, BC_ENABLED=True)

cfg = Config.create(smoke=SMOKE, **OVERRIDES)
print("device=%s  SMOKE=%s  ARCH=%s  HIDDEN=%d  ITERS=%d  B=%d  4p=%s"
      % (cfg.device, cfg.SMOKE, cfg.ARCH, cfg.HIDDEN, cfg.TOTAL_ITERS, cfg.B, cfg.FOURP_ENABLED))"""

BC_CELL = """# Optional BC warm-start: clone the medium bot into the gated-alloc heads (off unless
# cfg.BC_ENABLED). On a Run-All, the train cell then warm-starts from the saved BC init.
if cfg.BC_ENABLED:
    bc_pretrain(cfg)
    if RESUME_FROM is None:
        RESUME_FROM = cfg.BC_CKPT_PATH"""

TRAIN_CELL = ("net, hist, league = train(cfg, total_iters=cfg.TOTAL_ITERS, log_every=1, "
              "resume_from=RESUME_FROM)")

SAVE_CELL = "save_league_agents(cfg, league)"
PLOT_CELL = "plot_training_health(hist)"


def _span(node):
    """1-indexed inclusive line range a top-level node occupies (incl. multi-line imports)."""
    return range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1)


def process_module(name):
    """Return (docstring, [external import statements], body-without-imports-or-docstring)."""
    src = open(os.path.join(PKG_DIR, name + ".py"), encoding="utf-8").read()
    lines = src.splitlines()
    tree = ast.parse(src)
    drop = set()                       # 1-indexed line numbers to remove from the emitted body
    externals, futures = [], []

    doc = ast.get_docstring(tree, clean=False)
    if doc is not None and tree.body and isinstance(tree.body[0], ast.Expr):
        drop.update(_span(tree.body[0]))

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            futures.append(ast.get_source_segment(src, node)); drop.update(_span(node))
        elif isinstance(node, ast.ImportFrom) and (node.level or 0) > 0:
            drop.update(_span(node))                                   # relative -> drop entirely
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            externals.append(ast.get_source_segment(src, node)); drop.update(_span(node))

    body = "\n".join(l for i, l in enumerate(lines, 1) if i not in drop).strip("\n")
    if futures:
        body = "\n".join(futures) + "\n" + body
    return doc, externals, body


def _src_lines(text):
    """nbformat 'source' wants a list of lines, each terminated by '\\n' except the last."""
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src_lines(text)}


def _code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src_lines(text)}


def build_notebook():
    seen, imports = set(), []
    bodies = {}
    for name in MODULE_ORDER:
        doc, exts, body = process_module(name)
        bodies[name] = (doc, body)
        for e in exts:
            if e not in seen:
                seen.add(e); imports.append(e)

    cells = [_md(HEADER_MD), _md("## Imports"), _code("\n".join(imports))]
    for name in MODULE_ORDER:
        doc, body = bodies[name]
        head = "## `orbit_wars_v12.%s`" % name
        if doc:
            head += "\n\n" + doc.strip()
        cells += [_md(head), _code(body)]

    cells += [_md("## Run training"), _code(SETTINGS_CELL), _code(BC_CELL), _code(TRAIN_CELL),
              _md("## Save every league agent as weights"), _code(SAVE_CELL),
              _md("## Plot training-health curves"), _code(PLOT_CELL)]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def check(nb_path):
    """Exec the generated notebook on CPU (SMOKE) and assert it actually runs + evicts correctly."""
    import random
    import sys
    import types

    import torch

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OW_CKPT_DIR", os.path.join(REPO, "runs", "v12_nbcheck"))
    os.makedirs(os.environ["OW_CKPT_DIR"], exist_ok=True)
    os.chdir(REPO)

    doc = json.load(open(nb_path, encoding="utf-8"))
    # Register the exec namespace as a real module: a real notebook cell runs as `__main__`
    # (in sys.modules), which `from __future__ import annotations` + @dataclass needs to resolve
    # string annotations. A bare dict with __name__ not in sys.modules would crash the harness.
    mod = types.ModuleType("nbcheck")
    sys.modules["nbcheck"] = mod
    ns = mod.__dict__
    ns["__name__"] = "nbcheck"
    for c in doc["cells"]:                                # exec DEFINITION cells (stop at settings)
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "RUN SETTINGS" in src:
            break
        exec(compile(src, "nbcell", "exec"), ns)

    assert ns["F_DIM"] == 104, "flatten lost the one-hot obs (F_DIM != 104): %r" % ns.get("F_DIM")

    # tiny end-to-end train through the flattened defs (exercises rollout/ppo/league/recal/evict)
    cfg = ns["Config"].create(
        smoke=True, TOTAL_ITERS=4, EPISODE_STEPS=64, N_WORLDS=32, B=8, B_4P=8, FLEET_CAP=256,
        ELO_RECAL_ENVS=8, SELFPLAY_REFRESH=2, LEAGUE_MAX_SNAPSHOTS=3, LEARNER_RECAL_EVERY=2,
        N_HEADS=8, N_TX_LAYERS=2, N_HEAD_RES=1, device=torch.device("cpu"))   # CPU: fast, portable
    assert cfg.device.type == "cpu"
    net, hist, league = ns["train"](cfg, total_iters=4, log_every=1)
    assert hist["fmt"].count(2) >= 1 and hist["fmt"].count(4) >= 1, "expected both 2p and 4p iters"
    assert len(hist["learner_elo"]) == 4
    print("[check] tiny train ran: %d iters, league has %d members" % (len(hist["fmt"]), len(league.members)))

    # measured eviction semantics on the flattened League (stub the cfg-style eval funcs)
    o2, o4 = ns["_eval_score"], ns["_eval_winrate4"]
    F = {"starter": 0.99, "greedy": 0.99, "medium": 0.10}
    ns["_eval_score"] = lambda cfg, net, kind, worlds, n: F.get(kind, 0.5)
    ns["_eval_winrate4"] = lambda cfg, net, kind, worlds, n: F.get(kind, 0.5)
    mk = lambda kind, **kw: dict(net=None, kind=kind, label=kind, anchor=False, elo=1000.0,
                                 elo4=1000.0, n=0, n4=0, **kw)
    m_un, m_ma, m_wd = mk("medium"), mk("starter"), mk("greedy", wd_active=True)
    league.members = [m_un, m_ma, m_wd]
    league.evict_mastered_anchors(net, None, 8)
    assert m_un in league.members, "unmastered script wrongly evicted"
    assert m_ma not in league.members, "mastered ordinary script NOT removed"
    assert m_wd in league.members and m_wd["wd_active"] is False, "watchdog must retire to dormant"
    ns["_eval_score"], ns["_eval_winrate4"] = o2, o4
    print("[check] measured eviction OK: unmastered survives | mastered removed | watchdog->dormant")
    print("\nCHECK OK: generated notebook execs, trains, and evicts faithfully.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="output .ipynb path")
    ap.add_argument("--check", action="store_true", help="exec the generated nb (SMOKE) to verify")
    args = ap.parse_args()

    nb = build_notebook()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote %s  (%d cells from %d modules)"
          % (os.path.relpath(args.out, REPO), len(nb["cells"]), len(MODULE_ORDER)))

    if args.check:
        check(args.out)


if __name__ == "__main__":
    main()
