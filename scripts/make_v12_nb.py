"""Package the ``orbit_wars_v12`` Python package into a single self-contained notebook.

The PACKAGE is the source of truth. This script FLATTENS ``orbit_wars_v12/*.py`` into one
notebook so the notebook never has to be hand-edited (and so Claude can keep reasoning about the
small package files instead of one giant notebook):

  * external imports (torch/numpy/stdlib) are de-duplicated into ONE shared imports cell;
  * intra-package relative imports (``from .env import ...``) are DROPPED -- the flattened modules
    all share a single global namespace, so the names resolve directly. That no two cells ever
    define the same top-level name (so no cell can silently overwrite another) is ENFORCED on every
    build by :func:`_assert_no_cell_overwrites`, which raises if any clash is introduced;
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
MCTS_OUT = os.path.join(REPO, "notebooks", "setup1_v12_mcts_a100.ipynb")
GRPO_OUT = os.path.join(REPO, "notebooks", "setup1_v12_grpo_a100.ipynb")
GRPO_SMOKE_OUT = os.path.join(REPO, "notebooks", "setup1_v12_grpo_anticollapse.ipynb")

# Topological order: every module imports only from modules earlier in this list (verified against
# the import graph). Flattening in this order means every top-level name a module references is
# already defined when its cell runs.
# Base v12 (no MCTS). The optional `mcts` module is inserted before `plotting` when --mcts is
# passed (it depends only on env+policy, both earlier; nothing depends on it).
BASE_MODULES = ["config", "constants", "distributions", "worldgen", "policy", "env",
                "checkpoint", "ppo", "rollout", "bc", "league", "train", "plotting"]
# `grpo` is inserted right after `ppo` (it depends only on policy.evaluate + ppo.policy_surrogate,
# both earlier) when --grpo is passed; `train` references grpo_update only under cfg.ALGO=="grpo".

HEADER_MD = """# Orbit Wars -- v12 A100 run  *(GENERATED -- do not edit cells here)*

This notebook is **generated from the `orbit_wars_v12/` package** by `scripts/make_v12_nb.py`.
The package is the single source of truth: **edit the modules under `orbit_wars_v12/`, then run
`python scripts/make_v12_nb.py` to regenerate this notebook.** Hand-edits to the cells below are
overwritten on the next regen.

Layout: one shared **imports** cell, then **one cell per module** (relative imports stripped --
every definition shares one global namespace, exactly as the package does once imported), then the
**run** cells (settings -> optional BC warm-start -> train -> save league -> plot) that call the
package's public API."""

# Two run presets baked into the settings cell (the ONLY hand-edited cell). The base v12 nb gets
# BIG_SETTINGS; the --mcts nb gets SMALL_SETTINGS. SMOKE=True ignores OVERRIDES (tiny sanity run).
_SETTINGS_PRINT = '''cfg = Config.create(smoke=SMOKE, **(OVERRIDES if not SMOKE else {}))
print("device=%s  SMOKE=%s  ARCH=%s  HIDDEN=%d  ITERS=%d  B=%d  4p=%s"
      % (cfg.device, cfg.SMOKE, cfg.ARCH, cfg.HIDDEN, cfg.TOTAL_ITERS, cfg.B, cfg.FOURP_ENABLED))'''

BIG_SETTINGS = """# ============================= RUN SETTINGS =============================
# PRESET: BIG transformer, NO MCTS -- the raw-policy leaderboard lineage.
# GPU: Colab H100 (native bf16). Compute-bound: deep transformer + PPO+GAE self-play league with
# the advanced dual-Elo + MEASURED eviction (on start AND every recal -- v12 default).
# Colab gotcha: if a torch<->triton crash hits, `pip install triton==3.6.0` then RESTART runtime.
# Submission: ~43M params -> 173MB fp32 (OVER the 100MB cap) => ship FP16 (~87MB). Deploys GREEDY
# (no search) on the 2-core CPU: ~30-60ms/forward, inside the 1s/turn budget.
SMOKE       = False     # True -> tiny net + few iters (sanity run; OVERRIDES ignored)
RESUME_FROM = None      # path to a *_train_state.pt to resume from, else None
OVERRIDES   = dict(
    ARCH="transformer", HIDDEN=512, N_TX_LAYERS=12, N_HEADS=16, TX_MLP_RATIO=4,
    N_STEM_RES=2, N_HEAD_RES=4, VALUE_RES_BLOCKS=4,
    GRAD_CHECKPOINT=True, USE_COMPILE=True, AMP_DTYPE=torch.bfloat16,   # H100: native bf16
    NUM_GROUPS=16, GROUP_SIZE=16,        # B=256 (H100 can push 24x16=384 / 32x16=512 if VRAM allows)
    TOTAL_ITERS=2000, BC_ENABLED=True,   # BC warm-start (clone medium) then PPO league
    # auxiliary reward-prediction head (UNREAL): a shallow head off the SHARED trunk regresses the
    # immediate per-step reward as a representation aux. Does NOT change reward / advantage / critic --
    # only adds trunk gradient, which helps under the terminal-dominated reward. (heartbeat: `ar`.)
    AUX_REWARD_PRED=False, AUX_REWARD_COEF=0.25,   # True -> turn the aux head on
)
""" + _SETTINGS_PRINT

GRPO_SETTINGS = """# ============================= RUN SETTINGS =============================
# PRESET: BIG transformer, GRPO (critic-free) -- group-relative baseline instead of PPO+GAE.
# Each iter rolls NUM_GROUPS distinct worlds out GRPO_GROUP times; the advantage is the rollout's
# return STANDARDIZED within its world-group (no value head trained). Same clipped surrogate +
# entropy as PPO; optional KL-to-reference via GRPO_KL_COEF (beta>0 freezes the post-BC net as ref).
# Weights are interchangeable with the PPO nb (same Policy net) -- only the critic head goes unused,
# so a PPO-trained .pt warm-starts a GRPO run directly (RESUME_FROM=...), and vice-versa.
# GPU: Colab A100 / H100 (both native bf16). Colab gotcha: USE_COMPILE=True can hit a torch<->triton crash ->
# `pip install triton==3.6.0` then RESTART (or set USE_COMPILE=False). BC warm-start -> dual-Elo league UNCHANGED.
#
# ADVANTAGE ESTIMATOR = legacy per-group std-norm [B], the BATTLE-TESTED v9/v12 default. A controlled local
# 3070 Ti A/B (scripts/_gpu_ab*.py; shared BC init, h256 trunk, 2p, 100 iters) found that the Tier 0-2
# "anti-collapse" estimators -- rank[C1] AND drgrpo center-only[B] -- and clip-higher[D3] ALL drove the agent
# into the PASSIVITY basin (launch-rate -> 0.03-0.34, elo DECLINED), while legacy std-norm stayed STABLE and
# improving (elo 1497->1576). Bounded != good: rank discards win-MAGNITUDE, so it stops rewarding decisive
# aggression. The std rare-win amplification the proof flagged is double-edged -- it is also what keeps the
# agent aggressive. The it426->it526 collapse was a TRIGGER (an asymmetric "lose-slower" outcome decay) x that
# amplifier, NOT the estimator; the default keeps WIN_DECAY=LOSS_DECAY=1.0 (symmetric, no trigger). The
# estimator flags below are MUTUALLY EXCLUSIVE (precedence phi>sibling>analytic>rank>cvar>std silently picks ONE
# if several are on -- that was the "all flags on" A100 collapse: it ran sibling/phi-value, NOT rank). train.py
# now PRINTS the active estimator + WARNS on conflicts. Enable AT MOST ONE and re-ablate on the 3070 Ti (NOT A100).
SMOKE       = False     # True -> tiny net + few iters (sanity run; OVERRIDES ignored)
RESUME_FROM = None      # path to a *_train_state.pt to resume from (e.g. a prior GRPO/PPO ckpt), else None
OVERRIDES   = dict(
    ALGO="grpo", GRPO_GROUP=16, GRPO_KL_COEF=0.0, GRPO_OUTCOME_ONLY=False, GRPO_WHITEN=False,
    GRPO_STD_NORM=True, GRPO_LOO=False,    # [B] legacy per-group std-norm: the STABLE, battle-tested default
    # --- estimator flags: MUTUALLY EXCLUSIVE, ALL OFF. Local A/B: rank/drgrpo/clip-hi -> PASSIVITY. Do NOT
    #     enable for a real run without re-ablating on the 3070 Ti (train.py prints/warns which one is active). ---
    GRPO_RANK_ADV=False, GRPO_ANALYTIC_ADV=False, GRPO_PHI_VALUE=False, GRPO_SIBLING_BASELINE=False, GRPO_CVAR_ALPHA=0.0,
    # --- orthogonal levers (compose), also OFF pending ablation ---
    CLIP_HI=0.0, ADV_TARGET_RMS=0.0, GRPO_RB_GATE=False, RATIO_LENGTHNORM=False, GRPO_AUX_VALUE=False, GRPO_CRN=False,
    VALUE_WARMUP_ITERS=0,                  # no critic to warm up under GRPO
    ARCH="transformer", HIDDEN=512, N_TX_LAYERS=12, N_HEADS=16, TX_MLP_RATIO=4,
    N_STEM_RES=2, N_HEAD_RES=4, VALUE_RES_BLOCKS=4,
    GRAD_CHECKPOINT=True, USE_COMPILE=True, AMP_DTYPE=torch.bfloat16,   # A100/H100: native bf16
    NUM_GROUPS=16, GROUP_SIZE=16,          # B=256 = 16 worlds x 16 group samples/iter (A100 80GB can push 24-32 groups)
    TOTAL_ITERS=2000, BC_ENABLED=True,     # BC warm-start (clone medium) then GRPO league
)
""" + _SETTINGS_PRINT

GRPO_SMOKE_SETTINGS = '''# ===================== RUN SETTINGS: ANTI-COLLAPSE SMOKE EXPERIMENT =====================
# Proves the v12 anti-collapse fix (Tier 0-2) WITHOUT a GPU. The legacy GRPO path (GRPO_STD_NORM=True)
# death-spirals in self-play: as the snapshot pool strengthens the learner faces LOPSIDED losing groups
# where a rare win's advantage ~ sqrt((1-p)/p) is unbounded -> it chases lucky wins, launch-rate ratchets
# down, and it loses to its own snapshots and to `greedy` (the it426->it526 log). The fix bounds /
# re-weights the advantage. This notebook runs:
#   (1) advantage_bound_probe   -- DETERMINISTIC: the new [C1] rank advantage is bounded by sqrt3 for EVERY
#                                  group; legacy std/analytic grow with lopsidedness. Plus a difficulty-
#                                  inversion demo (legacy equalizes group energy; Dr.GRPO keeps p(1-p)).
#   (2) run_anticollapse_compare -- short A/B train(): legacy vs rank[C1] vs drgrpo[B], logging the
#                                  advantage-health / launch-rate / return curves.
# To REPRODUCE the real death-spiral: set RESUME_FROM to a strong *_train_state.pt (e.g. it426) and raise
# EXP_ITERS (~80) on a GPU; the legacy arm bends down while the anti-collapse arms hold.
SMOKE        = True       # tiny net (HIDDEN=64); CPU-friendly. False -> full-size net (needs a GPU)
RESUME_FROM  = None       # set to a *_train_state.pt to reproduce the real collapse from a strong ckpt
EXP_ITERS    = 30         # A/B iters per arm (bump to ~80 on a GPU to watch the legacy arm bend down)
EXP_OVERRIDES = dict(     # smoke-scale env (CPU-friendly). On a GPU, raise B / EPISODE_STEPS / N_WORLDS.
    GRPO_GROUP=16, EPISODE_STEPS=120, N_WORLDS=64, B=64, B_4P=64, FLEET_CAP=512,
    SELFPLAY_REFRESH=5, LEAGUE_MAX_SNAPSHOTS=4, ELO_RECAL_ENVS=16, FOURP_ENABLED=True,
)
print("ANTI-COLLAPSE SMOKE | SMOKE=%s iters/arm=%d resume=%s" % (SMOKE, EXP_ITERS, RESUME_FROM))'''

GRPO_EXP_CELL = '''# (1) DETERMINISTIC proof: the new advantage cannot blow up (+ difficulty-inversion demo)
probe = advantage_bound_probe(Config(ALGO="grpo"), G=16)

# (2) short A/B train(): legacy (collapsing path) vs rank[C1] vs drgrpo[B], same init, smoke scale
res = run_anticollapse_compare(base_overrides=EXP_OVERRIDES, iters=EXP_ITERS, smoke=SMOKE,
                               resume_from=RESUME_FROM, log_every=max(1, EXP_ITERS // 10))

# (3) plot the proof: advantage-bound curve + per-iter advantage-health / launch-rate / return
plot_anticollapse(probe, res, path="anticollapse.png")'''

SMALL_SETTINGS = """# ============================= RUN SETTINGS =============================
# PRESET: SMALL net + MCTS -- the "small model, search-amplified" lineage.
# GPU: Colab T4 (sm_75: fp16 tensor cores, NO bf16 => AMP_DTYPE=float16; compile OFF for Turing).
# Colab gotcha: if a torch<->triton crash hits, `pip install triton==3.6.0` then RESTART runtime.
# Submission: ~1M params -> ~4MB fp32 (far under the 100MB cap; keep fp32). CPU-deployable WITH
# MCTS: ~depth-30 at ~150-200 sims/turn given a lean single-state stepper; MctsAgent spends the
# 60s overage bank on CRITICAL turns (crit-scaled sims) under a wall-clock->greedy guard.
SMOKE       = False     # True -> tiny net + few iters (sanity run; OVERRIDES ignored)
RESUME_FROM = None      # path to a *_train_state.pt to resume from, else None
OVERRIDES   = dict(
    ARCH="trunk", USE_ATTENTION=True, USE_GLU=True, DEST_HEAD="bilinear",
    HIDDEN=192, N_RES_BLOCKS=6, VALUE_RES_BLOCKS=2,
    GRAD_CHECKPOINT=False, USE_COMPILE=False, USE_AMP=False,  # T4 has no bf16; fp16 overflows the -1e9 masks -> train fp32 (net is tiny, AMP buys ~nothing)
    NUM_GROUPS=16, GROUP_SIZE=16,        # B=256 (fits T4 16GB)
    TOTAL_ITERS=1500, BC_ENABLED=True,
)
""" + _SETTINGS_PRINT

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

MCTS_CELL = '''# AlphaZero-style inference search on the trained net (proof of concept): net = prior +
# sampled candidate actions, bounded ship/production heuristic at leaves, opponent = a fixed
# deterministic script (so the search is a SOUND single-agent lookahead here). This is the seed
# for the planned BC warm-start -> search-distillation loop; decoupled-UCT for self-play is next.
# Sizes scale with SMOKE so a sanity run stays fast. (B=1 search is launch-bound -- a demo, not
# the deploy path.)
# MctsAgent scales sims on CRITICAL (close) turns and spends the 60s bank under a wall-clock
# guard -- here capped small so the in-notebook demo stays quick. (B=1 search is launch-bound; the
# real deploy path is a lean single-state stepper, not GpuEnv.)
_mcts_sims  = 8 if cfg.SMOKE else 32
_mcts_steps = 40 if cfg.SMOKE else 120
_mcts_budget = 0.3 if cfg.SMOKE else 0.8
_mcts_worlds = make_world_pool(cfg, 2 if cfg.SMOKE else 4, base_seed=99991)
_mcts_agent = MctsAgent(cfg, net, opp_kind="greedy", n_sims=_mcts_sims, crit_sims=_mcts_sims * 3,
                        c_puct=1.5, K=6, turn_budget_s=_mcts_budget, crit_budget_s=_mcts_budget * 2)
_greedy_fn  = greedy_act_fn(cfg, net)
for _i, _w in enumerate(_mcts_worlds):
    _sm, _scm, _ = play_2p_vs_script(cfg, lambda e, t: _mcts_agent.act(e, t)[0], _w, "greedy", max_steps=_mcts_steps)
    _sg, _scg, _ = play_2p_vs_script(cfg, _greedy_fn, _w, "greedy", max_steps=_mcts_steps)
    print("world %d  MCTS %.1f (%.0f-%.0f)  |  greedy %.1f (%.0f-%.0f)  bank=%.1fs"
          % (_i, _sm, _scm[0], _scm[1], _sg, _scg[0], _scg[1], _mcts_agent.bank_s))'''


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


def _assign_names(tgt):
    """Flatten an assignment target into the simple Names it binds (incl. tuple/list unpacking)."""
    if isinstance(tgt, ast.Name):
        return [tgt.id]
    if isinstance(tgt, (ast.Tuple, ast.List)):
        return [nm for e in tgt.elts for nm in _assign_names(e)]
    return []                                          # attribute/subscript targets bind no new name


def _top_bindings(src):
    """{name: lineno} for every TOP-LEVEL binding a cell introduces -- defs, classes, module-level
    assignments, and import aliases. Names bound *inside* a function/class are locals and can never
    collide across cells, so they are deliberately excluded."""
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.setdefault(node.name, node.lineno)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for nm in _assign_names(tgt):
                    out.setdefault(nm, node.lineno)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.setdefault(node.target.id, node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue                                   # `from __future__ import X` binds nothing usable
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                out.setdefault((a.asname or a.name).split(".")[0], node.lineno)
    return out


def _assert_no_cell_overwrites(imports, module_order, bodies):
    """Hard invariant: the LIBRARY cells (the shared imports cell + one cell per module) must bind
    DISJOINT top-level names, so no cell can ever silently overwrite another cell's definition. The
    run/driver cells are intentionally exempt -- they are imperative notebook flow and legitimately
    rebind locals (``cfg`` / ``net`` / ``RESUME_FROM``). Raises with the offending names on a clash."""
    owner, clashes = {}, []
    cells = [("imports", "\n".join(imports))] + [(m, bodies[m][1]) for m in module_order]
    for cell, src in cells:
        for nm in _top_bindings(src):
            if nm in owner:
                clashes.append((nm, owner[nm], cell))
            else:
                owner[nm] = cell
    if clashes:
        raise ValueError(
            "flattened cells would overwrite each other -- top-level name clash(es):\n"
            + "\n".join("    %-24s defined in BOTH '%s' and '%s'" % c for c in clashes)
            + "\n(rename one, or keep the single definition in one module.)")
    return len(owner)


def _src_lines(text):
    """nbformat 'source' wants a list of lines, each terminated by '\\n' except the last."""
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _src_lines(text)}


def _code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src_lines(text)}


def build_notebook(with_mcts=False, with_grpo=False, with_grpo_smoke=False):
    module_order = list(BASE_MODULES)
    if with_mcts:
        module_order.insert(module_order.index("plotting"), "mcts")
    if with_grpo or with_grpo_smoke:
        module_order.insert(module_order.index("rollout"), "grpo")   # after ppo, before train
    if with_grpo_smoke:
        module_order.insert(module_order.index("plotting"), "grpo_diag")   # after train, before plotting
    seen, imports = set(), []
    bodies = {}
    for name in module_order:
        doc, exts, body = process_module(name)
        bodies[name] = (doc, body)
        for e in exts:
            if e not in seen:
                seen.add(e); imports.append(e)

    _assert_no_cell_overwrites(imports, module_order, bodies)   # no cell may shadow another's defs

    cells = [_md(HEADER_MD), _md("## Imports"), _code("\n".join(imports))]
    for name in module_order:
        doc, body = bodies[name]
        head = "## `orbit_wars_v12.%s`" % name
        if doc:
            head += "\n\n" + doc.strip()
        cells += [_md(head), _code(body)]

    if with_grpo_smoke:
        cells += [_md("## Anti-collapse smoke experiment"), _code(GRPO_SMOKE_SETTINGS),
                  _md("## Prove the reward will not collapse (bound probe + A/B train)"), _code(GRPO_EXP_CELL)]
        return _wrap(cells)

    settings = GRPO_SETTINGS if with_grpo else (SMALL_SETTINGS if with_mcts else BIG_SETTINGS)
    cells += [_md("## Run training"), _code(settings), _code(BC_CELL), _code(TRAIN_CELL),
              _md("## Save every league agent as weights"), _code(SAVE_CELL),
              _md("## Plot training-health curves"), _code(PLOT_CELL)]
    if with_mcts:
        cells += [_md("## MCTS inference-search (proof of concept)"), _code(MCTS_CELL)]
    return _wrap(cells)


def _wrap(cells):
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

    # cleanliness invariant, re-checked against the WRITTEN notebook: the flattened library cells
    # (everything before RUN SETTINGS) must define disjoint top-level names -- no cell overwrites another.
    lib_owner, dup = {}, []
    for c in doc["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "RUN SETTINGS" in src:
            break
        for nm in _top_bindings(src):
            (dup.append(nm) if nm in lib_owner else lib_owner.__setitem__(nm, True))
    assert not dup, "duplicate top-level names across library cells (cells overwrite each other): %s" % sorted(set(dup))
    print("[check] %d top-level names across library cells, all unique (no cell overwrites another)" % len(lib_owner))

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

    # measured eviction semantics on the flattened League: ONE standard for scripts AND neural league
    # players (snapshots + invited); self-anchors + the newest snapshot are protected. Stub the four
    # cfg-style eval funcs (scripts keyed by kind; neural members keyed by an `_evict` tag on the net).
    o2, o4 = ns["_eval_score"], ns["_eval_winrate4"]
    on2, on4 = ns["_score2_vs"], ns["_eval_winrate4_net"]
    F = {"starter": 0.99, "greedy": 0.99, "medium": 0.10}
    ns["_eval_score"] = lambda cfg, net, kind, worlds, n: F.get(kind, 0.5)
    ns["_eval_winrate4"] = lambda cfg, net, kind, worlds, n: F.get(kind, 0.5)
    ns["_score2_vs"] = lambda cfg, net, opp, worlds, n: getattr(opp, "_evict", 0.5)
    ns["_eval_winrate4_net"] = lambda cfg, net, opp, worlds, n: getattr(opp, "_evict", 0.5)
    SK = ns["SELF_KIND"]
    nnet = lambda v: types.SimpleNamespace(_evict=v, to=lambda *a, **k: None)
    def mk(kind, label, net=None, **kw):
        return dict(net=net, kind=kind, label=label, anchor=kw.pop("anchor", False),
                    pinned=kw.pop("pinned", False), elo=1000.0, elo4=1000.0, n=0, n4=0, **kw)
    m_un  = mk("medium", "medium")                                  # unmastered script -> survives
    m_ma  = mk("starter", "starter")                               # mastered script -> removed
    m_wd  = mk("greedy", "greedy", wd_active=True)                 # keep-kind watchdog -> dormant
    m_old = mk("snapshot", "it_old", net=nnet(0.99))              # mastered league player -> removed
    m_new = mk("snapshot", "it_new", net=nnet(0.99))              # newest snapshot -> protected
    m_slf = mk(SK, "selfA0", net=nnet(0.99), anchor=True, pinned=True)   # Elo reference -> never
    m_inv = mk("invited", "inv:x", net=nnet(0.99), anchor=True, pinned=True)   # mastered invited -> removed
    league.members = [m_un, m_ma, m_wd, m_old, m_new, m_slf, m_inv]
    league.evict_mastered(net, None, 8, neural=True)
    assert m_un in league.members, "unmastered script wrongly evicted"
    assert m_ma not in league.members, "mastered ordinary script NOT removed"
    assert m_wd in league.members and m_wd["wd_active"] is False, "watchdog must retire to dormant"
    assert m_old not in league.members, "mastered league player (snapshot) NOT removed"
    assert m_new in league.members, "newest snapshot (current self) must be protected"
    assert m_slf in league.members, "self-anchor (Elo reference) must never be evicted"
    assert m_inv not in league.members, "mastered invited league player NOT removed"
    ns["_eval_score"], ns["_eval_winrate4"] = o2, o4
    ns["_score2_vs"], ns["_eval_winrate4_net"] = on2, on4
    print("[check] measured eviction OK: scripts + league players under one standard | self-anchor & newest snapshot protected")

    if "grpo_update" in ns:                      # GRPO variant: exercise the critic-free path end-to-end
        cfgg = ns["Config"].create(
            smoke=True, ALGO="grpo", GRPO_GROUP=4, GRPO_KL_COEF=0.05, VALUE_WARMUP_ITERS=0,
            TOTAL_ITERS=4, EPISODE_STEPS=64, N_WORLDS=32, B=8, B_4P=8, FLEET_CAP=256,
            ELO_RECAL_ENVS=8, SELFPLAY_REFRESH=2, LEAGUE_MAX_SNAPSHOTS=3, LEARNER_RECAL_EVERY=2,
            N_HEADS=8, N_TX_LAYERS=2, N_HEAD_RES=1, device=torch.device("cpu"))
        _gnet, ghist, _gleague = ns["train"](cfgg, total_iters=4, log_every=1)
        assert ghist["fmt"].count(2) >= 1 and ghist["fmt"].count(4) >= 1, "GRPO: expected both 2p and 4p iters"
        assert len(ghist["learner_elo"]) == 4
        print("[check] GRPO tiny train ran: %d iters (critic-free group baseline + KL-to-ref)" % len(ghist["fmt"]))

    if "advantage_bound_probe" in ns:            # anti-collapse smoke nb: bound probe + tiny 3-arm A/B
        ns["advantage_bound_probe"](ns["Config"](ALGO="grpo"), G=16, verbose=False)
        _res = ns["run_anticollapse_compare"](
            base_overrides=dict(GRPO_GROUP=4, EPISODE_STEPS=48, N_WORLDS=24, B=8, B_4P=8, FLEET_CAP=256,
                                ELO_RECAL_ENVS=8, SELFPLAY_REFRESH=2, LEAGUE_MAX_SNAPSHOTS=3,
                                LEARNER_RECAL_EVERY=0, ELO_RECAL_EVERY=0, N_HEADS=8, N_TX_LAYERS=2,
                                N_HEAD_RES=1, FOURP_ENABLED=False, device=torch.device("cpu")),
            iters=2, smoke=True, log_every=99)
        assert set(_res["_arms"]) == {"legacy", "rank[C1]", "drgrpo[B]"}, "anti-collapse arms changed unexpectedly"
        for _a in _res["_arms"]:
            assert max(_res[_a]["adv_absmax"]) < 50.0, "%s advantage blew up in the A/B smoke" % _a
        print("[check] anti-collapse: bound probe PASS + 3-arm A/B ran, all advantages bounded")

    print("\nCHECK OK: generated notebook execs, trains, and evicts faithfully.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=None, help="output .ipynb path (default depends on --mcts/--grpo)")
    ap.add_argument("--mcts", action="store_true", help="include the MCTS module + inference-search demo")
    ap.add_argument("--grpo", action="store_true", help="critic-free GRPO variant (setup1_v12_grpo_a100.ipynb)")
    ap.add_argument("--grpo-smoke", dest="grpo_smoke", action="store_true",
                    help="GRPO anti-collapse smoke experiment (setup1_v12_grpo_anticollapse.ipynb)")
    ap.add_argument("--both", action="store_true", help="generate BOTH the base v12 and v12-mcts notebooks")
    ap.add_argument("--all", action="store_true", help="generate the base, MCTS, GRPO, and anti-collapse notebooks")
    ap.add_argument("--check", action="store_true", help="exec the generated nb(s) (SMOKE) to verify")
    args = ap.parse_args()
    if sum([args.mcts, args.grpo, args.grpo_smoke]) > 1:
        ap.error("--mcts / --grpo / --grpo-smoke are mutually exclusive")

    if args.all:
        variants = [(False, False, False, DEFAULT_OUT), (True, False, False, MCTS_OUT),
                    (False, True, False, GRPO_OUT), (False, False, True, GRPO_SMOKE_OUT)]
    elif args.both:
        variants = [(False, False, False, DEFAULT_OUT), (True, False, False, MCTS_OUT)]
    else:
        default_out = (GRPO_SMOKE_OUT if args.grpo_smoke else GRPO_OUT if args.grpo
                       else MCTS_OUT if args.mcts else DEFAULT_OUT)
        variants = [(args.mcts, args.grpo, args.grpo_smoke, args.out or default_out)]
    for with_mcts, with_grpo, with_grpo_smoke, out in variants:
        nb = build_notebook(with_mcts=with_mcts, with_grpo=with_grpo, with_grpo_smoke=with_grpo_smoke)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        n_modules = len(BASE_MODULES) + with_mcts + (with_grpo or with_grpo_smoke) + with_grpo_smoke
        tag = ", +MCTS" if with_mcts else (", +GRPO+diag" if with_grpo_smoke else ", +GRPO" if with_grpo else "")
        print("wrote %s  (%d cells from %d modules%s)"
              % (os.path.relpath(out, REPO), len(nb["cells"]), n_modules, tag))
        if args.check:
            check(out)


if __name__ == "__main__":
    main()
