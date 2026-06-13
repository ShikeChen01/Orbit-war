"""Generate notebooks/setup1_v7_a100.ipynb -- the A100/Colab run of the v7 recipe.

Copies setup1_v6_ppo.ipynb (which already contains the v7 patches: categorical WHERE,
gate-only entropy, BC warm-start, guardrails) and rewrites config-cell lines for the
A100 production run. Run-All = BC pretrain -> guardrailed PPO league from the BC init.

    python scripts/make_a100_nb.py [--hidden 768] [--layers 6] [--iters 600] [--b 256]
"""
import argparse, json, os, re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "notebooks", "legacy", "setup1_v6_ppo.ipynb")
DST = os.path.join(REPO, "notebooks", "legacy", "setup1_v7_a100.ipynb")

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=768)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--iters", type=int, default=600)
ap.add_argument("--b", type=int, default=256, help="parallel envs (NUM_GROUPS*GROUP_SIZE)")
args = ap.parse_args()
assert args.b % 16 == 0, "--b must be a multiple of 16"
ng, gs = args.b // 16, 16

doc = json.load(open(SRC, encoding="utf-8"))

# line-level rewrites of the CONFIG cell (left side must match the v6 notebook exactly)
REWRITES = [
    (r"HIDDEN          = 64  if SMOKE else 512", f"HIDDEN          = 64  if SMOKE else {args.hidden}"),
    (r"N_TX_LAYERS  = 4 ", f"N_TX_LAYERS  = {args.layers} "),
    (r"NUM_GROUPS      = 4   if SMOKE else 16", f"NUM_GROUPS      = 4   if SMOKE else {ng}"),
    (r"GROUP_SIZE      = 4   if SMOKE else 16", f"GROUP_SIZE      = 4   if SMOKE else {gs}"),
    (r"TOTAL_ITERS       = 4 if SMOKE else 1400", f"TOTAL_ITERS       = 4 if SMOKE else {args.iters}"),
    (r"SELFPLAY_REFRESH  = 5 if SMOKE else 100", "SELFPLAY_REFRESH  = 5 if SMOKE else 25"),
    (r"SNAPSHOT_WARMUP   = 0 if SMOKE else 100", "SNAPSHOT_WARMUP   = 0   # BC init is already strong -> snapshot from the start"),
    (r"LEAGUE_MAX_SNAPSHOTS = 4 if SMOKE else 12", "LEAGUE_MAX_SNAPSHOTS = 4 if SMOKE else 16"),
    (r"BC_ENABLED        = False", "BC_ENABLED        = True   # Run-All: BC pretrain -> PPO warm-start"),
    (r"BC_ROUNDS         = 12", "BC_ROUNDS         = 12"),
    (r"VALUE_WARMUP_ITERS = 0", "VALUE_WARMUP_ITERS = 5"),
    (r"LR_WARMUP_ITERS    = 0", "LR_WARMUP_ITERS    = 8"),
    (r"ENT_COEF_START = 0.02", "ENT_COEF_START = 0.005"),
    (r"ENT_COEF_END   = 0.002", "ENT_COEF_END   = 0.002"),
    (r"CKPT_EVERY = 100", "CKPT_EVERY = 20   # fixed-opponent gauntlet + best-ckpt cadence (dense: catch peaks/ratchets early)"),
    (r"ELO_RECAL_EVERY      = 0 if SMOKE else 1000", "ELO_RECAL_EVERY      = 0 if SMOKE else 500"),
    # v7 wiring: 12 heads (head_dim 64 at d=768, matches validated M geometry)
    (r"N_HEADS      = 8 ", "N_HEADS      = 12"),
    # post-stack decision depth: A/B'd at S-scale (runs/v6/headres{1,4}) -> +0.09 held-out
    # gauntlet on every checkpoint at zero wall-clock cost
    (r"N_HEAD_RES   = 1 ", "N_HEAD_RES   = 4 "),
    # v7 reward fix (the A100 passivity ratchet, 2026-06-09): full outcome signal + halved shaping.
    # gamma=0.997 already prefers fast wins; the decay only buried the win/loss under the shaping.
    (r"WIN_DECAY         = 0.995", "WIN_DECAY         = 1.0"),
    (r"LOSS_DECAY        = 0.995", "LOSS_DECAY        = 1.0"),
    (r"SHAPE_SHIP = 50.0", "SHAPE_SHIP = 25.0"),
    (r"SHAPE_PROD = 50.0", "SHAPE_PROD = 25.0"),
]

HEADER = """\
# Orbit Wars -- v7 A100 run (BC warm-start -> guardrailed PPO, categorical WHERE)

**This is the production A100/Colab notebook of the v7 recipe** (see `docs/rl_design_v7.pdf`).
Run-All does: (1) precompute worlds WITH official elliptical comets (bit-exact port of the
engine's `generate_comet_paths`, CPU-side), (2) BC-clone the medium bot into the gated heads
(~5 min), (3) guardrailed PPO against the Elo/PFSP league from that init.

**What changed vs the first (collapsed) A100 attempt** -- the passivity-ratchet post-mortem:
- `WIN_DECAY=LOSS_DECAY=1.0` + `SHAPE_*=25`: the outcome signal is no longer buried under the
  per-step shaping (the old config taught the gate to close even in won games). Validated
  100 iters at S-scale: gauntlet stable 0.70-0.84, Elo 1040->1172, no collapse.
- `[TRIPWIRE]` log line: fires when the launch-rate EMA drops below 40% of its early baseline --
  the old run decayed 1.3 -> 0.05 launches/step over 80 healthy-looking iters. If it fires, stop.
- Gauntlet/best-ckpt every 20 iters (the old run's peak at ~it13 fell between checkpoints).
- Official comets in training (previously straight-line approximations).
- `N_HEADS=12` (head_dim 64 at d=768, the validated M geometry).

**Colab checklist**
1. Runtime -> A100 GPU. Optionally `import os; os.environ['OW_CKPT_DIR']='/content/drive/MyDrive/ow_v7'`
   (after mounting Drive) BEFORE the config cell, so checkpoints + the resumable train state survive
   session death. Resume by setting `RESUME_FROM = TRAIN_STATE_PATH` and re-running the run cell.
2. Watch the first ~15 heartbeats: it1-5 must show `kl 0.000` (critic-only warmup), then KL should
   sit in the 0.02-0.05 band. If KL pins at the stop threshold every iter, halve LR.
3. The `[best]` line prints the fixed-opponent gauntlet -- that is the number that matters
   (training `wr` is against a moving PFSP opponent). Healthy run: gauntlet >= 0.75 and rising,
   Elo +~25/10 iters, launch-rate roughly flat near its BC level (~1.0), no `[TRIPWIRE]` lines.
"""

cfg_done = 0
for c in doc["cells"]:
    s = "".join(c["source"])
    if c["cell_type"] == "markdown" and s.startswith("# Orbit Wars"):
        c["source"] = HEADER.splitlines(keepends=True)
        cfg_done += 1
    elif c["cell_type"] == "code" and "SMOKE = " in s and "BC_ENABLED" in s:
        for old, new in REWRITES:
            assert old in s, "config line not found: %r" % old
            s = s.replace(old, new)
        c["source"] = s.splitlines(keepends=True)
        cfg_done += 1

assert cfg_done == 2, "expected header+config rewrite, got %d" % cfg_done
json.dump(doc, open(DST, "w", encoding="utf-8"), indent=1)
print("wrote", DST, f"(d={args.hidden} L={args.layers} B={args.b} iters={args.iters})")
