"""Generate notebooks/setup1_v11_a100.ipynb -- v9 league + CORRECTED league eviction.

Layered on notebooks/setup1_v9_a100.ipynb (the fully-patched v9.1..v9.9 MLP-league build). v10 is a
SEPARATE line (transformer trunk); v11 is a FRESH experiment on the proven v9 MLP base whose ONLY
change is the eviction trigger.

The v9 bug (diagnosed): the v9.9.3 wrapper `_v99_reground` ran an UNCONDITIONAL "init mass-evict" --
the instant the first self-anchor was frozen (the very first learner recalibration, ~iter
LEARNER_RECAL_EVERY) it dropped EVERY non-watchdog scripted member at once, regardless of whether the
learner had actually mastered them. That preempted the already-correct, every-recal, all-members,
mastery-gated `evict_dominated`, and made the MASTER_EVICT_*/windowed-WR standards moot for scripts.

v11 fix: REMOVE the init mass-evict. Eviction is now exactly what the standard describes -- it runs at
EVERY learner recalibration over ALL league members (`evict_dominated`, called from `_v99_reground`),
and a member is evicted only once the learner has MASTERED it (windowed WR: _dom2 AND _dom4). Keep-kinds
(random/greedy) are never removed; a re-mastered active watchdog retires to DORMANT. Nothing else about
the env, obs (F_DIM 104), arch, BC, or league machinery changes -- 2p/4p dynamics stay bit-exact to v9.

    python scripts/make_v11_nb.py
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "notebooks", "setup1_v9_a100.ipynb")
DST = os.path.join(REPO, "notebooks", "setup1_v11_a100.ipynb")

doc = json.load(open(SRC, encoding="utf-8"))


def cell_src(c):
    return "".join(c["source"])


def set_src(c, s):
    c["source"] = s.splitlines(keepends=True)


def find_cell(pred, what):
    hits = [c for c in doc["cells"] if pred(cell_src(c))]
    assert len(hits) == 1, "expected exactly 1 cell for %s, got %d" % (what, len(hits))
    return hits[0]


def patch(c, old, new, what, count=1):
    s = cell_src(c)
    n = s.count(old)
    assert n == count, "%s: expected %d occurrence(s) of %r, found %d" % (what, count, old[:70], n)
    set_src(c, s.replace(old, new))


# ============================================================================
# 0. header markdown -- rebrand v9 -> v11 + document the eviction fix
# ============================================================================
V11_NOTE = """

**v11 = v9 (MLP league) + corrected league eviction.** v9 dropped EVERY scripted member at once the
instant the first self-anchor was frozen (an unconditional "init mass-evict" on the first
recalibration), regardless of mastery -- so the windowed-WR / MASTER_EVICT_* standards never actually
gated a script. v11 removes that one-shot drop: eviction now runs at **every learner recalibration over
all league members** and removes a member only once the learner has **mastered** it (windowed WR:
2p mean(last `WR_WINDOW_2P`) >= `EVICT_WR_2P` AND 4p mean(last `WR_WINDOW_4P`) >= `EVICT_WR_4P`).
Keep-kinds (random/greedy) are never removed; a re-mastered active watchdog retires to DORMANT.
Env / obs (F_DIM 104) / arch / BC are identical to v9, so 2p/4p dynamics stay bit-exact -- this is a
FRESH run (fresh BC + league; point `OW_CKPT_DIR` at a new dir)."""

hdr = find_cell(lambda s: s.startswith("# Orbit Wars"), "header markdown")
set_src(hdr, cell_src(hdr).replace(
    "# Orbit Wars -- v9 A100 run (v8 league + per-seat ONE-HOT ownership)",
    "# Orbit Wars -- v11 A100 run (v9 league + every-recal mastery eviction)", 1) + V11_NOTE)

# ============================================================================
# 1. the eviction fix -- in the v9.9.2/v9.9.3 cell
# ============================================================================
ev = find_cell(lambda s: "v9.9.2: windowed-WR eviction" in s and "_v99_reground" in s, "eviction cell")

# 1a. rewrite the stale item-2 comment that described the (now removed) init mass-evict
patch(ev,
      "#   2. ORDINARY scripted bots (medium / intermediate / starter) are SCAFFOLDING: the instant the\n"
      "#      first self-anchor exists (the initial calibration), they are ALL evicted at once -- after BC\n"
      "#      the learner already beats them, so self-play is the curriculum from here on. (The windowed-WR\n"
      "#      graduated drop/evict below still governs any script that remains live, e.g. an activated watchdog.)\n",
      "#   2. ORDINARY scripted bots (medium / intermediate / starter) are SCAFFOLDING the learner sheds as\n"
      "#      it climbs past them. v11: eviction is MASTERY-GATED and runs at EVERY learner recalibration over\n"
      "#      ALL league members (evict_dominated, below) -- a member leaves the league only once the learner\n"
      "#      has actually mastered it (windowed WR: _dom2 AND _dom4), NOT blanket-dropped the instant the\n"
      "#      first self-anchor is frozen (the v9.9.3 init mass-evict, REMOVED in v11). Keep-kinds\n"
      "#      (random/greedy) are never removed; a re-mastered active watchdog retires to DORMANT instead.\n",
      "item-2 comment")

# 1b. REMOVE the unconditional init mass-evict block inside _v99_reground (line-splice: robust to the
#     trailing-space alignment in the source). Everything between `if self._self_anchors():` and the
#     following `if promoted:` is the init mass-evict; replace it with a v11 explainer comment.
lines = ev["source"]
i0 = next(k for k, l in enumerate(lines) if l.startswith("    if self._self_anchors():"))
i1 = next(k for k, l in enumerate(lines) if l.startswith("    if promoted:"))
assert i0 < i1, "init mass-evict block bounds not found (i0=%d i1=%d)" % (i0, i1)
# sanity: the block we're cutting is the one that prints "[evict init]"
assert any("[evict init]" in l for l in lines[i0:i1]), "expected [evict init] inside the cut block"
replacement = [
    "    # v11: init mass-evict REMOVED. Eviction is mastery-gated and runs at every recal via\n",
    "    # evict_dominated() below (all members, windowed-WR standard) -- scripts are no longer\n",
    "    # blanket-dropped the instant the first self-anchor is frozen.\n",
]
ev["source"] = lines[:i0] + replacement + lines[i1:]

# 1c. relabel the per-member eviction print (cosmetic, kills "history")
patch(ev, '[evict v9.9]', '[evict v11]', "evict print label")

# 1d. update the cell's status banner to describe the new behavior (preserve %s %.2f %d %d arg order)
patch(ev,
      'print("[v9.9.3] scripts auto-evicted at first self-anchor | watchdogs %s dormant->validate(tol %.2f)->league->dormant | windowed-WR (last %d/%d) graduated drop/evict for live scripts | neural seat-fill | promotion resample+recal | weak-boost OFF" % (\n',
      'print("[v11] mastery-evict EVERY recal over ALL members | watchdogs %s dormant->validate(tol %.2f)->league->dormant | windowed-WR standard last %d/%d | neural seat-fill | promotion resample+recal | weak-boost OFF" % (\n',
      "status banner")

# ============================================================================
# 2. cosmetics: plot title v9 -> v11
# ============================================================================
pl = find_cell(lambda s: "Training-health curves" in s, "plot cell")
patch(pl, 'fig.suptitle("Orbit Wars v9 -- training health', 'fig.suptitle("Orbit Wars v11 -- training health',
      "plot title")

json.dump(doc, open(DST, "w", encoding="utf-8"), indent=1)
print("wrote", DST, "(%d cells)" % len(doc["cells"]))
