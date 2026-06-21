"""End-to-end smoke of notebooks/setup1_v12_a100.ipynb: executes the REAL notebook cells in order
(config flipped to SMOKE + shrink overrides, with LEARNER_RECAL_EVERY>0 so the recal path runs),
then asserts the v12 invariants:

  - clean rebuild: NO monkeypatch cascade in the source (_v9*/_orig_/`League.`/OVERRIDES/weak-boost);
  - the new recal path executes at runtime (chain-init + _reground + validate_watchdogs + measured
    eviction) without error;
  - measured eviction semantics via a live unit-test with stubbed eval funcs: an unmastered script
    survives, a mastered ordinary script is removed, a mastered keep-kind watchdog retires to DORMANT;
  - v12.1 OR-gate: mastery in EITHER format alone (2p-only OR 4p-only) evicts; neither survives;
  - shared league invariants (one-hot obs F_DIM 104, mixed 2p/4p, dual Elo, gauntlets) + resume.

    python scripts/smoke_v12_nb.py
"""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
os.environ["OW_CKPT_DIR"] = os.path.join("runs", "v12_smoke")
os.makedirs(os.environ["OW_CKPT_DIR"], exist_ok=True)

NB = os.path.join("notebooks", "setup1_v12_a100.ipynb")
doc = json.load(open(NB, encoding="utf-8"))
ns = {"__name__": "nb"}

# static guards: the clean rebuild must NOT carry any of the monkeypatch cascade
_allsrc = "\n".join("".join(c["source"]) for c in doc["cells"] if c["cell_type"] == "code")
for bad in ["[evict init]", "_v9", "_orig_", "League._", "OVERRIDES", "_weak_boost",
            "evict_dominated", "WEAK_BOOST"]:
    assert bad not in _allsrc, "clean rebuild must not contain %r" % bad
for need in ["def evict_mastered(", "def _reground(", "def _ground_ratings(",
             "def validate_watchdogs(", "[evict]", "[recal]"]:
    assert need in _allsrc, "missing expected v12 symbol %r" % need

# post-config shrink (applied right after the config cell executes). LEARNER_RECAL_EVERY>0 so the
# recal/eviction path actually runs inside the short loop.
OVERRIDES = dict(
    TOTAL_ITERS=8, EPISODE_STEPS=96, N_WORLDS=48, B=8, B_4P=8, FLEET_CAP=256,
    ELO_RECAL_ENVS=8, CKPT_EVERY=3, SELFPLAY_REFRESH=2, LEAGUE_MAX_SNAPSHOTS=3,
    LEARNER_RECAL_EVERY=3, ELO_RECAL_EVERY=4,   # exercise BOTH lite (ground_members=False) + full recal
    BC_ROUNDS=1, BC_EPOCHS=1, VALUE_WARMUP_ITERS=1, LR_WARMUP_ITERS=2,
    N_HEADS=8, N_TX_LAYERS=2, N_HEAD_RES=1,   # SMOKE HIDDEN=64 not divisible by the A100's 12 heads
)

for i, c in enumerate(doc["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if "SMOKE = False " in src and "BC_ENABLED" in src:
        src = src.replace("SMOKE = False ", "SMOKE = True  ")
        exec(compile(src, "cell%d" % i, "exec"), ns)
        ns.update(OVERRIDES)
        print("[smoke] config overridden:", OVERRIDES)
        continue
    if src.startswith("RUN_SMOKE"):
        src = src.replace("RUN_SMOKE = False", "RUN_SMOKE = True")
    print("[smoke] exec cell %d: %s" % (i, src.splitlines()[0][:80]), flush=True)
    exec(compile(src, "cell%d" % i, "exec"), ns)

# --- shared league invariants ---
assert ns["F_DIM"] == 198, "v13 threat one-hot + per-planet motion obs (F_DIM 198), got %r" % ns["F_DIM"]
hist = ns["hist"]
n2, n4 = hist["fmt"].count(2), hist["fmt"].count(4)
print("\n[smoke] iterations: %d x 2p, %d x 4p" % (n2, n4))
assert n4 >= 1 and n2 >= 1, "expected BOTH formats in the mix"
assert len(hist["learner_elo4"]) == len(hist["learner_elo"]) == ns["TOTAL_ITERS"]
assert "gauntlet2" in hist and "gauntlet4" in hist and len(hist["gauntlet4"]) > 0

# the recal path must have actually run (chain-init froze a self-anchor)
lg = ns["league"]
assert any(m.get("kind") == ns["SELF_KIND"] for m in lg.members), "recal never ran (no self-anchor frozen)"
print("[smoke] recal path ran: %d self-anchor(s) frozen" % len(lg._self_anchors()))

# --- v12 measured-eviction semantics (stub the eval funcs to force mastery) ---
_orig_s2, _orig_w4 = ns["_eval_score"], ns["_eval_winrate4"]
_FORCE = {"starter": 0.99, "random": 0.99, "greedy": 0.99, "medium": 0.10}   # mastered except medium
ns["_eval_score"] = lambda net, kind, worlds, n: _FORCE.get(kind, 0.5)
ns["_eval_winrate4"] = lambda net, kind, worlds, n: _FORCE.get(kind, 0.5)
saved = list(lg.members)
m_un = dict(net=None, kind="medium", label="medium", anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0)
m_ma = dict(net=None, kind="starter", label="starter", anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0)
m_wd = dict(net=None, kind="greedy", label="greedy", anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0, wd_active=True)
lg.members = [m_un, m_ma, m_wd]
lg.evict_mastered(ns["net"], None, 8)
assert m_un in lg.members, "unmastered script was wrongly evicted (eviction not mastery-gated)"
assert m_ma not in lg.members, "mastered ordinary script was NOT removed (measured eviction broken)"
assert m_wd in lg.members, "keep-kind watchdog must never be removed"
assert m_wd["wd_active"] is False, "re-mastered active watchdog must retire to DORMANT"
lg.members = saved
ns["_eval_score"], ns["_eval_winrate4"] = _orig_s2, _orig_w4
print("[smoke] measured eviction: unmastered survives | mastered ordinary removed | watchdog -> dormant  OK")

# --- v12.1 OR-gate: mastery in EITHER format ALONE must evict (real kinds, both in _OPP_BY_KIND) ---
_S2 = {"starter": 0.99, "medium": 0.10, "intermediate": 0.10}   # starter mastered 2p-only
_W4 = {"starter": 0.10, "medium": 0.99, "intermediate": 0.10}   # medium  mastered 4p-only; intermediate neither
ns["_eval_score"]    = lambda net, kind, worlds, n: _S2[kind]
ns["_eval_winrate4"] = lambda net, kind, worlds, n: _W4[kind]
m_2p = dict(net=None, kind="starter",      label="starter",      anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0)
m_4p = dict(net=None, kind="medium",       label="medium",       anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0)
m_no = dict(net=None, kind="intermediate", label="intermediate", anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0)
lg.members = [m_2p, m_4p, m_no]
lg.evict_mastered(ns["net"], None, 8)
assert m_2p not in lg.members, "OR-gate: 2p-only mastery must evict (gate reverted to AND?)"
assert m_4p not in lg.members, "OR-gate: 4p-only mastery must evict (gate reverted to AND?)"
assert m_no in lg.members,     "OR-gate: neither-mastered script must survive"
lg.members = saved
ns["_eval_score"], ns["_eval_winrate4"] = _orig_s2, _orig_w4
print("[smoke] OR-gate eviction: 2p-only evicts | 4p-only evicts | neither survives  OK")

# resume round-trip (league state reloads + first recal at resume runs eviction)
net2, hist2, lg2 = ns["train"](total_iters=2, resume_from=ns["TRAIN_STATE_PATH"])
assert any(m["kind"] == "snapshot" for m in lg2.members), "snapshots lost across resume"

print("\nSMOKE V12 OK: clean rebuild, recal path runs, measured every-recal eviction, resume round-trip")
