"""End-to-end smoke of notebooks/setup1_v11_a100.ipynb: executes the REAL notebook cells in order
(config flipped to SMOKE + shrink overrides) so the full v11 path runs in minutes, then asserts the
v9 invariants (one-hot obs F_DIM 104, mixed 2p/4p, dual Elo, gauntlets, invited-member adapter,
resume round-trip) PLUS the v11 eviction fix:

  - the v9.9.3 unconditional "init mass-evict" is gone from the source (no [evict init]);
  - the eviction semantics are mastery-gated over ALL members and run every recal: a UnitTest on the
    live League shows an unmastered script survives, a mastered ordinary script is removed, and a
    mastered keep-kind (active watchdog) retires to DORMANT rather than being removed.

Mirrors scripts/smoke_v9_nb.py.

    python scripts/smoke_v11_nb.py
"""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
os.environ["OW_CKPT_DIR"] = os.path.join("runs", "v11_smoke")
os.makedirs(os.environ["OW_CKPT_DIR"], exist_ok=True)

NB = os.path.join("notebooks", "setup1_v11_a100.ipynb")
doc = json.load(open(NB, encoding="utf-8"))
ns = {"__name__": "nb"}

# static guard: the v11 fix must be baked into the notebook source itself
_allsrc = "\n".join("".join(c["source"]) for c in doc["cells"] if c["cell_type"] == "code")
assert "[evict init]" not in _allsrc, "v11 must NOT carry the unconditional init mass-evict"
assert "[v11] mastery-evict EVERY recal" in _allsrc, "v11 eviction banner missing"

# post-config shrink (applied right after the config cell executes)
OVERRIDES = dict(
    TOTAL_ITERS=8, EPISODE_STEPS=96, N_WORLDS=48, B=8, B_4P=8, FLEET_CAP=256,
    ELO_RECAL_ENVS=8, CKPT_EVERY=3, SELFPLAY_REFRESH=2, LEAGUE_MAX_SNAPSHOTS=3,
    BC_ROUNDS=1, BC_EPOCHS=1, VALUE_WARMUP_ITERS=1, LR_WARMUP_ITERS=2,
    N_HEADS=8, N_TX_LAYERS=2, N_HEAD_RES=1,   # SMOKE HIDDEN=64 is not divisible by the A100's 12 heads
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

# --- shared v9 invariants (unchanged by v11) ---
assert ns["F_DIM"] == 104, "v11 keeps the one-hot obs (F_DIM 104), got %r" % ns["F_DIM"]
hist = ns["hist"]
n2, n4 = hist["fmt"].count(2), hist["fmt"].count(4)
print("\n[smoke] iterations: %d x 2p, %d x 4p" % (n2, n4))
assert n4 >= 1 and n2 >= 1, "expected BOTH formats in the mix"
assert len(hist["learner_elo4"]) == len(hist["learner_elo"]) == ns["TOTAL_ITERS"]
assert "gauntlet2" in hist and "gauntlet4" in hist and len(hist["gauntlet4"]) > 0

# --- v11 eviction semantics: mastery-gated, all members, keep-kinds protected ---
lg = ns["league"]
WR2, WR4 = ns["WR_WINDOW_2P"], ns["WR_WINDOW_4P"]


def _mk(kind, mastered, wd_active=False):
    return dict(net=None, kind=kind, label=kind, anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0,
                wd_active=wd_active,
                wr_hist=[1.0] * WR2 if mastered else [0.0] * WR2,
                wr4_hist=[1.0] * WR4 if mastered else [0.0] * WR4)


saved = list(lg.members)
m_un = _mk("medium", mastered=False)             # not mastered -> must SURVIVE
m_ma = _mk("starter", mastered=True)             # mastered ordinary -> must be REMOVED
m_wd = _mk("greedy", mastered=True, wd_active=True)   # mastered keep-kind/watchdog -> RETIRE to dormant
lg.members = saved + [m_un, m_ma, m_wd]
lg.evict_dominated()
assert m_un in lg.members, "unmastered script was wrongly evicted (eviction not mastery-gated)"
assert m_ma not in lg.members, "mastered ordinary script was NOT evicted (every-recal eviction broken)"
assert m_wd in lg.members, "keep-kind watchdog must never be removed"
assert m_wd["wd_active"] is False, "re-mastered active watchdog must retire to DORMANT"
lg.members = saved
print("[smoke] eviction: unmastered survives | mastered ordinary removed | mastered watchdog -> dormant  OK")

# resume round-trip (league state reloads cleanly)
net2, hist2, lg2 = ns["train"](total_iters=2, resume_from=ns["TRAIN_STATE_PATH"])
assert any(m["kind"] == "snapshot" for m in lg2.members), "snapshots lost across resume"

print("\nSMOKE V11 OK: one-hot obs, mixed 2p/4p, mastery-gated every-recal eviction (no init mass-evict), resume round-trip")
