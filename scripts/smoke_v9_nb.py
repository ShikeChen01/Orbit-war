"""End-to-end smoke of notebooks/setup1_v9_a100.ipynb: executes the REAL notebook cells in
order (config flipped to SMOKE + extra shrink overrides) so the full v9 path runs in minutes:
BC warm-start -> mixed 2p/4p league training with ONE-HOT ownership obs (F_DIM 104), legacy
v7/v8-format invited members through the LegacyObsAdapter, dual Elo, bounded scheduler, mixed
gauntlet -> plot cell -> league export -> in-notebook shape smoke (incl. 4p). Mirrors
scripts/smoke_v8_nb.py.

    python scripts/smoke_v9_nb.py
"""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
os.environ["OW_CKPT_DIR"] = os.path.join("runs", "v9_smoke")
os.makedirs(os.environ["OW_CKPT_DIR"], exist_ok=True)

NB = os.path.join("notebooks", "setup1_v9_a100.ipynb")
doc = json.load(open(NB, encoding="utf-8"))
ns = {"__name__": "nb"}

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

assert ns["F_DIM"] == 104, "v9 must run the one-hot obs (F_DIM 104), got %r" % ns["F_DIM"]
hist = ns["hist"]
fmts = hist["fmt"]
n2, n4 = fmts.count(2), fmts.count(4)
print("\n[smoke] iterations: %d x 2p, %d x 4p (share_4p targets: %s)"
      % (n2, n4, ["%.2f" % s for s in hist["share_4p"]]))
assert n4 >= 1 and n2 >= 1, "expected BOTH formats in the mix"
assert n4 <= 6 * max(1, n2), "2p:4p ratio exceeded the 1:6 bound"
assert len(hist["learner_elo4"]) == len(hist["learner_elo"]) == ns["TOTAL_ITERS"]
assert "gauntlet2" in hist and "gauntlet4" in hist and len(hist["gauntlet4"]) > 0
lg = ns["league"]
inv = [m for m in lg.members if m["kind"] == "invited"]
print("[smoke] invited members: %s" % [(m["label"], m["elo"], m["elo4"]) for m in inv])
assert all(m["anchor"] for m in inv), "invited members must be Elo-anchored"
# legacy v7/v8-format members (F_DIM 101) must arrive WRAPPED in the obs adapter
legacy = [m for m in inv if int((m.get("cfg") or {}).get("F_DIM", 104)) == 101]
for m in legacy:
    assert type(m["net"]).__name__ == "LegacyObsAdapter", "legacy member %s not wrapped" % m["label"]
print("[smoke] legacy-format invited members wrapped: %d/%d" % (len(legacy), len(inv)))

# resume round-trip: league state (dual ratings + invited cfg + wrappers) must reload cleanly
net2, hist2, lg2 = ns["train"](total_iters=2, resume_from=ns["TRAIN_STATE_PATH"])
inv2 = [m for m in lg2.members if m["kind"] == "invited"]
assert len(inv2) == len(inv), "invited members lost across resume"
assert any(m["kind"] == "snapshot" for m in lg2.members), "snapshots lost across resume"
for m in inv2:
    if int((m.get("cfg") or {}).get("F_DIM", 104)) == 101:
        assert type(m["net"]).__name__ == "LegacyObsAdapter", "wrapper lost across resume"
print("\nSMOKE V9 OK: one-hot obs, mixed 2p/4p training, legacy-member adapter, resume round-trip")
