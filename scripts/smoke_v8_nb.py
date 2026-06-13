"""End-to-end smoke of notebooks/setup1_v8_a100.ipynb: executes the REAL notebook cells in
order (config flipped to SMOKE + extra shrink overrides) so the full v8 path runs in minutes:
BC warm-start -> mixed 2p/4p league training (invited-member discovery, dual Elo, bounded
scheduler, mixed gauntlet) -> plot cell -> league export -> in-notebook shape smoke (incl. 4p).

    python scripts/smoke_v8_nb.py
"""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
os.environ["OW_CKPT_DIR"] = os.path.join("runs", "v8_smoke")
os.makedirs(os.environ["OW_CKPT_DIR"], exist_ok=True)

NB = os.path.join("notebooks", "setup1_v8_a100.ipynb")
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

# weak-point matchmaking: synthetic detector check. Member 400 Elo below the learner ->
# expected ~0.91; actual EMA 0.70 -> gap ~0.21 -> boost capped at x3. Overperforming -> x1.
snap = next(m for m in lg.members if m["kind"] == "snapshot")
snap["elo"] = lg.learner_elo - 400.0
snap["wr"], snap["n"] = 0.70, 10
wb = lg._weak_boost(snap, 2)
assert abs(wb - ns["WEAK_BOOST_MAX"]) < 1e-9, "weak boost expected x%.1f, got %r" % (ns["WEAK_BOOST_MAX"], wb)
i = lg.members.index(snap)
assert lg._weights(2)[i] > lg._pfsp_weight(lg._p_beat(snap, 2)) * 2.5, "weak boost not applied in weights"
snap["wr"] = 0.95
assert lg._weak_boost(snap, 2) == 1.0, "boost must not fire when overperforming"
print("[smoke] weak-point matchmaking: x%.1f on the 0.91-expected/0.70-actual gap; x1.0 otherwise" % wb)

# resume round-trip: league state (dual ratings + invited cfg) must reload cleanly
net2, hist2, lg2 = ns["train"](total_iters=2, resume_from=ns["TRAIN_STATE_PATH"])
inv2 = [m for m in lg2.members if m["kind"] == "invited"]
assert len(inv2) == len(inv), "invited members lost across resume"
assert any(m["kind"] == "snapshot" for m in lg2.members), "snapshots lost across resume"
print("\nSMOKE V8 OK: mixed 2p/4p training, dual Elo, invited members, resume round-trip")
