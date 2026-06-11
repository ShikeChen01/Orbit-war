"""End-to-end smoke of notebooks/setup1_v7_a100.ipynb: executes the REAL notebook cells in
order (config flipped to SMOKE + extra shrink overrides) so the full v7 path runs in minutes:
BC warm-start -> guardrailed PPO league training -> plot cell -> league export -> in-notebook
shape smoke. Mirrors scripts/smoke_v8_nb.py for the v7 (2p-only) notebook.

    python scripts/smoke_v7_nb.py
"""
import json
import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
os.environ["OW_CKPT_DIR"] = os.path.join("runs", "v7_smoke")
os.makedirs(os.environ["OW_CKPT_DIR"], exist_ok=True)

NB = os.path.join("notebooks", "legacy", "setup1_v7_a100.ipynb")
doc = json.load(open(NB, encoding="utf-8"))
ns = {"__name__": "nb"}

# post-config shrink (applied right after the config cell executes)
OVERRIDES = dict(
    TOTAL_ITERS=8, EPISODE_STEPS=96, N_WORLDS=48, B=8, FLEET_CAP=256,
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
T = ns["TOTAL_ITERS"]
assert len(hist["iter"]) == T, "expected %d iters, got %d" % (T, len(hist["iter"]))
kl = hist["approx_kl"]
warm = ns["VALUE_WARMUP_ITERS"]
assert all(abs(k) < 1e-9 for k in kl[:warm]), "critic-only warmup must show kl 0.000: %r" % kl[:warm]
assert all(math.isfinite(k) for k in kl), "non-finite KL: %r" % kl
assert any(k > 0.0 for k in kl[warm:]), "policy never updated after warmup: %r" % kl
assert all(k < 0.5 for k in kl), "KL blew past the guardrail band: %r" % kl
assert all(math.isfinite(gn) for gn in hist["grad_norm"]), "non-finite grad norm"
assert any(l > 0.0 for l in hist["lnch_per_step"]), "launch-rate zero -- frozen policy"
assert len(hist["learner_elo"]) == T
print("\n[smoke] kl=%s" % ["%.3f" % k for k in kl])
print("[smoke] launch/step=%s elo=%s" % (["%.1f" % l for l in hist["lnch_per_step"]],
                                         ["%.0f" % e for e in hist["learner_elo"]]))
assert os.path.exists(ns["BC_CKPT_PATH"]), "BC checkpoint missing"
assert os.path.exists(ns["TRAIN_STATE_PATH"]), "resumable train state missing"

# resume round-trip: ratings + snapshots must reload cleanly
net2, hist2, lg2 = ns["train"](total_iters=2, resume_from=ns["TRAIN_STATE_PATH"])
assert any(m["kind"] == "snapshot" for m in lg2.members), "snapshots lost across resume"
print("\nSMOKE V7 OK: BC warm-start, %d PPO iters, guardrails, league, resume round-trip" % T)
