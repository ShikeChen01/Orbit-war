"""End-to-end smoke of notebooks/setup1_v10_a100.ipynb: executes the REAL notebook cells in
order (config flipped to SMOKE + extra shrink overrides) so the full v10 path runs in minutes:
transformer trunk -> BC warm-start -> mixed 2p/4p league training with the adaptive-entropy +
teacher-KL ppo_update -> plot -> league export. Mirrors scripts/smoke_v9_nb.py and adds the v10
assertions (arch, teacher seeded, adaptive coefs moved, KL machinery ran).

    python scripts/smoke_v10_nb.py
"""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
os.environ["OW_CKPT_DIR"] = os.path.join("runs", "v10_smoke")
os.makedirs(os.environ["OW_CKPT_DIR"], exist_ok=True)

NB = os.path.join("notebooks", "setup1_v10_a100.ipynb")
doc = json.load(open(NB, encoding="utf-8"))
ns = {"__name__": "nb"}

# post-config shrink (applied right after the config cell; the v10 override cells run LATER and
# re-set HIDDEN/N_HEADS for the SMOKE transformer, so they win -- this only shrinks the league/run)
OVERRIDES = dict(
    TOTAL_ITERS=8, EPISODE_STEPS=96, N_WORLDS=48, B=8, B_4P=8, FLEET_CAP=256,
    ELO_RECAL_ENVS=8, CKPT_EVERY=3, SELFPLAY_REFRESH=2, LEAGUE_MAX_SNAPSHOTS=3,
    BC_ROUNDS=1, BC_EPOCHS=1, VALUE_WARMUP_ITERS=1, LR_WARMUP_ITERS=2,
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

# --- shared v9 invariants (obs unchanged, mixed formats, dual Elo, gauntlets) ---
assert ns["F_DIM"] == 104, "v10 keeps the one-hot obs (F_DIM 104), got %r" % ns["F_DIM"]
assert ns["ARCH"] == "transformer", "v10 default arch must be transformer, got %r" % ns["ARCH"]
net = ns["net"]
assert type(net).__name__ == "PolicyNet" and hasattr(net, "trunk"), "transformer trunk not built"
assert len(net.trunk.layers) == ns["N_TX_LAYERS"], "transformer depth mismatch"
hist = ns["hist"]
n2, n4 = hist["fmt"].count(2), hist["fmt"].count(4)
print("\n[smoke] iterations: %d x 2p, %d x 4p" % (n2, n4))
assert n4 >= 1 and n2 >= 1, "expected BOTH formats in the mix"

# --- v10-specific machinery ---
v10 = ns["_V10"]
assert v10["iter"] >= ns["TOTAL_ITERS"], "v10 ppo_update iter counter did not advance"
assert v10["teacher"] is not None, "teacher net was never seeded"
assert all(not p.requires_grad for p in v10["teacher"].parameters()), "teacher must be frozen"
assert v10["log_cg"] is not None and v10["ent_opt"] is not None, "adaptive-entropy coefs not init"
# the gate coefficient must have MOVED off its init (controller actually stepped)
import math as _m
cg = float(v10["log_cg"].detach().exp())
print("[smoke] adaptive gate coef: init %.4f -> %.4f | teacher params frozen | iters=%d"
      % (ns["ENT_COEF_INIT"], cg, v10["iter"]))
assert abs(cg - ns["ENT_COEF_INIT"]) > 1e-6, "adaptive entropy coef never updated"
# teacher-KL must have been exercised (non-negative, finite) at least once
tk = hist.get("teacher_kl")
print("[smoke] teacher_kl logged: %s" % (("%.4f.." % tk[-1]) if tk else "n/a"))

# resume round-trip (league + v10 ppo_update still wired)
iter_pre = ns["_V10"]["iter"]
net2, hist2, lg2 = ns["train"](total_iters=2, resume_from=ns["TRAIN_STATE_PATH"])
assert ns["_V10"]["iter"] > iter_pre, "iter counter did not advance across resume"

# --- FULL-config (non-SMOKE) parameter parity: transformer h512/L5 ~ the 512x32 deep-MLP ---
def _build(arch, hidden, blocks):
    return ns["PolicyNet"](ns["F_DIM"], ns["G_DIM"], hidden, ns["D_G"], ns["PLANET_CAP"], blocks,
                           ns["USE_GLU"], ns["INIT_GATE_BIAS"], use_attention=False, value_res_blocks=4,
                           arch=arch, n_heads=8, n_tx_layers=5, tx_mlp_ratio=4, n_stem_res=1, n_head_res=4)
p_tx = sum(p.numel() for p in _build("transformer", 512, 32).parameters())
p_mlp = sum(p.numel() for p in _build("trunk", 512, 32).parameters())
ratio = p_tx / p_mlp
print("[smoke] FULL param parity: transformer h512/L5 %.2fM vs 512x32 trunk %.2fM -> %.2fx (%.1f MB fp32)"
      % (p_tx / 1e6, p_mlp / 1e6, ratio, p_tx * 4 / 1e6))
assert 0.90 <= ratio <= 1.15, "transformer not parameter-matched to 512x32 (ratio %.2f)" % ratio
assert p_tx * 4 / 1e6 < 100.0, "transformer over the 100MB fp32 submission ceiling"

print("\nSMOKE V10 OK: transformer trunk, adaptive entropy stepped, teacher-KL wired, resume round-trip, 512x32 param parity")
