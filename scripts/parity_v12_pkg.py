"""Numeric parity: orbit_wars_v12 (package) vs notebooks/setup1_v12_a100.ipynb (source of truth).

Forces CPU (deterministic fp32), execs the notebook's DEFINITION cells into a namespace, then runs
ONE seeded SMOKE `collect_ppo` + `ppo_update` on each side under an identical reseed and a FIXED
scripted opponent, and asserts the rollout stats, update stats, transition-buffer summaries, and
the post-update weight checksum match. A single iteration is the strongest tractable bit-exactness
guarantee for the refactor (full multi-iter runs share the same code paths).

    python scripts/parity_v12_pkg.py
"""
import json
import os
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""        # force CPU -> deterministic, machine-independent
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ["OW_CKPT_DIR"] = os.path.join("runs", "v12_parity")
os.makedirs(os.environ["OW_CKPT_DIR"], exist_ok=True)

import numpy as np
import torch

# small, identical config on both sides (ARCH=trunk SMOKE default; AMP is a no-op on CPU)
PAR = dict(B=8, B_4P=8, EPISODE_STEPS=40, N_WORLDS=32, FLEET_CAP=128, MINIBATCHES=4, UPDATE_EPOCHS=2)
SEED = 0


def reseed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def _checksum(net):
    return float(sum(p.detach().double().abs().sum().item() for p in net.parameters()))


# ===================== notebook side =====================
NB = os.path.join("notebooks", "setup1_v12_a100.ipynb")
doc = json.load(open(NB, encoding="utf-8"))
ns = {"__name__": "nb"}


def _skip(src):
    return (src.startswith("net, hist, league = train(") or "LEAGUE_OUT" in src
            or src.startswith("# Training-health") or src.startswith("RUN_SMOKE")
            or src.startswith("# ## 13b. BC"))


for c in doc["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if "SMOKE = False " in src and "BC_ENABLED" in src:        # config cell: flip + shrink
        src = src.replace("SMOKE = False ", "SMOKE = True  ")
        exec(compile(src, "cfg", "exec"), ns)
        ns.update(PAR)
        continue
    if _skip(src):
        continue
    exec(compile(src, "nbcell", "exec"), ns)

assert ns["DEVICE"].type == "cpu", "parity test must run on CPU"
seats = [{"kind": "starter", "net": None}]

reseed()
net_nb = ns["build_policy"]()
opt_nb = torch.optim.Adam(net_nb.parameters(), lr=ns["LR"], eps=ns["ADAM_EPS"])
env_nb = ns["GpuEnv"](ns["PLANET_CAP"], ns["FLEET_CAP"], ns["EPISODE_STEPS"], ns["SHIP_SPEED"], ns["DEVICE"])
pool_nb = ns["make_world_pool"](ns["N_WORLDS"], base_seed=0)
ns["CUR_ENT_COEF"] = ns["ENT_COEF_START"]; ns["PPO_POLICY_COEF"] = 1.0
tb_nb, rs_nb, _ = ns["collect_ppo"](env_nb, net_nb, pool_nb, 0, random.Random(7), seats, n_players=2, n_envs=ns["B"])
us_nb = ns["ppo_update"](net_nb, opt_nb, tb_nb)
chk_nb = _checksum(net_nb)
tbsum_nb = dict(n=tb_nb["n"], old_logp=float(tb_nb["old_logp"].double().sum()),
                adv=float(tb_nb["advantage"].double().abs().sum()), ret=float(tb_nb["returns"].double().sum()))

# ===================== package side =====================
from orbit_wars_v12 import Config, build_policy, GpuEnv
from orbit_wars_v12.rollout import collect_ppo
from orbit_wars_v12.ppo import ppo_update

cfg = Config.create(smoke=True, **PAR)
assert cfg.device.type == "cpu", "parity test must run on CPU"

reseed()
net_pk = build_policy(cfg)
opt_pk = torch.optim.Adam(net_pk.parameters(), lr=cfg.LR, eps=cfg.ADAM_EPS)
env_pk = GpuEnv(cfg)
pool_pk = __import__("orbit_wars_v12.worldgen", fromlist=["make_world_pool"]).make_world_pool(cfg, cfg.N_WORLDS, base_seed=0)
tb_pk, rs_pk, _ = collect_ppo(cfg, env_pk, net_pk, pool_pk, 0, random.Random(7), seats, n_players=2, n_envs=cfg.B)
us_pk = ppo_update(cfg, net_pk, opt_pk, tb_pk, ent_coef=cfg.ENT_COEF_START, policy_coef=1.0)
chk_pk = _checksum(net_pk)
tbsum_pk = dict(n=tb_pk["n"], old_logp=float(tb_pk["old_logp"].double().sum()),
                adv=float(tb_pk["advantage"].double().abs().sum()), ret=float(tb_pk["returns"].double().sum()))

# ===================== compare =====================
def _cmp(label, a, b, keys, atol=1e-6):
    worst = 0.0
    for k in keys:
        d = abs(float(a[k]) - float(b[k]))
        worst = max(worst, d)
        print("    %-14s nb=%-16.8g pkg=%-16.8g  |d|=%.3g" % (k, a[k], b[k], d))
    print("  %s: max |diff| = %.3g  (%s)" % (label, worst, "EXACT" if worst == 0 else ("ok" if worst <= atol else "FAIL")))
    assert worst <= atol, "%s parity exceeded tol %g (max %g)" % (label, atol, worst)
    return worst


print("\n=== rollout stats (collect_ppo) ===")
_cmp("rollout", rs_nb, rs_pk, ["mean_return", "mean_len", "win_rate", "score", "inv_per_step",
                               "lnch_per_step", "valid_per_step", "transitions", "r_outcome",
                               "r_capture", "r_launch", "r_milestone"])
print("=== transition buffer ===")
_cmp("buffer", tbsum_nb, tbsum_pk, ["n", "old_logp", "adv", "ret"])
print("=== update stats (ppo_update) ===")
_cmp("update", us_nb, us_pk, ["total", "policy", "vf", "entropy", "approx_kl", "clipfrac", "grad_norm"])
print("=== post-update weight checksum ===")
print("    nb=%.10g  pkg=%.10g  |d|=%.3g" % (chk_nb, chk_pk, abs(chk_nb - chk_pk)))
assert abs(chk_nb - chk_pk) <= 1e-5, "weight checksum diverged: %g vs %g" % (chk_nb, chk_pk)

print("\nPARITY OK: package matches the notebook on a seeded single collect_ppo + ppo_update (CPU).")
