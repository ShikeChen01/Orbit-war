"""Small hyperparameter sweep over the native trainer.

Runs each config as an isolated `train_native.py` subprocess (fresh GPU state, no leak
across runs), reusing the disk-cached world pool so only the first run pays world-gen.
Parses the final 'best vs starter' from each run and prints a leaderboard.

    python scripts/sweep_native.py --steps 1500000 --tag swp1
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

PY = sys.executable

# Each dict overrides train_native.py defaults. Keep it small; one-factor-around-a-base.
BASE = dict(prod_weight=20, ent_coef=0.02, angle_bins=16, lr=3e-4)
GRID = [
    {"name": "base"},
    {"name": "prod10", "prod_weight": 10},
    {"name": "prod40", "prod_weight": 40},
    {"name": "ent04", "ent_coef": 0.04},
    {"name": "ent01", "ent_coef": 0.01},
    {"name": "bins32", "angle_bins": 32},
    {"name": "bins64", "angle_bins": 64},
]


def run_one(cfg, steps, num_envs, eval_every, eval_games, seed, out):
    params = {**BASE, **{k: v for k, v in cfg.items() if k != "name"}}
    cmd = [PY, "scripts/train_native.py", "--total-steps", str(steps), "--num-envs",
           str(num_envs), "--eval-every", str(eval_every), "--eval-games", str(eval_games),
           "--selfplay-start-step", str(10**9), "--opp-self", "0", "--seed", str(seed),
           "--out", out]
    for k, v in params.items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    out_txt = p.stdout + p.stderr
    m = re.search(r"best vs starter=([\d.]+)%", out_txt)
    best = float(m.group(1)) if m else -1.0
    # last eval row's vs-random + margin
    rows = [ln for ln in out_txt.splitlines() if re.match(r"\s*\d+\s+[-\d.]", ln)]
    last = rows[-1] if rows else ""
    return best, last.strip(), dt, out_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=500_000)
    ap.add_argument("--eval-games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="swp")
    args = ap.parse_args()

    results = []
    for cfg in GRID:
        out = f"runs/native/{args.tag}_{cfg['name']}.pt"
        print(f"\n=== {cfg['name']}: {cfg} ===", flush=True)
        best, last, dt, _ = run_one(cfg, args.steps, args.num_envs, args.eval_every,
                                    args.eval_games, args.seed, out)
        print(f"  best vs starter = {best:.1f}%  ({dt/60:.1f} min)  last: {last}", flush=True)
        results.append((cfg["name"], best, last, dt))

    results.sort(key=lambda r: -r[1])
    print("\n==== SWEEP LEADERBOARD (best win% vs starter) ====")
    for name, best, last, dt in results:
        print(f"  {name:10} {best:6.1f}%   {last}")


if __name__ == "__main__":
    main()
