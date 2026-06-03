"""Train a PPO agent locally.

    python scripts/train.py --total-steps 200000 --run-name exp1
    python scripts/train.py --device cpu --total-steps 5000   # quick check

Any TrainConfig field can be overridden with --field-name (underscores -> dashes).
"""
from __future__ import annotations

import argparse
import dataclasses

from orbit_wars_rl.train.config import TrainConfig
from orbit_wars_rl.train.ppo import PPOTrainer


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(TrainConfig):
        if f.type in ("tuple", "tuple[str, ...]", "tuple[float, ...]"):
            continue  # keep tuple defaults; tweak in code if needed
        flag = "--" + f.name.replace("_", "-")
        py_type = {"int": int, "float": float, "str": str, "bool": str}.get(f.type, str)
        ap.add_argument(flag, dest=f.name, type=py_type, default=None)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    cfg = dataclasses.replace(TrainConfig(), **overrides)
    print("config:", cfg)
    PPOTrainer(cfg).train()


if __name__ == "__main__":
    main()
