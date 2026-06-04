"""One-time generator for the native trainer's world-pool file (.owp).

Generates worlds with the OFFICIAL map + comet-schedule generator (so they're drawn from
the exact competition distribution) and serializes them to the C++-owned .owp format via
the `_native.write_world_pool` binding -- the same format the native exes read. This is the
ONLY Python step in the native train/eval loop; after running it, train.exe/eval.exe need
no Python.

    python scripts/gen_world_pool.py --n 2000 --seed 0      --out runs/native/train.owp
    python scripts/gen_world_pool.py --n 200  --seed 900000 --out runs/native/eval.owp
"""
from __future__ import annotations

import argparse
import os

from orbit_wars_rl import _native as native
from orbit_wars_rl.native_worldgen import generate_pool_cached


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-comets", action="store_true",
                    help="comet-free worlds (instant gen); eval pools should stay comet-aware")
    ap.add_argument("--out", default="runs/native/train.owp")
    args = ap.parse_args()

    pool = generate_pool_cached(args.n, base_seed=args.seed, comets=not args.no_comets)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    native.write_world_pool(pool, args.out)
    print(f"wrote {len(pool)} worlds (comets={not args.no_comets}) -> {args.out}")


if __name__ == "__main__":
    main()
