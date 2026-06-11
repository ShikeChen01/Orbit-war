"""Headless v6 experiment runner.

Execs notebooks/setup1_v6_ppo.ipynb code cells (stopping before the run cell),
applies config overrides right after the CONFIG cell, then calls train().

    python scripts/exp_v6.py --out runs/v6/smoke --iters 12 \
        --set HIDDEN=256 --set N_TX_LAYERS=2 --set NUM_GROUPS=8 --set GROUP_SIZE=8

Overrides are python literals (ints/floats/bools/strings/tuples). B is recomputed
from NUM_GROUPS*GROUP_SIZE after overrides. Heartbeat goes to stdout; metrics.json
+ checkpoints land in --out (via OW_CKPT_DIR).
"""
from __future__ import annotations
import argparse, ast, json, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(REPO, "notebooks", "legacy", "setup1_v6_ppo.ipynb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run dir (checkpoints + metrics.json)")
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="config override, python literal (repeatable)")
    ap.add_argument("--resume", default=None, help="train-state path to resume from")
    ap.add_argument("--bc", action="store_true", help="run bc_pretrain() first, then train from it")
    ap.add_argument("--bc-only", action="store_true", help="run bc_pretrain() and exit")
    ap.add_argument("--nb", default=NB)
    args = ap.parse_args()

    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)
    os.environ["OW_CKPT_DIR"] = outdir          # config cell reads this for CKPT_DIR

    overrides = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        try:
            overrides[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            overrides[k] = v                     # bare string (e.g. ARCH=trunk)

    doc = json.load(open(args.nb, encoding="utf-8"))
    cells = [("".join(c["source"])) for c in doc["cells"] if c["cell_type"] == "code"]

    g = {"__name__": "__exp__"}
    applied = False
    for i, src in enumerate(cells):
        if "= train(" in src and "def train(" not in src:
            break                                # stop before the run cell
        exec(compile(src, f"nb_cell_{i}", "exec"), g)
        if not applied and "SMOKE" in g:         # CONFIG cell just ran -> apply overrides
            g.update(overrides)
            g["B"] = g["NUM_GROUPS"] * g["GROUP_SIZE"]
            # config cell already wrote CKPT paths from OW_CKPT_DIR; reseed if SEED overridden
            if "SEED" in overrides:
                import random, numpy as np, torch
                random.seed(g["SEED"]); np.random.seed(g["SEED"]); torch.manual_seed(g["SEED"])
            applied = True
            print("[exp_v6] overrides:", overrides, flush=True)
            print("[exp_v6] B=%d HIDDEN=%d ARCH=%s L=%s iters=%d out=%s" % (
                g["B"], g["HIDDEN"], g["ARCH"], g.get("N_TX_LAYERS"), args.iters, outdir), flush=True)

    resume = args.resume or g.get("RESUME_FROM")   # BC_ENABLED cell auto-chains via RESUME_FROM
    if args.bc or args.bc_only:
        t0 = time.time()
        bc_net = g["bc_pretrain"]()
        print("[exp_v6] BC done in %.1fs -> %s" % (time.time() - t0, g["BC_CKPT_PATH"]), flush=True)
        ew = g["make_world_pool"](64, base_seed=g["SEED"] + 777)
        for a in ("random", "starter", "greedy", "medium"):
            sc = g["_eval_score"](bc_net, a, ew, 64)
            print("[exp_v6] BC clone vs %-8s score %.3f (win+0.5*draw, greedy deploy, 64 envs)" % (a, sc), flush=True)
        if args.bc_only:
            return
        resume = g["BC_CKPT_PATH"]

    t0 = time.time()
    net, hist, league = g["train"](total_iters=args.iters, log_every=1, resume_from=resume)
    dt = time.time() - t0
    # train() itself saves CKPT_PATH, TRAIN_STATE_PATH and metrics.json into outdir
    print("[exp_v6] done: %d iters in %.1fs (%.1fs/iter)" % (args.iters, dt, dt / max(1, args.iters)), flush=True)


if __name__ == "__main__":
    main()
