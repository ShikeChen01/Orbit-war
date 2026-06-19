"""Record ONE 2p game of the transformer it426 (seat 0, blue) vs MLP-512x32 it1426 (seat 1, red)
in the v12 GpuEnv, then render it to GIF + keyframe PNG via scripts/render_recording.py.

The transformer loses this matchup ~92% head-up (see scripts/eval_tx_vs_mlps.py), so this picks a
REPRESENTATIVE loss: it scans a pool of worlds, takes the transformer's final ship-margin in each
(both greedy), and replays the loss whose margin is closest to the median loss (not a fluke blowout
nor a squeaker). That single world is then re-rolled tick-by-tick and dumped in the schema
render_recording.py expects (planets/fleets/comets per tick).

    .venv/Scripts/python scripts/render_tx_vs_it1426.py
    .venv/Scripts/python scripts/render_tx_vs_it1426.py --scan 48 --stride 3 --fps 12
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("MPLBACKEND", "Agg")

import json

import torch

from orbit_wars_v12 import Config, load_snapshot, make_world_pool
from orbit_wars_v12.constants import BOARD_SIZE, SUN_RADIUS
from orbit_wars_v12.env import GpuEnv, env_encode, env_step, settle
from orbit_wars_v12.policy import act

TX = "notebooks/weight/Transformers/512h-12n-16hd-4v/it426.pt"   # seat 0 (blue, the one we watch lose)
MLP = "notebooks/weight/MLPs/512x32/run3/it1426.pt"              # seat 1 (red)


def scan_margins(cfg, tx, mlp, worlds, n):
    """Batched greedy 2p game over `worlds`; return per-world final (s0 - s1) ship margin (tx=seat0)."""
    dev = cfg.device
    env = GpuEnv(cfg)
    env.reset([worlds[i % len(worlds)] for i in range(n)], n_players=2)
    active = torch.ones(n, device=dev)
    margin = torch.zeros(n, device=dev)
    with torch.no_grad():
        for t in range(env.T):
            a0 = act(cfg, tx, *env_encode(env, 0), greedy=True)[0]
            a1 = act(cfg, mlp, *env_encode(env, 1), greedy=True)[0]
            env_step(cfg, env, a0, seats=[{"pid": 1, "action": a1}], step_idx=t)
            s0, s1, al0, al1 = settle(env)
            term = (env.step_ct >= float(env.T - 2)) | (~(al0 & al1))
            newly = (active > 0.5) & term
            margin = torch.where(newly, s0 - s1, margin)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s0, s1, _, _ = settle(env)
        margin = torch.where(active > 0.5, s0 - s1, margin)
    return margin


def snap_frame(env, t, b=0):
    """Extract env's world `b` into the render_recording per-tick schema."""
    pa = env.p_alive[b] > 0.5
    planets, comets = [], []
    pis = env.p_is_comet[b]
    for i in torch.nonzero(pa, as_tuple=False).flatten().tolist():
        owner = int(env.p_owner[b, i].item())
        planets.append([i, owner, float(env.p_x[b, i]), float(env.p_y[b, i]),
                        float(env.p_radius[b, i]), int(round(float(env.p_ships[b, i]))),
                        int(round(float(env.p_prod[b, i])))])
        if pis[i].item() > 0.5:
            comets.append(i)
    fa = env.f_alive[b] > 0.5
    fleets = []
    for j in torch.nonzero(fa, as_tuple=False).flatten().tolist():
        fleets.append([int(env.f_owner[b, j].item()), float(env.f_x[b, j]), float(env.f_y[b, j]),
                       float(env.f_angle[b, j]), int(round(float(env.f_ships[b, j])))])
    return {"t": t, "planets": planets, "fleets": fleets, "comets": comets}


def record_one(cfg, tx, mlp, world):
    """Replay a single world tick-by-tick (B=1), recording every tick. Returns (ticks, final margin)."""
    env = GpuEnv(cfg)
    env.reset([world], n_players=2)
    ticks = [snap_frame(env, 0)]
    margin = 0.0
    with torch.no_grad():
        for t in range(env.T):
            a0 = act(cfg, tx, *env_encode(env, 0), greedy=True)[0]
            a1 = act(cfg, mlp, *env_encode(env, 1), greedy=True)[0]
            env_step(cfg, env, a0, seats=[{"pid": 1, "action": a1}], step_idx=t)
            ticks.append(snap_frame(env, t + 1))
            s0, s1, al0, al1 = settle(env)
            margin = float((s0 - s1)[0].item())
            if bool((env.step_ct[0] >= float(env.T - 2)) | (~(al0[0] & al1[0]))):
                break
    return ticks, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=48, help="worlds to scan for a representative loss")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=None, help="force a specific world index (skip scan)")
    ap.add_argument("--out", default=os.path.join("runs", "viz", "tx_it426_vs_it1426"))
    args = ap.parse_args()

    cfg = Config.create(smoke=False, CKPT_DIR=os.path.join("runs", "tx_eval"))
    dev = cfg.device
    tx, _ = load_snapshot(cfg, TX); tx = tx.to(dev).eval()
    mlp, _ = load_snapshot(cfg, MLP); mlp = mlp.to(dev).eval()
    worlds = make_world_pool(cfg, args.scan, base_seed=cfg.SEED)
    print("loaded tx-512h12L it426 (seat0/blue) vs mlp-512x32 it1426 (seat1/red) | device %s" % dev)

    if args.seed is not None:
        pick = args.seed % len(worlds)
        print("forced world index %d" % pick)
    else:
        margins = scan_margins(cfg, tx, mlp, worlds, args.scan)
        m = margins.tolist()
        losses = [(i, v) for i, v in enumerate(m) if v < 0]
        wins = sum(1 for v in m if v > 0)
        print("scan: %d worlds | tx wins %d, losses %d, draws %d | margin min/med/max = %.1f / %.1f / %.1f"
              % (len(m), wins, len(losses), len(m) - wins - len(losses),
                 min(m), sorted(m)[len(m) // 2], max(m)))
        if not losses:
            print("!! transformer did not lose any scanned world; rendering its worst margin instead")
            pick = min(range(len(m)), key=lambda i: m[i])
        else:
            med = sorted(v for _, v in losses)[len(losses) // 2]   # median loss margin
            pick = min(losses, key=lambda iv: abs(iv[1] - med))[0]
        print("picked world index %d (final margin tx-opp = %.1f ships)" % (pick, m[pick]))

    ticks, margin = record_one(cfg, tx, mlp, worlds[pick])
    outcome = 1.0 if margin > 0 else (-1.0 if margin < 0 else 0.0)
    doc = {"board_size": BOARD_SIZE, "sun_radius": SUN_RADIUS,
           "ego": "tx-512h12L it426", "opp": "mlp-512x32 it1426",
           "outcome": outcome, "final_margin": margin, "ticks": ticks}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json_path = args.out + ".json"
    with open(json_path, "w") as fh:
        json.dump(doc, fh)
    print("recorded %d ticks | final margin tx-opp = %.1f (%s) -> %s"
          % (len(ticks), margin, "LOSS" if margin < 0 else ("win" if margin > 0 else "draw"), json_path))

    cmd = [sys.executable, os.path.join("scripts", "render_recording.py"), json_path,
           "--out", args.out, "--stride", str(args.stride), "--fps", str(args.fps)]
    print("rendering:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("\nDone:")
    print("  GIF: %s" % os.path.abspath(args.out + ".gif"))
    print("  PNG: %s" % os.path.abspath(args.out + "_keyframes.png"))


if __name__ == "__main__":
    main()
