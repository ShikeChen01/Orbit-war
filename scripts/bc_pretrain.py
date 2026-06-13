"""Behavior-cloning warm start from the scripted `starter` into a TARGET-MODE policy.

Target-mode actions make starter's move exactly representable: "from planet P, send half
the ships at the nearest non-owned static planet" == (target_row of that planet, frac=0.5).
We roll out starter-vs-random games (native engine, fast), label every actionable planet
with starter's choice, and train the EntityPolicy by cross-entropy. The result plays at
~starter level immediately -- a launch pad for PPO/self-play to surpass it, skipping the
hard exploration that left from-scratch RL stuck at ~random level vs starter.

    python scripts/bc_pretrain.py --seeds 400 --epochs 8 --out runs/native/bc_start.pt
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from orbit_wars_rl import _native as native
from orbit_wars_rl.agents.ppo_policy import EntityPolicy
from orbit_wars_rl.agents.scripted import RandomAgent, StarterAgent
from orbit_wars_rl.env.game import CENTER, ROTATION_RADIUS_LIMIT
from orbit_wars_rl.native_worldgen import generate_world
from orbit_wars_rl.processors.observation import EntityObservation, N_ENTITY_FEATURES, N_GLOBAL_FEATURES

FRACTIONS = [0.25, 0.5, 0.75, 1.0]
FRAC_HALF = FRACTIONS.index(0.5)  # starter sends half
MAX_ENTITIES = 64
MIN_SHIPS = 20


def starter_target_id(planets, player, mp):
    """The planet id starter would target from planet `mp` (nearest non-owned static), or None."""
    if mp[1] != player or mp[5] <= 0 or mp[5] // 2 < MIN_SHIPS:
        return None
    best, closest = float("inf"), None
    for t in planets:
        if t[1] == player:
            continue
        if math.hypot(t[2] - CENTER, t[3] - CENTER) + t[4] < ROTATION_RADIUS_LIMIT:
            continue  # not static
        d = math.hypot(mp[2] - t[2], mp[3] - t[3])
        if d < best:
            best, closest = d, t
    return None if closest is None else closest[0]


def label_state(obs, proc):
    """Return (obs_arrays, labels[max_entities], loss_mask[max_entities]) for player 0."""
    obs0 = dict(obs)
    obs0["player"] = 0
    arrays, ctx = proc.process(obs0, {"episodeSteps": 500})
    planets = obs.get("planets", [])
    id_to_row = {int(pid): r for r, pid in enumerate(ctx["planet_ids"]) if int(pid) >= 0}
    labels = np.zeros((MAX_ENTITIES,), np.int64)
    loss_mask = np.zeros((MAX_ENTITIES,), np.float32)
    nf = len(FRACTIONS)
    for r in range(MAX_ENTITIES):
        if ctx["actionable"][r] < 0.5:
            continue
        loss_mask[r] = 1.0  # train on every actionable planet (noop is a valid label)
        pid = int(ctx["planet_ids"][r])
        mp = next((p for p in planets if int(p[0]) == pid), None)
        if mp is None:
            continue
        tid = starter_target_id(planets, 0, mp)
        if tid is not None and int(tid) in id_to_row:
            t_row = id_to_row[int(tid)]
            labels[r] = 1 + t_row * nf + FRAC_HALF
        # else: label stays 0 (noop) -- starter wouldn't launch from here yet
    return arrays, labels, loss_mask


def collect(seeds, steps, sample_every):
    """Roll out starter(p0) vs random(p1) and collect labeled states."""
    proc = EntityObservation()
    ent, em, am, gl, lab, lm = [], [], [], [], [], []
    for s in range(seeds):
        w = generate_world(s)
        state = {"planets": [p[:] for p in w["planets"]], "initial_planets": [p[:] for p in w["planets"]],
                 "fleets": [], "comets": [], "comet_planet_ids": [], "angular_velocity": w["angular_velocity"],
                 "next_fleet_id": 0, "step": 0, "num_agents": 2, "comet_schedule": w["comet_schedule"]}
        starter, rnd = StarterAgent(), RandomAgent(seed=s)
        for t in range(steps):
            a0 = starter.act({**state, "player": 0}, {"episodeSteps": 500})
            if t % sample_every == 0:
                arrays, labels, loss_mask = label_state(state, proc)
                if loss_mask.sum() > 0:
                    ent.append(arrays["entities"]); em.append(arrays["entity_mask"])
                    am.append(arrays["action_mask"]); gl.append(arrays["globals"])
                    lab.append(labels); lm.append(loss_mask)
            a1 = rnd.act({**state, "player": 1}, {"episodeSteps": 500})
            nxt = native.step_from_state(state, [a0, a1])
            nxt["step"] = t + 1; nxt["num_agents"] = 2; nxt["comet_schedule"] = w["comet_schedule"]
            state = nxt
            alive = {p[1] for p in state["planets"] if p[1] != -1} | {f[1] for f in state["fleets"]}
            if len(alive) <= 1:
                break
    return (np.stack(ent), np.stack(em), np.stack(am), np.stack(gl),
            np.stack(lab), np.stack(lm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--sample-every", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--launch-weight", type=float, default=12.0,
                    help="cap on the loss upweight for launch (vs noop) labels")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/native/bc_start.pt")
    args = ap.parse_args()

    import os
    # Feature count in the key: the cache stores encoded obs, so it must invalidate whenever
    # the encoder feature layout changes (e.g. the threat features bumped F 15 -> 20).
    cache = f"runs/_worldcache/bc_data_s{args.seeds}_t{args.steps}_e{args.sample_every}_f{N_ENTITY_FEATURES}.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        ent, em, am, gl, lab, lm = d["ent"], d["em"], d["am"], d["gl"], d["lab"], d["lm"]
        print(f"loaded {len(ent)} BC states from cache", flush=True)
    else:
        print(f"collecting BC data from {args.seeds} starter-vs-random games...", flush=True)
        t0 = time.perf_counter()
        ent, em, am, gl, lab, lm = collect(args.seeds, args.steps, args.sample_every)
        print(f"  {len(ent)} states in {time.perf_counter()-t0:.1f}s", flush=True)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, ent=ent, em=em, am=am, gl=gl, lab=lab, lm=lm)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    A = 1 + MAX_ENTITIES * len(FRACTIONS)
    pol = EntityPolicy(N_ENTITY_FEATURES, N_GLOBAL_FEATURES, A, args.hidden,
                       target_mode=True, num_fracs=len(FRACTIONS)).to(dev)
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    T = {k: torch.from_numpy(v).to(dev) for k, v in
         dict(ent=ent, em=em, am=am, gl=gl, lab=lab, lm=lm).items()}
    N = len(ent)

    # Class imbalance: most actionable planets aren't launching (starter waits for >=40
    # ships), so noop dominates and unweighted CE collapses to "always noop" (passive).
    # Upweight launch labels so BC actually learns the launches that matter.
    all_lab = T["lab"].reshape(-1)[T["lm"].reshape(-1) > 0.5]
    frac_launch = max(1e-3, float((all_lab > 0).float().mean()))
    launch_w = min(args.launch_weight, (1 - frac_launch) / frac_launch)
    print(f"launch fraction={frac_launch:.3f} -> launch_weight={launch_w:.2f}", flush=True)

    for ep in range(args.epochs):
        perm = torch.randperm(N, device=dev)
        tot = wsum = 0.0
        c_all = n_all = c_lau = n_lau = 0
        for i in range(0, N, args.batch):
            idx = perm[i:i + args.batch]
            obs = {"entities": T["ent"][idx], "entity_mask": T["em"][idx],
                   "action_mask": T["am"][idx], "globals": T["gl"][idx]}
            logits, _ = pol.forward(obs)
            B, E, Acl = logits.shape
            m = T["lm"][idx].reshape(-1) > 0.5
            lg = logits.reshape(-1, Acl)[m]
            tg = T["lab"][idx].reshape(-1)[m]
            w = torch.where(tg > 0, torch.full_like(tg, launch_w, dtype=torch.float), torch.ones_like(tg, dtype=torch.float))
            ce = F.cross_entropy(lg, tg, reduction="none")
            loss = (ce * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += (ce * w).sum().item(); wsum += w.sum().item()
            pred = lg.argmax(-1)
            c_all += (pred == tg).sum().item(); n_all += len(tg)
            lmask = tg > 0
            c_lau += (pred[lmask] == tg[lmask]).sum().item(); n_lau += int(lmask.sum().item())
        print(f"epoch {ep}: loss={tot/wsum:.4f} acc={c_all/n_all:.3f} "
              f"launch_acc={c_lau/max(1,n_lau):.3f}", flush=True)

    cfg = dict(n_entity_features=N_ENTITY_FEATURES, n_global_features=N_GLOBAL_FEATURES,
               actions_per_entity=A, hidden=args.hidden, target_mode=True, num_fracs=len(FRACTIONS))
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"policy_state": pol.state_dict(), "policy_config": cfg}, args.out)
    print(f"saved BC checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
