"""Inspect how a policy behaves over one game (is it passive? does it expand?).

Plays a checkpoint (player 0) vs a scripted opponent in the native engine via
`step_from_state` and prints planets/ships over time plus launch statistics -- the quick
"is this policy doing anything sane" check used to diagnose the passivity failure.

    python scripts/inspect_agent.py runs/native/run.pt --opponent starter --seed 777
"""
from __future__ import annotations

import argparse
import collections

from orbit_wars_rl import _native as native
from orbit_wars_rl.native_worldgen import generate_world
from orbit_wars_rl.agents.ppo_policy import PolicyAgent
from orbit_wars_rl.agents.scripted import RandomAgent, StarterAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--opponent", default="starter", choices=["starter", "random"])
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--episode-steps", type=int, default=500)
    args = ap.parse_args()

    ag = PolicyAgent.load(args.checkpoint, device="cpu")
    opp = StarterAgent() if args.opponent == "starter" else RandomAgent(seed=args.seed)
    w = generate_world(args.seed)
    state = {"planets": [p[:] for p in w["planets"]], "initial_planets": [p[:] for p in w["planets"]],
             "fleets": [], "comets": [], "comet_planet_ids": [], "angular_velocity": w["angular_velocity"],
             "next_fleet_id": 0, "step": 0, "num_agents": 2, "comet_schedule": w["comet_schedule"]}
    cfg = {"episodeSteps": args.episode_steps, "shipSpeed": 6.0, "cometSpeed": 4.0}

    launches = []
    fracs = collections.Counter()
    for t in range(args.episode_steps):
        a0 = ag.act(state, cfg)
        launches.append(len(a0))
        a1 = opp.act({**state, "player": 1}, cfg)
        nxt = native.step_from_state(state, [a0, a1])
        nxt["step"] = t + 1
        nxt["num_agents"] = 2
        nxt["comet_schedule"] = w["comet_schedule"]
        state = nxt
        if (t + 1) % 50 == 0 or t < 2:
            p0 = [p for p in state["planets"] if p[1] == 0]
            p1 = [p for p in state["planets"] if p[1] == 1]
            sh0 = sum(p[5] for p in p0) + sum(f[6] for f in state["fleets"] if f[1] == 0)
            sh1 = sum(p[5] for p in p1) + sum(f[6] for f in state["fleets"] if f[1] == 1)
            print(f"t={t+1:3d}  p0: {len(p0):2d} planets {sh0:6d} ships   "
                  f"p1: {len(p1):2d} planets {sh1:6d} ships   launches={len(a0)}")

    n = len(launches)
    print(f"\navg launches/turn={sum(launches)/n:.2f}  max={max(launches)}  "
          f"idle_turns={sum(1 for x in launches if x == 0)}/{n}")
    p0 = [p for p in state["planets"] if p[1] == 0]
    p1 = [p for p in state["planets"] if p[1] == 1]
    sh0 = sum(p[5] for p in p0) + sum(f[6] for f in state["fleets"] if f[1] == 0)
    sh1 = sum(p[5] for p in p1) + sum(f[6] for f in state["fleets"] if f[1] == 1)
    print(f"FINAL  p0={sh0} ({len(p0)} planets)  p1={sh1} ({len(p1)} planets)  "
          f"{'P0 WINS' if sh0 > sh1 else 'P1 WINS' if sh1 > sh0 else 'DRAW'}")


if __name__ == "__main__":
    main()
