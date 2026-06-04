"""Value-guided lookahead agent: the inference-time edge.

Kaggle inference is CPU-bound with a ~1s/turn budget, and we have a bit-exact, fast
forward model (the native step). So instead of playing the raw policy greedily, we:
  1. ask the policy for K candidate full-turn actions (greedy + samples),
  2. roll each forward H steps in the engine (both seats driven by the policy),
  3. score the resulting state with the value head (or a heuristic potential),
  4. play the best candidate.
This catches actions that look good locally but lead to bad positions, and lets a fixed
policy spend compute to play better -- exactly what the inference budget is for.

The forward model here is `native.step_from_state` (fast, local). For a Kaggle submission
the same logic runs on a pure-Python port of the step (the native .pyd isn't on the box);
the planning is identical. Future comet *spawns* are unknown to agents, so the rollout
assumes no new comets (existing ones on the board move along their known paths).

    python scripts/search_agent.py runs/native/sp1.pt --opponent starter --episodes 20 --K 8 --H 6
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from orbit_wars_rl.agents.base import Agent
from orbit_wars_rl.agents.ppo_policy import EntityPolicy, obs_to_tensors
from orbit_wars_rl.processors.action import PerPlanetAction, DEFAULT_FRACTIONS
from orbit_wars_rl.processors.observation import EntityObservation

PROD_WEIGHT = 20.0


def _heuristic_value(state: dict, player: int) -> float:
    s0 = s1 = 0
    pr0 = pr1 = 0.0
    for p in state["planets"]:
        if p[1] == player:
            s0 += p[5]; pr0 += p[6]
        elif p[1] == (1 - player):
            s1 += p[5]; pr1 += p[6]
    for f in state["fleets"]:
        if f[1] == player:
            s0 += f[6]
        elif f[1] == (1 - player):
            s1 += f[6]
    return (s0 - s1) + PROD_WEIGHT * (pr0 - pr1)


class SearchAgent(Agent):
    name = "search"

    def __init__(self, checkpoint: str, device: str = "cpu", K: int = 8, H: int = 6,
                 value: str = "head", opponent_model: str = "policy", episode_steps: int = 500,
                 temperature: float = 1.5, fast_encode: bool = True, mode: str = "joint",
                 backend: str = "native", time_budget: float = 0.85):
        ckpt = torch.load(checkpoint, map_location=device)
        cfg = ckpt["policy_config"]
        self.policy = EntityPolicy(**cfg).to(device).eval()
        self.policy.load_state_dict(ckpt["policy_state"])
        self.device = torch.device(device)
        self.target_mode = bool(cfg.get("target_mode", False))
        nf = len(DEFAULT_FRACTIONS)
        n_choices = (cfg["actions_per_entity"] - 1) // nf
        if self.target_mode:
            self.act_proc = PerPlanetAction(max_entities=n_choices, target_mode=True)
        else:
            self.act_proc = PerPlanetAction(angle_bins=n_choices)
        self.obs_proc = EntityObservation(max_entities=self.act_proc.max_entities,
                                          episode_steps=episode_steps)
        self.K, self.H, self.value_mode, self.opp_model = K, H, value, opponent_model
        self.episode_steps = episode_steps
        self.temperature = temperature
        self.fast_encode = fast_encode
        self.mode = mode
        self.backend = backend  # "native" (fast, local) or "py" (py_engine, Kaggle submission)
        self.time_budget = time_budget  # per-turn wall-clock guard (s); fall back to greedy

    # ---- policy helpers -----------------------------------------------------
    def _logits_value(self, state: dict, player: int):
        if self.fast_encode:
            arr, ctx = self._encode_fast(state, player)
        else:
            o = dict(state); o["player"] = player
            arr, ctx = self.obs_proc.process(o, {"episodeSteps": self.episode_steps})
        t = obs_to_tensors(arr, self.device, batched=False)
        with torch.no_grad():
            logits, value = self.policy.forward(t)
        return logits.squeeze(0), float(value.item()), ctx

    def _encode_fast(self, state: dict, player: int):
        """Features from the fast C++ encoder (parity-exact to EntityObservation); context
        (row->id/ships/x/y, actionable) computed cheaply in Python with the same own-first-
        then-by-id ordering. Avoids the O(fleets*planets) Python pressure loop -- the search
        bottleneck. (Local only; a Kaggle submission uses the Python encoder.)"""
        ME = self.act_proc.max_entities
        planets = state["planets"]
        order = sorted(range(len(planets)),
                       key=lambda i: (0 if planets[i][1] == player else 1, planets[i][0]))[:ME]
        planet_ids = np.full(ME, -1, np.int64)
        planet_ships = np.zeros(ME, np.int64)
        px = np.zeros(ME); py = np.zeros(ME); actionable = np.zeros(ME, np.float32)
        for row, i in enumerate(order):
            p = planets[i]
            planet_ids[row] = p[0]; planet_ships[row] = p[5]; px[row] = p[2]; py[row] = p[3]
            if p[1] == player and p[5] > 0:
                actionable[row] = 1.0
        from orbit_wars_rl import _native as native
        ent, em, am, gl = native.encode_state(state, player, ME, self.episode_steps)
        arr = {"entities": ent, "entity_mask": em, "action_mask": am, "globals": gl}
        ctx = {"planet_ids": planet_ids, "planet_ships": planet_ships,
               "planet_x": px, "planet_y": py, "actionable": actionable}
        return arr, ctx

    def _decode(self, classes, ctx):
        return self.act_proc.decode(np.asarray(classes), ctx)

    def _greedy_move(self, state: dict, player: int):
        logits, _, ctx = self._logits_value(state, player)
        return self._decode(logits.argmax(-1).cpu().numpy(), ctx)

    def _candidates(self, state: dict, player: int):
        logits, _, ctx = self._logits_value(state, player)
        E = logits.shape[0]
        cands = [self._decode(logits.argmax(-1).cpu().numpy(), ctx)]  # greedy
        cands.append([])  # all-noop (pure defense) -- the key alternative to over-extension
        probs = torch.softmax(logits / self.temperature, dim=-1)  # hotter -> more diverse
        for _ in range(max(0, self.K - 2)):
            sampled = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()
            cands.append(self._decode(sampled, ctx))
        return cands

    # ---- rollout ------------------------------------------------------------
    def _opp_move(self, state: dict, opp: int):
        if self.opp_model == "starter":
            from orbit_wars_rl.agents.scripted import StarterAgent
            return StarterAgent().act({**state, "player": opp}, {"episodeSteps": self.episode_steps})
        return self._greedy_move(state, opp)

    def _rollout_value(self, state: dict, player: int, first_action) -> float:
        opp = 1 - player
        s = state
        acts = [None, None]
        acts[player] = first_action
        acts[opp] = self._opp_move(s, opp)
        s = self._step(s, acts)
        for _ in range(self.H - 1):
            if self._winner_decided(s):
                break
            acts[player] = self._greedy_move(s, player)
            acts[opp] = self._greedy_move(s, opp)
            s = self._step(s, acts)
        if self.value_mode == "heuristic":
            return _heuristic_value(s, player)
        _, v, _ = self._logits_value(s, player)
        return v

    def _step(self, state: dict, acts):
        if self.backend == "py":
            from orbit_wars_rl import py_engine
            nxt = py_engine.step(state, acts)
        else:
            from orbit_wars_rl import _native as native
            nxt = native.step_from_state(state, acts)
        nxt["step"] = state.get("step", 0) + 1
        nxt["num_agents"] = 2
        return nxt

    @staticmethod
    def _winner_decided(state: dict) -> bool:
        alive = {p[1] for p in state["planets"] if p[1] != -1} | {f[1] for f in state["fleets"]}
        return len(alive) <= 1

    # ---- per-planet 1-ply (coordinate ascent) ------------------------------
    def _planet_launch_options(self, state: dict, player: int, pid: int, T: int = 3):
        """Options for one planet: noop + launch-half at each of the T nearest non-owned
        planets. Directly poses the over-extension question (hold vs attack) per planet."""
        p = next((pp for pp in state["planets"] if pp[0] == pid), None)
        if p is None or p[5] <= 0:
            return [None]
        cand = []
        for t in state["planets"]:
            if t[0] == pid or t[1] == player:
                continue
            cand.append((math.hypot(p[2] - t[2], p[3] - t[3]), t))
        cand.sort(key=lambda x: x[0])
        opts = [None]
        for _, t in cand[:T]:
            ang = math.atan2(t[3] - p[3], t[2] - p[2])
            ships = max(1, min(int(0.5 * p[5]), p[5]))
            opts.append([pid, ang, ships])
        return opts

    def _per_planet_search(self, state: dict, player: int, t0: float = None):
        logits, _, ctx = self._logits_value(state, player)
        base = self._decode(logits.argmax(-1).cpu().numpy(), ctx)
        move_by_pid = {int(m[0]): list(m) for m in base}
        rows = [r for r in range(self.act_proc.max_entities) if ctx["actionable"][r] > 0.5]
        rows.sort(key=lambda r: -int(ctx["planet_ships"][r]))  # decide biggest garrisons first
        for r in rows:
            if t0 is not None and time.perf_counter() - t0 > self.time_budget:
                break  # wall-clock guard: keep the greedy choice for the rest
            pid = int(ctx["planet_ids"][r])
            best_v, best = -1e18, move_by_pid.get(pid)
            for opt in self._planet_launch_options(state, player, pid):
                joint = [m for q, m in move_by_pid.items() if q != pid]
                if opt is not None:
                    joint.append(opt)
                v = self._rollout_value(state, player, joint)
                if v > best_v:
                    best_v, best = v, opt
            if best is None:
                move_by_pid.pop(pid, None)
            else:
                move_by_pid[pid] = best
        return [m for m in move_by_pid.values()]

    # ---- Agent API ----------------------------------------------------------
    def act(self, obs, config):
        player = int(obs.get("player", 0))
        state = {
            "planets": [list(p) for p in obs.get("planets", [])],
            "fleets": [list(f) for f in obs.get("fleets", [])],
            "comets": [dict(g) if hasattr(g, "items") else g for g in obs.get("comets", [])],
            "comet_planet_ids": list(obs.get("comet_planet_ids", [])),
            "initial_planets": [list(p) for p in obs.get("initial_planets", [])],
            "angular_velocity": float(obs.get("angular_velocity", 0.0)),
            "next_fleet_id": int(obs.get("next_fleet_id", 0)),
            "step": int(obs.get("step", 0)),
            "num_agents": 2,
            "comet_schedule": [],  # future spawns are hidden; plan assuming none
        }
        t0 = time.perf_counter()
        if self.mode == "per_planet":
            return self._per_planet_search(state, player, t0)
        cands = self._candidates(state, player)
        best_v, best = -1e18, cands[0]  # greedy is cands[0] -> always a safe fallback
        for a in cands:
            v = self._rollout_value(state, player, a)
            if v > best_v:
                best_v, best = v, a
            if time.perf_counter() - t0 > self.time_budget:
                break  # wall-clock guard: never risk a TIMEOUT
        return best


def main():
    from orbit_wars_rl.agents.scripted import RandomAgent, StarterAgent
    from orbit_wars_rl.agents.ppo_policy import PolicyAgent
    from orbit_wars_rl.env.game import make_kaggle_env, scores_by_player

    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--opponent", default="starter", choices=["starter", "random"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--value", default="head", choices=["head", "heuristic"])
    ap.add_argument("--opp-model", default="policy", choices=["policy", "starter"])
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--mode", default="joint", choices=["joint", "per_planet"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare-greedy", action="store_true", help="also run the raw greedy policy")
    args = ap.parse_args()

    def run(agent_factory, label):
        wins = [0, 0, 0]
        for ep in range(args.episodes):
            a0 = agent_factory()
            a1 = StarterAgent() if args.opponent == "starter" else RandomAgent(seed=7000 + ep)
            env = make_kaggle_env(configuration={"seed": 7000 + ep})
            env.run([a0.to_kaggle_agent(), a1.to_kaggle_agent()])
            sc = scores_by_player(env.steps[-1][0]["observation"], 2)
            wins[0 if sc[0] > sc[1] else 1 if sc[1] > sc[0] else 2] += 1
        n = args.episodes
        print(f"{label}: p0 {wins[0]}/{n} ({wins[0]/n:.0%})  p1 {wins[1]}  draw {wins[2]}")

    run(lambda: SearchAgent(args.checkpoint, args.device, args.K, args.H, args.value,
                            args.opp_model, temperature=args.temperature, mode=args.mode),
        f"SEARCH(mode={args.mode},H={args.H},val={args.value},opp={args.opp_model}) vs {args.opponent}")
    if args.compare_greedy:
        run(lambda: PolicyAgent.load(args.checkpoint, device=args.device),
            f"GREEDY vs {args.opponent}")


if __name__ == "__main__":
    main()
