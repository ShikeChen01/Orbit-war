"""Local validation of submission_greedy.py in the REAL kaggle simulation.

Runs the exact submitted `agent(obs, config)` vs scripted baselines and reports win
rate, so a broken/weak agent is caught before spending a daily submission.

    .venv\\Scripts\\python.exe scripts\\eval_greedy.py --episodes 20
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbit_wars_rl.agents import NoopAgent, RandomAgent, StarterAgent  # noqa: E402
from orbit_wars_rl.agents.base import FunctionAgent  # noqa: E402
from orbit_wars_rl.env.game import make_kaggle_env, scores_by_player  # noqa: E402

from submission_greedy import agent as greedy_agent  # noqa: E402


def play_once(a0, a1, seed):
    env = make_kaggle_env(configuration={"seed": seed} if seed is not None else None)
    a0.reset(); a1.reset()
    env.run([a0.to_kaggle_agent(), a1.to_kaggle_agent()])
    scores = scores_by_player(env.steps[-1][0]["observation"], 2)
    winner = max(range(2), key=lambda i: scores[i]) if max(scores) > 0 else -1
    return winner, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    greedy = FunctionAgent(greedy_agent, name="greedy")
    opponents = {"noop": NoopAgent(), "random": RandomAgent(seed=0), "starter": StarterAgent()}

    for name, opp in opponents.items():
        w = [0, 0, 0]  # greedy, opp, draw
        sample = []
        for ep in range(args.episodes):
            winner, scores = play_once(greedy, opp, seed=1000 + ep)
            w[2 if winner == -1 else winner] += 1
            if ep < 3:
                sample.append(scores)
        n = args.episodes
        print(f"greedy vs {name:8s} | win {w[0]/n:5.0%}  lose {w[1]/n:5.0%}  draw {w[2]/n:5.0%}"
              f"  | sample scores {sample}")


if __name__ == "__main__":
    main()
