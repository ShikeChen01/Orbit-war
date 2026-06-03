"""Arena: play two agents against each other in the real kaggle simulation.

This uses the kaggle env directly (not the training wrapper), so it scores exactly
like the competition. Swap in any :class:`Agent` -- scripted or a trained policy.

    python scripts/play_episode.py --p0 starter --p1 random
    python scripts/play_episode.py --p0 runs/ppo_orbitwars/final.pt --p1 starter --episodes 20
"""
from __future__ import annotations

import argparse

from orbit_wars_rl.agents import RandomAgent, StarterAgent, NoopAgent
from orbit_wars_rl.env.game import make_kaggle_env, scores_by_player


def build_agent(spec: str):
    spec = spec.strip()
    if spec == "random":
        return RandomAgent()
    if spec == "starter":
        return StarterAgent()
    if spec == "noop":
        return NoopAgent()
    if spec.endswith(".pt"):
        from orbit_wars_rl.agents.ppo_policy import PolicyAgent  # lazy: needs torch

        return PolicyAgent.load(spec, device="cpu")
    raise SystemExit(f"unknown agent spec: {spec!r} (use random|starter|noop|<checkpoint>.pt)")


def play_once(a0, a1, num_players: int, seed: int | None) -> tuple[int, list[int]]:
    env = make_kaggle_env(configuration={"seed": seed} if seed is not None else None)
    agents = [a0.to_kaggle_agent(), a1.to_kaggle_agent()]
    if num_players == 4:
        agents += [a0.to_kaggle_agent(), a1.to_kaggle_agent()]
    a0.reset(); a1.reset()
    env.run(agents)
    final_obs = env.steps[-1][0]["observation"]
    scores = scores_by_player(final_obs, num_players)
    winner = max(range(num_players), key=lambda i: scores[i]) if max(scores) > 0 else -1
    return winner, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0", default="starter")
    ap.add_argument("--p1", default="random")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--players", type=int, default=2, choices=(2, 4))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    a0, a1 = build_agent(args.p0), build_agent(args.p1)
    wins = [0, 0, 0]  # p0, p1, draw
    for ep in range(args.episodes):
        seed = None if args.seed is None else args.seed + ep
        winner, scores = play_once(a0, a1, args.players, seed)
        bucket = 2 if winner == -1 else (0 if winner % 2 == 0 else 1)
        wins[bucket] += 1
        if args.episodes <= 5:
            print(f"ep {ep}: winner=p{winner} scores={scores}")
    n = args.episodes
    print(f"\n{args.p0} vs {args.p1} over {n} eps:")
    print(f"  p0 wins: {wins[0]} ({wins[0]/n:.0%})  p1 wins: {wins[1]} ({wins[1]/n:.0%})  draws: {wins[2]}")


if __name__ == "__main__":
    main()
