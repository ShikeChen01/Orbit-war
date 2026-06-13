"""4-player FREE-FOR-ALL in the official Kaggle orbit_wars engine (agents spec supports [2,4]).

Seats two v7 neural checkpoints (default it1041 + it101) plus two scripted/baseline fillers and
runs N games, ROTATING the seat assignment each game so no agent is pinned to a lucky seat. The
neural + scripted agents reuse the bit-exact obs->GpuEnv adapters from eval_kaggle_v7; in a 4-player
obs every non-self owner collapses to "enemy" in the 2-player GpuEnv view (the policy was trained
2-player, so it plays everyone-vs-me).

Reports each agent's mean placement (1=best..4=worst), 1st-place rate, and mean final reward.

    python scripts/sim_4player_v7.py --games 24
    python scripts/sim_4player_v7.py --seats it1041,it101,starter,greedy --games 24
"""
from __future__ import annotations
import os, sys, math, argparse, itertools
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # official engine is CPU, B=1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_kaggle_v7 import (load_nb_defs, build_actor_v7, make_learner_agent,
                            make_scripted_agent, OPP_CODE, NB_DEFAULT, REPO)

CKPT_DIR = os.path.join("notebooks", "weight", "Transformers")


def resolve_ckpt(p):
    for cand in (p, os.path.join(CKPT_DIR, p), os.path.join(CKPT_DIR, p + ".pt")):
        if os.path.exists(cand):
            return cand
    return None


def build_seat_agents(g, seat_specs):
    """seat_specs: list of labels. A label is a scripted-anchor name (starter/greedy/medium/...)
    or a checkpoint (it1041 / path). Returns {label: agent_fn} (one agent per UNIQUE label)."""
    agents = {}
    for lab in set(seat_specs):
        if lab in OPP_CODE:
            agents[lab] = make_scripted_agent(g, OPP_CODE[lab])
        else:
            ck = resolve_ckpt(lab)
            if ck is None:
                raise FileNotFoundError("seat %r is neither a scripted anchor nor a checkpoint" % lab)
            net, _, _ = build_actor_v7(g, ck, )  # build_actor_v7(g, ckpt)
            agents[lab] = make_learner_agent(g, net)
    return agents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seats", default="it1041,it101,starter,greedy",
                    help="4 comma-separated seat labels (checkpoint or anchor name)")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--nb", default=NB_DEFAULT)
    args = ap.parse_args()
    os.chdir(REPO)

    labels = [s.strip() for s in args.seats.split(",") if s.strip()]
    assert len(labels) == 4, "4-player FFA needs exactly 4 seat labels, got %d" % len(labels)

    g = load_nb_defs(args.nb)
    agents = build_seat_agents(g, labels)

    from kaggle_environments import make
    # rotate the 4 labels through the 4 physical seats so each label sees each seat equally
    rotations = list(itertools.islice(itertools.cycle(
        [labels[i:] + labels[:i] for i in range(4)]), args.games))

    stats = {lab: {"place_sum": 0, "first": 0, "reward_sum": 0.0, "n": 0} for lab in labels}
    print("4-PLAYER FFA | seats=%s | %d games | official engine (comets on)\n"
          % (labels, args.games), flush=True)

    for gi, roster_labels in enumerate(rotations):
        env = make("orbit_wars", configuration={"episodeSteps": args.steps}, debug=False)
        roster = [agents[lab] for lab in roster_labels]
        out = env.run(roster)
        rewards = [out[-1][i]["reward"] for i in range(4)]
        rewards = [(-1e18 if r is None else r) for r in rewards]
        # placement: 1 = highest reward (ties share the better rank average)
        order = sorted(range(4), key=lambda i: rewards[i], reverse=True)
        place = [0] * 4
        rank = 0
        while rank < 4:
            same = [order[rank]]
            j = rank + 1
            while j < 4 and rewards[order[j]] == rewards[order[rank]]:
                same.append(order[j]); j += 1
            avg_place = sum(range(rank + 1, j + 1)) / len(same)
            for s in same:
                place[s] = avg_place
            rank = j
        for i, lab in enumerate(roster_labels):
            st = stats[lab]
            st["place_sum"] += place[i]; st["reward_sum"] += rewards[i]; st["n"] += 1
            if place[i] == 1.0:
                st["first"] += 1
        if (gi + 1) % 4 == 0:
            print("  ... %d/%d games" % (gi + 1, args.games), flush=True)

    print("\n%-10s | games | mean-place | 1st-rate | mean-reward" % "agent")
    print("-" * 60)
    rows = sorted(labels, key=lambda l: stats[l]["place_sum"] / max(1, stats[l]["n"]))
    for lab in rows:
        st = stats[lab]; n = max(1, st["n"])
        print("%-10s | %5d | %10.3f | %7.1f%% | %.2f"
              % (lab, st["n"], st["place_sum"] / n, 100.0 * st["first"] / n, st["reward_sum"] / n))


if __name__ == "__main__":
    main()
