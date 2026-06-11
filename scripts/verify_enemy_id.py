"""Can the policy tell WHICH enemy owns a planet (per-seat enemy ID) in v8 / v9?

Empirical check of the ownership features (and threat signs / global aggregates) that
env_encode emits when planets are owned by seats 0..3:

  * v8 (LEGACY-COMPATIBLE lineage, scalar ownership, F_DIM 101): every non-ego owner collapses
    to the single code 2.0 BY DESIGN (everyone-vs-me; v7 weights drop in unchanged) -> the
    agent CANNOT tell enemies apart, and that is intended.
  * v9 (one-hot lineage, F_DIM 104): ownership is a 4-channel EGO-RELATIVE seat ONE-HOT at
    cols 7..10 [self, enemy+1, enemy+2, enemy+3], neutral = all-zero -> the agent can tell
    WHICH enemy owns a body; in 2p only the first two channels ever fire.

(v7, now in notebooks/legacy/, used the scalar `na = 2` formula that would alias owner ego+2
onto the EGO code if ever fed a 4p state -- the reason v8 collapses instead.)

    .venv/Scripts/python.exe scripts/verify_enemy_id.py
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import sys
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import eval_kaggle_v7 as EV   # noqa: E402  (notebook-defs loader works for v7 AND v8)


def four_player_state(g, n_players):
    """A v8-style env with one planet per seat 0..3 + one inbound fleet per enemy seat."""
    env = g["GpuEnv"](g["PLANET_CAP"], g["FLEET_CAP"], 500, g["SHIP_SPEED"], torch.device("cpu"))
    w = g["generate_world"](0)
    try:
        env.reset([w], n_players=n_players)   # v8
    except TypeError:
        env.reset([w])                        # v7 (2p only)
    # force a clean 4-seat ownership pattern on the first 5 alive planets
    seats = [0, 1, 2, 3]
    for i, p in enumerate(seats):
        env.p_owner[0, i] = float(p)
        env.p_ships[0, i] = 50.0
    env.p_owner[0, 4] = -1.0                  # neutral
    # one fleet per enemy seat, aimed at the ego planet (slot 0)
    for j, p in enumerate((1, 2, 3)):
        env.f_alive[0, j] = 1.0
        env.f_owner[0, j] = float(p)
        env.f_x[0, j] = env.p_x[0, 0] + 10.0 + j
        env.f_y[0, j] = env.p_y[0, 0]
        env.f_angle[0, j] = 3.14159265
        env.f_ships[0, j] = 10.0 + j
        env.f_seq[0, j] = float(j + 1)
    return env


def report(tag, g, n_players):
    env = four_player_state(g, n_players)
    nb = g["N_BODY_FEATURES"]
    onehot = nb == 14                                    # v8: 4-channel seat one-hot at cols 7..10
    print("== %s (%s ownership) ==" % (tag, "one-hot" if onehot else "scalar"))
    for ego in (0, 2):
        ent, em, am, gl = g["env_encode"](env, ego)
        if onehot:
            blk = ent[0, :5, 7:11]
            fmt = lambda r: "".join(str(int(v)) for v in r.tolist())
            codes = {int(env.p_owner[0, i].item()): fmt(blk[i]) for i in range(4)}
            codes["neutral"] = fmt(blk[4])
        else:
            b7 = ent[0, :5, 7].tolist()
            codes = {int(env.p_owner[0, i].item()): round(b7[i], 1) for i in range(4)}
            codes["neutral"] = round(b7[4], 1)
        # threat slots of the ego planet: sign of the 3 inbound enemy fleets
        sgn = ent[0, 0, nb::3]
        signs = sorted(set(round(v, 1) for v in sgn.tolist() if v != 0.0))
        print("  ego=%d: ownership by owner %s | inbound-fleet signs %s | g3(enemy ships, pooled)=%.3f"
              % (ego, codes, signs, gl[0, 3].item()))
        if ego == 0:
            distinct = len(set(str(codes[p]) for p in (1, 2, 3)))
            alias = any(str(codes[p]) == str(codes[0]) for p in (1, 2, 3))
            print("  -> distinct enemy codes: %d/3 %s%s"
                  % (distinct, "(CANNOT tell enemies apart)" if distinct == 1 else "",
                     "  !! owner %s ALIASES to the EGO code !!" %
                     [p for p in (1, 2, 3) if str(codes[p]) == str(codes[0])] if alias else ""))
    print()


def main():
    os.chdir(REPO)
    for tag, nb, npl in (("v8 (setup1_v8_a100)", "notebooks/setup1_v8_a100.ipynb", 4),
                         ("v9 (setup1_v9_a100)", "notebooks/setup1_v9_a100.ipynb", 4)):
        g = EV.load_nb_defs(nb)
        g["DEVICE"] = torch.device("cpu")
        report(tag, g, npl)


if __name__ == "__main__":
    main()
