"""Thin layer over the official kaggle_environments ``orbit_wars`` simulation.

The kaggle env *is* the agent environment (the full game simulation). This module
re-exports its constants / named tuples and adds small helpers (scoring, env
construction) so the rest of the codebase never imports kaggle internals directly.
"""
from __future__ import annotations

import orbit_wars_rl._bootstrap  # noqa: F401  (must run before kaggle_environments)

import logging

# kaggle_environments is very chatty at import (it eagerly loads every bundled env).
logging.getLogger("kaggle_environments").setLevel(logging.ERROR)

from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.orbit_wars.orbit_wars import (  # noqa: E402
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    Fleet,
    Planet,
)

ENV_NAME = "orbit_wars"

__all__ = [
    "ENV_NAME",
    "make_kaggle_env",
    "Planet",
    "Fleet",
    "BOARD_SIZE",
    "CENTER",
    "SUN_RADIUS",
    "ROTATION_RADIUS_LIMIT",
    "COMET_PRODUCTION",
    "scores_by_player",
    "player_score",
]


def make_kaggle_env(configuration: dict | None = None, debug: bool = False):
    """Create a raw kaggle_environments Orbit Wars environment."""
    return make(ENV_NAME, configuration=configuration or {}, debug=debug)


def scores_by_player(obs: dict, num_players: int = 4) -> list[int]:
    """Total ships (on owned planets + in owned fleets) per player.

    This mirrors the env's terminal scoring (``orbit_wars.py``): the winner is the
    player with the highest score. Neutral (owner ``-1``) ships are ignored.
    """
    scores = [0] * num_players
    for p in obs.get("planets", []):
        owner = p[1]
        if owner != -1 and 0 <= owner < num_players:
            scores[owner] += p[5]
    for f in obs.get("fleets", []):
        owner = f[1]
        if 0 <= owner < num_players:
            scores[owner] += f[6]
    return scores


def player_score(obs: dict, player: int, num_players: int = 4) -> int:
    return scores_by_player(obs, num_players)[player]
