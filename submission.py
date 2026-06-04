"""Kaggle Orbit Wars submission entrypoint (inference-search agent).

Uses the validated search agent on the **Kaggle-compatible path** -- no native `.pyd`:
  * forward model: `orbit_wars_rl.py_engine` (pure Python, bit-exact to the official engine),
  * observation encode: the Python `EntityObservation`,
  * a per-turn wall-clock guard that falls back to the greedy policy move (never TIMEOUT).

Config is the strongest in local eval (BC greedy 0% -> per-planet search **70%** vs starter):
mode=per_planet, H=8, heuristic value, opponent modeled as starter. (Joint mode K=8/H=10 gives
42%; per-planet -- decide each planet's hold-vs-attack by 1-ply lookahead -- is markedly better.)

To deploy on Kaggle this file plus the `orbit_wars_rl/` package (and the checkpoint) must be
present on the eval box -- package the repo as a Kaggle Utility Script / dataset and import
it, or inline the needed modules. See `docs/SUBMISSION.md`. Validate locally first:
    python scripts/play_episode.py  (or a couple games vs starter) -- it's faithful.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbit_wars_rl.agents.base import _as_dict  # noqa: E402
from scripts.search_agent import SearchAgent  # noqa: E402

_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "native", "bc_start.pt")

_AGENT = SearchAgent(
    _CKPT, device="cpu", K=8, H=8, value="heuristic", opponent_model="starter",
    mode="per_planet",   # decide each planet hold-vs-attack by 1-ply lookahead (70% vs starter)
    temperature=1.5,
    backend="py",        # py_engine forward model (no native .pyd on the eval box)
    fast_encode=False,   # Python EntityObservation encode (no native)
    time_budget=0.85,    # leave headroom under the 1s/turn budget; falls back to greedy
)


def agent(obs, config):
    return _AGENT.act(_as_dict(obs), _as_dict(config))
