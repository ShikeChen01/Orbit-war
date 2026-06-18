"""Orbit Wars v12 -- encapsulated port of ``notebooks/setup1_v12_a100.ipynb``.

A self-contained PPO training stack: a GPU-batched simulator (bit-exact to the official engine
incl. comets), a gated-allocation transformer/ResNet policy, dual-Elo PFSP self-play league with
measured eviction, BC warm-start, and the A100 accel path (torch.compile learner + grad
checkpointing). All run-tunable hyperparameters live on a single frozen :class:`Config`; fixed
game/feature constants live in :mod:`orbit_wars_v12.constants`.

Quick start::

    from orbit_wars_v12 import Config, Trainer
    cfg = Config.create(smoke=True)          # or Config.create(smoke=False) for the full run
    net, hist, league = Trainer(cfg).train()

This is a behaviour-faithful refactor of the v12 notebook (the notebook remains the source of
truth); it is a separate lineage from the older :mod:`orbit_wars_rl` native/gym package.
"""
from . import constants
from .bc import bc_pretrain
from .checkpoint import (build_from_cfg, discover_invited, load_snapshot, load_train_state,
                         save_ckpt, save_train_state)
from .config import Config
from .distributions import GatedAllocDist, GatedCatDist
from .env import GpuEnv, env_encode, env_step, settle, settle_n
from .league import League, eval_gauntlet, eval_gauntlet4
from .plotting import plot_training_health
from .policy import PolicyNet, act, build_policy, evaluate
from .ppo import ppo_update
from .rollout import collect_ppo
from .train import Trainer, anneal_ent_coef, save_league_agents, train
from .worldgen import build_world_pool, generate_world, make_world_pool

__all__ = [
    "constants", "Config",
    "GpuEnv", "env_encode", "env_step", "settle", "settle_n",
    "GatedAllocDist", "GatedCatDist",
    "PolicyNet", "build_policy", "act", "evaluate",
    "collect_ppo", "ppo_update",
    "League", "eval_gauntlet", "eval_gauntlet4",
    "Trainer", "train", "anneal_ent_coef", "save_league_agents",
    "bc_pretrain",
    "save_ckpt", "save_train_state", "load_train_state", "load_snapshot",
    "build_from_cfg", "discover_invited",
    "make_world_pool", "build_world_pool", "generate_world",
    "plot_training_health",
]
