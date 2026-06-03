"""Training configuration (single source of truth for hyperparameters)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class TrainConfig:
    # -- environment --
    num_players: int = 2
    max_entities: int = 64
    angle_bins: int = 16
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    episode_steps: int = 500
    reward_scale: float = 50.0
    terminal_bonus: float = 1.0
    reward_clip: float = 5.0

    # -- model --
    hidden: int = 128

    # -- PPO --
    total_steps: int = 1_000_000
    rollout_steps: int = 2048      # env steps collected per update
    update_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.997           # long horizon (up to 500 turns)
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4

    # -- self-play / opponents --
    # Pool of opponent kinds to sample from each episode: "random", "starter", "self".
    opponent_pool: tuple[str, ...] = ("random", "starter", "self")
    selfplay_start_step: int = 100_000   # only add frozen-self snapshots after this
    snapshot_every_steps: int = 100_000
    max_snapshots: int = 5

    # -- runtime --
    device: str = "cuda"
    seed: int = 0
    log_every_updates: int = 1
    eval_every_updates: int = 10
    eval_episodes: int = 20
    save_dir: str = "runs"
    run_name: str = "ppo_orbitwars"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fractions"] = list(self.fractions)
        d["opponent_pool"] = list(self.opponent_pool)
        return d
