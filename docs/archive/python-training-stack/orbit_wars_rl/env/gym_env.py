"""Single-agent Gymnasium training environment (local).

Wraps the kaggle Orbit Wars simulation via ``env.train([None, opponent, ...])`` so the
learner always controls player 0 while every other seat is driven by a pluggable
:class:`~orbit_wars_rl.agents.base.Agent` (random / starter / a frozen self-play policy).

The kaggle env only rewards at the end (+1 win / -1 loss), which is too sparse for PPO.
We add a dense shaping reward: the per-turn change in score margin
``my_score - best_opponent_score`` (total ships on planets + in fleets), plus a terminal
win/loss bonus.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import gymnasium as gym
import numpy as np

from orbit_wars_rl.env.game import make_kaggle_env, scores_by_player

if TYPE_CHECKING:  # avoid an env<->agents import cycle at runtime
    from orbit_wars_rl.agents.base import Agent
from orbit_wars_rl.processors.action import PerPlanetAction
from orbit_wars_rl.processors.base import ActionProcessor, ObservationProcessor
from orbit_wars_rl.processors.observation import EntityObservation


class OrbitWarsEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        obs_processor: Optional[ObservationProcessor] = None,
        act_processor: Optional[ActionProcessor] = None,
        opponents: "Optional[list[Agent]]" = None,
        num_players: int = 2,
        configuration: Optional[dict] = None,
        reward_scale: float = 50.0,
        terminal_bonus: float = 1.0,
        reward_clip: float = 5.0,
        seed: Optional[int] = None,
        debug: bool = False,
    ):
        super().__init__()
        assert num_players in (2, 4), "Orbit Wars supports 2 or 4 players."
        self.num_players = num_players
        self.configuration = dict(configuration or {})
        self.episode_steps = int(self.configuration.get("episodeSteps", 500))
        self.obs_processor = obs_processor or EntityObservation(episode_steps=self.episode_steps)
        self.act_processor = act_processor or PerPlanetAction()
        if opponents is None:
            from orbit_wars_rl.agents.scripted import RandomAgent  # lazy: breaks import cycle

            opponents = [RandomAgent(seed=seed) for _ in range(num_players - 1)]
        self.opponents = opponents
        assert len(self.opponents) == num_players - 1, "need one opponent per non-learner seat"
        self.reward_scale = reward_scale
        self.terminal_bonus = terminal_bonus
        self.reward_clip = reward_clip
        self.debug = debug

        self.observation_space = self.obs_processor.build_space(self.configuration)
        self.action_space = self.act_processor.build_space(self.configuration)

        self._kenv = None
        self._trainer = None
        self._context: dict = {}
        self._raw_obs: dict = {}
        self._prev_margin: float = 0.0
        self._rng = np.random.default_rng(seed)

    # -- gym API -------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        cfg = dict(self.configuration)
        # Fresh planet/comet layout each episode unless the caller pinned a seed.
        cfg.setdefault("seed", int(self._rng.integers(0, 2**31 - 1)))

        self._kenv = make_kaggle_env(configuration=cfg, debug=self.debug)
        for opp in self.opponents:
            opp.reset()
        opponent_agents = [opp.to_kaggle_agent() for opp in self.opponents]
        self._trainer = self._kenv.train([None, *opponent_agents])

        raw = self._trainer.reset()
        self._raw_obs = raw
        obs_arrays, self._context = self.obs_processor.process(raw, cfg)
        self._prev_margin = self._margin(raw)
        return obs_arrays, {"raw_obs": raw}

    def step(self, action):
        cfg = dict(self.configuration)
        moves = self.act_processor.decode(action, self._context)
        raw, kaggle_reward, done, info = self._trainer.step(moves)
        self._raw_obs = raw

        margin = self._margin(raw)
        dense = (margin - self._prev_margin) / self.reward_scale
        self._prev_margin = margin
        reward = float(np.clip(dense, -self.reward_clip, self.reward_clip))

        terminated = bool(done)
        if terminated and kaggle_reward is not None:
            # kaggle terminal reward: +1 win / -1 loss for player 0.
            reward += self.terminal_bonus * float(np.sign(kaggle_reward))

        obs_arrays, self._context = self.obs_processor.process(raw, cfg)
        info = dict(info or {})
        info.update({"raw_obs": raw, "score_margin": margin, "kaggle_reward": kaggle_reward})
        if terminated:
            info["winner"] = self._winner(raw)
        return obs_arrays, reward, terminated, False, info

    def render(self):
        if not self._raw_obs:
            return ""
        return self._kenv.render(mode="ansi") if self._kenv else ""

    # -- helpers -------------------------------------------------------------
    def _margin(self, raw: dict) -> float:
        scores = scores_by_player(raw, self.num_players)
        mine = scores[0]
        others = [s for i, s in enumerate(scores) if i != 0]
        return float(mine - (max(others) if others else 0))

    def _winner(self, raw: dict) -> int:
        scores = scores_by_player(raw, self.num_players)
        best = max(scores)
        return int(np.argmax(scores)) if best > 0 else -1

    @property
    def raw_obs(self) -> dict:
        return self._raw_obs
