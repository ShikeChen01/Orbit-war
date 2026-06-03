"""PPO trainer with a self-play opponent pool (single local environment).

Deliberately framework-light: just torch. Collects fixed-length rollouts against an
opponent sampled per-episode (random / starter / a frozen snapshot of the policy),
computes GAE, and runs clipped PPO updates. Designed to be readable and to scale up
later (vectorized envs, Vertex AI) without changing the abstraction layer.
"""
from __future__ import annotations

import copy
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from orbit_wars_rl.agents.ppo_policy import EntityPolicy, PolicyAgent, obs_to_tensors
from orbit_wars_rl.agents.scripted import RandomAgent, StarterAgent
from orbit_wars_rl.env.gym_env import OrbitWarsEnv
from orbit_wars_rl.processors.action import PerPlanetAction
from orbit_wars_rl.processors.observation import (
    N_ENTITY_FEATURES,
    N_GLOBAL_FEATURES,
    EntityObservation,
)
from orbit_wars_rl.train.config import TrainConfig


class PPOTrainer:
    def __init__(self, config: TrainConfig):
        self.cfg = config
        self.device = torch.device(
            config.device if (config.device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)

        self.obs_proc = EntityObservation(
            max_entities=config.max_entities, episode_steps=config.episode_steps
        )
        self.act_proc = PerPlanetAction(
            max_entities=config.max_entities,
            angle_bins=config.angle_bins,
            fractions=config.fractions,
        )
        self.policy_config = dict(
            n_entity_features=N_ENTITY_FEATURES,
            n_global_features=N_GLOBAL_FEATURES,
            actions_per_entity=self.act_proc.actions_per_entity,
            hidden=config.hidden,
        )
        self.policy = EntityPolicy(**self.policy_config).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate, eps=1e-5)

        self.snapshots: deque[dict] = deque(maxlen=config.max_snapshots)
        self.env = OrbitWarsEnv(
            obs_processor=self.obs_proc,
            act_processor=self.act_proc,
            opponents=self._sample_opponents(step=0),
            num_players=config.num_players,
            configuration={"episodeSteps": config.episode_steps},
            reward_scale=config.reward_scale,
            terminal_bonus=config.terminal_bonus,
            reward_clip=config.reward_clip,
            seed=config.seed,
        )

        self.global_step = 0
        self.updates = 0
        self._cur_obs = None
        self._ep_ret = 0.0
        self._last_snap_bucket = -1
        self.ep_returns: deque[float] = deque(maxlen=100)
        self.ep_wins: deque[float] = deque(maxlen=100)
        self.run_dir = os.path.join(config.save_dir, config.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

    # -- opponents -----------------------------------------------------------
    def _sample_opponents(self, step: int) -> list:
        opponents = []
        for _ in range(self.cfg.num_players - 1):
            opponents.append(self._sample_one_opponent(step))
        return opponents

    def _sample_one_opponent(self, step: int):
        pool = list(self.cfg.opponent_pool)
        if step < self.cfg.selfplay_start_step or not self.snapshots:
            pool = [k for k in pool if k != "self"] or ["starter"]
        kind = pool[self.rng.integers(len(pool))]
        if kind == "random":
            return RandomAgent(seed=int(self.rng.integers(0, 2**31 - 1)))
        if kind == "starter":
            return StarterAgent()
        # frozen self snapshot, played greedily on CPU
        state = self.snapshots[self.rng.integers(len(self.snapshots))]
        snap = EntityPolicy(**self.policy_config)
        snap.load_state_dict(state)
        return PolicyAgent(snap, self.obs_proc, self.act_proc, device="cpu", deterministic=False)

    def _maybe_snapshot(self):
        if (
            self.global_step >= self.cfg.selfplay_start_step
            and self.global_step // self.cfg.snapshot_every_steps
            > getattr(self, "_last_snap_bucket", -1)
        ):
            self._last_snap_bucket = self.global_step // self.cfg.snapshot_every_steps
            self.snapshots.append(copy.deepcopy({k: v.cpu() for k, v in self.policy.state_dict().items()}))

    # -- rollout -------------------------------------------------------------
    def collect_rollout(self):
        cfg = self.cfg
        T = cfg.rollout_steps
        obs_buf = {k: np.zeros((T, *space.shape), np.float32) for k, space in self.env.observation_space.spaces.items()}
        actions = np.zeros((T, cfg.max_entities), np.int64)
        logprobs = np.zeros(T, np.float32)
        values = np.zeros(T, np.float32)
        rewards = np.zeros(T, np.float32)
        dones = np.zeros(T, np.float32)

        if self._cur_obs is None:
            self._cur_obs, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
        obs = self._cur_obs
        ep_ret = self._ep_ret

        for t in range(T):
            for k in obs_buf:
                obs_buf[k][t] = obs[k]
            tens = obs_to_tensors(obs, self.device, batched=False)
            action, logp, value = self.policy.act(tens)
            a = action.squeeze(0).cpu().numpy()
            obs, reward, term, trunc, info = self.env.step(a)
            actions[t] = a
            logprobs[t] = logp.item()
            values[t] = value.item()
            rewards[t] = reward
            dones[t] = float(term or trunc)
            ep_ret += reward
            self.global_step += 1

            if term or trunc:
                self.ep_returns.append(ep_ret)
                self.ep_wins.append(1.0 if info.get("winner") == 0 else 0.0)
                ep_ret = 0.0
                self._maybe_snapshot()
                self.env.opponents = self._sample_opponents(self.global_step)
                obs, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))

        # bootstrap value for the last (possibly non-terminal) state
        with torch.no_grad():
            last_value = self.policy.forward(obs_to_tensors(obs, self.device))[1].item()
        self._cur_obs = obs
        self._ep_ret = ep_ret

        advantages, returns = self._gae(rewards, values, dones, last_value)
        return obs_buf, actions, logprobs, values, advantages, returns

    def _gae(self, rewards, values, dones, last_value):
        cfg = self.cfg
        T = len(rewards)
        adv = np.zeros(T, np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    # -- update --------------------------------------------------------------
    def update(self, obs_buf, actions, old_logprobs, old_values, advantages, returns):
        cfg = self.cfg
        T = cfg.rollout_steps
        obs_t = {k: torch.as_tensor(v, device=self.device) for k, v in obs_buf.items()}
        act_t = torch.as_tensor(actions, device=self.device)
        old_logp_t = torch.as_tensor(old_logprobs, device=self.device)
        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        idx = np.arange(T)
        stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "n": 0}
        for _ in range(cfg.update_epochs):
            self.rng.shuffle(idx)
            for start in range(0, T, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                mb_obs = {k: v[mb] for k, v in obs_t.items()}
                logp, entropy, value = self.policy.evaluate_actions(mb_obs, act_t[mb])
                ratio = torch.exp(logp - old_logp_t[mb])
                a = adv_t[mb]
                pg1 = -a * ratio
                pg2 = -a * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((value - ret_t[mb]) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    stats["pg_loss"] += pg_loss.item()
                    stats["v_loss"] += v_loss.item()
                    stats["entropy"] += ent.item()
                    stats["approx_kl"] += ((old_logp_t[mb] - logp).mean()).item()
                    stats["n"] += 1
        for k in ("pg_loss", "v_loss", "entropy", "approx_kl"):
            stats[k] /= max(1, stats["n"])
        return stats

    # -- main loop -----------------------------------------------------------
    def train(self):
        cfg = self.cfg
        start = time.time()
        while self.global_step < cfg.total_steps:
            rollout = self.collect_rollout()
            stats = self.update(*rollout)
            self.updates += 1

            if self.updates % cfg.log_every_updates == 0:
                sps = int(self.global_step / max(1e-9, time.time() - start))
                avg_ret = np.mean(self.ep_returns) if self.ep_returns else float("nan")
                win = np.mean(self.ep_wins) if self.ep_wins else float("nan")
                print(
                    f"upd {self.updates:4d} | step {self.global_step:>8d} | sps {sps:5d} "
                    f"| ep_ret {avg_ret:7.3f} | winrate {win:5.2f} "
                    f"| pg {stats['pg_loss']:+.3f} v {stats['v_loss']:.3f} "
                    f"ent {stats['entropy']:.3f} kl {stats['approx_kl']:+.4f} "
                    f"| snaps {len(self.snapshots)}",
                    flush=True,
                )
            if self.updates % cfg.eval_every_updates == 0:
                self.save("latest.pt")
        self.save("final.pt")

    def save(self, name: str):
        path = os.path.join(self.run_dir, name)
        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "policy_config": self.policy_config,
                "train_config": self.cfg.to_dict(),
                "global_step": self.global_step,
            },
            path,
        )
        return path
