"""Entity-based actor-critic policy and its Agent wrapper.

Architecture (matches the chosen design: per-planet entity encoder -> per-planet
action head, shared critic):

    entities (B,E,F) --shared MLP-->  tok (B,E,h)
    context = masked-mean(tok) ++ globals  --MLP-->  core (B,h)
    actor:  MLP([tok_e ++ core]) -> logits (B,E,A)   (masked to legal launches)
    critic: MLP(core)            -> value  (B,)

Each entity row is an independent categorical over ``actions_per_entity`` classes
(class 0 = no-op). Padding / non-owned rows are masked to the no-op class, so they
contribute zero log-prob and zero entropy. The turn's log-prob is the sum over rows.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from orbit_wars_rl.agents.base import Agent
from orbit_wars_rl.processors.action import DEFAULT_FRACTIONS, PerPlanetAction
from orbit_wars_rl.processors.observation import EntityObservation

_NEG_INF = -1e9


def _mlp(sizes: list[int], act=nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class EntityPolicy(nn.Module):
    def __init__(
        self,
        n_entity_features: int,
        n_global_features: int,
        actions_per_entity: int,
        hidden: int = 128,
        target_mode: bool = False,
        num_fracs: int = 4,
    ):
        super().__init__()
        self.actions_per_entity = actions_per_entity
        self.target_mode = target_mode
        self.num_fracs = num_fracs
        self.hidden = hidden
        self.entity_encoder = _mlp([n_entity_features, hidden, hidden])
        self.context_encoder = _mlp([hidden + n_global_features, hidden, hidden])
        self.critic = _mlp([hidden, hidden, 1])
        # Residual GLU block per encoder: out = x + W_o( (W_v x) * sigmoid(W_g x) ).
        self.ent_glu_gate = nn.Linear(hidden, hidden)
        self.ent_glu_val = nn.Linear(hidden, hidden)
        self.ent_glu_out = nn.Linear(hidden, hidden)
        self.ctx_glu_gate = nn.Linear(hidden, hidden)
        self.ctx_glu_val = nn.Linear(hidden, hidden)
        self.ctx_glu_out = nn.Linear(hidden, hidden)
        if target_mode:
            # Pointer actor: logit(r,t,f) = <q_f(tok_r,core), k(tok_t)>; the score depends
            # on BOTH launcher and target encodings, so target selection (e.g. "nearest
            # non-owned static") is representable -- an MLP over (tok_r,core) alone cannot
            # pick a target because it has no per-target information.
            self.actor_q = nn.Linear(hidden + hidden, num_fracs * hidden)
            self.actor_k = nn.Linear(hidden, hidden)
            self.actor_noop = nn.Linear(hidden + hidden, 1)
        else:
            self.actor = _mlp([hidden + hidden, hidden, actions_per_entity])

    def forward(self, obs: dict[str, torch.Tensor]):
        """Returns (logits (B,E,A), value (B,)). Logits are masked to legal actions."""
        ent = obs["entities"]                      # (B,E,F)
        ent_mask = obs["entity_mask"]              # (B,E)
        act_mask = obs["action_mask"]              # (B,E)
        glob = obs["globals"]                      # (B,G)

        tok = self.entity_encoder(ent)             # (B,E,h)
        tok = tok + self.ent_glu_out(self.ent_glu_val(tok) * torch.sigmoid(self.ent_glu_gate(tok)))

        # Masked mean over real entities -> board summary.
        m = ent_mask.unsqueeze(-1)                 # (B,E,1)
        summed = (tok * m).sum(dim=1)
        count = m.sum(dim=1).clamp_min(1.0)
        pooled = summed / count                    # (B,h)
        core = self.context_encoder(torch.cat([pooled, glob], dim=-1))  # (B,h)
        core = core + self.ctx_glu_out(self.ctx_glu_val(core) * torch.sigmoid(self.ctx_glu_gate(core)))

        core_b = core.unsqueeze(1).expand(-1, tok.size(1), -1)          # (B,E,h)
        if self.target_mode:
            B, E, h = tok.shape
            q = self.actor_q(torch.cat([tok, core_b], dim=-1)).reshape(B, E, self.num_fracs, h)
            k = self.actor_k(tok)                                      # (B,E,h)
            score = torch.einsum("brfh,bth->brtf", q, k) / (h ** 0.5)  # (B,r,t,f)
            noop = self.actor_noop(torch.cat([tok, core_b], dim=-1))   # (B,E,1)
            logits = torch.cat([noop, score.reshape(B, E, E * self.num_fracs)], dim=-1)
        else:
            logits = self.actor(torch.cat([tok, core_b], dim=-1))      # (B,E,A)
        logits = self._mask_logits(logits, act_mask, ent_mask)
        value = self.critic(core).squeeze(-1)                          # (B,)
        return logits, value

    def _mask_logits(self, logits: torch.Tensor, act_mask: torch.Tensor,
                     ent_mask: torch.Tensor) -> torch.Tensor:
        if self.target_mode:
            # Classes 1.. are (target_row t, frac f). Allow only if launcher r is
            # actionable, target t is real, and t != r. Class 0 (noop) always allowed.
            B, E, A = logits.shape
            nf = (A - 1) // E
            noop = logits[..., :1]
            launch = logits[..., 1:].reshape(B, E, E, nf)              # (B,r,t,f)
            launcher_ok = (act_mask > 0.5).view(B, E, 1, 1)
            tgt_real = (ent_mask > 0.5).view(B, 1, E, 1)
            eye = torch.eye(E, device=logits.device, dtype=torch.bool).view(1, E, E, 1)
            allowed = launcher_ok & tgt_real & (~eye)                 # (B,E,E,1)
            launch = torch.where(allowed, launch, torch.full_like(launch, _NEG_INF))
            return torch.cat([noop, launch.reshape(B, E, A - 1)], dim=-1)
        # Angle-bin mode: allow class 0 everywhere, classes >0 only where act_mask == 1.
        allowed = torch.ones_like(logits, dtype=torch.bool)
        non_actionable = act_mask < 0.5                                # (B,E)
        allowed[..., 1:] = ~non_actionable.unsqueeze(-1)
        return torch.where(allowed, logits, torch.full_like(logits, _NEG_INF))

    # -- rollout / training helpers -----------------------------------------
    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()  # (B,E)
        logp = self._summed_logprob(dist, action, obs["entity_mask"])
        return action, logp, value

    def evaluate_actions(self, obs: dict[str, torch.Tensor], action: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        logp = self._summed_logprob(dist, action, obs["entity_mask"])
        ent_mask = obs["entity_mask"]
        entropy = (dist.entropy() * ent_mask).sum(dim=1)
        return logp, entropy, value

    @staticmethod
    def _summed_logprob(dist: Categorical, action: torch.Tensor, ent_mask: torch.Tensor):
        return (dist.log_prob(action) * ent_mask).sum(dim=1)


def obs_to_tensors(obs: dict, device: torch.device, batched: bool = False) -> dict:
    """Numpy obs dict -> float tensors on ``device`` (adds batch dim if needed)."""
    out = {}
    for k, v in obs.items():
        t = torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
        out[k] = t if batched else t.unsqueeze(0)
    return out


class PolicyAgent(Agent):
    """Wrap an :class:`EntityPolicy` as a playable :class:`Agent`.

    Used for evaluation, as a frozen self-play opponent, and as the basis for the
    competition submission. Defaults to greedy (deterministic) play.
    """

    name = "ppo"

    def __init__(
        self,
        policy: EntityPolicy,
        obs_processor: Optional[EntityObservation] = None,
        act_processor: Optional[PerPlanetAction] = None,
        device: str = "cpu",
        deterministic: bool = True,
    ):
        self.policy = policy.to(device).eval()
        self.device = torch.device(device)
        # Build processors that MATCH the policy head, so decode never mismatches the
        # trained action layout (target-based vs angle-bin; #bins / #targets).
        target_mode = bool(getattr(policy, "target_mode", False))
        nf = len(DEFAULT_FRACTIONS)
        if act_processor is None:
            if target_mode:
                max_entities = (policy.actions_per_entity - 1) // nf
                act_processor = PerPlanetAction(max_entities=max_entities, target_mode=True)
            else:
                angle_bins = (policy.actions_per_entity - 1) // nf
                act_processor = PerPlanetAction(angle_bins=angle_bins)
        self.obs_processor = obs_processor or EntityObservation(max_entities=act_processor.max_entities)
        self.act_processor = act_processor
        self.deterministic = deterministic

    def act(self, obs: dict, config: dict) -> list[list]:
        obs_arrays, context = self.obs_processor.process(obs, config)
        tensors = obs_to_tensors(obs_arrays, self.device, batched=False)
        action, _, _ = self.policy.act(tensors, deterministic=self.deterministic)
        return self.act_processor.decode(action.squeeze(0).cpu().numpy(), context)

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "PolicyAgent":
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["policy_config"]
        policy = EntityPolicy(**cfg)
        policy.load_state_dict(ckpt["policy_state"])
        return cls(policy, device=device, **kwargs)
