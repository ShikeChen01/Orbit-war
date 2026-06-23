"""Gated per-source action distributions (notebook cell 9).

The actor emits, per owned source planet, a WHERE distribution over destinations plus a launch
GATE. On fire, the planet dispatches its FULL garrison routed by the WHERE row; on no-fire it
holds. Two variants share the same ``(B, E, E+1)`` action bundle (``[...,:E]`` = allocation rows,
``[..., E]`` = fire in ``{0,1}``):

* :class:`GatedAllocDist` -- Dirichlet rows (v6 A/B).
* :class:`GatedCatDist`   -- categorical destinations (v7 default; bounded log-prob, BC-stable).

Both take the :class:`~orbit_wars_v12.config.Config` so the clamps / gate trims / entropy weights
that were notebook globals are read off ``cfg``.
"""
import torch
import torch.nn.functional as F


def _rb_gate_where(cfg, fire, where_lp, p):
    """[A1] The WHERE term of the per-planet log-prob. Default = sampled ``fire * where_lp``.
    With ``cfg.GRPO_RB_GATE`` it is Rao-Blackwellized over the launch Bernoulli: a straight-through
    that keeps the FORWARD value at ``fire * where_lp`` (so the importance ratio stays an honest
    on-policy ratio) while routing the GRADIENT through ``p.detach() * where_lp`` -- the conditional
    expectation of the WHERE score over the gate, removing the gate's 0/1 sampling variance exactly
    (E[fire]=p) without injecting any spurious gate-direction gradient."""
    if not cfg.GRPO_RB_GATE:
        return fire * where_lp
    pc = p.detach()
    return (fire * where_lp).detach() + pc * where_lp - (pc * where_lp).detach()


class GatedAllocDist:
    """Gated per-source Dirichlet ALLOCATION actor (v6).
      WHERE  dest_logits (B,E,E) -> alpha = softplus(logits)*kappa + eps -> Dirichlet rows over dests.
      GATE   gate_logits (B,E)   -> p = clamp((sigmoid-LO)/(HI-LO),0,1), trimmed to [eps,1-eps]; fire ~ Bernoulli(p).
    On fire the planet dispatches its FULL garrison routed by its WHERE row; on no-fire it holds.
    Action (B,E,E+1): [...,:E]=allocation rows, [...,E]=fire in {0,1}. Log-prob/entropy SUM over owned sources;
    the WHERE term only counts when firing.
    v6: greedy()/DEPLOY uses a TEMPERATURE-SHARPENED softmax over the LIVE+REACHABLE off-diagonal dest logits
    (concentration-aware) so the deployed policy lands a CONCENTRATED strike -- the Dirichlet MEAN spreads the
    full garrison thin and the decode floor then dropped every launch (the residual greedy-passive bug)."""
    def __init__(self, cfg, dest_logits, gate_logits, owned, alive=None, reach=None, ships=None,
                 kappa=None, deploy=False):
        self.cfg = cfg
        kappa = cfg.ALLOC_KAPPA if kappa is None else kappa
        self.owned = owned if owned.dim() == 2 else owned.squeeze(-1)        # (B,E) legal sources
        logits = dest_logits.clamp(-cfg.LOGIT_CLAMP, cfg.LOGIT_CLAMP)
        self.alpha = F.softplus(logits) * kappa + 1e-3                       # (B,E,E)
        self.dist = torch.distributions.Dirichlet(self.alpha)
        g = torch.sigmoid(gate_logits if gate_logits.dim() == 2 else gate_logits.squeeze(-1))   # (B,E)
        if deploy:                                                          # DEPLOY: hard deadzone (fire/hold trim)
            p = ((g - cfg.GATE_TRIM_LO) / (cfg.GATE_TRIM_HI - cfg.GATE_TRIM_LO)).clamp(0.0, 1.0)
        else:                                                              # TRAIN: SMOOTH fire-prob so the gate head
            p = g                                                          #   keeps a full-support gradient (the
                                                                           #   deadzone clamp zeroed it -> grad vanish)
        self.p = p.clamp(cfg.GATE_EPS, 1.0 - cfg.GATE_EPS)                  # (B,E) trainable fire-prob
        E = logits.shape[-1]
        eye = torch.eye(E, dtype=torch.bool, device=logits.device).unsqueeze(0)
        dead = eye.expand_as(logits).clone()                                # drop self (decode drops it)
        if alive is not None:
            dead = dead | (alive < 0.5).unsqueeze(1)                        # drop dead dests
        if reach is not None:
            dead = dead | (reach < 0.5)                                     # drop sun-blocked dests
        self.glogits = logits.masked_fill(dead, -1e9)                       # masked logits for the greedy WHERE

    def _bundle(self, alloc, fire):
        return torch.cat([alloc, fire.unsqueeze(-1)], -1)                   # (B,E,E+1)

    def sample(self, crn_group=0):    # crn_group accepted for call-site parity; CRN coupling (docs/grpo_v12.tex [D])
        #                               is implemented for the (default) categorical WHERE, not the Dirichlet
        return self._bundle(self.dist.rsample(), torch.bernoulli(self.p))  # WHERE simplex + Bernoulli fire

    def greedy(self):
        alloc = torch.softmax(self.glogits / self.cfg.GREEDY_TAU, -1)      # sharpened, mask-aware -> concentrated
        fire = torch.bernoulli(self.p) if self.cfg.GREEDY_SAMPLE_GATE else (self.p >= 0.5).float()
        return self._bundle(alloc, fire.to(alloc.dtype))                   # DEPLOY: gate fires at the TRAINED rate

    def log_prob(self, a):
        E = self.alpha.shape[-1]
        alloc = a[..., :E].clamp_min(1e-6); alloc = alloc / alloc.sum(-1, keepdim=True)   # project to simplex
        fire = a[..., E]                                                   # (B,E) in {0,1}
        where_lp = self.dist.log_prob(alloc)                              # (B,E)
        gate_lp = fire * torch.log(self.p.clamp_min(1e-8)) + (1.0 - fire) * torch.log((1.0 - self.p).clamp_min(1e-8))
        lp = gate_lp + _rb_gate_where(self.cfg, fire, where_lp, self.p)   # [A1] Rao-Blackwell de-noise (or fire*where_lp)
        return (lp * self.owned).sum(1)                                   # (B,)

    def entropy(self):
        p = self.p
        gate_e = -(p * torch.log(p.clamp_min(1e-8)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-8)))
        # Dirichlet differential entropy is unboundedly NEGATIVE when peaked -> as a bonus it
        # blurs a BC-sharpened WHERE head (measured: loss +20, gn 1380, kl 4.8). Gate-only by default.
        ent = gate_e + self.cfg.ENT_DIR_COEF * p * self.dist.entropy()
        return (ent * self.owned).sum(1)                                  # (B,)


class GatedCatDist:
    """Gated per-source CATEGORICAL destination actor (v7 default).
      WHERE  masked logits glogits (B,E,E) -> Categorical over dests; action row = one-hot(dest).
      GATE   identical to GatedAllocDist (smooth p at train, hard trim at deploy).
    Same (B,E,E+1) action bundle as the Dirichlet -> decode/buffers/league untouched.
    log_prob is bounded-sensitivity (log_softmax), which is what makes PPO-after-BC stable."""
    def __init__(self, cfg, dest_logits, gate_logits, owned, alive=None, reach=None, ships=None, deploy=False):
        self.cfg = cfg
        self.owned = owned if owned.dim() == 2 else owned.squeeze(-1)
        logits = dest_logits.clamp(-cfg.LOGIT_CLAMP, cfg.LOGIT_CLAMP)
        g = torch.sigmoid(gate_logits if gate_logits.dim() == 2 else gate_logits.squeeze(-1))
        if deploy:
            p = ((g - cfg.GATE_TRIM_LO) / (cfg.GATE_TRIM_HI - cfg.GATE_TRIM_LO)).clamp(0.0, 1.0)
        else:
            p = g
        self.p = p.clamp(cfg.GATE_EPS, 1.0 - cfg.GATE_EPS)
        E = logits.shape[-1]
        eye = torch.eye(E, dtype=torch.bool, device=logits.device).unsqueeze(0)
        dead = eye.expand_as(logits).clone()
        if alive is not None:
            dead = dead | (alive < 0.5).unsqueeze(1)
        if reach is not None:
            dead = dead | (reach < 0.5)
        self.glogits = logits.masked_fill(dead, -1e9)
        self.lsm = torch.log_softmax(self.glogits, -1)                      # (B,E,E)

    def _bundle(self, dest, fire):
        alloc = torch.zeros_like(self.lsm)
        alloc.scatter_(2, dest.unsqueeze(-1), 1.0)                          # one-hot WHERE row
        return torch.cat([alloc, fire.unsqueeze(-1)], -1)

    def sample(self, crn_group=0):
        B_, E, _ = self.lsm.shape
        if crn_group and crn_group > 1 and B_ % crn_group == 0:
            # CRN (docs/grpo_v12.tex [D]): share the WHERE Gumbel + GATE uniform noise across the
            # crn_group rollouts of one world-group (envs are laid out in contiguous groups), so only
            # the ACTING agent's own randomness varies within a group -> lower-variance group baseline.
            G = crn_group; nW = B_ // G
            u = torch.rand(nW, 1, E, E, device=self.lsm.device).clamp_(1e-9, 1.0)
            gumbel = -torch.log(-torch.log(u))                              # Gumbel-max == Categorical(lsm)
            dest = (self.lsm.view(nW, G, E, E) + gumbel).argmax(-1).reshape(B_, E)
            ug = torch.rand(nW, 1, E, device=self.p.device)
            fire = (ug < self.p.view(nW, G, E)).to(self.lsm.dtype).reshape(B_, E)
            return self._bundle(dest, fire)
        dest = torch.distributions.Categorical(logits=self.lsm.reshape(-1, E)).sample().view(B_, E)
        return self._bundle(dest, torch.bernoulli(self.p))

    def greedy(self):
        fire = torch.bernoulli(self.p) if self.cfg.GREEDY_SAMPLE_GATE else (self.p >= 0.5).float()
        return self._bundle(self.glogits.argmax(-1), fire.to(self.lsm.dtype))

    def log_prob(self, a):
        E = self.lsm.shape[-1]
        dest = a[..., :E].argmax(-1)                                        # rows are one-hot
        fire = a[..., E]
        where_lp = self.lsm.gather(2, dest.unsqueeze(-1)).squeeze(-1)
        gate_lp = fire * torch.log(self.p.clamp_min(1e-8)) + (1.0 - fire) * torch.log((1.0 - self.p).clamp_min(1e-8))
        lp = gate_lp + _rb_gate_where(self.cfg, fire, where_lp, self.p)    # [A1] Rao-Blackwell de-noise (or fire*where_lp)
        return (lp * self.owned).sum(1)

    def entropy(self):
        p = self.p
        gate_e = -(p * torch.log(p.clamp_min(1e-8)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-8)))
        cat_e = -(self.lsm.exp() * self.lsm.clamp_min(-30.0)).sum(-1)       # bounded (<= ln E)
        ent = gate_e + self.cfg.ENT_DIR_COEF * p * cat_e
        return (ent * self.owned).sum(1)
