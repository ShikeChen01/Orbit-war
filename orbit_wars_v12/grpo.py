"""Group-Relative Policy Optimization update -- a critic-free PPO variant.

GRPO replaces PPO's *learned* value baseline with a GROUP baseline: several rollouts share one
initial world (assigned in :func:`orbit_wars_v12.rollout.collect_ppo` when ``cfg.ALGO == "grpo"``),
and each rollout's advantage is its return STANDARDIZED within its group. The advantage is therefore
precomputed in the rollout and stored in ``tb["advantage"]`` -- exactly like PPO -- so this update
just consumes it. Versus :func:`orbit_wars_v12.ppo.ppo_update` the only differences are:

  * **no value-function loss** -- the policy net's critic head is left untrained and unused;
  * an **optional KL-to-reference penalty** (DeepSeekMath ``beta`` = ``cfg.GRPO_KL_COEF``), estimated
    with Schulman's unbiased k3 estimator against a frozen reference net.

Everything else (clipped surrogate, entropy bonus, KL early-stop, grad clipping, GPU stat
accumulators) is shared with PPO. ``policy_coef`` is accepted for call-site parity but unused: there
is no critic to warm up, so the policy is always trained at full weight.
"""
import torch

from .constants import DTYPE
from .policy import evaluate
from .ppo import policy_surrogate


def grpo_update(cfg, net, opt, tb, ent_coef, policy_coef, ref_net=None):
    """``cfg.UPDATE_EPOCHS`` minibatch passes of the clipped surrogate on the group-relative
    advantages already in ``tb`` -- NO value loss, optional KL-to-``ref_net``. The KL-to-ref value is
    logged in the ``vf`` slot (there is no critic) so the training heartbeat prints unchanged."""
    N = tb["n"]; mb = cfg.MINIBATCHES; mbsize = N // mb
    s = {"total": 0.0, "policy": 0.0, "vf": 0.0, "entropy": 0.0, "sigma": 0.0,
         "approx_kl": 0.0, "clipfrac": 0.0, "grad_norm": 0.0}
    if mbsize == 0:
        return s
    nsteps = 0
    dev = tb["entities"].device
    _acc = {k: torch.zeros((), device=dev) for k in            # GPU-side stat accumulators: flush ONCE
            ("total", "policy", "vf", "entropy", "sigma", "clipfrac", "grad_norm")}   # at the end
    beta = cfg.GRPO_KL_COEF if ref_net is not None else 0.0
    for epoch in range(cfg.UPDATE_EPOCHS):
        perm = torch.randperm(N, device=dev)
        ep_kl = []
        for b in range(mb):
            mi = perm[b * mbsize:(b + 1) * mbsize]
            ent = tb["entities"].index_select(0, mi)          # already on GPU (no .to(DEVICE))
            em = tb["entity_mask"].index_select(0, mi)
            am = tb["action_mask"].index_select(0, mi)
            gl = tb["globals"].index_select(0, mi)
            act_mb = tb["action"].index_select(0, mi)
            oldlp = tb["old_logp"].index_select(0, mi)
            adv = tb["advantage"].index_select(0, mi)

            logp, entropy, _value = evaluate(cfg, net, ent, em, am, gl, act_mb)   # value head ignored
            pol = policy_surrogate(cfg, logp, oldlp, adv, cfg.CLIP)
            n_owned = am.sum(1).clamp_min(1.0)                # per-owned-source mean entropy (scale-correct)
            ent_b = (entropy / n_owned).mean()
            loss = pol - ent_coef * ent_b
            kl_ref = torch.zeros((), device=dev)
            if beta > 0.0:                                    # KL(pi || pi_ref), unbiased k3 estimator
                with torch.no_grad():
                    rlp, _, _ = evaluate(cfg, ref_net, ent, em, am, gl, act_mb)
                logr = (rlp - logp).clamp(-cfg.LOGRATIO_CLAMP, cfg.LOGRATIO_CLAMP)   # log(pi_ref/pi)
                kl_ref = (torch.exp(logr) - logr - 1.0).mean()                       # >= 0, grad flows via logp
                loss = loss + beta * kl_ref

            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.MAX_GRAD_NORM)
            if torch.isfinite(gnorm):
                opt.step()

            with torch.no_grad():
                lr = (logp - oldlp); lrc = lr.clamp(-cfg.LOGRATIO_CLAMP, cfg.LOGRATIO_CLAMP)
                ratio = torch.exp(lrc)
                mb_kl = ((ratio - 1.0) - lrc).mean().item()   # k3 KL(old||new) for the early-stop branch
                _acc["total"] += loss.detach(); _acc["policy"] += pol.detach()
                _acc["vf"] += kl_ref.detach()                 # no critic: surface KL-to-ref in the vf slot
                _acc["entropy"] += ent_b.detach(); _acc["sigma"] += ent_b.detach()
                s["approx_kl"] += mb_kl
                _acc["clipfrac"] += (torch.abs(ratio - 1.0) > cfg.CLIP).to(DTYPE).mean()
                _acc["grad_norm"] += gnorm.detach()
                ep_kl.append(mb_kl)
            nsteps += 1
            if cfg.KL_STOP_MINIBATCH and mb_kl > cfg.KL_HARD_MULT * cfg.KL_TARGET:
                break
        if cfg.KL_STOP_MINIBATCH and ep_kl and ep_kl[-1] > cfg.KL_HARD_MULT * cfg.KL_TARGET:
            break
        if cfg.KL_EARLYSTOP and ep_kl and (sum(ep_kl) / len(ep_kl)) > cfg.KL_TARGET:
            break
    for k in _acc:                       # ONE GPU->CPU sync for all logging stats
        s[k] = _acc[k].item()
    if nsteps:
        for k in s:
            s[k] /= nsteps
    return s
