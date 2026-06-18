"""PPO clipped-surrogate update (notebook cell 25).

``ppo_update`` runs ``cfg.UPDATE_EPOCHS`` minibatch passes with KL early-stop, PopArt value
targets, grad clipping, and GPU-side stat accumulators flushed once at the end. The two values the
notebook mutated as globals are now explicit args: ``ent_coef`` (the annealed entropy coefficient,
was ``CUR_ENT_COEF``) and ``policy_coef`` (0 during critic warmup, was ``PPO_POLICY_COEF``).
"""
import torch
import torch.nn.functional as F

from .constants import DTYPE
from .policy import evaluate

_HALF_LOG2PIE_C = 1.4189385332046727   # (kept from the notebook; vestigial Gaussian-actor constant)


def policy_surrogate(cfg, logp, oldlp, adv, clip):
    ratio = torch.exp((logp - oldlp).clamp(-cfg.LOGRATIO_CLAMP, cfg.LOGRATIO_CLAMP))
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv
    return -torch.minimum(unclipped, clipped).mean()


def ppo_update(cfg, net, opt, tb, ent_coef, policy_coef):
    N = tb["n"]; mb = cfg.MINIBATCHES; mbsize = N // mb
    s = {"total": 0.0, "policy": 0.0, "vf": 0.0, "entropy": 0.0, "sigma": 0.0,
         "approx_kl": 0.0, "clipfrac": 0.0, "grad_norm": 0.0}
    if mbsize == 0:
        return s
    nsteps = 0
    dev = tb["entities"].device
    _acc = {k: torch.zeros((), device=dev) for k in            # GPU-side stat accumulators: flush ONCE at
            ("total", "policy", "vf", "entropy", "sigma", "clipfrac", "grad_norm")}   # the end (was ~6 .item()/mb)
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
            ret = tb["returns"].index_select(0, mi)

            logp, entropy, value = evaluate(cfg, net, ent, em, am, gl, act_mb)
            pol = policy_surrogate(cfg, logp, oldlp, adv, cfg.CLIP)
            vtarget = net.popart.normalize(ret) if (cfg.USE_POPART and getattr(net, "popart", None) is not None) else ret
            vloss = F.mse_loss(value, vtarget)
            n_owned = am.sum(1).clamp_min(1.0)                # per-owned-source mean entropy (scale-correct)
            ent_b = (entropy / n_owned).mean()
            pc = policy_coef   # 0 during value warmup
            loss = pc * pol + cfg.VF_COEF * vloss - (pc * ent_coef) * ent_b

            opt.zero_grad()
            loss.backward()
            if pc == 0.0:                                  # critic warmup: val-only step --
                for pn, pp in net.named_parameters():      # shared-trunk vf grads otherwise
                    if not pn.startswith('val_'):          # drag the sharp BC policy around
                        pp.grad = None
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.MAX_GRAD_NORM)
            if torch.isfinite(gnorm):
                opt.step()
            if cfg.USE_POPART and getattr(net, "popart", None) is not None:
                net.popart.update(ret, net.val_out)           # output-preserving stats update

            with torch.no_grad():
                lr = (logp - oldlp); lrc = lr.clamp(-cfg.LOGRATIO_CLAMP, cfg.LOGRATIO_CLAMP)
                ratio = torch.exp(lrc)
                mb_kl = ((ratio - 1.0) - lrc).mean().item()   # k3 KL estimator (>=0)
                _acc["total"] += loss.detach(); _acc["policy"] += pol.detach(); _acc["vf"] += vloss.detach()
                _acc["entropy"] += ent_b.detach(); _acc["sigma"] += ent_b.detach()
                s["approx_kl"] += mb_kl   # mb_kl already on CPU (the KL early-stop branch needs it)
                _acc["clipfrac"] += (torch.abs(ratio - 1.0) > cfg.CLIP).to(DTYPE).mean()
                _acc["grad_norm"] += gnorm.detach()
                ep_kl.append(mb_kl)
            nsteps += 1
            if cfg.KL_STOP_MINIBATCH and mb_kl > cfg.KL_HARD_MULT * cfg.KL_TARGET:
                break
        if cfg.KL_STOP_MINIBATCH and ep_kl and ep_kl[-1] > cfg.KL_HARD_MULT * cfg.KL_TARGET:
            break
        if cfg.KL_EARLYSTOP and ep_kl and (sum(ep_kl) / len(ep_kl)) > cfg.KL_TARGET:   # end-of-epoch (mean), not mid-minibatch
            break
    for k in _acc:                       # ONE GPU->CPU sync for all logging stats (was ~6 per minibatch)
        s[k] = _acc[k].item()
    if nsteps:
        for k in s:
            s[k] /= nsteps
    return s
