"""Mechanism probe: log-prob sensitivity of the WHERE distribution wrt its logits,
Dirichlet vs categorical, at (a) calm init and (b) BC-sharpened logits.

Two quantities per regime/distribution, over E=48-dim rows (CPU, no training):
  grad-norm   ||d(-log pi(a))/d logits||_2 for sampled actions  (how hard one SGD step kicks)
  delta-logp  |log pi'(a) - log pi(a)| after a unit-direction logit perturbation of size eps
              (how many NATS one weight-step of a given size moves a stored action's log-prob;
               PPO's clip/KL react to exactly this)

    python scripts/probe_dist_sensitivity.py
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
E, N = 48, 4096          # row dim, sample rows
EPS = 1e-2               # logit perturbation magnitude (unit direction * EPS)


def dir_logprob(logits, x):
    alpha = F.softplus(logits) + 1e-3
    return torch.distributions.Dirichlet(alpha).log_prob(x)


def cat_logprob(logits, idx):
    return torch.log_softmax(logits, -1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def probe(name, logits):
    out = {}
    # --- Dirichlet ---
    lg = logits.clone().requires_grad_(True)
    alpha = F.softplus(lg) + 1e-3
    x = torch.distributions.Dirichlet(alpha.detach()).sample().clamp_min(1e-6)
    x = x / x.sum(-1, keepdim=True)
    lp = dir_logprob(lg, x)
    g = torch.autograd.grad(lp.sum(), lg)[0]
    gn_dir = g.norm(dim=-1)
    d = torch.randn_like(logits); d = d / d.norm(dim=-1, keepdim=True)
    dlp_dir = (dir_logprob(logits + EPS * d, x) - dir_logprob(logits, x)).abs()
    # --- categorical (same logits) ---
    lg = logits.clone().requires_grad_(True)
    idx = torch.distributions.Categorical(logits=logits).sample()
    lp = cat_logprob(lg, idx)
    g = torch.autograd.grad(lp.sum(), lg)[0]
    gn_cat = g.norm(dim=-1)
    dlp_cat = (cat_logprob(logits + EPS * d, idx) - cat_logprob(logits, idx)).abs()

    def q(t):
        return "med %8.3f  p99 %10.3f  max %12.3f" % (t.median(), t.quantile(0.99), t.max())
    print(f"[{name}]  alpha range: {(F.softplus(logits)+1e-3).min():.3f}..{(F.softplus(logits)+1e-3).max():.3f}")
    print(f"  grad-norm  dir: {q(gn_dir)}")
    print(f"  grad-norm  cat: {q(gn_cat)}")
    print(f"  dlogp@{EPS:g}  dir: {q(dlp_dir)}   (nats moved by one {EPS:g}-size logit step)")
    print(f"  dlogp@{EPS:g}  cat: {q(dlp_cat)}")
    print()


# (a) calm init: INIT_DEST_SCALE=0.02 -> logits ~ N(0, 0.02^2); softplus(0)=0.693 -> alpha<1 EVERYWHERE
probe("calm init (alpha~0.69 < 1: corner-spiked density)", torch.randn(N, E) * 0.02)
# (b) generic untrained spread
probe("random init (logits ~ N(0,1))", torch.randn(N, E))
# (c) BC-sharpened: one strong target logit, rest suppressed (clamped at LOGIT_CLAMP=8)
sharp = torch.full((N, E), -6.0)
sharp[torch.arange(N), torch.randint(0, E, (N,))] = 6.0
sharp += torch.randn(N, E) * 0.5
probe("BC-sharpened (+6 target / -6 rest)", sharp.clamp(-8, 8))
