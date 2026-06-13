// The action distribution: a squashed diagonal Gaussian over the E x 2K continuous controls.
// A single env-step's action is the joint choice across all planet slots, so the joint log-prob
// and entropy are sums over slots and the 2K params. Header-only (elementwise tensor ops).
#pragma once
#include <torch/torch.h>

namespace ow {

// === Continuous actor: squashed diagonal Gaussian (docs/rl_math.pdf sec:dist) ================
// Over the E x 2K controls, per component a = sigmoid(z), z ~ N(mu, sigma^2), sigma = exp(varsigma).
// Joint log-prob / entropy are SUMS over ALL E planet slots AND all 2K params (NOT masked -- the
// invalid-dispatch penalty, not a mask, drives phantom/unowned slots toward "no launch"). Header-
// only: it is just a few elementwise ops on the (B,E,2K) mean/log-std tensors.
class SquashedGaussian {
public:
    // mean, logstd: (B,E,3K). logstd is assumed already clipped to [varsigma_min, varsigma_max].
    // OPTIONAL mask: (B,E) per-planet, 1=active / 0=masked (e.g. unowned/empty planet). When given,
    // masked planets are FORCED to the no-op action (all-zero -> phi=0 -> never commits) and are
    // EXCLUDED from the joint log-prob and entropy sums, so the policy only "decides" on the legal
    // (owned) slots -- the gradient stops being spent on, and diluted by, the ~E*K phantom/unowned
    // controls. Without a mask the actor is unmasked (legality learned from the dispatch penalty).
    SquashedGaussian(torch::Tensor mean, torch::Tensor logstd, torch::Tensor mask = {})
        : mean_(std::move(mean)), logstd_(std::move(logstd)), std_(torch::exp(logstd_)) {
        if (mask.defined()) mask_ = (mask.dim() == 2 ? mask.unsqueeze(-1) : mask);  // (B,E,1) broadcast
    }

    // a sampled action in (0,1), shape (B,E,3K). reparameterized (mu + sigma*eps), then squashed.
    // masked planets -> all-zero action (phi=0 -> no commit in decode).
    torch::Tensor sample() const {
        auto a = torch::sigmoid(mean_ + std_ * torch::randn_like(mean_));
        return mask_.defined() ? a * mask_ : a;
    }
    // greedy action: squash the mean (no exploration), shape (B,E,3K).
    torch::Tensor greedy() const {
        auto a = torch::sigmoid(mean_);
        return mask_.defined() ? a * mask_ : a;
    }

    // (B,) joint log-prob of action a (B,E,3K), incl. the sigmoid change-of-variables term.
    torch::Tensor log_prob(const torch::Tensor& a) const {
        auto ac = a.clamp(1e-6, 1.0 - 1e-6);
        auto z = torch::log(ac) - torch::log1p(-ac);            // logit(a) recovers the latent z
        // clamp mean in the density: |mean|>15 already saturates sigmoid(mean) to ~0/1, so this
        // leaves the action distribution effectively unchanged but bounds (z-mean)^2 (and its
        // gradient) when a deep net momentarily emits an extreme mean -- keeps log-prob finite.
        auto mu = mean_.clamp(-15.0, 15.0);
        auto logN = -0.5 * (z - mu).pow(2) / (std_ * std_)      // log N(z; mu, sigma^2)
                    - logstd_ - 0.5 * kLog2Pi;
        auto logjac = torch::log(ac) + torch::log1p(-ac);       // log|da/dz| = log(a(1-a))
        auto term = logN - logjac;                              // (B,E,3K)
        if (mask_.defined()) term = term * mask_;  // masked planets contribute 0 (no grad to them)
        return term.sum(2).sum(1);
    }
    // (B,) entropy bonus: closed-form latent-Gaussian differential entropy, summed (= total log-std
    // plus a constant). Drives the exploration scale directly (dH/dvarsigma = 1).
    torch::Tensor entropy() const {
        auto e = logstd_ + kHalfLog2PiE;                        // (B,E,3K)
        if (mask_.defined()) e = e * mask_;
        return e.sum(2).sum(1);
    }

private:
    static constexpr double kLog2Pi = 1.8378770664093453;       // log(2*pi)
    static constexpr double kHalfLog2PiE = 1.4189385332046727;  // 0.5*log(2*pi*e)
    torch::Tensor mean_, logstd_, std_, mask_;
};

// === v5 TARGET actor: per-planet (destination categorical) x (phi squashed-Gaussian) ============
// For each OWNED planet the action is (dest, phi): a destination planet sampled from a categorical
// over the E obs slots (masked to LIVE planets), and a launch fraction phi = sigmoid(N(mu,sigma^2))
// of that planet's ships -- identical to the continuous actor's phi. The joint log-prob/entropy sum
// over owned planets only (unowned slots are forced to no-op and dropped, like SquashedGaussian's
// mask). The stored action is (B,E,2): [...,0]=dest row index (float), [...,1]=phi in (0,1).
class TargetActorDist {
public:
    // dest_logits (B,E,E); phi_mean/phi_logstd (B,E,1) [logstd pre-clipped]; owned (B,E) = legal
    // source planets; alive (B,E) = valid destination slots.
    TargetActorDist(torch::Tensor dest_logits, torch::Tensor phi_mean, torch::Tensor phi_logstd,
                    torch::Tensor owned, torch::Tensor alive)
        : owned_(owned.dim() == 2 ? owned : owned.squeeze(-1)),
          pmean_(phi_mean.squeeze(-1)),                       // (B,E)
          plogstd_(phi_logstd.squeeze(-1)),                   // (B,E)
          pstd_(torch::exp(phi_logstd.squeeze(-1))) {
        auto dead_dest = (alive < 0.5).unsqueeze(1);          // (B,1,E) broadcast over the source dim
        ml_ = dest_logits.masked_fill(dead_dest, -1e9);       // (B,E,E) only live destinations remain
        dlogp_ = torch::log_softmax(ml_, -1);                 // (B,E,E)
    }

    // sampled action (B,E,2): [dest (long-as-float), phi in (0,1)]. Unowned planets -> all-zero
    // (phi=0 -> never commits; dest 0 is harmless because the slot is dropped from log-prob).
    torch::Tensor sample() const {
        long B = ml_.size(0), E = ml_.size(1);
        auto p = torch::softmax(ml_, -1).reshape({B * E, E});
        auto dest = torch::multinomial(p, 1).reshape({B, E}).to(torch::kFloat32);  // (B,E)
        auto phi = torch::sigmoid(pmean_ + pstd_ * torch::randn_like(pmean_));      // (B,E)
        return mask_action(torch::stack({dest, phi}, -1));
    }
    // greedy: argmax destination + squashed mean phi.
    torch::Tensor greedy() const {
        auto dest = std::get<1>(ml_.max(-1)).to(torch::kFloat32);  // (B,E)
        auto phi = torch::sigmoid(pmean_);                          // (B,E)
        return mask_action(torch::stack({dest, phi}, -1));
    }

    // (B,) joint log-prob: sum over OWNED planets of [log p(dest) + squashed-Gaussian log p(phi)].
    torch::Tensor log_prob(const torch::Tensor& a) const {
        long E = ml_.size(1);
        auto dest = a.select(-1, 0).to(torch::kLong).clamp(0, E - 1);      // (B,E)
        auto dest_lp = dlogp_.gather(2, dest.unsqueeze(-1)).squeeze(-1);   // (B,E)
        auto phi = a.select(-1, 1);                                        // (B,E)
        auto ac = phi.clamp(1e-6, 1.0 - 1e-6);
        auto z = torch::log(ac) - torch::log1p(-ac);
        auto mu = pmean_.clamp(-15.0, 15.0);
        auto logN = -0.5 * (z - mu).pow(2) / (pstd_ * pstd_) - plogstd_ - 0.5 * kLog2Pi;
        auto logjac = torch::log(ac) + torch::log1p(-ac);                  // log|da/dz| = log(a(1-a))
        auto phi_lp = logN - logjac;                                       // (B,E)
        return ((dest_lp + phi_lp) * owned_).sum(1);                       // owned-masked sum
    }
    // (B,) entropy: categorical entropy + phi latent-Gaussian entropy, owned-masked sum.
    torch::Tensor entropy() const {
        auto cat_e = -(torch::softmax(ml_, -1) * dlogp_).sum(-1);         // (B,E)
        auto phi_e = plogstd_ + kHalfLog2PiE;                            // (B,E)
        return ((cat_e + phi_e) * owned_).sum(1);
    }

private:
    torch::Tensor mask_action(torch::Tensor a) const {  // zero out unowned planets (-> no-op)
        return a * owned_.unsqueeze(-1);
    }
    static constexpr double kLog2Pi = 1.8378770664093453;
    static constexpr double kHalfLog2PiE = 1.4189385332046727;
    torch::Tensor owned_, pmean_, plogstd_, pstd_, ml_, dlogp_;
};

}  // namespace ow
