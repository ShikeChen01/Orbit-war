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
    // mean, logstd: (B,E,2K). logstd is assumed already clipped to [varsigma_min, varsigma_max].
    SquashedGaussian(torch::Tensor mean, torch::Tensor logstd)
        : mean_(std::move(mean)), logstd_(std::move(logstd)), std_(torch::exp(logstd_)) {}

    // a sampled action in (0,1), shape (B,E,2K). reparameterized (mu + sigma*eps), then squashed.
    torch::Tensor sample() const {
        return torch::sigmoid(mean_ + std_ * torch::randn_like(mean_));
    }
    // greedy action: squash the mean (no exploration), shape (B,E,2K).
    torch::Tensor greedy() const { return torch::sigmoid(mean_); }

    // (B,) joint log-prob of action a (B,E,2K), incl. the sigmoid change-of-variables term.
    torch::Tensor log_prob(const torch::Tensor& a) const {
        auto ac = a.clamp(1e-6, 1.0 - 1e-6);
        auto z = torch::log(ac) - torch::log1p(-ac);            // logit(a) recovers the latent z
        auto logN = -0.5 * (z - mean_).pow(2) / (std_ * std_)   // log N(z; mu, sigma^2)
                    - logstd_ - 0.5 * kLog2Pi;
        auto logjac = torch::log(ac) + torch::log1p(-ac);       // log|da/dz| = log(a(1-a))
        return (logN - logjac).sum(2).sum(1);
    }
    // (B,) entropy bonus: closed-form latent-Gaussian differential entropy, summed (= total log-std
    // plus a constant). Drives the exploration scale directly (dH/dvarsigma = 1).
    torch::Tensor entropy() const { return (logstd_ + kHalfLog2PiE).sum(2).sum(1); }

private:
    static constexpr double kLog2Pi = 1.8378770664093453;       // log(2*pi)
    static constexpr double kHalfLog2PiE = 1.4189385332046727;  // 0.5*log(2*pi*e)
    torch::Tensor mean_, logstd_, std_;
};

}  // namespace ow
