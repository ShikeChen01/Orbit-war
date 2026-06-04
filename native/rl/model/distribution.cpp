// MaskedPerEntityCategorical implementation. The joint action of one env-step is the product
// over real entities, so log_prob / entropy / KL are entity sums weighted by the entity mask
// (Eqs. 9-11 of docs/rl_math.pdf). Masked classes sit at NEG_INF so they contribute ~0.
#include "rl/model/distribution.hpp"

namespace ow {

MaskedPerEntityCategorical::MaskedPerEntityCategorical(torch::Tensor logits,
                                                       torch::Tensor entity_mask)
    : logits_(std::move(logits)), entity_mask_(std::move(entity_mask)) {
    log_probs_ = torch::log_softmax(logits_, -1);
}

torch::Tensor MaskedPerEntityCategorical::sample() const {
    auto p = torch::softmax(logits_, -1);
    auto B = p.size(0), E = p.size(1), A = p.size(2);
    return torch::multinomial(p.view({B * E, A}), 1).view({B, E});
}

torch::Tensor MaskedPerEntityCategorical::greedy() const {
    return std::get<1>(logits_.max(-1));
}

torch::Tensor MaskedPerEntityCategorical::log_prob(const torch::Tensor& action) const {
    auto chosen = log_probs_.gather(-1, action.unsqueeze(-1)).squeeze(-1);  // (B,E)
    return (chosen * entity_mask_).sum(1);                                  // (B,)
}

torch::Tensor MaskedPerEntityCategorical::entropy() const {
    auto p = torch::exp(log_probs_);
    auto ent = -(p * log_probs_).sum(-1);                                   // (B,E)
    return (ent * entity_mask_).sum(1);                                     // (B,)
}

torch::Tensor MaskedPerEntityCategorical::kl_to(const MaskedPerEntityCategorical& ref) const {
    auto p = torch::exp(log_probs_);
    auto kl = (p * (log_probs_ - ref.log_probs_)).sum(-1);                  // (B,E)
    return (kl * entity_mask_).sum(1);                                      // (B,)
}

}  // namespace ow
