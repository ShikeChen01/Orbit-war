// The action distribution: a masked Categorical over each entity's class logits. A single
// env-step's action is the *joint* choice across all real entities, so the joint log-prob
// and entropy are sums over entities (weighted by the entity mask). This type makes every
// quantity GRPO needs explicit -- sampling, the log-prob for the importance ratio, the
// entropy bonus, and the full-distribution KL to a reference policy.
//
// Interface only -- math lives in distribution.cpp.
#pragma once
#include <torch/torch.h>

namespace ow {

class MaskedPerEntityCategorical {
public:
    // logits (B,E,A) already masked (illegal classes at -inf); entity_mask (B,E) marks
    // real entities. log_softmax is computed once and cached.
    MaskedPerEntityCategorical(torch::Tensor logits, torch::Tensor entity_mask);

    torch::Tensor sample() const;                               // (B,E) int64, multinomial
    torch::Tensor greedy() const;                               // (B,E) int64, argmax
    torch::Tensor log_prob(const torch::Tensor& action) const; // (B,) summed over entities
    torch::Tensor entropy() const;                             // (B,) summed over entities
    torch::Tensor kl_to(const MaskedPerEntityCategorical& ref) const;  // (B,) KL(this||ref)

private:
    torch::Tensor logits_;       // (B,E,A)
    torch::Tensor entity_mask_;  // (B,E)
    torch::Tensor log_probs_;    // (B,E,A) = log_softmax(logits_, -1)
};

}  // namespace ow
