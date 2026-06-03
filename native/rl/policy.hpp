// EntityPolicy in LibTorch, mirroring orbit_wars_rl/agents/ppo_policy.py:EntityPolicy.
// Layer shapes/order match so weights can be loaded into the Python model 1:1.
#pragma once
#include <torch/torch.h>

namespace ow {

constexpr double NEG_INF = -1e9;

struct EntityPolicyImpl : torch::nn::Module {
    int A;
    torch::nn::Linear ent0{nullptr}, ent2{nullptr}, ctx0{nullptr}, ctx2{nullptr};
    torch::nn::Linear act0{nullptr}, act2{nullptr}, crit0{nullptr}, crit2{nullptr};

    EntityPolicyImpl(int n_entity_features, int n_global_features, int actions_per_entity,
                     int hidden = 128)
        : A(actions_per_entity) {
        ent0 = register_module("ent0", torch::nn::Linear(n_entity_features, hidden));
        ent2 = register_module("ent2", torch::nn::Linear(hidden, hidden));
        ctx0 = register_module("ctx0", torch::nn::Linear(hidden + n_global_features, hidden));
        ctx2 = register_module("ctx2", torch::nn::Linear(hidden, hidden));
        act0 = register_module("act0", torch::nn::Linear(hidden + hidden, hidden));
        act2 = register_module("act2", torch::nn::Linear(hidden, actions_per_entity));
        crit0 = register_module("crit0", torch::nn::Linear(hidden, hidden));
        crit2 = register_module("crit2", torch::nn::Linear(hidden, 1));
    }

    // Returns {logits (B,E,A) masked, value (B,)}.
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& entities,
                                                    const torch::Tensor& entity_mask,
                                                    const torch::Tensor& action_mask,
                                                    const torch::Tensor& globals) {
        auto tok = torch::relu(ent0->forward(entities));
        tok = ent2->forward(tok);  // (B,E,h)
        auto m = entity_mask.unsqueeze(-1);                      // (B,E,1)
        auto pooled = (tok * m).sum(1) / m.sum(1).clamp_min(1.0);  // (B,h)
        auto core = torch::relu(ctx0->forward(torch::cat({pooled, globals}, -1)));
        core = ctx2->forward(core);                              // (B,h)
        auto core_b = core.unsqueeze(1).expand({-1, tok.size(1), -1});
        auto h = torch::relu(act0->forward(torch::cat({tok, core_b}, -1)));
        auto logits = act2->forward(h);                          // (B,E,A)
        // Mask: class 0 always allowed; classes >0 only where action_mask==1.
        auto non_actionable = (action_mask < 0.5).unsqueeze(-1);  // (B,E,1)
        auto cls_index = torch::arange(A, logits.options()).view({1, 1, A});
        auto blocked = (cls_index > 0) & non_actionable;        // (B,E,A)
        logits = torch::where(blocked, torch::full_like(logits, NEG_INF), logits);
        auto vh = torch::relu(crit0->forward(core));
        auto value = crit2->forward(vh).squeeze(-1);            // (B,)
        return {logits, value};
    }
};
TORCH_MODULE(EntityPolicy);

// Per-entity categorical helpers (log-prob summed over real entities).
inline torch::Tensor summed_logprob(const torch::Tensor& logits, const torch::Tensor& action,
                                    const torch::Tensor& entity_mask) {
    auto logp = torch::log_softmax(logits, -1);                  // (B,E,A)
    auto chosen = logp.gather(-1, action.unsqueeze(-1)).squeeze(-1);  // (B,E)
    return (chosen * entity_mask).sum(1);                        // (B,)
}

inline torch::Tensor entropy_sum(const torch::Tensor& logits, const torch::Tensor& entity_mask) {
    auto logp = torch::log_softmax(logits, -1);
    auto p = torch::exp(logp);
    auto ent = -(p * logp).sum(-1);                              // (B,E)
    return (ent * entity_mask).sum(1);                           // (B,)
}

// Sample per-entity actions from masked logits. Returns (action (B,E) int64).
inline torch::Tensor sample_actions(const torch::Tensor& logits) {
    auto p = torch::softmax(logits, -1);                        // (B,E,A)
    auto B = p.size(0), E = p.size(1), A = p.size(2);
    auto flat = p.view({B * E, A});
    auto a = torch::multinomial(flat, 1).view({B, E});
    return a;
}

inline torch::Tensor greedy_actions(const torch::Tensor& logits) {
    return std::get<1>(logits.max(-1));
}

}  // namespace ow
