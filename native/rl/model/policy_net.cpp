// PolicyNet implementation. Ported 1:1 from the old EntityPolicyImpl::forward (policy.hpp)
// with the critic head removed -- same ops/shapes, so a BC checkpoint reproduces identical
// logits (verified by ow_test_model).
#include "rl/model/policy_net.hpp"

#include <cmath>

namespace ow {

namespace { constexpr double NEG_INF = -1e9; }  // masked-class logit (kept local to this TU)

PolicyNetImpl::PolicyNetImpl(const ModelConfig& c) : cfg(c), A(c.actions_per_entity()) {
    int F = c.n_entity_features, G = c.n_global_features, h = c.hidden;
    int nf = (int)c.fractions.size();
    ent0 = register_module("ent0", torch::nn::Linear(F, h));
    ent2 = register_module("ent2", torch::nn::Linear(h, h));
    ctx0 = register_module("ctx0", torch::nn::Linear(h + G, h));
    ctx2 = register_module("ctx2", torch::nn::Linear(h, h));
    ent_glu_gate = register_module("ent_glu_gate", torch::nn::Linear(h, h));
    ent_glu_val = register_module("ent_glu_val", torch::nn::Linear(h, h));
    ent_glu_out = register_module("ent_glu_out", torch::nn::Linear(h, h));
    ctx_glu_gate = register_module("ctx_glu_gate", torch::nn::Linear(h, h));
    ctx_glu_val = register_module("ctx_glu_val", torch::nn::Linear(h, h));
    ctx_glu_out = register_module("ctx_glu_out", torch::nn::Linear(h, h));
    if (c.target_mode) {
        aq = register_module("aq", torch::nn::Linear(h + h, nf * h));
        ak = register_module("ak", torch::nn::Linear(h, h));
        anoop = register_module("anoop", torch::nn::Linear(h + h, 1));
    } else {
        act0 = register_module("act0", torch::nn::Linear(h + h, h));
        act2 = register_module("act2", torch::nn::Linear(h, A));
    }
}

torch::Tensor PolicyNetImpl::forward(const torch::Tensor& entities,
                                     const torch::Tensor& entity_mask,
                                     const torch::Tensor& action_mask,
                                     const torch::Tensor& globals) {
    auto tok = torch::relu(ent0->forward(entities));
    tok = ent2->forward(tok);                                  // (B,E,h)
    tok = tok + ent_glu_out->forward(ent_glu_val->forward(tok) *
                                     torch::sigmoid(ent_glu_gate->forward(tok)));  // residual GLU
    auto m = entity_mask.unsqueeze(-1);                        // (B,E,1)
    auto pooled = (tok * m).sum(1) / m.sum(1).clamp_min(1.0);  // (B,h)
    auto core = torch::relu(ctx0->forward(torch::cat({pooled, globals}, -1)));
    core = ctx2->forward(core);                                // (B,h)
    core = core + ctx_glu_out->forward(ctx_glu_val->forward(core) *
                                       torch::sigmoid(ctx_glu_gate->forward(core)));  // residual GLU
    auto core_b = core.unsqueeze(1).expand({-1, tok.size(1), -1});  // (B,E,h)
    int nf = (int)cfg.fractions.size();

    torch::Tensor logits;
    if (cfg.target_mode) {
        long B = tok.size(0), E = tok.size(1);
        auto q = aq->forward(torch::cat({tok, core_b}, -1)).reshape({B, E, nf, cfg.hidden});
        auto k = ak->forward(tok);                             // (B,E,h)
        auto score = torch::einsum("brfh,bth->brtf", {q, k}) / std::sqrt((double)cfg.hidden);
        auto noop = anoop->forward(torch::cat({tok, core_b}, -1));   // (B,E,1)
        logits = torch::cat({noop, score.reshape({B, E, E * nf})}, -1);  // (B,E,A)
    } else {
        auto hh = torch::relu(act0->forward(torch::cat({tok, core_b}, -1)));
        logits = act2->forward(hh);
    }

    if (cfg.target_mode) {
        long B = logits.size(0), E = logits.size(1);
        long nfl = (A - 1) / E;
        auto noop = logits.narrow(-1, 0, 1);
        auto launch = logits.narrow(-1, 1, A - 1).reshape({B, E, E, nfl});  // (B,r,t,f)
        auto launcher_ok = (action_mask > 0.5f).view({B, E, 1, 1});
        auto tgt_real = (entity_mask > 0.5f).view({B, 1, E, 1});
        auto eye = torch::eye(E, logits.options()).view({1, E, E, 1});
        auto allowed = launcher_ok & tgt_real & (eye < 0.5);
        launch = torch::where(allowed, launch, torch::full_like(launch, NEG_INF));
        logits = torch::cat({noop, launch.reshape({B, E, A - 1})}, -1);
    } else {
        auto non_actionable = (action_mask < 0.5).unsqueeze(-1);
        auto cls_index = torch::arange(A, logits.options()).view({1, 1, A});
        auto blocked = (cls_index > 0) & non_actionable;
        logits = torch::where(blocked, torch::full_like(logits, NEG_INF), logits);
    }
    return logits;
}

}  // namespace ow
