// EntityPolicy in LibTorch, mirroring orbit_wars_rl/agents/ppo_policy.py:EntityPolicy.
// Layer shapes/order match so weights can be loaded into the Python model 1:1.
#pragma once
#include <torch/torch.h>

#include <map>
#include <string>

namespace ow {

constexpr double NEG_INF = -1e9;

struct EntityPolicyImpl : torch::nn::Module {
    int A, hidden, num_fracs;
    bool target_mode;  // true: actor classes are (target_row, frac); false: (angle_bin, frac)
    torch::nn::Linear ent0{nullptr}, ent2{nullptr}, ctx0{nullptr}, ctx2{nullptr};
    torch::nn::Linear act0{nullptr}, act2{nullptr}, crit0{nullptr}, crit2{nullptr};
    // Residual GLU blocks (one per encoder): out = x + W_o( (W_v x) * sigmoid(W_g x) ).
    torch::nn::Linear ent_glu_gate{nullptr}, ent_glu_val{nullptr}, ent_glu_out{nullptr};
    torch::nn::Linear ctx_glu_gate{nullptr}, ctx_glu_val{nullptr}, ctx_glu_out{nullptr};
    // Target-mode pointer actor: logit(r,t,f) = <q_f(tok_r,core), k(tok_t)> + noop head.
    torch::nn::Linear aq{nullptr}, ak{nullptr}, anoop{nullptr};

    EntityPolicyImpl(int n_entity_features, int n_global_features, int actions_per_entity,
                     int hidden_ = 128, bool target_mode_ = false, int num_fracs_ = 4)
        : A(actions_per_entity), hidden(hidden_), num_fracs(num_fracs_), target_mode(target_mode_) {
        ent0 = register_module("ent0", torch::nn::Linear(n_entity_features, hidden));
        ent2 = register_module("ent2", torch::nn::Linear(hidden, hidden));
        ctx0 = register_module("ctx0", torch::nn::Linear(hidden + n_global_features, hidden));
        ctx2 = register_module("ctx2", torch::nn::Linear(hidden, hidden));
        crit0 = register_module("crit0", torch::nn::Linear(hidden, hidden));
        crit2 = register_module("crit2", torch::nn::Linear(hidden, 1));
        ent_glu_gate = register_module("ent_glu_gate", torch::nn::Linear(hidden, hidden));
        ent_glu_val = register_module("ent_glu_val", torch::nn::Linear(hidden, hidden));
        ent_glu_out = register_module("ent_glu_out", torch::nn::Linear(hidden, hidden));
        ctx_glu_gate = register_module("ctx_glu_gate", torch::nn::Linear(hidden, hidden));
        ctx_glu_val = register_module("ctx_glu_val", torch::nn::Linear(hidden, hidden));
        ctx_glu_out = register_module("ctx_glu_out", torch::nn::Linear(hidden, hidden));
        if (target_mode) {
            aq = register_module("aq", torch::nn::Linear(hidden + hidden, num_fracs * hidden));
            ak = register_module("ak", torch::nn::Linear(hidden, hidden));
            anoop = register_module("anoop", torch::nn::Linear(hidden + hidden, 1));
        } else {
            act0 = register_module("act0", torch::nn::Linear(hidden + hidden, hidden));
            act2 = register_module("act2", torch::nn::Linear(hidden, actions_per_entity));
        }
    }

    // Returns {logits (B,E,A) masked, value (B,)}.
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& entities,
                                                    const torch::Tensor& entity_mask,
                                                    const torch::Tensor& action_mask,
                                                    const torch::Tensor& globals) {
        auto tok = torch::relu(ent0->forward(entities));
        tok = ent2->forward(tok);  // (B,E,h)
        tok = tok + ent_glu_out->forward(ent_glu_val->forward(tok) *
                                         torch::sigmoid(ent_glu_gate->forward(tok)));  // residual GLU
        auto m = entity_mask.unsqueeze(-1);                      // (B,E,1)
        auto pooled = (tok * m).sum(1) / m.sum(1).clamp_min(1.0);  // (B,h)
        auto core = torch::relu(ctx0->forward(torch::cat({pooled, globals}, -1)));
        core = ctx2->forward(core);                              // (B,h)
        core = core + ctx_glu_out->forward(ctx_glu_val->forward(core) *
                                           torch::sigmoid(ctx_glu_gate->forward(core)));  // residual GLU
        auto core_b = core.unsqueeze(1).expand({-1, tok.size(1), -1});  // (B,E,h)
        torch::Tensor logits;
        if (target_mode) {
            // Pointer actor: per-launcher query (one per frac) dotted with per-target key.
            long B = tok.size(0), E = tok.size(1);
            auto q = aq->forward(torch::cat({tok, core_b}, -1));     // (B,E, nf*h)
            q = q.reshape({B, E, num_fracs, hidden});               // (B,r,f,h)
            auto k = ak->forward(tok);                              // (B,E,h)  per-target key
            // score(r,t,f) = <q[r,f], k[t]> / sqrt(h)
            auto score = torch::einsum("brfh,bth->brtf", {q, k}) / std::sqrt((double)hidden);
            auto noop = anoop->forward(torch::cat({tok, core_b}, -1));  // (B,E,1)
            // launch class index = 1 + t*nf + f; score is (B,r,t,f) contiguous, so the
            // (t,f) reshape yields exactly t*nf+f.
            logits = torch::cat({noop, score.reshape({B, E, E * num_fracs})}, -1);  // (B,E,A)
        } else {
            auto h = torch::relu(act0->forward(torch::cat({tok, core_b}, -1)));
            logits = act2->forward(h);                          // (B,E,A)
        }
        if (target_mode) {
            // Classes 1.. are (target_row t, frac f). Allow only if launcher r is
            // actionable, target t is a real entity, and t != r. Class 0 (noop) always ok.
            long B = logits.size(0), E = logits.size(1);
            long nf = (A - 1) / E;
            auto noop = logits.narrow(-1, 0, 1);                       // (B,E,1)
            auto launch = logits.narrow(-1, 1, A - 1).reshape({B, E, E, nf});  // (B,r,t,f)
            auto launcher_ok = (action_mask > 0.5f).view({B, E, 1, 1});
            auto tgt_real = (entity_mask > 0.5f).view({B, 1, E, 1});
            auto eye = torch::eye(E, logits.options()).view({1, E, E, 1});
            auto allowed = launcher_ok & tgt_real & (eye < 0.5);      // (B,E,E,1) bool
            launch = torch::where(allowed, launch, torch::full_like(launch, NEG_INF));
            logits = torch::cat({noop, launch.reshape({B, E, A - 1})}, -1);
        } else {
            // Angle-bin mode: class 0 always allowed; classes >0 only where action_mask==1.
            auto non_actionable = (action_mask < 0.5).unsqueeze(-1);  // (B,E,1)
            auto cls_index = torch::arange(A, logits.options()).view({1, 1, A});
            auto blocked = (cls_index > 0) & non_actionable;        // (B,E,A)
            logits = torch::where(blocked, torch::full_like(logits, NEG_INF), logits);
        }
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

// Load weights keyed by the *Python* EntityPolicy.state_dict() names (e.g.
// "entity_encoder.0.weight") into a C++ EntityPolicy. Inverse of
// Trainer::get_state_dict; lets the arena play exported checkpoints.
inline void load_python_state_dict(EntityPolicy& policy,
                                   const std::map<std::string, torch::Tensor>& sd) {
    auto load = [&](const std::string& key, torch::nn::Linear& lin) {
        auto wi = sd.find(key + ".weight");
        auto bi = sd.find(key + ".bias");
        TORCH_CHECK(wi != sd.end() && bi != sd.end(), "missing weights for ", key);
        torch::NoGradGuard ng;
        lin->weight.copy_(wi->second.reshape(lin->weight.sizes()));
        lin->bias.copy_(bi->second.reshape(lin->bias.sizes()));
    };
    auto load_opt = [&](const std::string& key, torch::nn::Linear& lin) {  // critic is optional:
        auto wi = sd.find(key + ".weight");                                // GRPO checkpoints omit
        auto bi = sd.find(key + ".bias");                                  // it; arena eval uses
        if (wi == sd.end() || bi == sd.end()) return;                      // logits only, so a
        torch::NoGradGuard ng;                                             // random critic is fine.
        lin->weight.copy_(wi->second.reshape(lin->weight.sizes()));
        lin->bias.copy_(bi->second.reshape(lin->bias.sizes()));
    };
    load("entity_encoder.0", policy->ent0); load("entity_encoder.2", policy->ent2);
    load("context_encoder.0", policy->ctx0); load("context_encoder.2", policy->ctx2);
    load("ent_glu_gate", policy->ent_glu_gate); load("ent_glu_val", policy->ent_glu_val);
    load("ent_glu_out", policy->ent_glu_out);
    load("ctx_glu_gate", policy->ctx_glu_gate); load("ctx_glu_val", policy->ctx_glu_val);
    load("ctx_glu_out", policy->ctx_glu_out);
    load_opt("critic.0", policy->crit0); load_opt("critic.2", policy->crit2);
    if (policy->target_mode) {
        load("actor_q", policy->aq); load("actor_k", policy->ak); load("actor_noop", policy->anoop);
    } else {
        load("actor.0", policy->act0); load("actor.2", policy->act2);
    }
}

}  // namespace ow
