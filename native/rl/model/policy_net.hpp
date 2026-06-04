// The agent's network: entity encoder + context encoder + actor head -> per-entity action
// logits. There is **no value head** -- GRPO needs no critic. Parameter names mirror the
// Python EntityPolicy (entity_encoder.* / context_encoder.* / actor_*) so behavior-cloning
// checkpoints load 1:1; the (unused) "critic.*" tensors in a BC checkpoint are simply
// ignored on load.
//
// Interface only -- the forward pass and module registration live in policy_net.cpp.
#pragma once
#include <torch/torch.h>

#include "rl/config.hpp"

namespace ow {

struct PolicyNetImpl : torch::nn::Module {
    explicit PolicyNetImpl(const ModelConfig& cfg);

    // entities (B,E,F), entity_mask (B,E), action_mask (B,E), globals (B,G)
    //   -> logits (B,E,A), with illegal classes set to NEG_INF.
    // Angle mode: classes are (angle_bin, fraction). Target mode: a pointer actor scores
    // (launcher r, target t, fraction f) = <q_f(tok_r, core), k(tok_t)> + a noop head.
    torch::Tensor forward(const torch::Tensor& entities,
                          const torch::Tensor& entity_mask,
                          const torch::Tensor& action_mask,
                          const torch::Tensor& globals);

    ModelConfig cfg;
    int A;  // actions_per_entity, cached from cfg

    torch::nn::Linear ent0{nullptr}, ent2{nullptr};   // entity encoder (2-layer MLP)
    torch::nn::Linear ctx0{nullptr}, ctx2{nullptr};   // context (pooled + globals) encoder (2-layer MLP)
    // Residual GLU blocks (one per encoder): out = x + W_o( (W_v x) * sigmoid(W_g x) ).
    torch::nn::Linear ent_glu_gate{nullptr}, ent_glu_val{nullptr}, ent_glu_out{nullptr};
    torch::nn::Linear ctx_glu_gate{nullptr}, ctx_glu_val{nullptr}, ctx_glu_out{nullptr};
    torch::nn::Linear act0{nullptr}, act2{nullptr};   // angle-mode actor MLP
    torch::nn::Linear aq{nullptr}, ak{nullptr}, anoop{nullptr};  // target-mode pointer actor
};
TORCH_MODULE(PolicyNet);

}  // namespace ow
