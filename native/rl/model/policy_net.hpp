// The agent's network: a per-planet trunk + a separate board-globals embedding feeding two
// continuous action heads (mean and log-std of a squashed diagonal Gaussian over E x 2K controls).
// An OPTIONAL value head (critic) is built only for PPO+GAE (cfg.use_value_head); GRPO omits it.
// Architecture (docs/rl_math.pdf sec:policy):
//   trunk:   tok = proj(x);  tok += GLU(tok) [if use_glu];  tok += resblock_i(tok) x n_res
//   globals: g' = relu(g_embed(globals))            (a SEPARATE path, broadcast to every planet)
//   heads:   mu = mu_head([tok; g']);  logstd = logstd_head([tok; g'])  OR  shared vector
// Two A/B switches live in ModelConfig: `use_glu` (trunk) and `std_state_dependent` (log-std).
//
// Interface only -- the forward pass and module registration live in policy_net.cpp.
#pragma once
#include <torch/torch.h>

#include <utility>
#include <vector>

#include "rl/config.hpp"

namespace ow {

struct PolicyNetImpl : torch::nn::Module {
    explicit PolicyNetImpl(const ModelConfig& cfg);

    // forward output: the two Gaussian head tensors plus (when use_value_head) the critic's V(s).
    struct Out {
        torch::Tensor mean;    // continuous actor: (B,E,3K) Gaussian mean; target actor: (B,E,1) phi mean
        torch::Tensor logstd;  // matching shape, clipped to [logstd_min, logstd_max]
        torch::Tensor dest_logits;  // TARGET actor only: (B,E,E) destination logits (else undefined)
        torch::Tensor value;   // (B,) critic value, or undefined when use_value_head==false
    };
    // entities (B,E,F), entity_mask (B,E) [gates attention keys + value pool], action_mask (B,E)
    //   [unused], globals (B,G) -> Out. The value head pools the post-trunk tokens (masked mean over
    //   live planets) with the globals embedding -> a 2-layer MLP -> scalar V(s).
    Out forward(const torch::Tensor& entities, const torch::Tensor& entity_mask,
                const torch::Tensor& action_mask, const torch::Tensor& globals);

    ModelConfig cfg;
    int nap;  // n_action_params = 3*fleets_per_planet (dx,dy,phi per fleet), cached from cfg

    // per-planet trunk
    torch::nn::Linear proj{nullptr};                                          // F -> d
    // one self-attention block over the E planet tokens: lets each planet SEE the others' features
    // (positions) so it can aim a launch AT a specific target planet -- the per-planet+global trunk
    // alone cannot represent targeted headings (cross-planet info is absent). Masked by entity_mask.
    torch::nn::Linear attn_q{nullptr}, attn_k{nullptr}, attn_v{nullptr}, attn_o{nullptr};
    torch::nn::Linear glu_gate{nullptr}, glu_val{nullptr}, glu_out{nullptr};  // one GLU block (opt)
    std::vector<torch::nn::Linear> res_a, res_b;                              // n_res_blocks res MLPs
    // pre-LayerNorm keeps the deep trunk's activations bounded so the head means do not saturate
    // sigmoid (which was zeroing the aiming gradient -> headings never learned). Standard pre-norm.
    torch::nn::LayerNorm ln_attn{nullptr}, ln_out{nullptr};
    std::vector<torch::nn::LayerNorm> ln_res;
    // separate board-globals path + heads (operate on [tok; g'], dim d + d_g)
    torch::nn::Linear g_embed{nullptr};                                       // G -> d_g
    torch::nn::Linear mu_h1{nullptr}, mu_h2{nullptr};  // MLP mean head: nonlinear so it can turn the
                                                       // relative positions (from attention) into an
                                                       // angle (atan2) -- a linear head cannot.
    torch::nn::Linear logstd_head{nullptr};                                   // d+d_g -> 2K (state-dep)
    torch::Tensor logstd_param;                                               // 2K vector (state-indep)
    // --- v5 TARGET actor heads (built only when cfg.target_actor) ---
    torch::nn::Linear dest_head{nullptr};       // d+d_g -> E   (destination-planet logits per planet)
    torch::nn::Linear phi_mu{nullptr};          // d+d_g -> 1   (phi launch-fraction mean)
    torch::nn::Linear phi_logstd{nullptr};      // d+d_g -> 1   (phi log-std; state-dependent variant)
    torch::Tensor phi_logstd_param;             // 1-vector phi log-std (state-independent variant)
    // optional critic (PPO): masked-mean pool of the trunk tokens + g' -> MLP -> scalar V(s)
    torch::nn::Linear val_h1{nullptr}, val_h2{nullptr};                       // d+d_g -> d -> 1
};
TORCH_MODULE(PolicyNet);

}  // namespace ow
