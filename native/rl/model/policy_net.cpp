// PolicyNet (v4, continuous): a per-planet trunk + a separate board-globals embedding feeding two
// Gaussian heads. Layout (docs/rl_math.pdf sec:policy):
//   tok = proj(x);  tok += GLU(tok) [if use_glu];  tok += resblock_i(tok) x n_res
//   g'  = relu(g_embed(globals));   h = [tok ; g'_broadcast]   (dim d + d_g)
//   mean = mu_head(h);   logstd = logstd_head(h)  OR  shared learnable vector  -> both (B,E,2K)
// No mask is applied (the actor learns the legal action space from the invalid-dispatch penalty);
// no value head (GRPO has no critic).
#include "rl/model/policy_net.hpp"

#include <cmath>
#include <string>

namespace ow {

PolicyNetImpl::PolicyNetImpl(const ModelConfig& c) : cfg(c), twoK(2 * c.fleets_per_planet) {
    int F = c.n_entity_features, h = c.hidden, G = c.n_global_features, dg = c.d_g;
    proj = register_module("proj", torch::nn::Linear(F, h));
    if (c.use_glu) {
        glu_gate = register_module("glu_gate", torch::nn::Linear(h, h));
        glu_val = register_module("glu_val", torch::nn::Linear(h, h));
        glu_out = register_module("glu_out", torch::nn::Linear(h, h));
    }
    for (int i = 0; i < c.n_res_blocks; ++i) {
        res_a.push_back(register_module("res" + std::to_string(i) + "a", torch::nn::Linear(h, h)));
        res_b.push_back(register_module("res" + std::to_string(i) + "b", torch::nn::Linear(h, h)));
    }
    g_embed = register_module("g_embed", torch::nn::Linear(G, dg));
    mu_head = register_module("mu_head", torch::nn::Linear(h + dg, twoK));
    {
        // Prior: start the policy nearly INERT so it learns to act, instead of spamming ~E*K illegal
        // dispatches and slowly suppressing them. Small final-layer weights make every head output
        // start ~= its bias regardless of the deep random trunk (without this, a deep scratch trunk
        // emits large, varied means -> ~half of all E*K slots commit illegally from step 1). A
        // strongly negative phi bias (sigmoid(-5)=0.0067 << tau_act) keeps almost nothing committed
        // at init; the policy then LEARNS to raise phi on owned planets. phi = odd (alpha,phi) comps.
        torch::NoGradGuard ng;
        mu_head->weight.mul_(c.init_mu_scale);
        mu_head->bias.zero_();
        for (int k = 0; k < c.fleets_per_planet; ++k) mu_head->bias[2 * k + 1].fill_(c.init_phi_bias);
    }
    if (c.std_state_dependent) {
        logstd_head = register_module("logstd_head", torch::nn::Linear(h + dg, twoK));
    } else {
        // one shared learnable log-std vector (state-independent exploration); init sigma ~ 1.
        logstd_param = register_parameter("logstd_param", torch::zeros({twoK}));
    }
}

std::pair<torch::Tensor, torch::Tensor> PolicyNetImpl::forward(const torch::Tensor& entities,
                                                              const torch::Tensor& entity_mask,
                                                              const torch::Tensor& action_mask,
                                                              const torch::Tensor& globals) {
    (void)entity_mask;   // not masked: log-prob/entropy sum over ALL slots (penalty teaches legality)
    (void)action_mask;
    auto tok = proj->forward(entities);  // (B,E,d)
    if (cfg.use_glu) {
        tok = tok +
              glu_out->forward(glu_val->forward(tok) * torch::sigmoid(glu_gate->forward(tok)));
    }
    for (size_t i = 0; i < res_a.size(); ++i)                  // n_res_blocks residual MLP blocks
        tok = tok + res_b[i]->forward(torch::relu(res_a[i]->forward(tok)));

    long B = tok.size(0), E = tok.size(1);
    auto gp = torch::relu(g_embed->forward(globals));            // (B,d_g)
    auto gpb = gp.unsqueeze(1).expand({B, E, cfg.d_g});          // broadcast to every planet
    auto h = torch::cat({tok, gpb}, -1);                         // (B,E,d+d_g)

    auto mean = mu_head->forward(h);                             // (B,E,2K)
    torch::Tensor logstd;
    if (cfg.std_state_dependent) {
        logstd = logstd_head->forward(h);                       // (B,E,2K)
    } else {
        logstd = logstd_param.view({1, 1, twoK}).expand({B, E, twoK});
    }
    logstd = logstd.clamp(cfg.logstd_min, cfg.logstd_max);
    return {mean, logstd};
}

}  // namespace ow
