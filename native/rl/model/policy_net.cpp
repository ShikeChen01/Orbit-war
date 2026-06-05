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
    attn_q = register_module("attn_q", torch::nn::Linear(h, h));
    attn_k = register_module("attn_k", torch::nn::Linear(h, h));
    attn_v = register_module("attn_v", torch::nn::Linear(h, h));
    attn_o = register_module("attn_o", torch::nn::Linear(h, h));
    ln_attn = register_module("ln_attn", torch::nn::LayerNorm(torch::nn::LayerNormOptions({h})));
    ln_out = register_module("ln_out", torch::nn::LayerNorm(torch::nn::LayerNormOptions({h})));
    if (c.use_glu) {
        glu_gate = register_module("glu_gate", torch::nn::Linear(h, h));
        glu_val = register_module("glu_val", torch::nn::Linear(h, h));
        glu_out = register_module("glu_out", torch::nn::Linear(h, h));
    }
    for (int i = 0; i < c.n_res_blocks; ++i) {
        res_a.push_back(register_module("res" + std::to_string(i) + "a", torch::nn::Linear(h, h)));
        res_b.push_back(register_module("res" + std::to_string(i) + "b", torch::nn::Linear(h, h)));
        ln_res.push_back(register_module("ln_res" + std::to_string(i),
                                         torch::nn::LayerNorm(torch::nn::LayerNormOptions({h}))));
    }
    g_embed = register_module("g_embed", torch::nn::Linear(G, dg));
    mu_h1 = register_module("mu_h1", torch::nn::Linear(h + dg, h));
    mu_h2 = register_module("mu_h2", torch::nn::Linear(h, twoK));
    {
        // Prior: start the policy nearly INERT so it learns to act, instead of spamming ~E*K illegal
        // dispatches and slowly suppressing them. Small final-layer weights make every head output
        // start ~= its bias regardless of the deep random trunk (without this, a deep scratch trunk
        // emits large, varied means -> ~half of all E*K slots commit illegally from step 1). A
        // strongly negative phi bias (sigmoid(-5)=0.0067 << tau_act) keeps almost nothing committed
        // at init; the policy then LEARNS to raise phi on owned planets. phi = odd (alpha,phi) comps.
        torch::NoGradGuard ng;
        mu_h2->weight.mul_(c.init_mu_scale);
        mu_h2->bias.zero_();
        for (int k = 0; k < c.fleets_per_planet; ++k) mu_h2->bias[2 * k + 1].fill_(c.init_phi_bias);
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
    (void)action_mask;
    auto tok = proj->forward(entities);  // (B,E,d)
    {
        // masked single-head self-attention over planets: each planet attends to all LIVE planets,
        // so it can read their positions and aim a launch at a specific target (impossible for the
        // per-planet trunk alone). entity_mask (B,E): 1 live / 0 padding -> dead keys are masked out.
        auto xn = ln_attn->forward(tok);
        auto Q = attn_q->forward(xn), Kt = attn_k->forward(xn), Vt = attn_v->forward(xn);
        auto scores = torch::matmul(Q, Kt.transpose(1, 2)) / std::sqrt((double)cfg.hidden);  // (B,E,E)
        auto dead_key = (entity_mask < 0.5).unsqueeze(1);  // (B,1,E)
        scores = scores.masked_fill(dead_key, -1e9);
        auto attn = torch::matmul(torch::softmax(scores, -1), Vt);  // (B,E,d)
        tok = tok + attn_o->forward(attn);
    }
    if (cfg.use_glu) {
        tok = tok +
              glu_out->forward(glu_val->forward(tok) * torch::sigmoid(glu_gate->forward(tok)));
    }
    for (size_t i = 0; i < res_a.size(); ++i)  // n_res_blocks pre-norm residual MLP blocks
        tok = tok + res_b[i]->forward(torch::relu(res_a[i]->forward(ln_res[i]->forward(tok))));
    tok = ln_out->forward(tok);  // final norm so the head means stay in sigmoid's gradient range

    long B = tok.size(0), E = tok.size(1);
    auto gp = torch::relu(g_embed->forward(globals));            // (B,d_g)
    auto gpb = gp.unsqueeze(1).expand({B, E, cfg.d_g});          // broadcast to every planet
    auto h = torch::cat({tok, gpb}, -1);                         // (B,E,d+d_g)

    auto mean = mu_h2->forward(torch::relu(mu_h1->forward(h)));  // (B,E,2K) nonlinear mean head
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
