// Agent implementation: forward+sample (rollout/opponent), re-score (update), frozen clone
// (opponent pool / KL reference), and checkpoint I/O. Weight keys mirror the Python
// EntityPolicy; "critic.*" tensors in a BC checkpoint are ignored (GRPO has no critic).
#include "rl/model/agent.hpp"

#include <sstream>

#include "rl/model/distribution.hpp"

namespace ow {

Agent::Agent(const ModelConfig& cfg, torch::Device dev) : device(dev) {
    net = PolicyNet(cfg);
    net->to(device);
}

Agent::Decision Agent::act(const torch::Tensor& ent, const torch::Tensor& em,
                           const torch::Tensor& am, const torch::Tensor& gl, bool greedy) {
    torch::NoGradGuard ng;
    auto logits = net->forward(ent, em, am, gl);
    MaskedPerEntityCategorical dist(logits, em);
    auto action = greedy ? dist.greedy() : dist.sample();
    return {action, dist.log_prob(action)};
}

Agent::Score Agent::evaluate(const torch::Tensor& ent, const torch::Tensor& em,
                             const torch::Tensor& am, const torch::Tensor& gl,
                             const torch::Tensor& action) {
    auto logits = net->forward(ent, em, am, gl);
    MaskedPerEntityCategorical dist(logits, em);
    return {dist.log_prob(action), dist.entropy(), logits};
}

Agent Agent::clone_frozen() const {
    Agent c;
    c.device = device;
    c.net = PolicyNet(net->cfg);
    std::stringstream ss;          // serialize -> deserialize is the proven deep-copy path
    torch::save(net, ss);
    torch::load(c.net, ss);
    c.net->to(device);
    c.net->eval();
    return c;
}

void Agent::load_state_dict(const std::map<std::string, torch::Tensor>& sd) {
    auto load = [&](const std::string& key, torch::nn::Linear& lin) {
        auto wi = sd.find(key + ".weight");
        auto bi = sd.find(key + ".bias");
        if (wi == sd.end() || bi == sd.end()) return;  // tolerate missing (e.g. critic.*)
        torch::NoGradGuard ng;
        lin->weight.copy_(wi->second.reshape(lin->weight.sizes()).to(device));
        lin->bias.copy_(bi->second.reshape(lin->bias.sizes()).to(device));
    };
    load("entity_encoder.0", net->ent0);
    load("entity_encoder.2", net->ent2);
    load("context_encoder.0", net->ctx0);
    load("context_encoder.2", net->ctx2);
    load("ent_glu_gate", net->ent_glu_gate);
    load("ent_glu_val", net->ent_glu_val);
    load("ent_glu_out", net->ent_glu_out);
    load("ctx_glu_gate", net->ctx_glu_gate);
    load("ctx_glu_val", net->ctx_glu_val);
    load("ctx_glu_out", net->ctx_glu_out);
    if (net->cfg.target_mode) {
        load("actor_q", net->aq);
        load("actor_k", net->ak);
        load("actor_noop", net->anoop);
    } else {
        load("actor.0", net->act0);
        load("actor.2", net->act2);
    }
}

std::map<std::string, torch::Tensor> Agent::state_dict() const {
    std::map<std::string, torch::Tensor> w;
    auto put = [&](const std::string& k, const torch::nn::Linear& lin) {
        w[k + ".weight"] = lin->weight.detach().to(torch::kCPU).contiguous();
        w[k + ".bias"] = lin->bias.detach().to(torch::kCPU).contiguous();
    };
    put("entity_encoder.0", net->ent0);
    put("entity_encoder.2", net->ent2);
    put("context_encoder.0", net->ctx0);
    put("context_encoder.2", net->ctx2);
    put("ent_glu_gate", net->ent_glu_gate);
    put("ent_glu_val", net->ent_glu_val);
    put("ent_glu_out", net->ent_glu_out);
    put("ctx_glu_gate", net->ctx_glu_gate);
    put("ctx_glu_val", net->ctx_glu_val);
    put("ctx_glu_out", net->ctx_glu_out);
    if (net->cfg.target_mode) {
        put("actor_q", net->aq);
        put("actor_k", net->ak);
        put("actor_noop", net->anoop);
    } else {
        put("actor.0", net->act0);
        put("actor.2", net->act2);
    }
    return w;
}

}  // namespace ow
