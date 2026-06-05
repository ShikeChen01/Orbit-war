// Agent implementation (v4, continuous): forward+sample (rollout/opponent), re-score (update),
// frozen clone (KL reference), and checkpoint I/O. The policy is a squashed diagonal Gaussian; an
// action is the (E x 2K) matrix of squashed controls. Checkpoint keys mirror the module names so
// the pure-Python serving twin can load them. No critic (GRPO).
#include "rl/model/agent.hpp"

#include <sstream>
#include <string>

#include "rl/model/distribution.hpp"

namespace ow {

Agent::Agent(const ModelConfig& cfg, torch::Device dev) : device(dev) {
    net = PolicyNet(cfg);
    net->to(device);
}

Agent::Decision Agent::act(const torch::Tensor& ent, const torch::Tensor& em,
                           const torch::Tensor& am, const torch::Tensor& gl, bool greedy) {
    torch::NoGradGuard ng;
    auto [mean, logstd] = net->forward(ent, em, am, gl);
    SquashedGaussian dist(mean, logstd);
    auto action = greedy ? dist.greedy() : dist.sample();
    return {action, dist.log_prob(action)};
}

Agent::Score Agent::evaluate(const torch::Tensor& ent, const torch::Tensor& em,
                             const torch::Tensor& am, const torch::Tensor& gl,
                             const torch::Tensor& action) {
    auto [mean, logstd] = net->forward(ent, em, am, gl);
    SquashedGaussian dist(mean, logstd);
    return {dist.log_prob(action), dist.entropy()};
}

Agent Agent::clone_frozen() const {
    Agent c;
    c.device = device;
    c.net = PolicyNet(net->cfg);
    std::stringstream ss;  // serialize -> deserialize is the proven deep-copy path
    torch::save(net, ss);
    torch::load(c.net, ss);
    c.net->to(device);
    c.net->eval();
    return c;
}

void Agent::load_state_dict(const std::map<std::string, torch::Tensor>& sd) {
    auto load = [&](const std::string& key, torch::nn::Linear& lin) {
        if (!lin) return;
        auto wi = sd.find(key + ".weight");
        auto bi = sd.find(key + ".bias");
        if (wi == sd.end() || bi == sd.end()) return;  // tolerate missing
        torch::NoGradGuard ng;
        lin->weight.copy_(wi->second.reshape(lin->weight.sizes()).to(device));
        lin->bias.copy_(bi->second.reshape(lin->bias.sizes()).to(device));
    };
    load("proj", net->proj);
    if (net->cfg.use_glu) {
        load("glu_gate", net->glu_gate);
        load("glu_val", net->glu_val);
        load("glu_out", net->glu_out);
    }
    for (size_t i = 0; i < net->res_a.size(); ++i) {
        load("res" + std::to_string(i) + "a", net->res_a[i]);
        load("res" + std::to_string(i) + "b", net->res_b[i]);
    }
    load("g_embed", net->g_embed);
    load("mu_head", net->mu_head);
    if (net->cfg.std_state_dependent) {
        load("logstd_head", net->logstd_head);
    } else {
        auto it = sd.find("logstd_param");
        if (it != sd.end() && net->logstd_param.defined()) {
            torch::NoGradGuard ng;
            net->logstd_param.copy_(it->second.reshape(net->logstd_param.sizes()).to(device));
        }
    }
}

std::map<std::string, torch::Tensor> Agent::state_dict() const {
    std::map<std::string, torch::Tensor> w;
    auto put = [&](const std::string& k, const torch::nn::Linear& lin) {
        if (!lin) return;
        w[k + ".weight"] = lin->weight.detach().to(torch::kCPU).contiguous();
        w[k + ".bias"] = lin->bias.detach().to(torch::kCPU).contiguous();
    };
    put("proj", net->proj);
    if (net->cfg.use_glu) {
        put("glu_gate", net->glu_gate);
        put("glu_val", net->glu_val);
        put("glu_out", net->glu_out);
    }
    for (size_t i = 0; i < net->res_a.size(); ++i) {
        put("res" + std::to_string(i) + "a", net->res_a[i]);
        put("res" + std::to_string(i) + "b", net->res_b[i]);
    }
    put("g_embed", net->g_embed);
    put("mu_head", net->mu_head);
    if (net->cfg.std_state_dependent) {
        put("logstd_head", net->logstd_head);
    } else if (net->logstd_param.defined()) {
        w["logstd_param"] = net->logstd_param.detach().to(torch::kCPU).contiguous();
    }
    return w;
}

}  // namespace ow
