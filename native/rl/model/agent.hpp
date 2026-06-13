// The Agent: the single thing the rollout, the GRPO update, the arena, and the self-play
// league talk to. It wraps a PolicyNet and owns the policy-side operations -- turning an
// observation into an action (rollout / opponent play) and re-scoring a stored action
// under the current weights (the update). A frozen clone is the unit stored in the
// opponent pool. Checkpoint I/O uses the Python EntityPolicy state_dict key names so BC
// warm-starts and the .owc/.pt tooling interoperate; "critic.*" keys are tolerated on load
// and never emitted (GRPO has no critic).
//
// Interface only -- implementation in agent.cpp.
#pragma once
#include <map>
#include <string>

#include <torch/torch.h>

#include "rl/config.hpp"
#include "rl/model/policy_net.hpp"

namespace ow {

class Agent {
public:
    Agent() = default;
    Agent(const ModelConfig& cfg, torch::Device device);

    // --- rollout / opponent play: forward + sample (or greedy), no grad ---
    struct Decision {
        torch::Tensor action;    // (B,E,2K) float squashed-Gaussian action in (0,1)
        torch::Tensor log_prob;  // (B,) behavior log-prob of the joint action
        torch::Tensor value;     // (B,) critic V(s) (PPO); undefined when no value head
    };
    Decision act(const torch::Tensor& entities, const torch::Tensor& entity_mask,
                 const torch::Tensor& action_mask, const torch::Tensor& globals,
                 bool greedy = false);

    // --- the update: re-score a stored action under the current (grad-enabled) policy ---
    struct Score {
        torch::Tensor log_prob;  // (B,) current log-prob of the stored action
        torch::Tensor entropy;   // (B,) policy entropy (for the bonus)
        torch::Tensor value;     // (B,) critic V(s) WITH grad (PPO value loss); undefined for GRPO
    };
    Score evaluate(const torch::Tensor& entities, const torch::Tensor& entity_mask,
                   const torch::Tensor& action_mask, const torch::Tensor& globals,
                   const torch::Tensor& action);

    // Deep copy in eval mode -- the snapshot stored in the self-play pool / used as the
    // fixed KL reference.
    Agent clone_frozen() const;

    // Checkpoint interop (Python EntityPolicy state_dict keys; critic optional/ignored).
    void load_state_dict(const std::map<std::string, torch::Tensor>& sd);
    std::map<std::string, torch::Tensor> state_dict() const;

    PolicyNet net{nullptr};
    torch::Device device = torch::kCPU;
};

}  // namespace ow
