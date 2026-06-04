// Grouped episodic rollout for GRPO. Lays out B = num_groups * group_size envs; every env in
// a group plays the SAME world against the SAME opponent, so the group return mean/std is a
// valid baseline (the only within-group variation is the policy's own sampling). Collects full
// trajectories to terminal, computes per-episode returns, group-normalizes -> per-transition
// advantage, and returns a flat TrajectoryBatch. (v1 opponents: random/starter. Self-play
// policy opponents are a follow-up.)
#pragma once
#include <random>
#include <vector>

#include <torch/torch.h>

#include "core/state.hpp"
#include "rl/config.hpp"
#include "rl/model/agent.hpp"
#include "rl/trajectory.hpp"

namespace ow {

struct RolloutStats {
    double mean_return = 0.0, mean_len = 0.0, win_rate = 0.0;
    long episodes = 0, transitions = 0;
};

class GroupedRollout {
public:
    GroupedRollout(const TrainConfig& cfg, torch::Device device);
    void set_world_pool(std::vector<GameState> pool) {
        world_pool_ = std::move(pool);
        cursor_ = 0;
    }
    // One GRPO iteration: sample the policy, collect grouped full episodes, group-normalize the
    // returns. (ref_logp for the KL anchor is computed in the update, not here -- it depends only
    // on the fixed taken action, so deferring it avoids a per-step reference forward.)
    TrajectoryBatch collect(Agent& policy, RolloutStats& stats);

private:
    TrainConfig cfg_;
    torch::Device device_;
    std::vector<GameState> world_pool_;
    size_t cursor_ = 0;
    std::mt19937_64 rng_;
};

}  // namespace ow
