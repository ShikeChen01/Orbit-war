// Grouped episodic rollout for GRPO. Lays out B = num_groups * group_size envs; every env in
// a group plays the SAME world against the SAME opponent, so the group return mean/std is a
// valid baseline (the only within-group variation is the policy's own sampling). Collects full
// trajectories to terminal, computes per-episode returns, group-normalizes -> per-transition
// advantage, and returns a flat TrajectoryBatch. (v1 opponents: random/starter. Self-play
// policy opponents are a follow-up.)
#pragma once
#include <memory>
#include <random>
#include <vector>

#include <torch/torch.h>

#include "core/state.hpp"
#include "rl/config.hpp"
#include "rl/gpu_env.hpp"
#include "rl/model/agent.hpp"
#include "rl/trajectory.hpp"

namespace ow {

struct RolloutStats {
    double mean_return = 0.0, mean_len = 0.0, win_rate = 0.0;
    double mean_invalid = 0.0;   // committed-but-failed fleet dispatches per env-step (the penalty x_t)
    double mean_launches = 0.0;  // executed launches per env-step
    double mean_valid = 0.0;     // executed launches that will LAND on a planet per env-step
    // mean per-EPISODE reward by docs/set-ups/1.md channel (REAL units, before any ppo_reward_scale):
    // outcome (decayed win / loss / draw), capture (+/-), first-N dispatch, fleet-hit (capped),
    // production-milestone (per ship-count doubling). r_hit is legacy (kept for CSV compat, usually 0).
    double r_outcome = 0.0, r_capture = 0.0, r_dispatch = 0.0, r_hit = 0.0, r_prod_milestone = 0.0;
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

    // GPU-batched variant: the entire encode->act->step loop runs on-device via GpuEnv; the
    // trajectory is offloaded to host. Same TrajectoryBatch / reward scheme as collect().
    TrajectoryBatch collect_gpu(Agent& policy, RolloutStats& stats);

    // Curriculum: the trainer sets the active stage each iteration (1 noop / 2 random / 3 starter /
    // 4 self-play+starter+random when stage3_iters>0; legacy 1/2/3 otherwise). 0 = static cfg.stage.
    void set_stage(int s) { cur_stage_ = s; }

    // Stage-4 self-play opponent: a frozen policy snapshot the trainer refreshes. nullptr -> stage 4
    // falls back to starter-only. Borrowed pointer; the trainer owns the snapshot's lifetime.
    void set_opponent(Agent* a) { opponent_ = a; }

private:
    int cur_stage_ = 0;
    Agent* opponent_ = nullptr;
    TrainConfig cfg_;
    torch::Device device_;
    std::vector<GameState> world_pool_;
    size_t cursor_ = 0;
    std::mt19937_64 rng_;
    std::unique_ptr<GpuEnv> gpu_;  // lazily built on first collect_gpu (sized to B, planet_cap)

    // Persistent PINNED host buffers for the GPU rollout's per-step D2H offload. Allocated once
    // (lazily, shape-guarded) and REUSED every iteration -- the old code reallocated ~1GB of pinned
    // memory per iteration, which churns the page-locked pool. Only the [0,Tu) prefix is read each
    // iteration and every kept step is fully overwritten, so no clearing is needed between iters.
    torch::Tensor ent_buf_, em_buf_, am_buf_, gl_buf_, act_buf_, oldlogp_buf_, valid_buf_,
        rew_buf_, val_buf_, done_buf_;
    int buf_T_ = 0, buf_B_ = 0, buf_Ec_ = 0, buf_F_ = 0, buf_Gl_ = 0, buf_nap_ = 0;
};

}  // namespace ow
