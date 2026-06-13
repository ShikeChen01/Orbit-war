// GRPO trainer: orchestration only. Owns the policy, the frozen BC reference (KL anchor),
// the optimizer, the grouped rollout, and the arena eval. The loop is
//   collect -> group advantage -> clipped surrogate + beta*KL - ent*H update -> (every 200k
//   env-steps) arena eval + log line + metrics.csv row + best/last checkpoint.
// All artifacts go to a per-run results folder.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "core/state.hpp"
#include "io/serialize.hpp"
#include "rl/arena_cont.hpp"
#include "rl/config.hpp"
#include "rl/model/agent.hpp"
#include "rl/rollout.hpp"

namespace ow {

class GrpoTrainer {
public:
    GrpoTrainer(const TrainConfig& cfg, std::string run_dir);
    void set_world_pool(std::vector<GameState> train, std::vector<GameState> eval);
    void load_init(const std::map<std::string, torch::Tensor>& sd);  // BC -> policy AND reference
    void init_scratch();  // from-scratch: random policy, optimizer, no KL anchor (kl_beta=0)
    void train();

private:
    struct UpdateStats {
        double total = 0, policy = 0, kl = 0, entropy = 0, sigma = 0, adv_mean = 0, adv_std = 0,
               approx_kl = 0, clipfrac = 0, vf = 0;  // vf = value (critic) MSE loss, PPO only
    };
    UpdateStats update(const TrajectoryBatch& tb);
    // Full eval (arena win-rates) + metrics row + checkpoint + printed line. Runs at the checkpoint
    // cadence. ckpt_tag (when non-empty) also writes run_dir/ckpt/<tag>.owc (the progression).
    void log_and_eval(double sps, const RolloutStats& rs, const UpdateStats& us,
                      const std::string& ckpt_tag = "");
    // Cheap per-iteration heartbeat: a metrics row (win-rates carried from the last eval) + a compact
    // printed line. NO arena eval, NO checkpoint -- so it can run every iteration.
    void heartbeat(double sps, const RolloutStats& rs, const UpdateStats& us);
    void write_metrics_row(double sps, const RolloutStats& rs, const UpdateStats& us,
                           double wr_rand, double wr_start, double margin, double eplen);
    CheckpointMeta meta() const;

    TrainConfig cfg_;
    torch::Device device_;
    std::string run_dir_;
    Agent policy_, reference_;
    Agent opp_snapshot_;  // stage-4 self-play opponent: a frozen, periodically-refreshed snapshot
    bool opp_set_ = false;  // whether opp_snapshot_ has been built + handed to the rollout
    std::shared_ptr<torch::optim::Adam> opt_;
    GroupedRollout rollout_;
    ArenaCont arena_cont_;
    std::vector<GameState> eval_worlds_;
    long global_step_ = 0;
    int iter_ = 0;
    int cur_stage_ = 1;  // active curriculum stage (for logging)
    double best_wr_starter_ = -1.0;
    bool csv_open_ = false;
    // last arena-eval results, carried forward into the per-iter heartbeat metrics rows
    double last_wr_rand_ = 0, last_wr_start_ = 0, last_margin_ = 0, last_eplen_ = 0;
};

}  // namespace ow
