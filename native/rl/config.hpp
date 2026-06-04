// Configuration for the native RL stack, split by concern so the agent model, the GRPO
// algorithm, the rollout/env, and the self-play league each own their knobs (no single
// god-struct). `TrainConfig` aggregates them for a run. Pure data -- no logic beyond a
// couple of derived sizes. Note there is **no value/critic config** (no vf_coef, no
// gae_lambda, no value_warmup): GRPO has no value network; its baseline is the group.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ow {

// --- the agent model (network shape) ---------------------------------------
struct ModelConfig {
    int n_entity_features = 15;   // overwritten from encode.hpp at build time
    int n_global_features = 10;
    int hidden = 128;
    int max_entities = 64;
    int angle_bins = 16;          // angle-mode action choices (ignored when target_mode)
    bool target_mode = true;      // pointer actor: per-planet action = (target entity, fraction)
    std::vector<double> fractions = {0.25, 0.5, 0.75, 1.0};

    int n_choices() const { return target_mode ? max_entities : angle_bins; }
    int actions_per_entity() const { return 1 + n_choices() * (int)fractions.size(); }
};

// --- optimizer -------------------------------------------------------------
struct OptimConfig {
    double lr = 3e-4;
    double max_grad_norm = 0.5;
    double adam_eps = 1e-5;
};

// --- GRPO algorithm --------------------------------------------------------
// Group Relative Policy Optimization: sample a *group* of trajectories from one start,
// and use the group's own return statistics as the baseline (replacing the critic).
struct GrpoConfig {
    int group_size = 8;     // trajectories per group sharing one (world, opponent) -> the baseline set
    int num_groups = 64;    // groups per iteration; env count = group_size * num_groups
    double clip = 0.2;      // PPO-style ratio clip on the surrogate
    double kl_beta = 0.04;  // weight of the KL-to-reference penalty (0 disables the anchor)
    double ent_coef = 0.01; // entropy bonus
    double adv_eps = 1e-4;  // group-std floor when normalizing the advantage
    bool whiten_advantage = true;  // divide by group std (false = mean-subtract only)
    int update_epochs = 2;  // passes over the collected trajectories
    int minibatches = 8;
    // Group-relative return = dense_weight * (discounted shaped reward) + outcome_weight * (+/-1 win).
    // dense_weight=0 -> pure-outcome GRPO; outcome_weight=0 -> shaped-only.
    double dense_weight = 1.0;
    double outcome_weight = 1.0;
    double gamma = 1.0;     // return discount (1.0 = undiscounted Monte-Carlo return; GRPO-classic)
};

// --- rollout / environment shaping reward ----------------------------------
struct RolloutConfig {
    int episode_steps = 500;
    double reward_scale = 50.0;   // divides the raw production-margin delta
    double reward_clip = 5.0;     // clamp on the per-step shaped reward
    double prod_weight = 20.0;    // production-margin weight in the potential
    double terminal_bonus = 1.0;  // magnitude of the +/- terminal win signal
};

// --- self-play league ------------------------------------------------------
// A real league (fixes the single-snapshot / blind-schedule defects of the old trainer):
// a bounded pool of historical snapshots, win-GATED promotion, and a tunable opponent mix.
struct SelfPlayConfig {
    long start_step = 2'000'000;      // begin adding the policy to the opponent mix
    long snapshot_every = 500'000;    // cadence of promotion *attempts*
    double promote_winrate = 0.55;    // gate: snapshot only if current beats the pool by this
    int pool_capacity = 10;           // historical snapshots retained (sampled per episode)
    double mix_random = 1.0, mix_starter = 1.0, mix_pool = 2.0;  // opponent mix once live
};

// --- top-level training run ------------------------------------------------
struct TrainConfig {
    ModelConfig model;
    OptimConfig optim;
    GrpoConfig grpo;
    RolloutConfig rollout;
    SelfPlayConfig selfplay;
    long total_steps = 8'000'000;
    long eval_every = 400'000;
    std::string device = "cuda";
    uint64_t seed = 0;

    int num_envs() const { return grpo.group_size * grpo.num_groups; }
};

}  // namespace ow
