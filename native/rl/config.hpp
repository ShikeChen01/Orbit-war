// Configuration for the native RL stack, split by concern so the agent model, the GRPO
// algorithm, the rollout/env, and the self-play league each own their knobs (no single
// god-struct). `TrainConfig` aggregates them for a run. Pure data -- no logic beyond a
// couple of derived sizes. GRPO needs no critic (its baseline is the group); the optional
// PPO+GAE path adds one (PpoConfig + ModelConfig::use_value_head), selected by TrainConfig::algo.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ow {

// --- the agent model (network shape) ---------------------------------------
struct ModelConfig {
    int n_entity_features = 15;   // overwritten from encode.hpp at build time
    int n_global_features = 10;
    int hidden = 128;             // d: trunk width
    int max_entities = 40;        // E: planet slots (docs/rl_math.pdf)

    // --- v4 CONTINUOUS actor (squashed diagonal Gaussian over E x 3K controls) ----------------
    // Each planet emits K fleet TRIPLES (dx, dy, phi): (dx,dy) is a DIRECTION VECTOR and phi the
    // launch fraction, so 3K params/planet. Decode aims via angle = atan2(2*dy-1, 2*dx-1) -- aiming
    // is then LINEAR in relative position (attention-computable) instead of the nonlinear scalar
    // atan2(target-self) a linear head cannot represent. The trunk is per-planet (proj -> [GLU] ->
    // res MLP); the G board globals enter through a SEPARATE embedding g'=relu(W_g g) broadcast to
    // the mean/log-std heads (NOT merged per row).
    int fleets_per_planet = 5;    // K: max fleets launched per planet per step -> 3K action params
    int d_g = 32;                 // board-globals embedding dim (separate broadcast path)
    bool use_glu = true;          // trunk A/B: GLU block on (true) vs plain stacked-MLP (false)
    int n_res_blocks = 2;         // residual MLP blocks after the (optional) GLU
    bool std_state_dependent = true;  // exploration A/B: log-std from a head (true) vs a shared
                                      // learnable vector (false, "stable" global exploration)
    bool use_value_head = false;      // PPO: add a critic head (masked-mean trunk pool + g' -> V(s)).
                                      // false = GRPO (no critic; the group is the baseline).
    bool action_mask_policy = false;  // MASK the actor to legal (owned, ships>0) planet slots: force
                                      // masked slots to no-op and drop them from the log-prob/entropy
                                      // sums, so the gradient concentrates on the ~owned*K controls
                                      // that matter and invalid dispatch -> 0 structurally. false =
                                      // the no-mask design (legality learned from the I penalty).
    double act_threshold = 0.05;  // tau_act: a fleet is "committed" when phi >= this
    double logstd_min = -2.0, logstd_max = 1.0;  // [varsigma_min, varsigma_max] clip on log-std
                                                 // (tight range keeps deep-net log-prob stable)
    // --- exploration anneal: force sigma DOWN, then hand it to the learnable head ----------------
    // The entropy bonus pins sigma at the cap forever -> noise-dominated action -> policy gradient
    // cancels (frozen). Two-phase fix (sigma_decay_iters>0): PHASE 1 (iters [0,sigma_decay_iters))
    // force the cap down from logstd_max -> logstd_max_end (e.g. sigma 1.0 -> 0.05); PHASE 2 release
    // the cap to logstd_max_post and let the state-dependent head choose sigma in [exp(logstd_min),
    // exp(logstd_max_post)] (e.g. [0.01, 0.2]) -- re-widen where uncertain, tighten where confident.
    // If sigma_decay_iters==0 and logstd_max_end!=logstd_max, fall back to a linear decay over the
    // whole run (by training-progress fraction).
    double logstd_max_end = 1.0;  // phase-1 forced-decay target for the cap (also the linear-anneal end)
    double logstd_max_post = 1.0; // phase-2 cap once the head takes over (head ranges to exp(this))
    int sigma_decay_iters = 0;    // >0: phase-1 length in ITERATIONS (fast forced decay); 0: off/linear
    // Calm init: scale the final (mu) layer so outputs start ~= bias regardless of the deep trunk,
    // and bias phi negative so few slots commit at init. LESS calm (smaller |bias|, larger scale)
    // = more initial commits = more mistakes to learn from; TOO calm = inert, learns nothing.
    double init_mu_scale = 0.02;   // mu_head weight multiplier at init
    double init_phi_bias = -4.0;   // mu_head phi-bias at init (sigmoid(-4)=0.018; w/ sigma=1 ~14% commit)

    // --- v5 TARGET actor: replace the unlearnable continuous aiming (dx,dy) with a DESTINATION ----
    // categorical. Per planet the model picks (a) a destination planet (softmax over the E obs slots,
    // masked to live planets) and (b) phi -- the launch fraction of THAT planet's ships (continuous,
    // squashed-Gaussian, identical to the old phi). A heuristic turns the destination into a heading
    // (atan2 toward it -- the "shortest path"), so the model never learns aiming. ONE launch per
    // planet per step. Stored action is (dest_row, phi) per planet -> 2 params. Selected by
    // --target-actor; false keeps the (dx,dy,phi) continuous actor (and parity/BC stay intact).
    bool target_actor = false;
    int n_action_params() const {
        return target_actor ? 2 : 3 * fleets_per_planet;  // target: (dest,phi); else 3K (dx,dy,phi)
    }
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

// --- PPO + GAE (actor-critic) ----------------------------------------------
// Optional alternative to GRPO. A value head (the critic) baselines each step and GAE assigns
// credit temporally, so the env's ACTION-LOCAL aiming reward (valid-landing carrot + miss stick,
// computed at the launch step) actually reaches the aiming gradient -- the per-step signal GRPO
// destroys by collapsing each trajectory to one capped episode return (docs/bc_vs_grpo.pdf).
// Enabled via TrainConfig::algo=="ppo" (which also sets ModelConfig::use_value_head). PPO reuses
// GrpoConfig's clip/ent_coef/update_epochs/minibatches/kl_beta.
struct PpoConfig {
    double gae_lambda = 0.95;  // GAE(lambda): bias/variance trade for the per-step advantage
    double gamma = 0.99;       // PPO return discount (GAE applies it; reward must NOT pre-discount)
    double vf_coef = 0.5;      // weight of the value (MSE-to-returns) loss in the total loss
    bool value_clip = false;   // PPO value-clipping (off by default; flag reserved)
};

// --- rollout / environment shaping reward ----------------------------------
struct RolloutConfig {
    int episode_steps = 500;
    // --- GPU-batched env (on-device rollout). gpu_env=true routes collect() through GpuEnv so the
    // whole loop (encode->forward->step) stays on the GPU; the trajectory is offloaded to host. The
    // obs/action dim E = planet_cap (>= max planets + 4 comet slots); fleet_cap bounds in-flight fleets.
    bool gpu_env = false;
    int planet_cap = 48;
    int fleet_cap = 1024;
    double reward_scale = 50.0;   // divides the raw production-margin delta
    double reward_clip = 5.0;     // clamp on the per-step shaped reward
    double prod_weight = 20.0;    // production-margin weight in the potential
    double terminal_bonus = 1.0;  // magnitude of the +/- terminal win signal
    // Curriculum stage: 1 = solo expansion (passive opponent), 2 = 1v1 vs starter,
    // 3 = mixed opponents. Stage only sets the opponent; the reward below is shared.
    int stage = 2;
    // 3-stage curriculum auto-ramp WITHIN one run, by training-progress fraction:
    //   step/total < stage1_frac -> stage 1 (passive);  < stage2_frac -> stage 2 (starter);
    //   otherwise -> stage 3 (mix). curriculum=false pins the fixed `stage` for the whole run.
    bool curriculum = true;
    double stage1_frac = 0.40;
    double stage2_frac = 0.75;
    // Reward = production-only dense + an ILLEGAL-dispatch penalty, with a loss forfeiting all:
    //   r_t = prod_reward_weight * d(own production)
    //       - illegal_launch_penalty * #launches-chosen-from-a-planet-not-owned-(or with 0 ships)
    //   R   = (win ? D : -D) + outcome_weight * o,   D = sum_t gamma^t r_t.
    // The action head is NOT ownership-masked: the policy may try to launch from any planet, and
    // learns the legal "dynamic action space" from the penalty (no reward/penalty for *where* a
    // legal launch goes -- we let the net be creative with the heuristic destination launches).
    // SMOOTH, BOUNDED per-game reward (each channel is accumulated over the episode, then capped):
    //   R = outcome + P(prod, +/-cap) + V(valid, 0..cap) - I(invalid, uncapped) + S(suppression)
    // outcome is applied ONLY at stage >= 2 (stage 1 is pure "valid dispatch + better production").
    double prod_reward_weight = 2.5;      // w_prod: rate of production gain into P
    double prod_reward_cap = 100.0;       // per-game cap on the production reward |P|
    double prod_reward_decay = 0.997;     // delta_Pi: per-STEP multiplicative decay on the production
                                          // term (weights step t's prod gain by decay^t). <1 front-
                                          // loads early expansion and shrinks production's share of the
                                          // return so the aiming (V/M) + outcome terms drive the
                                          // group-relative advantage. 1.0 = no decay (legacy).
    double valid_launch_reward = 0.02;    // w_valid: rate of valid (landing) launches into V
    double valid_reward_cap = 30.0;       // per-game cap on the valid-launch reward V
    double illegal_launch_penalty = 0.005;// w_inv: rate of invalid dispatches into I (uncapped:
                                          // a hard cap would remove the gradient while invalid is high)
    double miss_launch_penalty = 0.01;    // w_miss: rate of MISSED launches into M (uncapped). A miss
                                          // is a legal, in-budget launch from an OWNED planet whose
                                          // heading lands on no planet (= launches - valid): ships
                                          // thrown into space. This is the aiming STICK that
                                          // complements the valid-launch carrot. GUARDRAIL: keep
                                          // w_miss <= valid_launch_reward so a >50%-accurate launcher
                                          // stays net-positive (else "do nothing" becomes optimal ->
                                          // the passive basin the carrot-only design avoided).
    double win_bonus = 300.0;             // outcome (stage>=2): + on a win
    double loss_penalty = 100.0;          // outcome (stage>=2): - on a loss (draw = 0)
    bool loss_forfeit = false;            // legacy/unused in the bounded scheme (outcome is explicit)
    // Enemy-suppression (stage>=2, after warmup): reward bending the enemy's production growth down.
    //   g_t = d(enemy production); gbar EMA(beta); y_t = g_t - gbar_{t-1}; term = -w_e*y_t.
    double enemy_growth_weight = 0.5;     // w_e
    int enemy_growth_warmup = 50;         // x_0: steps before the suppression term switches on
    double enemy_growth_ema = 0.1;        // beta: EMA window (1.0 -> one-step second difference)

    // --- docs/set-ups/1.md reward set (event-based; all OFF by default so legacy runs are
    //     unchanged). When these are on the run typically zeroes the production/valid/miss/illegal
    //     terms above (the spec's reward REPLACES the production-shaping reward). ---
    double win_decay = 1.0;               // per-STEP multiplicative decay on the WIN bonus:
                                          // win value = win_bonus * win_decay^(episode length).
                                          // <1 rewards winning fast (0.995 = -0.5%/step). draw 0.
    double loss_decay = 1.0;              // same, for the LOSS: loss value = -loss_penalty *
                                          // loss_decay^(episode length). 1.0 = constant -loss_penalty;
                                          // set == win_decay for a symmetric "flip of winning" curve
                                          // (e.g. win_bonus=loss_penalty=1000, both *0.995^len).
    double capture_reward = 0.0;          // +v per planet the ego GAINS this step, -v per planet it
                                          // LOSES (owner flip to/from ego among alive slots).
    double dispatch_reward = 0.0;         // +r for each of the first `dispatch_reward_count` committed
    int dispatch_reward_count = 50;       // legal ego launches in a game (a "get moving" carrot).
    double fleet_hit_base = 0.0;          // per EGO fleet that hits a planet (any owner): reward
    double fleet_hit_ship_weight = 0.0;   //   fleet_hit_base + fleet_hit_ship_weight * ships_on_fleet,
    double fleet_hit_cap = 300.0;         //   accumulated then capped at fleet_hit_cap per game.
    double ppo_reward_scale = 1.0;        // PPO-only: divide the assembled per-step reward AND the
                                          // outcome by this before GAE. Shrinks the value targets to a
                                          // fittable range (advantages are re-normalized, so the policy
                                          // objective is unchanged); 1.0 = no scaling.
    double selfplay_prob = 0.5;           // stage 4: fraction of iterations played vs the frozen
                                          // self-play snapshot (else vs starter).
    double decay_start_step = 0.0;        // win/loss decay only kicks in AFTER this episode step:
                                          // value = bonus * decay^max(0, len - decay_start_step).
                                          // 0 = decay from step 0 (legacy). e.g. 100 = flat for the
                                          // first 100 steps, then decays (rewards a decisive mid-game).
    double prod_milestone_reward = 0.0;   // +r each time the ego's TOTAL ship count (planets+fleets)
    double prod_milestone_base = 100.0;   // crosses a doubling threshold base, 2*base, 4*base, ...
                                          // (the "bring back the production reward" channel; rewards a
                                          // growing economy, monotonic -- losing ships never penalizes).
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
    PpoConfig ppo;
    RolloutConfig rollout;
    SelfPlayConfig selfplay;
    std::string algo = "grpo";  // "grpo" (group baseline) or "ppo" (value head + GAE)
    long total_steps = 8'000'000;
    long eval_every = 400'000;
    // --- ITERATION-driven schedule (overrides step-based when stage1_iters>0 / total_iters>0) ---
    // Curriculum advances by iteration count: stage 1 for [0,s1), stage 2 for [s1,s1+s2), stage 3 after.
    // Checkpoints + log/eval fire every ckpt_every_early iters in stages 1&2 and ckpt_every_late in
    // stage 3, each written to run_dir/ckpt/itNNNN.owc (a progression for the behavior visualizations).
    int stage1_iters = 0;
    int stage2_iters = 0;
    int stage3_iters = 0;        // 4-stage curriculum: [s1) noop, [s1,s1+s2) random, [s1+s2,
                                 // s1+s2+s3) starter, [s1+s2+s3,) self-play+starter (stage 4).
                                 // 0 keeps the legacy 3-stage schedule (stage 3 = mix from s1+s2).
    long total_iters = 0;        // >0: stop after this many iterations (else stop at total_steps)
    int ckpt_every_early = 50;   // checkpoint/eval cadence (iters) during stage 1 & 2
    int ckpt_every_late = 100;   // checkpoint/eval cadence (iters) during stage 3
    std::string device = "cuda";
    uint64_t seed = 0;

    int num_envs() const { return grpo.group_size * grpo.num_groups; }
    bool is_ppo() const { return algo == "ppo"; }
};

}  // namespace ow
