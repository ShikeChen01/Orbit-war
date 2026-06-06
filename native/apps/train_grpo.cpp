// GRPO training entry point. Warm-starts from a BC checkpoint (which also becomes the fixed
// KL reference), trains with the native grouped-rollout GRPO trainer, and writes a results
// folder (metrics.csv + best/last .owc + config.txt). No Python in the loop.
//   native\run.cmd ow_train_grpo --init-from runs/native/bc_start.owc \
//       --worlds runs/native/train.owp --eval-worlds runs/native/eval.owp \
//       --total-steps 30000000 --run-dir runs/grpo/run1
#include <cstdio>
#include <map>
#include <string>

#include "apps/cli.hpp"
#include "core/encode.hpp"
#include "io/serialize.hpp"
#include "rl/config.hpp"
#include "rl/grpo_trainer.hpp"

#ifdef _WIN32
// Force-load torch_cuda.dll so its static initializers register the CUDA backend (MSVC drops it
// otherwise -- see native-cpp-build-toolchain memory). Without this .to(kCUDA) silently fails and
// the trainer falls back to CPU. torch\lib must be on PATH (native\run.cmd does this).
extern "C" __declspec(dllimport) void* __stdcall LoadLibraryA(const char*);
#endif

int main(int argc, char** argv) try {
    setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered so output survives an abort
#ifdef _WIN32
    LoadLibraryA("c10_cuda.dll");
    LoadLibraryA("torch_cuda.dll");
#endif
    ow::Args a(argc, argv);
    std::string worlds = a.s("worlds", "runs/native/train.owp");
    std::string evalw = a.s("eval-worlds", "runs/native/eval.owp");
    std::string init = a.s("init-from", "scratch");  // continuous v4 trains from scratch by default
    std::string run_dir = a.s("run-dir", "runs/grpo/run");

    // From-scratch (`--init-from none|scratch`): size the model from CLI + encode.hpp constants,
    // random-init, and disable the KL anchor (there is no reference to anchor to). Otherwise size
    // from the BC checkpoint and use it as the fixed KL reference.
    bool scratch = (init == "none" || init == "scratch" || init.empty());
    ow::CheckpointMeta meta;
    std::map<std::string, torch::Tensor> sd;
    if (!scratch) sd = ow::read_checkpoint(init, meta);

    ow::TrainConfig cfg;
    cfg.model.n_entity_features = ow::N_ENTITY_FEATURES;
    cfg.model.n_global_features = ow::N_GLOBAL_FEATURES;
    cfg.model.hidden = scratch ? a.i("hidden", 896) : meta.hidden;       // ~15M params at d=896,n_res=8
    cfg.model.n_res_blocks = a.i("n-res-blocks", 8);                      // trunk depth (residual MLPs)
    cfg.model.max_entities = scratch ? a.i("max-entities", 40) : meta.max_entities;
    cfg.rollout.episode_steps = a.i("episode-steps", scratch ? 500 : meta.episode_steps);
    // --- continuous actor (v4): the two A/B switches + shape knobs ---
    cfg.model.fleets_per_planet = a.i("fleets", 5);                       // K
    cfg.model.d_g = a.i("d-g", 32);                                       // board-globals embed dim
    cfg.model.use_glu = a.i("use-glu", 1) != 0;                           // trunk A/B
    cfg.model.std_state_dependent = a.i("std-state-dependent", 1) != 0;   // exploration A/B
    cfg.model.action_mask_policy = a.i("action-mask", 0) != 0;            // mask actor to legal slots
    cfg.model.target_actor = a.i("target-actor", 0) != 0;                 // v5 (dest,phi); else (dx,dy,phi)
    cfg.model.act_threshold = a.f("act-threshold", 0.05);                 // tau_act
    cfg.model.logstd_min = a.f("logstd-min", -2.0);
    cfg.model.logstd_max = a.f("logstd-max", 1.0);
    cfg.model.logstd_max_end = a.f("logstd-max-end", cfg.model.logstd_max);  // phase-1 decay target
    cfg.model.logstd_max_post = a.f("logstd-max-post", cfg.model.logstd_max_end);  // phase-2 head cap
    cfg.model.sigma_decay_iters = a.i("sigma-decay-iters", 0);  // >0: force decay over N iters then head
    cfg.model.init_mu_scale = a.f("init-mu-scale", 0.02);  // policy "calmness" at init (tunable)
    cfg.model.init_phi_bias = a.f("init-phi-bias", -4.0);  // less negative = less calm = more commits
    // --- reward ---
    cfg.rollout.stage = a.i("stage", 2);  // fixed stage when curriculum is off
    // 3-stage curriculum auto-ramp within the run (on by default): passive -> starter -> mix
    cfg.rollout.curriculum = a.i("curriculum", 1) != 0;
    cfg.rollout.stage1_frac = a.f("stage1-frac", 0.40);
    cfg.rollout.stage2_frac = a.f("stage2-frac", 0.75);
    cfg.rollout.prod_reward_weight = a.f("prod-reward-weight", 2.5);
    cfg.rollout.prod_reward_cap = a.f("prod-reward-cap", 100.0);
    cfg.rollout.prod_reward_decay = a.f("prod-reward-decay", 0.997);  // per-step prod decay (front-load)
    cfg.rollout.valid_launch_reward = a.f("valid-launch-reward", 0.02);
    cfg.rollout.valid_reward_cap = a.f("valid-reward-cap", 30.0);
    cfg.rollout.illegal_launch_penalty = a.f("illegal-launch-penalty", 0.005);
    cfg.rollout.miss_launch_penalty = a.f("miss-launch-penalty", 0.01);  // aiming STICK (<= valid reward)
    cfg.rollout.win_bonus = a.f("win-bonus", 300.0);
    cfg.rollout.loss_penalty = a.f("loss-penalty", 100.0);
    cfg.rollout.enemy_growth_weight = a.f("enemy-growth-weight", 0.5);
    cfg.rollout.enemy_growth_warmup = a.i("enemy-growth-warmup", 50);
    cfg.rollout.enemy_growth_ema = a.f("enemy-growth-ema", 0.1);
    // --- docs/set-ups/1.md event reward set (off by default) ---
    cfg.rollout.win_decay = a.f("win-decay", 1.0);                       // win bonus * win_decay^len
    cfg.rollout.loss_decay = a.f("loss-decay", 1.0);                     // loss penalty * loss_decay^len
    cfg.rollout.capture_reward = a.f("capture-reward", 0.0);             // +/- per planet gained/lost
    cfg.rollout.dispatch_reward = a.f("dispatch-reward", 0.0);           // first-N committed launches
    cfg.rollout.dispatch_reward_count = a.i("dispatch-count", 50);      // N
    cfg.rollout.fleet_hit_base = a.f("fleet-hit-base", 0.0);             // per ego fleet that hits
    cfg.rollout.fleet_hit_ship_weight = a.f("fleet-hit-ship-weight", 0.0);
    cfg.rollout.fleet_hit_cap = a.f("fleet-hit-cap", 300.0);             // per-game cap on hit reward
    cfg.rollout.ppo_reward_scale = a.f("reward-scale", 1.0);             // value-target rescale (PPO)
    cfg.rollout.selfplay_prob = a.f("selfplay-prob", 0.5);              // stage-4 self-play fraction
    cfg.rollout.decay_start_step = a.f("decay-start-step", 0.0);         // win/loss flat until this step
    cfg.rollout.prod_milestone_reward = a.f("prod-milestone-reward", 0.0);  // +r per ego ship doubling
    cfg.rollout.prod_milestone_base = a.f("prod-milestone-base", 100.0);    // first doubling threshold (then x2)
    // --- GPU-batched on-device rollout (the whole loop stays on the GPU) ---
    cfg.rollout.gpu_env = a.i("gpu-env", 0) != 0;
    cfg.rollout.planet_cap = a.i("planet-cap", 48);  // obs/action dim E (>= max planets + 4 comets)
    cfg.rollout.fleet_cap = a.i("fleet-cap", 1024);
    // TARGET actor: the destination categorical is over the E=planet_cap obs slots, so the model's
    // E must equal planet_cap (and the CPU eval obs must use the same width). Force them equal.
    if (cfg.model.target_actor) cfg.model.max_entities = cfg.rollout.planet_cap;
    cfg.total_steps = a.l("total-steps", 30'000'000);
    // --- iteration-driven schedule (curriculum + checkpoints by ITER) ---
    cfg.stage1_iters = a.i("stage1-iters", 0);
    cfg.stage2_iters = a.i("stage2-iters", 0);
    cfg.stage3_iters = a.i("stage3-iters", 0);  // >0 enables the 4-stage (self-play) curriculum
    cfg.total_iters = a.l("total-iters", 0);
    cfg.ckpt_every_early = a.i("ckpt-every-early", 50);
    cfg.ckpt_every_late = a.i("ckpt-every-late", 100);
    cfg.grpo.group_size = a.i("group-size", 8);
    cfg.grpo.num_groups = a.i("num-groups", 64);
    cfg.grpo.kl_beta = scratch ? 0.0 : a.f("kl-beta", 0.04);  // no KL anchor when from scratch
    cfg.grpo.clip = a.f("clip", 0.2);
    cfg.grpo.ent_coef = a.f("ent-coef", 0.01);
    cfg.grpo.gamma = a.f("gamma", 1.0);
    cfg.grpo.dense_weight = a.f("dense-weight", 1.0);
    cfg.grpo.outcome_weight = a.f("outcome-weight", 1.0);
    cfg.grpo.update_epochs = a.i("update-epochs", 2);
    cfg.grpo.minibatches = a.i("minibatches", 8);
    cfg.optim.lr = a.f("lr", 3e-4);
    // --- algo: GRPO (group baseline, default) or PPO+GAE (value head + per-step credit) ---
    cfg.algo = a.s("algo", "grpo");
    cfg.model.use_value_head = cfg.is_ppo() || (a.i("use-value-head", 0) != 0);
    cfg.ppo.gae_lambda = a.f("gae-lambda", 0.95);
    cfg.ppo.vf_coef = a.f("vf-coef", 0.5);
    cfg.ppo.gamma = a.f("ppo-gamma", 0.99);  // PPO discount (GAE applies it); GRPO uses --gamma (=1.0)
    cfg.selfplay.mix_random = a.f("mix-random", 1.0);
    cfg.selfplay.mix_starter = a.f("mix-starter", 1.0);
    cfg.device = a.s("device", "cuda");
    cfg.seed = (uint64_t)a.l("seed", 0);

    ow::GrpoTrainer trainer(cfg, run_dir);
    trainer.set_world_pool(ow::read_world_pool(worlds), ow::read_world_pool(evalw));
    if (scratch) trainer.init_scratch(); else trainer.load_init(sd);
    trainer.train();
    return 0;
} catch (const std::exception& e) {
    fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
}
