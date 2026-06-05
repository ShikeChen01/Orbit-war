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
    cfg.model.act_threshold = a.f("act-threshold", 0.05);                 // tau_act
    cfg.model.logstd_min = a.f("logstd-min", -2.0);
    cfg.model.logstd_max = a.f("logstd-max", 1.0);
    // --- reward ---
    cfg.rollout.stage = a.i("stage", 2);  // fixed stage when curriculum is off
    // 3-stage curriculum auto-ramp within the run (on by default): passive -> starter -> mix
    cfg.rollout.curriculum = a.i("curriculum", 1) != 0;
    cfg.rollout.stage1_frac = a.f("stage1-frac", 0.40);
    cfg.rollout.stage2_frac = a.f("stage2-frac", 0.75);
    cfg.rollout.prod_reward_weight = a.f("prod-reward-weight", 2.5);
    cfg.rollout.prod_reward_cap = a.f("prod-reward-cap", 200.0);
    cfg.rollout.valid_launch_reward = a.f("valid-launch-reward", 0.02);
    cfg.rollout.valid_reward_cap = a.f("valid-reward-cap", 30.0);
    cfg.rollout.illegal_launch_penalty = a.f("illegal-launch-penalty", 0.005);
    cfg.rollout.win_bonus = a.f("win-bonus", 300.0);
    cfg.rollout.loss_penalty = a.f("loss-penalty", 100.0);
    cfg.rollout.enemy_growth_weight = a.f("enemy-growth-weight", 0.5);
    cfg.rollout.enemy_growth_warmup = a.i("enemy-growth-warmup", 50);
    cfg.rollout.enemy_growth_ema = a.f("enemy-growth-ema", 0.1);
    // --- GPU-batched on-device rollout (the whole loop stays on the GPU) ---
    cfg.rollout.gpu_env = a.i("gpu-env", 0) != 0;
    cfg.rollout.planet_cap = a.i("planet-cap", 48);  // obs/action dim E (>= max planets + 4 comets)
    cfg.rollout.fleet_cap = a.i("fleet-cap", 1024);
    cfg.total_steps = a.l("total-steps", 30'000'000);
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
