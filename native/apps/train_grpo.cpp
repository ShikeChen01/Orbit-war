// GRPO training entry point. Warm-starts from a BC checkpoint (which also becomes the fixed
// KL reference), trains with the native grouped-rollout GRPO trainer, and writes a results
// folder (metrics.csv + best/last .owc + config.txt). No Python in the loop.
//   native\run.cmd ow_train_grpo --init-from runs/native/bc_start.owc \
//       --worlds runs/native/train.owp --eval-worlds runs/native/eval.owp \
//       --total-steps 30000000 --run-dir runs/grpo/run1
#include <cstdio>
#include <string>

#include "apps/cli.hpp"
#include "core/encode.hpp"
#include "io/serialize.hpp"
#include "rl/config.hpp"
#include "rl/grpo_trainer.hpp"

int main(int argc, char** argv) {
    ow::Args a(argc, argv);
    std::string worlds = a.s("worlds", "runs/native/train.owp");
    std::string evalw = a.s("eval-worlds", "runs/native/eval.owp");
    std::string init = a.s("init-from", "runs/native/bc_start.owc");
    std::string run_dir = a.s("run-dir", "runs/grpo/run");

    ow::CheckpointMeta meta;
    auto sd = ow::read_checkpoint(init, meta);  // sizes the model + is the KL reference

    ow::TrainConfig cfg;
    cfg.model.n_entity_features = ow::N_ENTITY_FEATURES;
    cfg.model.n_global_features = ow::N_GLOBAL_FEATURES;
    cfg.model.hidden = meta.hidden;
    cfg.model.angle_bins = meta.angle_bins;
    cfg.model.max_entities = meta.max_entities;
    cfg.model.target_mode = meta.target_mode != 0;
    cfg.model.fractions = meta.fractions;
    cfg.rollout.episode_steps = a.i("episode-steps", meta.episode_steps);
    cfg.rollout.prod_weight = a.f("prod-weight", 20.0);
    cfg.total_steps = a.l("total-steps", 30'000'000);
    cfg.grpo.group_size = a.i("group-size", 8);
    cfg.grpo.num_groups = a.i("num-groups", 64);
    cfg.grpo.kl_beta = a.f("kl-beta", 0.04);
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
    trainer.load_init(sd);
    trainer.train();
    return 0;
}
