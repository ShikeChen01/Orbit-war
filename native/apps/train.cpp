// Native (Python-free) training entry point. Reads a cached world pool (.owp), runs the
// C++ PPO trainer with periodic C++-arena eval, and writes .owc checkpoints. The dev loop
// becomes:  edit -> native\build.cmd -> native\run.cmd ow_train ...  with no Python and no
// .pyd reimport in the path. (One-time world-pool gen is the only Python step; see
// scripts/gen_world_pool.py.)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>

#include "apps/cli.hpp"
#include "io/serialize.hpp"
#include "rl/arena.hpp"
#include "rl/trainer.hpp"

int main(int argc, char** argv) {
    ow::Args a(argc, argv);

    ow::TrainerConfig cfg;
    cfg.num_envs = a.i("num-envs", 512);
    cfg.rollout_steps = a.i("rollout-steps", 16);
    cfg.total_steps = (int)a.l("total-steps", 8'000'000);
    cfg.hidden = a.i("hidden", 128);
    cfg.angle_bins = a.i("angle-bins", 16);
    cfg.target_mode = a.flag("target-mode");
    cfg.episode_steps = a.i("episode-steps", 500);
    cfg.selfplay_start_step = (int)a.l("selfplay-start-step", 2'000'000);
    cfg.snapshot_every = (int)a.l("snapshot-every", 500'000);
    cfg.prod_weight = a.f("prod-weight", 20.0);
    cfg.value_warmup_updates = a.i("value-warmup-updates", 0);
    cfg.reward_scale = a.f("reward-scale", 50.0);
    cfg.ent_coef = a.f("ent-coef", 0.02);
    cfg.lr = a.f("lr", 3e-4);
    cfg.gamma = a.f("gamma", 0.997);
    cfg.clip = a.f("clip", 0.2);
    cfg.opp_random = a.f("opp-random", 1.0);
    cfg.opp_starter = a.f("opp-starter", 1.0);
    cfg.opp_self = a.f("opp-self", 2.0);
    cfg.device = a.s("device", "cuda");
    cfg.seed = (uint64_t)a.l("seed", 0);

    long eval_every = a.l("eval-every", 400'000);
    std::string worlds_path = a.s("worlds", "runs/native/train.owp");
    std::string eval_path = a.s("eval-worlds", "runs/native/eval.owp");
    std::string out = a.s("out", "runs/native/run.owc");
    std::string init_from = a.s("init-from", "");
    std::string out_last = out;
    {
        auto pos = out_last.rfind(".owc");
        if (pos != std::string::npos) out_last.replace(pos, 4, ".last.owc");
        else out_last += ".last";
    }

    printf("loading worlds: %s + %s\n", worlds_path.c_str(), eval_path.c_str());
    auto train_pool = ow::read_world_pool(worlds_path);
    auto eval_worlds = ow::read_world_pool(eval_path);
    int eval_games = (int)eval_worlds.size();
    printf("train worlds=%zu eval worlds=%d | device=%s target_mode=%d\n",
           train_pool.size(), eval_games, cfg.device.c_str(), (int)cfg.target_mode);

    ow::CheckpointMeta meta;
    meta.n_entity_features = ow::N_ENTITY_FEATURES;
    meta.n_global_features = ow::N_GLOBAL_FEATURES;
    meta.hidden = cfg.hidden;
    meta.angle_bins = cfg.angle_bins;
    meta.max_entities = cfg.max_entities;
    meta.num_fracs = (int)cfg.fractions.size();
    meta.episode_steps = cfg.episode_steps;
    meta.target_mode = cfg.target_mode ? 1 : 0;
    meta.fractions = cfg.fractions;

    ow::Trainer trainer(cfg);
    trainer.set_world_pool(train_pool);
    if (!init_from.empty()) {
        ow::CheckpointMeta m2;
        auto sd = ow::read_checkpoint(init_from, m2);
        trainer.load_weights(sd);
        printf("warm-started policy from %s\n", init_from.c_str());
    }

    ow::Arena arena(cfg.max_entities, cfg.angle_bins, cfg.fractions, cfg.hidden,
                    cfg.episode_steps, cfg.device, cfg.target_mode);

    printf("%10s %8s %7s %8s %9s %5s %8s %5s\n", "step", "ret", "vsRAND", "vsSTART",
           "margin", "len", "sps", "best");
    long total = cfg.total_steps, done = 0;
    double best = -1.0;
    auto t_start = std::chrono::steady_clock::now();
    while (done < total) {
        long chunk = std::min<long>(eval_every, total - done);
        auto t0 = std::chrono::steady_clock::now();
        trainer.train(chunk);
        done += chunk;
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double sps = chunk / std::max(dt, 1e-9);

        auto sd = trainer.get_state_dict();
        arena.load_p0(sd);
        auto r_rand = arena.play(eval_worlds, 0, true, 12345);
        auto r_start = arena.play(eval_worlds, 1, true, 12345);
        double wr_rand = eval_games ? 100.0 * r_rand.p0_wins / eval_games : 0.0;
        double wr_start = eval_games ? 100.0 * r_start.p0_wins / eval_games : 0.0;

        ow::write_checkpoint(out_last, sd, meta);
        const char* flag = "";
        if (wr_start > best) {
            best = wr_start;
            ow::write_checkpoint(out, sd, meta);
            flag = "*";
        }
        printf("%10ld %8.2f %6.1f%% %7.1f%% %+9.1f %5.0f %8.0f %5s\n", done,
               trainer.recent_return(), wr_rand, wr_start, r_start.mean_margin,
               r_start.mean_len, sps, flag);
        fflush(stdout);
    }
    double total_dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    printf("\ndone in %.1f min | best vs starter=%.1f%% -> %s (last -> %s)\n",
           total_dt / 60.0, best, out.c_str(), out_last.c_str());
    return 0;
}
