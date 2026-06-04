// Native checkpoint evaluator: load a .owc policy + a .owp world pool and play vs the
// scripted baselines in the C++ arena (the project's fitness function). No Python.
//   native\run.cmd ow_eval --ckpt runs/native/run.owc --worlds runs/native/eval.owp
#include <cstdio>
#include <string>

#include "apps/cli.hpp"
#include "io/serialize.hpp"
#include "rl/arena.hpp"

int main(int argc, char** argv) {
    ow::Args a(argc, argv);
    std::string ckpt = a.s("ckpt", "runs/native/run.owc");
    std::string worlds_path = a.s("worlds", "runs/native/eval.owp");
    std::string device = a.s("device", "cuda");
    bool sample = a.flag("sample");  // default greedy (deterministic)
    uint64_t seed = (uint64_t)a.l("seed", 12345);

    ow::CheckpointMeta meta;
    auto sd = ow::read_checkpoint(ckpt, meta);
    auto worlds = ow::read_world_pool(worlds_path);
    int n = (int)worlds.size();
    printf("ckpt=%s worlds=%d hidden=%d target_mode=%d episode_steps=%d\n", ckpt.c_str(), n,
           meta.hidden, (int)meta.target_mode, meta.episode_steps);

    ow::Arena arena(meta.max_entities, meta.angle_bins, meta.fractions, meta.hidden,
                    meta.episode_steps, device, meta.target_mode != 0);
    arena.load_p0(sd);
    auto r_rand = arena.play(worlds, 0, !sample, seed);
    auto r_start = arena.play(worlds, 1, !sample, seed);
    printf("vs random : winrate=%5.1f%% margin=%+8.1f len=%.0f\n",
           n ? 100.0 * r_rand.p0_wins / n : 0.0, r_rand.mean_margin, r_rand.mean_len);
    printf("vs starter: winrate=%5.1f%% margin=%+8.1f len=%.0f\n",
           n ? 100.0 * r_start.p0_wins / n : 0.0, r_start.mean_margin, r_start.mean_len);
    return 0;
}
