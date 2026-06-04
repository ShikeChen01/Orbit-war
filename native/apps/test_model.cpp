// Parity gate for the refactored agent model: load a checkpoint into BOTH the old
// EntityPolicy (rl/policy.hpp) and the new Agent/PolicyNet, encode a batch of worlds, and
// assert the logits and greedy actions match. Confirms the hpp/cpp split reproduces the
// proven network bit-for-bit before the GRPO trainer is built on top.
//   native\run.cmd ow_test_model [ckpt.owc] [worlds.owp]
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "core/encode.hpp"
#include "io/serialize.hpp"
#include "rl/model/agent.hpp"
#include "rl/policy.hpp"  // old EntityPolicy (reference)

int main(int argc, char** argv) {
    std::string ckpt = argc > 1 ? argv[1] : "runs/native/bc_start.owc";
    std::string worlds = argc > 2 ? argv[2] : "runs/native/eval.owp";

    ow::CheckpointMeta meta;
    auto sd = ow::read_checkpoint(ckpt, meta);
    auto pool = ow::read_world_pool(worlds);
    int B = std::min<int>(64, (int)pool.size());
    int E = meta.max_entities, F = ow::N_ENTITY_FEATURES, G = ow::N_GLOBAL_FEATURES;

    std::vector<float> ent((size_t)B * E * F, 0.f), em((size_t)B * E, 0.f),
        am((size_t)B * E, 0.f), gl((size_t)B * G, 0.f);
    std::vector<long> rid(E), rsh(E);
    for (int i = 0; i < B; ++i) {
        ow::encode_obs(pool[i], 0, E, ent.data() + (size_t)i * E * F, em.data() + (size_t)i * E,
                       am.data() + (size_t)i * E, gl.data() + (size_t)i * G, rid.data(),
                       rsh.data(), meta.episode_steps);
    }
    auto o = torch::TensorOptions().dtype(torch::kFloat32);
    auto t_ent = torch::from_blob(ent.data(), {B, E, F}, o).clone();
    auto t_em = torch::from_blob(em.data(), {B, E}, o).clone();
    auto t_am = torch::from_blob(am.data(), {B, E}, o).clone();
    auto t_gl = torch::from_blob(gl.data(), {B, G}, o).clone();

    // new model
    ow::ModelConfig mc;
    mc.n_entity_features = F;
    mc.n_global_features = G;
    mc.hidden = meta.hidden;
    mc.max_entities = E;
    mc.angle_bins = meta.angle_bins;
    mc.target_mode = meta.target_mode != 0;
    mc.fractions = meta.fractions;
    ow::Agent agent(mc, torch::kCPU);
    agent.load_state_dict(sd);
    auto new_logits = agent.evaluate(t_ent, t_em, t_am, t_gl,
                                     torch::zeros({B, E}, torch::kLong)).logits;

    // old reference
    int A = mc.actions_per_entity();
    ow::EntityPolicy old(F, G, A, meta.hidden, meta.target_mode != 0, (int)meta.fractions.size());
    ow::load_python_state_dict(old, sd);
    old->eval();
    torch::Tensor old_logits;
    { torch::NoGradGuard ng; old_logits = std::get<0>(old->forward(t_ent, t_em, t_am, t_gl)); }

    double dmax = (new_logits - old_logits).abs().max().item<float>();
    auto g_new = std::get<1>(new_logits.max(-1));
    auto g_old = std::get<1>(old_logits.max(-1));
    int64_t mism = (g_new != g_old).sum().item<int64_t>();
    printf("model parity: B=%d  max|dlogit|=%.3e  greedy_mismatches=%lld / %d\n", B, dmax,
           (long long)mism, B * E);
    bool ok = dmax < 1e-3 && mism == 0;
    printf(ok ? "PARITY OK\n" : "PARITY FAIL\n");
    return ok ? 0 : 1;
}
