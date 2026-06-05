// Behavior-clone the scripted starter into the continuous PolicyNet -> bc.owc, a warm start for
// GRPO (`--init-from runs/native/bc.owc`). Rationale: from-scratch GRPO cannot bootstrap against the
// starter -- once the policy loses every game in a group, all group returns are ~equal so the
// group-relative advantage carries no signal, and it collapses to passivity. Cloning the starter's
// atan2 aiming first gives RL a competent, ACTIVE base (greedy means already above the commit
// threshold) to finetune from. Supervised: regress the squashed mean to the starter's (alpha,phi).
//   native\run.cmd ow_bc --worlds runs/native/train.owp --eval-worlds runs/native/eval.owp \
//       --hidden 512 --n-res-blocks 5 --out runs/native/bc.owc
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "apps/cli.hpp"
#include "core/agents.hpp"
#include "core/encode.hpp"
#include "core/sim.hpp"
#include "core/state.hpp"
#include "io/serialize.hpp"
#include "rl/arena_cont.hpp"
#include "rl/config.hpp"
#include "rl/model/agent.hpp"
#include "rl/reward.hpp"

#ifdef _WIN32
extern "C" __declspec(dllimport) void* __stdcall LoadLibraryA(const char*);
#endif

using namespace ow;
using torch::indexing::None;
using torch::indexing::Slice;

int main(int argc, char** argv) try {
    setvbuf(stdout, nullptr, _IONBF, 0);
#ifdef _WIN32
    LoadLibraryA("c10_cuda.dll");
    LoadLibraryA("torch_cuda.dll");
#endif
    Args a(argc, argv);
    std::string worlds_path = a.s("worlds", "runs/native/train.owp");
    std::string eval_path = a.s("eval-worlds", "runs/native/eval.owp");
    std::string out = a.s("out", "runs/native/bc.owc");
    long n_samples = a.l("samples", 100000);
    int epochs = a.i("epochs", 25);
    double lr = a.f("lr", 1e-3);
    int gen_steps = a.i("gen-steps", 250);

    torch::Device dev(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    printf("cuda=%d\n", (int)torch::cuda::is_available());

    ModelConfig cfg;
    cfg.n_entity_features = N_ENTITY_FEATURES;
    cfg.n_global_features = N_GLOBAL_FEATURES;
    cfg.hidden = a.i("hidden", 512);
    cfg.n_res_blocks = a.i("n-res-blocks", 5);
    cfg.max_entities = a.i("max-entities", 40);
    cfg.fleets_per_planet = a.i("fleets", 5);
    cfg.d_g = a.i("d-g", 32);
    cfg.use_glu = a.i("use-glu", 1) != 0;
    cfg.std_state_dependent = a.i("std-state-dependent", 1) != 0;
    cfg.act_threshold = a.f("act-threshold", 0.03);
    cfg.logstd_min = a.f("logstd-min", -2.0);
    cfg.logstd_max = a.f("logstd-max", 1.0);
    cfg.init_mu_scale = a.f("init-mu-scale", 1.0);  // BC fits the means -> use full-scale init
    cfg.init_phi_bias = a.f("init-phi-bias", -2.0);

    const int E = cfg.max_entities, F = N_ENTITY_FEATURES, G = N_GLOBAL_FEATURES;
    const int K = cfg.fleets_per_planet, twoK = 2 * K;
    const double TWO_PI = 2.0 * PI;

    auto pool = read_world_pool(worlds_path);
    if (pool.empty()) { printf("empty world pool %s\n", worlds_path.c_str()); return 1; }

    // --- generate (obs for player 0, starter-target) samples. player0 = starter (the expert we
    //     imitate); player1 alternates starter/random for state diversity. ---
    printf("generating up to %ld samples (gen_steps=%d)...\n", n_samples, gen_steps);
    std::vector<float> ent, em, am, gl, talpha, tphi, cmask;
    ent.reserve((size_t)n_samples * E * F);
    Config econf{500, 6.0, 4.0};
    std::mt19937_64 rng(12345);
    long S = 0;
    for (size_t w = 0; S < n_samples; ++w) {
        GameState s = pool[w % pool.size()];
        bool opp_random = (w % 2 == 1);
        for (int t = 0; t < gen_steps && S < n_samples; ++t) {
            std::vector<float> e(E * F, 0.f), m(E, 0.f), amask(E, 0.f), g(G, 0.f);
            std::vector<long> rid(E, -1), rsh(E, 0);
            encode_obs(s, 0, E, e.data(), m.data(), amask.data(), g.data(), rid.data(), rsh.data(), 500);
            std::vector<float> ta(E * K, 0.f), tp(E * K, 0.f), cm(E * K, 0.f);
            Action mv0 = starter_agent(s, 0);
            for (const auto& mvv : mv0) {
                int r = -1;
                for (int rr = 0; rr < E; ++rr)
                    if (rid[rr] == mvv.from_id) { r = rr; break; }
                if (r < 0) continue;
                double ang = mvv.angle;
                while (ang < 0) ang += TWO_PI;
                double alpha = std::fmod(ang, TWO_PI) / TWO_PI;
                long sh = rsh[r] > 0 ? rsh[r] : 1;
                double phi = std::min(1.0, std::max(0.0, (double)mvv.ships / (double)sh));
                ta[r * K + 0] = (float)alpha;
                tp[r * K + 0] = (float)phi;
                cm[r * K + 0] = 1.f;
            }
            ent.insert(ent.end(), e.begin(), e.end());
            em.insert(em.end(), m.begin(), m.end());
            am.insert(am.end(), amask.begin(), amask.end());
            gl.insert(gl.end(), g.begin(), g.end());
            talpha.insert(talpha.end(), ta.begin(), ta.end());
            tphi.insert(tphi.end(), tp.begin(), tp.end());
            cmask.insert(cmask.end(), cm.begin(), cm.end());
            ++S;
            Action mv1 = opp_random ? random_agent(s, 1, rng) : starter_agent(s, 1);
            std::vector<Action> acts = {mv0, mv1};
            ow::step(s, acts, econf);
            s.step += 1;
            if (is_terminal(s, 500)) break;
        }
    }
    printf("collected %ld samples\n", S);

    auto fo = torch::TensorOptions().dtype(torch::kFloat32);
    auto T_ent = torch::from_blob(ent.data(), {S, E, F}, fo).clone();
    auto T_em = torch::from_blob(em.data(), {S, E}, fo).clone();
    auto T_am = torch::from_blob(am.data(), {S, E}, fo).clone();
    auto T_gl = torch::from_blob(gl.data(), {S, G}, fo).clone();
    auto T_ta = torch::from_blob(talpha.data(), {S, E, K}, fo).clone();
    auto T_tp = torch::from_blob(tphi.data(), {S, E, K}, fo).clone();
    auto T_cm = torch::from_blob(cmask.data(), {S, E, K}, fo).clone();

    Agent policy(cfg, dev);
    policy.net->train();
    torch::optim::Adam opt(policy.net->parameters(), torch::optim::AdamOptions(lr));

    const long B = 4096;
    for (int ep = 0; ep < epochs; ++ep) {
        auto perm = torch::randperm(S, torch::TensorOptions().dtype(torch::kLong));
        double ep_loss = 0, ep_phi = 0, ep_al = 0;
        int nb = 0;
        for (long off = 0; off + B <= S; off += B) {
            auto mi = perm.narrow(0, off, B);
            auto ent_b = T_ent.index_select(0, mi).to(dev);
            auto em_b = T_em.index_select(0, mi).to(dev);
            auto am_b = T_am.index_select(0, mi).to(dev);
            auto gl_b = T_gl.index_select(0, mi).to(dev);
            auto ta_b = T_ta.index_select(0, mi).to(dev);
            auto tp_b = T_tp.index_select(0, mi).to(dev);
            auto cm_b = T_cm.index_select(0, mi).to(dev);

            auto mean = policy.net->forward(ent_b, em_b, am_b, gl_b).first;  // (B,E,2K)
            auto act = torch::sigmoid(mean);
            auto a_alpha = act.index({Slice(), Slice(), Slice(0, None, 2)});  // (B,E,K)
            auto a_phi = act.index({Slice(), Slice(), Slice(1, None, 2)});    // (B,E,K)

            // phi: BALANCED by the commit mask. ~3 of ~200 slots/sample are launches (target ~0.5)
            // vs ~197 zeros; a plain mean(mse) is dominated by the zeros so the policy outputs phi~0
            // everywhere (inert!). Average the committed and non-committed terms separately.
            auto phi_commit = (cm_b * (a_phi - tp_b).pow(2)).sum() / (cm_b.sum() + 1e-6);
            auto phi_zero = ((1.0 - cm_b) * a_phi.pow(2)).sum() / ((1.0 - cm_b).sum() + 1e-6);
            auto phi_loss = phi_commit + 0.2 * phi_zero;
            auto ang_pred = a_alpha * TWO_PI, ang_tgt = ta_b * TWO_PI;
            auto dcos = torch::cos(ang_pred) - torch::cos(ang_tgt);
            auto dsin = torch::sin(ang_pred) - torch::sin(ang_tgt);
            auto alpha_loss = (cm_b * (dcos * dcos + dsin * dsin)).sum() / (cm_b.sum() + 1e-6);
            auto loss = phi_loss + 0.3 * alpha_loss;

            opt.zero_grad();
            loss.backward();
            opt.step();
            ep_loss += loss.item<double>();
            ep_phi += phi_loss.item<double>();
            ep_al += alpha_loss.item<double>();
            ++nb;
        }
        if (nb) printf("epoch %2d/%d  loss %.5f (phi %.5f  alpha %.5f)\n", ep + 1, epochs,
                       ep_loss / nb, ep_phi / nb, ep_al / nb);
    }

    // --- eval the cloned (greedy) policy in the bit-exact arena ---
    auto evalw = read_world_pool(eval_path);
    policy.net->eval();
    ArenaCont arena(cfg.max_entities, cfg.fleets_per_planet, cfg.act_threshold, 500, dev);
    auto rS = arena.play(policy, evalw, 1, 12345);
    auto rR = arena.play(policy, evalw, 0, 12345);
    int n = std::max(1, (int)evalw.size());
    printf("BC eval: vsSTART %.1f%% (margin %.1f, len %.0f) | vsRAND %.1f%%\n",
           100.0 * rS.p0_wins / n, rS.mean_margin, rS.mean_len, 100.0 * rR.p0_wins / n);

    CheckpointMeta meta;
    meta.n_entity_features = cfg.n_entity_features;
    meta.n_global_features = cfg.n_global_features;
    meta.hidden = cfg.hidden;
    meta.max_entities = cfg.max_entities;
    meta.episode_steps = 500;
    write_checkpoint(out, policy.state_dict(), meta);
    printf("wrote %s  (finetune with: --init-from %s --hidden %d --n-res-blocks %d)\n", out.c_str(),
           out.c_str(), cfg.hidden, cfg.n_res_blocks);
    return 0;
} catch (const std::exception& e) {
    fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
}
