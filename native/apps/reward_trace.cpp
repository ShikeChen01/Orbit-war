// Single-game reward-curve diagnostic. Plays ONE episode and logs, per step, every reward channel
// (mirrors rl/rollout.cpp collect(): production P with the per-step decay, valid-landing V, illegal
// I, MISS M, suppression S) plus the realized game standing (ego vs enemy ships/production) and the
// reward-to-go. Purpose: pick the reward-shaping knobs (rho_m / w_v / cap / decay / K) by READING the
// magnitudes off a real game instead of guessing -- e.g. trace the starter (expert) to confirm good
// play scores positive and the miss penalty does not crush it, then trace random to see bad play.
//
//   native\run.cmd ow_reward_trace --ego starter --opponent starter --stage 2 --out runs/trace_start.csv
//   native\run.cmd ow_reward_trace --ego random  --opponent noop    --stage 1 --out runs/trace_rand.csv
//   native\run.cmd ow_reward_trace --ego policy --ckpt runs/grpo/dirvec2/last.owc --hidden 512 --fleets 8 ...
//
// NB: GRPO has no critic, so the "advantage the model thinks it has" is shown as the *realized*
// standing (ship margin) and the reward-to-go from each step -- the honest per-stage value, not a
// learned estimate.
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "apps/cli.hpp"
#include "core/agents.hpp"
#include "core/encode.hpp"
#include "core/sim.hpp"
#include "io/serialize.hpp"
#include "rl/config.hpp"
#include "rl/model/agent.hpp"
#include "rl/reward.hpp"

#ifdef _WIN32
extern "C" __declspec(dllimport) void* __stdcall LoadLibraryA(const char*);
#endif

using namespace ow;

// ego/enemy total ships (owned planets + in-flight fleets) -- the realized game standing.
static void side_ships(const GameState& s, long& s0, long& s1) {
    s0 = s1 = 0;
    for (const auto& p : s.planets) { if (p.owner == 0) s0 += p.ships; else if (p.owner == 1) s1 += p.ships; }
    for (const auto& f : s.fleets) { if (f.owner == 0) s0 += f.ships; else if (f.owner == 1) s1 += f.ships; }
}

int main(int argc, char** argv) try {
    setvbuf(stdout, nullptr, _IONBF, 0);
    Args a(argc, argv);
    std::string ego = a.s("ego", "starter");           // starter | random | policy
    std::string opps = a.s("opponent", "starter");     // starter | random | noop
    std::string ckpt = a.s("ckpt", "");                // required iff ego==policy
    std::string worlds = a.s("worlds", "runs/native/eval.owp");
    std::string out = a.s("out", "runs/reward_trace.csv");
    std::string record_json = a.s("record-json", "");  // if set, dump per-tick board geometry for render
    int world_idx = a.i("world", 0);
    int stage = a.i("stage", 2);                        // outcome O + suppression S apply at stage>=2
    bool greedy = a.i("greedy", 1) != 0;
    std::string device_s = a.s("device", "cpu");
    uint64_t seed = (uint64_t)a.l("seed", 0);

    // --- reward knobs: start from the shipped defaults, override from CLI (same names as ow_train_grpo) ---
    RolloutConfig rc;
    rc.prod_reward_weight = a.f("prod-reward-weight", rc.prod_reward_weight);
    rc.prod_reward_cap = a.f("prod-reward-cap", rc.prod_reward_cap);
    rc.prod_reward_decay = a.f("prod-reward-decay", rc.prod_reward_decay);
    rc.valid_launch_reward = a.f("valid-launch-reward", rc.valid_launch_reward);
    rc.valid_reward_cap = a.f("valid-reward-cap", rc.valid_reward_cap);
    rc.illegal_launch_penalty = a.f("illegal-launch-penalty", rc.illegal_launch_penalty);
    rc.miss_launch_penalty = a.f("miss-launch-penalty", rc.miss_launch_penalty);
    rc.win_bonus = a.f("win-bonus", rc.win_bonus);
    rc.loss_penalty = a.f("loss-penalty", rc.loss_penalty);
    rc.enemy_growth_weight = a.f("enemy-growth-weight", rc.enemy_growth_weight);
    rc.enemy_growth_warmup = a.i("enemy-growth-warmup", rc.enemy_growth_warmup);
    rc.enemy_growth_ema = a.f("enemy-growth-ema", rc.enemy_growth_ema);
    rc.episode_steps = a.i("episode-steps", rc.episode_steps);
    double gamma = a.f("gamma", 1.0);
    double act_thr = a.f("act-threshold", 0.03);
    int K = a.i("fleets", 5);
    int E = a.i("max-entities", 40);

    // --- optional policy (ego==policy): rebuild the model shape from the .owc meta + CLI shape knobs ---
    torch::Device dev(device_s == "cuda" ? torch::kCUDA : torch::kCPU);
    Agent policy;
    bool use_policy = (ego == "policy");
    if (use_policy) {
#ifdef _WIN32
        if (dev.is_cuda()) { LoadLibraryA("c10_cuda.dll"); LoadLibraryA("torch_cuda.dll"); }
#endif
        if (ckpt.empty()) { printf("ego=policy needs --ckpt\n"); return 1; }
        CheckpointMeta meta;
        auto sd = read_checkpoint(ckpt, meta);
        ModelConfig mc;
        mc.n_entity_features = N_ENTITY_FEATURES;
        mc.n_global_features = N_GLOBAL_FEATURES;
        mc.hidden = a.i("hidden", meta.hidden);
        mc.max_entities = a.i("max-entities", meta.max_entities);
        mc.fleets_per_planet = K;
        mc.d_g = a.i("d-g", 32);
        mc.use_glu = a.i("use-glu", 1) != 0;
        mc.n_res_blocks = a.i("n-res-blocks", 5);
        mc.std_state_dependent = a.i("std-state-dependent", 1) != 0;
        mc.act_threshold = act_thr;
        E = mc.max_entities;
        K = mc.fleets_per_planet;
        policy = Agent(mc, dev);
        policy.load_state_dict(sd);
        policy.net->eval();
        rc.episode_steps = a.i("episode-steps", meta.episode_steps);
    }

    auto pool = read_world_pool(worlds);
    if (pool.empty()) { printf("empty world pool: %s\n", worlds.c_str()); return 1; }
    GameState s = pool[world_idx % pool.size()];
    Config econf{rc.episode_steps, 6.0, 4.0};
    std::mt19937_64 rng(seed);
    const int nap = 3 * K, T = rc.episode_steps, F = N_ENTITY_FEATURES, G = N_GLOBAL_FEATURES;
    const int x0 = rc.enemy_growth_warmup;

    // per-step records (CSV written after the game so we can also emit reward-to-go)
    struct Row { int step; double dpe, dpn; long lnch, valid, miss, inval; double pI, vI, iI, mI, sI;
                 double cumP, cumV, cumI, cumM, cumS; long es, ens; double eprod, enprod; };
    std::vector<Row> rows;

    double P = 0, V = 0, I = 0, M = 0, S = 0, gpow = 1.0, ppow = 1.0, gbar = 0.0;
    double prev0 = my_production(s, 0), prev1 = my_production(s, 1);
    double outcome = 0.0;

    // Optional per-tick board-geometry capture for the behavior visualizations. Each frame is one
    // JSON object: planets [id,owner,x,y,radius,ships,prod], fleets [owner,x,y,angle,ships], comet ids.
    std::vector<std::string> frames;
    auto capture = [&](const GameState& g) {
        if (record_json.empty()) return;
        char b[64];
        std::string j = "{\"t\":" + std::to_string(g.step) + ",\"planets\":[";
        for (size_t i = 0; i < g.planets.size(); ++i) {
            const auto& p = g.planets[i];
            snprintf(b, sizeof(b), "[%ld,%d,%.3f,%.3f,%.3f,%ld,%d]", p.id, p.owner, p.x, p.y,
                     p.radius, p.ships, p.production);
            j += b; if (i + 1 < g.planets.size()) j += ',';
        }
        j += "],\"fleets\":[";
        for (size_t i = 0; i < g.fleets.size(); ++i) {
            const auto& f = g.fleets[i];
            snprintf(b, sizeof(b), "[%d,%.3f,%.3f,%.4f,%ld]", f.owner, f.x, f.y, f.angle, f.ships);
            j += b; if (i + 1 < g.fleets.size()) j += ',';
        }
        j += "],\"comets\":[";
        for (size_t i = 0; i < g.comet_planet_ids.size(); ++i) {
            j += std::to_string(g.comet_planet_ids[i]);
            if (i + 1 < g.comet_planet_ids.size()) j += ',';
        }
        j += "]}";
        frames.push_back(std::move(j));
    };

    for (int t = 0; t < T; ++t) {
        capture(s);  // board state the policy sees this tick
        // --- ego action ---
        Action move0;
        long invalid = 0;
        std::vector<long> rid(E, -1), rsh(E, 0);
        {
            std::vector<float> ent((size_t)E * F, 0.f), em(E, 0.f), am(E, 0.f), gl(G, 0.f);
            encode_obs(s, 0, E, ent.data(), em.data(), am.data(), gl.data(), rid.data(), rsh.data(), T);
            if (use_policy) {
                auto fo = torch::TensorOptions().dtype(torch::kFloat32);
                auto te = torch::from_blob(ent.data(), {1, E, F}, fo).to(dev);
                auto tm = torch::from_blob(em.data(), {1, E}, fo).to(dev);
                auto ta = torch::from_blob(am.data(), {1, E}, fo).to(dev);
                auto tg = torch::from_blob(gl.data(), {1, G}, fo).to(dev);
                auto act = policy.act(te, tm, ta, tg, greedy).action.to(torch::kCPU).contiguous();
                int inv = 0;
                move0 = decode_action_continuous(act.data_ptr<float>(), am.data(), rid.data(),
                                                 rsh.data(), E, K, act_thr, &inv);
                invalid = inv;
            } else if (ego == "random") {
                move0 = random_agent(s, 0, rng);
            } else {
                move0 = starter_agent(s, 0);
            }
        }
        // valid = launches whose heading lands on a planet (mirror rollout collect())
        long valid = 0;
        for (const auto& mv : move0) {
            const Planet* lp = nullptr;
            for (const auto& p : s.planets) if (p.id == mv.from_id) { lp = &p; break; }
            if (!lp) continue;
            double cx = std::cos(mv.angle), cy = std::sin(mv.angle);
            Fleet f{0, 0, lp->x + cx * (lp->radius + 0.1), lp->y + cy * (lp->radius + 0.1),
                    mv.angle, mv.from_id, mv.ships};
            if (fleet_target(f, s.planets).first >= 0) ++valid;
        }
        long lnch = (long)move0.size(), miss = lnch - valid;

        // --- opponent ---
        Action move1;
        if (opps == "random") move1 = random_agent(s, 1, rng);
        else if (opps == "starter") move1 = starter_agent(s, 1);
        // "noop" -> empty (passive)

        std::vector<Action> acts = {move0, move1};
        ow::step(s, acts, econf);
        s.step += 1;

        double p0 = my_production(s, 0), p1 = my_production(s, 1);
        double dpe = p0 - prev0, dpn = p1 - prev1;
        double y_t = dpn - gbar;
        gbar = (1.0 - rc.enemy_growth_ema) * gbar + rc.enemy_growth_ema * dpn;
        double pI = gpow * ppow * rc.prod_reward_weight * dpe;
        double vI = gpow * rc.valid_launch_reward * (double)valid;
        double iI = gpow * rc.illegal_launch_penalty * (double)invalid;
        double mI = gpow * rc.miss_launch_penalty * (double)miss;
        double sI = (stage >= 2 && t >= x0) ? gpow * (-rc.enemy_growth_weight * y_t) : 0.0;
        P += pI; V += vI; I += iI; M += mI; S += sI;
        gpow *= gamma; ppow *= rc.prod_reward_decay;
        prev0 = p0; prev1 = p1;

        long es, ens; side_ships(s, es, ens);
        rows.push_back({s.step, dpe, dpn, lnch, valid, miss, invalid, pI, vI, iI, mI, sI,
                        P, V, I, M, S, es, ens, p0, p1});
        if (is_terminal(s, T)) { outcome = outcome_sign(s); break; }
    }
    if (rows.empty()) { printf("no steps?\n"); return 1; }
    if (outcome == 0.0) outcome = outcome_sign(s);  // hit the cap

    if (!record_json.empty()) {  // write the captured frames (+ a final frame) as one JSON document
        capture(s);
        std::ofstream jf(record_json);
        jf << "{\"board_size\":" << BOARD_SIZE << ",\"center\":" << CENTER
           << ",\"sun_radius\":" << SUN_RADIUS << ",\"ego\":\"" << ego << "\",\"opp\":\"" << opps
           << "\",\"outcome\":" << outcome << ",\"ticks\":[\n";
        for (size_t i = 0; i < frames.size(); ++i) { jf << frames[i]; if (i + 1 < frames.size()) jf << ",\n"; }
        jf << "\n]}\n";
        jf.close();
        printf("wrote %s  (%zu frames)\n", record_json.c_str(), frames.size());
    }

    // capped channels + final return (mirrors collect())
    double Pc = std::max(-rc.prod_reward_cap, std::min(rc.prod_reward_cap, P));
    double Vc = std::max(0.0, std::min(rc.valid_reward_cap, V));
    double O = (stage >= 2) ? (outcome > 0 ? rc.win_bonus : (outcome < 0 ? -rc.loss_penalty : 0.0)) : 0.0;
    double Rfinal = O + Pc + Vc - I - M + S;

    // reward-to-go per step (dense, pre-cap): R_dense_total - cum_dense(t). A per-stage "value" proxy.
    double dense_total = rows.back().cumP + rows.back().cumV - rows.back().cumI - rows.back().cumM +
                         rows.back().cumS;
    std::ofstream csv(out);
    csv << "step,dprod_ego,dprod_enemy,launches,valid,miss,invalid,"
           "P_inc,V_inc,I_inc,M_inc,S_inc,cumP,cumV,cumI,cumM,cumS,cumR,rtg,"
           "ego_ships,enemy_ships,ship_margin,ego_prod,enemy_prod\n";
    for (const auto& r : rows) {
        double cumR = r.cumP + r.cumV - r.cumI - r.cumM + r.cumS;
        double rtg = dense_total - cumR;  // dense reward still to come from this step onward
        csv << r.step << ',' << r.dpe << ',' << r.dpn << ',' << r.lnch << ',' << r.valid << ','
            << r.miss << ',' << r.inval << ',' << r.pI << ',' << r.vI << ',' << r.iI << ',' << r.mI
            << ',' << r.sI << ',' << r.cumP << ',' << r.cumV << ',' << r.cumI << ',' << r.cumM << ','
            << r.cumS << ',' << cumR << ',' << rtg << ',' << r.es << ',' << r.ens << ','
            << (r.es - r.ens) << ',' << r.eprod << ',' << r.enprod << '\n';
    }
    csv.close();

    // --- stdout summary: per-channel totals + per-third attribution (see the decay/miss effect) ---
    long Ln = 0, Vl = 0, Ms = 0, Iv = 0;
    double thP[3] = {0}, thM[3] = {0}, thV[3] = {0};
    int n = (int)rows.size();
    for (int k = 0; k < n; ++k) {
        const auto& r = rows[k];
        Ln += r.lnch; Vl += r.valid; Ms += r.miss; Iv += r.inval;
        int third = std::min(2, (int)((long)k * 3 / n));
        thP[third] += r.pI; thM[third] += r.mI; thV[third] += r.vI;
    }
    printf("\n=== reward trace: ego=%s opp=%s stage=%d world=%d  (%d steps, outcome %+.0f) ===\n",
           ego.c_str(), opps.c_str(), stage, world_idx, n, outcome);
    printf("knobs: w_prod=%.3g cap=%.0f decay=%.4g | w_v=%.3g vcap=%.0f | rho=%.4g rho_m=%.4g\n",
           rc.prod_reward_weight, rc.prod_reward_cap, rc.prod_reward_decay, rc.valid_launch_reward,
           rc.valid_reward_cap, rc.illegal_launch_penalty, rc.miss_launch_penalty);
    printf("launches/st %.2f  valid/st %.2f  miss/st %.2f  invalid/st %.2f  (hit-rate %.0f%%)\n",
           (double)Ln / n, (double)Vl / n, (double)Ms / n, (double)Iv / n,
           Ln ? 100.0 * Vl / Ln : 0.0);
    printf("channels (capped): O=%+.1f  P=%+.1f (raw %+.1f)  V=%+.1f (raw %+.1f)  I=%-.1f  M=%-.1f  S=%+.1f\n",
           O, Pc, P, Vc, V, I, M, S);
    printf("RETURN R = %+.1f\n", Rfinal);
    printf("per-third  P: %+.1f %+.1f %+.1f   V: %+.1f %+.1f %+.1f   M: %-.1f %-.1f %-.1f\n",
           thP[0], thP[1], thP[2], thV[0], thV[1], thV[2], thM[0], thM[1], thM[2]);
    printf("wrote %s  (%d rows)\n", out.c_str(), n);
    return 0;
} catch (const std::exception& e) {
    fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
}
