#include "rl/rollout.hpp"

#include <algorithm>
#include <execution>

#include "core/agents.hpp"
#include "core/encode.hpp"
#include "core/sim.hpp"
#include "rl/algo/grpo.hpp"
#include "rl/reward.hpp"

namespace ow {

GroupedRollout::GroupedRollout(const TrainConfig& cfg, torch::Device device)
    : cfg_(cfg), device_(device), rng_(cfg.seed * 2654435761ull + 12345ull) {}

TrajectoryBatch GroupedRollout::collect(Agent& policy, RolloutStats& stats) {
    const int G = cfg_.grpo.group_size, N = cfg_.grpo.num_groups, B = N * G;
    const int E = cfg_.model.max_entities, F = N_ENTITY_FEATURES, Gl = N_GLOBAL_FEATURES;
    const int T = cfg_.rollout.episode_steps, nf = (int)cfg_.model.fractions.size();
    const double pw = cfg_.rollout.prod_weight, rscale = cfg_.rollout.reward_scale,
                 rclip = cfg_.rollout.reward_clip, gamma = cfg_.grpo.gamma;
    Config econf{cfg_.rollout.episode_steps, 6.0, 4.0};

    // --- assign one (world, opponent) per group; all G envs in a group share them ---
    std::vector<GameState> st(B);
    std::vector<int> opp(B);  // 0 random, 1 starter
    std::vector<std::mt19937_64> erng(B);
    {
        double wr = cfg_.selfplay.mix_random, ws = cfg_.selfplay.mix_starter;
        std::uniform_real_distribution<double> u(0.0, wr + ws);
        for (int j = 0; j < N; ++j) {
            const GameState& world = world_pool_[cursor_++ % world_pool_.size()];
            int o = (u(rng_) < wr) ? 0 : 1;
            for (int g = 0; g < G; ++g) {
                int i = j * G + g;
                st[i] = world;
                opp[i] = o;
                erng[i].seed(rng_());
            }
        }
    }

    std::vector<char> active(B, 1);
    std::vector<double> dense_ret(B, 0.0), gpow(B, 1.0), prev_pot(B), outcome(B, 0.0);
    std::vector<int> ep_len(B, T);
    for (int i = 0; i < B; ++i) prev_pot[i] = potential(st[i], pw);

    auto fo = torch::TensorOptions().dtype(torch::kFloat32);
    auto lo = torch::TensorOptions().dtype(torch::kLong);
    auto ent_buf = torch::zeros({T, B, E, F}, fo);
    auto em_buf = torch::zeros({T, B, E}, fo);
    auto am_buf = torch::zeros({T, B, E}, fo);
    auto gl_buf = torch::zeros({T, B, Gl}, fo);
    auto act_buf = torch::zeros({T, B, E}, lo);
    auto oldlogp_buf = torch::zeros({T, B}, fo);
    auto valid_buf = torch::zeros({T, B}, fo);

    std::vector<int> idx(B);
    for (int i = 0; i < B; ++i) idx[i] = i;
    std::vector<std::vector<long>> rid(B, std::vector<long>(E, -1)), rsh(B, std::vector<long>(E, 0));

    int last_t = 0;
    for (int t = 0; t < T; ++t) {
        last_t = t;
        std::vector<float> ent((size_t)B * E * F, 0.f), em((size_t)B * E, 0.f),
            am((size_t)B * E, 0.f), gl((size_t)B * Gl, 0.f);
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
            encode_obs(st[i], 0, E, ent.data() + (size_t)i * E * F, em.data() + (size_t)i * E,
                       am.data() + (size_t)i * E, gl.data() + (size_t)i * Gl, rid[i].data(),
                       rsh[i].data(), cfg_.rollout.episode_steps);
        });
        auto t_ent = torch::from_blob(ent.data(), {B, E, F}, fo).clone();
        auto t_em = torch::from_blob(em.data(), {B, E}, fo).clone();
        auto t_am = torch::from_blob(am.data(), {B, E}, fo).clone();
        auto t_gl = torch::from_blob(gl.data(), {B, Gl}, fo).clone();
        ent_buf[t] = t_ent;
        em_buf[t] = t_em;
        am_buf[t] = t_am;
        gl_buf[t] = t_gl;

        auto ent_d = t_ent.to(device_), em_d = t_em.to(device_), am_d = t_am.to(device_),
             gl_d = t_gl.to(device_);
        auto dec = policy.act(ent_d, em_d, am_d, gl_d, false);
        act_buf[t] = dec.action.to(torch::kCPU);
        oldlogp_buf[t] = dec.log_prob.to(torch::kCPU);

        auto act_t = act_buf[t];  // named lvalue; accessor can't bind to a temporary
        auto a0 = act_t.accessor<int64_t, 2>();
        std::vector<float> validv(B, 0.f);
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
            if (!active[i]) return;
            validv[i] = 1.f;
            std::vector<long> cls0(E);
            std::vector<float> am0(E);
            for (int r = 0; r < E; ++r) {
                cls0[r] = a0[i][r];
                am0[r] = am[(size_t)i * E + r];
            }
            Action move0 =
                cfg_.model.target_mode
                    ? decode_action_target(st[i], cls0.data(), am0.data(), rid[i].data(),
                                           rsh[i].data(), E, nf, cfg_.model.fractions)
                    : decode_action(cls0.data(), am0.data(), rid[i].data(), rsh[i].data(), E,
                                    cfg_.model.angle_bins, cfg_.model.fractions);
            Action move1 = (opp[i] == 0) ? random_agent(st[i], 1, erng[i]) : starter_agent(st[i], 1);
            std::vector<Action> acts = {move0, move1};
            ow::step(st[i], acts, econf);
            st[i].step += 1;
            double pot = potential(st[i], pw);
            double dense = std::max(-rclip, std::min(rclip, (pot - prev_pot[i]) / rscale));
            dense_ret[i] += gpow[i] * dense;
            gpow[i] *= gamma;
            prev_pot[i] = pot;
            if (is_terminal(st[i], cfg_.rollout.episode_steps)) {
                active[i] = 0;
                outcome[i] = outcome_sign(st[i]);
                ep_len[i] = st[i].step;
            }
        });
        valid_buf[t] = torch::from_blob(validv.data(), {B}, fo).clone();
        if (std::none_of(active.begin(), active.end(), [](char c) { return c != 0; })) break;
    }
    for (int i = 0; i < B; ++i)
        if (active[i]) outcome[i] = outcome_sign(st[i]);  // hit the step cap

    // --- per-episode return + group-relative advantage ---
    std::vector<float> Rv(B);
    for (int i = 0; i < B; ++i)
        Rv[i] = (float)(cfg_.grpo.dense_weight * dense_ret[i] +
                        cfg_.grpo.outcome_weight * outcome[i]);
    auto R = torch::from_blob(Rv.data(), {B}, fo).clone();
    std::vector<int64_t> gidv(B);
    for (int i = 0; i < B; ++i) gidv[i] = i / G;
    auto gid = torch::from_blob(gidv.data(), {B}, lo).clone();
    auto adv_env = group_advantage(R, gid, N, cfg_.grpo.adv_eps, cfg_.grpo.whiten_advantage);

    // --- flatten valid transitions ---
    int Tu = last_t + 1;
    auto keep = valid_buf.narrow(0, 0, Tu).reshape({-1}).nonzero().squeeze(-1);  // (Nkeep,)
    auto adv_full = adv_env.unsqueeze(0).expand({Tu, B}).reshape({-1});
    auto sel = [&](torch::Tensor x) { return x.index_select(0, keep); };

    TrajectoryBatch tb;
    tb.entities = sel(ent_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, E, F}));
    tb.entity_mask = sel(em_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, E}));
    tb.action_mask = sel(am_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, E}));
    tb.globals = sel(gl_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, Gl}));
    tb.action = sel(act_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, E}));
    tb.old_logp = sel(oldlogp_buf.narrow(0, 0, Tu).reshape({(long)Tu * B}));
    tb.advantage = adv_full.index_select(0, keep);  // ref_logp computed in the update
    tb.n_transitions = keep.size(0);

    double sumR = 0, sumlen = 0, wins = 0;
    for (int i = 0; i < B; ++i) {
        sumR += Rv[i];
        sumlen += ep_len[i];
        wins += outcome[i] > 0 ? 1.0 : 0.0;
    }
    stats.mean_return = sumR / B;
    stats.mean_len = sumlen / B;
    stats.win_rate = wins / B;
    stats.episodes = B;
    stats.transitions = tb.n_transitions;
    return tb;
}

}  // namespace ow
