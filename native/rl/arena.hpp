// Fast, faithful native arena: play a trained EntityPolicy against a fixed opponent
// (random / starter / a second policy) in the bit-exact C++ engine WITH comets, scoring
// exactly like the competition (terminal total-ship score; highest wins). This is the
// project's fitness function -- ~50x faster than the Python kaggle arena and identical in
// outcome (parity-proven step + matching scoring/termination), so every architecture and
// training change can be measured in seconds.
#pragma once
#include <torch/torch.h>

#include <execution>
#include <random>
#include <tuple>
#include <vector>

#include "core/agents.hpp"
#include "core/encode.hpp"
#include "core/sim.hpp"
#include "core/state.hpp"
#include "rl/policy.hpp"

namespace ow {

struct ArenaResult {
    int p0_wins = 0, p1_wins = 0, draws = 0;
    double mean_margin = 0.0;   // mean (p0_score - p1_score) at game end
    double mean_len = 0.0;      // mean episode length in steps
};

class Arena {
public:
    torch::Device device;
    int max_entities, angle_bins, episode_steps, hidden;
    int actions_per_entity, n_entity_features, n_global_features;
    bool target_mode;
    std::vector<double> fractions;
    Config cfg;
    EntityPolicy pol0{nullptr}, pol1{nullptr};
    bool have_pol1 = false;

    Arena(int max_entities_, int angle_bins_, std::vector<double> fracs, int hidden_,
          int episode_steps_, const std::string& dev, bool target_mode_ = false)
        : device(dev == "cuda" && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          max_entities(max_entities_), angle_bins(angle_bins_), episode_steps(episode_steps_),
          hidden(hidden_), target_mode(target_mode_), fractions(std::move(fracs)),
          cfg(Config{episode_steps_, 6.0, 4.0}) {
        n_entity_features = N_ENTITY_FEATURES;
        n_global_features = N_GLOBAL_FEATURES;
        int nf = (int)fractions.size();
        actions_per_entity = target_mode ? 1 + max_entities * nf : 1 + angle_bins * nf;
    }

    void load_p0(const std::map<std::string, torch::Tensor>& sd) {
        pol0 = EntityPolicy(n_entity_features, n_global_features, actions_per_entity, hidden,
                            target_mode, (int)fractions.size());
        load_python_state_dict(pol0, sd);
        pol0->to(device); pol0->eval();
    }
    void load_p1(const std::map<std::string, torch::Tensor>& sd) {
        pol1 = EntityPolicy(n_entity_features, n_global_features, actions_per_entity, hidden,
                            target_mode, (int)fractions.size());
        load_python_state_dict(pol1, sd);
        pol1->to(device); pol1->eval();
        have_pol1 = true;
    }

    // Play every world once to completion. opponent: 0=random, 1=starter, 2=policy(p1).
    // p0 always uses pol0. Returns aggregate win/loss/draw + mean score margin.
    ArenaResult play(std::vector<GameState> worlds, int opponent, bool deterministic,
                     uint64_t seed) {
        int B = (int)worlds.size();
        ArenaResult R;
        if (B == 0) return R;

        std::vector<GameState> st = std::move(worlds);
        std::vector<char> done(B, 0);
        std::vector<int> margin(B, 0);
        std::vector<int> length(B, episode_steps);
        std::vector<std::mt19937_64> rng(B);
        for (int i = 0; i < B; ++i) rng[i].seed(seed + 1234567ull * (i + 1));

        std::vector<int> idx(B);
        for (int i = 0; i < B; ++i) idx[i] = i;
        std::vector<std::vector<long>> rid0(B, std::vector<long>(max_entities, -1));
        std::vector<std::vector<long>> rsh0(B, std::vector<long>(max_entities, 0));
        std::vector<std::vector<long>> rid1(B, std::vector<long>(max_entities, -1));
        std::vector<std::vector<long>> rsh1(B, std::vector<long>(max_entities, 0));

        int remaining = B;
        for (int t = 0; t < episode_steps && remaining > 0; ++t) {
            auto a0 = policy_actions(pol0, st, done, 0, rid0, rsh0, deterministic);
            torch::Tensor a1;
            if (opponent == 2 && have_pol1)
                a1 = policy_actions(pol1, st, done, 1, rid1, rsh1, deterministic);
            auto a0acc = a0.accessor<int64_t, 2>();
            torch::TensorAccessor<int64_t, 2> a1acc =
                (a1.defined() ? a1.accessor<int64_t, 2>() : a0.accessor<int64_t, 2>());

            std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
                if (done[i]) return;
                GameState& s = st[i];
                // player 0 (the policy under test)
                std::vector<long> cls0(max_entities);
                std::vector<float> am0(max_entities);
                for (int r = 0; r < max_entities; ++r) {
                    cls0[r] = a0acc[i][r];
                    am0[r] = (rid0[i][r] >= 0 && owned_with_ships(s, 0, rid0[i][r])) ? 1.f : 0.f;
                }
                Action m0 = decode_mv(s, cls0.data(), am0.data(), rid0[i].data(), rsh0[i].data());
                // player 1 (opponent)
                Action m1;
                if (opponent == 0) {
                    m1 = random_agent(s, 1, rng[i]);
                } else if (opponent == 1) {
                    m1 = starter_agent(s, 1);
                } else if (opponent == 2 && have_pol1) {
                    std::vector<long> cls1(max_entities);
                    std::vector<float> am1(max_entities);
                    for (int r = 0; r < max_entities; ++r) {
                        cls1[r] = a1acc[i][r];
                        am1[r] = (rid1[i][r] >= 0 && owned_with_ships(s, 1, rid1[i][r])) ? 1.f : 0.f;
                    }
                    m1 = decode_mv(s, cls1.data(), am1.data(), rid1[i].data(), rsh1[i].data());
                }
                std::vector<Action> acts = {m0, m1};
                ow::step(s, acts, cfg);
                s.step += 1;
                if (is_terminal(s)) {
                    done[i] = 1;
                    margin[i] = score_margin(s);
                    length[i] = s.step;
                }
            });
            // tally newly-finished games
            remaining = 0;
            for (int i = 0; i < B; ++i) if (!done[i]) remaining++;
        }

        long sum_margin = 0, sum_len = 0;
        for (int i = 0; i < B; ++i) {
            if (!done[i]) { margin[i] = score_margin(st[i]); length[i] = st[i].step; }  // hit cap
            sum_margin += margin[i];
            sum_len += length[i];
            if (margin[i] > 0) R.p0_wins++;
            else if (margin[i] < 0) R.p1_wins++;
            else R.draws++;
        }
        R.mean_margin = (double)sum_margin / B;
        R.mean_len = (double)sum_len / B;
        return R;
    }

private:
    Action decode_mv(const GameState& s, const long* cls, const float* am, const long* rid,
                     const long* rsh) const {
        if (target_mode)
            return decode_action_target(s, cls, am, rid, rsh, max_entities,
                                        (int)fractions.size(), fractions);
        return decode_action(cls, am, rid, rsh, max_entities, angle_bins, fractions);
    }
    static bool owned_with_ships(const GameState& s, int player, long pid) {
        for (const auto& p : s.planets)
            if (p.id == pid) return p.owner == player && p.ships > 0;
        return false;
    }
    static int score_margin(const GameState& s) {
        long s0 = 0, s1 = 0;
        for (const auto& p : s.planets) { if (p.owner == 0) s0 += p.ships; else if (p.owner == 1) s1 += p.ships; }
        for (const auto& f : s.fleets) { if (f.owner == 0) s0 += f.ships; else if (f.owner == 1) s1 += f.ships; }
        return (int)(s0 - s1);
    }
    bool is_terminal(const GameState& s) const {
        if (s.step >= episode_steps - 2) return true;
        bool p0 = false, p1 = false;
        for (const auto& p : s.planets) { if (p.owner == 0) p0 = true; else if (p.owner == 1) p1 = true; }
        for (const auto& f : s.fleets) { if (f.owner == 0) p0 = true; else if (f.owner == 1) p1 = true; }
        return !(p0 && p1);
    }

    // Encode all live envs for `player`, run the policy, return greedy/sampled classes
    // (B,E) int64 on CPU. Caches row->planet id/ships for action decode.
    torch::Tensor policy_actions(EntityPolicy& pol, const std::vector<GameState>& st,
                                 const std::vector<char>& done, int player,
                                 std::vector<std::vector<long>>& rid,
                                 std::vector<std::vector<long>>& rsh, bool deterministic) {
        int B = (int)st.size();
        size_t F = N_ENTITY_FEATURES, G = N_GLOBAL_FEATURES;
        std::vector<float> ent((size_t)B * max_entities * F, 0.f), em((size_t)B * max_entities, 0.f),
            am((size_t)B * max_entities, 0.f), gl((size_t)B * G, 0.f);
        std::vector<int> idx(B);
        for (int i = 0; i < B; ++i) idx[i] = i;
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
            if (done[i]) return;
            encode_obs(st[i], player, max_entities, ent.data() + (size_t)i * max_entities * F,
                       em.data() + (size_t)i * max_entities, am.data() + (size_t)i * max_entities,
                       gl.data() + (size_t)i * G, rid[i].data(), rsh[i].data(), episode_steps);
        });
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto t_ent = torch::from_blob(ent.data(), {B, max_entities, (long)F}, opts).to(device);
        auto t_em = torch::from_blob(em.data(), {B, max_entities}, opts).to(device);
        auto t_am = torch::from_blob(am.data(), {B, max_entities}, opts).to(device);
        auto t_gl = torch::from_blob(gl.data(), {B, (long)G}, opts).to(device);
        torch::NoGradGuard ng;
        auto [logits, val] = pol->forward(t_ent, t_em, t_am, t_gl);
        auto a = deterministic ? greedy_actions(logits) : sample_actions(logits);
        return a.to(torch::kCPU).contiguous();
    }
};

}  // namespace ow
