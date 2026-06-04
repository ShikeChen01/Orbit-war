// Batched Orbit Wars env (2-player) for native PPO. Steps B independent games in
// parallel, encodes observations to torch tensors, computes the dense shaping reward
// (mirrors OrbitWarsEnv), and resets finished games from a world pool. Internal to the
// Trainer -- never crosses the Python boundary, so it uses torch::Tensor freely.
#pragma once
#include <torch/torch.h>

#include <execution>
#include <random>
#include <vector>

#include "core/agents.hpp"
#include "core/encode.hpp"
#include "core/sim.hpp"
#include "core/state.hpp"

namespace ow {

struct ObsTensors {
    torch::Tensor entities, entity_mask, action_mask, globals;  // CPU float32
};

enum OppKind { OPP_RANDOM = 0, OPP_STARTER = 1, OPP_EXTERNAL = 2 };

class BatchedEnv {
public:
    int B, max_entities, angle_bins, episode_steps, num_agents;
    std::vector<double> fractions;
    Config cfg;
    double reward_scale, terminal_bonus, reward_clip, prod_weight;

    BatchedEnv(int batch, int max_entities_, int angle_bins_, std::vector<double> fracs,
               Config cfg_, int episode_steps_, double reward_scale_ = 50.0,
               double terminal_bonus_ = 1.0, double reward_clip_ = 5.0, double prod_weight_ = 0.0,
               uint64_t seed = 0)
        : B(batch), max_entities(max_entities_), angle_bins(angle_bins_),
          episode_steps(episode_steps_), num_agents(2), fractions(std::move(fracs)), cfg(cfg_),
          reward_scale(reward_scale_), terminal_bonus(terminal_bonus_), reward_clip(reward_clip_),
          prod_weight(prod_weight_) {
        states.resize(B);
        prev_margin.resize(B, 0.0);
        opp_kind.resize(B, OPP_STARTER);
        rng.resize(B);
        for (int i = 0; i < B; ++i) rng[i].seed(seed + 1234567u * (i + 1));
        for (int pl = 0; pl < 2; ++pl) {
            row_pid[pl].assign(B, std::vector<long>(max_entities, -1));
            row_ships[pl].assign(B, std::vector<long>(max_entities, 0));
        }
        idx_.resize(B);
        for (int i = 0; i < B; ++i) idx_[i] = i;
    }

    void set_world_pool(std::vector<GameState> pool) { world_pool = std::move(pool); pool_cursor = 0; }
    void set_opp_weights(double w_random, double w_starter, double w_self) {
        ow_random = w_random; ow_starter = w_starter; ow_self = w_self;
    }
    void set_selfplay_available(bool avail) { selfplay_available = avail; }
    void set_target_mode(bool t) { target_mode = t; }

    // Decode a per-entity class vector into a move list (target-based or angle-bin).
    Action decode(const GameState& s, const long* cls, const float* am, const long* rid,
                  const long* rsh) const {
        if (target_mode)
            return decode_action_target(s, cls, am, rid, rsh, max_entities,
                                        (int)fractions.size(), fractions);
        return decode_action(cls, am, rid, rsh, max_entities, angle_bins, fractions);
    }

    void reset_all() {
        for (int i = 0; i < B; ++i) load_world(i);
    }

    // Encode all envs for `player`; caches row->planet mappings for action decode.
    ObsTensors observe(int player) {
        size_t F = N_ENTITY_FEATURES, G = N_GLOBAL_FEATURES;
        std::vector<float> ent((size_t)B * max_entities * F), em((size_t)B * max_entities),
            am((size_t)B * max_entities), gl((size_t)B * G);
        std::for_each(std::execution::par, idx_.begin(), idx_.end(), [&](int i) {
            encode_obs(states[i], player, max_entities, ent.data() + (size_t)i * max_entities * F,
                       em.data() + (size_t)i * max_entities, am.data() + (size_t)i * max_entities,
                       gl.data() + (size_t)i * G, row_pid[player][i].data(),
                       row_ships[player][i].data(), episode_steps);
        });
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        ObsTensors o;
        o.entities = torch::from_blob(ent.data(), {B, max_entities, (long)F}, opts).clone();
        o.entity_mask = torch::from_blob(em.data(), {B, max_entities}, opts).clone();
        o.action_mask = torch::from_blob(am.data(), {B, max_entities}, opts).clone();
        o.globals = torch::from_blob(gl.data(), {B, (long)G}, opts).clone();
        return o;
    }

    // Advance all envs. p0_action (B,E) int64 CPU. opp_action used only for OPP_EXTERNAL envs.
    // Returns {reward (B,) float, done (B,) bool} CPU.
    std::pair<torch::Tensor, torch::Tensor> step(const torch::Tensor& p0_action,
                                                 const torch::Tensor& opp_action) {
        auto p0 = p0_action.to(torch::kCPU).contiguous();
        auto a0 = p0.accessor<int64_t, 2>();
        bool have_opp = opp_action.defined() && opp_action.numel() > 0;
        torch::Tensor opp_cpu;
        if (have_opp) opp_cpu = opp_action.to(torch::kCPU).contiguous();

        std::vector<float> reward(B, 0.f);
        std::vector<uint8_t> done(B, 0);

        std::for_each(std::execution::par, idx_.begin(), idx_.end(), [&](int i) {
            GameState& s = states[i];
            // Decode learner (player 0) action.
            std::vector<long> cls0(max_entities);
            for (int r = 0; r < max_entities; ++r) cls0[r] = a0[i][r];
            std::vector<float> am0(max_entities);
            for (int r = 0; r < max_entities; ++r)
                am0[r] = (row_pid[0][i][r] >= 0 && states[i].planets.size() &&
                          owned_with_ships(s, 0, row_pid[0][i][r])) ? 1.f : 0.f;
            Action move0 = decode(s, cls0.data(), am0.data(), row_pid[0][i].data(),
                                  row_ships[0][i].data());
            // Opponent (player 1).
            Action move1;
            if (opp_kind[i] == OPP_RANDOM) {
                move1 = random_agent(s, 1, rng[i]);
            } else if (opp_kind[i] == OPP_STARTER) {
                move1 = starter_agent(s, 1);
            } else if (have_opp) {  // OPP_EXTERNAL self-play
                auto a1 = opp_cpu.accessor<int64_t, 2>();
                std::vector<long> cls1(max_entities);
                for (int r = 0; r < max_entities; ++r) cls1[r] = a1[i][r];
                std::vector<float> am1(max_entities);
                for (int r = 0; r < max_entities; ++r)
                    am1[r] = (row_pid[1][i][r] >= 0 && owned_with_ships(s, 1, row_pid[1][i][r])) ? 1.f : 0.f;
                move1 = decode(s, cls1.data(), am1.data(), row_pid[1][i].data(),
                               row_ships[1][i].data());
            }

            std::vector<Action> acts = {move0, move1};
            ow::step(s, acts, cfg);
            s.step += 1;

            double margin = potential(s);
            double dense = (margin - prev_margin[i]) / reward_scale;
            dense = std::max(-reward_clip, std::min(reward_clip, dense));
            prev_margin[i] = margin;

            bool term = terminal(s);
            if (term) {
                int w = winner(s);
                dense += terminal_bonus * (w == 0 ? 1.0 : -1.0);
            }
            reward[i] = (float)dense;
            done[i] = term ? 1 : 0;
        });

        auto r = torch::from_blob(reward.data(), {B}, torch::kFloat32).clone();
        auto d = torch::from_blob(done.data(), {B}, torch::kUInt8).clone().to(torch::kBool);
        return {r, d};
    }

    // Reload finished envs from the pool. Call after reading the step's reward/done.
    void reset_done(const torch::Tensor& done) {
        auto dcpu = done.to(torch::kCPU).contiguous();
        auto da = dcpu.accessor<bool, 1>();
        for (int i = 0; i < B; ++i)
            if (da[i]) load_world(i);
    }

    int winrate_window_wins = 0, winrate_window_games = 0;  // simple running stats
    double mean_return = 0.0;

private:
    std::vector<GameState> states;
    std::vector<double> prev_margin;
    std::vector<int> opp_kind;
    std::vector<std::mt19937_64> rng;
    std::vector<int> idx_;
    std::vector<std::vector<long>> row_pid[2];
    std::vector<std::vector<long>> row_ships[2];
    std::vector<GameState> world_pool;
    size_t pool_cursor = 0;
    double ow_random = 1.0, ow_starter = 1.0, ow_self = 0.0;
    bool selfplay_available = false;
    bool target_mode = false;

    static bool owned_with_ships(const GameState& s, int player, long pid) {
        for (const auto& p : s.planets)
            if (p.id == pid) return p.owner == player && p.ships > 0;
        return false;
    }

    double score_margin(const GameState& s) const {
        long s0 = 0, s1 = 0;
        for (const auto& p : s.planets) {
            if (p.owner == 0) s0 += p.ships; else if (p.owner == 1) s1 += p.ships;
        }
        for (const auto& f : s.fleets) { if (f.owner == 0) s0 += f.ships; else if (f.owner == 1) s1 += f.ships; }
        return (double)(s0 - s1);
    }

    // Reward potential: ship margin + prod_weight * production margin. Adding the
    // production advantage makes *capturing a planet* immediately rewarding (it raises
    // your production this step), which counteracts the short-term ship cost of
    // expansion -- the local optimum that made the ship-only reward collapse to passivity.
    double potential(const GameState& s) const {
        long s0 = 0, s1 = 0;
        double pr0 = 0.0, pr1 = 0.0;
        for (const auto& p : s.planets) {
            if (p.owner == 0) { s0 += p.ships; pr0 += p.production; }
            else if (p.owner == 1) { s1 += p.ships; pr1 += p.production; }
        }
        for (const auto& f : s.fleets) { if (f.owner == 0) s0 += f.ships; else if (f.owner == 1) s1 += f.ships; }
        return (double)(s0 - s1) + prod_weight * (pr0 - pr1);
    }

    bool terminal(const GameState& s) const {
        if (s.step >= episode_steps - 2) return true;
        bool p0alive = false, p1alive = false;
        for (const auto& p : s.planets) { if (p.owner == 0) p0alive = true; else if (p.owner == 1) p1alive = true; }
        for (const auto& f : s.fleets) { if (f.owner == 0) p0alive = true; else if (f.owner == 1) p1alive = true; }
        return !(p0alive && p1alive);  // <=1 alive
    }

    int winner(const GameState& s) const {
        long s0 = 0, s1 = 0;
        for (const auto& p : s.planets) { if (p.owner == 0) s0 += p.ships; else if (p.owner == 1) s1 += p.ships; }
        for (const auto& f : s.fleets) { if (f.owner == 0) s0 += f.ships; else if (f.owner == 1) s1 += f.ships; }
        if (s0 == s1) return -1;
        return s0 > s1 ? 0 : 1;
    }

    void load_world(int i) {
        if (world_pool.empty()) return;
        states[i] = world_pool[pool_cursor % world_pool.size()];
        pool_cursor++;
        prev_margin[i] = potential(states[i]);
        // Resample opponent kind.
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double total = ow_random + ow_starter + (selfplay_available ? ow_self : 0.0);
        double x = u(rng[i]) * total;
        if (x < ow_random) opp_kind[i] = OPP_RANDOM;
        else if (x < ow_random + ow_starter) opp_kind[i] = OPP_STARTER;
        else opp_kind[i] = OPP_EXTERNAL;
    }
};

}  // namespace ow
