// Continuous eval arena: play a trained continuous Agent (greedy) against a fixed scripted
// opponent (random / starter) in the bit-exact C++ engine WITH comets, scored like the
// competition (terminal total-ship margin; highest wins). Unlike rl/arena.hpp (which wraps the
// legacy discrete EntityPolicy twin) this drives the SAME PolicyNet used in training, via Agent,
// so there is no second network implementation to keep in sync. This is the GRPO fitness signal.
#pragma once
#include <torch/torch.h>

#include <execution>
#include <random>
#include <vector>

#include "core/agents.hpp"
#include "core/encode.hpp"
#include "core/sim.hpp"
#include "core/state.hpp"
#include "rl/model/agent.hpp"

namespace ow {

struct ArenaContResult {
    int p0_wins = 0, p1_wins = 0, draws = 0;
    double mean_margin = 0.0;  // mean (p0_score - p1_score) at game end
    double mean_len = 0.0;     // mean episode length in steps
};

class ArenaCont {
public:
    ArenaCont(int max_entities, int fleets_per_planet, double act_threshold, int episode_steps,
              torch::Device device)
        : E_(max_entities),
          K_(fleets_per_planet),
          act_thr_(act_threshold),
          episode_steps_(episode_steps),
          device_(device),
          cfg_(Config{episode_steps, 6.0, 4.0}) {}

    // Play every world once to completion; opponent 0=random, 1=starter. p0 plays greedily.
    ArenaContResult play(Agent& p0, const std::vector<GameState>& worlds, int opponent,
                         uint64_t seed) {
        ArenaContResult R;
        int B = (int)worlds.size();
        if (B == 0) return R;
        const int F = N_ENTITY_FEATURES, G = N_GLOBAL_FEATURES, nap = 3 * K_;  // (dx,dy,phi) per fleet

        std::vector<GameState> st = worlds;
        std::vector<char> done(B, 0);
        std::vector<int> margin(B, 0), length(B, episode_steps_);
        std::vector<std::mt19937_64> rng(B);
        std::vector<int> idx(B);
        for (int i = 0; i < B; ++i) { rng[i].seed(seed + 1234567ull * (i + 1)); idx[i] = i; }
        std::vector<std::vector<long>> rid(B, std::vector<long>(E_, -1)),
            rsh(B, std::vector<long>(E_, 0));

        int remaining = B;
        for (int t = 0; t < episode_steps_ && remaining > 0; ++t) {
            std::vector<float> ent((size_t)B * E_ * F, 0.f), em((size_t)B * E_, 0.f),
                am((size_t)B * E_, 0.f), gl((size_t)B * G, 0.f);
            std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
                if (done[i]) return;
                encode_obs(st[i], 0, E_, ent.data() + (size_t)i * E_ * F,
                           em.data() + (size_t)i * E_, am.data() + (size_t)i * E_,
                           gl.data() + (size_t)i * G, rid[i].data(), rsh[i].data(), episode_steps_);
            });
            auto fo = torch::TensorOptions().dtype(torch::kFloat32);
            auto t_ent = torch::from_blob(ent.data(), {B, E_, F}, fo).to(device_);
            auto t_em = torch::from_blob(em.data(), {B, E_}, fo).to(device_);
            auto t_am = torch::from_blob(am.data(), {B, E_}, fo).to(device_);
            auto t_gl = torch::from_blob(gl.data(), {B, G}, fo).to(device_);
            auto act = p0.act(t_ent, t_em, t_am, t_gl, /*greedy=*/true).action.to(torch::kCPU).contiguous();
            const float* ad = act.data_ptr<float>();

            std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
                if (done[i]) return;
                GameState& s = st[i];
                std::vector<float> am0(E_);
                for (int r = 0; r < E_; ++r) am0[r] = am[(size_t)i * E_ + r];
                Action m0 = decode_action_continuous(ad + (size_t)i * E_ * nap, am0.data(),
                                                     rid[i].data(), rsh[i].data(), E_, K_, act_thr_);
                Action m1 = (opponent == 0) ? random_agent(s, 1, rng[i]) : starter_agent(s, 1);
                std::vector<Action> acts = {m0, m1};
                ow::step(s, acts, cfg_);
                s.step += 1;
                if (terminal(s)) { done[i] = 1; margin[i] = score_margin(s); length[i] = s.step; }
            });
            remaining = 0;
            for (int i = 0; i < B; ++i) if (!done[i]) remaining++;
        }

        long sum_margin = 0, sum_len = 0;
        for (int i = 0; i < B; ++i) {
            if (!done[i]) { margin[i] = score_margin(st[i]); length[i] = st[i].step; }
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
    int E_, K_;
    double act_thr_;
    int episode_steps_;
    torch::Device device_;
    Config cfg_;

    static int score_margin(const GameState& s) {
        long s0 = 0, s1 = 0;
        for (const auto& p : s.planets) { if (p.owner == 0) s0 += p.ships; else if (p.owner == 1) s1 += p.ships; }
        for (const auto& f : s.fleets) { if (f.owner == 0) s0 += f.ships; else if (f.owner == 1) s1 += f.ships; }
        return (int)(s0 - s1);
    }
    bool terminal(const GameState& s) const {
        if (s.step >= episode_steps_ - 2) return true;
        bool p0 = false, p1 = false;
        for (const auto& p : s.planets) { if (p.owner == 0) p0 = true; else if (p.owner == 1) p1 = true; }
        for (const auto& f : s.fleets) { if (f.owner == 0) p0 = true; else if (f.owner == 1) p1 = true; }
        return !(p0 && p1);
    }
};

}  // namespace ow
