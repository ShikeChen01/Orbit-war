// Reward / termination / scoring for the 2-player training regime, lifted out of the env so
// the rollout and the trainer share one definition. (4-player FFA would generalize potential()
// to a margin-over-strongest-rival and winner() to the max-score seat -- see docs/rl_math.pdf.)
#pragma once
#include "core/state.hpp"

namespace ow {

// Production-aware potential: ship margin + prod_weight * production margin (player 0 - 1).
inline double potential(const GameState& s, double prod_weight) {
    long s0 = 0, s1 = 0;
    double pr0 = 0.0, pr1 = 0.0;
    for (const auto& p : s.planets) {
        if (p.owner == 0) { s0 += p.ships; pr0 += p.production; }
        else if (p.owner == 1) { s1 += p.ships; pr1 += p.production; }
    }
    for (const auto& f : s.fleets) {
        if (f.owner == 0) s0 += f.ships;
        else if (f.owner == 1) s1 += f.ships;
    }
    return (double)(s0 - s1) + prod_weight * (pr0 - pr1);
}

inline bool is_terminal(const GameState& s, int episode_steps) {
    if (s.step >= episode_steps - 2) return true;
    bool a = false, b = false;
    for (const auto& p : s.planets) { if (p.owner == 0) a = true; else if (p.owner == 1) b = true; }
    for (const auto& f : s.fleets) { if (f.owner == 0) a = true; else if (f.owner == 1) b = true; }
    return !(a && b);  // <= 1 side alive
}

// +1 if player 0 has strictly more total ships at the end, -1 if fewer, 0 on a tie.
inline double outcome_sign(const GameState& s) {
    long s0 = 0, s1 = 0;
    for (const auto& p : s.planets) { if (p.owner == 0) s0 += p.ships; else if (p.owner == 1) s1 += p.ships; }
    for (const auto& f : s.fleets) { if (f.owner == 0) s0 += f.ships; else if (f.owner == 1) s1 += f.ships; }
    if (s0 == s1) return 0.0;
    return s0 > s1 ? 1.0 : -1.0;
}

}  // namespace ow
