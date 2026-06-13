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

// Stage-1 (solo) potential: reward owning planets + production + (a little) ships, from
// `player`'s view ONLY -- the opponent is passive, so there is no margin to subtract. The
// per-step delta of this, discounted by gamma<1, makes "capture the whole map as fast as
// possible" optimal: a capture is a big immediate jump (cap_weight + prod_weight*prod),
// and earlier jumps are worth more under the discount (the takeover-time factor). The small
// ship_log term keeps ship growth a tie-breaker, not a hoarding incentive. Training-only
// (never in the serving path), so it needs no Python parity.
inline double solo_potential(const GameState& s, int player, double cap_w, double prod_w,
                             double ship_w) {
    long ships = 0;
    double prod = 0.0;
    int planets = 0;
    for (const auto& p : s.planets)
        if (p.owner == player) { ships += p.ships; prod += p.production; planets++; }
    for (const auto& f : s.fleets)
        if (f.owner == player) ships += f.ships;
    double slog = std::log1p((double)std::max<long>(0, ships)) / std::log(1000.0);
    return cap_w * (double)planets + prod_w * prod + ship_w * slog;
}

// Fraction of live planets owned by `player` (stage-1 graded outcome: 1.0 == conquered all).
inline double planet_fraction_owned(const GameState& s, int player) {
    int mine = 0, total = 0;
    for (const auto& p : s.planets) { total++; if (p.owner == player) mine++; }
    return total > 0 ? (double)mine / (double)total : 0.0;
}

// Total production of `player`'s owned planets -- the only quantity the dense reward rewards
// growing (a capture raises it by the captured planet's production; a loss lowers it). It is
// read directly from the environment each step.
inline double my_production(const GameState& s, int player) {
    double pr = 0.0;
    for (const auto& p : s.planets)
        if (p.owner == player) pr += p.production;
    return pr;
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
