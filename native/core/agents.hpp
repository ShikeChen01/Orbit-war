// Scripted opponents in C++ (mirror orbit_wars_rl/agents/scripted.py) for the
// non-learner seat during native rollouts.
#pragma once
#include <cmath>
#include <limits>
#include <random>

#include "core/state.hpp"

namespace ow {

// Half the garrison from each owned planet, random direction. Mirrors RandomAgent.
inline Action random_agent(const GameState& s, int player, std::mt19937_64& rng,
                           long min_ships = 20) {
    Action moves;
    std::uniform_real_distribution<double> ang(0.0, 2.0 * 3.141592653589793);
    for (const auto& p : s.planets) {
        if (p.owner == player && p.ships > 0) {
            long ships = p.ships / 2;
            if (ships >= min_ships) moves.push_back(Move{p.id, ang(rng), ships});
        }
    }
    return moves;
}

// Each owned planet sends half its ships at the nearest static, non-owned planet.
// Mirrors StarterAgent.
inline Action starter_agent(const GameState& s, int player, long min_ships = 20) {
    Action moves;
    for (const auto& mp : s.planets) {
        if (mp.owner != player || mp.ships <= 0) continue;
        const Planet* best = nullptr;
        double bestd = std::numeric_limits<double>::infinity();
        for (const auto& t : s.planets) {
            if (t.owner == player) continue;
            double orb = std::sqrt((t.x - CENTER) * (t.x - CENTER) + (t.y - CENTER) * (t.y - CENTER));
            if (orb + t.radius < ROTATION_RADIUS_LIMIT) continue;  // not static
            double d = std::sqrt((mp.x - t.x) * (mp.x - t.x) + (mp.y - t.y) * (mp.y - t.y));
            if (d < bestd) { bestd = d; best = &t; }
        }
        if (best) {
            long ships = mp.ships / 2;
            if (ships >= min_ships) {
                double angle = std::atan2(best->y - mp.y, best->x - mp.x);
                moves.push_back(Move{mp.id, angle, ships});
            }
        }
    }
    return moves;
}

}  // namespace ow
