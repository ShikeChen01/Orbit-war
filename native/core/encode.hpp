// Observation encoder + action decoder + scripted opponents, mirroring the Python
// processors (EntityObservation / PerPlanetAction) and scripted agents EXACTLY.
// Torch-free: fills caller-provided float/long buffers. Parity-checked in tests.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "core/state.hpp"

namespace ow {

constexpr int N_ENTITY_FEATURES = 20;  // 15 base + 5 incoming-fleet threat features
constexpr int N_GLOBAL_FEATURES = 10;
inline const double SHIP_LOG_DENOM = std::log(1000.0);
inline const double DIAG_HALF = std::sqrt(BOARD_SIZE * BOARD_SIZE + BOARD_SIZE * BOARD_SIZE) / 2.0;
constexpr double PRESSURE_RADIUS = 25.0;
constexpr double PI = 3.141592653589793;  // == Python math.pi
constexpr double THREAT_MAX_SPEED = 6.0;  // == Config.shipSpeed; fleet speed cap for the ETA model
constexpr double THREAT_ETA_TAU = 8.0;    // imminence = exp(-eta/tau); ~half-life of 5-6 ticks
constexpr double THREAT_MARGIN_SCALE = 100.0;  // hold-margin -> tanh divisor

inline double ship_log(double n) {
    return std::log1p(std::max(0.0, n)) / SHIP_LOG_DENOM;
}

// Fleet ship-speed (mirrors sim.hpp fleet movement): larger fleets move faster, capped.
inline double fleet_speed(long ships) {
    double n = (double)std::max<long>(1, ships);
    double v = 1.0 + (THREAT_MAX_SPEED - 1.0) * std::pow(std::log(n) / std::log(1000.0), 1.5);
    return std::min(v, THREAT_MAX_SPEED);
}

// Closed-form straight-line threat projection: the planet INDEX this fleet first reaches and
// the ETA in ticks, or {-1, 0} if it crosses the sun or hits nothing ahead. This approximates
// the swept collision (it ignores planet rotation and re-launch self-hits) but is computed
// identically on the Python side, so the observation stays parity-exact. It is used ONLY for
// the threat features -- never for game mechanics, which remain the exact ported sim.
inline std::pair<int, double> fleet_target(const Fleet& f, const std::vector<Planet>& planets) {
    double dx = std::cos(f.angle), dy = std::sin(f.angle);
    int best = -1;
    double best_t = 1e18;
    for (size_t i = 0; i < planets.size(); ++i) {
        const Planet& p = planets[i];
        double ox = f.x - p.x, oy = f.y - p.y;
        double tca = -(ox * dx + oy * dy);   // distance along ray to closest approach
        if (tca < 0.0) continue;             // planet is behind the fleet
        double perp2 = ox * ox + oy * oy - tca * tca;
        double rr = p.radius * p.radius;
        if (perp2 > rr) continue;            // ray misses the planet disk
        double t = tca - std::sqrt(rr - perp2);  // distance to first intersection
        if (t < 0.0) t = 0.0;
        if (t < best_t) { best_t = t; best = (int)i; }
    }
    if (best < 0) return {-1, 0.0};
    double sx = f.x - CENTER, sy = f.y - CENTER;  // sun occlusion: dies if it crosses the sun first
    double tcs = -(sx * dx + sy * dy);
    if (tcs >= 0.0) {
        double perp2 = sx * sx + sy * sy - tcs * tcs;
        double sr = SUN_RADIUS * SUN_RADIUS;
        if (perp2 <= sr) {
            double t_sun = tcs - std::sqrt(sr - perp2);
            if (t_sun >= 0.0 && t_sun < best_t) return {-1, 0.0};
        }
    }
    return {best, best_t / fleet_speed(f.ships)};
}

// Row ordering used by both encode and decode: own planets first, then by id.
inline std::vector<int> entity_order(const GameState& s, int player, int max_entities) {
    std::vector<int> idx(s.planets.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        int ka = (s.planets[a].owner == player) ? 0 : 1;
        int kb = (s.planets[b].owner == player) ? 0 : 1;
        if (ka != kb) return ka < kb;
        return s.planets[a].id < s.planets[b].id;
    });
    if ((int)idx.size() > max_entities) idx.resize(max_entities);
    return idx;
}

// Fill entity/mask/global buffers for `player`. Also outputs per-row planet id and ship
// count (for action decoding). Buffers must be sized (max_entities*F), (max_entities),
// (N_GLOBAL_FEATURES). Mirrors EntityObservation.process.
inline void encode_obs(const GameState& s, int player, int max_entities,
                       float* entities, float* entity_mask, float* action_mask,
                       float* globals, long* row_planet_id, long* row_planet_ships,
                       int episode_steps) {
    std::memset(entities, 0, sizeof(float) * max_entities * N_ENTITY_FEATURES);
    std::memset(entity_mask, 0, sizeof(float) * max_entities);
    std::memset(action_mask, 0, sizeof(float) * max_entities);
    for (int i = 0; i < max_entities; ++i) { row_planet_id[i] = -1; row_planet_ships[i] = 0; }

    std::vector<long> comet_set = s.comet_planet_ids;  // small; linear lookup ok
    auto is_comet = [&](long id) {
        return std::find(comet_set.begin(), comet_set.end(), id) != comet_set.end();
    };

    // Per-planet fleet pressure (raw proximity sum) + projected incoming threat (who a fleet
    // will actually reach, and when). Threat is the signal a reactive policy needs to defend:
    // a fleet inside the pressure radius but aimed away is harmless; one aimed in from afar is
    // the danger. fleet_target() filters out fleets that hit nothing.
    size_t np = s.planets.size();
    std::vector<double> enemy_press(np, 0.0), ally_press(np, 0.0);
    std::vector<double> inc_enemy(np, 0.0), inc_ally(np, 0.0);
    std::vector<double> eta_enemy(np, 1e18), eta_ally(np, 1e18);
    for (const auto& f : s.fleets) {
        for (size_t i = 0; i < np; ++i) {
            if (distance2(f.x, f.y, s.planets[i].x, s.planets[i].y) <= PRESSURE_RADIUS) {
                if (f.owner == player) ally_press[i] += f.ships;
                else enemy_press[i] += f.ships;
            }
        }
        auto tgt = fleet_target(f, s.planets);
        if (tgt.first < 0) continue;
        int ti = tgt.first;
        double eta = tgt.second;
        if (f.owner == player) {
            inc_ally[ti] += (double)f.ships;
            if (eta < eta_ally[ti]) eta_ally[ti] = eta;
        } else {
            inc_enemy[ti] += (double)f.ships;
            if (eta < eta_enemy[ti]) eta_enemy[ti] = eta;
        }
    }

    auto order = entity_order(s, player, max_entities);
    int n = (int)order.size();
    for (int row = 0; row < n; ++row) {
        int i = order[row];
        const Planet& p = s.planets[i];
        float* e = entities + row * N_ENTITY_FEATURES;
        double dx = (p.x - CENTER) / CENTER, dy = (p.y - CENTER) / CENTER;
        double dist_center = std::sqrt((p.x - CENTER) * (p.x - CENTER) + (p.y - CENTER) * (p.y - CENTER));
        e[0] = (p.owner == player) ? 1.f : 0.f;
        e[1] = (p.owner != player && p.owner != -1) ? 1.f : 0.f;
        e[2] = (p.owner == -1) ? 1.f : 0.f;
        e[3] = (float)(p.x / BOARD_SIZE);
        e[4] = (float)(p.y / BOARD_SIZE);
        e[5] = (float)dx;
        e[6] = (float)dy;
        e[7] = (float)(dist_center / DIAG_HALF);
        e[8] = (float)(p.radius / 3.0);
        e[9] = (float)ship_log((double)p.ships);
        e[10] = (float)(p.production / 5.0);
        e[11] = is_comet(p.id) ? 1.f : 0.f;
        e[12] = ((dist_center + p.radius) < ROTATION_RADIUS_LIMIT) ? 1.f : 0.f;
        e[13] = (float)ship_log(enemy_press[i]);
        e[14] = (float)ship_log(ally_press[i]);
        // Incoming-fleet threat (projected): how much / how soon, plus a net hold-margin.
        double ie = inc_enemy[i], ia = inc_ally[i], ee = eta_enemy[i], ea = eta_ally[i];
        e[15] = (float)ship_log(ie);                                       // enemy ships inbound
        e[16] = ie > 0.0 ? (float)std::exp(-ee / THREAT_ETA_TAU) : 0.f;    // enemy imminence (->1 soon)
        e[17] = (float)ship_log(ia);                                       // my reinforcements inbound
        e[18] = ia > 0.0 ? (float)std::exp(-ea / THREAT_ETA_TAU) : 0.f;    // reinforcement imminence
        // Hold-margin at impact: current ships + production banked until impact + help - attack.
        double margin = ie > 0.0 ? ((double)p.ships + (double)p.production * ee + ia - ie) : 0.0;
        e[19] = ie > 0.0 ? (float)std::tanh(margin / THREAT_MARGIN_SCALE) : 0.f;  // >0 hold, <0 fall
        entity_mask[row] = 1.f;
        row_planet_id[row] = p.id;
        row_planet_ships[row] = p.ships;
        if (p.owner == player && p.ships > 0) action_mask[row] = 1.f;
    }

    // Globals.
    long my_ships = 0, enemy_ships = 0, my_fleet = 0, enemy_fleet = 0;
    int my_planets = 0, enemy_planets = 0, neutral_planets = 0;
    for (const auto& p : s.planets) {
        if (p.owner == player) { my_ships += p.ships; my_planets++; }
        else if (p.owner == -1) neutral_planets++;
        else { enemy_ships += p.ships; enemy_planets++; }
    }
    for (const auto& f : s.fleets) {
        if (f.owner == player) my_fleet += f.ships; else enemy_fleet += f.ships;
    }
    int total_planets = std::max<int>(1, (int)s.planets.size());
    globals[0] = (float)((double)s.step / std::max(1, episode_steps));
    globals[1] = (float)(s.angular_velocity * 10.0);
    globals[2] = (float)ship_log((double)my_ships);
    globals[3] = (float)ship_log((double)enemy_ships);
    globals[4] = (float)((double)my_planets / total_planets);
    globals[5] = (float)((double)enemy_planets / total_planets);
    globals[6] = (float)((double)neutral_planets / total_planets);
    globals[7] = (float)ship_log((double)my_fleet);
    globals[8] = (float)ship_log((double)enemy_fleet);
    globals[9] = (float)((double)n / max_entities);
}

// TARGET-BASED decode. class 0 = noop; else class-1 splits into (target_row, frac_bin):
// target_row = launch / num_fracs, frac_bin = launch % num_fracs. The launch angle is
// computed to aim straight at the target entity's current position (like the scripted
// starter's atan2), so every launch actually hits something -- this removes the precise-
// aiming barrier that 16 angle bins impose and makes exploration land real captures.
// Ownership-safe (action_mask) + target must be a real, different entity.
inline Action decode_action_target(const GameState& s, const long* classes,
                                   const float* action_mask, const long* row_planet_id,
                                   const long* row_planet_ships, int max_entities, int num_fracs,
                                   const std::vector<double>& fractions, long min_ships = 1) {
    Action moves;
    auto pos_of = [&](long id, double& x, double& y) -> bool {
        for (const auto& p : s.planets)
            if (p.id == id) { x = p.x; y = p.y; return true; }
        return false;
    };
    for (int row = 0; row < max_entities; ++row) {
        long c = classes[row];
        if (c <= 0) continue;
        if (action_mask[row] < 0.5f) continue;
        long pid = row_planet_id[row];
        if (pid < 0) continue;
        long avail = row_planet_ships[row];
        if (avail <= 0) continue;
        long launch = c - 1;
        long target_row = launch / num_fracs;
        long fi = launch % num_fracs;
        if (target_row < 0 || target_row >= max_entities || target_row == row) continue;
        long tpid = row_planet_id[target_row];
        if (tpid < 0) continue;  // padding / invalid target
        double fx, fy, tx, ty;
        if (!pos_of(pid, fx, fy) || !pos_of(tpid, tx, ty)) continue;
        double angle = std::atan2(ty - fy, tx - fx);
        long ships = (long)(fractions[fi] * avail);
        ships = std::max(min_ships, std::min(ships, avail));
        if (ships > 0) moves.push_back(Move{pid, angle, ships});
    }
    return moves;
}

// Per-planet action classes -> moves. class 0 = noop; else (angle_bin, frac_bin).
// Mirrors PerPlanetAction.decode (ownership-safe via action_mask).
inline Action decode_action(const long* classes, const float* action_mask,
                            const long* row_planet_id, const long* row_planet_ships,
                            int max_entities, int angle_bins,
                            const std::vector<double>& fractions, long min_ships = 1) {
    Action moves;
    for (int row = 0; row < max_entities; ++row) {
        long c = classes[row];
        if (c <= 0) continue;
        if (action_mask[row] < 0.5f) continue;
        long pid = row_planet_id[row];
        if (pid < 0) continue;
        long avail = row_planet_ships[row];
        if (avail <= 0) continue;
        long launch = c - 1;  // index into angle_bins*fractions
        long k = launch % angle_bins;
        long fi = launch / angle_bins;
        double angle = 2.0 * PI * (double)k / angle_bins;
        double frac = fractions[fi];
        long ships = (long)(frac * avail);
        ships = std::max(min_ships, std::min(ships, avail));
        if (ships > 0) moves.push_back(Move{pid, angle, ships});
    }
    return moves;
}

}  // namespace ow
