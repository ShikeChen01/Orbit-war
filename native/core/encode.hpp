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

// v4 LEAN per-planet encoding (docs/rl_math.pdf sec "Observation encoding"):
//   F = 11 body + 21 threat = 32 features/planet; the G=10 board globals are a SEPARATE input
//   (no longer merged into each row -- the net has a dedicated globals embedding path).
//   Body[0..10]  : x, y, orbital speed, cw/ccw sign, radius, production, ships(log), ownership
//                  code (0 neutral/1 ego/2-4 enemy), dist-to-center, comet flag, action mask.
//   Threat[11..31]: N_SOON soonest + N_BIG largest inbound fleets, 3 feats each [sgn, ETA, ships].
constexpr int N_SOON = 2;   // per-planet inbound fleets kept by soonest arrival (smallest ETA)
constexpr int N_BIG = 5;    // per-planet inbound fleets kept by largest ship count
constexpr int N_THREAT_FLEETS = N_SOON + N_BIG;  // 7 slots x 3 feats = 21
constexpr int N_BODY_FEATURES = 11;
constexpr int N_ENTITY_FEATURES = N_BODY_FEATURES + 3 * N_THREAT_FLEETS;  // 11 + 21 = 32
constexpr int N_GLOBAL_FEATURES = 10;
inline const double SHIP_LOG_DENOM = std::log(1000.0);
inline const double DIAG_HALF = std::sqrt(BOARD_SIZE * BOARD_SIZE + BOARD_SIZE * BOARD_SIZE) / 2.0;
constexpr double PI = 3.141592653589793;  // == Python math.pi
constexpr double THREAT_MAX_SPEED = 6.0;  // == Config.shipSpeed; fleet speed cap for the ETA model
constexpr double THREAT_ETA_SCALE = 100.0;  // raw ETA (ticks) feature divisor (linear, not imminence)

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
    size_t np = s.planets.size();
    auto labs = [](long v) { return v < 0 ? -v : v; };

    // --- Globals (board summary): computed once and written to the SEPARATE `globals` buffer.
    // The net embeds these on their own path and broadcasts to the heads (NOT merged per row). ---
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
    int total_planets = std::max<int>(1, (int)np);
    float g[N_GLOBAL_FEATURES];
    g[0] = (float)((double)s.step / std::max(1, episode_steps));
    g[1] = (float)(s.angular_velocity * 10.0);
    g[2] = (float)ship_log((double)my_ships);
    g[3] = (float)ship_log((double)enemy_ships);
    g[4] = (float)((double)my_planets / total_planets);
    g[5] = (float)((double)enemy_planets / total_planets);
    g[6] = (float)((double)neutral_planets / total_planets);
    g[7] = (float)ship_log((double)my_fleet);
    g[8] = (float)ship_log((double)enemy_fleet);
    g[9] = (float)((double)std::min<int>((int)np, max_entities) / max_entities);
    for (int k = 0; k < N_GLOBAL_FEATURES; ++k) globals[k] = g[k];

    // --- Top-N inbound fleets per planet (by |ships| desc): replaces the aggregate-threat
    // scalars. fleet_target gives the planet a fleet reaches + its ETA; misses are dropped.
    // Signed ships: +mine (reinforcement) / -enemy. (FFA -> there are no allies.) ---
    std::vector<std::vector<std::pair<long, double>>> inbound(np);  // (signed_ships, eta_ticks)
    for (const auto& f : s.fleets) {
        auto tgt = fleet_target(f, s.planets);
        if (tgt.first < 0) continue;
        long signed_ships = (f.owner == player) ? f.ships : -f.ships;
        inbound[(size_t)tgt.first].push_back({signed_ships, tgt.second});
    }

    auto order = entity_order(s, player, max_entities);
    int n = (int)order.size();
    int na = std::max(1, s.num_agents);
    for (int row = 0; row < n; ++row) {
        int i = order[row];
        const Planet& p = s.planets[i];
        float* e = entities + row * N_ENTITY_FEATURES;
        double dist_center =
            std::sqrt((p.x - CENTER) * (p.x - CENTER) + (p.y - CENTER) * (p.y - CENTER));
        bool comet = is_comet(p.id);
        // A body orbits only if it is not a comet and sits inside the rotation limit; comets and
        // far "static stars" do not rotate (orbital velocity 0). cw/ccw is the board's spin sign.
        bool rotating = (!comet) && ((dist_center + p.radius) < ROTATION_RADIUS_LIMIT);
        double vmag = rotating ? std::abs(s.angular_velocity) * dist_center : 0.0;  // tangential speed
        double cw = rotating ? (s.angular_velocity >= 0.0 ? 1.0 : -1.0) : 0.0;
        // ownership code, ego-relative: 0 neutral, 1 ego, 2..(na) enemy seats (stable under rotation)
        int own_code = (p.owner == -1) ? 0 : (1 + ((p.owner - player + na) % na));

        // --- 11 body features ---
        e[0] = (float)(p.x / BOARD_SIZE);
        e[1] = (float)(p.y / BOARD_SIZE);
        e[2] = (float)(vmag / THREAT_MAX_SPEED);  // orbital speed, relative to ship speed
        e[3] = (float)cw;                          // +1 ccw / -1 cw / 0 not orbiting
        e[4] = (float)(p.radius / 3.0);            // radius (hit geometry)
        e[5] = (float)(p.production / 5.0);         // production (reward-defining; redundant w/ radius)
        e[6] = (float)ship_log((double)p.ships);   // ships (log)
        e[7] = (float)own_code;                    // ownership code 0/1/2-4
        e[8] = (float)(dist_center / DIAG_HALF);   // distance to board center
        e[9] = comet ? 1.f : 0.f;                  // comet flag
        e[10] = (p.owner == player && p.ships > 0) ? 1.f : 0.f;  // action mask u_e

        // --- 21 threat features: N_SOON soonest-arriving then N_BIG largest inbound fleets ---
        // each slot: [sgn (+1 mine / -1 enemy / 0 empty), raw ETA (ticks, scaled), ship_log]
        auto& vec = inbound[(size_t)i];  // (signed_ships, eta_ticks)
        std::vector<int> by_eta(vec.size()), by_ships(vec.size());
        std::iota(by_eta.begin(), by_eta.end(), 0);
        std::iota(by_ships.begin(), by_ships.end(), 0);
        std::stable_sort(by_eta.begin(), by_eta.end(),
                         [&](int a, int b) { return vec[a].second < vec[b].second; });
        std::stable_sort(by_ships.begin(), by_ships.end(),
                         [&](int a, int b) { return labs(vec[a].first) > labs(vec[b].first); });
        auto fill = [&](int slot, int vi) {
            float* fe = e + N_BODY_FEATURES + 3 * slot;
            long sh = vec[vi].first;
            fe[0] = sh >= 0 ? 1.f : -1.f;                        // owner sign
            fe[1] = (float)(vec[vi].second / THREAT_ETA_SCALE);  // raw ETA (ticks), scaled
            fe[2] = (float)ship_log((double)labs(sh));           // ships (log)
        };
        for (int k = 0; k < N_SOON && k < (int)by_eta.size(); ++k) fill(k, by_eta[k]);
        for (int k = 0; k < N_BIG && k < (int)by_ships.size(); ++k) fill(N_SOON + k, by_ships[k]);

        entity_mask[row] = 1.f;
        row_planet_id[row] = p.id;
        row_planet_ships[row] = p.ships;
        if (p.owner == player && p.ships > 0) action_mask[row] = 1.f;
    }
}

// CONTINUOUS decode (docs/rl_math.pdf sec:decode). `actions` is row-major (E x 3K): per planet,
// K fleet TRIPLES [dx, dy, phi] in [0,1]. (dx,dy) is a DIRECTION VECTOR (recentred to [-1,1] as
// 2a-1); the heading is theta = atan2(2*dy-1, 2*dx-1) -- so aiming is LINEAR in relative position
// (attention-friendly), unlike the old scalar 2*pi*alpha a linear head could not represent. A fleet
// is COMMITTED when phi >= act_threshold; it launches n = floor(phi * S) ships (S = planet ships at
// decode time), with the engine's sequential deduction mimicked (later over-budget fleets drop).
// `*out_invalid` accumulates the COMMITTED fleets that fail to execute -- on a phantom slot, an
// unowned/0-ship planet, or past the ship budget -- i.e. the x_t penalized by the reward. The actor
// is NOT masked: it may commit anywhere and learns the legal action space from that penalty.
inline Action decode_action_continuous(const float* actions, const float* action_mask,
                                       const long* row_planet_id, const long* row_planet_ships,
                                       int max_entities, int fleets_per_planet, double act_threshold,
                                       int* out_invalid = nullptr) {
    Action moves;
    int invalid = 0;
    const int nap = 3 * fleets_per_planet;  // (dx,dy,phi) per fleet
    for (int row = 0; row < max_entities; ++row) {
        const float* a = actions + (size_t)row * nap;
        bool legal = (action_mask[row] >= 0.5f) && (row_planet_id[row] >= 0);
        long pid = row_planet_id[row];
        long S = row_planet_ships[row];  // ships available at decode time (pre-production)
        long remaining = S;
        for (int k = 0; k < fleets_per_planet; ++k) {
            double dx = 2.0 * (double)a[3 * k] - 1.0;      // direction vector, recentred to [-1,1]
            double dy = 2.0 * (double)a[3 * k + 1] - 1.0;
            double phi = (double)a[3 * k + 2];
            if (phi < act_threshold) continue;  // not committed -> genuine no-op, no penalty
            if (!legal) { ++invalid; continue; }
            long n = (long)std::floor(phi * (double)S);
            if (n >= 1 && remaining >= n) {
                double angle = std::atan2(dy, dx);  // direction vector -> heading (linear aiming)
                moves.push_back(Move{pid, angle, n});
                remaining -= n;
            } else {
                ++invalid;  // n < 1, or beyond the remaining ship budget (over-dispatch)
            }
        }
    }
    if (out_invalid) *out_invalid = invalid;
    return moves;
}

}  // namespace ow
