// Deterministic per-step simulation, ported 1:1 from REFERENCE_orbit_wars.py interpreter
// (lines ~409-717), EXCLUDING the RNG comet-spawn block (431-474). Spawning is a separate
// concern (native worldgen / parity injection), keeping RNG out of the hot path.
#pragma once
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include "core/state.hpp"

namespace ow {

// Minimum distance from point p to segment v-w. Mirrors point_to_segment_distance.
inline double point_to_segment_distance(double px, double py, double vx, double vy,
                                        double wx, double wy) {
    double l2 = (vx - wx) * (vx - wx) + (vy - wy) * (vy - wy);
    if (l2 == 0.0) return distance2(px, py, vx, vy);
    double t = ((px - vx) * (wx - vx) + (py - vy) * (wy - vy)) / l2;
    t = std::max(0.0, std::min(1.0, t));
    double projx = vx + t * (wx - vx);
    double projy = vy + t * (wy - vy);
    return distance2(px, py, projx, projy);
}

// True iff fleet A->B and planet P0->P1 come within r for some t in [0,1]. Mirrors swept_pair_hit.
inline bool swept_pair_hit(double Ax, double Ay, double Bx, double By, double P0x, double P0y,
                           double P1x, double P1y, double r) {
    double d0x = Ax - P0x, d0y = Ay - P0y;
    double dvx = (Bx - Ax) - (P1x - P0x);
    double dvy = (By - Ay) - (P1y - P0y);
    double a = dvx * dvx + dvy * dvy;
    double b = 2.0 * (d0x * dvx + d0y * dvy);
    double c = d0x * d0x + d0y * d0y - r * r;
    if (a < 1e-12) return c <= 0.0;
    double disc = b * b - 4.0 * a * c;
    if (disc < 0.0) return false;
    double sq = std::sqrt(disc);
    double t1 = (-b - sq) / (2.0 * a);
    double t2 = (-b + sq) / (2.0 * a);
    return t2 >= 0.0 && t1 <= 1.0;
}

// Advance the game one tick in place. `actions[i]` is player i's move list.
inline void step(GameState& s, const std::vector<Action>& actions, const Config& cfg) {
    // --- Comet expiration before launch (lines 409-429) ---
    {
        std::unordered_set<long> expired;
        for (auto& g : s.comets) {
            long idx = g.path_index;
            for (size_t i = 0; i < g.planet_ids.size(); ++i) {
                if (idx >= (long)g.paths[i].size()) expired.insert(g.planet_ids[i]);
            }
        }
        if (!expired.empty()) remove_planets(s, expired);
    }

    // --- Comet spawn (lines 431-474), schedule-driven (RNG pre-resolved in Python) ---
    // Fires when (step + 1) hits a scheduled spawn step. Each event adds 4 symmetric
    // neutral "comet" planets starting off-board at (-99,-99) with path_index = -1; the
    // movement block below advances them onto path[0]. Mirrors the reference exactly:
    // new ids are max(current planet id) + 1 (comets never coexist across spawns, so
    // this is the original max), rows [pid,-1,-99,-99,COMET_RADIUS,ships,COMET_PRODUCTION].
    for (const auto& sc : s.comet_schedule) {
        if (sc.spawn_step != s.step + 1) continue;
        long next_id = 0;
        bool any = false;
        for (const auto& p : s.planets) {
            if (!any || p.id >= next_id) { next_id = p.id; any = true; }
        }
        next_id = any ? next_id + 1 : 0;
        CometGroup group;
        group.path_index = -1;
        group.paths = sc.paths;
        for (size_t i = 0; i < sc.paths.size(); ++i) {
            long pid = next_id + (long)i;
            group.planet_ids.push_back(pid);
            s.comet_planet_ids.push_back(pid);
            Planet pl{pid, -1, -99.0, -99.0, COMET_RADIUS, sc.ships, COMET_PRODUCTION};
            s.planets.push_back(pl);
            s.initial_planets.push_back(pl);
        }
        s.comets.push_back(std::move(group));
    }

    // --- 0. Fleet launch (lines 477-509) ---
    for (int pid = 0; pid < s.num_agents && pid < (int)actions.size(); ++pid) {
        for (const auto& mv : actions[pid]) {
            long ships = mv.ships;  // already integer
            Planet* from = find_planet(s, mv.from_id);
            if (from && from->owner == pid && from->ships >= ships && ships > 0) {
                from->ships -= ships;
                double sx = from->x + std::cos(mv.angle) * (from->radius + 0.1);
                double sy = from->y + std::sin(mv.angle) * (from->radius + 0.1);
                s.fleets.push_back(Fleet{s.next_fleet_id, pid, sx, sy, mv.angle, mv.from_id, ships});
                s.next_fleet_id += 1;
            }
        }
    }

    // --- 1. Production (lines 512-514) ---
    for (auto& p : s.planets)
        if (p.owner != -1) p.ships += p.production;

    // --- 2. Planet end-of-tick positions (lines 519-566) ---
    struct Path { double ox, oy, nx, ny; bool check; };
    std::unordered_map<long, Path> planet_paths;
    std::unordered_set<long> comet_pid_set(s.comet_planet_ids.begin(), s.comet_planet_ids.end());
    std::unordered_map<long, const Planet*> initial_by_id;
    for (auto& p : s.initial_planets) initial_by_id[p.id] = &p;

    std::vector<long> expired_after_move;

    for (auto& p : s.planets) {
        if (comet_pid_set.count(p.id)) continue;
        double nx = p.x, ny = p.y;
        auto it = initial_by_id.find(p.id);
        if (it != initial_by_id.end()) {
            const Planet* ip = it->second;
            double dx = ip->x - CENTER, dy = ip->y - CENTER;
            double r = std::sqrt(dx * dx + dy * dy);
            if (r + p.radius < ROTATION_RADIUS_LIMIT) {
                double initial_angle = std::atan2(dy, dx);
                double current_angle = initial_angle + s.angular_velocity * (double)s.step;
                nx = CENTER + r * std::cos(current_angle);
                ny = CENTER + r * std::sin(current_angle);
            }
        }
        planet_paths[p.id] = Path{p.x, p.y, nx, ny, true};
    }

    for (auto& g : s.comets) {
        g.path_index += 1;
        long idx = g.path_index;
        for (size_t i = 0; i < g.planet_ids.size(); ++i) {
            long pid = g.planet_ids[i];
            Planet* p = find_planet(s, pid);
            if (!p) continue;
            double ox = p->x, oy = p->y;
            if (idx >= (long)g.paths[i].size()) {
                expired_after_move.push_back(pid);
                planet_paths[pid] = Path{ox, oy, ox, oy, true};
            } else {
                double nxp = g.paths[i][idx].first, nyp = g.paths[i][idx].second;
                bool check = ox >= 0;  // first placement uses off-board placeholder
                planet_paths[pid] = Path{ox, oy, nxp, nyp, check};
            }
        }
    }

    // --- 3. Fleet movement + continuous collision (lines 568-609) ---
    double max_speed = cfg.shipSpeed;
    std::unordered_set<long> fleets_to_remove;
    // per-planet arriving (owner, ships) in fleet-encounter order, planet order preserved
    std::unordered_map<long, std::vector<std::pair<int, long>>> combat_lists;
    for (auto& p : s.planets) combat_lists[p.id];  // init in planet order (insertion)

    for (auto& f : s.fleets) {
        double speed = 1.0 + (max_speed - 1.0) *
                                  std::pow(std::log((double)f.ships) / std::log(1000.0), 1.5);
        speed = std::min(speed, max_speed);
        double ox = f.x, oy = f.y;
        f.x += std::cos(f.angle) * speed;
        f.y += std::sin(f.angle) * speed;
        double nx = f.x, ny = f.y;

        bool hit = false;
        for (auto& p : s.planets) {
            auto pit = planet_paths.find(p.id);
            if (pit == planet_paths.end() || !pit->second.check) continue;
            const Path& pp = pit->second;
            if (swept_pair_hit(ox, oy, nx, ny, pp.ox, pp.oy, pp.nx, pp.ny, p.radius)) {
                combat_lists[p.id].push_back({f.owner, f.ships});
                fleets_to_remove.insert(f.id);
                hit = true;
                break;
            }
        }
        if (hit) continue;
        if (!(0 <= f.x && f.x <= BOARD_SIZE && 0 <= f.y && f.y <= BOARD_SIZE)) {
            fleets_to_remove.insert(f.id);
            continue;
        }
        if (point_to_segment_distance(CENTER, CENTER, ox, oy, nx, ny) < SUN_RADIUS) {
            fleets_to_remove.insert(f.id);
            continue;
        }
    }

    // --- 4. Apply planet movement (lines 611-615) ---
    for (auto& p : s.planets) {
        auto it = planet_paths.find(p.id);
        if (it != planet_paths.end()) { p.x = it->second.nx; p.y = it->second.ny; }
    }

    // Remove expired comets immediately (lines 617-631)
    if (!expired_after_move.empty()) {
        std::unordered_set<long> ex(expired_after_move.begin(), expired_after_move.end());
        remove_planets(s, ex);
    }

    // Remove fleets that hit a planet / went OOB / crossed the sun (line 633)
    s.fleets.erase(std::remove_if(s.fleets.begin(), s.fleets.end(),
                                  [&](const Fleet& f) { return fleets_to_remove.count(f.id) != 0; }),
                   s.fleets.end());

    // --- 5. Combat resolution (lines 635-675), planet order ---
    for (auto& p : s.planets) {
        auto cit = combat_lists.find(p.id);
        if (cit == combat_lists.end() || cit->second.empty()) continue;
        auto& arriving = cit->second;

        // Sum ships per owner, preserving first-seen order.
        std::vector<std::pair<int, long>> player_ships;  // (owner, ships)
        for (auto& ow_ships : arriving) {
            bool found = false;
            for (auto& ps : player_ships)
                if (ps.first == ow_ships.first) { ps.second += ow_ships.second; found = true; break; }
            if (!found) player_ships.push_back(ow_ships);
        }
        if (player_ships.empty()) continue;

        std::stable_sort(player_ships.begin(), player_ships.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        int top_player = player_ships[0].first;
        long top_ships = player_ships[0].second;
        int survivor_owner;
        long survivor_ships;
        if (player_ships.size() > 1) {
            long second_ships = player_ships[1].second;
            survivor_ships = top_ships - second_ships;
            if (player_ships[0].second == player_ships[1].second) survivor_ships = 0;
            survivor_owner = survivor_ships > 0 ? top_player : -1;
        } else {
            survivor_owner = top_player;
            survivor_ships = top_ships;
        }

        if (survivor_ships > 0) {
            if (p.owner == survivor_owner) {
                p.ships += survivor_ships;
            } else {
                p.ships -= survivor_ships;
                if (p.ships < 0) { p.owner = survivor_owner; p.ships = -p.ships; }
            }
        }
    }
}

}  // namespace ow
