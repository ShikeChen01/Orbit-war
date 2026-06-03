// Array-of-Structs game state, a faithful mirror of the kaggle obs lists.
// (Phase 2 converts the hot path to batched Struct-of-Arrays; AoS here keeps the
// Phase-1 port 1:1 with REFERENCE_orbit_wars.py for parity.)
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace ow {

constexpr double BOARD_SIZE = 100.0;
constexpr double CENTER = BOARD_SIZE / 2.0;
constexpr double SUN_RADIUS = 10.0;
constexpr double ROTATION_RADIUS_LIMIT = 50.0;
constexpr double COMET_RADIUS = 1.0;
constexpr int COMET_PRODUCTION = 1;

// planet row: [id, owner, x, y, radius, ships, production]
struct Planet {
    long id;
    int owner;        // -1 neutral, else player id
    double x, y;
    double radius;
    long ships;
    int production;
};

// fleet row: [id, owner, x, y, angle, from_planet_id, ships]
struct Fleet {
    long id;
    int owner;
    double x, y;
    double angle;
    long from_planet_id;
    long ships;
};

struct CometGroup {
    std::vector<long> planet_ids;
    std::vector<std::vector<std::pair<double, double>>> paths;  // per comet, list of (x,y)
    long path_index;
};

struct Config {
    int episodeSteps = 500;
    double shipSpeed = 6.0;
    double cometSpeed = 4.0;
};

struct GameState {
    std::vector<Planet> planets;
    std::vector<Fleet> fleets;
    std::vector<CometGroup> comets;
    std::vector<long> comet_planet_ids;
    std::vector<Planet> initial_planets;
    double angular_velocity = 0.0;
    long next_fleet_id = 0;
    int step = 0;
    int num_agents = 2;
};

// A single player's action: list of [from_planet_id, angle, num_ships].
struct Move {
    long from_id;
    double angle;
    long ships;
};
using Action = std::vector<Move>;

inline double distance2(double ax, double ay, double bx, double by) {
    double dx = ax - bx, dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

inline Planet* find_planet(GameState& s, long id) {
    for (auto& p : s.planets)
        if (p.id == id) return &p;
    return nullptr;
}

// Remove a set of planet ids from planets / initial_planets / comet bookkeeping.
// Mirrors the expired-comet pruning in the interpreter (lines 418-429 / 620-631).
template <class Set>
inline void remove_planets(GameState& s, const Set& ids) {
    auto gone = [&](long id) { return ids.find(id) != ids.end(); };
    s.planets.erase(std::remove_if(s.planets.begin(), s.planets.end(),
                                   [&](const Planet& p) { return gone(p.id); }),
                    s.planets.end());
    s.initial_planets.erase(std::remove_if(s.initial_planets.begin(), s.initial_planets.end(),
                                           [&](const Planet& p) { return gone(p.id); }),
                            s.initial_planets.end());
    s.comet_planet_ids.erase(std::remove_if(s.comet_planet_ids.begin(), s.comet_planet_ids.end(),
                                            [&](long id) { return gone(id); }),
                             s.comet_planet_ids.end());
    for (auto& g : s.comets) {
        g.planet_ids.erase(std::remove_if(g.planet_ids.begin(), g.planet_ids.end(),
                                          [&](long id) { return gone(id); }),
                           g.planet_ids.end());
    }
    s.comets.erase(std::remove_if(s.comets.begin(), s.comets.end(),
                                  [&](const CometGroup& g) { return g.planet_ids.empty(); }),
                   s.comets.end());
}

}  // namespace ow
