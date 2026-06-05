// nanobind module: Phase-0 smoke fns + Phase-1 single-step parity entry point.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <torch/torch.h>

#include <vector>

#include <map>
#include <string>

#include "core/encode.hpp"
#include "core/sim.hpp"
#include "core/state.hpp"
#include "io/serialize.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// --- helpers: kaggle obs dict <-> ow::GameState -----------------------------
static std::vector<double> row_to_doubles(nb::handle row) {
    std::vector<double> v;
    for (nb::handle e : nb::borrow(row)) v.push_back(nb::cast<double>(e));
    return v;
}

// Parse a pre-generated comet schedule (list of {spawn_step, paths, ships}) into
// ScheduledComet events. Mirrors native_worldgen.build_comet_schedule.
static std::vector<ow::ScheduledComet> schedule_from_handle(nb::handle h) {
    std::vector<ow::ScheduledComet> out;
    for (nb::handle ev : nb::cast<nb::list>(h)) {
        nb::dict ed = nb::cast<nb::dict>(ev);
        ow::ScheduledComet sc;
        sc.spawn_step = nb::cast<int>(ed["spawn_step"]);
        sc.ships = nb::cast<long>(ed["ships"]);
        for (nb::handle path : nb::cast<nb::list>(ed["paths"])) {
            std::vector<std::pair<double, double>> pts;
            for (nb::handle pt : nb::borrow(path)) {
                auto xy = row_to_doubles(pt);
                pts.push_back({xy[0], xy[1]});
            }
            sc.paths.push_back(std::move(pts));
        }
        out.push_back(std::move(sc));
    }
    return out;
}

static ow::GameState state_from_dict(nb::dict d) {
    ow::GameState s;
    for (nb::handle row : nb::cast<nb::list>(d["planets"])) {
        auto r = row_to_doubles(row);
        s.planets.push_back(ow::Planet{(long)r[0], (int)r[1], r[2], r[3], r[4], (long)r[5], (int)r[6]});
    }
    for (nb::handle row : nb::cast<nb::list>(d["initial_planets"])) {
        auto r = row_to_doubles(row);
        s.initial_planets.push_back(ow::Planet{(long)r[0], (int)r[1], r[2], r[3], r[4], (long)r[5], (int)r[6]});
    }
    for (nb::handle row : nb::cast<nb::list>(d["fleets"])) {
        auto r = row_to_doubles(row);
        s.fleets.push_back(ow::Fleet{(long)r[0], (int)r[1], r[2], r[3], r[4], (long)r[5], (long)r[6]});
    }
    for (nb::handle pid : nb::cast<nb::list>(d["comet_planet_ids"]))
        s.comet_planet_ids.push_back(nb::cast<long>(pid));
    for (nb::handle g : nb::cast<nb::list>(d["comets"])) {
        nb::dict gd = nb::cast<nb::dict>(g);
        ow::CometGroup cg;
        for (nb::handle pid : nb::cast<nb::list>(gd["planet_ids"]))
            cg.planet_ids.push_back(nb::cast<long>(pid));
        for (nb::handle path : nb::cast<nb::list>(gd["paths"])) {
            std::vector<std::pair<double, double>> pts;
            for (nb::handle pt : nb::borrow(path)) {
                auto xy = row_to_doubles(pt);
                pts.push_back({xy[0], xy[1]});
            }
            cg.paths.push_back(std::move(pts));
        }
        cg.path_index = nb::cast<long>(gd["path_index"]);
        s.comets.push_back(std::move(cg));
    }
    s.angular_velocity = nb::cast<double>(d["angular_velocity"]);
    s.next_fleet_id = nb::cast<long>(d["next_fleet_id"]);
    s.step = nb::cast<int>(d["step"]);
    s.num_agents = nb::cast<int>(d["num_agents"]);
    if (d.contains("comet_schedule")) s.comet_schedule = schedule_from_handle(d["comet_schedule"]);
    return s;
}

static nb::list planets_to_list(const std::vector<ow::Planet>& ps) {
    nb::list out;
    for (const auto& p : ps) {
        nb::list r;
        r.append(p.id); r.append(p.owner); r.append(p.x); r.append(p.y);
        r.append(p.radius); r.append(p.ships); r.append(p.production);
        out.append(r);
    }
    return out;
}

static nb::dict state_to_dict(const ow::GameState& s) {
    nb::dict d;
    d["planets"] = planets_to_list(s.planets);
    d["initial_planets"] = planets_to_list(s.initial_planets);
    nb::list fleets;
    for (const auto& f : s.fleets) {
        nb::list r;
        r.append(f.id); r.append(f.owner); r.append(f.x); r.append(f.y);
        r.append(f.angle); r.append(f.from_planet_id); r.append(f.ships);
        fleets.append(r);
    }
    d["fleets"] = fleets;
    nb::list cpids;
    for (long pid : s.comet_planet_ids) cpids.append(pid);
    d["comet_planet_ids"] = cpids;
    nb::list comets;
    for (const auto& g : s.comets) {
        nb::dict gd;
        nb::list pids;
        for (long pid : g.planet_ids) pids.append(pid);
        gd["planet_ids"] = pids;
        nb::list paths;
        for (const auto& path : g.paths) {
            nb::list pl;
            for (const auto& pt : path) {
                nb::list xy; xy.append(pt.first); xy.append(pt.second);
                pl.append(xy);
            }
            paths.append(pl);
        }
        gd["paths"] = paths;
        gd["path_index"] = g.path_index;
        comets.append(gd);
    }
    d["comets"] = comets;
    d["angular_velocity"] = s.angular_velocity;
    d["next_fleet_id"] = s.next_fleet_id;
    d["step"] = s.step;
    d["num_agents"] = s.num_agents;
    return d;
}

static std::vector<ow::Action> actions_from_list(nb::list actions) {
    std::vector<ow::Action> out;
    for (nb::handle pa : actions) {
        ow::Action a;
        for (nb::handle mv : nb::borrow(pa)) {
            auto m = row_to_doubles(mv);
            a.push_back(ow::Move{(long)m[0], m[1], (long)m[2]});
        }
        out.push_back(std::move(a));
    }
    return out;
}

// Advance one deterministic tick. `obs` is a kaggle-style state dict (plus num_agents);
// returns the next-state dict. Does NOT spawn comets (parity excludes spawn steps).
static nb::dict step_from_state(nb::dict obs, nb::list actions, double shipSpeed,
                                double cometSpeed, int episodeSteps) {
    ow::GameState s = state_from_dict(obs);
    ow::Config cfg{episodeSteps, shipSpeed, cometSpeed};
    auto acts = actions_from_list(actions);
    ow::step(s, acts, cfg);
    return state_to_dict(s);
}

// Owned numpy array from a heap buffer (capsule frees it).
template <class T>
static nb::ndarray<nb::numpy, T> own_array(T* data, std::vector<size_t> shape) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.data(), owner);
}

// Encode a single state for `player` -> (entities, entity_mask, action_mask, globals)
// as numpy arrays. Used to parity-check against the Python EntityObservation.
static nb::tuple encode_state(nb::dict obs, int player, int max_entities, int episode_steps) {
    ow::GameState s = state_from_dict(obs);
    int F = ow::N_ENTITY_FEATURES, G = ow::N_GLOBAL_FEATURES;
    float* ent = new float[(size_t)max_entities * F];
    float* emask = new float[max_entities];
    float* amask = new float[max_entities];
    float* glob = new float[G];
    std::vector<long> rid(max_entities), rsh(max_entities);
    ow::encode_obs(s, player, max_entities, ent, emask, amask, glob, rid.data(), rsh.data(),
                   episode_steps);
    return nb::make_tuple(
        own_array(ent, {(size_t)max_entities, (size_t)F}),
        own_array(emask, {(size_t)max_entities}),
        own_array(amask, {(size_t)max_entities}),
        own_array(glob, {(size_t)G}));
}

// Build a GameState from a world dict (planets + angular_velocity; no fleets/comets).
static ow::GameState world_from_dict(nb::dict d) {
    ow::GameState s;
    for (nb::handle row : nb::cast<nb::list>(d["planets"])) {
        auto r = row_to_doubles(row);
        s.planets.push_back(ow::Planet{(long)r[0], (int)r[1], r[2], r[3], r[4], (long)r[5], (int)r[6]});
    }
    s.initial_planets = s.planets;
    s.angular_velocity = nb::cast<double>(d["angular_velocity"]);
    s.next_fleet_id = 0;
    s.step = 0;
    s.num_agents = 2;
    if (d.contains("comet_schedule")) s.comet_schedule = schedule_from_handle(d["comet_schedule"]);
    return s;
}

NB_MODULE(orbitwars_native, m) {
    m.doc() = "Orbit Wars native core (C++/LibTorch)";
    // Phase-0 smoke
    m.def("hello", []() { return "orbitwars_native online"; });
    m.def("torch_version", []() { return std::string(TORCH_VERSION); });
    m.def("cuda_available", []() { return torch::cuda::is_available(); });
    m.def("cuda_device_count", []() { return static_cast<int>(torch::cuda::device_count()); });
    m.def("gpu_matmul_ok", []() {
        if (!torch::cuda::is_available()) return false;
        auto a = torch::randn({64, 64}, torch::device(torch::kCUDA));
        auto b = torch::randn({64, 64}, torch::device(torch::kCUDA));
        auto c = torch::matmul(a, b);
        return std::isfinite(c.sum().item<float>());
    });
    // Phase-1 parity entry point
    m.def("step_from_state", &step_from_state, "obs"_a, "actions"_a, "shipSpeed"_a = 6.0,
          "cometSpeed"_a = 4.0, "episodeSteps"_a = 500);
    m.def("encode_state", &encode_state, "obs"_a, "player"_a, "max_entities"_a = 64,
          "episode_steps"_a = 500);

    // Serialize a list of world dicts (from native_worldgen) to the C++-owned .owp format
    // the native train/eval executables read. One-time pool gen -- see scripts/gen_world_pool.py.
    m.def("write_world_pool", [](nb::list worlds, const std::string& path) {
        std::vector<ow::GameState> pool;
        for (nb::handle w : worlds) pool.push_back(world_from_dict(nb::cast<nb::dict>(w)));
        ow::write_world_pool(pool, path);
    }, "worlds"_a, "path"_a);
}
