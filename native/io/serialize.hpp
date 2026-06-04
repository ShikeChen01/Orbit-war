// C++-owned binary formats for the native (Python-free) train/eval loop:
//   .owp  world pool        -- planets + angular_velocity + comet schedule
//   .owc  policy checkpoint -- self-describing meta + named float32 tensors
//
// Both are little-endian (every producer/consumer is x86-64 LE: this machine and the
// Kaggle eval box). The world-pool format lives ONLY here -- Python writes it through the
// `write_world_pool` binding (one-time pool gen), the native exes read it. The .owc format
// is ALSO implemented in pure Python (orbit_wars_rl/native_ckpt.py) so the Kaggle
// submission -- which has no compiled extension -- can load a trained policy. Keep the two
// .owc codecs in sync; the byte layout is documented identically in both files.
#pragma once
#include <torch/torch.h>

#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/encode.hpp"
#include "core/state.hpp"

namespace ow {

namespace detail {
template <class T>
inline void wr(std::ostream& o, T v) {
    o.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <class T>
inline T rd(std::istream& in) {
    T v;
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}
inline void expect_magic(std::istream& in, const char* want, const char* what) {
    char m[4];
    in.read(m, 4);
    if (std::string(m, 4) != std::string(want, 4))
        throw std::runtime_error(std::string("bad magic for ") + what);
}
}  // namespace detail

// ---- world pool (.owp) -----------------------------------------------------
// magic 'OWP1' ; int32 num_worlds ; then per world:
//   float64 angular_velocity
//   int32 num_planets ; per planet: int64 id, int32 owner, f64 x,y,radius, int64 ships, int32 prod
//   int32 num_events  ; per event: int32 spawn_step, int64 ships, int32 num_paths,
//                       per path: int32 num_points, per point: f64 x, f64 y
inline void write_world_pool(const std::vector<GameState>& pool, const std::string& path) {
    std::ofstream o(path, std::ios::binary);
    if (!o) throw std::runtime_error("cannot open for write: " + path);
    using namespace detail;
    o.write("OWP1", 4);
    wr<int32_t>(o, (int32_t)pool.size());
    for (const auto& s : pool) {
        wr<double>(o, s.angular_velocity);
        wr<int32_t>(o, (int32_t)s.planets.size());
        for (const auto& p : s.planets) {
            wr<int64_t>(o, (int64_t)p.id);
            wr<int32_t>(o, (int32_t)p.owner);
            wr<double>(o, p.x);
            wr<double>(o, p.y);
            wr<double>(o, p.radius);
            wr<int64_t>(o, (int64_t)p.ships);
            wr<int32_t>(o, (int32_t)p.production);
        }
        wr<int32_t>(o, (int32_t)s.comet_schedule.size());
        for (const auto& ev : s.comet_schedule) {
            wr<int32_t>(o, (int32_t)ev.spawn_step);
            wr<int64_t>(o, (int64_t)ev.ships);
            wr<int32_t>(o, (int32_t)ev.paths.size());
            for (const auto& pts : ev.paths) {
                wr<int32_t>(o, (int32_t)pts.size());
                for (const auto& pt : pts) {
                    wr<double>(o, pt.first);
                    wr<double>(o, pt.second);
                }
            }
        }
    }
}

inline std::vector<GameState> read_world_pool(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open world pool: " + path);
    using namespace detail;
    expect_magic(in, "OWP1", "world pool");
    int32_t n = rd<int32_t>(in);
    std::vector<GameState> pool;
    pool.reserve(n);
    for (int32_t i = 0; i < n; ++i) {
        GameState s;
        s.angular_velocity = rd<double>(in);
        int32_t np = rd<int32_t>(in);
        s.planets.reserve(np);
        for (int32_t j = 0; j < np; ++j) {
            Planet p;
            p.id = (long)rd<int64_t>(in);
            p.owner = (int)rd<int32_t>(in);
            p.x = rd<double>(in);
            p.y = rd<double>(in);
            p.radius = rd<double>(in);
            p.ships = (long)rd<int64_t>(in);
            p.production = (int)rd<int32_t>(in);
            s.planets.push_back(p);
        }
        int32_t ne = rd<int32_t>(in);
        for (int32_t e = 0; e < ne; ++e) {
            ScheduledComet sc;
            sc.spawn_step = (int)rd<int32_t>(in);
            sc.ships = (long)rd<int64_t>(in);
            int32_t npaths = rd<int32_t>(in);
            for (int32_t pa = 0; pa < npaths; ++pa) {
                int32_t npts = rd<int32_t>(in);
                std::vector<std::pair<double, double>> pts;
                pts.reserve(npts);
                for (int32_t k = 0; k < npts; ++k) {
                    double x = rd<double>(in), y = rd<double>(in);
                    pts.push_back({x, y});
                }
                sc.paths.push_back(std::move(pts));
            }
            s.comet_schedule.push_back(std::move(sc));
        }
        s.initial_planets = s.planets;
        s.next_fleet_id = 0;
        s.step = 0;
        s.num_agents = 2;
        pool.push_back(std::move(s));
    }
    return pool;
}

// ---- checkpoint (.owc) -----------------------------------------------------
// magic 'OWC1'
// int32 n_entity_features, n_global_features, hidden, angle_bins, max_entities,
//       num_fracs, episode_steps
// uint8 target_mode
// int32 num_fractions ; float64 fractions[num_fractions]
// int32 num_tensors ; per tensor: int32 name_len, name bytes, int32 ndim,
//       int32 dims[ndim], float32 data[prod(dims)]  (row-major / contiguous)
struct CheckpointMeta {
    int32_t n_entity_features = N_ENTITY_FEATURES;
    int32_t n_global_features = N_GLOBAL_FEATURES;
    int32_t hidden = 128, angle_bins = 16, max_entities = 64, num_fracs = 4;
    int32_t episode_steps = 500;
    uint8_t target_mode = 0;
    std::vector<double> fractions = {0.25, 0.5, 0.75, 1.0};
};

inline void write_checkpoint(const std::string& path,
                             const std::map<std::string, torch::Tensor>& sd,
                             const CheckpointMeta& meta) {
    std::ofstream o(path, std::ios::binary);
    if (!o) throw std::runtime_error("cannot open for write: " + path);
    using namespace detail;
    o.write("OWC1", 4);
    wr<int32_t>(o, meta.n_entity_features);
    wr<int32_t>(o, meta.n_global_features);
    wr<int32_t>(o, meta.hidden);
    wr<int32_t>(o, meta.angle_bins);
    wr<int32_t>(o, meta.max_entities);
    wr<int32_t>(o, meta.num_fracs);
    wr<int32_t>(o, meta.episode_steps);
    wr<uint8_t>(o, meta.target_mode);
    wr<int32_t>(o, (int32_t)meta.fractions.size());
    for (double f : meta.fractions) wr<double>(o, f);
    wr<int32_t>(o, (int32_t)sd.size());
    for (const auto& kv : sd) {
        auto t = kv.second.to(torch::kCPU).to(torch::kFloat32).contiguous();
        wr<int32_t>(o, (int32_t)kv.first.size());
        o.write(kv.first.data(), kv.first.size());
        wr<int32_t>(o, (int32_t)t.dim());
        for (int d = 0; d < t.dim(); ++d) wr<int32_t>(o, (int32_t)t.size(d));
        o.write(reinterpret_cast<const char*>(t.data_ptr<float>()), t.numel() * sizeof(float));
    }
}

inline std::map<std::string, torch::Tensor> read_checkpoint(const std::string& path,
                                                            CheckpointMeta& meta) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open checkpoint: " + path);
    using namespace detail;
    expect_magic(in, "OWC1", "checkpoint");
    meta.n_entity_features = rd<int32_t>(in);
    meta.n_global_features = rd<int32_t>(in);
    meta.hidden = rd<int32_t>(in);
    meta.angle_bins = rd<int32_t>(in);
    meta.max_entities = rd<int32_t>(in);
    meta.num_fracs = rd<int32_t>(in);
    meta.episode_steps = rd<int32_t>(in);
    meta.target_mode = rd<uint8_t>(in);
    int32_t nfr = rd<int32_t>(in);
    meta.fractions.clear();
    for (int32_t i = 0; i < nfr; ++i) meta.fractions.push_back(rd<double>(in));
    int32_t nt = rd<int32_t>(in);
    std::map<std::string, torch::Tensor> sd;
    for (int32_t i = 0; i < nt; ++i) {
        int32_t nl = rd<int32_t>(in);
        std::string name(nl, '\0');
        in.read(name.data(), nl);
        int32_t ndim = rd<int32_t>(in);
        std::vector<int64_t> shape(ndim);
        int64_t numel = 1;
        for (int32_t d = 0; d < ndim; ++d) {
            shape[d] = rd<int32_t>(in);
            numel *= shape[d];
        }
        auto t = torch::empty(shape, torch::kFloat32);
        in.read(reinterpret_cast<char*>(t.data_ptr<float>()), numel * sizeof(float));
        sd[name] = t;
    }
    return sd;
}

}  // namespace ow
