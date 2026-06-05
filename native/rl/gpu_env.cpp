// GpuEnv implementation -- the Orbit Wars sim as batched LibTorch tensor ops (see gpu_env.hpp).
// Every method below is a vectorized mirror of core/sim.hpp + core/encode.hpp; the PARITY test
// (apps/parity_gpu) drives this and ow::step from identical states and asserts they agree.
#include "rl/gpu_env.hpp"

#include <algorithm>
#include <cmath>

#include "core/encode.hpp"  // feature-layout constants (N_ENTITY_FEATURES, N_SOON, ...)
#include "core/state.hpp"   // BOARD_SIZE, CENTER, SUN_RADIUS, ROTATION_RADIUS_LIMIT, COMET_*

namespace ow {

using torch::Tensor;
using namespace torch::indexing;

namespace {
constexpr double kPI = 3.141592653589793;
constexpr double kBIG = 1e18;

// log1p(max(0,x)) / log(1000), elementwise (mirrors encode.hpp ship_log).
inline Tensor ship_log_t(const Tensor& x) {
    return torch::log1p(x.clamp_min(0.0)) / std::log(1000.0);
}

// Fleet ship-speed (mirrors sim.hpp): 1 + (vmax-1)*(log(ships)/log(1000))^1.5, capped, ships>=1.
inline Tensor fleet_speed_t(const Tensor& ships, double vmax) {
    Tensor n = ships.clamp_min(1.0);
    Tensor v = 1.0 + (vmax - 1.0) * torch::pow(torch::log(n) / std::log(1000.0), 1.5);
    return v.clamp_max(vmax);
}
}  // namespace

GpuEnv::GpuEnv(const GpuEnvConfig& cfg, torch::Device device) : cfg_(cfg), dev_(device) {}

// ---------------------------------------------------------------------------------------------
// reset: pack B CPU GameStates into device SoA tensors. One-time per rollout (not per step).
// ---------------------------------------------------------------------------------------------
void GpuEnv::reset(const std::vector<GameState>& worlds) {
    B_ = (int)worlds.size();
    const int Cs = cfg_.comet_slots;
    int max_real = 0;
    for (const auto& w : worlds) max_real = std::max(max_real, (int)w.planets.size());
    if (cfg_.planet_cap == 0) cfg_.planet_cap = max_real + Cs;
    const int Ec = cfg_.planet_cap;
    TORCH_CHECK(max_real + Cs <= Ec, "planet_cap=", Ec, " too small for ", max_real,
                " planets + ", Cs, " comet slots");
    const int Ev = cfg_.max_events;
    int Lmax = 1;
    for (const auto& w : worlds)
        for (const auto& sc : w.comet_schedule)
            for (const auto& path : sc.paths) Lmax = std::max(Lmax, (int)path.size());

    const int B = B_;
    std::vector<float> p_alive(B * Ec, 0), p_owner(B * Ec, -1), px(B * Ec, 0), py(B * Ec, 0),
        prad(B * Ec, 0), pshp(B * Ec, 0), pprod(B * Ec, 0), pcom(B * Ec, 0), pix(B * Ec, 0),
        piy(B * Ec, 0), prot(B * Ec, 0);
    std::vector<float> angv(B, 0);
    std::vector<int> stp(B, 0);
    std::vector<int> c_spawn(B * Ev, -1);
    std::vector<float> c_ships(B * Ev, 0);
    std::vector<float> c_path(B * Ev * Cs * Lmax * 2, 0);
    std::vector<int> c_plen(B * Ev * Cs, 0);

    for (int b = 0; b < B; ++b) {
        const GameState& w = worlds[b];
        int n = std::min((int)w.planets.size(), max_real);
        for (int i = 0; i < n; ++i) {
            const Planet& pl = w.planets[i];
            bool comet = std::find(w.comet_planet_ids.begin(), w.comet_planet_ids.end(), pl.id) !=
                         w.comet_planet_ids.end();
            size_t s = (size_t)b * Ec + i;
            p_alive[s] = 1.f;
            p_owner[s] = (float)pl.owner;
            px[s] = (float)pl.x;
            py[s] = (float)pl.y;
            prad[s] = (float)pl.radius;
            pshp[s] = (float)pl.ships;
            pprod[s] = (float)pl.production;
            pcom[s] = comet ? 1.f : 0.f;
            pix[s] = (float)pl.x;  // current == initial at reset (step 0)
            piy[s] = (float)pl.y;
            double r = std::sqrt((pl.x - CENTER) * (pl.x - CENTER) + (pl.y - CENTER) * (pl.y - CENTER));
            prot[s] = (!comet && (r + pl.radius < ROTATION_RADIUS_LIMIT)) ? 1.f : 0.f;
        }
        angv[b] = (float)w.angular_velocity;
        stp[b] = w.step;
        int e = 0;
        for (const auto& sc : w.comet_schedule) {
            if (e >= Ev) break;
            c_spawn[(size_t)b * Ev + e] = sc.spawn_step;
            c_ships[(size_t)b * Ev + e] = (float)sc.ships;
            int ncp = std::min((int)sc.paths.size(), Cs);
            for (int ci = 0; ci < ncp; ++ci) {
                const auto& path = sc.paths[ci];
                c_plen[((size_t)b * Ev + e) * Cs + ci] = (int)path.size();
                for (int j = 0; j < (int)path.size() && j < Lmax; ++j) {
                    size_t base = ((((size_t)b * Ev + e) * Cs + ci) * Lmax + j) * 2;
                    c_path[base + 0] = (float)path[j].first;
                    c_path[base + 1] = (float)path[j].second;
                }
            }
            ++e;
        }
    }

    auto fo = torch::TensorOptions().dtype(torch::kFloat32);
    auto io = torch::TensorOptions().dtype(torch::kInt32);
    auto mk = [&](std::vector<float>& v, std::vector<int64_t> shape) {
        return torch::from_blob(v.data(), shape, fo).clone().to(dev_);
    };
    auto mki = [&](std::vector<int>& v, std::vector<int64_t> shape) {
        return torch::from_blob(v.data(), shape, io).clone().to(dev_);
    };
    t_.p_alive = mk(p_alive, {B, Ec});
    t_.p_owner = mk(p_owner, {B, Ec});
    t_.p_x = mk(px, {B, Ec});
    t_.p_y = mk(py, {B, Ec});
    t_.p_radius = mk(prad, {B, Ec});
    t_.p_ships = mk(pshp, {B, Ec});
    t_.p_prod = mk(pprod, {B, Ec});
    t_.p_is_comet = mk(pcom, {B, Ec});
    t_.p_init_x = mk(pix, {B, Ec});
    t_.p_init_y = mk(piy, {B, Ec});
    t_.p_rotates = mk(prot, {B, Ec});

    t_.f_alive = torch::zeros({B, cfg_.fleet_cap}, fo).to(dev_);
    t_.f_owner = torch::zeros({B, cfg_.fleet_cap}, fo).to(dev_);
    t_.f_x = torch::zeros({B, cfg_.fleet_cap}, fo).to(dev_);
    t_.f_y = torch::zeros({B, cfg_.fleet_cap}, fo).to(dev_);
    t_.f_angle = torch::zeros({B, cfg_.fleet_cap}, fo).to(dev_);
    t_.f_ships = torch::zeros({B, cfg_.fleet_cap}, fo).to(dev_);

    t_.c_spawn_step = mki(c_spawn, {B, Ev});
    t_.c_ships = mk(c_ships, {B, Ev});
    t_.c_path = mk(c_path, {B, Ev, Cs, Lmax, 2});
    t_.c_path_len = mki(c_plen, {B, Ev, Cs});
    t_.c_active = torch::full({B}, -1, io).to(dev_);
    t_.c_index = torch::zeros({B}, io).to(dev_);
    t_.c_next_ev = torch::zeros({B}, io).to(dev_);

    t_.ang_vel = mk(angv, {B});
    t_.step = mki(stp, {B});
    t_.done = torch::zeros({B}, fo).to(dev_);
}

// ---------------------------------------------------------------------------------------------
// fleet_target: closed-form ray-vs-disk + sun occlusion for M virtual fleets per env against the
// CURRENT planets. Returns {tgt (B,M) long: planet slot or -1, eta (B,M) float}. Mirrors
// encode.hpp fleet_target. Used by encode (M=Fc threat) and by the valid-launch reward (M=Ec*K).
// ---------------------------------------------------------------------------------------------
static std::pair<Tensor, Tensor> fleet_target_batch(const EnvTensors& t, const Tensor& fx,
                                                     const Tensor& fy, const Tensor& fang,
                                                     const Tensor& fships, double vmax) {
    // fx,fy,fang,fships: (B,M). planets: (B,Ec).
    Tensor dx = torch::cos(fang).unsqueeze(2);  // (B,M,1)
    Tensor dy = torch::sin(fang).unsqueeze(2);
    Tensor pxr = t.p_x.unsqueeze(1);  // (B,1,Ec)
    Tensor pyr = t.p_y.unsqueeze(1);
    Tensor rr = (t.p_radius * t.p_radius).unsqueeze(1);  // (B,1,Ec)
    Tensor alive = t.p_alive.unsqueeze(1) > 0.5;
    Tensor ox = fx.unsqueeze(2) - pxr;  // (B,M,Ec)
    Tensor oy = fy.unsqueeze(2) - pyr;
    Tensor tca = -(ox * dx + oy * dy);                  // (B,M,Ec)
    Tensor perp2 = ox * ox + oy * oy - tca * tca;
    Tensor hit = (tca >= 0.0) & (perp2 <= rr) & alive;  // (B,M,Ec)
    Tensor t_int = tca - torch::sqrt((rr - perp2).clamp_min(0.0));
    t_int = t_int.clamp_min(0.0);
    Tensor tvals = torch::where(hit, t_int, torch::full_like(t_int, kBIG));  // (B,M,Ec)
    auto best = tvals.min(2);                                                // over Ec
    Tensor best_t = std::get<0>(best);                                       // (B,M)
    Tensor best_e = std::get<1>(best);                                       // (B,M) long
    Tensor any_hit = best_t < kBIG;

    // sun occlusion: if the ray crosses the sun before best_t, it dies (target -1).
    Tensor sx = fx - CENTER, sy = fy - CENTER;       // (B,M)
    Tensor dxx = torch::cos(fang), dyy = torch::sin(fang);
    Tensor tcs = -(sx * dxx + sy * dyy);             // (B,M)
    Tensor sperp2 = sx * sx + sy * sy - tcs * tcs;
    double sr = SUN_RADIUS * SUN_RADIUS;
    Tensor t_sun = tcs - torch::sqrt((sr - sperp2).clamp_min(0.0));
    Tensor sun_block = (tcs >= 0.0) & (sperp2 <= sr) & (t_sun >= 0.0) & (t_sun < best_t);
    Tensor valid = any_hit & (~sun_block);

    Tensor tgt = torch::where(valid, best_e, torch::full_like(best_e, -1));
    Tensor eta = best_t / fleet_speed_t(fships, vmax);
    eta = torch::where(valid, eta, torch::zeros_like(eta));
    return {tgt, eta};
}

// ---------------------------------------------------------------------------------------------
// launch_fleets: append committed launches (owner,from_slot,angle,ships,commit) -- all (B,L) --
// into the fleet pool, placing each into a distinct free slot. Ship deduction happens in step().
// ---------------------------------------------------------------------------------------------
void GpuEnv::launch_fleets(const Tensor& owner, const Tensor& from_slot, const Tensor& angle,
                           const Tensor& ships, const Tensor& commit) {
    const int B = B_, Fc = cfg_.fleet_cap;
    const long L = from_slot.size(1);
    auto fo = torch::TensorOptions().dtype(torch::kFloat32).device(dev_);

    // free-slot ranks: for each env, the r-th free fleet slot.
    Tensor free = t_.f_alive < 0.5;                                  // (B,Fc) bool
    Tensor freef = free.to(torch::kFloat32);
    Tensor fr = torch::cumsum(freef, 1) - 1.0;                       // (B,Fc) rank of each free slot
    Tensor n_free = freef.sum(1, /*keepdim=*/false);                 // (B,)
    // rank -> slot map (size Fc+1; col Fc is a dump for non-free)
    Tensor idx = torch::where(free, fr.to(torch::kLong), torch::full_like(fr.to(torch::kLong), Fc));
    Tensor rank_to_slot = torch::full({B, Fc + 1}, (long)Fc,
                                      torch::TensorOptions().dtype(torch::kLong).device(dev_));
    Tensor slot_src = torch::arange(Fc, torch::TensorOptions().dtype(torch::kLong).device(dev_))
                          .unsqueeze(0).expand({B, Fc});
    rank_to_slot.scatter_(1, idx, slot_src);
    rank_to_slot = rank_to_slot.index({Slice(), Slice(0, Fc)});      // (B,Fc)

    Tensor commitb = commit > 0.5;                                   // (B,L)
    Tensor crank = (torch::cumsum(commit.to(torch::kFloat32), 1) - 1.0).to(torch::kLong);  // (B,L)
    Tensor place = commitb & (crank < n_free.unsqueeze(1).to(torch::kLong));               // (B,L)
    Tensor gslot = rank_to_slot.gather(1, crank.clamp(0, Fc - 1));   // (B,L) target fleet slot
    Tensor dump = torch::full_like(gslot, Fc);
    Tensor wslot = torch::where(place, gslot, dump);                 // (B,L), Fc => dump

    // augmented (B,Fc+1) targets so non-placed launches land in the dump column, then slice off.
    auto scatter_into = [&](Tensor field, const Tensor& vals) {
        Tensor aug = torch::cat({field, torch::zeros({B, 1}, fo)}, 1);  // (B,Fc+1)
        aug.scatter_(1, wslot, vals);
        return aug.index({Slice(), Slice(0, Fc)});
    };
    // origin position = planet pos + dir*(radius+0.1)
    Tensor opx = t_.p_x.gather(1, from_slot.clamp(0, t_.p_x.size(1) - 1));
    Tensor opy = t_.p_y.gather(1, from_slot.clamp(0, t_.p_x.size(1) - 1));
    Tensor orad = t_.p_radius.gather(1, from_slot.clamp(0, t_.p_x.size(1) - 1));
    Tensor sx = opx + torch::cos(angle) * (orad + 0.1);
    Tensor sy = opy + torch::sin(angle) * (orad + 0.1);

    t_.f_alive = scatter_into(t_.f_alive, torch::ones({B, L}, fo));
    t_.f_owner = scatter_into(t_.f_owner, owner);
    t_.f_x = scatter_into(t_.f_x, sx);
    t_.f_y = scatter_into(t_.f_y, sy);
    t_.f_angle = scatter_into(t_.f_angle, angle);
    t_.f_ships = scatter_into(t_.f_ships, ships);
}

void GpuEnv::opponent_action(int opponent, Tensor& angle, Tensor& ships, Tensor& commit) {
    const int B = B_, Ec = cfg_.planet_cap;
    auto fo = torch::TensorOptions().dtype(torch::kFloat32).device(dev_);
    angle = torch::zeros({B, Ec}, fo);
    ships = torch::zeros({B, Ec}, fo);
    commit = torch::zeros({B, Ec}, fo);
    if (opponent == 2) return;  // noop / passive
    TORCH_CHECK(false, "scripted opponent ", opponent, " not vectorized yet (next pass)");
}

GpuEnv::Obs GpuEnv::encode(int ego) {
    TORCH_CHECK(false, "GpuEnv::encode not implemented yet (next pass)");
    return {};
}

// ---------------------------------------------------------------------------------------------
// step: one tick. Order mirrors ow::step exactly (comet expire -> spawn -> launch -> production
// -> compute planet new pos -> fleet move+collision -> apply pos / remove -> combat).
// ---------------------------------------------------------------------------------------------
GpuEnv::StepOut GpuEnv::step(const Tensor& ego_action, int opponent, double act_threshold) {
    const int B = B_, Ec = cfg_.planet_cap, Fc = cfg_.fleet_cap;
    const int K = (int)(ego_action.size(2) / 2);
    const int Cs = cfg_.comet_slots, Ev = (int)t_.c_path.size(1), Lmax = (int)t_.c_path.size(3);
    const double vmax = cfg_.ship_speed;
    auto fo = torch::TensorOptions().dtype(torch::kFloat32).device(dev_);
    auto lo = torch::TensorOptions().dtype(torch::kLong).device(dev_);
    const int ego = 0, enemy = 1;
    const auto cslice = Slice(Ec - Cs, Ec);  // reserved comet planet slots

    Tensor prod_ego0 = (t_.p_prod * (t_.p_owner == (double)ego) * (t_.p_alive > 0.5)).sum(1);
    Tensor prod_enemy0 = (t_.p_prod * (t_.p_owner == (double)enemy) * (t_.p_alive > 0.5)).sum(1);

    // --- comet TOP-expiration (ow::step lines 409-429): remove comets whose path index (from the
    //     previous tick) has run off the end. For the symmetric equal-length paths here this is
    //     usually a no-op (the post-move expiry below already removed them), kept for fidelity. ---
    {
        Tensor active = t_.c_active >= 0;                                          // (B,)
        Tensor ev = t_.c_active.clamp_min(0).to(torch::kLong);                     // (B,)
        Tensor cplen = t_.c_path_len.gather(1, ev.view({B, 1, 1}).expand({B, 1, Cs})).squeeze(1);
        Tensor expired = active.unsqueeze(1) & (t_.c_index.unsqueeze(1) >= cplen);  // (B,Cs)
        Tensor cs_alive = t_.p_alive.index({Slice(), cslice});
        cs_alive = torch::where(expired, torch::zeros_like(cs_alive), cs_alive);
        t_.p_alive.index_put_({Slice(), cslice}, cs_alive);
        Tensor still = (cs_alive > 0.5).any(1);  // (B,)
        t_.c_active = torch::where(active & (~still), torch::full_like(t_.c_active, -1), t_.c_active);
    }

    // --- comet spawn (ow::step lines 431-474): when step+1 hits the next scheduled spawn, write the
    //     4 reserved slots as neutral comets at the off-board placeholder (-99,-99); the movement
    //     block advances them onto path[0] this same tick. ---
    {
        Tensor nev = t_.c_next_ev.clamp(0, Ev - 1).to(torch::kLong);              // (B,)
        Tensor spawn_step = t_.c_spawn_step.gather(1, nev.unsqueeze(1)).squeeze(1);
        Tensor ev_ships = t_.c_ships.gather(1, nev.unsqueeze(1)).squeeze(1);
        Tensor fire = (t_.c_next_ev < Ev) & ((t_.step + 1) == spawn_step);        // (B,)
        Tensor fb = fire.unsqueeze(1);                                            // (B,1)
        auto put = [&](torch::Tensor& field, const Tensor& val) {
            field.index_put_({Slice(), cslice},
                             torch::where(fb, val, field.index({Slice(), cslice})));
        };
        Tensor ones_cs = torch::ones({B, Cs}, fo), zeros_cs = torch::zeros({B, Cs}, fo);
        put(t_.p_alive, ones_cs);
        put(t_.p_owner, torch::full({B, Cs}, -1.0, fo));
        put(t_.p_ships, ev_ships.unsqueeze(1).expand({B, Cs}));
        put(t_.p_prod, torch::full({B, Cs}, (double)COMET_PRODUCTION, fo));
        put(t_.p_radius, torch::full({B, Cs}, COMET_RADIUS, fo));
        put(t_.p_is_comet, ones_cs);
        put(t_.p_x, torch::full({B, Cs}, -99.0, fo));
        put(t_.p_y, torch::full({B, Cs}, -99.0, fo));
        put(t_.p_rotates, zeros_cs);
        t_.c_active = torch::where(fire, t_.c_next_ev, t_.c_active);
        t_.c_index = torch::where(fire, torch::full_like(t_.c_index, -1), t_.c_index);
        t_.c_next_ev = torch::where(fire, t_.c_next_ev + 1, t_.c_next_ev);
    }

    // --- decode ego action (mirror decode_action_continuous), sequential per-planet budget ---
    Tensor legal = (t_.p_owner == (double)ego) & (t_.p_alive > 0.5) & (t_.p_ships > 0.0);  // (B,Ec)
    Tensor S = t_.p_ships;                       // ships at decode time (per planet)
    Tensor remaining = S.clone();
    std::vector<Tensor> e_ang, e_shp, e_can;     // per-k launch tensors (B,Ec)
    Tensor invalid = torch::zeros({B}, fo);
    Tensor launches = torch::zeros({B}, fo);
    for (int k = 0; k < K; ++k) {
        Tensor alpha = ego_action.index({Slice(), Slice(), 2 * k});
        Tensor phi = ego_action.index({Slice(), Slice(), 2 * k + 1});
        Tensor commit = phi >= act_threshold;                       // (B,Ec) bool
        Tensor n = torch::floor(phi * S);                           // ships to send
        Tensor ok = commit & legal & (n >= 1.0) & (remaining >= n);  // launches this k
        // committed-but-failed (illegal planet / n<1 / over the per-planet ship budget) == ref invalid.
        Tensor inv_k = commit & (~ok);
        remaining = remaining - torch::where(ok, n, torch::zeros_like(n));
        invalid = invalid + (inv_k.to(torch::kFloat32)).sum(1);
        launches = launches + (ok.to(torch::kFloat32)).sum(1);
        Tensor angle = 2.0 * kPI * alpha;
        e_ang.push_back(angle);
        e_shp.push_back(torch::where(ok, n, torch::zeros_like(n)));
        e_can.push_back(ok.to(torch::kFloat32));
    }

    // valid launches: committed ego launches whose heading lands on a planet (pre-step planets).
    Tensor valid = torch::zeros({B}, fo);
    for (int k = 0; k < K; ++k) {
        Tensor can = e_can[k] > 0.5;                                      // (B,Ec)
        Tensor ox = t_.p_x + torch::cos(e_ang[k]) * (t_.p_radius + 0.1);  // launch just outside planet
        Tensor oy = t_.p_y + torch::sin(e_ang[k]) * (t_.p_radius + 0.1);
        auto tgt_eta = fleet_target_batch(t_, ox, oy, e_ang[k], e_shp[k].clamp_min(1.0), vmax);
        Tensor lands = (std::get<0>(tgt_eta) >= 0) & can;                 // (B,Ec)
        valid = valid + lands.to(torch::kFloat32).sum(1);
    }

    // --- opponent launches (noop for now) ---
    Tensor o_ang, o_shp, o_can;
    opponent_action(opponent, o_ang, o_shp, o_can);

    // --- deduct ships from origin planets for all committed launches (ego per-k + opponent) ---
    Tensor ded = torch::zeros({B, Ec}, fo);
    for (int k = 0; k < K; ++k) ded = ded + e_shp[k];   // e_shp already 0 where !ok
    ded = ded + o_shp;                                   // opponent deductions
    t_.p_ships = t_.p_ships - ded;

    // --- place fleets: concat ego (Ec*K) + opponent (Ec) launches, scatter into free slots ---
    {
        std::vector<Tensor> ow_, sl_, an_, sh_, cm_;
        Tensor slot_idx = torch::arange(Ec, torch::TensorOptions().dtype(torch::kLong).device(dev_))
                              .unsqueeze(0).expand({B, Ec});
        for (int k = 0; k < K; ++k) {
            ow_.push_back(torch::full({B, Ec}, (double)ego, fo));
            sl_.push_back(slot_idx);
            an_.push_back(e_ang[k]);
            sh_.push_back(e_shp[k]);
            cm_.push_back(e_can[k]);
        }
        ow_.push_back(torch::full({B, Ec}, (double)enemy, fo));
        sl_.push_back(slot_idx);
        an_.push_back(o_ang);
        sh_.push_back(o_shp);
        cm_.push_back(o_can);
        Tensor owner = torch::cat(ow_, 1);
        Tensor from_slot = torch::cat(sl_, 1);
        Tensor angle = torch::cat(an_, 1);
        Tensor ships = torch::cat(sh_, 1);
        Tensor commit = torch::cat(cm_, 1);
        launch_fleets(owner, from_slot, angle, ships, commit);
    }

    // --- production (owned planets gain production) ---
    t_.p_ships = t_.p_ships + t_.p_prod * (t_.p_owner != -1.0) * (t_.p_alive > 0.5);

    // --- compute planet NEW positions (orbit), not yet applied ---
    Tensor stepf = t_.step.to(torch::kFloat32);  // (B,)
    Tensor dxc = t_.p_init_x - CENTER, dyc = t_.p_init_y - CENTER;
    Tensor r = torch::sqrt(dxc * dxc + dyc * dyc);
    Tensor ia = torch::atan2(dyc, dxc);
    Tensor ca = ia + t_.ang_vel.unsqueeze(1) * stepf.unsqueeze(1);
    Tensor rot = t_.p_rotates > 0.5;
    Tensor nx = torch::where(rot, CENTER + r * torch::cos(ca), t_.p_x);
    Tensor ny = torch::where(rot, CENTER + r * torch::sin(ca), t_.p_y);
    Tensor old_px = t_.p_x, old_py = t_.p_y;  // swept-collision uses old->new planet path

    // comet path movement (ow::step lines 519-566): advance the live group's path index and place
    // each comet at path[idx]; an index past the path end keeps the old pos (stationary) and is
    // flagged expired -> removed AFTER collision (so it still collides this tick, like the ref).
    Tensor comet_expired = torch::zeros({B, Cs}, torch::TensorOptions().dtype(torch::kBool).device(dev_));
    {
        Tensor active = t_.c_active >= 0;                       // (B,)
        Tensor ev = t_.c_active.clamp_min(0).to(torch::kLong);  // (B,)
        Tensor idx_new = t_.c_index + 1;                        // (B,) int32
        Tensor evx = ev.view({B, 1, 1, 1, 1}).expand({B, 1, Cs, Lmax, 2});
        Tensor cp_ev = t_.c_path.gather(1, evx).squeeze(1);    // (B,Cs,Lmax,2)
        Tensor idxc = idx_new.clamp(0, Lmax - 1).to(torch::kLong).view({B, 1, 1, 1}).expand({B, Cs, 1, 2});
        Tensor cpos = cp_ev.gather(2, idxc).squeeze(2);        // (B,Cs,2)
        Tensor cplen = t_.c_path_len.gather(1, ev.view({B, 1, 1}).expand({B, 1, Cs})).squeeze(1);
        Tensor within = active.unsqueeze(1) & (idx_new.unsqueeze(1) < cplen);  // (B,Cs)
        Tensor curx = nx.index({Slice(), cslice}), cury = ny.index({Slice(), cslice});
        nx.index_put_({Slice(), cslice}, torch::where(within, cpos.index({Slice(), Slice(), 0}), curx));
        ny.index_put_({Slice(), cslice}, torch::where(within, cpos.index({Slice(), Slice(), 1}), cury));
        t_.c_index = torch::where(active, idx_new, t_.c_index);
        comet_expired = active.unsqueeze(1) & (idx_new.unsqueeze(1) >= cplen);  // (B,Cs) removed below
    }

    // --- fleet movement + swept collision against planet paths (the big GPU compute) ---
    Tensor falive = t_.f_alive > 0.5;                       // (B,Fc)
    Tensor speed = fleet_speed_t(t_.f_ships, vmax);         // (B,Fc)
    Tensor fox = t_.f_x, foy = t_.f_y;
    Tensor fnx = fox + torch::cos(t_.f_angle) * speed;
    Tensor fny = foy + torch::sin(t_.f_angle) * speed;

    // swept_pair_hit over (B,Fc,Ec): fleet old->new vs planet old->new.
    Tensor Ax = fox.unsqueeze(2), Ay = foy.unsqueeze(2);   // (B,Fc,1)
    Tensor Bx = fnx.unsqueeze(2), By = fny.unsqueeze(2);
    Tensor P0x = old_px.unsqueeze(1), P0y = old_py.unsqueeze(1);  // (B,1,Ec)
    Tensor P1x = nx.unsqueeze(1), P1y = ny.unsqueeze(1);
    Tensor rad = t_.p_radius.unsqueeze(1);
    Tensor palive = t_.p_alive.unsqueeze(1) > 0.5;
    Tensor d0x = Ax - P0x, d0y = Ay - P0y;
    Tensor dvx = (Bx - Ax) - (P1x - P0x), dvy = (By - Ay) - (P1y - P0y);
    Tensor a = dvx * dvx + dvy * dvy;
    Tensor b = 2.0 * (d0x * dvx + d0y * dvy);
    Tensor c = d0x * d0x + d0y * d0y - rad * rad;
    Tensor disc = b * b - 4.0 * a * c;
    Tensor sq = torch::sqrt(disc.clamp_min(0.0));
    Tensor t1 = (-b - sq) / (2.0 * a);
    Tensor t2 = (-b + sq) / (2.0 * a);
    Tensor hit_quad = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0);
    Tensor hit_lin = (a < 1e-12) & (c <= 0.0);
    Tensor hit = torch::where(a < 1e-12, hit_lin, hit_quad);   // (B,Fc,Ec)
    // ow::step `check` flag: a comet on its first (off-board, x=-99) placement does not collide.
    Tensor pcheck = (t_.p_is_comet.unsqueeze(1) < 0.5) | (old_px.unsqueeze(1) >= 0.0);  // (B,1,Ec)
    hit = hit & palive & pcheck & falive.unsqueeze(2);

    // first hit = smallest planet slot index that is hit (matches sim.hpp planet-order break).
    Tensor slotf = torch::arange(Ec, fo).view({1, 1, Ec});
    Tensor order = torch::where(hit, slotf, torch::full_like(slotf, (double)Ec));
    auto firsthit = order.min(2);
    Tensor tgt_slot = std::get<1>(firsthit);                  // (B,Fc) long
    Tensor has_hit = std::get<0>(firsthit) < (double)Ec;      // (B,Fc)

    // OOB / sun removal for non-hitting fleets (point-to-segment distance to the sun center).
    Tensor oob = (fnx < 0.0) | (fnx > BOARD_SIZE) | (fny < 0.0) | (fny > BOARD_SIZE);
    Tensor vx = fox, vy = foy, wx = fnx, wy = fny;
    Tensor l2 = (vx - wx) * (vx - wx) + (vy - wy) * (vy - wy);
    Tensor tt = ((CENTER - vx) * (wx - vx) + (CENTER - vy) * (wy - vy)) / l2.clamp_min(1e-12);
    tt = tt.clamp(0.0, 1.0);
    Tensor prx = vx + tt * (wx - vx), pry = vy + tt * (wy - vy);
    Tensor sundist = torch::sqrt((CENTER - prx) * (CENTER - prx) + (CENTER - pry) * (CENTER - pry));
    Tensor sun_hit = (l2 > 0.0) & (sundist < SUN_RADIUS);     // l2==0 handled like distance to point below
    Tensor sun_pt = torch::sqrt((CENTER - vx) * (CENTER - vx) + (CENTER - vy) * (CENTER - vy)) < SUN_RADIUS;
    sun_hit = torch::where(l2 > 0.0, sun_hit, sun_pt);

    Tensor remove_fleet = falive & (has_hit | oob | sun_hit);
    // a fleet that hits is the only one contributing to combat; OOB/sun ones just vanish.
    Tensor contributes = falive & has_hit;

    // --- remove comets that ran off their path this tick (ow::step lines 617-631), BEFORE combat
    //     so a just-expired comet takes no combat (its slot is no longer alive). ---
    {
        Tensor cs_alive = t_.p_alive.index({Slice(), cslice});
        cs_alive = torch::where(comet_expired, torch::zeros_like(cs_alive), cs_alive);
        t_.p_alive.index_put_({Slice(), cslice}, cs_alive);
        Tensor active = t_.c_active >= 0;
        Tensor still = (cs_alive > 0.5).any(1);
        t_.c_active = torch::where(active & (~still), torch::full_like(t_.c_active, -1), t_.c_active);
    }

    // --- combat: scatter arriving ships per owner into (B,Ec), then elementwise resolve ---
    Tensor arr0 = torch::zeros({B, Ec}, fo);
    Tensor arr1 = torch::zeros({B, Ec}, fo);
    {
        Tensor cf = contributes.to(torch::kFloat32);
        Tensor s0 = t_.f_ships * cf * (t_.f_owner == (double)ego);     // (B,Fc)
        Tensor s1 = t_.f_ships * cf * (t_.f_owner == (double)enemy);
        Tensor tslot = tgt_slot.clamp(0, Ec - 1);
        arr0.scatter_add_(1, tslot, s0);
        arr1.scatter_add_(1, tslot, s1);
    }
    Tensor has0 = arr0 > 0.0, has1 = arr1 > 0.0;
    Tensor top = torch::maximum(arr0, arr1);
    Tensor second = torch::minimum(arr0, arr1);
    Tensor both = has0 & has1;
    Tensor surv_ships = torch::where(both, top - second, top);        // single owner -> its full sum
    Tensor tie = both & (arr0 == arr1);
    surv_ships = torch::where(tie, torch::zeros_like(surv_ships), surv_ships);
    // survivor owner: 0 if arr0>arr1, 1 if arr1>arr0, -1 on tie/none
    Tensor surv_owner = torch::full({B, Ec}, -1.0, fo);
    surv_owner = torch::where((arr0 > arr1), torch::zeros_like(surv_owner), surv_owner);
    surv_owner = torch::where((arr1 > arr0), torch::ones_like(surv_owner), surv_owner);
    Tensor any_arr = has0 | has1;

    // apply to planets (only where there was combat and survivors)
    Tensor apply = any_arr & (surv_ships > 0.0) & (t_.p_alive > 0.5);
    Tensor same = (t_.p_owner == surv_owner);
    Tensor reinforce = apply & same;
    Tensor attack = apply & (~same);
    t_.p_ships = torch::where(reinforce, t_.p_ships + surv_ships, t_.p_ships);
    Tensor after = t_.p_ships - surv_ships;                           // attack branch
    Tensor flips = attack & (after < 0.0);
    t_.p_ships = torch::where(attack, torch::where(after < 0.0, -after, after), t_.p_ships);
    t_.p_owner = torch::where(flips, surv_owner, t_.p_owner);

    // --- apply planet new positions; clear removed fleets ---
    t_.p_x = nx;
    t_.p_y = ny;
    Tensor keep = falive & (~remove_fleet);
    Tensor keepf = keep.to(torch::kFloat32);
    t_.f_alive = keepf;
    t_.f_owner = t_.f_owner * keepf;
    t_.f_x = fnx * keepf;  // write back the MOVED position (fnx/fny), not the pre-move one
    t_.f_y = fny * keepf;
    t_.f_angle = t_.f_angle * keepf;
    t_.f_ships = t_.f_ships * keepf;

    // --- increment step ---
    t_.step = t_.step + 1;

    Tensor prod_ego1 = (t_.p_prod * (t_.p_owner == (double)ego) * (t_.p_alive > 0.5)).sum(1);
    Tensor prod_enemy1 = (t_.p_prod * (t_.p_owner == (double)enemy) * (t_.p_alive > 0.5)).sum(1);

    StepOut out;
    out.invalid = invalid;
    out.valid = valid;
    out.launches = launches;
    out.dprod_ego = prod_ego1 - prod_ego0;
    out.dprod_enemy = prod_enemy1 - prod_enemy0;
    out.newly_done = torch::zeros({B}, fo);  // terminal handling added with rollout integration
    return out;
}

}  // namespace ow
