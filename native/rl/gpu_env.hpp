// GpuEnv: the Orbit Wars game, batched across B environments and run ENTIRELY on the GPU as
// LibTorch tensor ops (no CUDA toolkit / custom kernel -- pure eager tensors). It replaces the
// CPU-bound rollout (encode -> forward -> sim with the GPU idle ~70%) so the whole loop stays
// on-device and the deep policy forward + the swept fleet/planet collision keep the GPU
// compute-bound. See docs/rl_math.pdf sec:gpuenv and memory gpu-batched-sim.
//
// Design (Structure-of-Arrays; leading dim B = #envs):
//   * Planets live in FIXED slots [0, planet_cap). No entity_order / sorting -- the continuous
//     actor is permutation-equivariant per planet, so row order is irrelevant. The last
//     `comet_slots` (=4) rows are reserved for the at-most-one live comet group.
//   * Fleets live in a fixed pool [0, fleet_cap) with an alive mask; launches scatter into free
//     slots, arrivals/OOB/sun clear them.
//   * 2-player combat vectorizes: scatter-add arriving ships per owner into (B,Ec), then
//     elementwise top/second/survivor (the reference's stable-sort/first-seen order only bites
//     for >2 owners, which never happens here).
//   * Comets are a per-env baked schedule (spawn step + ships + precomputed path positions);
//     position is overwritten each tick, but ships/owner evolve through combat like any planet.
//
// PARITY: this is a THIRD engine and must match the rules of the C++ ow::step (core/sim.hpp),
// which is the Kaggle submission engine. Tested against it via apps/parity_gpu.
#pragma once
#include <torch/torch.h>

#include <vector>

#include "core/state.hpp"

namespace ow {

struct GpuEnvConfig {
    int planet_cap = 0;     // Ec: planet slots/env incl. comet_slots. 0 => derive (max over worlds + comet_slots)
    int fleet_cap = 1024;   // Fc: max simultaneous in-flight fleets/env (overflow launches dropped; asserted in parity)
    int comet_slots = 4;    // reserved planet slots for the (single) live comet group
    int max_events = 8;     // comet spawn events/env to bake (steps 50/150/250/350/450 => 5; pad)
    double ship_speed = 6.0;
    double comet_speed = 4.0;
    int episode_steps = 500;
    bool target_actor = false;  // v5: ego/opp action is (dest_slot, phi) per planet (one launch),
                                // heading = atan2 toward the chosen destination. false = (dx,dy,phi)x K.
};

// All tensors on `device`. Integer-valued quantities (ships, production, owner, step) are kept in
// float32 -- exact for these magnitudes (< 2^24) and cheaper than int64 on GPU; owner uses {-1,0,1}
// with the alive mask distinguishing empty slots.
struct EnvTensors {
    // --- planets (B, Ec) ---
    torch::Tensor p_alive;     // 1.0 live, 0.0 empty slot
    torch::Tensor p_owner;     // -1 neutral, 0/1 player
    torch::Tensor p_x, p_y;    // position
    torch::Tensor p_radius;
    torch::Tensor p_ships;     // integer-valued
    torch::Tensor p_prod;      // production (integer-valued)
    torch::Tensor p_is_comet;  // 1.0 if a comet body
    torch::Tensor p_init_x, p_init_y;  // initial position (orbit reference, non-comet)
    torch::Tensor p_rotates;   // 1.0 if (r_init + radius < ROTATION_RADIUS_LIMIT) and not a comet

    // --- fleets (B, Fc) ---
    torch::Tensor f_alive;
    torch::Tensor f_owner;     // 0/1
    torch::Tensor f_x, f_y;
    torch::Tensor f_angle;
    torch::Tensor f_ships;     // integer-valued
    torch::Tensor f_seq;       // monotonic launch-order stamp (mirrors ref next_fleet_id ordering)

    // --- comet schedule, baked per env ---
    torch::Tensor c_spawn_step;  // (B, Ev) int32, -1 padded
    torch::Tensor c_ships;       // (B, Ev) float
    torch::Tensor c_path;        // (B, Ev, comet_slots, Lmax, 2) float, positions per path index
    torch::Tensor c_path_len;    // (B, Ev, comet_slots) int32

    // --- comet runtime ---
    torch::Tensor c_active;   // (B,) int32: active event index, or -1 if none live
    torch::Tensor c_index;    // (B,) int32: current path index of the live group
    torch::Tensor c_next_ev;  // (B,) int32: next event to spawn

    // --- per-env bookkeeping ---
    torch::Tensor ang_vel;  // (B,) float angular velocity
    torch::Tensor step;     // (B,) int32 current tick
    torch::Tensor done;     // (B,) 1.0 if terminal
};

class GpuEnv {
   public:
    GpuEnv(const GpuEnvConfig& cfg, torch::Device device);

    // Pack B CPU GameStates (a world-pool sample) into device tensors and zero the runtime state.
    void reset(const std::vector<GameState>& worlds);

    int batch() const { return B_; }
    int planet_cap() const { return cfg_.planet_cap; }
    int fleet_cap() const { return cfg_.fleet_cap; }
    const EnvTensors& tensors() const { return t_; }
    const GpuEnvConfig& config() const { return cfg_; }

    // Observation for `ego` (docs sec:obs): F = N_ENTITY_FEATURES per planet + N_GLOBAL_FEATURES.
    struct Obs {
        torch::Tensor entities;     // (B, Ec, F)
        torch::Tensor entity_mask;  // (B, Ec)
        torch::Tensor action_mask;  // (B, Ec)  owned & ships>0
        torch::Tensor globals;      // (B, Gl)
    };
    Obs encode(int ego = 0);

    // Decode the ego action (B, Ec, 2K) of squashed controls into launches, generate the opponent's
    // launches, apply both, and advance one tick. The opponent is either a scripted one (`opponent`:
    // 0 random / 1 starter / 2 noop) OR -- when `opp_action` is non-null (self-play) -- a player-1
    // neural action (B, Ec, 3K) decoded exactly like the ego action. Returns the per-env reward
    // ingredients so the trainer can build the reward on-device.
    struct StepOut {
        torch::Tensor invalid;       // (B,) committed-but-failed ego dispatches (x_t)
        torch::Tensor valid;         // (B,) ego launches whose heading lands on a planet
        torch::Tensor launches;      // (B,) ego launches committed
        torch::Tensor dprod_ego;     // (B,) ego production delta this tick
        torch::Tensor dprod_enemy;   // (B,) enemy production delta this tick
        torch::Tensor newly_done;    // (B,) 1.0 if this tick made the env terminal
        torch::Tensor captured;      // (B,) planets the ego GAINED this tick (owner flip to ego)
        torch::Tensor lost;          // (B,) planets the ego LOST this tick (owner flip away from ego)
        torch::Tensor fleet_hits;    // (B,) ego fleets that hit a planet this tick (any owner planet)
        torch::Tensor fleet_hit_ships;  // (B,) total ships on those hitting ego fleets
    };
    StepOut step(const torch::Tensor& ego_action, int opponent, double act_threshold,
                 const torch::Tensor* opp_action = nullptr);

   private:
    GpuEnvConfig cfg_;
    torch::Device dev_;
    int B_ = 0;
    EnvTensors t_;

    // launch helper: append the given per-env launches (owner, from-slot, angle, ships, valid-mask)
    // into the fleet pool, deducting ships from the origin planet. Used for both ego and opponent.
    void launch_fleets(const torch::Tensor& owner, const torch::Tensor& from_slot,
                       const torch::Tensor& angle, const torch::Tensor& ships,
                       const torch::Tensor& commit, const torch::Tensor& seq);
    // scripted opponent launches as tensors (per env, per planet slot): {angle, ships, commit}.
    void opponent_action(int opponent, torch::Tensor& angle, torch::Tensor& ships,
                         torch::Tensor& commit);
};

}  // namespace ow
