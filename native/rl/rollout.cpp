#include "rl/rollout.hpp"

#include <algorithm>
#include <execution>

#include "core/agents.hpp"
#include "core/encode.hpp"
#include "core/sim.hpp"
#include "rl/algo/grpo.hpp"
#include "rl/reward.hpp"

namespace ow {

GroupedRollout::GroupedRollout(const TrainConfig& cfg, torch::Device device)
    : cfg_(cfg), device_(device), rng_(cfg.seed * 2654435761ull + 12345ull) {}

// One GRPO iteration with the CONTINUOUS actor (docs/rl_math.pdf sec:reward):
//   per-step dense  r_t = clip( ( w_Pi*dProd0 - rho*x_t - w_e*y_t*1[t>=x0] ) / sigma, -kappa, kappa )
//     dProd0 = gain in ego production; x_t = committed fleets that failed to execute (invalid);
//     y_t = enemy_growth_t - EMA(enemy_growth) -> penalize the enemy ACCELERATING.
//   return  R = 2 w_o + w_d D (win) | 1 w_o (draw) | -w_d D (loss),   D = sum_t gamma^t r_t.
TrajectoryBatch GroupedRollout::collect(Agent& policy, RolloutStats& stats) {
    const int G = cfg_.grpo.group_size, N = cfg_.grpo.num_groups, B = N * G;
    const int E = cfg_.model.max_entities, F = N_ENTITY_FEATURES, Gl = N_GLOBAL_FEATURES;
    const int K = cfg_.model.fleets_per_planet, nap = 3 * K;  // (dx,dy,phi) per fleet
    const int T = cfg_.rollout.episode_steps;
    const double gamma = cfg_.grpo.gamma;
    const double w_prod = cfg_.rollout.prod_reward_weight, prod_cap = cfg_.rollout.prod_reward_cap,
                 prod_decay = cfg_.rollout.prod_reward_decay,
                 w_valid = cfg_.rollout.valid_launch_reward, valid_cap = cfg_.rollout.valid_reward_cap,
                 w_inv = cfg_.rollout.illegal_launch_penalty,
                 w_miss = cfg_.rollout.miss_launch_penalty,
                 w_e = cfg_.rollout.enemy_growth_weight, beta = cfg_.rollout.enemy_growth_ema,
                 win_bonus = cfg_.rollout.win_bonus, loss_penalty = cfg_.rollout.loss_penalty;
    const int x0 = cfg_.rollout.enemy_growth_warmup;
    const double act_thr = cfg_.model.act_threshold;
    const int stage = cur_stage_ ? cur_stage_ : cfg_.rollout.stage;  // curriculum override
    Config econf{cfg_.rollout.episode_steps, 6.0, 4.0};

    // --- assign one (world, opponent) per group; all G envs in a group share them ---
    std::vector<GameState> st(B);
    std::vector<int> opp(B);  // 0 random, 1 starter, 2 noop (passive)
    std::vector<std::mt19937_64> erng(B);
    {
        double wr = cfg_.selfplay.mix_random, ws = cfg_.selfplay.mix_starter;
        std::uniform_real_distribution<double> u(0.0, wr + ws);
        for (int j = 0; j < N; ++j) {
            const GameState& world = world_pool_[cursor_++ % world_pool_.size()];
            int o = (stage == 1) ? 2 : (stage == 2) ? 1 : ((u(rng_) < wr) ? 0 : 1);
            for (int g = 0; g < G; ++g) {
                int i = j * G + g;
                st[i] = world;
                opp[i] = o;
                erng[i].seed(rng_());
            }
        }
    }

    std::vector<char> active(B, 1);
    // Per-game reward channels (accumulated over the episode, capped at the end). prodR/validR get
    // capped; invR (invalid penalty) is uncapped; suppR (enemy suppression) only fills at stage>=2;
    // the win/draw/loss outcome is added only at stage>=2 (stage 1 = pure dispatch+production).
    std::vector<double> prodR(B, 0.0), validR(B, 0.0), invR(B, 0.0), missR(B, 0.0), suppR(B, 0.0),
        gpow(B, 1.0), ppow(B, 1.0), prev_p0(B), prev_p1(B), gbar(B, 0.0), outcome(B, 0.0);
    std::vector<long> inv_acc(B, 0), lnch_acc(B, 0), valid_acc(B, 0), step_acc(B, 0);
    std::vector<int> ep_len(B, T);
    for (int i = 0; i < B; ++i) { prev_p0[i] = my_production(st[i], 0); prev_p1[i] = my_production(st[i], 1); }

    auto fo = torch::TensorOptions().dtype(torch::kFloat32);
    auto ent_buf = torch::zeros({T, B, E, F}, fo);
    auto em_buf = torch::zeros({T, B, E}, fo);
    auto am_buf = torch::zeros({T, B, E}, fo);
    auto gl_buf = torch::zeros({T, B, Gl}, fo);
    auto act_buf = torch::zeros({T, B, E, nap}, fo);
    auto oldlogp_buf = torch::zeros({T, B}, fo);
    auto valid_buf = torch::zeros({T, B}, fo);

    std::vector<int> idx(B);
    for (int i = 0; i < B; ++i) idx[i] = i;
    std::vector<std::vector<long>> rid(B, std::vector<long>(E, -1)), rsh(B, std::vector<long>(E, 0));

    int last_t = 0;
    for (int t = 0; t < T; ++t) {
        last_t = t;
        std::vector<float> ent((size_t)B * E * F, 0.f), em((size_t)B * E, 0.f),
            am((size_t)B * E, 0.f), gl((size_t)B * Gl, 0.f);
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
            encode_obs(st[i], 0, E, ent.data() + (size_t)i * E * F, em.data() + (size_t)i * E,
                       am.data() + (size_t)i * E, gl.data() + (size_t)i * Gl, rid[i].data(),
                       rsh[i].data(), cfg_.rollout.episode_steps);
        });
        auto t_ent = torch::from_blob(ent.data(), {B, E, F}, fo).clone();
        auto t_em = torch::from_blob(em.data(), {B, E}, fo).clone();
        auto t_am = torch::from_blob(am.data(), {B, E}, fo).clone();
        auto t_gl = torch::from_blob(gl.data(), {B, Gl}, fo).clone();
        ent_buf[t] = t_ent;
        em_buf[t] = t_em;
        am_buf[t] = t_am;
        gl_buf[t] = t_gl;

        auto dec = policy.act(t_ent.to(device_), t_em.to(device_), t_am.to(device_),
                              t_gl.to(device_), false);
        auto a_cpu = dec.action.to(torch::kCPU).contiguous();  // (B,E,2K) float
        act_buf[t] = a_cpu;
        oldlogp_buf[t] = dec.log_prob.to(torch::kCPU);
        const float* ad = a_cpu.data_ptr<float>();

        std::vector<float> validv(B, 0.f);
        std::for_each(std::execution::par, idx.begin(), idx.end(), [&](int i) {
            if (!active[i]) return;
            validv[i] = 1.f;
            std::vector<float> am0(E);
            for (int r = 0; r < E; ++r) am0[r] = am[(size_t)i * E + r];
            int invalid = 0;
            Action move0 = decode_action_continuous(ad + (size_t)i * E * nap, am0.data(),
                                                    rid[i].data(), rsh[i].data(), E, K, act_thr,
                                                    &invalid);
            // VALID launches: legal+committed fleets whose heading actually lands on a planet
            // (closed-form ray-vs-disk + sun occlusion). This is the aiming reward signal.
            int valid = 0;
            for (const auto& mv : move0) {
                const Planet* lp = nullptr;
                for (const auto& p : st[i].planets) if (p.id == mv.from_id) { lp = &p; break; }
                if (!lp) continue;
                double cx = std::cos(mv.angle), cy = std::sin(mv.angle);
                Fleet f{0, 0, lp->x + cx * (lp->radius + 0.1), lp->y + cy * (lp->radius + 0.1),
                        mv.angle, mv.from_id, mv.ships};
                if (fleet_target(f, st[i].planets).first >= 0) ++valid;
            }
            Action move1;  // opp 2 (noop) leaves it empty -> the passive stage-1 opponent
            if (opp[i] == 0) move1 = random_agent(st[i], 1, erng[i]);
            else if (opp[i] == 1) move1 = starter_agent(st[i], 1);
            std::vector<Action> acts = {move0, move1};
            ow::step(st[i], acts, econf);
            st[i].step += 1;

            double p0 = my_production(st[i], 0), p1 = my_production(st[i], 1);
            double dprod = p0 - prev_p0[i];
            double g_t = p1 - prev_p1[i];                 // enemy production growth this step
            double y_t = g_t - gbar[i];                   // vs the enemy's own recent average
            gbar[i] = (1.0 - beta) * gbar[i] + beta * g_t;
            prodR[i] += gpow[i] * ppow[i] * w_prod * dprod;     // production gain (decayed, capped)
            validR[i] += gpow[i] * w_valid * (double)valid;     // valid landing launches (capped)
            invR[i] += gpow[i] * w_inv * (double)invalid;       // invalid dispatches (uncapped penalty)
            missR[i] += gpow[i] * w_miss * (double)((long)move0.size() - valid);  // MISSED launches:
            // executed from an owned planet but landed on no planet (= launches - valid). Uncapped
            // aiming stick (the ships are thrown away); pushes phi+heading toward real targets.
            if (stage >= 2 && t >= x0) suppR[i] += gpow[i] * (-w_e * y_t);  // bend enemy growth down
            gpow[i] *= gamma;
            ppow[i] *= prod_decay;  // production-only per-step decay (front-load expansion)
            prev_p0[i] = p0;
            prev_p1[i] = p1;

            inv_acc[i] += invalid;
            lnch_acc[i] += (long)move0.size();
            valid_acc[i] += valid;
            step_acc[i] += 1;
            if (is_terminal(st[i], cfg_.rollout.episode_steps)) {
                active[i] = 0;
                outcome[i] = outcome_sign(st[i]);
                ep_len[i] = st[i].step;
            }
        });
        valid_buf[t] = torch::from_blob(validv.data(), {B}, fo).clone();
        if (std::none_of(active.begin(), active.end(), [](char c) { return c != 0; })) break;
    }
    for (int i = 0; i < B; ++i)
        if (active[i]) outcome[i] = outcome_sign(st[i]);  // hit the step cap

    // --- per-episode return on the non-negative outcome ladder (loss forfeits the dense return) ---
    std::vector<float> Rv(B);
    for (int i = 0; i < B; ++i) {
        double P = std::max(-prod_cap, std::min(prod_cap, prodR[i]));  // production reward, capped +/-
        double V = std::max(0.0, std::min(valid_cap, validR[i]));      // valid-launch reward, capped
        double o = outcome[i];
        // outcome only at stage >= 2; stage 1 is pure dispatch + production (no win/draw/loss reward)
        double O = (stage >= 2) ? (o > 0.0 ? win_bonus : (o < 0.0 ? -loss_penalty : 0.0)) : 0.0;
        Rv[i] = (float)(O + P + V - invR[i] - missR[i] + suppR[i]);
    }
    auto R = torch::from_blob(Rv.data(), {B}, fo).clone();
    std::vector<int64_t> gidv(B);
    for (int i = 0; i < B; ++i) gidv[i] = i / G;
    auto gid = torch::from_blob(gidv.data(), {B}, torch::TensorOptions().dtype(torch::kLong)).clone();
    auto adv_env = group_advantage(R, gid, N, cfg_.grpo.adv_eps, cfg_.grpo.whiten_advantage);

    // --- flatten valid transitions ---
    int Tu = last_t + 1;
    auto keep = valid_buf.narrow(0, 0, Tu).reshape({-1}).nonzero().squeeze(-1);  // (Nkeep,)
    auto adv_full = adv_env.unsqueeze(0).expand({Tu, B}).reshape({-1});
    auto sel = [&](torch::Tensor x) { return x.index_select(0, keep); };

    TrajectoryBatch tb;
    tb.entities = sel(ent_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, E, F}));
    tb.entity_mask = sel(em_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, E}));
    tb.action_mask = sel(am_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, E}));
    tb.globals = sel(gl_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, Gl}));
    tb.action = sel(act_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, E, nap}));
    tb.old_logp = sel(oldlogp_buf.narrow(0, 0, Tu).reshape({(long)Tu * B}));
    tb.advantage = adv_full.index_select(0, keep);  // ref_logp computed in the update
    tb.n_transitions = keep.size(0);

    double sumR = 0, sumlen = 0, wins = 0;
    long inv_sum = 0, lnch_sum = 0, valid_sum = 0, step_sum = 0;
    for (int i = 0; i < B; ++i) {
        sumR += Rv[i];
        sumlen += ep_len[i];
        wins += outcome[i] > 0 ? 1.0 : 0.0;
        inv_sum += inv_acc[i];
        lnch_sum += lnch_acc[i];
        valid_sum += valid_acc[i];
        step_sum += step_acc[i];
    }
    stats.mean_return = sumR / B;
    stats.mean_len = sumlen / B;
    stats.win_rate = wins / B;
    stats.mean_invalid = step_sum ? (double)inv_sum / step_sum : 0.0;
    stats.mean_launches = step_sum ? (double)lnch_sum / step_sum : 0.0;
    stats.mean_valid = step_sum ? (double)valid_sum / step_sum : 0.0;
    stats.episodes = B;
    stats.transitions = tb.n_transitions;
    return tb;
}

// GPU-batched rollout: the whole encode->act->step loop runs on-device via GpuEnv; obs/action/logp
// are offloaded to host each step (VRAM stays bounded). Reward/terminal/outcome mirror collect()
// + reward.hpp. Opponent is uniform across the batch this iteration (within a group the env shares
// world AND opponent, so the group return mean/std stays a valid GRPO baseline).
TrajectoryBatch GroupedRollout::collect_gpu(Agent& policy, RolloutStats& stats) {
    using torch::Tensor;
    const int G = cfg_.grpo.group_size, N = cfg_.grpo.num_groups, B = N * G;
    const int Ec = cfg_.rollout.planet_cap, F = N_ENTITY_FEATURES, Gl = N_GLOBAL_FEATURES;
    const int K = cfg_.model.fleets_per_planet, nap = 3 * K, T = cfg_.rollout.episode_steps;  // (dx,dy,phi)
    const double gamma = cfg_.grpo.gamma;
    const double w_prod = cfg_.rollout.prod_reward_weight, prod_cap = cfg_.rollout.prod_reward_cap,
                 prod_decay = cfg_.rollout.prod_reward_decay,
                 w_valid = cfg_.rollout.valid_launch_reward, valid_cap = cfg_.rollout.valid_reward_cap,
                 w_inv = cfg_.rollout.illegal_launch_penalty,
                 w_miss = cfg_.rollout.miss_launch_penalty, w_e = cfg_.rollout.enemy_growth_weight,
                 beta = cfg_.rollout.enemy_growth_ema, win_bonus = cfg_.rollout.win_bonus,
                 loss_penalty = cfg_.rollout.loss_penalty;
    const int x0 = cfg_.rollout.enemy_growth_warmup;
    const double act_thr = cfg_.model.act_threshold;
    const int stage = cur_stage_ ? cur_stage_ : cfg_.rollout.stage;
    const auto kF = torch::kFloat32;

    int opp = (stage == 1) ? 2 : (stage == 2) ? 1 : 0;
    if (stage >= 3) {
        double wr = cfg_.selfplay.mix_random, ws = cfg_.selfplay.mix_starter;
        std::uniform_real_distribution<double> u(0.0, wr + ws);
        opp = (u(rng_) < wr) ? 0 : 1;
    }

    std::vector<GameState> st(B);
    for (int j = 0; j < N; ++j) {
        const GameState& world = world_pool_[cursor_++ % world_pool_.size()];
        for (int g = 0; g < G; ++g) st[j * G + g] = world;
    }
    if (!gpu_ || gpu_->batch() != B || gpu_->planet_cap() != Ec) {
        GpuEnvConfig gc;
        gc.planet_cap = Ec;
        gc.fleet_cap = cfg_.rollout.fleet_cap;
        gc.episode_steps = T;
        gpu_ = std::make_unique<GpuEnv>(gc, device_);
    }
    gpu_->reset(st);

    auto fo = torch::TensorOptions().dtype(kF).device(device_);
    // Pinned host buffers so the per-step D2H offload is async (non_blocking) and does not stall the
    // CPU from enqueuing the next step's kernels -- keeps the GPU fed (compute-bound, not transfer).
    auto cpuf = torch::TensorOptions().dtype(kF).pinned_memory(device_.is_cuda());
    auto ent_buf = torch::zeros({T, B, Ec, F}, cpuf), em_buf = torch::zeros({T, B, Ec}, cpuf),
         am_buf = torch::zeros({T, B, Ec}, cpuf), gl_buf = torch::zeros({T, B, Gl}, cpuf),
         act_buf = torch::zeros({T, B, Ec, nap}, cpuf), oldlogp_buf = torch::zeros({T, B}, cpuf),
         valid_buf = torch::zeros({T, B}, cpuf);

    Tensor active = torch::ones({B}, fo), prodR = torch::zeros({B}, fo),
           validR = torch::zeros({B}, fo), invR = torch::zeros({B}, fo),
           missR = torch::zeros({B}, fo),
           suppR = torch::zeros({B}, fo), gpow = torch::ones({B}, fo), ppow = torch::ones({B}, fo),
           gbar = torch::zeros({B}, fo),
           outcome = torch::zeros({B}, fo), inv_sum = torch::zeros({B}, fo),
           lnch_sum = torch::zeros({B}, fo), valid_sum = torch::zeros({B}, fo),
           step_sum = torch::zeros({B}, fo);

    // {s0, s1, side0_alive, side1_alive} from the current env (reward.hpp scoring).
    auto settle = [&]() {
        const EnvTensors& E_ = gpu_->tensors();
        Tensor pa = E_.p_alive > 0.5, fa = E_.f_alive > 0.5;
        Tensor o0 = (E_.p_owner == 0.0) & pa, o1 = (E_.p_owner == 1.0) & pa;
        Tensor g0 = (E_.f_owner == 0.0) & fa, g1 = (E_.f_owner == 1.0) & fa;
        Tensor s0 = (E_.p_ships * o0.to(kF)).sum(1) + (E_.f_ships * g0.to(kF)).sum(1);
        Tensor s1 = (E_.p_ships * o1.to(kF)).sum(1) + (E_.f_ships * g1.to(kF)).sum(1);
        return std::make_tuple(s0, s1, o0.any(1) | g0.any(1), o1.any(1) | g1.any(1));
    };

    int last_t = 0;
    for (int t = 0; t < T; ++t) {
        last_t = t;
        auto obs = gpu_->encode(0);
        auto dec = policy.act(obs.entities, obs.entity_mask, obs.action_mask, obs.globals, false);
        ent_buf[t].copy_(obs.entities, /*non_blocking=*/true);
        em_buf[t].copy_(obs.entity_mask, true);
        am_buf[t].copy_(obs.action_mask, true);
        gl_buf[t].copy_(obs.globals, true);
        act_buf[t].copy_(dec.action, true);
        oldlogp_buf[t].copy_(dec.log_prob, true);
        valid_buf[t].copy_(active, true);  // keep this transition iff the env was active before stepping

        auto out = gpu_->step(dec.action, opp, act_thr);
        Tensor a = active;
        prodR = prodR + a * gpow * ppow * (w_prod * out.dprod_ego);  // production decayed per step
        validR = validR + a * gpow * (w_valid * out.valid);
        invR = invR + a * gpow * (w_inv * out.invalid);
        missR = missR + a * gpow * (w_miss * (out.launches - out.valid));  // MISSED launches (= executed
        // from an owned planet but landed on no planet): uncapped aiming stick, mirrors collect().
        Tensor y_t = out.dprod_enemy - gbar;
        gbar = (1.0 - beta) * gbar + beta * out.dprod_enemy;
        if (stage >= 2 && t >= x0) suppR = suppR - a * gpow * (w_e * y_t);
        gpow = gpow * gamma;
        ppow = ppow * prod_decay;  // production-only per-step decay (front-load expansion)
        inv_sum = inv_sum + a * out.invalid;
        lnch_sum = lnch_sum + a * out.launches;
        valid_sum = valid_sum + a * out.valid;
        step_sum = step_sum + a;

        auto [s0, s1, side0, side1] = settle();
        Tensor step_now = gpu_->tensors().step.to(kF);  // = t+1
        Tensor term = (step_now >= (double)(T - 2)) | (~(side0 & side1));
        Tensor newly = (active > 0.5) & term;
        outcome = torch::where(newly, torch::sign(s0 - s1), outcome);
        active = torch::where(term, torch::zeros_like(active), active);
        if ((t & 15) == 15 && active.sum().item<float>() == 0.0f) break;
    }
    {  // settle any env still alive at the step cap
        auto [s0, s1, side0, side1] = settle();
        outcome = torch::where(active > 0.5, torch::sign(s0 - s1), outcome);
    }

    Tensor P = prodR.clamp(-prod_cap, prod_cap), V = validR.clamp(0.0, valid_cap);
    Tensor O = (stage >= 2)
                   ? torch::where(outcome > 0.0, torch::full_like(outcome, win_bonus),
                                  torch::where(outcome < 0.0, torch::full_like(outcome, -loss_penalty),
                                               torch::zeros_like(outcome)))
                   : torch::zeros_like(outcome);
    Tensor R = O + P + V - invR - missR + suppR;  // (B,)
    Tensor gid = torch::arange(B, torch::TensorOptions().dtype(torch::kLong).device(device_))
                     .floor_divide(G);  // int64 group id (plain / would true-divide to float)
    Tensor adv_env = group_advantage(R, gid, N, cfg_.grpo.adv_eps, cfg_.grpo.whiten_advantage);

    if (device_.is_cuda()) torch::cuda::synchronize();  // ensure all async D2H offloads landed
    int Tu = last_t + 1;
    Tensor keep = valid_buf.narrow(0, 0, Tu).reshape({-1}).nonzero().squeeze(-1);
    Tensor adv_full = adv_env.to(torch::kCPU).unsqueeze(0).expand({Tu, B}).reshape({-1});
    auto sel = [&](torch::Tensor x) { return x.index_select(0, keep); };

    TrajectoryBatch tb;
    tb.entities = sel(ent_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, Ec, F}));
    tb.entity_mask = sel(em_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, Ec}));
    tb.action_mask = sel(am_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, Ec}));
    tb.globals = sel(gl_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, Gl}));
    tb.action = sel(act_buf.narrow(0, 0, Tu).reshape({(long)Tu * B, Ec, nap}));
    tb.old_logp = sel(oldlogp_buf.narrow(0, 0, Tu).reshape({(long)Tu * B}));
    tb.advantage = adv_full.index_select(0, keep);
    tb.n_transitions = keep.size(0);

    double step_total = step_sum.sum().item<float>();
    stats.mean_return = R.mean().item<float>();
    stats.mean_len = step_total / B;
    stats.win_rate = (outcome > 0.0).to(kF).mean().item<float>();
    stats.mean_invalid = step_total > 0 ? inv_sum.sum().item<float>() / step_total : 0.0;
    stats.mean_launches = step_total > 0 ? lnch_sum.sum().item<float>() / step_total : 0.0;
    stats.mean_valid = step_total > 0 ? valid_sum.sum().item<float>() / step_total : 0.0;
    stats.episodes = B;
    stats.transitions = tb.n_transitions;
    return tb;
}

}  // namespace ow
