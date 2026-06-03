// Native PPO trainer: rollout + GAE + clipped update, all in C++/LibTorch. Mirrors the
// Python PPOTrainer. The whole loop runs here; Python only supplies worlds and reads
// weights/stats out.
#pragma once
#include <torch/torch.h>

#include <deque>
#include <map>
#include <string>

#include "rl/batched_env.hpp"
#include "rl/policy.hpp"

namespace ow {

struct TrainerConfig {
    int num_envs = 256;
    int rollout_steps = 16;
    int total_steps = 2'000'000;
    int update_epochs = 4;
    int minibatches = 8;
    double gamma = 0.997, gae_lambda = 0.95, clip = 0.2;
    double ent_coef = 0.01, vf_coef = 0.5, max_grad_norm = 0.5, lr = 3e-4;
    int hidden = 128, max_entities = 64, angle_bins = 16, episode_steps = 500;
    std::vector<double> fractions = {0.25, 0.5, 0.75, 1.0};
    double reward_scale = 50.0, terminal_bonus = 1.0, reward_clip = 5.0;
    int selfplay_start_step = 200'000, snapshot_every = 200'000;
    double opp_random = 1.0, opp_starter = 1.0, opp_self = 2.0;
    std::string device = "cuda";
    uint64_t seed = 0;
};

class Trainer {
public:
    TrainerConfig cfg;
    torch::Device device;
    EntityPolicy policy{nullptr}, frozen{nullptr};
    std::shared_ptr<torch::optim::Adam> opt;
    BatchedEnv env;
    int n_entity_features, n_global_features, actions_per_entity;
    long global_step = 0;
    int updates = 0;
    std::deque<float> ep_returns, ep_wins;

    Trainer(TrainerConfig c)
        : cfg(c),
          device(c.device == "cuda" && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
          env(c.num_envs, c.max_entities, c.angle_bins, c.fractions,
              Config{c.episode_steps, 6.0, 4.0}, c.episode_steps, c.reward_scale,
              c.terminal_bonus, c.reward_clip, c.seed) {
        torch::manual_seed(c.seed);
        n_entity_features = N_ENTITY_FEATURES;
        n_global_features = N_GLOBAL_FEATURES;
        actions_per_entity = 1 + c.angle_bins * (int)c.fractions.size();
        policy = EntityPolicy(n_entity_features, n_global_features, actions_per_entity, c.hidden);
        policy->to(device);
        opt = std::make_shared<torch::optim::Adam>(policy->parameters(),
                                                   torch::optim::AdamOptions(c.lr).eps(1e-5));
        env.set_opp_weights(c.opp_random, c.opp_starter, c.opp_self);
        env.set_selfplay_available(false);
    }

    void set_world_pool(std::vector<GameState> pool) { env.set_world_pool(std::move(pool)); }

    void train(long total_steps) {
        env.reset_all();
        auto obs = to_device(env.observe(0));
        std::vector<float> ep_ret(cfg.num_envs, 0.f);

        int B = cfg.num_envs, T = cfg.rollout_steps, E = cfg.max_entities;
        int F = n_entity_features, G = n_global_features;
        auto fopt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto lopt = torch::TensorOptions().dtype(torch::kLong).device(device);
        // Rollout buffers on device.
        auto b_ent = torch::zeros({T, B, E, F}, fopt);
        auto b_em = torch::zeros({T, B, E}, fopt);
        auto b_am = torch::zeros({T, B, E}, fopt);
        auto b_gl = torch::zeros({T, B, G}, fopt);
        auto b_act = torch::zeros({T, B, E}, lopt);
        auto b_logp = torch::zeros({T, B}, fopt);
        auto b_val = torch::zeros({T, B}, fopt);
        auto b_rew = torch::zeros({T, B}, fopt);
        auto b_done = torch::zeros({T, B}, fopt);

        long target = global_step + total_steps;
        while (global_step < target) {
            for (int t = 0; t < T; ++t) {
                b_ent[t] = obs.entities; b_em[t] = obs.entity_mask;
                b_am[t] = obs.action_mask; b_gl[t] = obs.globals;
                torch::Tensor action, logp, value;
                {
                    torch::NoGradGuard ng;
                    auto [logits, val] = policy->forward(obs.entities, obs.entity_mask,
                                                         obs.action_mask, obs.globals);
                    action = sample_actions(logits);
                    logp = summed_logprob(logits, action, obs.entity_mask);
                    value = val;
                }
                // Self-play opponent action (frozen policy) if active.
                torch::Tensor opp_action;
                if (env_selfplay_active()) {
                    auto obs1 = to_device(env.observe(1));
                    torch::NoGradGuard ng;
                    auto [l1, v1] = frozen->forward(obs1.entities, obs1.entity_mask,
                                                    obs1.action_mask, obs1.globals);
                    opp_action = sample_actions(l1).to(torch::kCPU);
                }
                auto [reward, done] = env.step(action.to(torch::kCPU), opp_action);
                env.reset_done(done);

                b_act[t] = action; b_logp[t] = logp; b_val[t] = value;
                b_rew[t] = reward.to(device); b_done[t] = done.to(torch::kFloat32).to(device);

                // stats
                auto rcpu = reward.accessor<float, 1>();
                auto dcpu = done.to(torch::kCPU);
                auto da = dcpu.accessor<bool, 1>();
                for (int i = 0; i < B; ++i) {
                    ep_ret[i] += rcpu[i];
                    if (da[i]) {
                        ep_returns.push_back(ep_ret[i]);
                        ep_wins.push_back(rcpu[i] > 0 ? 1.f : 0.f);  // terminal bonus sign proxy
                        if (ep_returns.size() > 200) { ep_returns.pop_front(); ep_wins.pop_front(); }
                        ep_ret[i] = 0.f;
                    }
                }
                global_step += B;
                obs = to_device(env.observe(0));
            }

            // Bootstrap value.
            torch::Tensor last_val;
            { torch::NoGradGuard ng;
              last_val = std::get<1>(policy->forward(obs.entities, obs.entity_mask,
                                                     obs.action_mask, obs.globals)); }
            auto [adv, ret] = compute_gae(b_rew, b_val, b_done, last_val);
            ppo_update(b_ent, b_em, b_am, b_gl, b_act, b_logp, adv, ret);
            updates++;
            maybe_snapshot();
            if (updates % 1 == 0) log_progress();
        }
    }

    // Weights with Python EntityPolicy state_dict keys, as CPU tensors.
    std::map<std::string, torch::Tensor> get_state_dict() {
        std::map<std::string, torch::Tensor> w;
        auto put = [&](const std::string& k, torch::nn::Linear& lin) {
            w[k + ".weight"] = lin->weight.detach().to(torch::kCPU).contiguous();
            w[k + ".bias"] = lin->bias.detach().to(torch::kCPU).contiguous();
        };
        put("entity_encoder.0", policy->ent0); put("entity_encoder.2", policy->ent2);
        put("context_encoder.0", policy->ctx0); put("context_encoder.2", policy->ctx2);
        put("actor.0", policy->act0); put("actor.2", policy->act2);
        put("critic.0", policy->crit0); put("critic.2", policy->crit2);
        return w;
    }

    // Greedy per-entity classes for player 0 on a single state (for export parity tests).
    std::vector<long> greedy_classes(const GameState& s) {
        int E = cfg.max_entities, F = n_entity_features, G = n_global_features;
        std::vector<float> ent((size_t)E * F), em(E), am(E), gl(G);
        std::vector<long> rid(E), rsh(E);
        encode_obs(s, 0, E, ent.data(), em.data(), am.data(), gl.data(), rid.data(), rsh.data(),
                   cfg.episode_steps);
        auto fo = torch::TensorOptions().dtype(torch::kFloat32);
        auto t_ent = torch::from_blob(ent.data(), {1, E, F}, fo).to(device);
        auto t_em = torch::from_blob(em.data(), {1, E}, fo).to(device);
        auto t_am = torch::from_blob(am.data(), {1, E}, fo).to(device);
        auto t_gl = torch::from_blob(gl.data(), {1, G}, fo).to(device);
        torch::NoGradGuard ng;
        auto [logits, val] = policy->forward(t_ent, t_em, t_am, t_gl);
        auto a = std::get<1>(logits.max(-1)).to(torch::kCPU).view({E}).contiguous();
        auto acc = a.accessor<int64_t, 1>();
        std::vector<long> out(E);
        for (int i = 0; i < E; ++i) out[i] = acc[i];
        return out;
    }

    double recent_winrate() {
        if (ep_wins.empty()) return -1.0;
        double s = 0; for (float w : ep_wins) s += w; return s / ep_wins.size();
    }
    double recent_return() {
        if (ep_returns.empty()) return 0.0;
        double s = 0; for (float r : ep_returns) s += r; return s / ep_returns.size();
    }

private:
    bool selfplay_on = false;
    bool env_selfplay_active() { return selfplay_on && frozen; }

    ObsTensors to_device(const ObsTensors& o) {
        ObsTensors d;
        d.entities = o.entities.to(device); d.entity_mask = o.entity_mask.to(device);
        d.action_mask = o.action_mask.to(device); d.globals = o.globals.to(device);
        return d;
    }

    std::pair<torch::Tensor, torch::Tensor> compute_gae(const torch::Tensor& rew,
                                                        const torch::Tensor& val,
                                                        const torch::Tensor& done,
                                                        const torch::Tensor& last_val) {
        int T = rew.size(0);
        auto adv = torch::zeros_like(rew);
        torch::Tensor last_gae = torch::zeros_like(last_val);
        for (int t = T - 1; t >= 0; --t) {
            torch::Tensor next_val = (t == T - 1) ? last_val : val[t + 1];
            auto nonterminal = 1.0 - done[t];
            auto delta = rew[t] + cfg.gamma * next_val * nonterminal - val[t];
            last_gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last_gae;
            adv[t] = last_gae;
        }
        return {adv, adv + val};
    }

    void ppo_update(torch::Tensor ent, torch::Tensor em, torch::Tensor am, torch::Tensor gl,
                    torch::Tensor act, torch::Tensor old_logp, torch::Tensor adv,
                    torch::Tensor ret) {
        int T = ent.size(0), B = ent.size(1), E = ent.size(2), F = ent.size(3), G = gl.size(2);
        long N = (long)T * B;
        ent = ent.reshape({N, E, F}); em = em.reshape({N, E}); am = am.reshape({N, E});
        gl = gl.reshape({N, G}); act = act.reshape({N, E});
        old_logp = old_logp.reshape({N}); adv = adv.reshape({N}); ret = ret.reshape({N});
        adv = (adv - adv.mean()) / (adv.std() + 1e-8);

        long mb = N / cfg.minibatches;
        for (int epoch = 0; epoch < cfg.update_epochs; ++epoch) {
            auto perm = torch::randperm(N, torch::TensorOptions().dtype(torch::kLong).device(device));
            for (int b = 0; b < cfg.minibatches; ++b) {
                auto mbidx = perm.narrow(0, b * mb, mb);
                auto [logits, value] = policy->forward(ent.index({mbidx}), em.index({mbidx}),
                                                       am.index({mbidx}), gl.index({mbidx}));
                auto logp = summed_logprob(logits, act.index({mbidx}), em.index({mbidx}));
                auto entropy = entropy_sum(logits, em.index({mbidx})).mean();
                auto a = adv.index({mbidx});
                auto ratio = torch::exp(logp - old_logp.index({mbidx}));
                auto pg1 = -a * ratio;
                auto pg2 = -a * torch::clamp(ratio, 1 - cfg.clip, 1 + cfg.clip);
                auto pg_loss = torch::max(pg1, pg2).mean();
                auto v_loss = 0.5 * (value - ret.index({mbidx})).pow(2).mean();
                auto loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy;
                opt->zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(policy->parameters(), cfg.max_grad_norm);
                opt->step();
            }
        }
    }

    void maybe_snapshot() {
        if (global_step < cfg.selfplay_start_step) return;
        long bucket = global_step / std::max<long>(1, cfg.snapshot_every);
        if (bucket > last_snap_bucket) {
            last_snap_bucket = bucket;
            frozen = EntityPolicy(n_entity_features, n_global_features, actions_per_entity, cfg.hidden);
            std::stringstream ss;
            torch::save(policy, ss);
            torch::load(frozen, ss);
            frozen->to(device);
            frozen->eval();
            selfplay_on = true;
            env.set_selfplay_available(true);
        }
    }

    void log_progress() {
        printf("upd %4d | step %9ld | ep_ret %7.3f | winrate %5.2f | selfplay %d\n", updates,
               global_step, recent_return(), recent_winrate(), (int)selfplay_on);
        fflush(stdout);
    }

    long last_snap_bucket = -1;
};

}  // namespace ow
