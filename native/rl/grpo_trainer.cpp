#include "rl/grpo_trainer.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "rl/algo/grpo.hpp"

namespace ow {
namespace fs = std::filesystem;

GrpoTrainer::GrpoTrainer(const TrainConfig& cfg, std::string run_dir)
    : cfg_(cfg),
      device_(cfg.device == "cuda" && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      run_dir_(std::move(run_dir)),
      rollout_(cfg, device_),
      arena_(cfg.model.max_entities, cfg.model.angle_bins, cfg.model.fractions, cfg.model.hidden,
             cfg.rollout.episode_steps, cfg.device, cfg.model.target_mode) {
    torch::manual_seed(cfg.seed);
    policy_ = Agent(cfg.model, device_);
}

void GrpoTrainer::set_world_pool(std::vector<GameState> train, std::vector<GameState> eval) {
    rollout_.set_world_pool(std::move(train));
    eval_worlds_ = std::move(eval);
}

void GrpoTrainer::load_init(const std::map<std::string, torch::Tensor>& sd) {
    policy_.load_state_dict(sd);
    reference_ = policy_.clone_frozen();  // fixed KL anchor = the BC policy
    opt_ = std::make_shared<torch::optim::Adam>(
        policy_.net->parameters(), torch::optim::AdamOptions(cfg_.optim.lr).eps(cfg_.optim.adam_eps));
}

CheckpointMeta GrpoTrainer::meta() const {
    CheckpointMeta m;
    m.n_entity_features = cfg_.model.n_entity_features;
    m.n_global_features = cfg_.model.n_global_features;
    m.hidden = cfg_.model.hidden;
    m.angle_bins = cfg_.model.angle_bins;
    m.max_entities = cfg_.model.max_entities;
    m.num_fracs = (int)cfg_.model.fractions.size();
    m.episode_steps = cfg_.rollout.episode_steps;
    m.target_mode = cfg_.model.target_mode ? 1 : 0;
    m.fractions = cfg_.model.fractions;
    return m;
}

GrpoTrainer::UpdateStats GrpoTrainer::update(const TrajectoryBatch& tb) {
    UpdateStats s;
    long Nn = tb.n_transitions;
    int mb = cfg_.grpo.minibatches;
    long mbsize = Nn / mb;
    if (mbsize == 0) return s;
    s.adv_mean = tb.advantage.mean().item<float>();
    s.adv_std = tb.advantage.std().item<float>();
    int nsteps = 0;
    for (int e = 0; e < cfg_.grpo.update_epochs; ++e) {
        auto perm = torch::randperm(Nn, torch::TensorOptions().dtype(torch::kLong));
        for (int b = 0; b < mb; ++b) {
            auto mi = perm.narrow(0, (long)b * mbsize, mbsize);
            auto ent = tb.entities.index_select(0, mi).to(device_);
            auto em = tb.entity_mask.index_select(0, mi).to(device_);
            auto am = tb.action_mask.index_select(0, mi).to(device_);
            auto gl = tb.globals.index_select(0, mi).to(device_);
            auto act = tb.action.index_select(0, mi).to(device_);
            auto oldlp = tb.old_logp.index_select(0, mi).to(device_);
            auto adv = tb.advantage.index_select(0, mi).to(device_);

            auto sc = policy_.evaluate(ent, em, am, gl, act);
            torch::Tensor reflp;  // reference log-prob of the (fixed) taken action -- no grad
            { torch::NoGradGuard ng; reflp = reference_.evaluate(ent, em, am, gl, act).log_prob; }
            auto pol = policy_surrogate(sc.log_prob, oldlp, adv, cfg_.grpo.clip);
            auto kl = kl_penalty(sc.log_prob, reflp);
            auto ent_b = sc.entropy.mean();
            auto loss = pol + cfg_.grpo.kl_beta * kl - cfg_.grpo.ent_coef * ent_b;

            opt_->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(policy_.net->parameters(), cfg_.optim.max_grad_norm);
            opt_->step();

            torch::NoGradGuard ng;
            auto ratio = torch::exp(sc.log_prob - oldlp);
            s.total += loss.item<float>();
            s.policy += pol.item<float>();
            s.kl += kl.item<float>();
            s.entropy += ent_b.item<float>();
            s.approx_kl += (oldlp - sc.log_prob).mean().item<float>();
            s.clipfrac += (torch::abs(ratio - 1.0) > cfg_.grpo.clip).to(torch::kFloat32).mean().item<float>();
            nsteps++;
        }
    }
    if (nsteps) {
        s.total /= nsteps; s.policy /= nsteps; s.kl /= nsteps; s.entropy /= nsteps;
        s.approx_kl /= nsteps; s.clipfrac /= nsteps;
    }
    return s;
}

void GrpoTrainer::log_and_eval(double sps, const RolloutStats& rs, const UpdateStats& us) {
    auto sd = policy_.state_dict();
    arena_.load_p0(sd);
    auto r_rand = arena_.play(eval_worlds_, 0, true, 12345);
    auto r_start = arena_.play(eval_worlds_, 1, true, 12345);
    int n = (int)eval_worlds_.size();
    double wr_rand = n ? 100.0 * r_rand.p0_wins / n : 0.0;
    double wr_start = n ? 100.0 * r_start.p0_wins / n : 0.0;

    fs::create_directories(run_dir_);
    std::string csv = run_dir_ + "/metrics.csv";
    if (!csv_open_) {
        std::ofstream h(csv, std::ios::trunc);
        h << "step,iter,sps,ep_return,wr_random,wr_starter,margin_starter,ep_len,"
             "loss_total,loss_policy,loss_kl,loss_entropy,adv_mean,adv_std,approx_kl,clipfrac\n";
        csv_open_ = true;
    }
    {
        std::ofstream f(csv, std::ios::app);
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "%ld,%d,%.0f,%.3f,%.2f,%.2f,%.1f,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f\n",
                 global_step_, iter_, sps, rs.mean_return, wr_rand, wr_start, r_start.mean_margin,
                 r_start.mean_len, us.total, us.policy, us.kl, us.entropy, us.adv_mean, us.adv_std,
                 us.approx_kl, us.clipfrac);
        f << buf;
    }

    ow::write_checkpoint(run_dir_ + "/last.owc", sd, meta());
    const char* flag = "";
    if (wr_start > best_wr_starter_) {
        best_wr_starter_ = wr_start;
        ow::write_checkpoint(run_dir_ + "/best.owc", sd, meta());
        flag = " *best";
    }
    printf("step %9ld | vsRAND %5.1f%% | vsSTART %5.1f%% | ret %7.2f | loss %7.3f "
           "(pol %.3f kl %.3f H %.3f) | sps %5.0f%s\n",
           global_step_, wr_rand, wr_start, rs.mean_return, us.total, us.policy, us.kl, us.entropy,
           sps, flag);
    fflush(stdout);
}

void GrpoTrainer::train() {
    fs::create_directories(run_dir_);
    {
        std::ofstream c(run_dir_ + "/config.txt", std::ios::trunc);
        c << "total_steps=" << cfg_.total_steps << "\nnum_envs=" << cfg_.num_envs()
          << " (group_size=" << cfg_.grpo.group_size << " num_groups=" << cfg_.grpo.num_groups
          << ")\nkl_beta=" << cfg_.grpo.kl_beta << " clip=" << cfg_.grpo.clip
          << " ent_coef=" << cfg_.grpo.ent_coef << " gamma=" << cfg_.grpo.gamma
          << "\ndense_weight=" << cfg_.grpo.dense_weight
          << " outcome_weight=" << cfg_.grpo.outcome_weight << "\nlr=" << cfg_.optim.lr
          << " target_mode=" << cfg_.model.target_mode << " hidden=" << cfg_.model.hidden
          << "\nstage=" << cfg_.rollout.stage
          << " prod_reward_weight=" << cfg_.rollout.prod_reward_weight
          << " launch_hit_reward=" << cfg_.rollout.launch_hit_reward
          << " launch_miss_penalty=" << cfg_.rollout.launch_miss_penalty
          << " loss_forfeit=" << cfg_.rollout.loss_forfeit
          << " n_entity_features=" << cfg_.model.n_entity_features << "\n";
    }
    printf("GRPO: %d envs (%dx%d) | kl_beta=%.3f | total_steps=%ld | run_dir=%s\n",
           cfg_.num_envs(), cfg_.grpo.num_groups, cfg_.grpo.group_size, cfg_.grpo.kl_beta,
           cfg_.total_steps, run_dir_.c_str());

    long next_log = 0;  // first iteration logs the warm-started baseline
    while (global_step_ < cfg_.total_steps) {
        auto t0 = std::chrono::steady_clock::now();
        RolloutStats rs;
        auto tb = rollout_.collect(policy_, rs);
        auto us = update(tb);
        global_step_ += rs.transitions;
        iter_++;
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double sps = rs.transitions / std::max(dt, 1e-9);
        if (global_step_ >= next_log) {
            log_and_eval(sps, rs, us);
            next_log = ((global_step_ / 200000) + 1) * 200000;
        }
    }
    printf("done. best vsSTART=%.1f%% | results in %s\n", best_wr_starter_, run_dir_.c_str());
}

}  // namespace ow
