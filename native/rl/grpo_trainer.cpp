#include "rl/grpo_trainer.hpp"

#include <chrono>
#include <cmath>
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
      arena_cont_(cfg.model.max_entities, cfg.model.fleets_per_planet, cfg.model.act_threshold,
                  cfg.rollout.episode_steps, device_) {
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

void GrpoTrainer::init_scratch() {
    // No BC: the policy keeps its random init. reference_ is set (for the uniform update path) but
    // the KL term is off (cfg_.grpo.kl_beta == 0), so it never anchors to the random weights.
    reference_ = policy_.clone_frozen();
    opt_ = std::make_shared<torch::optim::Adam>(
        policy_.net->parameters(), torch::optim::AdamOptions(cfg_.optim.lr).eps(cfg_.optim.adam_eps));
}

CheckpointMeta GrpoTrainer::meta() const {
    // Continuous v4: the .owc records the obs/trunk shape; the actor head shape is implied by
    // fleets_per_planet and rebuilt from ModelConfig on load (legacy discrete fields stay default).
    CheckpointMeta m;
    m.n_entity_features = cfg_.model.n_entity_features;
    m.n_global_features = cfg_.model.n_global_features;
    m.hidden = cfg_.model.hidden;
    m.max_entities = cfg_.model.max_entities;
    m.episode_steps = cfg_.rollout.episode_steps;
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
    const int E = (int)tb.entities.size(1), nap = cfg_.model.n_action_params();  // E = actual obs dim
    const double kHalfLog2PiE = 1.4189385332046727;  // entropy const per component
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
            auto pol = policy_surrogate(sc.log_prob, oldlp, adv, cfg_.grpo.clip);
            // PPO critic: regress V(s) onto the GAE value target (returns). This is the dense,
            // low-variance signal that GRPO's group baseline lacks (docs/bc_vs_grpo.pdf).
            torch::Tensor vloss = torch::zeros({}, sc.log_prob.options());
            if (cfg_.model.use_value_head) {
                auto ret = tb.returns.index_select(0, mi).to(device_);
                vloss = torch::mse_loss(sc.value, ret);
            }
            // Skip the reference forward entirely when there is no KL anchor (kl_beta==0, e.g. from
            // scratch) -- it was a full extra net forward per minibatch computed only to be * 0.
            torch::Tensor kl = torch::zeros({}, sc.log_prob.options());
            if (cfg_.grpo.kl_beta > 0.0) {
                torch::NoGradGuard ng;
                auto reflp = reference_.evaluate(ent, em, am, gl, act).log_prob;
                kl = kl_penalty(sc.log_prob, reflp);
            }
            // per-component entropy (scale-invariant in E,K) so ent_coef behaves like standard PPO
            // -- the raw sum over 400 components would otherwise swamp the policy gradient.
            auto ent_b = sc.entropy.mean() / (double)(E * nap);
            auto loss = pol + cfg_.ppo.vf_coef * vloss + cfg_.grpo.kl_beta * kl
                      - cfg_.grpo.ent_coef * ent_b;

            opt_->zero_grad();
            loss.backward();
            double gnorm = torch::nn::utils::clip_grad_norm_(policy_.net->parameters(),
                                                             cfg_.optim.max_grad_norm);
            if (std::isfinite(gnorm)) opt_->step();  // skip a poisoned (non-finite) gradient step

            torch::NoGradGuard ng;
            auto ratio = torch::exp(sc.log_prob - oldlp);
            s.total += loss.item<float>();
            s.policy += pol.item<float>();
            s.vf += vloss.item<float>();
            s.kl += kl.item<float>();
            s.entropy += ent_b.item<float>();  // per-component entropy (logstd_mean + const)
            double mean_logstd = ent_b.item<float>() - kHalfLog2PiE;  // back out mean log-std
            s.sigma += std::exp(mean_logstd);
            s.approx_kl += (oldlp - sc.log_prob).mean().item<float>();
            s.clipfrac += (torch::abs(ratio - 1.0) > cfg_.grpo.clip).to(torch::kFloat32).mean().item<float>();
            nsteps++;
        }
    }
    if (nsteps) {
        s.total /= nsteps; s.policy /= nsteps; s.kl /= nsteps; s.entropy /= nsteps;
        s.sigma /= nsteps; s.approx_kl /= nsteps; s.clipfrac /= nsteps; s.vf /= nsteps;
    }
    return s;
}

void GrpoTrainer::write_metrics_row(double sps, const RolloutStats& rs, const UpdateStats& us,
                                    double wr_rand, double wr_start, double margin, double eplen) {
    fs::create_directories(run_dir_);
    std::string csv = run_dir_ + "/metrics.csv";
    if (!csv_open_) {
        std::ofstream h(csv, std::ios::trunc);
        h << "step,iter,sps,ep_return,wr_random,wr_starter,margin_starter,ep_len,valid_per_step,"
             "invalid_per_step,launch_per_step,r_outcome,r_capture,r_dispatch,r_prod,"
             "loss_total,loss_policy,loss_kl,entropy,sigma,"
             "adv_mean,adv_std,approx_kl,clipfrac,loss_vf\n";
        csv_open_ = true;
    }
    // best-effort: if Excel holds a write-lock on the CSV this open just fails and the row is skipped
    // (the trainer never blocks); use progress.log for the live, lock-free view.
    std::ofstream f(csv, std::ios::app);
    char buf[800];
    snprintf(buf, sizeof(buf),
             "%ld,%d,%.0f,%.3f,%.2f,%.2f,%.1f,%.0f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,"
             "%.4f,%.4f,%.4f,%.4f,%.4f,"
             "%.4f,%.4f,%.4f,%.3f,%.4f\n",
             global_step_, iter_, sps, rs.mean_return, wr_rand, wr_start, margin, eplen,
             rs.mean_valid, rs.mean_invalid, rs.mean_launches,
             rs.r_outcome, rs.r_capture, rs.r_dispatch, rs.r_prod_milestone,
             us.total, us.policy, us.kl, us.entropy,
             us.sigma, us.adv_mean, us.adv_std, us.approx_kl, us.clipfrac, us.vf);
    f << buf;
}

void GrpoTrainer::heartbeat(double sps, const RolloutStats& rs, const UpdateStats& us) {
    // per-iter row reuses the last eval's win-rates (a step function between evals)
    write_metrics_row(sps, rs, us, last_wr_rand_, last_wr_start_, last_margin_, last_eplen_);
    char line[640];
    snprintf(line, sizeof(line),
             "step %9ld it%4d s%d | ret %8.2f R[o%6.0f c%5.0f d%4.0f p%4.0f] | "
             "valid/st %.2f inv/st %.2f lnch/st %.2f | "
             "H %5.2f sig %.3f | loss %7.3f (pol %.3f vf %.3f) kl %.3f cf %.2f | sps %5.0f\n",
             global_step_, iter_, cur_stage_, rs.mean_return, rs.r_outcome, rs.r_capture,
             rs.r_dispatch, rs.r_prod_milestone, rs.mean_valid, rs.mean_invalid,
             rs.mean_launches, us.entropy, us.sigma, us.total, us.policy, us.vf, us.approx_kl,
             us.clipfrac, sps);
    fputs(line, stdout);
    fflush(stdout);
    std::ofstream plog(run_dir_ + "/progress.log", std::ios::app);
    plog << line;
}

void GrpoTrainer::log_and_eval(double sps, const RolloutStats& rs, const UpdateStats& us,
                              const std::string& ckpt_tag) {
    auto sd = policy_.state_dict();
    auto r_rand = arena_cont_.play(policy_, eval_worlds_, 0, 12345);
    auto r_start = arena_cont_.play(policy_, eval_worlds_, 1, 12345);
    int n = (int)eval_worlds_.size();
    double wr_rand = n ? 100.0 * r_rand.p0_wins / n : 0.0;
    double wr_start = n ? 100.0 * r_start.p0_wins / n : 0.0;
    last_wr_rand_ = wr_rand; last_wr_start_ = wr_start;        // carry into the heartbeat rows
    last_margin_ = r_start.mean_margin; last_eplen_ = r_start.mean_len;

    write_metrics_row(sps, rs, us, wr_rand, wr_start, r_start.mean_margin, r_start.mean_len);

    ow::write_checkpoint(run_dir_ + "/last.owc", sd, meta());
    if (!ckpt_tag.empty()) {  // per-iteration progression checkpoint for the behavior visualizations
        fs::create_directories(run_dir_ + "/ckpt");
        ow::write_checkpoint(run_dir_ + "/ckpt/" + ckpt_tag + ".owc", sd, meta());
    }
    const char* flag = "";
    if (wr_start > best_wr_starter_) {
        best_wr_starter_ = wr_start;
        ow::write_checkpoint(run_dir_ + "/best.owc", sd, meta());
        flag = " *best";
    }
    char line[640];
    snprintf(line, sizeof(line),
             "step %9ld it%4d s%d | vsRAND %5.1f%% | vsSTART %5.1f%% | ret %8.2f "
             "R[o%6.0f c%5.0f d%4.0f p%4.0f] | H %5.2f sig %.3f | "
             "valid/st %.2f inv/st %.2f lnch/st %.2f | loss %7.3f (pol %.3f vf %.3f) kl %.3f cf %.2f | "
             "sps %5.0f%s\n",
             global_step_, iter_, cur_stage_, wr_rand, wr_start, rs.mean_return,
             rs.r_outcome, rs.r_capture, rs.r_dispatch, rs.r_prod_milestone, us.entropy, us.sigma,
             rs.mean_valid, rs.mean_invalid, rs.mean_launches, us.total, us.policy, us.vf, us.approx_kl,
             us.clipfrac, sps, flag);
    fputs(line, stdout);
    fflush(stdout);
    // Live, lock-free progress view: tail -f runs/.../progress.log  (safe to watch while training;
    // do NOT open metrics.csv in Excel mid-run -- that can lock out the CSV append above).
    std::ofstream plog(run_dir_ + "/progress.log", std::ios::app);  // truncated once at train() start
    plog << line;
}

void GrpoTrainer::train() {
    fs::create_directories(run_dir_);
    { std::ofstream(run_dir_ + "/progress.log", std::ios::trunc); }  // start each run with a fresh log
    {
        std::ofstream c(run_dir_ + "/config.txt", std::ios::trunc);
        c << "total_steps=" << cfg_.total_steps << "\nnum_envs=" << cfg_.num_envs()
          << " (group_size=" << cfg_.grpo.group_size << " num_groups=" << cfg_.grpo.num_groups
          << ")\n# actor: continuous squashed-Gaussian over E x 3K (dx,dy,phi per fleet)\n"
          << "max_entities(E)=" << cfg_.model.max_entities
          << " fleets_per_planet(K)=" << cfg_.model.fleets_per_planet
          << " hidden(d)=" << cfg_.model.hidden << " d_g=" << cfg_.model.d_g
          << " use_glu=" << cfg_.model.use_glu
          << " std_state_dependent=" << cfg_.model.std_state_dependent
          << " action_mask=" << cfg_.model.action_mask_policy
          << " target_actor=" << cfg_.model.target_actor
          << " act_threshold=" << cfg_.model.act_threshold
          << " logstd_clip=[" << cfg_.model.logstd_min << "," << cfg_.model.logstd_max << "]"
          << " logstd_max_end=" << cfg_.model.logstd_max_end
          << " logstd_max_post=" << cfg_.model.logstd_max_post
          << " sigma_decay_iters=" << cfg_.model.sigma_decay_iters
          << " n_entity_features=" << cfg_.model.n_entity_features << "\n# grpo\n"
          << "kl_beta=" << cfg_.grpo.kl_beta << " clip=" << cfg_.grpo.clip
          << " ent_coef=" << cfg_.grpo.ent_coef << " gamma=" << cfg_.grpo.gamma
          << " update_epochs=" << cfg_.grpo.update_epochs << " minibatches=" << cfg_.grpo.minibatches
          << " dense_weight=" << cfg_.grpo.dense_weight
          << " outcome_weight=" << cfg_.grpo.outcome_weight << " lr=" << cfg_.optim.lr
          << "\n# algo: " << cfg_.algo
          << (cfg_.is_ppo() ? "" : " (group baseline; no critic)")
          << " use_value_head=" << cfg_.model.use_value_head
          << " gae_lambda=" << cfg_.ppo.gae_lambda << " ppo_gamma=" << cfg_.ppo.gamma
          << " vf_coef=" << cfg_.ppo.vf_coef
          << "\n# reward\n"
          << "curriculum=" << cfg_.rollout.curriculum
          << " stage1_frac=" << cfg_.rollout.stage1_frac
          << " stage2_frac=" << cfg_.rollout.stage2_frac
          << " stage=" << cfg_.rollout.stage
          << " prod_reward_weight=" << cfg_.rollout.prod_reward_weight
          << " prod_reward_cap=" << cfg_.rollout.prod_reward_cap
          << " prod_reward_decay=" << cfg_.rollout.prod_reward_decay
          << " valid_launch_reward=" << cfg_.rollout.valid_launch_reward
          << " valid_reward_cap=" << cfg_.rollout.valid_reward_cap
          << " illegal_launch_penalty=" << cfg_.rollout.illegal_launch_penalty
          << " miss_launch_penalty=" << cfg_.rollout.miss_launch_penalty
          << " win_bonus=" << cfg_.rollout.win_bonus
          << " loss_penalty=" << cfg_.rollout.loss_penalty
          << " enemy_growth_weight=" << cfg_.rollout.enemy_growth_weight
          << " enemy_growth_warmup=" << cfg_.rollout.enemy_growth_warmup
          << " enemy_growth_ema=" << cfg_.rollout.enemy_growth_ema
          << "\n# set-ups/1.md event reward + curriculum\n"
          << "win_decay=" << cfg_.rollout.win_decay
          << " capture_reward=" << cfg_.rollout.capture_reward
          << " dispatch_reward=" << cfg_.rollout.dispatch_reward
          << " dispatch_reward_count=" << cfg_.rollout.dispatch_reward_count
          << " fleet_hit_base=" << cfg_.rollout.fleet_hit_base
          << " fleet_hit_ship_weight=" << cfg_.rollout.fleet_hit_ship_weight
          << " fleet_hit_cap=" << cfg_.rollout.fleet_hit_cap
          << " ppo_reward_scale=" << cfg_.rollout.ppo_reward_scale
          << " selfplay_prob=" << cfg_.rollout.selfplay_prob
          << " decay_start_step=" << cfg_.rollout.decay_start_step
          << " prod_milestone_reward=" << cfg_.rollout.prod_milestone_reward
          << " prod_milestone_base=" << cfg_.rollout.prod_milestone_base
          << " n_res_blocks=" << cfg_.model.n_res_blocks
          << " stage_iters=[" << cfg_.stage1_iters << "," << cfg_.stage2_iters << ","
          << cfg_.stage3_iters << "] total_iters=" << cfg_.total_iters << "\n";
    }
    printf("GRPO: %d envs (%dx%d) | kl_beta=%.3f | total_steps=%ld | run_dir=%s\n",
           cfg_.num_envs(), cfg_.grpo.num_groups, cfg_.grpo.group_size, cfg_.grpo.kl_beta,
           cfg_.total_steps, run_dir_.c_str());

    long next_log = 0;  // first iteration logs the warm-started baseline
    const double ls_max_start = cfg_.model.logstd_max, ls_max_end = cfg_.model.logstd_max_end;
    const bool iter_mode = cfg_.total_iters > 0;  // ITERATION-driven schedule (curriculum + ckpts)
    while (iter_mode ? (iter_ < cfg_.total_iters) : (global_step_ < cfg_.total_steps)) {
        auto t0 = std::chrono::steady_clock::now();
        // Exploration anneal: force the log-std CAP down so sigma falls and the policy hands off
        // explore->exploit (else the entropy bonus pins sigma at the cap -> noise-dominated action ->
        // the policy gradient cancels -> frozen). Mutates the LIVE net's cfg (rollout + update in this
        // iter share the same cap -> the PPO ratio stays consistent). Two modes:
        //   sigma_decay_iters>0: PHASE 1 (iter<N) force cap ls_max_start -> ls_max_end (fast, by iter);
        //                        PHASE 2 release cap to logstd_max_post -> head chooses sigma in
        //                        [exp(logstd_min), exp(logstd_max_post)].
        //   else if ls_max_end!=ls_max_start: linear decay over the whole run (by step fraction).
        {
            double cur_max = cfg_.model.logstd_max;
            if (cfg_.model.sigma_decay_iters > 0) {
                int N = cfg_.model.sigma_decay_iters;
                if (iter_ < N) {
                    double f = (double)iter_ / (double)N;             // 0 -> ~1 over the first N iters
                    cur_max = ls_max_start + (ls_max_end - ls_max_start) * f;
                } else {
                    cur_max = cfg_.model.logstd_max_post;             // hand control to the head
                }
            } else if (ls_max_end != ls_max_start) {
                double prog = cfg_.total_steps > 0 ? (double)global_step_ / (double)cfg_.total_steps : 1.0;
                if (prog > 1.0) prog = 1.0;
                cur_max = ls_max_start + (ls_max_end - ls_max_start) * prog;
            }
            if (cur_max < cfg_.model.logstd_min) cur_max = cfg_.model.logstd_min;
            policy_.net->cfg.logstd_max = cur_max;
        }
        // 3-stage curriculum. ITERATION-based in iter_mode (stage 1 for [0,s1), stage 2 for
        // [s1,s1+s2), stage 3 after; s1=0 skips stage 1 entirely); else the legacy step-fraction
        // schedule; else a fixed stage.
        if (iter_mode) {
            if (cfg_.stage3_iters > 0) {  // 4-stage curriculum: noop / random / starter / self-play+starter
                int s1 = cfg_.stage1_iters, s2 = cfg_.stage2_iters, s3 = cfg_.stage3_iters;
                cur_stage_ = iter_ < s1 ? 1 : iter_ < s1 + s2 ? 2 : iter_ < s1 + s2 + s3 ? 3 : 4;
            } else {
                cur_stage_ = iter_ < cfg_.stage1_iters ? 1
                           : iter_ < cfg_.stage1_iters + cfg_.stage2_iters ? 2 : 3;
            }
        } else if (cfg_.rollout.curriculum) {
            double frac = cfg_.total_steps > 0 ? (double)global_step_ / (double)cfg_.total_steps : 1.0;
            cur_stage_ = frac < cfg_.rollout.stage1_frac ? 1
                       : frac < cfg_.rollout.stage2_frac ? 2 : 3;
        } else {
            cur_stage_ = cfg_.rollout.stage;
        }
        rollout_.set_stage(cur_stage_);
        // Stage-4 self-play: (re)build the frozen opponent snapshot at entry and every
        // ckpt_every_late iters so the opponent is a LAGGING mirror that tracks the improving policy.
        if (cur_stage_ == 4) {
            int cad = std::max(1, cfg_.ckpt_every_late);
            if (!opp_set_ || iter_ % cad == 0) {
                opp_snapshot_ = policy_.clone_frozen();
                rollout_.set_opponent(&opp_snapshot_);
                opp_set_ = true;
            }
        }
        RolloutStats rs;
        auto tb = cfg_.rollout.gpu_env ? rollout_.collect_gpu(policy_, rs)
                                       : rollout_.collect(policy_, rs);
        auto us = update(tb);
        global_step_ += rs.transitions;
        iter_++;
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double sps = rs.transitions / std::max(dt, 1e-9);
        if (iter_mode) {
            // Print a heartbeat EVERY iteration; run the full eval + save a distinct progression
            // checkpoint only at the cadence (50 in stages 1&2, 100 in stage 3), plus iter 1 (the
            // random baseline) and the final iter.
            int cad = (cur_stage_ <= 2) ? cfg_.ckpt_every_early : cfg_.ckpt_every_late;
            if (iter_ == 1 || iter_ % cad == 0 || iter_ == cfg_.total_iters) {
                char tag[32];
                snprintf(tag, sizeof(tag), "it%04d", iter_);
                log_and_eval(sps, rs, us, tag);
            } else {
                heartbeat(sps, rs, us);
            }
        } else if (global_step_ >= next_log) {
            log_and_eval(sps, rs, us);
            next_log = ((global_step_ / 200000) + 1) * 200000;
        }
    }
    printf("done. best vsSTART=%.1f%% | results in %s\n", best_wr_starter_, run_dir_.c_str());
}

}  // namespace ow
