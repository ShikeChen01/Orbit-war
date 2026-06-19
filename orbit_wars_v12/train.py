"""Elo-league-driven training loop (notebook cells 30 + 35).

:class:`Trainer` owns the :class:`~orbit_wars_v12.config.Config` and runs the loop: PFSP-sample an
opponent (2p or 4p per the dual-Elo schedule), roll out, PPO-update, update Elo, periodic recals +
measured eviction, and best-by-mixed-gauntlet checkpointing. There are NO hand-coded stages -- the
league IS the curriculum. The notebook's runtime globals are gone: the annealed entropy coefficient
and the value-warmup policy coefficient are computed per-iter and passed into ``ppo_update``.
"""
import copy
import json
import os
import random
import time

import torch

from .checkpoint import _freeze_snapshot, load_train_state, save_ckpt, save_train_state
from .env import GpuEnv
from .grpo import grpo_update
from .league import League, eval_gauntlet, eval_gauntlet4
from .policy import build_policy
from .ppo import ppo_update
from .rollout import collect_ppo
from .worldgen import make_world_pool


def anneal_ent_coef(cfg, it):
    if cfg.ENT_DECAY_ITERS <= 0:
        return cfg.ENT_COEF_END
    f = min(1.0, it / cfg.ENT_DECAY_ITERS)
    return cfg.ENT_COEF_START + (cfg.ENT_COEF_END - cfg.ENT_COEF_START) * f


class Trainer:
    """Holds the config and drives one or more train() calls. Returns (net, hist, league)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def train(self, total_iters=None, log_every=1, resume_from=None):
        cfg = self.cfg
        total_iters = cfg.TOTAL_ITERS if total_iters is None else total_iters
        resume_from = cfg.RESUME_FROM if resume_from is None else resume_from

        net = build_policy(cfg)
        try:
            opt = torch.optim.Adam(net.parameters(), lr=cfg.LR, eps=cfg.ADAM_EPS, fused=(cfg.device.type == 'cuda'))
        except Exception:
            opt = torch.optim.Adam(net.parameters(), lr=cfg.LR, eps=cfg.ADAM_EPS)
        net_fwd = torch.compile(net, mode=cfg.COMPILE_MODE) if cfg.USE_COMPILE else net   # compile LEARNER forward only;
        #          snapshots (deepcopy), recals and saves keep the eager `net` to avoid recompile churn.
        env = GpuEnv(cfg)
        rng = random.Random(cfg.SEED * 2654435761 + 12345)
        start_it = 0
        best_wr = -1.0

        league = League(cfg, rng)
        for ck in cfg.LEAGUE_INIT_CKPTS:
            try:
                m = league.add_checkpoint(ck)
                print("league seeded with %s  <- %s" % (m["label"], ck))
            except Exception as e:
                print("  !! skipped league ckpt %s -> %r" % (ck, e))

        if resume_from:
            start_it, best_wr = load_train_state(cfg, net, opt, league, resume_from)
            print("resumed from %s -> global iter %d, best_wr %.2f, %d league members"
                  % (resume_from, start_it, best_wr, len(league.members)))

        # GRPO reference policy for the optional KL penalty: freeze the (post-resume/BC) net once.
        ref_net = None
        if cfg.ALGO == "grpo" and cfg.GRPO_KL_COEF > 0.0:
            ref_net = _freeze_snapshot(copy.deepcopy(net))
            print("  [grpo] KL-to-reference enabled (beta=%.3g) -> froze current policy as reference" % cfg.GRPO_KL_COEF)

        hist = {"iter": [], "return": [], "win_rate": [],
                "lnch_per_step": [], "approx_kl": [], "clipfrac": [], "sigma": [], "grad_norm": [],
                "r_outcome": [], "r_capture": [], "r_milestone": [], "r_launch": [],
                "learner_elo": [], "learner_elo4": [], "share_4p": [], "fmt": []}

        pool, pool_round = None, -1
        # held-out gauntlet worlds: FIXED seed range never touched by training rounds, so the
        # best-ckpt score measures generalization and stays comparable across the whole run.
        heldout_pool = make_world_pool(cfg, cfg.ELO_RECAL_ENVS, base_seed=cfg.SEED + 999_999_937)
        if cfg.FOURP_ENABLED:
            league.calibrate_anchor_elo4(net, heldout_pool)   # v9.3: measured (not 2p-copied) 4p anchors
        if cfg.INVITE_MEASURE:
            league.calibrate_invited(heldout_pool)   # v9.4: measured invited anchors (stamps are stale-scale)
        acc4 = 0.0       # error-diffusion accumulator -> the realized 2p:4p ratio equals share_4p
        for step in range(total_iters):
            it = start_it + step
            rnd = (it // cfg.WORLD_RESAMPLE_EVERY) if cfg.WORLD_RESAMPLE_EVERY > 0 else 0
            _force = getattr(league, "_force_resample", False)   # v9.9: a promotion asked for fresh worlds
            if rnd != pool_round or _force:
                if _force:
                    league._force_resample = False
                    league._resample_salt = getattr(league, "_resample_salt", 0) + 1
                _salt = getattr(league, "_resample_salt", 0)
                pool = make_world_pool(cfg, cfg.N_WORLDS, base_seed=cfg.SEED + rnd * cfg.N_WORLDS + 7919 * _salt)
                cursor, pool_round = 0, rnd
                print("  [worlds] pool round %d @ it%d (base_seed %d)%s" % (
                    rnd, it, cfg.SEED + rnd * cfg.N_WORLDS + 7919 * _salt,
                    " [promotion resample #%d]" % _salt if _force else ""))
            if cfg.ELO_RECAL_EVERY > 0 and step > 0 and it % cfg.ELO_RECAL_EVERY == 0:  # re-ground BOTH rating scales
                league.recalibrate_elo(net, pool)
                if cfg.FOURP_ENABLED:
                    league.recalibrate_elo4(net, pool)
            elif cfg.LEARNER_RECAL_EVERY > 0 and step > 0 and it % cfg.LEARNER_RECAL_EVERY == 0:
                league.recalibrate_learner(net, pool)   # v9.2: cheap learner-only re-ground (blended)
            t0 = time.time()
            ent_coef = anneal_ent_coef(cfg, it)   # entropy anneal (anti collapse -> exploit)
            # v7 guardrails: linear LR warmup + critic-only warmup (both per train() CALL, not global iter)
            # GRPO has no critic to warm up -> always full policy weight.
            policy_coef = 1.0 if cfg.ALGO == "grpo" else (0.0 if step < cfg.VALUE_WARMUP_ITERS else 1.0)
            if cfg.LR_WARMUP_ITERS > 0:
                wf = min(1.0, (step + 1) / float(cfg.LR_WARMUP_ITERS))
                for pg in opt.param_groups:
                    pg['lr'] = cfg.LR * wf

            # grow the pool: snapshot the learner (after a short anchors-only warmup) every SELFPLAY_REFRESH iters
            if (it >= cfg.SNAPSHOT_WARMUP) and (step == 0 or it % max(1, cfg.SELFPLAY_REFRESH) == 0):
                league.add_learner_snapshot(net, it + 1)

            # ---- v8 match-type scheduler: dual-Elo gap -> bounded 2p/4p mix (1:1 .. 1:6) ----
            s4 = league.share_4p() if cfg.FOURP_ENABLED else 0.0
            acc4 += s4
            if acc4 >= 1.0:
                fmt = 4; acc4 -= 1.0
            else:
                fmt = 2

            # pick the opponent seat(s) for this rollout: PFSP over the whole league per format
            if fmt == 2:
                seats = [league.sample(net, fmt=2)]
            else:
                seats = league.sample_seats(net, k=3, fmt=4)
            for m in seats:
                if m["net"] is not None:
                    m["net"].to(cfg.device)

            tb, rs, cursor = collect_ppo(cfg, env, net_fwd, pool, cursor, rng, seats,
                                         n_players=fmt, n_envs=(cfg.B if fmt == 2 else cfg.B_4P))
            if cfg.ALGO == "grpo":
                us = grpo_update(cfg, net_fwd, opt, tb, ent_coef, policy_coef, ref_net=ref_net)
            else:
                us = ppo_update(cfg, net_fwd, opt, tb, ent_coef, policy_coef)

            # dual Elo: 2p = single update; 4p = pairwise vs every seat (cross-coupled inside)
            if fmt == 2:
                league.update_elo(seats[0], rs["score"])
            else:
                league.update_elo_4p(seats, rs["seat_scores"])
            if (not league.advanced) and league.learner_elo >= cfg.ADVANCE_TRIGGER_ELO:   # curriculum: confirm with an all-anchor recal (fresh seeds)
                league.recalibrate_elo(net, make_world_pool(cfg, cfg.ELO_RECAL_ENVS, base_seed=cfg.SEED + 104729 + 7919 * (it + 1)), all_anchors=True)
                if league.learner_elo >= cfg.ADVANCE_CONFIRM_ELO:
                    league.advanced = True
                    print("  [advance] learner hit %.0f ELO; all-anchor recal %.0f >= %d -> UNLOCK greedy/medium" % (cfg.ADVANCE_TRIGGER_ELO, league.learner_elo, cfg.ADVANCE_CONFIRM_ELO))
                else:
                    print("  [advance] learner hit %.0f ELO but all-anchor recal %.0f < %d -> stay in base regime" % (cfg.ADVANCE_TRIGGER_ELO, league.learner_elo, cfg.ADVANCE_CONFIRM_ELO))
            for m in seats:
                if m["net"] is not None:
                    m["net"].to("cpu")           # keep only the live policy resident on the GPU

            dt = time.time() - t0
            sps = rs["transitions"] / max(dt, 1e-9)

            hist["iter"].append(it + 1)
            hist["return"].append(rs["mean_return"]); hist["win_rate"].append(rs["win_rate"])
            hist["lnch_per_step"].append(rs["lnch_per_step"]); hist["approx_kl"].append(us["approx_kl"])
            hist["clipfrac"].append(us["clipfrac"]); hist["sigma"].append(us["sigma"]); hist["grad_norm"].append(us["grad_norm"])
            hist["r_outcome"].append(rs["r_outcome"]); hist["r_capture"].append(rs["r_capture"])
            hist["r_milestone"].append(rs["r_milestone"]); hist["r_launch"].append(rs["r_launch"])
            hist["learner_elo"].append(league.learner_elo)
            hist["learner_elo4"].append(league.learner_elo4)
            hist["share_4p"].append(s4); hist["fmt"].append(fmt)

            # v7 tripwire: gate-collapse detector (launch rate decaying under healthy-looking KL)
            _l = hist["lnch_per_step"]
            if len(_l) >= cfg.TRIPWIRE_BASE_ITERS and (it + 1) % 10 == 0:
                _base = sum(_l[:cfg.TRIPWIRE_BASE_ITERS]) / cfg.TRIPWIRE_BASE_ITERS
                _ema = sum(_l[-5:]) / len(_l[-5:])
                if _base > 0 and _ema < cfg.TRIPWIRE_FRAC * _base:
                    print("  [TRIPWIRE] launch/st EMA %.2f < %.0f%% of early baseline %.2f -> passivity ratchet suspected; check gauntlet+elo before continuing" % (_ema, cfg.TRIPWIRE_FRAC * 100, _base))

            if (it + 1) % log_every == 0 or step == 0 or step == total_iters - 1:
                # heartbeat: R[o c p ln] = outcome, capture, prod-milestone, launch (real units)
                opp_lab = "+".join(m["label"] for m in seats)
                print("it%4d %dp | ret %8.2f wr %.2f sc %.2f | R[o %7.1f c %6.1f p %5.1f ln %5.1f] | "
                      "lnch/st %.2f | kl %.3f cf %.2f gn %.1f | loss %7.3f (pol %.3f vf %.3f) | "
                      "elo2 %5.0f elo4 %5.0f s4 %.2f vs %-22s | sps %5.0f"
                      % (it + 1, fmt, rs["mean_return"], rs["win_rate"], rs["score"],
                         rs["r_outcome"], rs["r_capture"], rs["r_milestone"], rs["r_launch"],
                         rs["lnch_per_step"], us["approx_kl"], us["clipfrac"], us["grad_norm"],
                         us["total"], us["policy"], us["vf"],
                         league.learner_elo, league.learner_elo4, s4, opp_lab[:22], sps))

            # checkpoint: periodic + best-by-MIXED-GAUNTLET (2p + 4p; fixed opponents, held-out worlds)
            if ((it + 1) % max(1, cfg.CKPT_EVERY) == 0) or (step == total_iters - 1):
                save_ckpt(cfg, net, cfg.CKPT_PATH, meta={"iter": it + 1, "win_rate": rs["win_rate"]})
                save_train_state(net, opt, league, it + 1, best_wr, cfg.TRAIN_STATE_PATH)
                g2 = eval_gauntlet(cfg, net, heldout_pool)
                g4 = eval_gauntlet4(cfg, net, heldout_pool) if cfg.FOURP_ENABLED else g2
                gscore = (1.0 - cfg.GAUNTLET_4P_W) * g2 + cfg.GAUNTLET_4P_W * g4
                hist.setdefault("gauntlet", []).append(gscore)
                hist.setdefault("gauntlet2", []).append(g2)
                hist.setdefault("gauntlet4", []).append(g4)
                if gscore > best_wr:
                    best_wr = gscore
                    save_ckpt(cfg, net, cfg.BEST_CKPT_PATH, meta={"iter": it + 1, "gauntlet": float(gscore),
                                                                  "gauntlet2": float(g2), "gauntlet4": float(g4)})
                    print("  [best] mixed gauntlet %.3f (2p %.3f / 4p %.3f) -> saved best" % (gscore, g2, g4))
                print(league.leaderboard(), end="")

        save_ckpt(cfg, net, cfg.CKPT_PATH, meta={"iter": start_it + total_iters, "win_rate": hist["win_rate"][-1]})
        save_train_state(net, opt, league, start_it + total_iters, best_wr, cfg.TRAIN_STATE_PATH)
        print("saved final checkpoint ->", cfg.CKPT_PATH, "| best mixed gauntlet %.2f ->" % best_wr, cfg.BEST_CKPT_PATH)
        print(league.leaderboard(), end="")
        json.dump({k: [float(x) for x in v] for k, v in hist.items()},
                  open(os.path.join(cfg.CKPT_DIR, "metrics.json"), "w"))
        print("saved metrics ->", os.path.join(cfg.CKPT_DIR, "metrics.json"))
        return net, hist, league


def train(cfg, total_iters=None, log_every=1, resume_from=None):
    """Convenience: build a Trainer and run one train() call. Returns (net, hist, league)."""
    return Trainer(cfg).train(total_iters=total_iters, log_every=log_every, resume_from=resume_from)


def save_league_agents(cfg, league):
    """Save every league member that carries weights as a standalone .pt (notebook cell 35)."""
    out = os.path.join(cfg.CKPT_DIR, "league_agents")
    os.makedirs(out, exist_ok=True)
    saved = 0
    for m in sorted(league.members, key=lambda mm: -mm["elo"]):
        if m["net"] is None:                       # scripted anchors (random/starter) -> no weights
            continue
        safe = "".join(c if c.isalnum() else "_" for c in m["label"])
        path = os.path.join(out, "%s_elo%04d.pt" % (safe, round(m["elo"])))
        save_ckpt(cfg, m["net"], path,
                  meta={"elo": m["elo"], "elo4": m.get("elo4"), "label": m["label"],
                        "games": m["n"], "games4": m.get("n4", 0)})
        saved += 1
        print("saved %-12s elo %6.0f  n=%-4d -> %s" % (m["label"], m["elo"], m["n"], path))
    print("saved %d league agents -> %s" % (saved, out))
    return saved
