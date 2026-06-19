"""One PPO iteration: on-GPU rollout + v5 reward assembly + GAE (notebook cell 23).

``collect_ppo`` plays the learner (player 0) against ``seats`` (league members filling pids
1..n_players-1) across ``cfg.B`` parallel envs, with seat-swap world tiling, potential-based
reward shaping (or the legacy dense channels), terminal outcome payment, and an advantage estimate
-- GPU GAE for PPO, or a per-group standardized return for GRPO (``cfg.ALGO == "grpo"``) -- returning
a flattened transition buffer + heartbeat stats. n_players=2 is the v7 path.
"""
import torch

from .constants import DTYPE, F_DIM, G_DIM, ship_log_t
from .env import _KIND2OPP, _SEAT_ROT, env_encode, env_step, settle_n
from .policy import act


def _potentials(env):
    """Policy-invariant shaping potentials (Ng et al.), N-player: ship-margin and production
    share vs the STRONGEST opponent (reduces exactly to the v7 pair when n_players == 2)."""
    s, _ = settle_n(env)
    phi_ship = ship_log_t(s[:, 0]) - ship_log_t(s[:, 1:].max(1).values)
    pa = env.p_alive > 0.5
    N = int(getattr(env, "n_players", 2))
    prods = [(env.p_prod * ((env.p_owner == float(p)) & pa).to(DTYPE)).sum(1) for p in range(N)]
    pe = prods[0]
    pen = torch.stack(prods[1:], 1).max(1).values
    tot = (env.p_prod * pa.to(DTYPE)).sum(1).clamp_min(1.0)
    return phi_ship, (pe - pen) / tot


def collect_ppo(cfg, env, net, world_pool, cursor, rng, seats, n_players=2, n_envs=None):
    """One PPO iteration (learner = player 0) vs `seats` -- league members filling pids
    1..n_players-1. Scripted anchors run fully on-GPU; neural members act per step (no-grad).
    On-GPU buffers + GPU GAE; reward = potential shaping + outcome. n_players=2 is the v7 path.
    stats["seat_scores"][k] = the learner's mean pairwise score vs seat k (dual-Elo signal)."""
    Bn = int(n_envs or cfg.B)
    Ec, T = env.Ec, env.T
    dev = env.dev
    specs = []
    for i, m in enumerate(seats):
        pid = i + 1
        if m.get("net") is None:
            specs.append({"pid": pid, "script": _KIND2OPP[m["kind"]]})
        else:
            specs.append({"pid": pid, "net": m["net"]})

    # ---- seat-swap render: tile W=Bn//P distinct worlds P-fold and stamp a per-seat board
    # rotation (env._rot_k) so the ego (pid 0, fixed) plays every seat's geometry in this batch.
    # The 4-fold-symmetric board makes k*90deg an EXACT seat remap; sharing each world across the
    # P seat-blocks lets the normalized advantages cancel that world's positional bias.
    P = int(n_players)
    grpo = (cfg.ALGO == "grpo")
    phi_value = grpo and cfg.GRPO_PHI_VALUE and cfg.USE_POTENTIAL_SHAPING   # [C] Phi-as-value GAE (needs the potentials)
    crn = grpo and cfg.GRPO_CRN                                             # [D] common-random-numbers opponent coupling
    if grpo:
        # GRPO render: tile nW distinct worlds into contiguous groups of G envs (one group = one
        # world rolled out G times) so the per-group baseline cancels that world's difficulty. Each
        # group gets a SINGLE seat geometry, cycling _SEAT_ROT across groups so the ego still trains
        # on every seat without breaking within-group homogeneity.
        G = cfg.GRPO_GROUP or cfg.GROUP_SIZE
        if G <= 0 or Bn % G != 0:
            G = Bn                                         # fallback: the whole batch is one group
        nW = Bn // G
        base = [world_pool[(cursor + i) % len(world_pool)] for i in range(nW)]
        cursor += nW
        worlds = [w for w in base for _ in range(G)]       # base[0]xG, base[1]xG, ... (contiguous groups)
        if cfg.SEAT_SWAP and P in _SEAT_ROT:
            rots = _SEAT_ROT[P]
            rk = torch.empty(Bn, 1, dtype=torch.long, device=dev)
            for g in range(nW):
                rk[g * G:(g + 1) * G, 0] = rots[g % len(rots)]   # one seat geometry per group
            env._rot_k = rk; env._rot_aug = True
        else:
            env._rot_k = None; env._rot_aug = False
        _grp = (G, nW)
    elif cfg.SEAT_SWAP and P in _SEAT_ROT and Bn % P == 0:
        W = Bn // P
        base = [world_pool[(cursor + i) % len(world_pool)] for i in range(W)]
        cursor += W
        worlds = base * P                                  # P seat-blocks of the same W worlds
        rk = torch.empty(Bn, 1, dtype=torch.long, device=dev)
        for b, kk in enumerate(_SEAT_ROT[P]):
            rk[b * W:(b + 1) * W, 0] = kk                  # block b -> seat rotation kk
        env._rot_k = rk; env._rot_aug = True
        _grp = None
    else:
        worlds = [world_pool[(cursor + i) % len(world_pool)] for i in range(Bn)]
        cursor += Bn
        env._rot_k = None; env._rot_aug = False
        _grp = None
    env.reset(worlds, n_players=n_players)

    # on-GPU rollout buffers (compute-bound: no host copies, no per-step D2H sync)
    ent_buf = torch.zeros(T, Bn, Ec, F_DIM, device=dev)
    em_buf = torch.zeros(T, Bn, Ec, device=dev); am_buf = torch.zeros(T, Bn, Ec, device=dev)
    gl_buf = torch.zeros(T, Bn, G_DIM, device=dev); act_buf = torch.zeros(T, Bn, Ec, Ec + 1, device=dev)
    oldlp_buf = torch.zeros(T, Bn, device=dev); valid_buf = torch.zeros(T, Bn, device=dev)
    rew_buf = torch.zeros(T, Bn, device=dev); val_buf = torch.zeros(T, Bn, device=dev); done_buf = torch.zeros(T, Bn, device=dev)
    phi_buf = torch.zeros(T, Bn, device=dev) if phi_value else None        # [C] Phi(s_t) per step, fed to the V_hat baseline

    active = torch.ones(Bn, device=dev); outcome = torch.zeros(Bn, device=dev)
    seat_sc = torch.zeros(len(specs), Bn, device=dev)   # pairwise score vs each seat, frozen at term
    inv_sum = torch.zeros(Bn, device=dev); lnch_sum = torch.zeros(Bn, device=dev)
    valid_sum = torch.zeros(Bn, device=dev); step_sum = torch.zeros(Bn, device=dev)
    launch_paid = torch.zeros(Bn, device=dev); milestone_prev = torch.zeros(Bn, device=dev)
    captureR = torch.zeros(Bn, device=dev); launchR = torch.zeros(Bn, device=dev); prodMR = torch.zeros(Bn, device=dev)

    phi_ship_prev, phi_prod_prev = _potentials(env)   # shaping potentials at s_0

    def _scores_outcome(s_all):
        """Pairwise scores vs every seat + the scalar outcome in [-1, 1]."""
        s0 = s_all[:, 0]
        scs = []
        for sp in specs:
            sk = s_all[:, sp["pid"]]
            scs.append(torch.where(s0 > sk, torch.ones_like(s0),
                                   torch.where(s0 < sk, torch.zeros_like(s0), torch.full_like(s0, 0.5))))
        if n_players == 2 or cfg.OUTCOME_4P == "winner":
            oc = torch.sign(s0 - s_all[:, 1:].max(1).values)    # official: top score only
        else:
            oc = 2.0 * torch.stack(scs, 0).mean(0) - 1.0        # placement-linear (beat-all = +1)
        return scs, oc

    last_t = 0
    for t in range(T):
        last_t = t
        if phi_value:                       # Phi(s_t): phi_*_prev still holds the current state (s_0, or s_t from the prior step)
            phi_buf[t] = cfg.SHAPE_SHIP * phi_ship_prev + cfg.SHAPE_PROD * phi_prod_prev
        ent, em, am, gl = env_encode(env, 0)
        a_t, logp, value = act(cfg, net, ent, em, am, gl, greedy=False)
        ent_buf[t] = ent; em_buf[t] = em; am_buf[t] = am; gl_buf[t] = gl
        act_buf[t] = a_t; oldlp_buf[t] = logp; valid_buf[t] = active; val_buf[t] = value

        step_seats = []
        for sp in specs:
            if "net" in sp:
                with torch.no_grad():
                    oe, om, oa, og = env_encode(env, sp["pid"])
                    # [D] CRN: share this opponent's sampling noise across the G rollouts of each world-group
                    # (the ego, sampled above, keeps independent noise -> the group baseline isolates ego variance).
                    o_act, _, _ = act(cfg, sp["net"], oe, om, oa, og, greedy=False,
                                      crn_group=(_grp[0] if crn else 0))
                step_seats.append({"pid": sp["pid"], "action": o_act})
            else:
                step_seats.append({"pid": sp["pid"], "script": sp["script"]})
        out = env_step(cfg, env, a_t, seats=step_seats, step_idx=t)
        a = active

        if cfg.USE_POTENTIAL_SHAPING:
            phi_ship_now, phi_prod_now = _potentials(env)
            cap_t = a * (cfg.SHAPE_SHIP * (cfg.GAMMA * phi_ship_now - phi_ship_prev))   # ship-margin shaping
            mr_t = a * (cfg.SHAPE_PROD * (cfg.GAMMA * phi_prod_now - phi_prod_prev))    # production-share shaping
            launch_t = torch.zeros_like(cap_t)
            phi_ship_prev, phi_prod_prev = phi_ship_now, phi_prod_now
        else:
            s_all0, _ = settle_n(env)
            s0n = s_all0[:, 0]
            cap_t = a * (cfg.CAPTURE_REWARD * ((out.captured + cfg.CAPTURE_PROD_SCALE * out.captured_prod)
                                               - cfg.CAPTURE_LOSS_FRAC * (out.lost + cfg.CAPTURE_PROD_SCALE * out.lost_prod)))
            m_now = torch.where(s0n >= cfg.PROD_MILESTONE_BASE,
                                torch.floor(torch.log2(s0n.clamp_min(cfg.PROD_MILESTONE_BASE) / cfg.PROD_MILESTONE_BASE)) + 1.0,
                                torch.zeros_like(s0n))
            mr_t = a * (cfg.PROD_MILESTONE_REWARD * (m_now - milestone_prev).clamp_min(0.0))
            milestone_prev = torch.maximum(milestone_prev, m_now)
            if t < cfg.LAUNCH_WINDOW:
                _enemy_ships = out.launched_ships - out.owned_launch_ships
                _self_ships = torch.clamp(out.owned_launch_ships, max=cfg.SELF_LAUNCH_CAP)
                _disp_raw = cfg.LAUNCH_REWARD * _enemy_ships + cfg.SELF_LAUNCH_REWARD * _self_ships
                launch_t = a * torch.clamp(_disp_raw, max=cfg.LAUNCH_STEP_CAP)
            else:
                launch_t = torch.zeros_like(cap_t)
            launch_room = (cfg.LAUNCH_GAME_CAP - launch_paid).clamp_min(0.0)
            launch_t = torch.minimum(launch_t, launch_room)
            launch_paid = launch_paid + launch_t
        captureR = captureR + cap_t; prodMR = prodMR + mr_t; launchR = launchR + launch_t
        dense_t = torch.zeros_like(cap_t) if phi_value else (cap_t + mr_t + launch_t)   # [C] Phi -> value, not reward
        rew_buf[t] = dense_t / cfg.PPO_REWARD_SCALE

        inv_sum = inv_sum + a * out.invalid
        lnch_sum = lnch_sum + a * out.launches
        valid_sum = valid_sum + a * out.valid
        step_sum = step_sum + a

        s_all, alive_all = settle_n(env)
        step_now = env.step_ct
        n_alive = alive_all.to(DTYPE).sum(1)
        term = (step_now >= float(T - 2)) | (n_alive <= 1.0) | (~alive_all[:, 0])   # ego dead -> decided
        done_buf[t] = ((active > 0.5) & term).to(DTYPE)        # terminal on cap OR death (no spurious bootstrap)
        newly = (active > 0.5) & term
        scs, oc = _scores_outcome(s_all)
        for k in range(len(specs)):
            seat_sc[k] = torch.where(newly, scs[k], seat_sc[k])
        outcome = torch.where(newly, oc, outcome)
        active = torch.where(term, torch.zeros_like(active), active)
        # NOTE: no per-step active.sum().item() early-break -> that is a CUDA sync. Finished envs are masked.

    s_all, alive_all = settle_n(env)
    scs, oc = _scores_outcome(s_all)
    still = active > 0.5
    for k in range(len(specs)):
        seat_sc[k] = torch.where(still, scs[k], seat_sc[k])
    outcome = torch.where(still, oc, outcome)

    bootstrap = torch.zeros(Bn, device=dev)
    if not grpo:                                           # GRPO has no critic to bootstrap from
        with torch.no_grad():
            fe, fm, fa_, fg = env_encode(env, 0)
            _, _, vT = act(cfg, net, fe, fm, fa_, fg, greedy=False)
            bootstrap = vT * active

    Tu = last_t + 1
    rew = rew_buf[:Tu].clone(); val = val_buf[:Tu]; done = done_buf[:Tu]; alive = valid_buf[:Tu]
    length = alive.sum(0)
    len_eff = (length - cfg.DECAY_START_STEP).clamp_min(0.0)
    win_val = torch.pow(torch.full_like(outcome, cfg.WIN_DECAY), len_eff) * cfg.WIN_BONUS
    loss_val = torch.pow(torch.full_like(outcome, cfg.LOSS_DECAY), len_eff) * cfg.LOSS_PENALTY
    # fractional outcomes (4p placement mode) scale the outcome magnitudes linearly;
    # in 2p outcome is exactly {-1, 0, +1} -> identical to the v7 payment
    Ot = torch.where(outcome > 0.0, outcome * win_val, outcome * loss_val)
    r_outcome_log = Ot.mean().item()
    if not cfg.USE_POTENTIAL_SHAPING:
        disp_cashout = torch.clamp(cfg.LAUNCH_WINDOW - length, min=0.0) * cfg.LAUNCH_STEP_CAP * (outcome > 0.0).to(outcome.dtype)
        disp_cashout = torch.minimum(disp_cashout, (cfg.LAUNCH_GAME_CAP - launch_paid).clamp_min(0.0))
        launchR = launchR + disp_cashout
        Ot = (Ot + disp_cashout) / cfg.PPO_REWARD_SCALE
    else:
        Ot = Ot / cfg.PPO_REWARD_SCALE
    last_idx = (length - 1.0).clamp_min(0.0).long().unsqueeze(0)
    rew.scatter_add_(0, last_idx, Ot.unsqueeze(0))

    if grpo and not phi_value:
        # ---- flat group-relative advantage: one scalar per rollout, broadcast over its steps ----
        G, nW = _grp
        Re = Ot if cfg.GRPO_OUTCOME_ONLY else (rew * alive).sum(0)   # (Bn,) per-env return
        Rg = Re.view(nW, G)
        if cfg.GRPO_ANALYTIC_ADV and n_players == 2:
            # [A] closed-form binary advantage at a Beta(alpha,alpha)-shrunk per-group win-rate p~ (docs/grpo_v12.tex):
            #     A_win = sqrt((1-p~)/p~),  A_loss = -sqrt(p~/(1-p~)),  draws -> 0.  No divide-by-std, so the
            #     within-loss-group length signal vanishes and the p~ shrink removes the empty-group corner.
            og = outcome.view(nW, G)
            win = (og > 0.5).to(DTYPE); loss = (og < -0.5).to(DTYPE)
            a0 = cfg.GRPO_ANALYTIC_ALPHA
            p_hat = (win.sum(1, keepdim=True) + a0) / (G + 2.0 * a0)
            Ae = (win * torch.sqrt((1.0 - p_hat) / p_hat) - loss * torch.sqrt(p_hat / (1.0 - p_hat))).reshape(Bn)
        else:
            # [B] center (optionally leave-one-out RLOO) then optionally divide by the per-group std.
            sum_g = Rg.sum(1, keepdim=True)
            mu = (sum_g - Rg) / (G - 1) if (cfg.GRPO_LOO and G > 1) else sum_g / G
            cen = Rg - mu
            Ae = (cen / (Rg.std(1, keepdim=True) + cfg.GRPO_ADV_EPS) if cfg.GRPO_STD_NORM else cen).reshape(Bn)
        adv = Ae.unsqueeze(0).expand(Tu, Bn) * alive
        ret_full = Re.unsqueeze(0).expand(Tu, Bn).reshape(-1)        # placeholder (no critic target)
        adv_full = adv.reshape(-1)
    elif grpo and phi_value:
        # ---- [C] per-step GAE with a hand-built value V_hat(s_t) = mu_g + a*(Phi(s_t)-Phi(s_0)).
        # The shared root mu_g (= group baseline) cancels world difficulty; a*dPhi propagates the
        # sparse outcome to the moves that improved position. a is the OLS slope of outcome on terminal dPhi.
        G, nW = _grp
        Re = Ot if cfg.GRPO_OUTCOME_ONLY else (rew * alive).sum(0)
        Rg = Re.view(nW, G); sum_g = Rg.sum(1, keepdim=True)
        mu_g = (sum_g - Rg) / (G - 1) if (cfg.GRPO_LOO and G > 1) else (sum_g / G).expand(nW, G)
        mu = mu_g.reshape(Bn)                                       # V_hat(s_0), one baseline per rollout
        phi = phi_buf[:Tu]; phi0 = phi[0]
        dphi = phi - phi0.unsqueeze(0)                              # (Tu,Bn) positional change vs the start state
        li = (length - 1.0).clamp_min(0.0).long().unsqueeze(0)
        dphi_T = dphi.gather(0, li).squeeze(0)                       # (Bn,) terminal positional change
        a_slope = ((((Re - mu) * dphi_T).sum()) / (dphi_T * dphi_T).sum().clamp_min(1e-6)).clamp(0.0, cfg.GRPO_PHI_A_MAX)
        Vhat = (mu.unsqueeze(0) + a_slope * dphi) * alive           # (Tu,Bn)
        adv = torch.zeros(Tu, Bn, device=dev)
        A = torch.zeros(Bn, device=dev)
        nextval = (mu + a_slope * (phi[Tu - 1] - phi0)) * active     # bootstrap from V_hat (active envs only)
        for t in range(Tu - 1, -1, -1):
            al = alive[t]; notdone = 1.0 - done[t]
            delta = rew[t] + cfg.GAMMA * nextval * notdone - Vhat[t]
            A = delta + cfg.GAMMA * cfg.GAE_LAMBDA * notdone * A
            adv[t] = A * al
            nextval = Vhat[t]; A = A * al
        ret_full = (adv + Vhat).reshape(-1); adv_full = adv.reshape(-1)
    else:
        # GAE on GPU (no .cpu(), no synchronize)
        adv = torch.zeros(Tu, Bn, device=dev)
        A = torch.zeros(Bn, device=dev); nextval = bootstrap.clone()
        for t in range(Tu - 1, -1, -1):
            al = alive[t]; notdone = 1.0 - done[t]
            delta = rew[t] + cfg.GAMMA * nextval * notdone - val[t]
            A = delta + cfg.GAMMA * cfg.GAE_LAMBDA * notdone * A
            adv[t] = A * al
            nextval = val[t]
            A = A * al
        ret_full = (adv + val).reshape(-1); adv_full = adv.reshape(-1)
    mean_return_log = (rew.sum(0).mean().item()) * cfg.PPO_REWARD_SCALE

    keep = valid_buf[:Tu].reshape(-1).nonzero().squeeze(-1)
    def sel(x): return x.index_select(0, keep)
    tb = {}
    tb["entities"] = sel(ent_buf[:Tu].reshape(Tu * Bn, Ec, F_DIM))
    tb["entity_mask"] = sel(em_buf[:Tu].reshape(Tu * Bn, Ec))
    tb["action_mask"] = sel(am_buf[:Tu].reshape(Tu * Bn, Ec))
    tb["globals"] = sel(gl_buf[:Tu].reshape(Tu * Bn, G_DIM))
    tb["action"] = sel(act_buf[:Tu].reshape(Tu * Bn, Ec, Ec + 1))
    tb["old_logp"] = sel(oldlp_buf[:Tu].reshape(Tu * Bn))
    adv_s = sel(adv_full)
    # The flat per-group path WITH std-norm is already unit-scaled (leave it). Every other path --
    # center-only [B], phi-value [C], explicit GRPO_WHITEN, or PPO -- gets ONE global scale, which
    # (unlike the per-group std) preserves the cross-group p(1-p) gradient weighting (docs/grpo_v12.tex [B]).
    if grpo and not phi_value and cfg.GRPO_STD_NORM and not cfg.GRPO_WHITEN:
        tb["advantage"] = adv_s
    else:
        tb["advantage"] = (adv_s - adv_s.mean()) / (adv_s.std() + 1e-8)
    tb["returns"] = sel(ret_full)
    tb["n"] = keep.shape[0]
    del ent_buf, em_buf, am_buf, gl_buf, act_buf, oldlp_buf, val_buf, rew_buf, done_buf, valid_buf, adv, adv_full, ret_full

    step_total = step_sum.sum().item()
    stats = {
        "mean_return": mean_return_log,
        "mean_len": step_total / Bn,
        "win_rate": (seat_sc.min(0).values >= 0.999).to(DTYPE).mean().item(),   # beat EVERY opponent
        "score": seat_sc.mean().item(),
        "seat_scores": [float(seat_sc[k].mean().item()) for k in range(len(specs))],
        "fmt": n_players,
        "inv_per_step": (inv_sum.sum().item() / step_total) if step_total > 0 else 0.0,
        "lnch_per_step": (lnch_sum.sum().item() / step_total) if step_total > 0 else 0.0,
        "valid_per_step": (valid_sum.sum().item() / step_total) if step_total > 0 else 0.0,
        "transitions": tb["n"],
        "r_outcome": r_outcome_log,
        "r_capture": captureR.mean().item(),      # = ship-margin shaping when USE_POTENTIAL_SHAPING
        "r_launch": launchR.mean().item(),
        "r_milestone": prodMR.mean().item(),      # = production-share shaping when USE_POTENTIAL_SHAPING
    }
    return tb, stats, cursor
