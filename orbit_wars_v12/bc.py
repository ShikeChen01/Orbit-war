"""Behaviour-cloning warm-start (notebook cell 32).

Clones the MEDIUM scripted rule directly into the gated-alloc heads so PPO starts from a
committing, aiming policy (the from-scratch bootstrap is the proven passivity trap). Expert =
medium (nearest non-owned planet, lead-aim) expressed as one-hot WHERE rows + fire; states are
visited by the clone itself vs a mixed scripted-opponent schedule; loss = BCE(gate) + CE(masked
WHERE) on firing sources. Saves ``cfg.BC_CKPT_PATH``.
"""
import random

import torch

from .checkpoint import save_ckpt
from .constants import BIG, DTYPE, PLANET_CAP
from .env import GpuEnv, env_encode, env_step, settle
from .policy import _make_dist, build_policy
from .worldgen import make_world_pool


def medium_targets_ego(env):
    """The medium rule for EGO (owner 0): (fire*, tgt) both (B,Ec)."""
    half = torch.floor(env.p_ships / 2.0)
    base = (env.p_owner == 0.0) & (env.p_alive > 0.5) & (half >= 20.0)
    sx = env.p_x.unsqueeze(2); sy = env.p_y.unsqueeze(2)
    tx = env.p_x.unsqueeze(1); ty = env.p_y.unsqueeze(1)
    vt = (env.p_alive > 0.5) & (env.p_owner != 0.0)
    d = torch.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)
    dmask = torch.where(vt.unsqueeze(1), d, torch.full_like(d, BIG))
    bestd, tgt = dmask.min(2)
    fire = base & (bestd < BIG * 0.5)
    return fire, tgt


def bc_expert_action(env):
    """(B,Ec,Ec+1) gated-alloc action bundle of the expert: one-hot WHERE rows + fire."""
    fire, tgt = medium_targets_ego(env)
    B_, Ec = fire.shape
    alloc = torch.zeros(B_, Ec, Ec, dtype=DTYPE, device=env.dev)
    alloc.scatter_(2, tgt.unsqueeze(-1), 1.0)
    return torch.cat([alloc, fire.to(DTYPE).unsqueeze(-1)], -1), fire, tgt


def bc_pretrain(cfg, net=None, rounds=None, epochs=None, lr=None, log=True):
    """Supervised clone of the medium rule. Returns the net; saves cfg.BC_CKPT_PATH."""
    rounds = cfg.BC_ROUNDS if rounds is None else rounds
    epochs = cfg.BC_EPOCHS if epochs is None else epochs
    lr = cfg.BC_LR if lr is None else lr
    net = net or build_policy(cfg)
    try:
        opt = torch.optim.Adam(net.parameters(), lr=lr, eps=cfg.ADAM_EPS, fused=(cfg.device.type == 'cuda'))
    except Exception:
        opt = torch.optim.Adam(net.parameters(), lr=lr, eps=cfg.ADAM_EPS)
    env = GpuEnv(cfg)
    pool = make_world_pool(cfg, max(cfg.B, 256), base_seed=cfg.SEED + 900001)
    opps = [1, 0, 3, 1, 4, 1, 3, 0, 1, 3, 4, 1]          # starter-heavy mix, incl. medium/greedy
    rng = random.Random(cfg.SEED + 31337)
    for r in range(rounds):
        worlds = [pool[rng.randrange(len(pool))] for _ in range(cfg.B)]
        env.reset(worlds)
        opp = opps[r % len(opps)]
        ents, ems, ams, gls, fires, tgts, actives = [], [], [], [], [], [], []
        active = torch.ones(cfg.B, device=cfg.device)
        with torch.no_grad():
            for t in range(env.T):
                ent, em, am, gl = env_encode(env, 0)
                a_t, fire, tgt = bc_expert_action(env)
                # stash on CPU: a full on-GPU rollout of encoded states is ~0.6 GB
                ents.append(ent.cpu()); ems.append(em.cpu()); ams.append(am.cpu()); gls.append(gl.cpu())
                fires.append(fire.cpu()); tgts.append(tgt.cpu()); actives.append(active.cpu().clone())
                env_step(cfg, env, a_t, opp, None, step_idx=t)
                s0, s1, side0, side1 = settle(env)
                active = active * (side0 & side1).to(DTYPE)
        ent = torch.cat(ents); em = torch.cat(ems); am = torch.cat(ams); gl = torch.cat(gls)
        fire = torch.cat(fires); tgt = torch.cat(tgts); act_m = torch.cat(actives)
        keep = act_m > 0.5                                    # drop post-terminal states
        ent, em, am, gl, fire, tgt = ent[keep], em[keep], am[keep], gl[keep], fire[keep], tgt[keep]
        N = ent.shape[0]
        mbs = max(8, cfg.BC_MINIBATCH // PLANET_CAP)          # boards per minibatch (~85 @ 4096/48)
        stats = [0.0, 0.0, 0.0, 0]
        for ep in range(epochs):
            perm = torch.randperm(N)
            for b in range(0, N, mbs):
                mi = perm[b:b + mbs]
                ent_d, em_d, am_d, gl_d = (x[mi].to(cfg.device) for x in (ent, em, am, gl))
                fire_d, tgt_d = fire[mi].to(cfg.device), tgt[mi].to(cfg.device)
                dist, _, _ = _make_dist(cfg, net, ent_d, em_d, am_d, gl_d)
                own = am_d > 0.5
                f = fire_d.to(DTYPE)
                p = dist.p                                     # smooth fire-prob (post-fix)
                bce = -(f * torch.log(p.clamp_min(1e-8)) + (1 - f) * torch.log((1 - p).clamp_min(1e-8)))
                bce = (bce * own.to(DTYPE)).sum() / own.sum().clamp_min(1)
                lsm = torch.log_softmax(dist.glogits, -1)      # masked WHERE logits (alive+reach+no-self)
                ce_all = -lsm.gather(2, tgt_d.unsqueeze(-1)).squeeze(-1)
                ok = own & fire_d & (dist.glogits.gather(2, tgt_d.unsqueeze(-1)).squeeze(-1) > -1e8)
                ce = (ce_all * ok.to(DTYPE)).sum() / ok.sum().clamp_min(1)
                loss = bce + ce
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.MAX_GRAD_NORM)
                opt.step()
                with torch.no_grad():
                    hit = ((lsm.argmax(-1) == tgt_d) & ok).sum() / ok.sum().clamp_min(1)
                    gate_acc = (((p > 0.5) == fire_d) & own).sum() / own.sum().clamp_min(1)
                stats[0] += loss.item(); stats[1] += hit.item(); stats[2] += gate_acc.item(); stats[3] += 1
        if log:
            print("BC round %2d/%d opp=%d N=%6d | loss %.4f | dest-acc %.3f gate-acc %.3f"
                  % (r + 1, rounds, opp, N, stats[0] / stats[3], stats[1] / stats[3], stats[2] / stats[3]), flush=True)
    save_ckpt(cfg, net, cfg.BC_CKPT_PATH, meta={"iter": 0, "bc": True})
    print("saved BC init ->", cfg.BC_CKPT_PATH)
    return net
