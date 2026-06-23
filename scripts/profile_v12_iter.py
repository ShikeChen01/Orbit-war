"""Profile one v12 training iteration to find where A100 GPU util is being lost.

Run ON THE A100 (Colab) against the SAME config as the notebook you're training
(the deep blockseq h256 trunk by default). It answers three questions:

  1. rollout vs update  -- which half of the iter dominates (expect rollout).
  2. rollout phase split -- encode / ego-act / OPP-act / env_step / settle+reward.
     The opponent-act phase is the eager (uncompiled) league-net forward; if it is
     large, compiling/caching the opponent forward is the win.
  3. GPU-busy %         -- sum(self_cuda_time) / wall_time. THIS is your "52% util":
     <70% => dispatch-bound (CUDA graphs / bigger B / fewer kernels), not compute-bound.

It also sweeps B (NUM_GROUPS) so you can see util rise as the kernels get bigger.

Usage (on the GPU box):
    python scripts/profile_v12_iter.py                 # deep cfg, 2p + 4p, B-sweep
    python scripts/profile_v12_iter.py --fmt 2 --iters 8
    python scripts/profile_v12_iter.py --no-sweep --trace   # + chrome trace
"""
import argparse
import time

import torch

from orbit_wars_v12.config import Config
from orbit_wars_v12.env import GpuEnv, maybe_compile_encode, env_encode, env_step, settle_n
from orbit_wars_v12.policy import build_policy, act
from orbit_wars_v12.ppo import ppo_update
from orbit_wars_v12.grpo import grpo_update
from orbit_wars_v12.rollout import collect_ppo
from orbit_wars_v12.worldgen import make_world_pool

# the deep blockseq trunk the notebook trains (h256, attn every 12). Override on CLI.
_DEEP_SPEC = "res12,attn1," * 11
_DEEP_SPEC = (("res12,attn1," * 11) + "res12").rstrip(",")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build(args):
    spec = args.spec or _DEEP_SPEC
    cfg = Config.create(**dict(
        ARCH="blockseq", TRUNK_SPEC=spec, HIDDEN=256, N_HEADS=8,
        ALGO=args.algo, VALUE_RES_BLOCKS=2,
        GRAD_CHECKPOINT=not args.no_ckpt,
        USE_COMPILE=not args.no_compile, COMPILE_MODE=args.compile_mode,
        AMP_DTYPE=torch.bfloat16,
        NUM_GROUPS=args.groups, GROUP_SIZE=args.group_size,
        GRPO_GROUP=(args.group_size if args.algo == "grpo" else 0),
        EPISODE_STEPS=args.steps, N_WORLDS=max(256, args.groups * args.group_size * 2),
        TOTAL_ITERS=1,
    ))
    net = build_policy(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.LR, eps=cfg.ADAM_EPS,
                           fused=(cfg.device.type == "cuda"))
    net_fwd = torch.compile(net, mode=cfg.COMPILE_MODE) if cfg.USE_COMPILE else net
    maybe_compile_encode(cfg)
    env = GpuEnv(cfg)
    pool = make_world_pool(cfg, cfg.N_WORLDS, base_seed=cfg.SEED)
    return cfg, net, net_fwd, opt, env, pool


def make_seats(cfg, net, fmt, neural_opp):
    """Realistic seats: neural opponents = a frozen copy of the learner (captures the
    EAGER opponent-forward cost). Set --scripted to use the medium bot instead (cheaper)."""
    import copy
    n_opp = fmt - 1
    seats = []
    for i in range(n_opp):
        if neural_opp:
            opp = copy.deepcopy(net).to(cfg.device).eval()
            for p in opp.parameters():
                p.requires_grad_(False)
            seats.append({"net": opp, "label": "snap%d" % i, "kind": "neural"})
        else:
            seats.append({"net": None, "kind": "medium", "label": "medium"})
    return seats


def phase_timed_rollout(cfg, env, net_fwd, pool, seats, fmt, n_envs):
    """A stripped copy of collect_ppo's INNER LOOP with per-phase CUDA timers. NOT used for
    correctness -- only to attribute rollout wall-time to encode/ego-act/opp-act/step/settle.
    Mirrors the real loop's calls so the ratios transfer."""
    from orbit_wars_v12.env import _KIND2OPP
    acc = {"encode": 0.0, "ego_act": 0.0, "opp_act": 0.0, "step": 0.0, "settle": 0.0}

    def tic():
        _sync(); return time.perf_counter()

    def toc(k, t):
        _sync(); acc[k] += time.perf_counter() - t

    Bn = n_envs
    worlds = [pool[i % len(pool)] for i in range(Bn)]
    env._rot_k = None; env._rot_aug = False
    env.reset(worlds, n_players=fmt)
    specs = []
    for i, m in enumerate(seats):
        pid = i + 1
        specs.append({"pid": pid, "net": m["net"]} if m.get("net") is not None
                     else {"pid": pid, "script": _KIND2OPP[m["kind"]]})
    for t in range(env.T):
        ts = tic(); ent, em, am, gl = env_encode(env, 0); toc("encode", ts)
        ts = tic(); a_t, logp, value = act(cfg, net_fwd, ent, em, am, gl, greedy=False); toc("ego_act", ts)
        step_seats = []
        for sp in specs:
            if "net" in sp:
                ts = tic()
                with torch.no_grad():
                    oe, om, oa, og = env_encode(env, sp["pid"])
                    o_act, _, _ = act(cfg, sp["net"], oe, om, oa, og, greedy=False)
                toc("opp_act", ts)
                step_seats.append({"pid": sp["pid"], "action": o_act})
            else:
                step_seats.append({"pid": sp["pid"], "script": sp["script"]})
        ts = tic(); env_step(cfg, env, a_t, seats=step_seats, step_idx=t); toc("step", ts)
        ts = tic(); settle_n(env); toc("settle", ts)
    return acc


def run_one(cfg, net, net_fwd, opt, env, pool, fmt, args, label=""):
    n_envs = cfg.B if fmt == 2 else cfg.B_4P
    seats = make_seats(cfg, net, fmt, neural_opp=not args.scripted)
    update = grpo_update if cfg.ALGO in ("grpo", "mc") else ppo_update

    # warmup: trigger compile + cudnn autotune so timings are steady-state
    for _ in range(args.warmup):
        tb, rs, _ = collect_ppo(cfg, env, net_fwd, pool, 0, None, seats,
                                n_players=fmt, n_envs=n_envs)
        update(cfg, net_fwd, opt, tb, 0.01, 1.0)
    _sync()

    roll_t = upd_t = 0.0
    for _ in range(args.iters):
        t0 = time.perf_counter(); _sync()
        tb, rs, _ = collect_ppo(cfg, env, net_fwd, pool, 0, None, seats,
                                n_players=fmt, n_envs=n_envs)
        _sync(); t1 = time.perf_counter()
        update(cfg, net_fwd, opt, tb, 0.01, 1.0)
        _sync(); t2 = time.perf_counter()
        roll_t += t1 - t0; upd_t += t2 - t1
    roll_t /= args.iters; upd_t /= args.iters
    iter_t = roll_t + upd_t

    print("\n=== %s  fmt=%dp  B=%d  steps=%d  trans=%d ===" % (label, fmt, n_envs, env.T, tb["n"]))
    print("  iter      %7.1f ms" % (iter_t * 1e3))
    print("  rollout   %7.1f ms  (%4.1f%%)" % (roll_t * 1e3, 100 * roll_t / iter_t))
    print("  update    %7.1f ms  (%4.1f%%)" % (upd_t * 1e3, 100 * upd_t / iter_t))
    print("  throughput %6.0f transitions/s" % (tb["n"] / iter_t))

    # rollout phase split (one extra rollout, instrumented)
    ph = phase_timed_rollout(cfg, env, net_fwd, pool, seats, fmt, n_envs)
    tot = sum(ph.values()) or 1.0
    print("  -- rollout phases (instrumented) --")
    for k in ("encode", "ego_act", "opp_act", "step", "settle"):
        print("     %-8s %7.1f ms  (%4.1f%%)" % (k, ph[k] * 1e3, 100 * ph[k] / tot))

    # GPU-busy %: sum of kernel self-CUDA-time over one iter / wall. THIS is your util number.
    if torch.cuda.is_available():
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            tb, rs, _ = collect_ppo(cfg, env, net_fwd, pool, 0, None, seats,
                                    n_players=fmt, n_envs=n_envs)
            update(cfg, net_fwd, opt, tb, 0.01, 1.0)
            _sync(); wall = time.perf_counter() - t0
        cuda_us = sum(e.self_cuda_time_total for e in prof.key_averages())
        print("  GPU-busy  %4.1f%%  (sum kernel CUDA time %.1f ms / wall %.1f ms)"
              % (100 * (cuda_us / 1e6) / wall, cuda_us / 1e3, wall * 1e3))
        print("  -- top 12 kernels by self-CUDA time --")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))
        if args.trace:
            path = "trace_%s_%dp.json" % (label or "run", fmt)
            prof.export_chrome_trace(path)
            print("  chrome trace ->", path, "(open in chrome://tracing or perfetto.dev)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "grpo", "mc"])
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--fmt", type=int, default=0, help="0 = both 2p and 4p")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--scripted", action="store_true", help="medium-bot opp instead of neural copy")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-ckpt", action="store_true", help="disable grad checkpointing")
    ap.add_argument("--compile-mode", default="default", help="default | reduce-overhead (CUDA graphs)")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    cfg, net, net_fwd, opt, env, pool = build(args)
    print("device=%s  ARCH=%s  HIDDEN=%d  blocks(spec)=%s  compile=%s/%s  grad_ckpt=%s"
          % (cfg.device, cfg.ARCH, cfg.HIDDEN, cfg.TRUNK_SPEC.count(",") + 1,
             cfg.USE_COMPILE, cfg.COMPILE_MODE, cfg.GRAD_CHECKPOINT))

    fmts = [2, 4] if args.fmt == 0 else [args.fmt]
    for fmt in fmts:
        run_one(cfg, net, net_fwd, opt, env, pool, fmt, args, label="baseline")

    if not args.no_sweep:
        print("\n########## B-sweep (NUM_GROUPS x GROUP_SIZE, 2p, util vs batch) ##########")
        for groups in (8, 16, 32, 48):
            args.groups = groups
            cfg, net, net_fwd, opt, env, pool = build(args)
            run_one(cfg, net, net_fwd, opt, env, pool, 2, args, label="B=%d" % (groups * args.group_size))


if __name__ == "__main__":
    main()
