"""Measure the env-throughput cost of comets: A/B the COMETS_ENABLED branch.

Loads the notebook env defs (generate_world / GpuEnv / env_step) up to env_step,
then steps a full 500-tick game (the 5 comet spawns at 50/150/.../450 fire) with
comets ON vs OFF — everything else identical — and reports steps/sec.

    python scripts/bench_comet_speed.py [--B 256] [--passes 3] [--device cuda]
"""
import argparse, json, os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=256)
ap.add_argument("--passes", type=int, default=3)
ap.add_argument("--warmup", type=int, default=60)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--nb", default=os.path.join(REPO, "notebooks", "legacy", "setup1_v6_ppo.ipynb"))
args = ap.parse_args()

dev = torch.device(args.device)

# ---- load notebook defs up to and including env_step ----
doc = json.load(open(args.nb, encoding="utf-8"))
g = {"__name__": "__bench__"}
for c in doc["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    exec(compile(s, "nb", "exec"), g)
    if "def env_step(" in s:
        break

B = args.B
worlds = [g["generate_world"](1000 + i) for i in range(B)]
env = g["GpuEnv"](g["PLANET_CAP"], B, 500, g["SHIP_SPEED"], dev)
env.reset(worlds)
Ec = env.Ec
zero_act = torch.zeros(B, Ec, Ec + 1, device=dev)


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()


def run_game(comets_enabled, n_steps=500):
    g["COMETS_ENABLED"] = comets_enabled
    env.reset(worlds)
    sync()
    t0 = time.perf_counter()
    for t in range(n_steps):
        g["env_step"](env, zero_act, 2, None, step_idx=t)
    sync()
    return n_steps / (time.perf_counter() - t0)


# warmup (kernel autotune + allocator) on each branch
g["COMETS_ENABLED"] = True
env.reset(worlds)
for t in range(args.warmup):
    g["env_step"](env, zero_act, 2, None, step_idx=t)
g["COMETS_ENABLED"] = False
env.reset(worlds)
for t in range(args.warmup):
    g["env_step"](env, zero_act, 2, None, step_idx=t)
sync()

on, off = [], []
for _ in range(args.passes):
    on.append(run_game(True))     # interleave to average out clock/thermal drift
    off.append(run_game(False))

mean = lambda xs: sum(xs) / len(xs)
on_m, off_m = mean(on), mean(off)
gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
print(f"\ndevice={gpu}  B={B}  steps/game=500  passes={args.passes}")
print(f"comets ON : {on_m:8.1f} sps   (B*sps={on_m*B:,.0f} env-steps/s)   runs={[f'{x:.0f}' for x in on]}")
print(f"comets OFF: {off_m:8.1f} sps   (B*sps={off_m*B:,.0f} env-steps/s)   runs={[f'{x:.0f}' for x in off]}")
print(f"\ncomet overhead: {(off_m/on_m - 1)*100:+.1f}% slower with comets "
      f"({off_m - on_m:+.0f} sps; per-step +{(1/on_m - 1/off_m)*1e6:.1f} us/game-step)")
