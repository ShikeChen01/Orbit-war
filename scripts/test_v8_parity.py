"""2-player parity: the v8 N-player env must be BIT-EXACT to the v7 env on the 2p path.

Execs the env cells of BOTH notebooks (SMOKE config) in separate namespaces, then steps both
envs 70 ticks (comet spawn at t=50 included) with IDENTICAL random gated-alloc ego actions vs
the scripted medium bot, comparing planet/fleet state and the encoded obs exactly every tick.

    python scripts/test_v8_parity.py
"""
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
os.environ.setdefault("OW_CKPT_DIR", os.path.join("runs", "v8_parity"))
os.makedirs(os.environ["OW_CKPT_DIR"], exist_ok=True)

STOP_AFTER = "def build_policy"   # exec code cells up to & incl. the policy cell


def load_ns(nb_path):
    doc = json.load(open(nb_path, encoding="utf-8"))
    ns = {"__name__": "nb"}
    for c in doc["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "SMOKE = False " in src and "BC_ENABLED" in src:
            src = src.replace("SMOKE = False ", "SMOKE = True  ")
        exec(compile(src, nb_path, "exec"), ns)
        if STOP_AFTER in src:
            break
    return ns


def main():
    ns7 = load_ns(os.path.join("notebooks", "legacy", "setup1_v7_a100.ipynb"))
    ns8 = load_ns(os.path.join("notebooks", "setup1_v8_a100.ipynb"))

    n_envs, steps = 4, 70
    w7 = [ns7["generate_world"](s) for s in range(n_envs)]
    w8 = [ns8["generate_world"](s) for s in range(n_envs)]
    for a, b in zip(w7, w8):
        assert a["planets"] == b["planets"] and a["angular_velocity"] == b["angular_velocity"], \
            "world gen diverged"
    print("worlds identical (%d seeds)" % n_envs)

    def make_env(ns, worlds):
        env = ns["GpuEnv"](ns["PLANET_CAP"], ns["FLEET_CAP"], 200, ns["SHIP_SPEED"], ns["DEVICE"])
        env.reset(worlds)
        return env

    env7 = make_env(ns7, w7)
    env8 = make_env(ns8, w8)
    Ec = ns7["PLANET_CAP"]
    dev = ns7["DEVICE"]

    g = torch.Generator(device="cpu").manual_seed(1234)
    fields = ("p_ships", "p_owner", "p_alive", "p_x", "p_y",
              "f_alive", "f_owner", "f_x", "f_y", "f_ships")
    for t in range(steps):
        alloc = torch.rand(n_envs, Ec, Ec, generator=g)
        fire = (torch.rand(n_envs, Ec, generator=g) < 0.3).float()
        a = torch.cat([alloc, fire.unsqueeze(-1)], -1).to(dev)
        ns7["env_step"](env7, a.clone(), 3, None, step_idx=t)   # vs scripted medium
        ns8["env_step"](env8, a.clone(), 3, None, step_idx=t)
        for f in fields:
            x7, x8 = getattr(env7, f), getattr(env8, f)
            assert torch.equal(x7, x8), "t=%d field %s diverged (max |d| = %g)" % (
                t, f, (x7 - x8).abs().max().item())
        e7 = ns7["env_encode"](env7, 0)
        e8 = ns8["env_encode"](env8, 0)
        for i, (a7, a8) in enumerate(zip(e7, e8)):
            assert torch.equal(a7, a8), "t=%d encode output %d diverged" % (t, i)
    print("PARITY OK: %d ticks, %d envs -- planet/fleet state + encodings bit-exact (2p)" %
          (steps, n_envs))


if __name__ == "__main__":
    main()
