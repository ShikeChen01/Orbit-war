"""2-player parity: the v9 one-hot-obs env must keep v8's DYNAMICS bit-exact on the 2p path.

Execs the env cells of BOTH active notebooks (SMOKE config) in separate namespaces, then steps
both envs 70 ticks (comet spawn at t=50 included) with IDENTICAL random gated-alloc ego actions
vs the scripted medium bot. State must match bit-exactly every tick; encodings must agree on all
shared features, and v9's 4-channel ownership one-hot must be consistent with v8's scalar code
(self<->1, enemy<->2, channels 2/3 silent in 2p).

    python scripts/test_v9_parity.py
"""
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
os.environ.setdefault("OW_CKPT_DIR", os.path.join("runs", "v9_parity"))
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
    ns8 = load_ns(os.path.join("notebooks", "setup1_v8_a100.ipynb"))
    ns9 = load_ns(os.path.join("notebooks", "setup1_v9_a100.ipynb"))
    assert ns8["F_DIM"] == 101 and ns9["F_DIM"] == 104, (ns8["F_DIM"], ns9["F_DIM"])

    n_envs, steps = 4, 70
    w8 = [ns8["generate_world"](s) for s in range(n_envs)]
    w9 = [ns9["generate_world"](s) for s in range(n_envs)]
    for a, b in zip(w8, w9):
        assert a["planets"] == b["planets"] and a["angular_velocity"] == b["angular_velocity"], \
            "world gen diverged"
    print("worlds identical (%d seeds)" % n_envs)

    def make_env(ns, worlds):
        env = ns["GpuEnv"](ns["PLANET_CAP"], ns["FLEET_CAP"], 200, ns["SHIP_SPEED"], ns["DEVICE"])
        env.reset(worlds)
        return env

    env8 = make_env(ns8, w8)
    env9 = make_env(ns9, w9)
    Ec = ns8["PLANET_CAP"]
    dev = ns8["DEVICE"]

    g = torch.Generator(device="cpu").manual_seed(1234)
    fields = ("p_ships", "p_owner", "p_alive", "p_x", "p_y",
              "f_alive", "f_owner", "f_x", "f_y", "f_ships")
    for t in range(steps):
        alloc = torch.rand(n_envs, Ec, Ec, generator=g)
        fire = (torch.rand(n_envs, Ec, generator=g) < 0.3).float()
        a = torch.cat([alloc, fire.unsqueeze(-1)], -1).to(dev)
        ns8["env_step"](env8, a.clone(), 3, None, step_idx=t)   # vs scripted medium
        ns9["env_step"](env9, a.clone(), 3, None, step_idx=t)
        for f in fields:
            x8, x9 = getattr(env8, f), getattr(env9, f)
            assert torch.equal(x8, x9), "t=%d field %s diverged (max |d| = %g)" % (
                t, f, (x8 - x9).abs().max().item())
        e8 = ns8["env_encode"](env8, 0)
        e9 = ns9["env_encode"](env9, 0)
        # v9 ownership = 4-channel one-hot at cols 7..10 (v8: scalar code at col 7)
        ent8, ent9 = e8[0], e9[0]
        assert torch.equal(ent9[..., :7], ent8[..., :7]), "t=%d body[:7] diverged" % t
        assert torch.equal(ent9[..., 11:], ent8[..., 8:]), "t=%d body[8:]+threat diverged" % t
        b7 = ent8[..., 7]
        oh = ent9[..., 7:11]
        assert torch.equal(oh[..., 0], (b7 == 1.0).to(b7.dtype)), "t=%d self channel" % t
        assert torch.equal(oh[..., 1], (b7 == 2.0).to(b7.dtype)), "t=%d enemy channel" % t
        assert float(oh[..., 2:].abs().max()) == 0.0, "t=%d seat ch 2/3 fired in 2p" % t
        for i, (a8, a9) in enumerate(zip(e8[1:], e9[1:]), 1):
            assert torch.equal(a8, a9), "t=%d encode output %d diverged" % (t, i)
    print("PARITY OK: %d ticks, %d envs -- 2p state bit-exact; encodings equal on shared"
          " features + v9 ownership one-hot consistent with the v8 scalar" % (steps, n_envs))


if __name__ == "__main__":
    main()
