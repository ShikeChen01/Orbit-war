"""Verify the official-comet integration in setup1_v6_ppo.ipynb.

1. BIT-EXACT: the notebook's _comet_paths_official == REFERENCE_orbit_wars.generate_comet_paths
   for identical (planets, ang_vel, spawn_step, rng) across many seeds; ships draw order matches.
2. E2E (CPU GpuEnv): comets spawn at the schedule, positions replay path waypoints exactly,
   expiry frees the slot at path end, at most 4 comets alive.

    python scripts/test_official_comets.py
"""
import json, os, random, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
# exec the REFERENCE functions without importing the module (it loads orbit_wars.json at import)
_ref_src = open(os.path.join(REPO, "REFERENCE_orbit_wars.py"), encoding="utf-8").read()
_ref_ns = {}
exec(compile(_ref_src[:_ref_src.index("def interpreter(")], "REFERENCE_orbit_wars.py", "exec"), _ref_ns)


class _Ref:
    generate_comet_paths = staticmethod(_ref_ns["generate_comet_paths"])


ref = _Ref()

import torch

# ---- load notebook defs up to and including env_step ----
doc = json.load(open(os.path.join(REPO, "notebooks", "legacy", "setup1_v6_ppo.ipynb"), encoding="utf-8"))
g = {"__name__": "__comet_test__"}
for c in doc["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    exec(compile(s, "nb", "exec"), g)
    if "def env_step(" in s:
        break

# ---------------------------------------------------------------- 1. bit-exact port
n_events = n_match = n_none = 0
for seed in range(25):
    w = g["generate_world"](seed)
    for s in g["COMET_SPAWN_STEPS"]:
        r1 = random.Random(f"orbit_wars-comet-{seed}-{s}")
        r2 = random.Random(f"orbit_wars-comet-{seed}-{s}")
        ours = g["_comet_paths_official"](w["planets"], w["angular_velocity"], s, 4.0, r1)
        offi = ref.generate_comet_paths(w["planets"], w["angular_velocity"], s, set(), 4.0, r2)
        n_events += 1
        if ours is None and offi is None:
            n_none += 1; n_match += 1
            continue
        assert (ours is None) == (offi is None), f"seed {seed} step {s}: None mismatch"
        assert len(ours) == len(offi) == 4
        for m in range(4):
            assert len(ours[m]) == len(offi[m]), f"seed {seed} step {s} m{m}: len"
            for k, (a, b) in enumerate(zip(ours[m], offi[m])):
                assert a[0] == b[0] and a[1] == b[1], f"seed {seed} step {s} m{m} k{k}: {a} != {b}"
        # ships: same rng stream position -> identical min-of-4 draw
        s1 = min(r1.randint(1, 99), r1.randint(1, 99), r1.randint(1, 99), r1.randint(1, 99))
        s2 = min(r2.randint(1, 99), r2.randint(1, 99), r2.randint(1, 99), r2.randint(1, 99))
        assert s1 == s2
        # and the world dict stored exactly this
        e = list(g["COMET_SPAWN_STEPS"]).index(s)
        assert int(w["comet_len"][e]) == min(len(offi[0]), g["COMET_MAX_LEN"])
        assert float(w["comet_ships"][e]) == float(s1)
        for m in range(4):
            for k in range(int(w["comet_len"][e])):
                assert abs(w["comet_paths"][e, m, k, 0] - offi[m][k][0]) < 1e-4
                assert abs(w["comet_paths"][e, m, k, 1] - offi[m][k][1]) < 1e-4
        n_match += 1
print(f"[1] bit-exact port: {n_match}/{n_events} spawn events match (incl. {n_none} both-None)")

# ---------------------------------------------------------------- 2. e2e playback on CPU
cpu = torch.device("cpu")
worlds = [g["generate_world"](100 + i) for i in range(4)]
env = g["GpuEnv"](g["PLANET_CAP"], 256, 500, g["SHIP_SPEED"], cpu)
env.reset(worlds)
B, Ec = env.B, env.Ec
zero_act = torch.zeros(B, Ec, Ec + 1)
SPAWNS = list(g["COMET_SPAWN_STEPS"])
seen = {b: 0 for b in range(B)}
for t in range(300):
    g["env_step"](env, zero_act, 2, None, step_idx=t)   # noop vs noop
    n_comets = ((env.p_is_comet > 0.5) & (env.p_alive > 0.5)).sum(1)
    assert int(n_comets.max()) <= 4, f"t={t}: >4 comets alive"
    for e, s in enumerate(SPAWNS):
        if not (s <= t < s + g["COMET_MAX_LEN"]):
            continue
        k = t - s + 1                                  # waypoint shown after this tick's movement
        for b in range(B):
            L = int(worlds[b]["comet_len"][e])
            if L == 0:
                continue
            for m in range(4):
                sl = int(env.c_slot[b, e, m])
                if k < L:
                    assert sl >= 0, f"b{b} e{e} m{m} t{t}: comet missing"
                    ex = worlds[b]["comet_paths"][e, m, k]
                    px, py = float(env.p_x[b, sl]), float(env.p_y[b, sl])
                    assert abs(px - ex[0]) < 1e-4 and abs(py - ex[1]) < 1e-4, \
                        f"b{b} e{e} m{m} t{t}: pos ({px:.3f},{py:.3f}) != path[{k}] ({ex[0]:.3f},{ex[1]:.3f})"
                    seen[b] += 1
                else:
                    assert sl == -1, f"b{b} e{e} m{m} t{t}: not expired at k={k} >= L={L}"
alive_end = int(((env.p_is_comet > 0.5) & (env.p_alive > 0.5)).sum())
print(f"[2] e2e playback: {sum(seen.values())} comet-tick positions verified exact; "
      f"expiry clean; comets alive at t=300 (events 0-2 done): {alive_end}")
print("ALL OK")
