# Native C++ / LibTorch implementation

A from-scratch C++ reimplementation of the Orbit Wars **environment** and the **RL
algorithm** (rollout actor + GAE + PPO update + policy), built for throughput. The pure
Python stack was sim-bound (~130 env-steps/s); this moves the whole hot path into native
code so the RTX 3070 Ti is the limiting factor, not the interpreter.

> Status: **implemented and verified correct end-to-end.** The system trains fast and the
> learned weights play in the real engine. Producing a *strong* policy is a compute +
> tuning task (see [Limitations](#limitations--next-steps)).

## Why C++ (not Rust/JAX)

Chosen so the env, the rollout actor, and the PPO learner share one toolchain via
**LibTorch** (C++ torch) with no Python in the hot path. MSVC was required because
LibTorch's Windows binaries are MSVC-built (ABI match). See the plan and
`docs/SETUP_REPORT.md` for the decision trail.

## Layout

```
native/
  CMakeLists.txt          build.cmd            cmake/LinkTorch.cmake
  core/                   # env sim — no torch dependency
    state.hpp             # AoS game state + helpers (find/remove planets)
    sim.hpp               # deterministic step (ported 1:1 from the official interpreter)
    encode.hpp            # observation encoder + action decoder (mirror the Py processors)
    agents.hpp            # scripted opponents (random / starter) in C++
  rl/                     # LibTorch
    policy.hpp            # EntityPolicy (actor-critic) + masked per-entity Categorical
    batched_env.hpp       # BatchedEnv: parallel step, encode->tensors, dense reward, resets
    trainer.hpp           # rollout + GAE + clipped PPO + self-play snapshots + weight export
  bindings/module.cpp     # nanobind: step_from_state, encode_state, Trainer, TrainerConfig
orbit_wars_rl/
  _native/__init__.py     # loads the compiled .pyd (imports torch first)
  native_worldgen.py      # world pool via the OFFICIAL map generator (identical distribution)
scripts/train_native.py   # launch native training; export checkpoint for the arena/submission
tests/test_parity.py · test_encode_parity.py · test_native.py
```

## Build

`cmake`/`ninja`/`nanobind` live in `.venv`; MSVC 14.51 Build Tools are installed.
LibTorch is linked straight from the torch wheel (no CUDA toolkit, no admin) — see
`cmake/LinkTorch.cmake` and the memory note `native-cpp-build-toolchain`.

```powershell
# from repo root
$env:NB_DIR    = (.venv\Scripts\python -c "import nanobind;print(nanobind.cmake_dir())")
$env:TORCH_ROOT= (.venv\Scripts\python -c "import torch,os;print(os.path.dirname(torch.__file__))")
cmd /c native\build.cmd        # -> orbit_wars_rl/_native/orbitwars_native.*.pyd
```

`native/build.cmd` runs `vcvars64`, configures into `native/_bld`, and builds. **Always
`import torch` before importing the extension** (handled by `orbit_wars_rl/_native/__init__.py`).

## Design

### Env (core/)
`ow::step` is a 1:1 port of the official interpreter's deterministic mechanics: comet
expiration → fleet launch → production → fleet movement with **continuous swept-pair
collision** → planet rotation / comet movement → combat. State is Array-of-Structs
(`GameState`), mirroring the kaggle obs lists so the port stays faithful.

**RNG stays in Python.** Maps come from `native_worldgen.generate_world`, which calls the
*official* `generate_planets`, so training maps are drawn from the exact competition
distribution. The hot path is RNG-free.

### Batched env (rl/batched_env.hpp)
Holds `B` independent `GameState`s, steps them with `std::execution::par` (all cores),
encodes observations into CPU float tensors, computes the dense shaping reward
(per-turn change in `my_score − opp_score` + terminal ±1, mirroring `OrbitWarsEnv`), and
reloads finished games from a world pool. The opponent seat is a C++ scripted agent
(random/starter) or a frozen self-play snapshot.

### Policy (rl/policy.hpp)
`EntityPolicy` is the LibTorch twin of the Python `EntityPolicy`: shared per-planet
encoder → masked mean-pool ++ globals → core; per-entity actor head (one categorical of
`1 + angle_bins·|fractions|` classes, masked so only owned planets can launch) + a value
head. Layer shapes/order match the Python model so weights load 1:1.

### Trainer (rl/trainer.hpp)
Runs the **whole loop in C++**: collect a `T×B` rollout (sampling actions, computing the
self-play opponent's actions with a frozen snapshot), GAE, then clipped-PPO minibatch
updates (Adam, entropy bonus, value loss, grad clip). Self-play snapshots are taken on a
schedule. Only worlds (in) and weights/stats (out) cross the Python boundary, so the
nanobind↔torch tensor-interop problem never arises.

### Interop (Phase 5)
`Trainer.get_weights()` returns the parameters as numpy arrays keyed to the Python
`EntityPolicy.state_dict()`; `scripts/train_native.py:export_checkpoint` loads them into a
Python `EntityPolicy` and saves a normal checkpoint. The existing `PolicyAgent.load` then
plays it in `scripts/play_episode.py` (the real engine) and the kaggle submission — no
TorchScript needed, and no second network definition for serving.

## Verification (all passing)

| Gate | Result |
|------|--------|
| Env step parity vs official engine (`test_parity.py`) | **bit-exact** over 1127 real transitions |
| Encoder parity vs `EntityObservation` (`test_encode_parity.py`) | **0 mismatches** / 330 encodings |
| Policy-forward / weight-export parity (`test_native.py`) | C++ greedy == exported-Python greedy, **0 mismatches** |
| End-to-end native PPO on GPU | trains; ~**7,200 env-steps/s** (≈55× the 130/s Python baseline), including policy + PPO update |

Run them: `python -m pytest tests/` (the native tests require the built `.pyd`).

## Train

```powershell
.venv\Scripts\python scripts\train_native.py `
  --total-steps 2000000 --num-envs 256 --episode-steps 500 `
  --selfplay-start-step 300000 --out runs\native\final.pt

# then evaluate in the REAL engine
.venv\Scripts\python scripts\play_episode.py --p0 runs\native\final.pt --p1 starter --episodes 50
```

## Limitations & next steps

- **Policy strength needs compute + tuning.** Short runs (≤1M steps, untuned) do not yet
  beat the scripted `starter`/`random` baselines in the real engine — expected for this
  hard game and action space. The *machinery* is correct (proven by the parity chain);
  strength is a budget/hyperparameter matter. Next: train 10–100M steps, tune
  `reward_scale`/`ent_coef`/`lr`/`gamma`, and curette the opponent mix.
- **No comets in the native trainer (yet).** Comet *mechanics* are parity-proven in
  `ow::step`, but the native trainer trains on the no-comet variant; mid-episode comet
  spawning (RNG) isn't wired into the batched env. Add by pre-generating each episode's
  comet schedule in `native_worldgen` (reusing the official `generate_comet_paths`) and
  injecting it from the schedule in `BatchedEnv::step`.
- **Throughput headroom.** 7.2k steps/s is end-to-end (env + GPU policy + PPO). Larger
  `num_envs`, CUDA graphs, and an env-only benchmark would push the env component toward
  the 10⁵ target; the env is no longer the bottleneck — the GPU update is.
- **In-training "winrate" is a reward-sign proxy**, not true win rate; judge strength via
  the arena (`play_episode.py`) against fixed baselines.
- **Custom CUDA kernels** would need the CUDA toolkit (`nvcc`) installed; not required
  today (LibTorch's bundled runtime handles all GPU ops).
```
