# Native C++ / LibTorch implementation

A from-scratch C++ reimplementation of the Orbit Wars **environment** and the **RL
algorithm** (rollout actor + GAE + PPO update + policy), built for throughput. The pure
Python stack was sim-bound (~130 env-steps/s); this moves the whole hot path into native
code so the RTX 3070 Ti is the limiting factor, not the interpreter.

> Status: **implemented and verified correct end-to-end.** The system trains fast and the
> learned weights play in the real engine.
>
> **Update 2026-06-04 — the trainer is now GRPO and the policy beats starter.** The learner is
> **GRPO** (group-relative advantage, **no value network**; `ow_train_grpo`), not PPO/GAE. The
> policy is a **threat-aware (F=20) GLU/ResNet** pointer-actor, trained from a strong BC with a
> **production-only + loss-forfeit** reward. Result: **99.5% vs random, ~52–55% vs starter** (a
> *trained* policy, no inference search). Full math in `rl_math.pdf`; the journey in
> `EXPERIMENTS.md`. The PPO/GAE description below is the superseded original design (kept for the
> env/throughput details, which are unchanged).

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
| **Comet-spawn parity** incl. RNG spawn steps (`test_comet_parity.py`) | **bit-exact** over 1138 transitions (11 spawn steps) |
| Encoder parity vs `EntityObservation` (`test_encode_parity.py`) | **0 mismatches** / 330 encodings |
| Policy-forward / weight-export parity (`test_native.py`) | C++ greedy == exported-Python greedy, **0 mismatches** |
| **Native arena vs real kaggle engine** (`eval_native.py --crosscheck`) | **24/24 outcomes agree** on identical seeds |
| End-to-end native PPO on GPU | trains; ~**7,200 env-steps/s** (≈55× the 130/s Python baseline), including policy + PPO update |
| Fast arena throughput | **~250 full games/s** (≈1000× the Python arena) |

Run them: `python -m pytest tests/` (the native tests require the built `.pyd`).

## Evaluation (the fitness function)

`native/rl/arena.hpp` (`Arena`, bound as `native.Arena`) plays a loaded `EntityPolicy`
against a fixed opponent (random / starter / a second policy) in the bit-exact engine
*with comets*, scoring exactly like the competition (terminal total-ship score, highest
wins). It is the project's fitness function: ~250 full games/s vs ~0.25/s for the Python
arena, and outcome-identical to the real engine (24/24 crosscheck on matched seeds; tiny
score-margin differences only, from CUDA-vs-CPU policy float — the *decisions* match).

```powershell
# win rate vs the scripted baselines (256 comet-aware games each, in seconds)
.venv\Scripts\python scripts\eval_native.py runs\native\run.pt --opponent starter --games 256
# prove faithfulness against the real kaggle engine on identical seeds
.venv\Scripts\python scripts\eval_native.py runs\native\run.pt --opponent starter --crosscheck 40
```

## Train

`scripts/train_native.py` runs a **chunked** loop: train a chunk in C++, then evaluate
the live weights in the fast arena (true win rate vs the baselines) — a real learning
curve, not the reward-sign proxy. The best checkpoint by win-rate-vs-starter is saved to
`--out`; the latest to `*.last.pt`. The reward is a **production-aware potential**
(`Δ(ship_margin + prod_weight·prod_margin)`) so capturing a planet is immediately
rewarded — without this the policy collapses to passivity (see the failure-diagnosis
memo). The world pool is disk-cached (`generate_pool_cached`) so re-runs start instantly.

```powershell
.venv\Scripts\python scripts\train_native.py `
  --total-steps 8000000 --num-envs 512 --eval-every 400000 `
  --prod-weight 20 --ent-coef 0.02 --out runs\native\run.pt
```

## Strength pipeline (action space, BC, self-play, search)

Getting a *strong* policy (not just correct machinery) needed four changes, each gated by
the fast arena (see `docs/EXPERIMENTS.md` for numbers):

1. **Production-aware reward.** Dense reward = `Δ(ship_margin + prod_weight·prod_margin)/scale`.
   Ship-margin alone punishes the short-term cost of capturing a planet, so the policy
   collapsed to passivity; rewarding production margin makes a capture immediately positive.
2. **Target-based actions** (`--target-mode`). A per-planet action picks a *target entity*
   + ship-fraction and aims via `atan2` (like starter), instead of a coarse angle bin that
   can't hit small distant planets. `decode_action_target` in `core/encode.hpp`.
3. **Pointer actor** (the key architecture fix). The old MLP actor mapped
   `[tok_r ++ mean_pooled_core] → target logits`, which has **no per-target information** —
   it literally can't tell where target `t` is, so it can't pick good targets (from-scratch
   RL stalled at random-level; BC's launch-accuracy was ~0). The pointer actor scores
   `logit(r,t,f)=⟨q_f(tok_r,core), k(tok_t)⟩` so the score depends on both launcher and
   target. `EntityPolicy(target_mode=True)` builds it (C++ `aq/ak/anoop`, Python
   `actor_q/k/noop`); parity-verified C++↔Python.
4. **BC warm start → PPO finetune → self-play.** Target mode makes starter's move exactly
   representable, so `scripts/bc_pretrain.py` clones starter (class-weighted CE — noop is the
   majority class), then `train_native.py --init-from <bc>.pt` finetunes. PPO from a fresh
   BC policy has a random critic that wrecks the actor, so `--value-warmup-updates N` fits
   the critic first (policy frozen). Self-play (`--opp-self`) then drives past starter.

**Inference search** (`scripts/search_agent.py`): the CPU/1s-per-turn budget + the fast
exact forward model = value-guided lookahead. The policy proposes K candidate turns, each is
rolled out H steps (both seats by the policy) and scored by the value head (or the heuristic
potential); the best is played. ~60ms/turn at K=6,H=6 on CPU. For a Kaggle submission the
same logic runs on a pure-Python step port (the `.pyd` isn't on the eval box).

## Limitations & next steps

- **Policy strength needs compute + tuning.** Short runs (≤1M steps, untuned) do not yet
  beat the scripted `starter`/`random` baselines in the real engine — expected for this
  hard game and action space. The *machinery* is correct (proven by the parity chain);
  strength is a budget/hyperparameter matter. Next: train 10–100M steps, tune
  `reward_scale`/`ent_coef`/`lr`/`gamma`, and curette the opponent mix.
- **Comets: DONE.** Each world now carries a pre-generated comet schedule
  (`native_worldgen.build_comet_schedule`, official `generate_comet_paths` + the same
  seed-derived RNG the engine uses), injected deterministically inside `ow::step` at the
  scheduled steps. Bit-exact incl. spawn steps (`test_comet_parity.py`). Training and the
  arena now run the *full* competition game. Because the comet RNG is seeded identically,
  a native game from seed S has the same comets the real env makes for `{"seed": S}`.
- **Throughput headroom.** 7.2k steps/s is end-to-end (env + GPU policy + PPO). Larger
  `num_envs`, CUDA graphs, and an env-only benchmark would push the env component toward
  the 10⁵ target; the env is no longer the bottleneck — the GPU update is.
- **In-training "winrate" is a reward-sign proxy**, not true win rate; judge strength via
  the arena (`play_episode.py`) against fixed baselines.
- **Custom CUDA kernels** would need the CUDA toolkit (`nvcc`) installed; not required
  today (LibTorch's bundled runtime handles all GPU ops).
```
