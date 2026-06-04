# Orbit Wars RL

Reinforcement-learning setup for the Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)
competition. Training runs on a **native C++/LibTorch PPO** stack (env + policy + learner
in C++), ~55× faster end-to-end than a pure-Python loop, with the native env step
**bit-exact** to the official engine. A thin Python layer wraps the engine, holds the
swappable agent/representation abstractions, and serves trained policies.

> **New here? Read [`docs/SETUP.md`](docs/SETUP.md)**, then
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/NATIVE_CPP.md`](docs/NATIVE_CPP.md),
> and [`docs/GAME_REFERENCE.md`](docs/GAME_REFERENCE.md). The **strength work** (what makes a
> policy that plays well, with measured results) lives in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Layout

```
orbit_wars_rl/      env.game (engine wrapper) · agents · processors · ppo_policy (serving)
                    native_worldgen (worlds+comet schedules) · py_engine (pure-Python step,
                    for search submission) · _native (compiled core) · _bootstrap (SSL fix)
native/             C++/LibTorch: core (env sim + encoders) · rl (policy + batched env + PPO
                    + arena fitness fn). Policy supports target-mode pointer actor.
scripts/            train_native (chunked PPO + in-train arena eval; --target-mode,
                    --init-from, --value-warmup-updates) · bc_pretrain (clone starter) ·
                    eval_native (fast arena + --crosscheck) · search_agent (lookahead) ·
                    inspect_agent (behavior diagnosis) · sweep_native · play_episode
tests/              test_parity · test_comet_parity · test_encode_parity · test_native
deploy/vertex_ai/   cloud-training scaffold (for later)
docs/               SETUP · ARCHITECTURE · NATIVE_CPP · GAME_REFERENCE · EXPERIMENTS ·
                    SUBMISSION (how to deploy to Kaggle) · archive/
REFERENCE_*         verbatim copies of the official env spec/rules/tests (the port source)
```

The original pure-Python gym-env + PPO training loop is archived under
[`docs/archive/python-training-stack/`](docs/archive/python-training-stack).

## Setup

This machine needs `UV_SYSTEM_CERTS=1` for `uv` networking (TLS interception), and the
venv strips a bogus `SSLKEYLOGFILE` that otherwise crashes `import kaggle_environments`
(see [`docs/SETUP.md`](docs/SETUP.md)). The `.venv/` is already provisioned; to recreate
and build the native core:

```powershell
$env:UV_SYSTEM_CERTS = "1"
uv venv --python 3.12 .venv
uv pip install --python .venv -e .
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv cmake ninja nanobind
$env:NB_DIR    = (.venv\Scripts\python -c "import nanobind;print(nanobind.cmake_dir())")
$env:TORCH_ROOT= (.venv\Scripts\python -c "import torch,os;print(os.path.dirname(torch.__file__))")
cmd /c native\build.cmd
```

## Use it

```powershell
# 1) clone the starter heuristic into a target-mode policy (warm start)
.venv\Scripts\python.exe scripts\bc_pretrain.py --seeds 400 --epochs 15 --out runs\native\bc_start.pt

# 2) PPO finetune + self-play from the BC warm start (comet-aware, value-warmup stabilizes it)
.venv\Scripts\python.exe scripts\train_native.py --target-mode --init-from runs\native\bc_start.pt `
  --total-steps 6000000 --num-envs 512 --value-warmup-updates 25 --opp-self 1.5 --out runs\native\sp1.pt

# 3) evaluate FAST in the comet-aware native arena (~1000x the Python arena, outcome-identical)
.venv\Scripts\python.exe scripts\eval_native.py runs\native\bc_start.pt --opponent starter --games 256
.venv\Scripts\python.exe scripts\eval_native.py runs\native\bc_start.pt --opponent starter --crosscheck 20

# 4) inference-time lookahead on top of a fixed policy (the CPU-budget edge)
.venv\Scripts\python.exe scripts\search_agent.py runs\native\bc_start.pt --opponent starter `
  --episodes 10 --K 8 --H 10 --value heuristic --opp-model starter --compare-greedy

# arena in the REAL engine (random|starter|noop|<checkpoint>.pt); tests (all parity incl. comets)
.venv\Scripts\python.exe scripts\play_episode.py --p0 runs\native\bc_start.pt --p1 starter --episodes 50
.venv\Scripts\python.exe -m pytest
```

**State of play (see `docs/EXPERIMENTS.md`):** BC clones starter (96% vs random); from-scratch
and finetune RL beat random but plateau ~0–2% vs starter — the policy *over-extends* (expands
then loses undefended planets to starter's snowball). **Inference search fixes it — from the
same fixed BC policy: greedy 0% → joint search 42% → per-planet search 70% vs starter.**
Per-planet 1-ply lookahead (decide each planet hold-vs-attack, opponent modeled as starter,
bit-exact `py_engine` forward model — submits without the `.pyd`). The deployable agent is
`submission.py` (+ `docs/SUBMISSION.md`). Next levers: a learned win-value, attention encoder.

## Swapping things (the point of the abstraction layer)

- **Different opponent / self-play**: configure `opp_random`/`opp_starter`/`opp_self` on
  `native.TrainerConfig` (the native trainer's opponent mix).
- **Different state encoding**: implement a new `ObservationProcessor` (and mirror it in
  `native/core/encode.hpp`).
- **Different action parameterization**: implement a new `ActionProcessor`.
- **Use a trained policy anywhere**: `PolicyAgent.load("runs/.../final.pt")` — it's an
  `Agent`, and `.to_kaggle_agent()` produces the submission callable.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.
