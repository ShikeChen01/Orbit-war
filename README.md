# Orbit Wars RL

Reinforcement-learning setup for the Kaggle [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars)
competition. Training runs on a **native C++/LibTorch PPO** stack (env + policy + learner
in C++), ~55× faster end-to-end than a pure-Python loop, with the native env step
**bit-exact** to the official engine. A thin Python layer wraps the engine, holds the
swappable agent/representation abstractions, and serves trained policies.

> **New here? Read [`docs/SETUP.md`](docs/SETUP.md)**, then
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/NATIVE_CPP.md`](docs/NATIVE_CPP.md),
> and [`docs/GAME_REFERENCE.md`](docs/GAME_REFERENCE.md).

## Layout

```
orbit_wars_rl/      env.game (engine wrapper) · agents · processors · ppo_policy (serving)
                    native_worldgen · _native (compiled core) · _bootstrap (SSL fix)
native/             C++/LibTorch: core (env sim + encoders) · rl (policy + batched env + PPO)
scripts/            train_native · play_episode (arena)
tests/              test_parity · test_encode_parity · test_native
deploy/vertex_ai/   cloud-training scaffold (for later)
docs/               SETUP · ARCHITECTURE · NATIVE_CPP · GAME_REFERENCE · archive/
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
# train PPO natively on the GPU
.venv\Scripts\python.exe scripts\train_native.py --total-steps 2000000 --num-envs 256 --out runs\native\final.pt

# arena: any two agents in the REAL engine (random|starter|noop|<checkpoint>.pt)
.venv\Scripts\python.exe scripts\play_episode.py --p0 runs\native\final.pt --p1 starter --episodes 50

# tests (env step parity, encoder parity, native train+export parity)
.venv\Scripts\python.exe -m pytest
```

## Swapping things (the point of the abstraction layer)

- **Different opponent / self-play**: configure `opp_random`/`opp_starter`/`opp_self` on
  `native.TrainerConfig` (the native trainer's opponent mix).
- **Different state encoding**: implement a new `ObservationProcessor` (and mirror it in
  `native/core/encode.hpp`).
- **Different action parameterization**: implement a new `ActionProcessor`.
- **Use a trained policy anywhere**: `PolicyAgent.load("runs/.../final.pt")` — it's an
  `Agent`, and `.to_kaggle_agent()` produces the submission callable.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.
