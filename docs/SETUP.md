# Setup & onboarding

Current state of the project and how to get going. For the design, see
[`archive/ARCHITECTURE.md`](archive/ARCHITECTURE.md); for the native trainer, [`NATIVE_CPP.md`](NATIVE_CPP.md);
for the game rules, [`GAME_REFERENCE.md`](GAME_REFERENCE.md).

## What this is

An RL setup for the Kaggle **Orbit Wars** competition. The game engine is the official
`kaggle_environments` `orbit_wars` env. Training runs on a **native C++/LibTorch** PPO
stack (env + policy + learner in C++), ~55× faster end-to-end than the original Python
loop, with the native env step **bit-exact** to the official engine. A thin Python layer
provides the engine wrapper, the swappable agent/representation abstractions, and the
serving/eval path (load trained weights → play in the real engine / submit).

## Machine-specific gotchas (important)

This Windows machine runs a TLS-inspection driver. Two consequences:

1. **`uv` networking** needs the system cert store: set `UV_SYSTEM_CERTS=1` for any
   `uv pip` / `uv venv` command.
2. **`import kaggle_environments` would hard-abort** (`OPENSSL_Uplink ... no OPENSSL_Applink`)
   because the driver sets `SSLKEYLOGFILE` to a device path and uv's CPython OpenSSL lacks
   the applink stub. Fixed by stripping the bogus var in
   `.venv/Lib/site-packages/sitecustomize.py` **and** `orbit_wars_rl/_bootstrap.py` (imported
   first by the package). If a fresh venv crashes with the applink message, check
   `SSLKEYLOGFILE` first.

## Environment

- `.venv/` — uv-managed **CPython 3.12** (system Python is 3.14, too new for wheels).
- `kaggle-environments` (Orbit Wars **v1.0.9**), `gymnasium`, `numpy`, `truststore`.
- **`torch 2.11.0+cu128`** (CUDA), verified on an **RTX 3070 Ti**.
- Native build deps: `cmake`, `ninja`, `nanobind` (pip), MSVC 14.51 Build Tools.
- `open_spiel` must stay **uninstalled** (its import crashes on this machine; not needed).

Recreate:

```powershell
$env:UV_SYSTEM_CERTS = "1"
uv venv --python 3.12 .venv
uv pip install --python .venv -e .
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv cmake ninja nanobind
```

## Build the native core

```powershell
$env:NB_DIR    = (.venv\Scripts\python -c "import nanobind;print(nanobind.cmake_dir())")
$env:TORCH_ROOT= (.venv\Scripts\python -c "import torch,os;print(os.path.dirname(torch.__file__))")
cmd /c native\build.cmd     # -> orbit_wars_rl/_native/orbitwars_native.*.pyd
```

LibTorch is linked from the torch wheel (no CUDA toolkit / no admin needed). Details in
[`NATIVE_CPP.md`](NATIVE_CPP.md) and the memory note `native-cpp-build-toolchain`.

## Use it

```powershell
# train (native, GPU)
.venv\Scripts\python scripts\train_native.py --total-steps 2000000 --num-envs 256 --out runs\native\final.pt

# evaluate any two agents in the REAL engine (random|starter|noop|<checkpoint>.pt)
.venv\Scripts\python scripts\play_episode.py --p0 runs\native\final.pt --p1 starter --episodes 50

# tests (env step parity, encoder parity, native train+export parity)
.venv\Scripts\python -m pytest tests\
```

## Project layout

```
orbit_wars_rl/   env.game (engine wrapper) · agents · processors · ppo_policy (serving)
                 native_worldgen · _native (compiled core) · _bootstrap (SSL fix)
native/          C++/LibTorch: core (env sim + encoders) · rl (policy + batched env + PPO)
scripts/         train_native · play_episode (arena)
tests/           test_parity · test_encode_parity · test_native
deploy/vertex_ai/  cloud-training scaffold (later)
docs/            SETUP · ARCHITECTURE · NATIVE_CPP · GAME_REFERENCE · archive/
REFERENCE_*      verbatim copies of the official env spec/rules/tests (the port source)
```

The original pure-Python gym-env + PPO training loop is archived under
[`docs/archive/python-training-stack/`](archive/python-training-stack) (superseded by the
native trainer).

## Status / next steps

The native pipeline is complete and verified correct end-to-end (env/encoder/policy
parity, fast GPU training, weights play in the real engine). **A strong policy still needs
a long training run + tuning**, and the native trainer currently trains the no-comet
variant. See the limitations section of [`NATIVE_CPP.md`](NATIVE_CPP.md).
