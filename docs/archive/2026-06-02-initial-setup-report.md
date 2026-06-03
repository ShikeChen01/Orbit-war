> **ARCHIVED — historical snapshot (2026-06-02).** This documents the *initial pure-Python*
> milestone (env wrapper, abstraction layer, Python PPO on CUDA). It was superseded the same
> day by the native C++/LibTorch rewrite. For the current state see
> [`docs/SETUP.md`](../SETUP.md), [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md), and
> [`docs/NATIVE_CPP.md`](../NATIVE_CPP.md). The Python training stack described below now
> lives in `docs/archive/python-training-stack/`. The environment gotchas (esp. the
> `SSLKEYLOGFILE` fix) remain accurate and current.

# Orbit Wars RL — Setup Report (initial Python milestone)

What was built, why, and how it was verified. Date: 2026-06-02.

## Goal

Stand up a working RL setup for the Kaggle **Orbit Wars** competition with three pieces:

1. **Agent environment** — the full game simulation.
2. **Training environment** — local, RL-ready.
3. **An abstraction layer** to swap agents (and representations) in and out.

Decision (confirmed with you): **custom PyTorch PPO** on an **NVIDIA GPU (CUDA)**.

---

## 1. What the game actually is (investigation)

The Kaggle pages are JS-rendered, so the overview couldn't be scraped directly. The
authoritative source turned out to be the `kaggle_environments` package itself: the
starter `test.py` calls `make("orbit_wars")`, and the package ships the full game.

The real spec was read from the installed env files (copied locally as `REFERENCE_*`):

- `REFERENCE_orbit_wars.py` — the interpreter/simulation (812 lines).
- `REFERENCE_orbit_wars.json` — the env specification (v1.0.9).
- `REFERENCE_orbit_wars_README.md` — the official rules.
- `REFERENCE_test_orbit_wars.py` — official tests / usage patterns.

See **`docs/GAME_REFERENCE.md`** for the condensed rules. Key facts that shaped the code:

- Board 100×100 continuous, sun at center, 2 or 4 players, 500 turns.
- **Observation**: `planets`, `fleets`, `player`, `angular_velocity`, `initial_planets`,
  `comets`, `comet_planet_ids`, plus `step`/`remainingOverageTime`.
- **Action**: a *variable-length list* of `[from_planet_id, angle, num_ships]`.
- **Engine reward is sparse**: `0` every turn, then `+1` win / `-1` loss at the end
  (`orbit_wars.py` lines ~713-715). Score = total ships on owned planets + in fleets.
- Built-in opponents are registered as `random` and `starter`.
- Single-agent training uses `env.train([None, opponent])` →
  `trainer.step(moves) -> (obs, reward, done, info)`, learner is player 0.

---

## 2. The environment blocker (and the fix)

Getting `import kaggle_environments` to run at all took the most effort. The chain:

1. **Python 3.14.5** was the only interpreter installed — too new for `pygame`/`torch`
   wheels (the first install tried to *build* pygame from source and failed).
   → Used **`uv`** to provision a stable **Python 3.12** venv (`.venv/`).

2. **TLS interception**: this machine runs a monitoring/inspection driver. `uv`'s
   bundled cert store rejected the intercepted certs.
   → Set **`UV_SYSTEM_CERTS=1`** so `uv` uses the Windows cert store. (Use this for any
   `uv` network command on this machine.)

3. **`open_spiel`** got pulled in and its import crashed the process on Windows.
   → Uninstalled it; not needed for Orbit Wars (kaggle_environments skips it cleanly).

4. **The hard one — a hard process abort** with
   `OPENSSL_Uplink(...): no OPENSSL_Applink` on *any* `import kaggle_environments`.
   Root cause, after bisecting: the machine sets the env var **`SSLKEYLOGFILE`** to a
   device path (`\\.\nllMonFltProxy\...`, injected by the TLS-inspection driver).
   `kaggle_environments` creates a default SSL context at import (via `aiohttp`), and
   uv's CPython links an OpenSSL **built without the Windows "applink" stub**. When
   OpenSSL tries to open that keylog path through its file BIO, the process aborts.
   → Fix: strip the bogus `SSLKEYLOGFILE`. Done in two places (belt and suspenders):
     - `.venv/Lib/site-packages/sitecustomize.py` — every interpreter in this venv.
     - `orbit_wars_rl/_bootstrap.py` — imported first by the package, so any
       interpreter using the code is safe too.

This is documented inline in both files and in the project memory so it doesn't bite again.

---

## 3. What was installed

- `.venv/` — uv-managed **CPython 3.12.13**.
- `kaggle-environments` (Orbit Wars **v1.0.9**), `gymnasium`, `numpy`, `truststore`.
- **`torch 2.11.0+cu128`** from the PyTorch CUDA index — verified
  `torch.cuda.is_available() == True` on an **RTX 3070 Ti** (8 GB).
- `pytest` (dev).

Reproduce on this machine:

```powershell
$env:UV_SYSTEM_CERTS = "1"
uv venv --python 3.12 .venv
uv pip install --python .venv -e .
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
uv pip uninstall --python .venv open_spiel   # if it got pulled in
```

---

## 4. What was built

```
orbit_wars_rl/
  _bootstrap.py          SSLKEYLOGFILE fix (import-time)
  env/
    game.py              thin layer over kaggle orbit_wars: constants, scoring, make()
    gym_env.py           OrbitWarsEnv(gym.Env): dense reward + pluggable opponents
  agents/
    base.py              Agent ABC + to_kaggle_agent() + FunctionAgent
    scripted.py          RandomAgent, StarterAgent, NoopAgent
    ppo_policy.py        EntityPolicy (actor-critic) + PolicyAgent (policy-as-agent)
  processors/
    base.py              ObservationProcessor / ActionProcessor ABCs
    observation.py       EntityObservation: per-planet feature tensor + masks + globals
    action.py            PerPlanetAction: per-planet categorical -> kaggle moves
  train/
    config.py            TrainConfig dataclass (all hyperparameters)
    ppo.py               PPOTrainer: rollout + GAE + clipped PPO + self-play pool
scripts/
  smoke_test.py          end-to-end env check
  play_episode.py        arena: any two agents in the real simulation
  train.py               training entry point (CLI overrides any config field)
tests/test_env.py        pytest sanity (env, processors, policy masking)
deploy/vertex_ai/        cloud-training scaffold (for later)
docs/                    this report + architecture + game reference
```

See **`docs/ARCHITECTURE.md`** for the design of the abstraction layer and how to
swap agents / observation encodings / action parameterizations.

---

## 5. Verification (all passing)

- `python scripts/smoke_test.py` — full episode runs; random play loses to `starter`
  with a strongly negative shaped reward (signal is correct).
- `python scripts/play_episode.py --p0 starter --p1 random --episodes 6` —
  `starter` beats `random` **100%** (sanity).
- A tiny PPO run trains on **CUDA** end-to-end: rollouts, GAE, clipped updates,
  self-play snapshots, and win-rate logging all work.
- `pytest` — 4/4 tests pass (env reset/step, termination/scoring, action-processor
  ownership safety, policy forward + action masking).

---

## 6. Known limitations / next steps

- **Throughput is simulation-bound** (~130 env-steps/s): the kaggle game is pure Python
  and a fresh env is created per episode. The GPU is not the bottleneck. Biggest win is
  parallelizing env stepping (multiprocessing / vectorized rollouts) — a change in
  `train/ppo.py` only; the abstraction layer is unaffected.
- **Reward shaping** (`reward_scale`, `terminal_bonus`) and PPO hyperparameters are
  reasonable defaults, not tuned. Value loss starts high; expect to tune `reward_scale`.
- **Action representation** is a solid v1 (per-planet angle×fraction categorical).
  Alternatives (target-planet selection, continuous angle) are drop-in via a new
  `ActionProcessor`.
- **Fleets** are currently summarized into per-planet "pressure" features and globals,
  not modeled as their own action-relevant entities. A future `ObservationProcessor`
  can add a fleet entity stream.
- **Submission packaging** (single-file kaggle agent) is not done yet; `PolicyAgent`
  already exposes `to_kaggle_agent()`, so it's a short follow-up.
- **Cloud training** (`deploy/vertex_ai/`) is scaffolded but intentionally not wired up.
  Note: Docker is not installed locally — use Cloud Build, or install Docker.
