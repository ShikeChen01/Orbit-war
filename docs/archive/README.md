# Archive

Historical and superseded material — kept for provenance, **not** current. For the live
docs see [`../SETUP.md`](../SETUP.md), [`../ARCHITECTURE.md`](../ARCHITECTURE.md),
[`../NATIVE_CPP.md`](../NATIVE_CPP.md), and [`../GAME_REFERENCE.md`](../GAME_REFERENCE.md).

## Contents

- **`2026-06-02-initial-setup-report.md`** — point-in-time report of the initial
  pure-Python milestone (env wrapper, abstraction layer, Python PPO on CUDA, and the
  environment-debugging saga). Superseded the same day by the native rewrite. The
  `SSLKEYLOGFILE` / `UV_SYSTEM_CERTS` notes in it are still accurate.
- **`python-training-stack/`** — the original pure-Python **training** code, superseded by
  the native C++/LibTorch trainer. Preserved with its original layout (not importable —
  it's a reference snapshot):
  - `orbit_wars_rl/env/gym_env.py` — `OrbitWarsEnv(gym.Env)` (Gymnasium training wrapper,
    dense reward, pluggable opponents).
  - `orbit_wars_rl/train/` — `TrainConfig`, `PPOTrainer` (Python rollout + GAE + PPO + self-play).
  - `scripts/train.py`, `scripts/smoke_test.py` — Python training launcher + env smoke.
  - `tests/test_env.py` — tests for the Python gym env/processors/policy masking.

  Note: the abstractions these built on (`Agent`, the `processors`, `EntityPolicy`,
  `PolicyAgent`, the engine wrapper) are **still live** in `orbit_wars_rl/` — only the
  Python training *loop* was archived.

## Timeline

| When | Milestone | Outcome |
|------|-----------|---------|
| 2026-06-02 | **M1 — Python RL stack** | env wrapper + swappable agent/processor abstractions + Python PPO on CUDA. End-to-end working; bottleneck ~130 env-steps/s (pure-Python sim). |
| 2026-06-02 | **M2 — Native C++/LibTorch rewrite** | from-scratch C++ env (bit-exact to the official engine) + LibTorch `EntityPolicy` + native batched env + native PPO trainer. ~7,200 env-steps/s (≈55×). Weights export to the Python `PolicyAgent` for arena/submission. Python training loop archived here. |

### What M2 changed
- **Env / training loop**: Python `OrbitWarsEnv` + `PPOTrainer` → native `BatchedEnv` +
  `Trainer` (C++). The dense-reward shaping moved into `native/rl/batched_env.hpp`.
- **Throughput**: the M1 "next step — parallelize env stepping" was realized natively.
- **Kept**: the agent/representation abstractions and serving path; the Python `EntityPolicy`
  is now the weight-export target + forward-parity oracle.
- **Still open** (from M1, unchanged): a longer/tuned training run for absolute strength;
  fleets-as-entities; comets in the native trainer; single-file submission packaging;
  wiring up Vertex AI.
