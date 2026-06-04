# Architecture

The system has **one ground-truth game** and **two code paths over it**:

- a **native C++/LibTorch** stack that does the heavy lifting — the env, policy, and PPO
  trainer (the **primary training path**, see [`NATIVE_CPP.md`](NATIVE_CPP.md));
- a small **Python layer** that wraps the official engine, defines the swappable
  agent/representation abstractions, serves trained policies, and acts as the
  **spec + parity oracle** the C++ is verified against.

The original pure-Python gym-env + PPO training loop has been **archived**
(`docs/archive/python-training-stack/`); it was superseded by the native trainer (~55×
faster, bit-exact env). What remains in Python is the engine wrapper, the abstractions,
and the serving/eval path.

> **RL math:** the full policy architecture, observation, reward shaping, curriculum, and the
> training update are derived in [`rl_math.pdf`](rl_math.pdf) (source [`rl_math.tex`](rl_math.tex)).
>
> **Current native RL stack (2026-06-04):**
> 1. **Python-free native loop.** `native/apps/` builds `ow_train`/`ow_eval`/`ow_train_grpo`
>    executables that read a cached world pool (`.owp`) and read/write checkpoints (`.owc`) via
>    `native/io/serialize.hpp` — the dev loop is `native\build.cmd` → `native\run.cmd ow_train_grpo …`
>    with no Python. Python is demoted to one-time world-gen, BC warm-start, the Kaggle submission
>    (pure Python, reads `.owc` via `orbit_wars_rl/native_ckpt.py`), and the parity oracle.
> 2. **GRPO trainer (no value network).** `ow_train_grpo` runs grouped episodic rollouts →
>    group-relative advantage → clipped surrogate + KL-to-reference + entropy, with a clean
>    hpp/cpp split: `native/rl/config.hpp`, `native/rl/model/{policy_net,distribution,agent}`,
>    `native/rl/algo/grpo`, `native/rl/{rollout,grpo_trainer}`. Writes a results folder
>    (`metrics.csv` + `best/last.owc` + `config.txt`).
> 3. **Threat-aware observation (F=20).** The encoder adds five incoming-fleet features
>    (projected ships / ETA-imminence / hold-margin) so a reactive policy can *defend* — the gap
>    that previously forced inference-time search. Parity-exact C++↔Python (`test_encode_parity`).
> 4. **Three-stage curriculum** (`RolloutConfig::stage`): solo expansion (γ<1 takeover-time) →
>    1v1 vs starter → mixed opponents, each warm-started + KL-anchored from the previous stage.

```
                ┌──────────────────────────────────────────────┐
                │  kaggle_environments "orbit_wars" (engine)     │  ← ground truth
                └───────▲───────────────────────────┬───────────┘
        replay/eval     │                           │  spec to mirror
                        │                            ▼
   ┌───────────────┐    │      ┌───────────────────────────────────────────┐
   │  Python layer │────┘      │  Native C++/LibTorch  (native/)            │
   │ (live):       │           │   core/  env sim (bit-exact port)          │
   │  env.game     │  parity   │         + encoders + scripted opponents    │
   │  agents/      │◄─────────►│   rl/    EntityPolicy + BatchedEnv + PPO    │
   │  processors/  │  oracle   │          Trainer (rollout+GAE+update)       │
   │  PolicyAgent  │           └───────────────┬───────────────────────────┘
   └───────▲───────┘                           │ get_weights() (numpy)
           │ load weights → arena/submission   ▼
           └──────────────  Python EntityPolicy / PolicyAgent
```

## Ground truth — the game engine

`orbit_wars_rl/env/game.py` is a thin layer over the official `kaggle_environments`
Orbit Wars simulation: constants (`Planet`, `Fleet`, `CENTER`, …), `make_kaggle_env()`,
and scoring helpers. **Nothing else imports kaggle internals.** Both the parity tests and
the final competition submission run against this exact simulation — no train/serve
mismatch. The C++ `core/sim.hpp` is a 1:1 port of its interpreter and is verified
**bit-exact** against it (`tests/test_parity.py`).

## The swappable abstractions (live Python)

These are the conceptual core and are reused by both the serving path and the C++ port.

### Agents (`agents/`) — "switch who is playing"
`Agent` is the one interface everything plays through:

```python
class Agent:
    def reset(self): ...
    def act(self, obs, config) -> list[Move]   # [[from_id, angle, ships], ...]
    def to_kaggle_agent(self)                   # -> agent(obs, config) for kaggle/submission
```

- `RandomAgent`, `StarterAgent`, `NoopAgent` — baselines/opponents (no torch). Mirrored in
  C++ (`core/agents.hpp`) for native rollouts.
- `PolicyAgent` (`agents/ppo_policy.py`) — wraps an `EntityPolicy` as an `Agent`; used to
  **serve** native-trained weights in the arena and the kaggle submission.

### Processors (`processors/`) — "switch the representation"
- `EntityObservation` — each planet → a fixed 20-feature row (geometry, ownership, ships,
  production, fleet pressure, **+5 incoming-fleet threat features**: projected enemy/ally ships,
  ETA-imminence, hold-margin), padded to `max_entities`, with `entity_mask` + `action_mask` +
  a `globals` vector, all relative to the acting player. The C++ `core/encode.hpp` matches it
  **byte-for-byte** (`tests/test_encode_parity.py`).
- `PerPlanetAction` — one categorical per planet (class 0 = no-op). **Target mode** (default):
  class ≥1 names a `(target entity, ship-fraction)` and aims analytically to hit it,
  `1 + max_entities·|fractions|` classes; ownership-safe. (Angle mode names an angle bin instead.)

Want a different state encoding or action parameterization? Implement a new processor (and
mirror it in `core/encode.hpp`); the policy head size follows `actions_per_entity`.

### Policy (`agents/ppo_policy.py`)
`EntityPolicy` (torch): shared per-planet encoder → masked mean-pool ++ globals → core;
per-entity **pointer actor** (target-mode: each owned planet scores `(target, fraction)`
by `⟨q(tok,core), k(tok_target)⟩`, masked categorical). The Python net still carries a value
head for legacy PPO tooling, but **GRPO uses no critic** — the native training net
(`rl/model/policy_net`) drops it. The C++ `rl/policy.hpp` (arena) and `rl/model/policy_net`
(trainer) are twins with identical layer shapes, so BC/native weights load 1:1 — verified by
`tests/test_native.py` (C++ greedy == exported-Python greedy).

## Native training path (primary)

All in C++ (`native/rl/`), so observations stay as `torch::Tensor` and never cross the
Python boundary during training:

1. **`GroupedRollout`** (`rl/rollout`) lays out `B = num_groups · group_size` games where every
   game in a group shares one `(world, opponent)`, steps them in parallel
   (`std::execution::par`), encodes obs to tensors, accumulates the discounted shaped reward,
   and runs each to terminal. Opponent seat is stage-driven: noop (stage 1), starter (stage 2),
   or a random/starter mix (stage 3).
2. **`GrpoTrainer`** (`rl/grpo_trainer`) computes per-episode returns, **group-relative
   advantages** (no critic, no GAE), then clipped-surrogate + KL-to-reference + entropy minibatch
   updates (Adam, grad clip). Logs/evals vs random+starter every 200k steps into `metrics.csv`.
3. **Worlds** come from `native_worldgen.py`, which reuses the *official* map generator —
   so training maps match the competition distribution.

Python only: generate the world pool, call `trainer.train()`, then `trainer.get_weights()`.

## Serving / evaluation path (Python)

`scripts/train_native.py:export_checkpoint` copies `Trainer.get_weights()` (numpy, keyed
to the Python `EntityPolicy.state_dict()`) into a Python `EntityPolicy` and saves a normal
checkpoint. `PolicyAgent.load(ckpt)` then plays it through the **real engine** in
`scripts/play_episode.py` (the arena) and the kaggle submission — reusing the `Agent`
interface, no TorchScript, no second network definition.

## Data shapes (defaults)

| Tensor | Shape | Notes |
|--------|-------|-------|
| `entities` | `(B, 64, 20)` | per-planet features (15 base + 5 threat) |
| `entity_mask` / `action_mask` | `(B, 64)` | real planet / ownable-with-ships |
| `globals` | `(B, 10)` | board summary |
| action | `(B, 64)` int | per-planet class (target-mode: `1 + target·|F| + frac`) |
| logits | `(B, 64, 257)` | target-mode: `1 + 64·4`, masked to legal launches |

## Where things live

| Concern | File(s) |
|---------|---------|
| Game engine wrapper | `orbit_wars_rl/env/game.py` |
| Agent interface + baselines | `orbit_wars_rl/agents/{base,scripted}.py` |
| Representation (obs/action) | `orbit_wars_rl/processors/` |
| Serving a trained policy | `orbit_wars_rl/agents/ppo_policy.py` (`PolicyAgent`) |
| Native env + policy + PPO | `native/core/`, `native/rl/` |
| World generation | `orbit_wars_rl/native_worldgen.py` |
| Train / evaluate | `scripts/train_native.py`, `scripts/play_episode.py` |
| **Archived** Python gym-env + PPO loop | `docs/archive/python-training-stack/` |
