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
- `EntityObservation` — each planet → a fixed feature row, padded to `max_entities`, with
  `entity_mask` + `action_mask` + a `globals` vector, all relative to the acting player.
  The C++ `core/encode.hpp` matches it **byte-for-byte** (`tests/test_encode_parity.py`).
- `PerPlanetAction` — one categorical per planet (class 0 = no-op, else
  `(angle_bin, ship-fraction)`), `1 + angle_bins·|fractions|` classes; ownership-safe.

Want a different state encoding or action parameterization? Implement a new processor (and
mirror it in `core/encode.hpp`); the policy head size follows `actions_per_entity`.

### Policy (`agents/ppo_policy.py`)
`EntityPolicy` (torch): shared per-planet encoder → masked mean-pool ++ globals → core;
per-entity actor head (masked categorical) + value head. The C++ `rl/policy.hpp` is its
twin with identical layer shapes, so native weights load 1:1 — verified by
`tests/test_native.py` (C++ greedy == exported-Python greedy).

## Native training path (primary)

All in C++ (`native/rl/`), so observations stay as `torch::Tensor` and never cross the
Python boundary during training:

1. **`BatchedEnv`** holds `B` independent games, steps them in parallel
   (`std::execution::par`), encodes obs to tensors, computes the **dense reward**
   (per-turn change in `my_score − opp_score` + terminal ±1), and reloads finished games
   from a world pool. Opponent seat = scripted (random/starter) or a frozen self-play snapshot.
2. **`Trainer`** collects a `T×B` rollout, computes GAE, and runs clipped-PPO minibatch
   updates (Adam, entropy bonus, value loss, grad clip), taking self-play snapshots on a
   schedule.
3. **Worlds** come from `native_worldgen.py`, which reuses the *official* map generator —
   so training maps match the competition distribution. (Comets omitted for now; see
   `NATIVE_CPP.md`.)

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
| `entities` | `(B, 64, 15)` | per-planet features |
| `entity_mask` / `action_mask` | `(B, 64)` | real planet / ownable-with-ships |
| `globals` | `(B, 10)` | board summary |
| action | `(B, 64)` int | per-planet class in `[0, 65)` |
| logits | `(B, 64, 65)` | masked to legal launches |

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
