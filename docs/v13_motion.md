# v13 — Motion-Aware Per-Planet Encoding (Implementation Problem Setup)

Status: **IMPLEMENTED in the package** (F_DIM 194→198), smoke + checks green. Source of truth is the
package `orbit_wars_v12/`; the 9 notebooks were regenerated via `scripts/make_v12_nb.py --all`.

Verification (all pass): `scripts/_motion_feat_check.py` (feature values + adapter down-convert
198→194/104/101), `scripts/_win_bet_check.py` (the v12 confidence bonus / win-bet head still trains
and round-trips under F_DIM=198), `scripts/smoke_v12_pkg.py` (full rollout/PPO/league/recal/evict/
resume). The deploy-adapter two-frame persistence (§5.6) and ellipse Phase 2 (§3.3) remain TODO.

This document specifies the v13 observation change, the train/prod parity constraints
that gate it, and the full implementation work breakdown. The learning algorithm is
**unchanged**: PPO + GAE with the value head inline on the shared trunk (no separate
critic network).

---

## 1. Problem statement

The current observation is **motion-blind for two of the three body types** and never
encodes a body's heading as a vector.

What a body's motion looks like in the sim (verified against `env.py` / `worldgen.py`):

| Body type | Motion model | Per-tick update | Deterministic? |
|---|---|---|---|
| **Static planet** | none | `pos` constant | — |
| **Orbital planet** | circular, closed-form | `pos = CENTER + r·(cos(θ₀+ω·t), sin(θ₀+ω·t))` (env.py:736–740) | yes (closed-form) |
| **Comet — legacy** (`COMET_OFFICIAL=False`) | straight line, constant velocity | `pos += (p_comet_vx, p_comet_vy)` (env.py:739–740) | yes (seed) |
| **Comet — official** (`COMET_OFFICIAL=True`, default) | rotated Keplerian **ellipse**, arc-length waypoints | `pos = c_paths[…, k]` waypoint lookup (env.py:759–761) | yes (precomputed) |

Official comets sweep a rotated ellipse: eccentricity `e∈[0.75,0.93]`, semi-major
`a∈[60,150]`, rotation `φ∈[π/6,π/3]`, sampled at constant arc-length speed
`COMET_SPEED=4.0` into 5–40 waypoints (`worldgen.py:147–172`). The full path is
precomputed and deterministic.

What the observation encodes about that motion today (`_encode_core`, env.py:204–230):

- Body feature `b2 = vmag/THREAT_MAX_SPEED` where `vmag = |ang_vel|·dist`, and
  `b3 = cw = sign(ang_vel)` — **but both are gated on `rotating`, and
  `rotating = (~comet) & ((dist+radius) < ROTATION_RADIUS_LIMIT)`** (env.py:205).
- Therefore: **comets get `b2 = b3 = 0`** — their velocity is *completely absent* from
  the per-planet body block.
- `b2`/`b3` give orbital *speed magnitude + spin sign* but **never a velocity
  direction vector**.
- The observation is **single-frame** (no history channel), so the network cannot
  recover heading by differencing frames itself.

Consequence: the policy cannot see which way a comet is moving, cannot anticipate the
elliptical bend, and only gets orbital motion as a scalar speed. The only place comet
velocity currently enters anything is *indirectly*, inside the threat-lead intercept
ETA (env.py:145, `pvx = p_x − p_x_prev`), which is internal to the fleet-targeting math
and is not surfaced to the body features.

**Goal of v13:** give every planet an explicit, prod-faithful motion descriptor so the
model understands each body's physical movement.

---

## 2. The gating constraint: train/prod parity

**Cardinal rule: never feed the model a feature it cannot reconstruct from the
production observation.** Anything available only at training time (from worldgen
ground truth) is an oracle the served agent will not have, and the policy will overfit
to it.

What the **production (Kaggle) observation** provides each tick
(`_emit_obs`, `scripts/package_submission_v9.py:223–244`):

- per **planet**: `[id, owner, x, y, radius, ships, prod]`
- per **fleet**: `[id, owner, x, y, angle, _, ships]`
- top-level: `angular_velocity` (single global scalar), `comet_planet_ids`, `step`

Derivability in prod:

| Quantity | In prod obs directly? | Reconstructable in prod? |
|---|---|---|
| Planet / comet position | yes (`x,y`) | — |
| Orbital angular velocity | yes (global `angular_velocity`) | — |
| Orbital radius / full future orbit | no | yes, exactly (center = sun, `r = |pos−CENTER|`, signed `ω`) |
| **Body velocity** `(vx,vy)` | no | **yes** — 1-step backward finite-diff (stateful adapter keeps prev frame) |
| **Body curvature** `(ax,ay)` | no | **yes** — 2nd backward finite-diff (needs **two** prev frames) |
| Comet **ellipse params** `(e,a,φ,focus)` | no | **no** (clean) — needs an online conic fit over many frames; unreliable, esp. right after spawn |

This is the crux: **velocity and curvature are prod-faithful; ellipse parameters are
not.** Filling true `e,a,φ` from worldgen at training time while the served agent sees
zeros for a comet's first several ticks (exactly when capture decisions are made) is the
oracle trap. Ellipse params are therefore **deferred** (see §4).

---

## 3. v13 feature design

### 3.1 Uniform per-planet motion vectors

Add, for **every** planet (static / orbital / comet alike), two 2-vectors computed by
**backward finite difference** of position:

- **velocity** `(vx, vy) = pos_t − pos_{t−1}`
- **curvature / acceleration** `(ax, ay) = pos_t − 2·pos_{t−1} + pos_{t−2}`

Semantics by body type fall out automatically:

| Body | `(vx,vy)` | `(ax,ay)` |
|---|---|---|
| static | `0,0` | `0,0` |
| orbital | tangential chord | centripetal (inward) second difference |
| comet (legacy) | constant drift | `≈0` (straight line) |
| comet (official) | ellipse chord velocity | bend toward the curve |

Rationale for **backward** diff (not forward / not closed-form): production cannot see
the next tick, so `pos_t − pos_{t−1}` is the only definition reproducible bit-for-bit at
serve time. It is also exactly the quantity the threat-lead already uses internally
(env.py:145), so the threat path and the new body features stay consistent.

`vmag` (`b2`) and `cw` (`b3`) are **kept** as-is. They are redundant with `(vx,vy)` for
orbital bodies but cheap, preserve the existing orbital signal the league was trained
on, and `vmag` is the true instantaneous `|v|` whereas the backward chord slightly
underestimates it for fast rotation. No behavioural reason to remove them.

The comet flag (`b9`) is kept — it lets the model condition its interpretation of the
motion vectors on body type.

### 3.2 Feature layout change

```
N_BODY_FEATURES : 14 → 18      (+vx, +vy, +ax, +ay appended after b10/actable)
N_THREAT_FEATS  : 6            (unchanged)
N_THREAT_FLEETS : 30           (unchanged)
F_DIM           : 194 → 198    (18 body + 6·30 threat)
```

New body indices (appended at the end of the body block to localise the change):

```
14 = vx / V_SCALE
15 = vy / V_SCALE
16 = ax / A_SCALE
17 = ay / A_SCALE
```

Normalisation constants (to add to `constants.py`): propose `V_SCALE = THREAT_MAX_SPEED`
(= 6.0, matches the existing `b2` velocity normaliser; covers max orbital speed ≈3.5 and
`COMET_SPEED=4.0`) and a dedicated `A_SCALE` for the second difference (smaller
magnitude; to be set from a measured distribution of `|ax,ay|` during a short profiling
run — placeholder ≈ `COMET_SPEED` until measured).

### 3.3 Ellipse parameters — deferred (Phase 2)

The "fill them once we have them" intent is kept, but **only behind a parity-safe path**:

- Reserve feature slots for a compact orbit descriptor (e.g. `e`, `a`, normalised
  focus offset, `φ`) **plus a `valid` flag**.
- Populate them from a **single online estimator that runs identically in train and
  prod** (conic fit over accumulated observed positions), emitting `valid=0` and zeros
  until enough frames are seen.
- **Never** fill these slots from worldgen ground truth at training time.

This is out of scope for the first v13 cut; velocity + curvature already deliver the
local motion signal with zero parity risk. Ship Phase 1, then revisit ellipse params if
comet play still underperforms.

---

## 4. Algorithm (unchanged)

- **PPO + GAE**, value head inline on the **shared trunk** — no separate critic network.
- No change to losses, advantage estimation, league, eviction, or curriculum.
- Trunk arch options (`ARCH` mlp / blockseq / transformer) unchanged; they simply ingest
  the wider per-planet entity (`F_DIM 198`).

---

## 5. Implementation work breakdown (no code yet)

Ordered by dependency. Each item notes the parity hazard it must respect.

1. **`constants.py`** — `N_BODY_FEATURES 14→18`, recompute `F_DIM` (→198), add
   `V_SCALE`/`A_SCALE`. Update the `F_DIM_BINARY_THREAT (104)` / `F_DIM_SCALAR_OWNER
   (101)` comments and the body-index notes (the 4 new feats are *appended*, so the
   threat block offset shifts by +4).

2. **`env.py` — second history frame.** Today only `p_x_prev/p_y_prev` are tracked
   (set at env.py:842, init at reset). Add `p_x_prev2/p_y_prev2`, rotate them each step
   (`prev2 ← prev`, `prev ← old`). Init `prev = prev2 = pos` at reset.

3. **`env.py` — spawn-frame reset (parity-critical).** Comets spawn into recycled free
   slots; `p_x_prev` for that slot retains the *previous occupant's* position, so the
   spawn-tick finite-diff is garbage. On every comet spawn
   (`spawn_comets` / `spawn_comets_official`), set `p_x_prev = p_x_prev2 = p_x` (and y)
   for the spawned slots so velocity = curvature = 0 on the appearance frame. The prod
   adapter must do the same for first-seen bodies → train == prod on cold start.

4. **`env.py` — `_encode_core` body stack.** Compute `(vx,vy) = pos − prev`,
   `(ax,ay) = pos − 2·prev + prev2`, normalise, append as channels 14–17. Extend the
   `_encode_core` signature + the `env_encode` wrapper to pass `p_x_prev2/p_y_prev2`
   (mirror the existing `p_x_prev` plumbing, including the `getattr` fallback for
   torch.compile tracing).

5. **`checkpoint.py` — `LegacyObsAdapter`.** Down-conversion to 104/101 layouts must
   **drop** channels 14–17 (legacy nets never had them) and account for the shifted
   threat-block offset. Verify the adapter still reproduces the exact legacy vectors.

6. **Deploy adapter (`scripts/package_submission_*`).** Persist **two** previous frames
   across ticks; apply the same first-seen-body `prev = pos` reset. The agent is
   stateful between calls, so this is bookkeeping, not new information. Confirm the v12
   adapter (not just the v9 one read here) carries the prev-frame state.

7. **Regenerate notebooks** — `scripts/make_v12_nb.py` (and `--grpo`, `--blockseq`,
   etc. variants); `--check` execs them.

8. **Tests / parity:**
   - Encode parity: training `env_encode` must bit-match the deploy adapter for the new
     channels across all ego seats (extend the existing `parity_test`).
   - Spawn-frame test: assert `(vx,vy,ax,ay)=0` on a comet's spawn tick, correct on the
     next ticks, in both env and adapter.
   - History-rotation test: verify `prev2/prev` shift correctly and survive reset.
   - Smoke (`smoke_v12_*`) green.

---

## 6. Consequences & risks

- **Checkpoint break.** `F_DIM 194→198` invalidates existing neural checkpoints →
  **fresh BC warm-start + RL retrain** (same situation as the v12 threat-onehot change).
  Legacy 104/101 members remain playable via the adapter.
- **Parity is the whole ballgame.** The two hazards are (a) the spawn-frame stale-prev
  and (b) the deploy adapter persisting two frames. Both are covered by §5.3 and §5.6
  and must be locked by the §5.8 tests before any training spend.
- **Curvature magnitude** is small and arch-dependent; `A_SCALE` should be set from a
  measured `|ax,ay|` distribution, not guessed, to avoid a near-dead channel.
- **Cost** is negligible: +4 channels per planet entity, no change to the threat block,
  no change to rollout (which is the launch-bound bottleneck).

---

## 7. Explicitly out of scope

- Global fleet-entity stream (top-N fleets with start/dest/owner/size). Considered and
  **dropped**: prod exposes every fleet so it is *possible*, but a `FLEET_CAP=1024`
  second entity stream with cross-attention is heavy on the already launch-bound rollout
  for little gain over the per-planet top-30 threat blocks. Revisit only if attribution
  quality proves limiting.
- Ellipse-parameter features (Phase 2, gated on a parity-safe online estimator, §3.3).
- Any algorithm change — PPO + GAE with the inline value head stays.
