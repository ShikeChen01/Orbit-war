# Orbit Wars — Game Reference

Condensed from the official env spec (`kaggle_environments` `orbit_wars` v1.0.9). The
verbatim source is in the repo as `REFERENCE_orbit_wars_README.md` and
`REFERENCE_orbit_wars.py`.

## Objective

Conquer planets orbiting a central sun in continuous 2D space. **2 or 4 players**,
**500 turns**. At the end, score = total ships on owned planets + total ships in owned
fleets; **highest score wins**. The game can also end early by elimination.

## Board

- 100×100 continuous space, origin top-left. Sun at (50, 50), radius 10.
- Fleets that cross the sun are destroyed.
- Everything is placed with 4-fold mirror symmetry for fairness.

## Planets — `[id, owner, x, y, radius, ships, production]`

- `owner`: player 0-3, or `-1` neutral.
- `production`: 1-5 ships/turn when owned; `radius = 1 + ln(production)`.
- 20-40 planets (5-10 symmetric groups of 4). ≥3 groups static, ≥1 group orbiting.
- **Orbiting** planets (`orbital_radius + radius < 50`) rotate at a fixed
  `angular_velocity` (0.025-0.05 rad/turn, in the observation). Others are static.
- **Home planets**: one group; 2p start diagonally opposite; home starts with 10 ships.

## Fleets — `[id, owner, x, y, angle, from_planet_id, ships]`

- Travel in a straight line; ship count fixed in transit.
- Speed scales with size: `1 + (maxSpeed-1) * (log(ships)/log(1000))^1.5`
  (1 ship → 1.0/turn; ~1000 ships → max, default 6.0).
- Removed if they leave the board, cross the sun, or hit a planet (→ combat).
  Collision is continuous over the whole path segment.

## Comets

- Temporary objects on elliptical orbits; spawn in groups of 4 at steps
  50/150/250/350/450. Radius 1, production 1. They appear in `planets` and follow normal
  rules; `comet_planet_ids` lists which planet ids are comets. Leaving the board removes
  them (and their garrison). Removed *before* launches each turn.

## Turn order

1. comet expiration → 2. comet spawning → 3. **fleet launches (your action)** →
4. production → 5. fleet movement + collision checks → 6. planet rotation / comet
movement (can sweep fleets into combat) → 7. combat resolution.

## Combat

Arriving fleets grouped by owner (same-owner summed). Largest force fights second
largest; the difference survives. A surviving attacker either reinforces (same owner) or
fights the garrison — if attackers exceed the garrison the planet flips and keeps the
surplus. Exact ties destroy all attacking ships.

## Observation (per turn)

| Field | Meaning |
|-------|---------|
| `planets` | `[[id, owner, x, y, radius, ships, production], ...]` (incl. comets) |
| `fleets` | `[[id, owner, x, y, angle, from_planet_id, ships], ...]` |
| `player` | your id (0-3) |
| `angular_velocity` | orbiting-planet rotation (rad/turn) |
| `initial_planets` | positions at game start (predict orbits with this + angular_velocity) |
| `comets` / `comet_planet_ids` | comet group paths/indices and which ids are comets |
| `step` | current turn |
| `remainingOverageTime` | spare time budget (seconds) |

## Action

Return a list of moves: `[[from_planet_id, direction_angle, num_ships], ...]`.

- Launch only from planets you own; can't send more than the garrison.
- `angle` in radians (0 = right, π/2 = down). Fleet spawns just outside the planet.
- Multiple launches per turn allowed. Empty list `[]` = do nothing.

## Configuration defaults

| Param | Default |
|-------|---------|
| `episodeSteps` | 500 |
| `actTimeout` | 1 s/turn |
| `shipSpeed` | 6.0 |
| `sunRadius` | 10.0 |
| `boardSize` | 100.0 |
| `cometSpeed` | 4.0 |

## How this maps to the code

- Engine reward is `0` until the end, then `+1`/`-1`. `OrbitWarsEnv` adds a dense
  score-margin reward for RL (see `docs/archive/ARCHITECTURE.md`).
- `EntityObservation` encodes planets/comets as entity rows with masks; fleets are
  summarized into per-planet "pressure" + global features.
- `PerPlanetAction` maps a per-planet categorical to the `[from_id, angle, ships]` list.
