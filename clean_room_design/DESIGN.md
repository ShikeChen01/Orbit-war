# Orbit Wars — A Clean-Room Design for a Championship Agent

*An independent analysis and recommended solution.*

> **Provenance note.** This document was written from first principles. The
> problem specification below was reconstructed from **external** sources only —
> the public Kaggle competition page, the open-source `kaggle-environments`
> engine source, and the community bug report (Issue #1047) — and not from any
> code, notebook, or note in this repository. The proposed solution is my own
> reasoning about what the *best* approach is, grounded in the published
> literature. Where my conclusions happen to converge with work already done in
> this repository, that convergence is independent corroboration, not borrowing.

---

## 1. Problem setup

### 1.1 The game

Orbit Wars is a real-time strategy game played on a **continuous 100×100 board**
with a **sun of radius 10 at the center**. It is a modern reimagining of the
2010 *Planet Wars* / Galcon genre, with the twist that the battlefield *moves*.
Two formats are played: **1v1 (2-player)** and **free-for-all (4-player)**. A
game lasts **500 turns**.

**Planets.** The map is seeded with 5–10 planet groups (at least 3 of them
static). Each planet has:

- a **production** value `prod ∈ {1..5}`, generating `prod` ships per turn;
- a **radius** `r = 1 + ln(prod)` (bigger producers are bigger targets);
- an **orbit**: planets whose `orbital_radius + radius < 50` **rotate around the
  sun** at a fixed angular velocity; planets farther out are **static**.

Each player begins with a single home planet. Ownership of every other planet is
contested.

**Fleets.** The sole action is to **launch a fleet**:
`[from_planet_id, angle_in_radians, num_ships]`. Multiple launches may be issued
in one turn, from the same or different owned planets. A fleet spawns just
outside the source planet's radius, travels in a **straight line at a fixed
angle**, and moves at a speed that *increases with fleet size*:

```
speed = 1.0 + (max_speed − 1.0) · (ln(ships) / ln(1000))^1.5      (capped at 6)
```

So a 1-ship fleet crawls at ~1 unit/turn while a large fleet approaches the cap
of 6 units/turn — a genuine **logistics trade-off** between mass and speed.

**Combat.** When fleets reach a planet, ships are summed *per attacking player*.
The strongest attacker survives with `survivor = top_ships − second_ships`
(ties for first annihilate to 0). If the survivor is not the current owner and
the surviving force exceeds the garrison, the planet **flips ownership**.

**Comets.** At steps **50, 150, 250, 350, 450**, four comets spawn (one per
quadrant) on highly elliptical orbits at speed ~4. They have radius 1 and
production 1, behave as **first-class planets** (capturable, producing,
launchable, combat-resolving), and are **temporary** — they expire when they
leave the field.

**Win condition.** The game terminates when ≤1 player retains any planets/fleets,
or at the step horizon (effectively step 498). **Score = ships on owned planets +
ships in flight.** Most ships wins; everyone else scores −1.

**Observation.** Each turn an agent receives its `player` id, the global
`angular_velocity`, and variable-length lists of `planets`
(`[id, owner, x, y, radius, ships, production]`), `fleets`, and `comets`, plus
the current `step`.

**Ranking.** Submitted agents play each other; Kaggle maintains a skill
(Elo-like) rating from a **mix of 2p and 4p games**.

### 1.2 Known engine quirks (a real source of edge)

The reference engine is open source and deterministic, and the community has
documented exploitable inconsistencies (Issue #1047) that a top agent must model
*exactly*, not approximately:

1. **Top-minus-second combat** only ever considers the two largest attackers in
   a multi-way fight; a 3rd/4th attacker's ships are deleted with **zero
   effect**. In 4p this strongly rewards being the *single* concentrated
   attacker and punishes piling onto a contested planet.
2. **Comet expiry/collision race** can make it *better to miss* an expiring
   comet than to hit it.
3. **Coordinate storage** is `[id, owner, Y, X]` internally while the agent-side
   API presents `(x, y)` — a foot-gun for anyone computing angles from the wrong
   axis order.
4. **Termination at `episodeSteps − 2`** means the nominal final turn never
   resolves its movement/combat.

These are not bugs to wish away; they are part of the true MDP and the optimal
policy exploits them.

### 1.3 What makes this hard (the design-relevant structure)

| Property | Consequence for the agent |
|---|---|
| Continuous space, **moving** targets, size-dependent speed | Hitting anything requires **intercept (lead) prediction**, not naïve `atan2` aim. |
| Variable, unordered sets of planets/fleets/comets | Observations are **permutation-invariant sets**, not fixed vectors. |
| Action = *variable-length list* of (discrete source, continuous angle, integer count) | A **structured, hybrid, multi-action** policy — the crux of the problem. |
| Reward is **sparse and terminal** over 500 steps | Severe **credit-assignment** problem. |
| 2p *and* 4p, against an open pool of opponents | **Non-transitive** strategy space → needs population/league training; 4p adds **game theory** (don't be the leader). |
| Engine is a fast, exact, deterministic simulator you control | Enables **massive self-play** and optional **test-time search**. |
| Rotational + seat symmetry | Exploitable via **equivariance / augmentation**; ignoring it causes seat overfitting. |

---

## 2. Solution thesis (the one-paragraph version)

> Build a **bit-exact, GPU-vectorized replica** of the engine for throughput.
> Encode the entity sets with a **relational set-transformer** that is fed
> precomputed geometry (intercept times, threat) and is **ego/seat-equivariant**.
> Use a **hybrid action space**: the network chooses *what* (which target, what
> intent, what fraction of the garrison) via pointer/attention heads decoded
> **autoregressively** for multi-launch turns, while a **closed-form intercept
> solver chooses the angle** — removing the single hardest exploration problem.
> Train with **PPO+GAE** (a critic is essential for the sparse 500-step horizon),
> **warm-started by behavioral cloning** of a scripted teacher, inside an
> **AlphaStar-style league** with prioritized fictitious self-play and
> anchored ratings to handle non-transitivity across both formats. Add
> **critical-turn, policy-guided lookahead** at inference as a strength
> multiplier where the time budget allows.

The rest of this document defends each clause.

---

## 3. The decisive design decision: abstract the aiming

The most important judgment call is **what the network should and should not be
asked to learn**. The game has two layers:

- a **strategic** layer — which planets to take/defend, when to expand, how to
  allocate ships, how to position in a 4p FFA; and
- a **geometric** layer — given a chosen target and fleet size, find the firing
  angle that intercepts a moving planet/comet.

The geometric layer is a **solved problem in closed form**. Intercepting a target
moving with (locally) constant velocity `v_t` from a source, with a projectile of
known speed `s` (which we know, because we choose `ships` and `s` is a
deterministic function of it), is the classic *lead/intercept* problem: solve

```
‖ p_target + v_t · t − p_source ‖ = s · t
```

a quadratic (or a 1-step root-find when the target's curvature matters over the
flight time) for the time-to-impact `t`, then aim at the predicted impact point.
For orbiting planets one either linearizes per step or iterates 2–3 times to
convergence; either way it is microseconds of deterministic math.

**Asking a neural policy to discover this from a sparse win/loss signal is a
textbook hard-exploration trap.** A randomly initialized continuous-angle head
almost never lands a fleet, so it almost never wins, so it gets almost no
gradient toward aiming — the policy collapses to passivity. This is the same
pathology that makes sparse-reward continuous control notoriously sample-hungry.

**Recommendation: action abstraction.** The policy emits *target + intent +
ship-fraction*; a deterministic intercept module emits the *angle*. This is a
direct application of **temporal/operator abstraction and hierarchical RL**
(options framework; parameterized-action MDPs): expose the agent a small, highly
informative action interface and let a hand-built controller handle the
analytically-known part. The effective action space shrinks from
"continuous angle × continuous count × choice of source" to
"pointer to a target × a few intent/fraction buckets," which is *learnable* under
sparse reward.

I would keep one escape hatch: an **optional small learned angular residual**
(a few degrees) on top of the analytic angle, so the policy can still express
genuinely strategic non-intercept shots — e.g., aiming where an *enemy* fleet
will be, threading past the sun, or splitting a swarm. Best of both: tractable
core, expressive tail.

This single decision is, in my assessment, worth more than any architecture
choice downstream.

---

## 4. Observation encoding

### 4.1 A relational set-transformer over entities

Observations are variable-length, unordered sets of heterogeneous entities. The
right inductive bias is **permutation invariance with pairwise relational
reasoning**, i.e., **self-attention over entity tokens** — the same entity-encoder
pattern used by AlphaStar and the pooled-entity encoders of OpenAI Five, and
formalized by **Deep Sets** and the **Set Transformer**.

Tokenize each planet/comet/fleet into a feature vector:

- **Per-entity:** owner (as ego-relative seat, see §4.3), production, radius,
  current garrison/fleet size, orbital vs static, angular velocity, comet
  time-to-expire, and — importantly — **raw and log** ship counts (raw preserves
  the linearity that combat math is built on; log compresses the dynamic range
  for the encoder).
- **Ego-relative geometry:** position and velocity expressed relative to the
  *acting* planet and to the sun, not in absolute board coordinates.

### 4.2 Inject geometry as edge features (don't make the net rederive it)

The hardest thing for an attention layer to learn here is **time and reachability
geometry**. So compute it and hand it over as **relational/edge features** on the
attention:

- pairwise **time-to-intercept** between each of my planets and each target
  (using the §3 solver, for a couple of representative fleet sizes);
- **arrival vs. threat** timing: for each planet, the soonest an enemy fleet can
  arrive vs. the soonest I can reinforce;
- line-of-fire **occlusion by the sun**.

Feeding precomputed geometry is the analogue of AlphaStar's *scatter connections*
and OpenAI Five's hand-crafted relational features: it lets the network spend its
capacity on *strategy* rather than on rediscovering kinematics.

### 4.3 Symmetry: equivariance, ego-centric framing, and augmentation

The game has two symmetry groups that, if ignored, cause measurable failure:

1. **Rotational symmetry about the sun.** The physics are invariant to rotating
   the entire board.
2. **Seat-permutation symmetry.** Which player you are is arbitrary.

If you train predominantly as "player 0" on absolute coordinates, the network
**overfits to the absolute geometry of seat 0** and plays worse from other seats —
a real and easily-overlooked bug. Two complementary fixes, both standard:

- **Ego-centric / equivariant framing.** Express everything relative to the
  acting agent and canonicalize orientation, and encode ownership as an
  **ego-relative seat one-hot** (me / next / opposite / previous in 4p) so one
  network generalizes across seats — cf. **MDP homomorphic networks** and
  symmetry-aware RL.
- **Symmetry data augmentation.** Rotate the board and permute seats when
  generating training data, exactly as AlphaGo/AlphaZero exploited the 8-fold
  board symmetry of Go. Cheap, and it directly removes seat/orientation
  overfitting. (A *deterministic, exhaustive* seat-and-rotation sweep is stronger
  than random augmentation because it removes the variance entirely.)

---

## 5. Policy architecture (the action space)

The action is a **variable-length list** of launches, each a *(source, angle,
ships)* tuple, with the engine quirk that being one of several attackers can be
wasteful (§1.2). Three viable structures:

1. **Autoregressive multi-launch (recommended for the ceiling).** Decode launches
   one at a time — *pointer to source planet → pointer to target → ship-fraction
   bucket* — with a **STOP token**, capped at K launches/turn. This is exactly
   AlphaStar's **autoregressive action head with pointer networks** over a
   variable entity set, and it is the only structure that natively represents
   "issue a coordinated set of launches this turn" with intra-turn dependencies
   (e.g., "having committed planet A to attack, defend planet B").
2. **Per-owned-planet factorized heads (recommended baseline).** For each owned
   planet, independently output *(launch?, target pointer, fraction)*. Massively
   parallel and simple; coordination across planets flows only through the shared
   encoder. In practice this is a strong, fast 80%-solution and a great first
   milestone.
3. Continuous allocation simplices over targets. Avoid as the primary mechanism
   (see §5.1).

**Ship count → coarse fraction buckets, categorical.** Parameterize ships as a
**fraction of the available garrison** discretized into a few buckets (e.g.
{¼, ½, ¾, all}). Two reasons: (a) the speed-vs-size law is smooth and a handful of
buckets captures the meaningful regimes (fast small probe vs. slow decisive
hammer); (b) **categorical heads are far more stable than continuous heads under
sparse reward.**

### 5.1 Why categorical, not continuous (squashed-Gaussian/Dirichlet)

Continuous policy heads — a squashed Gaussian over the angle, or a Dirichlet over
an allocation simplex — are seductive but brittle here. Under sparse, long-horizon
reward they suffer high-variance reparameterized gradients and **mode collapse**:
the distribution narrows around whatever it stumbled into before it has learned
anything, and exploration dies. Discrete categorical heads with an entropy bonus
keep exploration alive and have well-behaved policy-gradient updates. (This is a
general property of discrete vs. continuous control under sparse reward; it is
*also* why abstracting the angle away entirely, per §3, is so valuable.)

---

## 6. Training algorithm

### 6.1 Core learner: PPO + GAE (a critic is not optional)

I recommend **Proximal Policy Optimization** with **Generalized Advantage
Estimation** as the core learner. PPO is the proven workhorse for large-scale
self-play (OpenAI Five), and **the critic is essential here, not a nicety.**

The reward is a single win/loss-flavored signal at step ~498. Over a 500-step
horizon, a learned **value function provides a low-variance per-step baseline and
bootstraps credit backward** through the long causal chain from "I launched the
right swarm at step 120" to "I had the most ships at step 498." Critic-free
estimators that rely on **group-relative baselines** (averaging the outcomes of a
batch of games from the same start) are attractive for their simplicity but are a
poor fit for this problem: with sparse terminal reward and a long horizon their
advantage estimates are high-variance, and they are prone to a **collapse mode**
where lopsided all-win or all-loss groups produce a degenerate, near-zero or
mis-scaled gradient that stalls learning and drifts the policy toward passivity.
GAE's bias/variance knob (`λ`) is precisely the tool the long horizon needs.

> If one *insists* on a critic-free method for engineering reasons, the
> mitigations are well known — drop the std-normalization (it amplifies
> all-losing groups), use leave-one-out baselines, bound the advantage, and add a
> small auxiliary value signal — but this is swimming upstream. The honest
> recommendation is: **use the critic.**

### 6.2 Reward shaping: potential-based, plus the true terminal signal

Train against the **true terminal win/loss** (so the objective is exactly the
competition's), but add **potential-based shaping** (Ng, Harada & Russell, 1999)
to densify learning. Use a potential `Φ` such as the smoothed lead in *total ships
(planets + fleets)* relative to opponents, or a territory/production-control proxy.
Because shaped reward of the form `r + γΦ(s') − Φ(s)` is **policy-invariant** in
the sense that it does not change the optimal policy, you get faster credit
assignment without biasing the agent away from actually winning. Keep the terminal
W/L term dominant so the agent never learns to farm shaping at the expense of the
result.

### 6.3 Warm-start with behavioral cloning

Cold-starting self-play from random weights on a sparse-reward game is close to
hopeless: random agents never win, so there is no signal to climb. **Bootstrap
with behavioral cloning of a competent scripted teacher** — a greedy
"local-capture / sniper" heuristic (capture the nearest weakly-held high-production
planet you can reach before it reinforces). This is the same move AlphaStar made
with human replays, adapted to a domain where a decent scripted bot is easy to
write. A few minutes of BC produces a non-passive policy that already beats the
starter bot; RL then has gradient to climb from day one. **This is one of the
highest-leverage, lowest-cost steps in the whole plan.**

### 6.4 Population / league training for non-transitivity

A single self-play agent chasing its own latest copy will **cycle** in a
non-transitive strategy space (A beats B beats C beats A) and **forget** how to
beat older styles. The competition is exactly such a space, and 4p FFA amplifies
it. The fix is an **AlphaStar-style league** with **Prioritized Fictitious
Self-Play (PFSP)**:

- Maintain a **population**: *main agents* (the ones you ship), *main exploiters*
  (train fresh against the current main agent to find its holes), and *league
  exploiters* (train against the whole league to find systemic weaknesses).
- **Match-make by win-rate-weighted sampling** (PFSP): spend training games
  against opponents you *almost* beat, which is where the learning gradient is
  richest.
- Keep **historical snapshots** as frozen opponents to prevent strategic
  forgetting.
- This is the practical instantiation of the game-theoretic ideas in **PSRO /
  policy-space response oracles**, the **double-oracle** method, and the
  **open-ended-learning / "gamescape"** analysis of non-transitive games
  (Balduzzi et al., 2019): you are iteratively approximating a Nash mixture over a
  growing population rather than hill-climbing a single point.

**Train one network across both formats**, conditioned on a 2p/4p flag and seat,
and **sample both 2p and 4p games** (the leaderboard mixes them). 4p must be in
the mix because it teaches game-theoretic behaviors a 2p-only agent never sees:
not over-committing, not being the visible leader who gets ganged up on,
opportunistic "kingmaking," and — given the top-minus-second combat quirk — never
being the wasted 3rd/4th attacker on a contested planet.

### 6.5 Rating must be anchored

A population's internal Elo will **inflate/saturate** if it is grounded only on
itself — the numbers go up while real strength plateaus, and the match-maker
starves the formats it has stopped sampling. **Ground ratings against a fixed set
of scripted anchor bots** of known strength so the rating reflects absolute, not
relative-to-a-drifting-pool, skill, and so format sampling stays balanced.

---

## 7. Inference-time search (a strength multiplier, not the foundation)

You *own* a fast, exact, deterministic simulator — which invites **test-time
search**. But two facts temper how far to push it:

1. The per-turn wall-clock budget on Kaggle is small.
2. Orbit Wars is **simultaneous-move** and (in FFA) **multi-agent**, so
   single-agent MCTS's "fixed opponent" assumption is unsound. Proper treatment
   needs **simultaneous-move MCTS / decoupled-UCT**, or solving each node as a
   matrix game via **regret matching / CFR** — and 4p is harder still.

My recommendation is therefore **staged**:

- **Lead with a strong reactive policy.** A well-trained net that runs in
  milliseconds is the reliable backbone and is what most of the strength comes
  from. This is the AlphaZero lesson read in reverse: the policy/value net is
  doing the heavy lifting; search refines it.
- **Add cheap, high-value lookahead on *critical* turns.** Two concrete forms:
  - **1-ply safety rollout.** Before committing a launch that empties a garrison,
    simulate forward with the engine to verify the planet won't be lost to
    already-in-flight enemy fleets. This is a few simulator steps and prevents the
    single most common blunder (over-extension).
  - **Policy-guided narrow search.** On decision-heavy turns (an imminent capture
    or defense), expand only the net's **top-k proposed actions** and evaluate
    leaves with the **value head** — AlphaZero's policy-as-prior narrowing,
    budgeted to the turns that matter. For 2p, use a decoupled/SM-MCTS node rule;
    for 4p, keep it shallow and opponent-modeled by the league policy.

Treat full search as a **stretch goal**. The expected-value ordering is: bit-exact
sim ≫ entity-transformer + hybrid action ≫ PPO+league ≫ BC warm-start ≫ critical
-turn search. Do them in that order.

---

## 8. Engineering substrate: throughput is the real bottleneck

Self-play RL is **rollout-bound**, not update-bound. The training signal is gated
by how many games you can simulate per second, and the simulator's cost is
dominated by resolving launches, movement, and combat across thousands of parallel
games. The substrate that makes everything above feasible:

- A **bit-exact, GPU-vectorized re-implementation** of the engine that steps
  thousands of games in lockstep as batched tensor ops — *including the official
  quirks of §1.2*, because if your training sim differs from the judge, you train
  the wrong policy and your aim is systematically off (the Y/X axis trap alone
  will wreck intercept prediction).
- Standard learner-side accelerations (mixed precision, fused optimizer, compiled
  forward pass, on-device statistics to avoid host syncs, gradient checkpointing
  for deep trunks) — but recognize these speed the *update*, while the *rollout*
  is the binding constraint, so the highest-value engineering is **vectorizing the
  environment itself**.
- A small, exact **parity harness** that replays official engine games against the
  GPU sim and asserts tick-by-tick equality. Without this you will silently train
  against a fork of the rules.

---

## 9. End-to-end recommended system

1. **Substrate.** Bit-exact GPU-vectorized simulator (thousands of parallel
   games), parity-checked against the official engine including its quirks.
2. **Perception.** Relational set-transformer over planet/comet/fleet tokens, fed
   precomputed intercept/threat geometry as edge features; ego-relative + seat-one
   -hot; rotation/seat symmetry augmentation (ideally exhaustive).
3. **Action.** Hybrid: pointer-to-target + intent + ship-fraction bucket, decoded
   autoregressively with a STOP token for multi-launch turns (factorized
   per-planet heads as the fast baseline); **angle from a closed-form intercept
   solver** with an optional small learned residual.
4. **Learner.** PPO + GAE with a value head; potential-based dense shaping plus
   the true terminal win/loss.
5. **Bootstrap.** Behavioral cloning from a scripted sniper/greedy teacher.
6. **Population.** AlphaStar-style league (main + main-exploiter + league-exploiter)
   with PFSP, historical snapshots, **mixed 2p/4p**, **anchor-grounded ratings**.
7. **Inference (stretch).** Critical-turn 1-ply safety rollouts and policy-guided
   top-k narrow search with value-head leaf evaluation.

---

## 10. Risks, trade-offs, and honest hedges

- **Throughput is the make-or-break.** If the GPU sim isn't fast and exact, none
  of the rest matters. Budget the most engineering here.
- **The hybrid action abstraction is a bet** that analytic aiming + a small
  residual covers the strategically useful shots. I am confident it covers the
  vast majority; the residual and the optional non-intercept escape hatch are the
  insurance. If profiling shows the policy *wants* exotic angles often, widen the
  residual before abandoning the abstraction.
- **4p game theory is genuinely hard** and partly non-stationary (it depends on
  who else submitted). The league + anchored ratings are the best available
  hedge, but expect 4p strength to be noisier and less transitive than 2p; do not
  over-tune to a single 4p snapshot.
- **Search may not clear the time budget.** Treat it as upside. A net-only agent
  must be strong enough to compete on its own.
- **Rating saturation will lie to you.** Trust **head-to-head vs. fixed anchors
  and held-out opponents**, not the population's internal Elo, when deciding what
  to ship.

---

## 11. Why I believe this is the right shape of solution

The recommendation is not arbitrary — each pillar maps to a published result on a
structurally similar problem:

- **Variable entity sets → attention encoders.** Deep Sets and the Set
  Transformer establish the permutation-invariant inductive bias; AlphaStar and
  OpenAI Five show it works at scale in RTS-like games.
- **Structured multi-action with selection from a set → autoregressive pointer
  heads.** Pointer Networks and AlphaStar's action architecture are the direct
  template.
- **Continuous-but-analytic sub-problem → action abstraction / parameterized
  actions.** The options framework and parameterized-action MDP work (Hausknecht
  & Stone) justify handing the kinematics to a solver and learning only the
  decision.
- **Sparse long-horizon credit → PPO + GAE + potential-based shaping.** Schulman
  et al. (PPO, GAE) and Ng et al. (shaping) are the canonical tools; OpenAI Five
  is the existence proof at horizon scale.
- **Cold-start → behavioral cloning.** AlphaStar's human-BC bootstrap, here with a
  scripted teacher.
- **Non-transitivity across an open pool → league / PFSP / PSRO.** AlphaStar's
  league, Lanctot et al.'s PSRO, the double-oracle method, and Balduzzi et al.'s
  open-ended-learning analysis are exactly about this failure mode.
- **Self-play strength → policy/value net first, search second.** The
  AlphaZero/MuZero line shows the net carries the load and search refines it — and
  the simultaneous-move, time-budgeted, multi-agent reality here pushes search
  toward "critical turns only."

Put together, Orbit Wars is most precisely characterized as **"AlphaStar's
problem (entity sets, structured actions, league self-play, non-transitivity) with
a continuous-control aiming sub-problem that is analytically solvable."** The best
solution is therefore the AlphaStar recipe, *minus* the part you can replace with
closed-form geometry, *plus* a bit-exact simulator that respects the engine's real
(quirky) physics.

---

## 12. References (external literature)

**Core RL algorithms**
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017).
- Schulman et al., *High-Dimensional Continuous Control Using Generalized
  Advantage Estimation* (GAE) (2016).
- Ng, Harada & Russell, *Policy Invariance Under Reward Transformations:
  Theory and Application to Reward Shaping* (1999).
- Sutton, Precup & Singh, *Between MDPs and semi-MDPs: A framework for temporal
  abstraction* (the options framework) (1999).

**Large-scale game agents (the closest analogs)**
- Vinyals et al., *Grandmaster level in StarCraft II using multi-agent
  reinforcement learning* (AlphaStar), Nature (2019). — league, PFSP, entity
  encoder, autoregressive action heads, scatter connections, human-BC, exploiters.
- Berner et al., *Dota 2 with Large Scale Deep Reinforcement Learning* (OpenAI
  Five) (2019). — large-scale PPO self-play, pooled entity observations, long
  horizons.
- Silver et al., *Mastering the game of Go without human knowledge* (AlphaGo Zero)
  (2017); *A general reinforcement learning algorithm…* (AlphaZero) (2018).
- Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a
  Learned Model* (MuZero) (2020).

**Set / pointer / relational architectures**
- Zaheer et al., *Deep Sets* (2017).
- Lee et al., *Set Transformer* (2019).
- Vinyals, Fortunato & Jaitly, *Pointer Networks* (2015).
- Vaswani et al., *Attention Is All You Need* (2017).

**Parameterized / hybrid action spaces**
- Hausknecht & Stone, *Deep Reinforcement Learning in Parameterized Action Space*
  (2016).
- Masson, Ranchod & Konidaris, *Reinforcement Learning with Parameterized
  Actions* (Q-PAMDP) (2016).

**Multi-agent, non-transitivity, and population/league methods**
- Lanctot et al., *A Unified Game-Theoretic Approach to Multiagent Reinforcement
  Learning* (PSRO) (2017).
- McMahan, Gordon & Blum, *Planning in the Presence of Cost Functions Controlled
  by an Adversary* (the double-oracle method) (2003).
- Balduzzi et al., *Open-ended Learning in Symmetric Zero-sum Games* (2019). —
  gamescapes / non-transitivity.
- Heinrich & Silver, *Deep Reinforcement Learning from Self-Play in
  Imperfect-Information Games* (NFSP) (2016).

**Simultaneous-move / imperfect-information search**
- Tak, Lanctot & Winands, *Monte Carlo Tree Search variants for simultaneous move
  games* (decoupled UCT / SM-MCTS) (2014).
- Zinkevich et al., *Regret Minimization in Games with Incomplete Information*
  (CFR) (2007).

**Symmetry / equivariance in RL**
- van der Pol et al., *MDP Homomorphic Networks: Group Symmetries in
  Reinforcement Learning* (2020).

**Problem sources (external, used for the spec in §1)**
- Kaggle, *Orbit Wars* competition page and rules:
  https://www.kaggle.com/competitions/orbit-wars
- Kaggle/`kaggle-environments` engine source (`envs/orbit_wars/orbit_wars.py`):
  https://github.com/Kaggle/kaggle-environments
- *Critical Game Logic Bugs in Orbit Wars Simulation*, Issue #1047:
  https://github.com/Kaggle/kaggle-environments/issues/1047
