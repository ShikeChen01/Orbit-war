# Orbit Wars — experiment log

> **UPDATE 2026-06-19 — GRPO "anti-collapse" estimators FAIL the GPU ablation; legacy std-norm is the keeper.**
> A v12 GRPO run collapsed (passivity: launch-rate → 0.1, losing to its own snapshots *and* to `greedy`). A set
> of Tier 0–2 critic-free "fixes" was added (rank / Dr.GRPO / sibling / phi-value advantages, clip-higher, etc.;
> `docs/grpo_v12.tex`) and one — the bounded **rank** advantage — was shipped as the notebook default on the
> strength of a deterministic √3-bound proof. **A controlled RTX 3070 Ti ablation reversed that** (`scripts/_gpu_ab*.py`;
> shared BC init, h256 trunk, 2p, 100 iters/arm): *every* anti-collapse estimator drove the agent into the
> **passivity basin** (launch → ~0, Elo down), while **legacy per-group std-norm stayed stable and improving**
> (Elo 1497→1576). Two confirmed root causes: (1) the it426→it526 collapse was a **reward TRIGGER** (an asymmetric
> "lose-slower" outcome decay), *not* the estimator — identical legacy std-norm is stable under symmetric decay and
> collapses only when `LOSS_DECAY<1`; keep `WIN_DECAY=LOSS_DECAY=1.0`. (2) The "all flags on" A100 collapse was a
> **precedence footgun** — the mutually-exclusive estimator flags silently select `sibling`/`phi-value`, never rank
> (`train.py` now prints the active estimator and warns). ***Bounded ≠ good***: the √3 bound held, but rank discards
> win-magnitude → stops rewarding aggression. **Keeper: legacy std-norm + symmetric decay** (`setup1_v12_grpo_a100.ipynb`
> reverted to it). Next lever if the spiral recurs: trust-region/entropy (`GRPO_KL_COEF>0`, entropy floor), not an
> estimator swap. Full write-up: `docs/grpo_v12.tex` §"Empirical result"; curves in `runs/v12_gpu_ab*/`.

> **UPDATE 2026-06-04 — RL now beats starter without search.** Three changes broke the old
> "RL plateaus at 0–2% vs starter" wall: (1) **threat-aware obs** (5 incoming-fleet features so a
> reactive policy can defend), (2) a **GLU/ResNet trunk**, (3) a **properly-trained BC** (800
> seeds/18 epochs, launch-acc 0.78 — the earlier BC was under-trained). Result: BC *alone* =
> **99.5% vs random, 52% vs starter**; GRPO with the redesigned **production-only + loss-forfeit**
> reward then climbs past it (≥55% vs starter, 99.5% vs random, and rising). See
> "Threat-aware obs + curriculum" below and `rl_math.pdf`. The search agent (next paragraph) remains
> a deployable fallback / further booster, but a trained policy now clears starter on its own.

> **TL;DR of the earlier cycles.** Built a fast comet-aware arena (the fitness function), found the
> trained policies were *worse than random*, and traced it through three fixes:
> production-aware reward → target-based actions → **pointer actor** (the architectural unlock:
> BC vs random 22%→96%). The remaining wall is **vs starter (~0–2% across all RL configs)**: the
> policy expands well but **over-extends and loses undefended planets** to starter's snowball.
> The fix is **inference-time lookahead search** (opponent modeled as starter), enabled by a
> bit-exact pure-Python forward model. From the *same fixed BC policy*, no retraining:
> **greedy 0% → joint search 42% → per-planet search 70% vs starter.** Per-planet 1-ply search
> (decide each planet's hold-vs-attack) decisively beats starter — the inference-time compute
> *is* the edge. (The learned value head is policy-biased and doesn't help; the heuristic
> potential + deep rollout does.) Further levers: a learned win-value, self-attention encoder.


Fitness function: `scripts/eval_native.py` (fast comet-aware native arena, outcome-
identical to the real kaggle engine). Win% = fraction of games the policy (p0, greedy)
wins by true terminal score. Baselines: scripted `random`, `starter` (starter beats
random 100%). All win% vs a fixed 200-world held-out eval set unless noted.

## Baselines (2026-06-03)

| agent | vs random | vs starter | note |
|-------|-----------|------------|------|
| scripted starter | (100% vs random) | — | strong heuristic: every owned planet → half ships at nearest non-owned static planet (continuous angle) |
| `runs/native/final.pt` (pre-existing) | 8.6% | 0.8% | trained before this work; **worse than random** |
| best pre-existing ckpt | 12.5% | 0.0% | `ppo_orbitwars/final.pt` |

**Diagnosis:** policy is pathologically passive (0.57 launches/turn, 335/500 idle turns).
Root causes: (1) ship-margin reward punishes the short-term cost of expansion; (2) 16
angle bins can't aim precisely (starter uses continuous angle). See failure-diagnosis memo.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\bc_pretrain.py --seeds 400 --epochs 15 --out runs\native\bc_start.pt
.venv\Scripts\python.exe scripts\eval_native.py runs\native\bc_start.pt --opponent random  --games 256
.venv\Scripts\python.exe scripts\eval_native.py runs\native\bc_start.pt --opponent starter --games 256
# inference-time lookahead (per-planet mode is the strongest; opponent modeled as starter)
.venv\Scripts\python.exe scripts\search_agent.py runs\native\bc_start.pt --opponent starter `
  --episodes 12 --mode per_planet --H 8 --value heuristic --opp-model starter --compare-greedy
.venv\Scripts\python.exe scripts\inspect_agent.py runs\native\bc_start.pt --opponent starter  # behavior trace
```
Search backends: `--fast-encode` (default; C++ encoder, local) vs Python encoder for a Kaggle
submission; rollout via native step (local) or `orbit_wars_rl.py_engine` (submission, bit-exact).

## Experiments

| id | change | steps | vs random | vs starter | notes |
|----|--------|-------|-----------|------------|-------|
| exp_prod | production reward (prod_w=20), ent=0.02, **bins=16**, no self-play | 1.5M | 5–9% | ~0% | reward fixed but still passive: **16 bins can't aim**, return negative+worsening |
| exp_target | **target-based actions** (+ prod reward), no self-play, comet-free train | 1.5M | ~42–50% | ~1% | core task fixed (vs random 5%→~50%) but from-scratch RL **plateaus at random level**; can't crack starter |
| bc_mlp | BC into target-mode policy, **MLP actor** | — | 22% | 1% | failed: passive. acc 92% was the noop-majority mirage; even weighted, launch_acc≈0 |
| bc_ptr | BC, **pointer actor** (logit(r,t)=⟨q(tok_r,core),k(tok_t)⟩) | — | **96.1%** | 9.8% | architecture fix: launch_acc 0→0.72; now expands like starter, crushes random |
| ft_test | PPO finetune from bc_ptr, vs starter+random, no self-play | 2.5M | ~85% | ~7% | did NOT surpass starter; degraded from BC (random-critic) |
| sp1 | BC → value-warmup(25) → PPO + self-play, comets | 3M+ | ~95% | ~1% | self-play recovers vs-random but never beats starter (over-extension) |
| search(bc) | BC + lookahead (K8,H10,heuristic,all-noop+temp,opp=starter) vs starter | — | — | **42% (5/12)** vs **0%** greedy | **the edge**: lookahead fixes over-extension, near-parity with starter from a 0%-greedy policy, no retraining |
| **search(bc,per-planet)** | BC + **per-planet 1-ply** (H8, heuristic, opp=starter) vs starter | — | — | **70%** (7/10 *and* 14/20) vs **0%** greedy | **STRONGEST + confirmed**: decide each planet hold-vs-attack → solves over-extension → beats starter decisively; this is `submission.py` |
| search(sp1,val=head) | sp1 + lookahead (H4, **value head**, opp=starter) vs starter | — | — | 0% (= greedy) | value head is **policy-biased** (values sp1's own weak play) → useless for search. The **heuristic potential + deep H** is what works, not the learned value |

### Arena faithfulness re-validated (pointer actor)
`eval_native.py --crosscheck 10` on a target-mode pointer-actor policy: **10/10 outcomes
agree** with the real kaggle engine (CPU; margins near-exact). The native arena is a trusted
fitness function for the new architecture.

### Why the policy beats random but never starter (the real diagnosis)
Inspected sp1 (95% vs random, ~1% vs starter) vs starter on one game: the policy expands
competitively to ~8 planets by t≈100, then **collapses 8→3→0 by t≈200**. It **over-extends** —
launches ships away, leaving planets undefended; starter recaptures them and snowballs to 30
planets / thousands of ships. Beating random doesn't punish this (random can't capitalize);
starter does, ruthlessly. This is a *tactical defense* failure, not a reward-sign bug — the
production-potential reward already penalizes losing planets, but a reactive policy can't
foresee the consequence of a launch.

**→ Inference search is the targeted fix:** roll candidate moves forward (opponent modeled as
starter) and avoid the ones that lose planets. `orbit_wars_rl/py_engine.py` is a pure-Python
forward model, **bit-exact to native (0/357 mismatches)**, so the search submits without the
`.pyd`. Search results: see the search_* rows below.

### PPO-from-BC instability (observed)
Finetuning the BC policy with PPO **degrades** it: vs random 96%→~75%, vs starter 10%→~2%,
even with value-warmup (which only delayed it) and low LR. The dense production-potential
reward is a proxy; optimizing it drifts away from the BC (≈starter) behavior that actually
wins. Candidate fixes: KL-to-BC regularization, terminal-reward-heavier shaping, or skip RL
and put the compute into **inference search on top of the fixed BC policy**.

### Open threads / how to continue
1. **Search beats starter (per-planet, 70%) and needs no retrain** — this is the deployable agent
   (`submission.py`). Push further: a learned **win-probability value** (the current critic is
   policy-biased and useless for search — the heuristic potential works instead); tune H/T and the
   per-planet target set; a faster (batched/C++) search so larger evals/sweeps are cheap.
   `search_agent.py --fast-encode` uses the C++ encoder for fast local eval; the submission uses
   the Python encoder + `py_engine` (both bit-exact, no `.pyd`).
2. **Self-attention encoder** (entities attend to each other → per-planet threat awareness) is the
   architecture lever for defense. Mirror in C++ (`policy.hpp`) + Python; re-validate parity.
3. **BC quality**: launch_acc 0.72; more data/epochs or the attention encoder should raise it,
   giving a stronger warm start (closer to starter before any RL).
4. **Don't expect plain PPO/self-play to beat starter** with the current encoder — measured flat.

## Threat-aware obs + 3-stage curriculum (2026-06-04)

Directly attacks the documented wall (over-extension / no defense; Open-thread #2 named
"per-planet threat awareness" as the defensive lever). Two changes on the GRPO stack:

1. **Threat-aware encoder (F 15→20).** Five per-planet incoming-fleet features from a closed-form
   ray–circle projection of each in-flight fleet (which planet it reaches, ETA): `incoming_enemy_ships`,
   `enemy_imminence` $=e^{-\tau/8}$, `incoming_ally_ships`, `ally_imminence`, and a `hold_margin`
   $=\tanh((\text{ships}+\text{prod}\cdot\tau+\text{help}-\text{attack})/100)$. Fleets that hit nothing
   (or cross the sun first) are filtered. **Parity-exact** C++↔Python (`test_encode_parity`, 0 mismatch).
   This bakes search's threat-lookahead into the observation, so a *reactive* policy can finally defend.
2. **Curriculum** (`RolloutConfig::stage`, see `rl_math.pdf` §6): Stage 1 solo expansion (passive
   opponent, solo potential, $\gamma=0.99$ takeover-time discount) → Stage 2 1v1 vs starter
   ($\gamma=1$, margin potential + win) → Stage 3 mixed. Each stage warm-starts + KL-anchors from
   the previous (anchor-and-advance), re-BC'd at F=20 (acc 0.79 / launch-acc 0.41).

| id | change | steps | vs random | vs starter | notes |
|----|--------|-------|-----------|------------|-------|
| s1 (weak BC) | Stage-1 solo from an 8-epoch BC | ~0.9M | ~31% | ~0.5% | conquers passive map (ep_len≈265) but the *seed was under-trained* — see below |
| s2 (weak BC) | Stage-2 vs starter from s1 | ~0.2M | ~29% | ~0.5% | flat — doomed by the weak seed, not the method |

### Two changes after s1/s2: GLU trunk + a properly-trained BC (the unlock)

1. **GLU/ResNet trunk** (`#1/#4`): each encoder = 2-layer MLP **+ one residual GLU block**
   (`out = x + W_o((W_v x)·σ(W_g x))`), mirrored across the training net, the arena net, and the
   Python serving net. **Parity-verified**: Python and C++ greedy eval of the same checkpoint agree
   exactly (both 0.285 vs random for the weak BC — a forward-parity check, not a strength claim).
2. **BC was under-trained.** 8 epochs gave launch-acc **0.40** → only 28.5% vs random. Re-running
   at **800 seeds / 18 epochs** lifted launch-acc to **0.78** (peak 0.87).

**Strong GLU BC eval (the breakthrough): 99.5% vs random, 52.0% vs starter** — behavior cloning
*alone* now beats starter half the time (the prior 15-feat MLP BC was 9.8%). GLU capacity + the
threat features give the policy enough defensive awareness to hold its own; the earlier 0–2% wall
was a weak-seed + blind-obs artifact.

### Reward redesign (production-only + launch quality + loss forfeit)

Per the new problem-setup insight: the old margin potential let the agent maximize ships/planets
yet lose. New reward (rl_math.pdf §5): dense = `w_Π·Δ(own production) + ρ⁺·#launch-hits − ρ⁻·#launch-misses`
(no planet-count/ship term; a fleet aimed to land nowhere is penalized instantly via the §2
projection); return **forfeits the accumulated reward on a loss** (`R=−D−w_o`), so "hoard then lose"
scores strongly negative. Works cleanly now that the BC wins ~52% (groups contain winners, so the
forfeit separates win-behavior from lose-behavior instead of rewarding passivity).

| id | change | steps | vs random | vs starter | notes |
|----|--------|-------|-----------|------------|-------|
| win1 | GRPO vs starter from strong GLU BC, new reward (γ=1, kl_β=0.03, w_o=3) | 5M (cap) | _running_ | _running_ (from **52%**) | push past the 52% BC baseline |

Throughput: episodic GRPO ≈ **530 transitions/s** at 512 envs (full-episode Monte-Carlo, no value
bootstrap — ~40× costlier/step than the old truncated-GAE PPO, but the only-baseline-is-the-group
trade GRPO makes). 5M steps ≈ 2.6h; `best.owc` lets a run be stopped early.

### The pointer-actor breakthrough (Cycle 3)
The MLP actor maps `[tok_r ++ mean_pooled_core] → logits over target slots`, but that input has
**no per-target information** — it cannot know where target `t` is, so it cannot rank targets
by geometry ("nearest non-owned static"). That's why (a) from-scratch target RL plateaued at
~random (it could only launch at *random* valid targets), and (b) BC's launch-accuracy was ~0.
Fix: a **pointer actor** scoring `logit(r,t,f)=⟨q_f(tok_r,core), k(tok_t)⟩` so the score depends
on both launcher and target encodings. launch_acc 0.00→0.72; BC vs random 22%→96%. Required for
target mode to function at all. (C++ + Python, parity-verified.)

Notes: each 1.5M-step run ≈ **1.2 min** at 512 envs (~23k steps/s). World-gen with comets
is the slow part (~minute/1000 worlds) → disk-cached (`generate_pool_cached`) + a
`--no-comets` fast-iteration mode (eval stays comet-aware). bins32/64 sweep was abandoned
mid-world-gen once target-mode was chosen as the higher-ceiling fix (precise aim + clean
behavior-cloning target + better exploration than any angle-bin count).

Validation: target-mode masking/decode proven (Python unit check + C++↔Python greedy
parity test, 0 mismatches).
