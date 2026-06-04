# Orbit Wars — experiment log

> **TL;DR of the session.** Built a fast comet-aware arena (the fitness function), found the
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
