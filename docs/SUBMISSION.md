# Submitting to Kaggle (Orbit Wars)

The competition is a **Simulations** comp: the submission is a Python agent
(`def agent(obs, config)`) that runs on a **CPU** match-runner with a **~1s/turn** budget
(+60s episode overage bank). The `.pyd` native core is **not** on the eval box, so anything
submitted must be pure-Python + whatever's installed there (numpy, almost certainly torch,
and `kaggle_environments` itself).

## What to submit

Two options, in increasing strength:

### A. Raw policy (simplest)
Wrap a trained checkpoint with `PolicyAgent` and emit `to_kaggle_agent()`. The pointer-actor
`EntityPolicy` forward is small and runs in well under the budget on CPU. The strongest raw
policy here is `runs/native/bc_start.pt` (BC clone of starter: ~96% vs random). Note raw
policies plateau ~0–2% vs starter (over-extension) — option B is the real edge.

### B. Inference search (the edge: 0% → **70%** vs starter)
Lookahead on a bit-exact forward model, opponent modeled as `starter`. The strongest config is
**`mode="per_planet"`** (decide each planet's hold-vs-attack by 1-ply lookahead, H=8) — 70% vs
starter from a 0%-greedy policy. (Joint mode with an all-noop candidate gives 42%.) Ready to
deploy: **`submission.py`** already wires this up on the native-free path. The building blocks: 

- **Forward model:** `orbit_wars_rl/py_engine.py` — pure-Python `step()`, **bit-exact** to the
  native engine (and thus the official one): `tests`/ad-hoc parity 0/357. No `.pyd` needed.
- **Search:** `scripts/search_agent.py:SearchAgent` with `backend="py"` (uses `py_engine`) and
  `fast_encode=False` (uses the Python `EntityObservation`). The winning config:
  `K=8, H=10, value="heuristic", opponent_model="starter", temperature=1.5` (or
  `mode="per_planet"`). Heuristic value (ship + production margin) beats the learned value
  head, which is policy-biased.

To package it as one self-contained file for Kaggle:
1. Concatenate, in one `.py`: the constants + `py_engine.step` (+ helpers), the
   `EntityObservation.process` logic (features + the cheap row context), the `EntityPolicy`
   class (torch) loaded from embedded/bundled weights, the scripted `starter` move (for the
   opponent model + the per-planet target options), and the `SearchAgent` rollout/search.
2. Embed the checkpoint: `base64`-encode `torch.save`'d `state_dict`, or upload a sibling
   `weights.pt` and `torch.load(os.path.join(os.path.dirname(__file__), "weights.pt"))`.
3. Define `def agent(obs, config): return _SEARCH.act(_as_dict(obs), _as_dict(config))`.
4. **Test locally first** (no Kaggle round-trip needed — it's faithful):
   `play_episode`-style vs `starter`, or `search_agent.py --opponent starter`.

Budget: per-turn search is ~60–250ms on CPU at K=8,H=10 (well inside 1s). Keep `H`/`K`
modest and add a wall-clock guard that falls back to the greedy policy move if a turn is
running long, so a slow board never trips `TIMEOUT`.

## Submitting (CLI, already set up)
`.venv\Scripts\kaggle.exe competitions submit -c orbit-wars -f <agent>.py -m "..."`
(auth + TLS fix per the `kaggle-cli-auth-setup` memo). ~5 submissions/day; read agent
stderr from the web (submissions → episode → Agent Logs), not the CLI.
