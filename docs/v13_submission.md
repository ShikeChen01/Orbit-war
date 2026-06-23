# v13 — Submission Build (train this, then ship)

Status: **IMPLEMENTED in the package**, smoke + `make_v12_nb.py --submission --check` green. Source of
truth is the package `orbit_wars_v12/`; the notebook `notebooks/setup1_v12_submission.ipynb` is
**generated** by `python scripts/make_v12_nb.py --submission`. Do not hand-edit the notebook — edit the
package and regenerate.

This is the single configuration we intend to **train and submit**. It pins one point in the v12/v13
design space:

| Axis | Choice |
|---|---|
| Base | v12 (battle-tested heuristic engine, full information, comets handled — trivial for the most part) |
| Method | **PPO + GAE**, shared-trunk critic head = **2 ResNet-MLP blocks** (`VALUE_RES_BLOCKS=2`) |
| Trunk | `blockseq`, h=256, 8 heads: `res16,attn1,res16,attn1,res32,attn1,res32,attn1,res32,attn1,res32` |
| Aux | one-sided confident-**win bet** head ON (`AUX_WIN_BET` + `AUX_WIN_BET_REWARD`) |
| Shaping | Ng et al. potential shaping ON, **coexisting** with the legacy dense channels |
| Reward | **win pool 1200** + **loss −1000 decayed** + dense channels under a shared **250** game cap |
| Deploy | sparse **critical-step MCTS** — 10 s on steps 0/10/20/30/40, greedy elsewhere |

All of the new behaviour is behind **default-OFF config flags**, so every other notebook in the repo is
byte-for-byte unchanged; only the submission preset turns them on.

---

## 1. Architecture

`ARCH="blockseq"`, `TRUNK_SPEC="res16,attn1,res16,attn1,res32,attn1,res32,attn1,res32,attn1,res32"`,
`HIDDEN=256`, `N_HEADS=8`.

* `res` = pre-LN ResNet-MLP block (per-planet, GLU).
* `attn` = **pure** cross-planet multi-head self-attention (no MLP), so planets exchange information.
* Ordering: 16 res → attn → 16 res → attn → 32 res → attn → 32 res → attn → 32 res → attn → 32 res.

Totals: **160 ResNet-MLP + 5 attention = 165 blocks**.

Parameter / size budget (h=256; ≈132k params per res block, ≈262k per attn block — see
[[v12-blockseq-depth-ceilings]]):

| Part | Count | Params |
|---|---|---|
| res blocks | 160 × ~132k | ~21.1M |
| attn blocks | 5 × ~262k | ~1.3M |
| proj-in / globals / WHERE+GATE heads / value head (2 res) / bet head | — | ~0.4M |
| **Total** | **~165 blocks** | **~22.8M → ~91 MB fp32** |

That is deliberately **at the edge of the 100 MB submission cap** (see [[fp32-submission-size-ceiling]]) —
ship **fp32**; if a final count tips it over 100 MB, fp16 halves it with negligible quality loss. The
attention blocks count ~2× a res block in params, so 160 res + 5 attn ≈ 170 res-equivalents, still under
the measured h=256 fp32 ceiling (~192 res-equivalents).

The critic is the standard v12 **shared-trunk head**, not a separate net: masked-mean pool the trunk
tokens → concat globals → 2 pre-LN ResNet-MLP blocks → scalar, PopArt-normalised.

---

## 2. Method — PPO + GAE

`ALGO="ppo"` (learned critic), `GAMMA=0.997`, `GAE_LAMBDA=0.98`, `CLIP=0.2`, `VF_COEF=0.5`,
`VALUE_WARMUP_ITERS=5` (critic-only warmup after BC before joint PPO), BC warm-start ON
(`BC_ENABLED=True`, clone of the medium bot), then the dual-Elo PFSP self-play league is the curriculum
(unchanged from v12). 2p/4p mixed at the pinned `S4=0.25` (3:1 2p:4p).

---

## 3. Reward design

All raw units are **pre** `PPO_REWARD_SCALE=100` (the buffer divides by it). Per-env, ego = seat 0.

### 3.1 Win pool (1200) — `USE_WIN_POOL=True`, `WIN_POOL=1200`

The win side is a **fixed pool of 1200**. Each alive step pays the survival "drip"
`ALIVE_REWARD = 2`. On a **win**, the ego collects the *rest* of the pool:

```
win_terminal = clamp(WIN_POOL − ALIVE_REWARD · len, min=0)         # remainder after the drip
won_game_total = ALIVE_REWARD·len + win_terminal = WIN_POOL = 1200  # constant, undecayed
```

So a won game totals **exactly 1200** regardless of length. There is no flat WIN_BONUS anymore; the drip
*is* the win reward, front-loaded for credit assignment. Discounting (γ=0.997) still rewards **winning
fast** — the large remainder arrives earlier, so it is discounted less. (Max game = 500 steps →
2·500 = 1000 ≤ 1200, so the remainder is always ≥ 200; the pool never drains mid-game.)

### 3.2 Loss (−1000, decayed) + drip clawback — `LOSS_PENALTY=1000`, `LOSS_DECAY=0.9995`

The loss side is unchanged in spirit: `loss_terminal = −LOSS_DECAY^(len−100) · 1000` ("lose slower" — a
late loss hurts slightly less). **But** the survival drip is **clawed back** at terminal on any non-win:

```
Ot −= (outcome < 0) · ALIVE_REWARD · len
lost_game_total = ALIVE_REWARD·len − decayed(1000) − ALIVE_REWARD·len = −decayed(1000)
```

so a long-surviving loss can **never net positive**. This is the fix for the documented **passivity
ratchet**: without the clawback, a 500-step loss collects +1000 of drip against only ~−820 decayed
penalty → **+180 for losing slowly**, which trains the agent to stall. With the clawback the loser nets
exactly `−decayed(1000)`. (Under discounting the *discounted* return keeps a small positive residual from
the drip arriving earlier than the clawback — intended; it preserves mid-episode credit while the sign of
the return is still dominated by the loss.)

### 3.3 One-sided confident-win bet — `AUX_WIN_BET=True`, `AUX_WIN_BET_REWARD=True`, `_COEF=1.0`

The shared trunk grows a scalar head that bets `b = tanh(head) ∈ [−1,1]` on the eventual outcome
`z ∈ [−1,1]`. Two effects:

1. **Representation aux** (`AUX_WIN_BET_COEF=0.25`): trained to maximise `b·z`, pressuring the trunk to
   discriminate winning vs losing states. Not in the advantage.
2. **Reward** (`AUX_WIN_BET_REWARD_COEF=1.0`): the *detached* bet also earns
   `coef · relu(tanh(b)) · max(z,0)` ≤ **1 per step**, paid **only on won games** and **exactly zero on a
   loss**. One-sided on purpose — the symmetric `b·z` reward is a passivity trap (it pays for confidently
   *losing* → the agent bets against itself and throws games).

> **Note — interpreting "AUX_reward=ON".** The package has two mutually exclusive aux heads:
> `AUX_REWARD_PRED` (predict the immediate reward) and `AUX_WIN_BET` (the bet head, which takes
> precedence). The submission spec asks for **betting**, which *requires* `AUX_WIN_BET`. So
> "AUX_reward=ON" is realised as `AUX_WIN_BET=True` (+ its one-sided reward). You cannot have both heads.

### 3.4 Dense channels under a shared 250 cap — `USE_DENSE_GAME_CAP=True`, `DENSE_GAME_CAP=250`

The three legacy dense channels — **capture**, **prod-milestone**, **launch** — now share **one** hard
game cap of **250** (raw). When a step would push their running sum over 250, the three are scaled down
*together* (their per-step ratio preserved); net-negative steps free room back. This supersedes the old
launch-only `LAUNCH_GAME_CAP=30` as the binding ceiling and guarantees the small shaped channels never
rival the ±1200/−1000 terminal outcome.

### 3.5 Potential shaping ON, coexisting — `USE_POTENTIAL_SHAPING=True`, `DENSE_WITH_SHAPING=True`

Ng et al. potential shaping (ship-margin + production-share, `SHAPE_SHIP=SHAPE_PROD=25`) is added on top.
It is policy-invariant and telescoping (it sums to `Φ_T − Φ_0`), so it is **not** inside the 250 cap.

> **⚠ FLAG — shaping vs dense are normally mutually exclusive.** In stock v12, `USE_POTENTIAL_SHAPING`
> *replaces* the capture/prod/launch channels (and zeroes launch). The spec asks for **both** potential
> shaping AND a capped capture/production/launch — which the stock code cannot express. We added a new
> flag `DENSE_WITH_SHAPING=True` that lets them **coexist** (legacy channels run *and* shaping is summed
> on top). The redundancy risk: **ship-margin shaping overlaps the capture channel** (both reward gaining
> ships/planets), so this double-counts board-control reward. If that biases training (watch `c` vs `sh`
> in the heartbeat, and `lnch/st` for passivity), drop to **shaping-only** by setting
> `DENSE_WITH_SHAPING=False` (or **dense-only** by setting `USE_POTENTIAL_SHAPING=False`) — each is a
> one-line change.

### 3.6 Reward summary

| Channel | Flag | Magnitude (raw) | In advantage? |
|---|---|---|---|
| Survival drip | `ALIVE_REWARD` | +2 / alive step | yes |
| Win terminal | `USE_WIN_POOL` | `1200 − 2·len` (won-game total = **1200**) | yes |
| Loss terminal | `LOSS_PENALTY`,`LOSS_DECAY` | `−decayed(1000)` (drip clawed back) | yes |
| Confident-win bet | `AUX_WIN_BET_REWARD` | ≤ +1 / step, won games only | yes |
| Capture + prod-milestone + launch | `USE_DENSE_GAME_CAP` | share one **250** game cap | yes |
| Potential shaping | `USE_POTENTIAL_SHAPING` | ship-margin + prod-share, ±, telescoping | yes (not capped) |

Heartbeat line now reads `R[o c p ln sh al wb]` = outcome, capture, prod-milestone, launch, shaping,
alive-survival, win-bet.

---

## 4. Deployment — critical-step MCTS

Inference (Kaggle, 2-core CPU, ~28 ms/forward fp32) spends a ~60 s compute bank on a few **early
critical turns** and plays greedy otherwise, because the game is decided in the opening.

```python
agent = MctsAgent(cfg, net, opp_kind="greedy", value="heuristic", K=8,
                  crit_schedule=(10, 50, 10.0), bank_s=60.0, reserve_s=10.0)
```

`crit_schedule=(every=10, until=50, secs=10.0)` → a deep **10 s** PUCT search on steps **0/10/20/30/40**
(five critical turns = **50 s** of the bank), a single greedy forward on every other turn, with a **10 s**
turbulence reserve that the schedule never touches.

At ~28 ms/forward, 10 s ≈ **~357 PUCT sims** per critical turn (1 sim ≈ 1 forward + 1 env step; the real
count is a bit lower from snapshot/step overhead). With prior-pruned breadth `b`, depth `D ≈ sims/b`:

| breadth b | 4 | 8 | 16 |
|---|---|---|---|
| depth D ≈ | ~89 | ~45 | ~22 |

So the requested **depth 10 is reached comfortably**, with breadth `K` as large as the bank allows.

> **Note on "depth 10".** Depth is **emergent** from sims/breadth, not a direct knob — `MctsAgent` is
> budget-driven (time + `K` breadth + `max_sims` cap). At the budget above, depth ≫ 10 is available; set
> `K` to trade breadth for depth.

> **Note on leaf value.** Default leaf value is the bounded **heuristic** (historically beats a
> policy-biased critic for leaf eval). Because PPO+GAE **does train the critic**, `value="net"` is now
> available too — worth an A/B at deploy.

---

## 5. How to build / verify / train

```bash
# regenerate the notebook from the package (package is source of truth)
python scripts/make_v12_nb.py --submission
# regenerate + execute a tiny CPU SMOKE end-to-end (rollout/PPO/league/recal/evict)
python scripts/make_v12_nb.py --submission --check
```

Then open `notebooks/setup1_v12_submission.ipynb` on an A100/H100, Run-All (BC warm-start → PPO league →
save league → plot → critical-step MCTS deploy demo). The reward flags live in the single hand-editable
**RUN SETTINGS** cell.

Verified green: `make_v12_nb.py --submission --check` (notebook execs + trains + evicts), and a targeted
CPU smoke with **all** submission flags on (`USE_WIN_POOL`, `DENSE_WITH_SHAPING`, `USE_DENSE_GAME_CAP`,
`AUX_WIN_BET_REWARD`) — both run without error and the new `r_shape` channel reports non-zero.

---

## 6. Flagged decisions (read these)

1. **Shaping + dense coexistence (§3.5).** Stock v12 makes them mutually exclusive; we added
   `DENSE_WITH_SHAPING` to honour "potential shaping ON" *and* "capped capture/production/launch". The
   **ship-margin/capture double-count** is the open risk — one-line fallbacks given.
2. **"AUX_reward=ON" → `AUX_WIN_BET` (§3.3).** Betting requires the win-bet head, which is mutually
   exclusive with the reward-prediction head, so the bet head is what's enabled.
3. **Loss-side drip clawback (§3.2).** The spec says "win pool 1200 / lose −1000". Paying the +2 drip to
   losers *and* a decayed −1000 makes a long loss net **positive** (passivity). We claw the drip back on
   non-wins so the loser nets exactly `−decayed(1000)`, matching the stated totals. If you instead want
   the loser to keep the survival incentive, set `USE_WIN_POOL=False` and use the legacy additive
   `WIN_BONUS`/`ALIVE_REWARD` — but that reopens the ratchet.
4. **Depth 10 is emergent (§4)**, not a parameter; the budget reaches it easily.
5. **Size is at the cap (§1).** ~91 MB fp32 — ship fp32; fp16 if a final count tips over 100 MB.

Related: [[v12-blockseq-mc]], [[v12-blockseq-depth-ceilings]], [[v12-aux-win-bet-head]],
[[v12-reward-rollback-decay-retraction]], [[v12-deep-lean-critstep-mcts]], [[fp32-submission-size-ceiling]],
[[v13-motion-features]].
