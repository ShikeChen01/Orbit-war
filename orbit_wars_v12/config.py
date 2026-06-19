"""Run-tunable hyperparameters for the v12 stack, as a single frozen :class:`Config`.

This replaces the notebook's flat module-level globals (cell 5). Field NAMES are kept UPPER to
match the notebook 1:1 (and the checkpoint ``config`` blob keys), so a port is a uniform
``NAME -> cfg.NAME`` rewrite and existing ``.pt`` files round-trip unchanged. Fixed game/feature
constants are NOT here -- see :mod:`orbit_wars_v12.constants`.

Use :meth:`Config.create` (mirrors the notebook config cell: applies the SMOKE branch, derives
``B``/paths/``device``, and seeds ``random``/``numpy``/``torch``). Plain ``Config()`` yields the
full-run defaults with no global side effects.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field, fields, replace
from typing import Optional

import numpy as np
import torch

# Model/optim fields that differ between a full run and a fast SMOKE sanity run. `create(smoke=True)`
# applies these on top of the full-run defaults declared on the dataclass.
_SMOKE_FIELDS = dict(
    HIDDEN=64, N_RES_BLOCKS=2, GRAD_CHECKPOINT=False, USE_COMPILE=False, VALUE_RES_BLOCKS=1,
    NUM_GROUPS=4, GROUP_SIZE=4, EPISODE_STEPS=200, FLEET_CAP=512, TOTAL_ITERS=4,
    SELFPLAY_REFRESH=5, N_WORLDS=256, WORLD_RESAMPLE_EVERY=0, ELO_RECAL_EVERY=0,
    LEARNER_RECAL_EVERY=0, LEAGUE_MAX_SNAPSHOTS=4, MINIBATCHES=16,
)

_DEFAULT_INVITE_CFG = {"HIDDEN": 256, "N_TX_LAYERS": 4, "N_HEADS": 8, "TX_MLP_RATIO": 4,
                       "N_STEM_RES": 1, "N_HEAD_RES": 1, "N_RES_BLOCKS": 6, "VALUE_RES_BLOCKS": 2,
                       "ARCH": "transformer", "USE_GLU": True, "USE_ATTENTION": True, "D_G": 32}


@dataclass(frozen=True)
class Config:
    """All v12 hyperparameters. Defaults = the delivered FULL run (SMOKE=False)."""

    # ---- record of the SMOKE flag used to build this config --------------------------------
    SMOKE: bool = False

    # ---- model size (full spec: width 512, deep ResNet trunk) ------------------------------
    HIDDEN: int = 512              # d: trunk width ("model dim")
    N_RES_BLOCKS: int = 86         # residual MLP blocks in the legacy TRUNK [512x86]
    GRAD_CHECKPOINT: bool = True   # activation-checkpoint trunk blocks (the 512x86 OOM fix)
    USE_COMPILE: bool = True       # torch.compile the LEARNER forward (fuses launch-bound kernels)
    COMPILE_MODE: str = "default"  # "default" = fuse, low VRAM | "reduce-overhead" = CUDA graphs
    ACT_FN: str = "tanh"           # ResNet-MLP activation: 'relu'|'tanh'|'gelu'|'silu'
    # ---- trunk architecture (stacked transformer w/ ResNet-MLP ends) -----------------------
    ARCH: str = "trunk"            # "trunk" = legacy (1x attn + N res-MLP) | "transformer" = stack
    N_TX_LAYERS: int = 6           # [transformer] encoder layers
    N_HEADS: int = 12              # [transformer] attention heads (HIDDEN must divide by this)
    TX_MLP_RATIO: int = 4          # [transformer] per-layer MLP hidden = ratio x HIDDEN
    N_STEM_RES: int = 1            # [transformer] ResNet-MLP blocks BEFORE the stack
    N_HEAD_RES: int = 4            # [transformer] ResNet-MLP blocks AFTER the stack
    VALUE_RES_BLOCKS: int = 4      # residual MLP blocks in the VALUE/critic head
    D_G: int = 32                  # board-globals embedding dim
    USE_GLU: bool = True
    USE_ATTENTION: bool = False    # cross-planet self-attention block (legacy trunk; A/B off)

    # ---- env / batch (B = NUM_GROUPS * GROUP_SIZE) -----------------------------------------
    NUM_GROUPS: int = 16
    GROUP_SIZE: int = 16
    B: int = 256                   # parallel envs (derived; create() recomputes from groups)
    EPISODE_STEPS: int = 500
    FLEET_CAP: int = 1024          # max simultaneous in-flight fleets/env
    COMETS_ENABLED: bool = True
    COMET_OFFICIAL: bool = True    # official elliptical waypoint comets (CPU precompute)
    COMET_MAX_LEN: int = 40        # official visible-path cap (5..40 waypoints)
    COMET_SPAWN_STEPS: tuple = (50, 150, 250, 350, 450)
    COMET_SPEED: float = 4.0
    COMET_PAIRS: int = 1
    COMET_SPAWN_RADIUS: float = 50.0
    COMET_PERI_MIN: float = 15.0
    COMET_PERI_MAX: float = 30.0
    COMET_EXPIRE_RADIUS: float = 56.0

    # ---- PPO + GAE -------------------------------------------------------------------------
    LR: float = 1e-4
    GAMMA: float = 0.997
    GAE_LAMBDA: float = 0.98
    CLIP: float = 0.2
    CLIP_HI: float = 0.0           # [D3/DAPO clip-higher] asymmetric UPPER clip 1+CLIP_HI (0 -> symmetric = CLIP)
    RATIO_LENGTHNORM: bool = False # [D1/GSPO] divide the per-step log-ratio by #owned planets (geom-mean ratio)
    VF_COEF: float = 0.5
    ENT_COEF: float = 0.005
    MINIBATCHES: int = 64
    UPDATE_EPOCHS: int = 3
    MAX_GRAD_NORM: float = 0.5
    KL_TARGET: float = 0.05        # stop the epoch once a minibatch KL exceeds this
    LOGRATIO_CLAMP: float = 4.0    # clamp log importance-ratio before exp
    LOGIT_CLAMP: float = 8.0       # clamp dest_logits before softplus
    ADAM_EPS: float = 1e-5

    # ---- auxiliary reward-prediction head (UNREAL-style; PPO path) --------------------------
    # A shallow head off the SHARED trunk regresses the immediate per-step reward r_t as a supervised
    # aux loss. It does NOT change the reward, the advantage, or the critic -- it only adds gradient to
    # the trunk representation, which helps when the RL signal is terminal-dominated. Default OFF, so the
    # net (and every existing checkpoint) is byte-identical until flipped; when on, the head is persisted
    # in the ckpt config blob and rebuilt per-member, so pre-aux league snapshots still load unchanged.
    AUX_REWARD_PRED: bool = False   # build + train the reward-prediction head (consumed by ppo_update only)
    AUX_REWARD_COEF: float = 0.25   # weight of the aux reward MSE loss (shared-trunk regularizer)

    # ---- GRPO (group-relative PO; critic-free alternative to PPO+GAE) -----------------------
    # ALGO selects the optimizer at rollout+update time. "grpo" rolls each world out GRPO_GROUP
    # times and standardizes each rollout's return WITHIN its group as the advantage (no learned
    # value baseline) -- same clipped surrogate + entropy, no value loss, optional KL-to-reference.
    ALGO: str = "ppo"               # "ppo" = PPO+GAE (learned critic) | "grpo" = group baseline (no critic)
    GRPO_GROUP: int = 0             # rollouts sharing one world for the relative baseline (0 -> GROUP_SIZE)
    GRPO_ADV_EPS: float = 1e-4      # std floor when standardizing returns within a group
    GRPO_WHITEN: bool = False       # also batch-whiten advantages on top of the per-group baseline
    GRPO_OUTCOME_ONLY: bool = False # baseline on the terminal outcome only (else the full shaped return)
    GRPO_KL_COEF: float = 0.0       # beta: KL(pi||pi_ref) penalty vs a frozen reference (0 -> off, no ref built)
    # ---- GRPO v12 variance-reduction tricks (docs/grpo_v12.tex). ALL default to the legacy path, so
    # an existing run is byte-for-byte unchanged until a flag is flipped; ablate one at a time.
    GRPO_STD_NORM: bool = True      # [B] divide the centered return by the PER-GROUP std (DeepSeek). False ->
    #                                 center-only + ONE global scale (Dr.GRPO: removes the difficulty-inversion bias)
    GRPO_LOO: bool = False          # [B] leave-one-out group mean baseline (RLOO: unbiased, decorrelates b from sample i)
    GRPO_ANALYTIC_ADV: bool = False # [A] 2p binary closed-form advantage at a Beta-shrunk win-rate (kills the
    #                                 divide-by-eps corner + the within-loss-group length signal); 4p falls back to [B]
    GRPO_ANALYTIC_ALPHA: float = 1.0  # [A] Beta(alpha,alpha) pseudocount shrinking p_hat off {0,1}
    GRPO_PHI_VALUE: bool = False    # [C] use the shaping potential Phi as an OLS-fit GAE value (per-step credit);
    #                                 turns Phi from reward shaping (cancelled by the group mean) into a baseline
    GRPO_PHI_A_MAX: float = 3.0     # [C] clamp on the fitted Phi->outcome slope (guards a degenerate OLS fit)
    GRPO_CRN: bool = False          # [D] common random numbers: couple NEURAL-opponent sampling noise within a
    #                                 world-group so the baseline isolates ego-action variance (lower-variance adv)
    # ---- v12 GRPO advantage-estimator + variance flags (Tier 0-2; docs/grpo_v12.tex). WARNING: these
    #      "anti-collapse" estimators (rank/cvar/sibling + center-only/Dr.GRPO) all caused PASSIVITY in the
    #      3070 Ti ablation (docs/grpo_v12.tex Section "Empirical result"); the KEEPER is the LEGACY default
    #      (GRPO_STD_NORM=True + symmetric decay). The ESTIMATOR flags are MUTUALLY EXCLUSIVE (train.py prints
    #      the active one + warns); all default off. Re-ablate on the 3070 Ti before any A100 use.
    ADV_TARGET_RMS: float = 0.0     # [Tier0/D4] pin the (center-only) advantage RMS to this (0 -> legacy whiten).
    #                                 Stops league-difficulty from becoming a stealth LR schedule via Adam's eps.
    GRPO_RANK_ADV: bool = False     # [C1] within-group RANK advantage (van der Waerden / NES rank-shaping): bounded
    #                                 in [-sqrt3,sqrt3] -- but EMPIRICALLY went passive (bounded != good; discards margin)
    GRPO_RB_GATE: bool = False      # [A1] Rao-Blackwellize the launch gate: de-noise the WHERE gradient with p_i
    #                                 (expected fire) instead of the sampled 0/1 fire_i (exact, zero extra fwd)
    GRPO_OPP_BASELINE_W: float = 0.0  # [E1] subtract the league's Elo-expected outcome vs THIS opponent before
    #                                 centering (a no-op under 1-opponent-per-iter centering; enables E2 stratification)
    GRPO_CVAR_ALPHA: float = 0.0    # [C2] risk-sensitive: baseline on the worst-alpha quantile (0 -> off). 0.5 = mild
    GRPO_CVAR_LAMBDA: float = 1.0   # [C2] extra gradient weight on the tail (worst-alpha) rollouts
    GRPO_AUX_VALUE: bool = False    # [B2/C5] train the (else-unused) value head by MC regression to the return as a
    #                                 representation aux (does NOT enter the critic-free advantage; weights stay PPO-compat)
    GRPO_AUX_COEF: float = 0.1      # [B2] weight of the value aux loss (kept small: shared-trunk regularizer)
    GRPO_SIBLING_BASELINE: bool = False  # [B1] per-step baseline from matched-prefix siblings (shared-root VinePPO):
    #                                 phi_ship-bucketed group mean at each step -> per-step credit (critic-free GAE)
    GRPO_SIBLING_BUCKET_DELTA: float = 0.5  # [B1] phi_ship bucket width (log-ship units) for sibling matching
    GRPO_SIBLING_MIN_PEERS: int = 4         # [B1] fall back to the whole-group mean when a bucket is thinner than this
    GRPO_OPP_STRATA: int = 1        # [E2] RESERVED (not yet wired): would sample this many opponents/iter, one per
    #                                 block of groups, so the baseline averages opponent strength. Needs a per-block
    #                                 opponent rollout (invasive hot-loop change); E1 above is its no-op-until-then companion.

    # ---- training length (the Elo/PFSP league IS the curriculum) ---------------------------
    TOTAL_ITERS: int = 600
    SELFPLAY_REFRESH: int = 25     # snapshot the learner into the pool every N iters
    SNAPSHOT_WARMUP: int = 0       # anchors-only warmup before the first learner snapshot

    # ---- self-play Elo LEAGUE (PFSP pool -- the sole opponent source) ----------------------
    LEAGUE_MAX_SNAPSHOTS: int = 16
    LEAGUE_INIT_CKPTS: list = field(default_factory=list)
    ELO_INIT: float = 0.0
    ELO_RANDOM: float = 0.0        # ANCHORED rating of the random bot -> pins the Elo scale
    ELO_STARTER: float = 1350.0
    ELO_MEDIUM: float = 1560.0
    ELO_GREEDY: float = 1500.0
    ELO_INTERMEDIATE: float = 1425.0
    ADVANCE_TRIGGER_ELO: float = 1500.0
    ADVANCE_CONFIRM_ELO: float = 1350.0
    ELO_K: float = 32.0
    ELO_SCALE: float = 300.0 / math.log10(9.0)   # 314.38: a 300-Elo gap = 90% expected
    ELO_GROUND_KINDS: tuple = ("greedy", "medium")
    MASTER_EVICT_2P_WR: float = 0.90
    MASTER_EVICT_4P_WR: float = 0.75
    MASTER_KEEP_KINDS: tuple = ("random", "greedy")
    PFSP_MODE: str = "even"        # "even" (p*(1-p)) / "hard" ((1-p)^P) / "uniform"
    PFSP_POWER: float = 2.0
    PFSP_FLOOR: float = 0.05
    MATCH_ELO_W: float = 0.7
    MATCH_FP_W: float = 0.3
    SCRIPT_MATCH_BOOST: float = 2.0
    SCRIPT_BOOST_DROP_WR: float = 0.90
    SCRIPT_BOOST_MIN_GAMES: int = 5
    WR_EMA_BETA: float = 0.1
    CAPTURE_RADIUS: float = 25.0
    CAPTURE_MARGIN: float = 2.0
    REBALANCE_PCT: float = 0.4
    REBALANCE_RADIUS: float = 30.0

    # ---- gated allocation action space + sun-reachability ----------------------------------
    REACH_MASK: bool = True
    LEAD_TARGET: bool = True
    WHERE_DIST: str = "categorical"   # 'categorical' (v7 default) | 'dirichlet' (v6 A/B)
    ALLOC_KAPPA: float = 1.0
    MIN_LAUNCH_SHIPS: int = 2
    GATE_TRIM_LO: float = 0.2
    GATE_TRIM_HI: float = 0.8
    GATE_EPS: float = 0.02
    GREEDY_SAMPLE_GATE: bool = True
    INIT_GATE_BIAS: float = -1.0
    POLICY_MODE: str = "ppo"
    LAUNCH_REWARD: float = 0.001
    SELF_LAUNCH_REWARD: float = 0.001
    SELF_LAUNCH_CAP: float = 10.0
    LAUNCH_STEP_CAP: float = 0.5
    LAUNCH_WINDOW: int = 150
    LAUNCH_GAME_CAP: float = 30.0

    # ---- reward (v5; docs/set-ups/1.md) ----------------------------------------------------
    WIN_BONUS: float = 1000.0
    LOSS_PENALTY: float = 1000.0
    WIN_DECAY: float = 1.0
    LOSS_DECAY: float = 1.0
    DECAY_START_STEP: float = 100.0
    CAPTURE_REWARD: float = 30.0
    CAPTURE_LOSS_FRAC: float = 0.9
    CAPTURE_PROD_SCALE: float = 0.2
    PROD_MILESTONE_REWARD: float = 20.0
    PROD_MILESTONE_BASE: float = 100.0
    PPO_REWARD_SCALE: float = 100.0

    # ---- v6: compute-bound A100 + gate/entropy/popart/shaping ------------------------------
    USE_AMP: bool = True
    AMP_DTYPE: torch.dtype = torch.bfloat16
    GREEDY_TAU: float = 0.5
    INIT_DEST_SCALE: float = 0.02
    DEST_HEAD: str = "bilinear"    # 'linear' (per-source) | 'bilinear' (query=src, key=dst)
    ENT_DIR_COEF: float = 0.0
    ENT_COEF_START: float = 0.005
    ENT_COEF_END: float = 0.002
    ENT_DECAY_ITERS: int = 300
    KL_EARLYSTOP: bool = True
    USE_POPART: bool = True
    POPART_BETA: float = 3e-4
    USE_POTENTIAL_SHAPING: bool = True
    SHAPE_SHIP: float = 25.0
    SHAPE_PROD: float = 25.0

    # ---- checkpointing (filled by create() from OW_CKPT_DIR) -------------------------------
    CKPT_DIR: str = "."
    CKPT_PATH: str = "setup1_target_policy.pt"
    BEST_CKPT_PATH: str = "setup1_target_policy_best.pt"
    TRAIN_STATE_PATH: str = "setup1_target_train_state.pt"
    CKPT_EVERY: int = 20

    # ---- BC warm-start + PPO stability guardrails (v7) -------------------------------------
    BC_ENABLED: bool = False
    BC_ROUNDS: int = 12
    BC_EPOCHS: int = 2
    BC_LR: float = 3e-4
    BC_MINIBATCH: int = 4096
    BC_CKPT_PATH: str = "bc_medium_init.pt"
    VALUE_WARMUP_ITERS: int = 5
    LR_WARMUP_ITERS: int = 16
    KL_STOP_MINIBATCH: bool = True
    TRIPWIRE_FRAC: float = 0.4
    TRIPWIRE_BASE_ITERS: int = 10
    KL_HARD_MULT: float = 1.5

    # ---- world pool ------------------------------------------------------------------------
    N_WORLDS: int = 2048
    SEED: int = 0
    WORLD_RESAMPLE_EVERY: int = 100
    ELO_RECAL_EVERY: int = 500
    RESUME_FROM: Optional[str] = None
    ELO_RECAL_ENVS: int = 64

    # ---- v8: mixed 2p/4p + dual Elo + invited league ---------------------------------------
    FOURP_ENABLED: bool = True
    B_4P: int = 256                # parallel envs per 4p iter (derived = B)
    OUTCOME_4P: str = "placement"  # "placement" (linear) | "winner" (official top-only)
    SEAT_SWAP: bool = True
    S4_BASE: float = 0.25          # [pin 3:1 2p:4p] baseline share of 4p iters
    S4_MIN: float = 0.25
    S4_MAX: float = 0.25           # upper == lower -> share_4p pinned at 0.25
    S4_GAIN: float = 1.0
    ELO_K4: float = 16.0
    ELO_COUPLING: float = 0.25
    INVITE_DIRS: list = field(default_factory=lambda: [
        os.environ.get("OW_LEAGUE_DIR", ""), "/content/drive/MyDrive/league_agents",
        "league_invited", "league_agents", os.path.join("notebooks", "league_agents")])
    INVITE_MAX: int = 4
    INVITE_DEFAULT_ELO: float = 750.0
    INVITE_ELO4_OFFSET: float = 0.0
    INVITE_MATCH_BOOST: float = 2.0
    INVITE_MASTER_WR: float = 0.75
    INVITE_MASTERED_W: float = 0.2
    INVITE_CFG: dict = field(default_factory=lambda: dict(_DEFAULT_INVITE_CFG))
    # ---- v9.2: scripted-exposure caps + rating-noise control -------------------------------
    SCRIPT_SHARE_CAP: float = 0.30
    MAX_SCRIPT_SEATS_4P: int = 1
    LEARNER_RECAL_EVERY: int = 50
    RECAL_BLEND: float = 0.5
    S4_EMA_BETA: float = 0.10
    GAUNTLET_NEURAL_W: float = 0.5
    INVITE_MEASURE: bool = True
    GAUNTLET_4P_W: float = 0.5
    # ---- v12: dormant calibration watchdogs ------------------------------------------------
    WATCHDOG_KINDS: tuple = ("random", "greedy")
    WATCHDOG_TOL: float = 0.10

    # ---- runtime ---------------------------------------------------------------------------
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # -----------------------------------------------------------------------------------------
    @classmethod
    def create(cls, smoke: bool = False, **overrides) -> "Config":
        """Build a config the way the notebook config cell does: apply the SMOKE branch, then
        any `overrides`, derive `B`/`B_4P`/checkpoint paths/`device`, seed RNGs, set matmul
        precision, and create the checkpoint dir. Overrides win over the SMOKE branch (so e.g.
        `B=8` directly is honoured, matching the notebook smoke harness)."""
        vals: dict = {"SMOKE": smoke}
        if smoke:
            vals.update(_SMOKE_FIELDS)
        vals.update(overrides)

        defaults = {f.name: f for f in fields(cls)}

        # B / B_4P (derive from groups unless given explicitly)
        ng = vals.get("NUM_GROUPS", defaults["NUM_GROUPS"].default)
        gs = vals.get("GROUP_SIZE", defaults["GROUP_SIZE"].default)
        vals.setdefault("B", ng * gs)
        vals.setdefault("B_4P", vals["B"])

        # checkpoint paths (Colab/Kaggle-friendly, like the notebook)
        ckpt_dir = (vals.get("CKPT_DIR") or os.environ.get("OW_CKPT_DIR")
                    or ("/kaggle/working" if os.path.isdir("/kaggle/working")
                        else "/content" if os.path.isdir("/content") else "."))
        vals["CKPT_DIR"] = ckpt_dir
        vals.setdefault("CKPT_PATH", os.path.join(ckpt_dir, "setup1_target_policy.pt"))
        vals.setdefault("BEST_CKPT_PATH", os.path.join(ckpt_dir, "setup1_target_policy_best.pt"))
        vals.setdefault("TRAIN_STATE_PATH", os.path.join(ckpt_dir, "setup1_target_train_state.pt"))
        vals.setdefault("BC_CKPT_PATH", os.path.join(ckpt_dir, "bc_medium_init.pt"))
        vals.setdefault("INVITE_DIRS", [os.environ.get("OW_LEAGUE_DIR", ""),
                                        "/content/drive/MyDrive/league_agents",
                                        os.path.join(ckpt_dir, "league_invited"), "league_agents",
                                        os.path.join("notebooks", "league_agents")])
        vals.setdefault("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # global side effects matching the notebook config cell
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.set_float32_matmul_precision("high")
        seed = int(vals.get("SEED", defaults["SEED"].default))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        valid = {f.name for f in fields(cls)}
        unknown = set(vals) - valid
        if unknown:
            raise TypeError("unknown Config fields: %s" % ", ".join(sorted(unknown)))
        return cls(**vals)

    def replace(self, **changes) -> "Config":
        """Return a copy with `changes` applied (does not re-seed or recompute derived fields)."""
        return replace(self, **changes)
