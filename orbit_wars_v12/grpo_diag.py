"""GRPO anti-collapse diagnostics + A/B experiment harness (the proof the reward will not collapse).

The collapsing run (it426 -> it526, ALGO=grpo) was a self-play passivity death-spiral. The ORIGINAL
HYPOTHESIS (which motivated this harness) blamed the legacy per-group-std baseline: as the snapshot
pool gets strong the learner faces increasingly LOPSIDED losing groups, where the per-group-std (and
analytic [A]) advantage of a rare win is ``sqrt((1-p)/p)`` -- UNBOUNDED as ``p -> 0``, so the policy
chases those few (often lucky) wins and launch-rate ratchets down. The GPU ablation below REJECTED
that hypothesis: legacy std-norm is in fact the stable KEEPER, and the estimator "fixes" are what
collapse. This module computes two things at SMOKE scale, without a GPU:

  * :func:`advantage_bound_probe` -- DETERMINISTIC: across win-counts ``k=0..G`` it computes the
    worst-case advantage magnitude under each estimator. The legacy ``std``/``analytic`` magnitude
    grows without bound as the group becomes one-sided; the [C1] RANK advantage is bounded by
    ``sqrt(3)`` for every group. This is the core "cannot blow up" proof, run on the SAME
    :func:`orbit_wars_v12.rollout.grpo_flat_advantage` the trainer uses.
  * :func:`run_anticollapse_compare` -- a short A/B ``train()`` (LEGACY vs the :data:`ANTICOLLAPSE`
    bundle) returning the advantage-health / launch-rate / return curves for :func:`plot_anticollapse`.

EMPIRICAL RESULT (RTX 3070 Ti ablation, ``scripts/_gpu_ab*.py``, shared init; full write-up in
``docs/v12_grpo.tex`` Section "Empirical result"): the bound holds -- rank advantage IS bounded by sqrt3 --
but **bounded != good**. EVERY estimator in :data:`ANTICOLLAPSE`/:data:`DRGRPO_BUNDLE`, plus sibling[B1] and
phi-value[C], drove the agent into the PASSIVITY basin (launch-rate -> ~0, Elo DOWN), while LEGACY per-group
std-norm stayed stable/improving (Elo 1497->1576). So the GRPO collapse is caused by the ESTIMATOR CHANGES,
not by the legacy baseline -- and NOT by the reward decay (an earlier "lose-slower decay" hypothesis did not
hold up). **These bundles are a DIAGNOSTIC/RESEARCH harness, NOT a recommended config:** train on legacy
std-norm (every estimator flag off); re-ablate any estimator on the 3070 Ti before trusting it.
"""
import torch

from .config import Config
from .rollout import grpo_flat_advantage
from .train import train

# Two CANDIDATE anti-collapse bundles -- both EMPIRICALLY FAILED the GPU ablation (caused passivity; see the
# module docstring). Kept for the A/B harness / research, NOT as a recommended config. They are ALTERNATIVES
# (rank short-circuits the center-only path):
#   ANTICOLLAPSE [C1] -- within-group RANK advantage: bounded in [-sqrt3,sqrt3], reward-scale-invariant. The
#                        bound held (the proof) but it discards win-MAGNITUDE -> stops rewarding aggression -> passivity.
#   DRGRPO_BUNDLE [B]  -- Dr.GRPO center-only + RLOO + pinned RMS. Also went passive in the ablation.
# [D3] clip-higher (in both) was a passivity ACCELERANT. The rest of Tier 1-2 (A1/D1/C2/B1/B2/E1) layer on via flags.
ANTICOLLAPSE = dict(
    GRPO_RANK_ADV=True,       # [C1] bounded, reward-scale-invariant within-group rank advantage -- the core lever
    CLIP_HI=0.28,             # [Tier0/D3 DAPO] asymmetric upper clip -> lets good rare launches through (anti-passivity)
)
DRGRPO_BUNDLE = dict(
    GRPO_STD_NORM=False,      # [B] center-only (Dr.GRPO): drop the per-group 1/sigma difficulty-inversion
    GRPO_LOO=True,            # [B] leave-one-out (RLOO) group-mean baseline -- unbiased
    ADV_TARGET_RMS=1.0,       # [Tier0/D4] pin the advantage RMS so league difficulty can't drift the effective LR
    CLIP_HI=0.28,             # [Tier0/D3 DAPO] anti-passivity asymmetric clip
)


def _scenario_returns(G, k, scale=10.0, jitter=0.01, seed=0):
    """One group of G rollouts with k wins / (G-k) losses: outcome in {-1,+1} and a return
    ``Re = scale*outcome + tiny jitter`` (the jitter = game-length-decay spread, so groups are never
    exactly degenerate). Returns (Re, outcome) on CPU, shaped (G,) == (nW=1 x G)."""
    g = torch.Generator().manual_seed(seed)
    outcome = torch.cat([torch.ones(k), -torch.ones(G - k)])
    Re = scale * outcome + jitter * torch.randn(G, generator=g)
    return Re, outcome


def advantage_bound_probe(cfg=None, G=16, scale=10.0, verbose=True):
    """Deterministic proof that the [C1] rank advantage is BOUNDED while the legacy std/analytic
    advantage is UNBOUNDED in matchup lopsidedness. For every win-count k in 0..G, build a synthetic
    group and run :func:`grpo_flat_advantage` under each estimator; report the worst-case |advantage|.

    Returns ``{estimator: {"absmax": [...per k...], "worst": float}}`` and asserts the key claim
    (rank stays <= ~sqrt3; legacy std blows past it on one-sided groups)."""
    cfg = cfg or Config(ALGO="grpo")
    # Only the SELF-SCALING estimators (each returns its own final magnitude). The center-only [B] path's
    # scale is set later by ADV_TARGET_RMS at the buffer level, and equals std-norm at a single group --
    # its real benefit (no difficulty inversion) appears ACROSS groups; see _difficulty_inversion below.
    variants = {
        "legacy_std":  cfg.replace(GRPO_STD_NORM=True,  GRPO_LOO=False, GRPO_RANK_ADV=False, GRPO_ANALYTIC_ADV=False),
        "analytic[A]": cfg.replace(GRPO_STD_NORM=False, GRPO_LOO=False, GRPO_RANK_ADV=False, GRPO_ANALYTIC_ADV=True),
        "rank[C1]":    cfg.replace(GRPO_STD_NORM=False, GRPO_LOO=True,  GRPO_RANK_ADV=True,  GRPO_ANALYTIC_ADV=False),
    }
    out = {name: {"absmax": []} for name in variants}
    ks = list(range(0, G + 1))
    for k in ks:
        Re, outcome = _scenario_returns(G, k, scale=scale)
        for name, vcfg in variants.items():
            Ae, _unit = grpo_flat_advantage(vcfg, Re, outcome, G, 1, n_players=2)
            out[name]["absmax"].append(float(Ae.abs().max().item()))
    for name in variants:
        out[name]["worst"] = max(out[name]["absmax"])
    out["_ks"] = ks; out["_G"] = G
    rank_worst = out["rank[C1]"]["worst"]; std_worst = out["legacy_std"]["worst"]
    bound = (3.0 ** 0.5) * 1.10            # sqrt(3) with 10% slack for ties/jitter
    if verbose:
        print("=== advantage_bound_probe (G=%d): worst-case |advantage| over win-counts k=0..%d ===" % (G, G))
        print("    (lopsided groups = few wins or few losses = exactly where the collapsing run lived)")
        for name in ("legacy_std", "analytic[A]", "rank[C1]"):
            spark = " ".join("%4.1f" % v for v in out[name]["absmax"])
            print("  %-12s worst %6.2f | k= %s" % (name, out[name]["worst"], spark))
        print("  --> rank[C1] worst %.2f <= sqrt3*1.1=%.2f  (BOUNDED for every group)   legacy_std worst %.2f "
              "(grows ~sqrt((1-p)/p) as the matchup gets one-sided -> chases lucky wins -> collapse)"
              % (rank_worst, bound, std_worst))
    assert rank_worst <= bound, "RANK advantage not bounded by ~sqrt3 (got %.3f) -- C1 miswired" % rank_worst
    assert std_worst > 1.5 * rank_worst, ("legacy std advantage did not exceed rank on lopsided groups "
                                          "(std %.2f vs rank %.2f) -- probe scenario too mild" % (std_worst, rank_worst))
    out["inversion"] = _difficulty_inversion(cfg, G=G, verbose=verbose)
    if verbose:
        print("  PASS: the new reward/advantage is provably bounded where the legacy one is not.\n")
    return out


def _final_advantage(cfg, Ae, unit_scaled):
    """Apply collect_ppo's buffer-level scaling so the probe reports the advantage the SURROGATE sees."""
    if unit_scaled and not cfg.GRPO_WHITEN:
        return Ae
    if cfg.ADV_TARGET_RMS > 0.0:
        return (Ae - Ae.mean()) * (cfg.ADV_TARGET_RMS / Ae.pow(2).mean().sqrt().clamp_min(1e-8))
    return (Ae - Ae.mean()) / (Ae.std() + 1e-8)


def _difficulty_inversion(cfg, G=16, verbose=True):
    """Multi-group demo of the Dr.GRPO point: one batch with groups of DIFFERENT difficulty (balanced +
    lopsided). Legacy per-group std normalizes EACH group to unit energy -> the easy and the (near-solved/
    near-hopeless) lopsided groups get EQUAL gradient weight (difficulty inversion). Dr.GRPO center-only +
    ADV_TARGET_RMS keeps ONE global scale -> the natural p(1-p) cross-group weighting survives; rank is
    bounded per group. Reports each group's final-advantage RMS under std vs Dr.GRPO vs rank."""
    wins = [G // 2, G // 2, max(1, G // 8), G - max(1, G // 8)]   # [balanced, balanced, very-hard, very-easy]
    nW = len(wins)
    Re = torch.cat([_scenario_returns(G, k, seed=k)[0] for k in wins])
    outcome = torch.cat([_scenario_returns(G, k, seed=k)[1] for k in wins])
    arms = {
        "legacy_std": cfg.replace(GRPO_STD_NORM=True,  GRPO_LOO=False, GRPO_RANK_ADV=False, ADV_TARGET_RMS=0.0),
        "drgrpo+rms": cfg.replace(GRPO_STD_NORM=False, GRPO_LOO=True,  GRPO_RANK_ADV=False, ADV_TARGET_RMS=1.0),
        "rank[C1]":   cfg.replace(GRPO_STD_NORM=False, GRPO_LOO=True,  GRPO_RANK_ADV=True,  ADV_TARGET_RMS=1.0),
    }
    res = {}
    for name, vcfg in arms.items():
        Ae, unit = grpo_flat_advantage(vcfg, Re, outcome, G, nW, n_players=2)
        fin = _final_advantage(vcfg, Ae, unit).view(nW, G)
        res[name] = [float(fin[w].pow(2).mean().sqrt().item()) for w in range(nW)]
    if verbose:
        print("  --- difficulty inversion (per-group advantage RMS; groups = %s wins/%d) ---" % (wins, G))
        for name in ("legacy_std", "drgrpo+rms", "rank[C1]"):
            print("      %-12s %s" % (name, "  ".join("%.2f" % v for v in res[name])))
        print("      legacy_std forces ~equal energy across easy/hard groups (inversion); drgrpo+rms keeps the")
        print("      hard/uncertain groups weighted, the near-solved ones down -- the gradient that actually helps.")
    res["_wins"] = wins
    return res


def _summarize(hist):
    """Compact end-of-run summary of an A/B arm: launch-rate trend + worst advantage + return trend."""
    ln = hist.get("lnch_per_step", []); amx = hist.get("adv_absmax", []); ret = hist.get("return", [])
    half = max(1, len(ln) // 2)
    return {
        "lnch_start": (sum(ln[:half]) / half) if ln else 0.0,
        "lnch_end": (sum(ln[half:]) / max(1, len(ln) - half)) if ln else 0.0,
        "adv_absmax_max": max(amx) if amx else 0.0,
        "ret_end": (sum(ret[-half:]) / max(1, len(ret[-half:]))) if ret else 0.0,
    }


def run_anticollapse_compare(base_overrides=None, iters=24, smoke=True, resume_from=None, log_every=4,
                             arms=None, **cfg_kw):
    """Short A/B ``train()`` comparing LEGACY GRPO (per-group std -- the collapsing path) against the
    anti-collapse bundles, from the SAME init, on SMOKE-scale params. ``arms`` maps name -> overrides
    (default: legacy / rank[C1] / drgrpo[B]). Returns ``{name: hist, ..., "summary": {...}}``. Pass
    ``resume_from=<...train_state.pt>`` to REPRODUCE the real collapse from a strong checkpoint (e.g.
    it426) on a GPU; the default fresh smoke run proves the new path stays bounded and launch-stable."""
    base = dict(ALGO="grpo", GRPO_GROUP=16, GRPO_KL_COEF=0.0, GRPO_OUTCOME_ONLY=False,
                VALUE_WARMUP_ITERS=0, TOTAL_ITERS=iters)
    base.update(base_overrides or {})
    base.update(cfg_kw)
    arms = arms or {"legacy": {}, "rank[C1]": ANTICOLLAPSE, "drgrpo[B]": DRGRPO_BUNDLE}
    res, names = {}, list(arms)
    for name in names:
        print("\n========== A/B arm: %s ==========" % name.upper())
        cfg = Config.create(smoke=smoke, **{**base, **arms[name]})
        _net, hist, _lg = train(cfg, total_iters=iters, log_every=log_every, resume_from=resume_from)
        res[name] = hist
    res["summary"] = {name: _summarize(res[name]) for name in names}
    res["_arms"] = names
    print("\n=== A/B summary (smoke-scale) ===")
    for name in names:
        s = res["summary"][name]
        print("  %-14s launch/st %.2f->%.2f | worst|adv| %6.2f | ret_end %8.1f"
              % (name, s["lnch_start"], s["lnch_end"], s["adv_absmax_max"], s["ret_end"]))
    return res


def plot_anticollapse(probe=None, compare=None, path=None):
    """Render the proof: (1) the bound probe (worst |adv| vs win-count k) and (2) the A/B health curves
    (advantage-absmax, launch-rate, return) for legacy vs anti-collapse. Saves to ``path`` if given."""
    import matplotlib.pyplot as plt          # backend is the notebook's (headless --check sets MPLBACKEND=Agg)
    ncol = (1 if probe else 0) + (3 if compare else 0)
    if ncol == 0:
        return None
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.6))
    axes = [axes] if ncol == 1 else list(axes)
    ax = iter(axes)
    if probe:
        a = next(ax); ks = probe["_ks"]
        for name in ("legacy_std", "analytic[A]", "rank[C1]"):
            a.plot(ks, probe[name]["absmax"], marker="o", ms=3, label=name)
        a.axhline(3.0 ** 0.5, ls="--", c="k", lw=0.8, label="sqrt3 (rank bound)")
        a.set_xlabel("# wins in group (G=%d)" % probe["_G"]); a.set_ylabel("worst |advantage|")
        a.set_title("advantage bound vs lopsidedness"); a.legend(fontsize=7)
    if compare:
        names = compare.get("_arms", [k for k in compare if k not in ("summary", "_arms")])
        colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple"]
        for key, ttl in (("adv_absmax", "advantage |max| per iter"),
                         ("lnch_per_step", "launch rate / step"), ("return", "mean return")):
            a = next(ax)
            for i, name in enumerate(names):
                a.plot(compare[name][key], c=colors[i % len(colors)], label=name)
            a.set_xlabel("iter"); a.set_title(ttl); a.legend(fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110); print("saved anti-collapse figure ->", path)
    return fig
