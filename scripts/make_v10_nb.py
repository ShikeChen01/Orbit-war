"""Generate notebooks/setup1_v10_a100.ipynb -- v9 league + the three Lux-S3 "worth stealing"
upgrades, layered as override cells (the v9.x patch idiom).

Layered on notebooks/setup1_v9_a100.ipynb (the fully-patched v9.1..v9.9 build). v10 adds, in two
override cells inserted BEFORE the BC cell so the clone + learner are built with the v10 arch:

  (1) CROSS-PLANET TRANSFORMER TRUNK as the default learner architecture (ARCH="transformer",
      h512/L5, head_dim 64). The v9 learner ran ARCH="trunk" + USE_ATTENTION=False = a deep MLP
      with ZERO cross-planet mixing (the toy-probe finding); the transformer mixes every planet
      against every other at each layer. Sized for PARAMETER PARITY with the 512x32 deep-MLP
      baseline (21.4M vs 20.6M params, 1.04x) so the comparison is architecture-vs-architecture,
      not capacity -- and 85.6 MB fp32 stays under the 100MB submission ceiling. Flip
      V10_ARCH="trunk" to run the matched 512x32 deep-MLP line + the RL features instead (and
      warm-startable). NOTE: the transformer line is a FRESH run (BC + train from scratch); v9
      trunk weights do not load into it.

  (2) ADAPTIVE ENTROPY: a per-head auto-tuned entropy coefficient (SAC-style dual gradient) so the
      policy's ACTUAL entropy tracks a TARGET that decays over the run -- replaces the fixed
      anneal_ent_coef linear schedule. GATE head on by default; WHERE head off (gate-only entropy
      is the proven v7/v9 setup; flip ENT_ADAPT_WHERE for the dual-head Lux experiment).

  (3) TEACHER-KL anti-forgetting: + beta * KL(student || teacher) in the PPO loss, teacher = a
      frozen rolling copy of the learner (seeded at the BC init, refreshed every N iters), beta
      anneals to 0. Pulls the policy back toward its best recent self -> kills the it726-style
      regression where a higher internal Elo loses H2H to its own ancestors.

Board-symmetry overfit is already handled by v9.9 rotation augmentation (kept as-is); v10 does NOT
re-do it as observation canonicalization.

    python scripts/make_v10_nb.py
"""
import json
import os
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "notebooks", "setup1_v9_a100.ipynb")
DST = os.path.join(REPO, "notebooks", "setup1_v10_a100.ipynb")


def cell_src(c):
    return "".join(c["source"])


def set_src(c, s):
    c["source"] = s.splitlines(keepends=True)


def find_cell(doc, pred, what):
    hits = [(i, c) for i, c in enumerate(doc["cells"]) if pred(cell_src(c))]
    assert len(hits) == 1, "expected exactly 1 cell for %s, got %d" % (what, len(hits))
    return hits[0]


# ----------------------------------------------------------------------------
# override cell 1/2 -- config
# ----------------------------------------------------------------------------
OVERRIDE_CONFIG = r'''# ============================================================================
# v10 OVERRIDES (1/2) -- config: transformer trunk (default) + adaptive entropy
# + teacher-KL anti-forgetting. Runs AFTER all v9.x cells and BEFORE the BC cell,
# so the BC clone AND the PPO learner are built with this architecture.
# ============================================================================

# ---- (1) trunk architecture ------------------------------------------------
# v10 default = the multi-head TRANSFORMER trunk (cross-planet attention at every
# layer) at the fp32-submittable h256/L6 geometry. The v9 learner ran ARCH="trunk"
# + USE_ATTENTION=False = a deep MLP with NO cross-planet mixing. Flip to "trunk"
# to warm-start the v9 deep-MLP line and get only the RL features (set RESUME_FROM
# in the config cell to the it726 train state for that path).
V10_ARCH = "transformer"     # "transformer" (v10 default, FRESH run) | "trunk" (v9 deep-MLP, warm-startable)
if V10_ARCH == "transformer":
    ARCH          = "transformer"
    HIDDEN        = 64 if SMOKE else 512   # h512/L5 = 21.4M params (85.6 MB fp32) = 1.04x the 512x32
    N_TX_LAYERS   = 2 if SMOKE else 5      #   deep-MLP baseline -> a FAIR, parameter-matched comparison
    N_HEADS       = 8                      # head_dim 64 at h512 (the v7-validated geometry); 64/8 OK in SMOKE
    TX_MLP_RATIO  = 4
    N_STEM_RES    = 1
    N_HEAD_RES    = 4
    USE_ATTENTION = False                  # the transformer trunk does its own attention
else:                                      # the 512x32 deep-MLP baseline + the v10 RL features (warm-startable)
    ARCH          = "trunk"
    HIDDEN        = 64 if SMOKE else 512
    N_RES_BLOCKS  = 2 if SMOKE else 32     # 512x32 = 20.6M params (82.5 MB fp32) -- the comparison baseline
    USE_ATTENTION = False
assert HIDDEN % N_HEADS == 0, "HIDDEN (%d) must be divisible by N_HEADS (%d)" % (HIDDEN, N_HEADS)

# ---- (2) adaptive entropy (auto-tuned coefficient per head, SAC-style) ------
# Replaces the FIXED anneal_ent_coef schedule: a log-coefficient per head is trained
# so the policy's ACTUAL entropy tracks a TARGET that decays over the run (Lux S3
# recipe). Anti-collapse early, exploit late, no hand-tuned coefficient.
ENT_ADAPT              = True
ENT_ADAPT_LR           = 1e-3      # Adam LR on the log-coefficients (gentle; scalar params)
ENT_TARGET_DECAY_ITERS = 400       # linear target-decay window (iters)
# GATE head (Bernoulli fire/hold; max entropy = ln 2 ~ 0.693 nats/source). Start conservative so
# the controller nudges rather than blurs the committed BC gate.
ENT_TARGET_GATE_START  = 0.30
ENT_TARGET_GATE_END    = 0.02
# WHERE head (Categorical over live dests; fire-weighted). OFF by default -- gate-only entropy is
# the proven v7/v9 setup (WHERE entropy blurs the BC-sharpened head). Flip for the dual-head Lux run.
ENT_ADAPT_WHERE        = False
ENT_TARGET_WHERE_START = 0.80
ENT_TARGET_WHERE_END   = 0.05
ENT_COEF_INIT          = ENT_COEF_START   # starting value of both auto-coefficients
ENT_COEF_MIN           = 1e-5
ENT_COEF_MAX           = 0.05             # cap aggression (keep the bonus from nuking BC)

# ---- (3) teacher-KL anti-forgetting ----------------------------------------
# + beta * KL(student || teacher) in the PPO loss; teacher = a FROZEN rolling copy of the learner
# (seeded at the BC init, refreshed every TEACHER_REFRESH_EVERY iters). beta anneals to 0.
TEACHER_KL_ENABLED     = True
TEACHER_KL_COEF        = 0.5       # starting KL weight
TEACHER_KL_END         = 0.0
TEACHER_KL_DECAY_ITERS = 400
TEACHER_REFRESH_EVERY  = 50        # iters between teacher <- learner refreshes

V10_VERBOSE = True                 # one compact [v10] diagnostic line per iteration
print("[v10] ARCH=%s HIDDEN=%d L=%s | ENT_ADAPT=%s (where=%s) | TEACHER_KL=%s"
      % (ARCH, HIDDEN, (N_TX_LAYERS if ARCH == "transformer" else N_RES_BLOCKS),
         ENT_ADAPT, ENT_ADAPT_WHERE, TEACHER_KL_ENABLED))
'''


# ----------------------------------------------------------------------------
# override cell 2/2 -- code (dist methods + teacher/adaptive-entropy ppo_update)
# ----------------------------------------------------------------------------
OVERRIDE_CODE = r'''# ============================================================================
# v10 OVERRIDES (2/2) -- code: adaptive-entropy + teacher-KL PPO update.
# Monkeypatches ppo_update (the only caller of evaluate; redefined to use _make_dist
# directly so it can reach both the student and teacher distributions) and adds
# entropy_terms / kl_to to the gated-action distributions. Runs AFTER the dist /
# net / ppo_update definitions and BEFORE the BC + run cells, so these win.
# ============================================================================
import copy as _v10_copy


# --- per-head entropy components + KL between two gated-action distributions ---
def _gcd_entropy_terms(self):
    """(gate_e, where_e): per-state SUM over owned sources of the gate (Bernoulli) and the
    fire-weighted WHERE (categorical) entropies. Each is (B,)."""
    p = self.p
    gate_e = -(p * torch.log(p.clamp_min(1e-8)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-8)))
    cat_e = -(self.lsm.exp() * self.lsm.clamp_min(-30.0)).sum(-1)        # bounded (<= ln E_live)
    where_e = p * cat_e
    return (gate_e * self.owned).sum(1), (where_e * self.owned).sum(1)


def _gcd_kl_to(self, other):
    """KL(self || other) for the factored gate*WHERE action, summed over owned sources -> (B,).
    Bernoulli gate KL + fire-weighted categorical WHERE KL (matches the log_prob factorization)."""
    ps, pt = self.p, other.p
    gate_kl = (ps * (torch.log(ps.clamp_min(1e-8)) - torch.log(pt.clamp_min(1e-8)))
               + (1.0 - ps) * (torch.log((1.0 - ps).clamp_min(1e-8)) - torch.log((1.0 - pt).clamp_min(1e-8))))
    ps_dest = self.lsm.exp()                                             # masked dests -> ~0 weight
    where_kl = (ps_dest * (self.lsm - other.lsm)).sum(-1)
    kl = gate_kl + ps * where_kl
    return (kl * self.owned).sum(1)


GatedCatDist.entropy_terms = _gcd_entropy_terms
GatedCatDist.kl_to = _gcd_kl_to


def _gad_entropy_terms(self):
    p = self.p
    gate_e = -(p * torch.log(p.clamp_min(1e-8)) + (1.0 - p) * torch.log((1.0 - p).clamp_min(1e-8)))
    where_e = p * self.dist.entropy()
    return (gate_e * self.owned).sum(1), (where_e * self.owned).sum(1)


def _gad_kl_to(self, other):
    ps, pt = self.p, other.p
    gate_kl = (ps * (torch.log(ps.clamp_min(1e-8)) - torch.log(pt.clamp_min(1e-8)))
               + (1.0 - ps) * (torch.log((1.0 - ps).clamp_min(1e-8)) - torch.log((1.0 - pt).clamp_min(1e-8))))
    where_kl = torch.distributions.kl_divergence(self.dist, other.dist)
    kl = gate_kl + ps * where_kl
    return (kl * self.owned).sum(1)


GatedAllocDist.entropy_terms = _gad_entropy_terms
GatedAllocDist.kl_to = _gad_kl_to


# --- v10 training state (iter counter, teacher net, adaptive-entropy coefs) ---
_V10 = {"iter": 0, "teacher": None, "log_cg": None, "log_cw": None, "ent_opt": None}


def _v10_anneal(start, end, it, span):
    if span <= 0:
        return end
    return start + (end - start) * min(1.0, it / span)


def _v10_init_coefs():
    lcg = torch.tensor(math.log(max(ENT_COEF_INIT, 1e-6)), device=DEVICE, requires_grad=True)
    lcw = torch.tensor(math.log(max(ENT_COEF_INIT, 1e-6)), device=DEVICE, requires_grad=True)
    _V10["log_cg"], _V10["log_cw"] = lcg, lcw
    _V10["ent_opt"] = torch.optim.Adam([lcg, lcw], lr=ENT_ADAPT_LR)


def ppo_update(net, opt, tb):
    N = tb["n"]; mb = MINIBATCHES; mbsize = N // mb
    s = {"total": 0.0, "policy": 0.0, "vf": 0.0, "entropy": 0.0, "sigma": 0.0,
         "approx_kl": 0.0, "clipfrac": 0.0, "grad_norm": 0.0,
         "ent_gate": 0.0, "ent_where": 0.0, "coef_gate": 0.0, "coef_where": 0.0, "teacher_kl": 0.0}
    if mbsize == 0:
        return s
    nsteps = 0
    dev = tb["entities"].device
    it = _V10["iter"]; _V10["iter"] += 1
    pc = globals().get('PPO_POLICY_COEF', 1.0)            # 0 during the value-warmup iters

    # entropy: adaptive targets + lazy coef init (else fall back to the fixed annealed coef)
    if ENT_ADAPT:
        if _V10["log_cg"] is None:
            _v10_init_coefs()
        tgt_g = _v10_anneal(ENT_TARGET_GATE_START, ENT_TARGET_GATE_END, it, ENT_TARGET_DECAY_ITERS)
        tgt_w = _v10_anneal(ENT_TARGET_WHERE_START, ENT_TARGET_WHERE_END, it, ENT_TARGET_DECAY_ITERS)
        coef_g = _V10["log_cg"].detach().exp().clamp(ENT_COEF_MIN, ENT_COEF_MAX)
        coef_w = _V10["log_cw"].detach().exp().clamp(ENT_COEF_MIN, ENT_COEF_MAX)
    else:
        coef_g = torch.tensor(float(CUR_ENT_COEF), device=dev); coef_w = torch.zeros((), device=dev)
        tgt_g = tgt_w = 0.0

    # teacher: seed at the (BC) init, rolling refresh; KL weight anneals to 0
    if TEACHER_KL_ENABLED:
        if _V10["teacher"] is None:
            _V10["teacher"] = _v10_copy.deepcopy(net).eval()
            for prm in _V10["teacher"].parameters():
                prm.requires_grad_(False)
        elif pc > 0.0 and TEACHER_REFRESH_EVERY > 0 and it % TEACHER_REFRESH_EVERY == 0:
            _V10["teacher"].load_state_dict(net.state_dict())
        tkl_coef = _v10_anneal(TEACHER_KL_COEF, TEACHER_KL_END, it, TEACHER_KL_DECAY_ITERS)
    else:
        tkl_coef = 0.0

    for epoch in range(UPDATE_EPOCHS):
        perm = torch.randperm(N, device=dev)
        ep_kl = []
        for b in range(mb):
            mi = perm[b * mbsize:(b + 1) * mbsize]
            ent = tb["entities"].index_select(0, mi)
            em = tb["entity_mask"].index_select(0, mi)
            am = tb["action_mask"].index_select(0, mi)
            gl = tb["globals"].index_select(0, mi)
            act_mb = tb["action"].index_select(0, mi)
            oldlp = tb["old_logp"].index_select(0, mi)
            adv = tb["advantage"].index_select(0, mi)
            ret = tb["returns"].index_select(0, mi)

            dist, value = _make_dist(net, ent, em, am, gl)
            logp = dist.log_prob(act_mb)
            ge_sum, we_sum = dist.entropy_terms()
            n_owned = am.sum(1).clamp_min(1.0)
            ent_g = (ge_sum / n_owned).mean()
            ent_w = (we_sum / n_owned).mean()

            pol = policy_surrogate(logp, oldlp, adv, CLIP)
            vtarget = net.popart.normalize(ret) if (USE_POPART and getattr(net, "popart", None) is not None) else ret
            vloss = F.mse_loss(value, vtarget)

            ent_bonus = coef_g * ent_g
            if ENT_ADAPT_WHERE:
                ent_bonus = ent_bonus + coef_w * ent_w

            if tkl_coef > 0.0 and _V10["teacher"] is not None:
                with torch.no_grad():
                    t_dist, _ = _make_dist(_V10["teacher"], ent, em, am, gl)
                tkl = (dist.kl_to(t_dist) / n_owned).mean()
            else:
                tkl = torch.zeros((), device=dev)

            loss = pc * pol + VF_COEF * vloss - pc * ent_bonus + pc * tkl_coef * tkl

            opt.zero_grad()
            loss.backward()
            if pc == 0.0:                                  # critic warmup: val-only step
                for pn, pp in net.named_parameters():
                    if not pn.startswith('val_'):
                        pp.grad = None
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            if torch.isfinite(gnorm):
                opt.step()
            if USE_POPART and getattr(net, "popart", None) is not None:
                net.popart.update(ret, net.val_out)

            # adaptive entropy: dual-gradient coefficient step (skip during value warmup)
            if ENT_ADAPT and pc > 0.0:
                lcg, lcw = _V10["log_cg"], _V10["log_cw"]
                a_loss = lcg.exp() * (ent_g.detach() - tgt_g)
                if ENT_ADAPT_WHERE:
                    a_loss = a_loss + lcw.exp() * (ent_w.detach() - tgt_w)
                _V10["ent_opt"].zero_grad(); a_loss.backward(); _V10["ent_opt"].step()
                with torch.no_grad():
                    coef_g = lcg.detach().exp().clamp(ENT_COEF_MIN, ENT_COEF_MAX)
                    coef_w = lcw.detach().exp().clamp(ENT_COEF_MIN, ENT_COEF_MAX)

            with torch.no_grad():
                lrt = (logp - oldlp); lrc = lrt.clamp(-LOGRATIO_CLAMP, LOGRATIO_CLAMP)
                ratio = torch.exp(lrc)
                mb_kl = ((ratio - 1.0) - lrc).mean().item()
                s["total"] += loss.item(); s["policy"] += pol.item(); s["vf"] += vloss.item()
                s["entropy"] += ent_g.item(); s["sigma"] += ent_g.item()
                s["ent_gate"] += ent_g.item(); s["ent_where"] += ent_w.item()
                s["coef_gate"] += float(coef_g); s["coef_where"] += float(coef_w)
                s["teacher_kl"] += float(tkl)
                s["approx_kl"] += mb_kl
                s["clipfrac"] += (torch.abs(ratio - 1.0) > CLIP).to(DTYPE).mean().item()
                s["grad_norm"] += float(gnorm)
                ep_kl.append(mb_kl)
            nsteps += 1
            if KL_STOP_MINIBATCH and mb_kl > KL_HARD_MULT * KL_TARGET:
                break
        if KL_STOP_MINIBATCH and ep_kl and ep_kl[-1] > KL_HARD_MULT * KL_TARGET:
            break
        if KL_EARLYSTOP and ep_kl and (sum(ep_kl) / len(ep_kl)) > KL_TARGET:
            break
    if nsteps:
        for k in s:
            s[k] /= nsteps
    if V10_VERBOSE:
        print("   [v10] Hg %.3f/%.3f cg %.4f | Hw %.3f/%.3f cw %.4f | tkl %.4f x%.3f"
              % (s["ent_gate"], tgt_g, s["coef_gate"], s["ent_where"], tgt_w, s["coef_where"],
                 s["teacher_kl"], tkl_coef), flush=True)
    return s
# ============================ end v10 overrides =============================
'''


def main():
    doc = json.load(open(SRC, encoding="utf-8"))
    whole = "".join(cell_src(c) for c in doc["cells"])
    assert "v9.9 OVERRIDES" in whole, "source must be the fully-patched v9 nb (v9.9 missing)"
    assert "v10 OVERRIDES" not in whole, "v10 already applied to the source nb"

    # header markdown -> v10 title + summary
    _, hdr = find_cell(doc, lambda s: s.startswith("# Orbit Wars"), "header markdown")
    h = cell_src(hdr).replace(
        "# Orbit Wars -- v9 A100 run (v8 league + per-seat ONE-HOT ownership)",
        "# Orbit Wars -- v10 A100 run (transformer trunk + adaptive entropy + teacher-KL)", 1)
    h += """

---
**v10 = v9 + three Lux-S3 \"worth stealing\" upgrades** (two override cells before the BC cell):
1. **Cross-planet transformer trunk** as the default learner arch (`ARCH=\"transformer\"`, h512/L5,
   head_dim 64). The v9 learner was a deep MLP with no cross-planet mixing; this attends every
   planet to every other at each layer. Sized for **parameter parity** with the 512x32 deep-MLP
   baseline (21.4M vs 20.6M, 1.04x; 85.6 MB fp32, under the submission ceiling) so the comparison
   is architecture-vs-architecture. This is a **fresh run** (BC + train from scratch) -- set
   `V10_ARCH=\"trunk\"` to run the matched 512x32 deep-MLP line + RL features instead.
2. **Adaptive entropy** -- per-head auto-tuned entropy coefficient (SAC-style) tracking a decaying
   target, replacing the fixed `anneal_ent_coef` schedule. Gate head on; WHERE head off by default.
3. **Teacher-KL anti-forgetting** -- `+ beta * KL(student||teacher)`, teacher = a frozen rolling
   copy of the learner; beta anneals to 0. Targets the it726-style \"higher Elo, loses to ancestor\"
   regression.

Board-symmetry overfit stays handled by **v9.9 rotation augmentation** (unchanged). Watch the
per-iter `[v10]` line: `Hg`/`Hw` = measured gate/WHERE entropy vs their targets, `cg`/`cw` = the
auto-coefficients, `tkl` = teacher KL x its annealed weight.
"""
    set_src(hdr, h)

    # plot title cosmetic
    _, pl = find_cell(doc, lambda s: "Training-health curves" in s, "plot cell")
    set_src(pl, cell_src(pl).replace("Orbit Wars v9 -- training health",
                                     "Orbit Wars v10 -- training health"))

    # insert the two override cells immediately BEFORE the BC cell
    bc_i, _ = find_cell(doc, lambda s: s.startswith("# ## 13b. BC warm-start"), "BC cell")
    for off, src in enumerate((OVERRIDE_CONFIG, OVERRIDE_CODE)):
        compile(src, "v10_override_%d" % off, "exec")           # syntax-validate before writing
        cell = {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": src.splitlines(keepends=True), "id": uuid.uuid4().hex[:8]}
        doc["cells"].insert(bc_i + off, cell)

    json.dump(doc, open(DST, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.load(open(DST, encoding="utf-8"))                      # round-trip validate
    print("wrote", DST, "(%d cells; v10 overrides inserted at %d/%d)"
          % (len(doc["cells"]), bc_i, bc_i + 1))


if __name__ == "__main__":
    main()
