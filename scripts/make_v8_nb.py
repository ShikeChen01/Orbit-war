"""Generate notebooks/setup1_v8_a100.ipynb -- the mixed 2p/4p league run (v8).

Builds on notebooks/setup1_v7_a100.ipynb (BC warm-start -> guardrailed PPO, categorical WHERE)
and rewrites the env/rollout/league/train cells for the v8 program:

  - N-player GpuEnv: official 4p seating (owners 0..3 on the symmetric home group) and the
    official top-vs-second combat resolve; 2p stays BIT-EXACT (scripts/test_v8_parity.py).
  - mixed 2p/4p training: each iteration is a 2p match or a 4p FFA; FFA seats 1-3 are
    PFSP-sampled league members (scripted anchors / snapshots / invited members).
  - dual correlated Elo (2p + 4p) per member + learner; cross-coupled updates; the rating gap
    drives the match-type mix, error-diffused and bounded to the 1:1 .. 1:6 (2p:4p) envelope.
  - invited league members: top-4 *.pt from the first existing INVITE_DIRS folder, Elo-anchored
    (filename `_elo<NNNN>` > ckpt meta > default); sampling boost while unmastered, then the
    matchup rate drops to a small floor.
  - best-ckpt = mixed gauntlet (2p + 4p) -- the it1041 lesson (2p-dominant, 4p-losing ckpts).

    python scripts/make_v8_nb.py
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "notebooks", "legacy", "setup1_v7_a100.ipynb")
DST = os.path.join(REPO, "notebooks", "setup1_v8_a100.ipynb")

doc = json.load(open(SRC, encoding="utf-8"))


def cell_src(c):
    return "".join(c["source"])


def set_src(c, s):
    c["source"] = s.splitlines(keepends=True)


def find_cell(pred, what):
    hits = [c for c in doc["cells"] if pred(cell_src(c))]
    assert len(hits) == 1, "expected exactly 1 cell for %s, got %d" % (what, len(hits))
    return hits[0]


def patch(c, old, new, what, count=1):
    s = cell_src(c)
    n = s.count(old)
    assert n == count, "%s: expected %d occurrence(s) of %r, found %d" % (what, count, old[:60], n)
    set_src(c, s.replace(old, new))


# ============================================================================
# 0. header markdown
# ============================================================================
HEADER = r"""# Orbit Wars -- v8 A100 run (mixed 2p/4p league, dual Elo, invited members)

**v8 = v7 (BC warm-start -> guardrailed PPO, categorical WHERE) + the 4-player program.**
The it1041 post-mortem: a checkpoint can DOMINATE heads-up (1.000 gauntlet) and still LOSE
4-player FFAs (29% top-rate vs it101's 58%) -- and the Kaggle leaderboard mixes both formats.
v8 trains both formats inside one league:

- **Mixed 2p/4p training.** Every iteration is a 2-player match OR a 4-player FFA in the same
  GpuEnv (N-player combat = the official top-vs-second resolve; official 4p seating: owners 0..3
  on one symmetric home group). The learner is always player 0; FFA seats 1-3 are PFSP-sampled
  league members. Neural seats act per step (no-grad); scripted anchors run fully on-GPU.
  The 2p path is bit-exact to v7 (`scripts/test_v8_parity.py`).
- **Dual correlated Elo.** The learner and every member carry a 2p rating AND a 4p rating.
  A 2p result updates elo2 (and drags elo4 by `ELO_COUPLING`); a 4p FFA becomes 3 pairwise
  updates at `ELO_K4/3` each (coupled back into elo2). Scripted + invited members are ANCHORED
  in both formats.
- **Gap-driven match mix, bounded 1:1 .. 1:6.** `share_4p = clip(S4_BASE + S4_GAIN*(elo2-elo4)/400,
  S4_MIN, S4_MAX)`. When the two ratings split apart, the lagging format gets more matches; the
  2p:4p ratio may tilt toward 4p but never beyond 1:6, and never below 1:1. The scheduler is
  error-diffused, so the REALIZED ratio equals the target (not just in expectation).
- **Invited league members.** The first existing folder in `INVITE_DIRS` is scanned for `*.pt`
  checkpoints (pre-trained smaller models; dims come from each ckpt's embedded config, else
  `INVITE_CFG` -- all invited members share one dimension). The TOP-4 by anchored Elo join as
  high-level bots; the anchor comes from the filename suffix `_elo<NNNN>` (e.g. `it1_elo1100.pt`),
  else ckpt meta `elo`, else `INVITE_DEFAULT_ELO`, and is NEVER updated. They sample at
  `INVITE_MATCH_BOOST`x while unmastered; once the learner's per-format score EMA passes
  `INVITE_MASTER_WR`, their matchup rate DROPS to `INVITE_MASTERED_W`x (anti-forgetting floor).
  High-level sparring -- especially 4p -- without spending compute training those models.
- **Weak-point matchmaking.** Per member and per format, the league compares the learner's ACTUAL
  score EMA against the Elo-EXPECTED score. When reality runs below theory (Elo says 0.90, the
  learner only scores 0.70), that member's matchup rate is boosted by
  `1 + WEAK_BOOST_GAIN*(expected - actual)` capped at `WEAK_BOOST_MAX` (x3 at the example gap).
  It ONLY targets weakness: overperforming opponents are never down-weighted, and the detector
  stays off until `WEAK_BOOST_MIN_GAMES` matches / outside the EMA-noise deadband. Active boosts
  show on the leaderboard as `weak2/weak4 x<mult>`.
- **Mixed best-ckpt gauntlet.** best = `(1-GAUNTLET_4P_W)`*2p-gauntlet + `GAUNTLET_4P_W`*4p-gauntlet
  (FFA vs the starter/greedy/intermediate trio), so 2p-only specialists no longer win the
  checkpoint race.

**Colab checklist** (unchanged from v7)
1. Runtime -> A100 GPU. Optionally `import os; os.environ['OW_CKPT_DIR']='/content/drive/MyDrive/ow_v8'`
   BEFORE the config cell (and `os.environ['OW_LEAGUE_DIR']=...` to point at the invited-members
   folder). Resume by setting `RESUME_FROM = TRAIN_STATE_PATH` and re-running the run cell.
2. Watch the first ~15 heartbeats: it1-5 must show `kl 0.000` (critic-only warmup), then KL in the
   0.02-0.05 band. The heartbeat now shows the format (`2p`/`4p`), both Elos and the 4p share.
3. The `[best]` line prints the MIXED gauntlet (2p / 4p breakdown alongside) -- that is the number
   that matters. Healthy run: both gauntlets rising, launch-rate roughly flat near its BC level,
   no `[TRIPWIRE]` lines, and `s4` drifting toward whichever format lags.
"""
hdr = find_cell(lambda s: s.startswith("# Orbit Wars"), "header markdown")
set_src(hdr, HEADER)

# ============================================================================
# 1. config cell: append the v8 block
# ============================================================================
V8_CONFIG = r"""

# ======================= v8 additions: mixed 2p/4p + dual Elo + invited league =======================
# ---- 4-player FFA training (player 0 = learner; seats 1..3 PFSP-sampled from the league) ----
FOURP_ENABLED   = True
B_4P            = B            # parallel envs per 4p iteration (learner buffers match 2p; each
                               # NEURAL opponent seat adds one no-grad forward per step)
OUTCOME_4P      = "placement"  # "placement": outcome = 2*mean(pairwise score) - 1 (placement-linear,
                               #   beat-all=+1 .. lose-all=-1; denser learning signal)
                               # "winner": official top-score-only (+1 top, -1 otherwise)
# ---- match-type scheduler: 2p vs 4p share driven by the DUAL-Elo gap ----
# share_4p = clip(S4_BASE + S4_GAIN*(elo2p - elo4p)/400, S4_MIN, S4_MAX). When the 4p rating falls
# behind the 2p rating the system schedules more 4p (and vice versa). Bounds = the 1:1 .. 1:6
# envelope: MORE 4p than 2p is allowed, but never beyond 6 4p-iters per 2p-iter and never fewer 4p
# than 2p. The train loop error-diffuses the share, so the realized ratio is exact.
S4_BASE         = 0.60         # baseline share of 4p iterations (0.60 = 2p:4p of 2:3)
S4_MIN          = 0.50         # lower bound  = 1:1 (2p:4p)
S4_MAX          = 6.0 / 7.0    # upper bound  = 1:6 (2p:4p)
S4_GAIN         = 1.0          # share shift per 400 Elo of (elo2p - elo4p) gap
# ---- dual correlated Elo (2p Elo + 4p Elo) ----
ELO_K4          = 32.0         # 4p K-factor; each of the 3 pairwise updates uses ELO_K4/3
ELO_COUPLING    = 0.25         # cross-format fraction: a 2p delta also moves the 4p rating by this
                               # much (and vice versa) -- the two ratings stay correlated
# ---- invited league members (pre-trained smaller models, treated as anchored high-level bots) ----
# The FIRST existing directory below is scanned for *.pt checkpoints and the TOP-INVITE_MAX by
# anchored Elo are invited. The anchor comes from the filename suffix `_elo<NNNN>` (e.g.
# it1_elo1100.pt), else the ckpt meta "elo", else INVITE_DEFAULT_ELO. Invited ratings are MANUALLY
# ANCHORED: they never drift. NOTE: CKPT_DIR/league_agents is the AUTO-SNAPSHOT dump of this run --
# deliberately NOT searched (use CKPT_DIR/league_invited or OW_LEAGUE_DIR for curated members).
INVITE_DIRS     = [os.environ.get("OW_LEAGUE_DIR", ""),
                   "/content/drive/MyDrive/league_agents",
                   os.path.join(CKPT_DIR, "league_invited"),
                   "league_agents",
                   os.path.join("notebooks", "league_agents")]
INVITE_MAX         = 4         # invite at most the top-4 members from the folder
INVITE_DEFAULT_ELO = 750.0     # anchored Elo when neither filename nor ckpt meta carries one
                               # (= the submitted it1041 weight's leaderboard level)
INVITE_ELO4_OFFSET = 0.0       # anchored 4p Elo = anchored 2p Elo + this offset
INVITE_MATCH_BOOST = 2.0       # x sampling weight while UNMASTERED (high-level sparring focus)
INVITE_MASTER_WR   = 0.75      # mastered once the learner's per-format score EMA >= this ...
INVITE_MASTERED_W  = 0.2       # ... then the matchup rate DROPS to this x (>0: anti-forgetting)
# fallback dims when an invited ckpt has no embedded config (all invited members share one dim)
INVITE_CFG = {"HIDDEN": 256, "N_TX_LAYERS": 4, "N_HEADS": 8, "TX_MLP_RATIO": 4,
              "N_STEM_RES": 1, "N_HEAD_RES": 1, "N_RES_BLOCKS": 6, "VALUE_RES_BLOCKS": 2,
              "ARCH": "transformer", "USE_GLU": True, "USE_ATTENTION": True, "D_G": 32}
# ---- weak-point matchmaking (per format) ----
# If the learner's ACTUAL score EMA vs a member runs below the Elo-EXPECTED score, that member is
# a weak point -> boost its matchup rate by 1 + WEAK_BOOST_GAIN*(expected - actual), capped at
# WEAK_BOOST_MAX. Example: Elo says 0.90 but reality is 0.70 -> gap 0.20 -> x3. Never fires the
# other way (overperforming opponents are NOT down-weighted) and never below the deadband.
WEAK_BOOST_ENABLED   = True
WEAK_BOOST_MIN_GAMES = 5       # per-format matches vs a member before the detector can fire
WEAK_BOOST_MIN_GAP   = 0.05    # deadband: ignore (expected - actual) gaps below this (EMA noise)
WEAK_BOOST_GAIN      = 10.0    # boost = 1 + gain * gap  (0.90 expected vs 0.70 actual -> x3)
WEAK_BOOST_MAX       = 3.0     # cap on the multiplier
# ---- mixed best-ckpt gauntlet ----
GAUNTLET_4P_W   = 0.5          # best score = (1-w)*2p gauntlet + w*4p gauntlet (it1041 lesson:
                               # 2p-dominant ckpts can LOSE 4p; the leaderboard mixes both)
"""
cfg = find_cell(lambda s: "SMOKE = " in s and "BC_ENABLED" in s, "config cell")
set_src(cfg, cell_src(cfg).rstrip("\n") + "\n" + V8_CONFIG.lstrip("\n"))

# ============================================================================
# 2. world gen: remember the symmetric home group (4p re-seating happens in reset)
# ============================================================================
wg = find_cell(lambda s: s.startswith("MIN_PLANET_GROUPS"), "world-gen cell")
patch(wg, """    num_groups = len(planets) // 4
    if num_groups > 0:
        base = rng.randint(0, num_groups - 1) * 4
        planets[base][1] = 0;      planets[base][5] = 10       # player 0 home
        planets[base + 3][1] = 1;  planets[base + 3][5] = 10   # player 1 home
    w = {"planets": planets, "angular_velocity": angular_velocity}""",
      """    num_groups = len(planets) // 4
    base = -1
    if num_groups > 0:
        base = rng.randint(0, num_groups - 1) * 4
        planets[base][1] = 0;      planets[base][5] = 10       # player 0 home
        planets[base + 3][1] = 1;  planets[base + 3][5] = 10   # player 1 home
    # v8: remember the home group; env.reset(n_players=4) re-seats owners 0..3 on base..base+3
    # exactly like the official engine's 4-player branch (same group => fair under 4-fold symmetry).
    w = {"planets": planets, "angular_velocity": angular_velocity, "home_base": base}""",
      "world-gen home_base")

# ============================================================================
# 3. GpuEnv: n_players-aware reset (official 4p seating)
# ============================================================================
ge = find_cell(lambda s: s.startswith("class GpuEnv"), "GpuEnv cell")
patch(ge, "        self.B = 0\n",
      "        self.B = 0\n        self.n_players = 2\n", "GpuEnv init n_players")
patch(ge, "    def reset(self, worlds):\n",
      "    def reset(self, worlds, n_players=2):\n        self.n_players = int(n_players)\n",
      "GpuEnv reset signature")
patch(ge, """            angv[b] = w["angular_velocity"]
""",
      """            angv[b] = w["angular_velocity"]
            if self.n_players >= 4:               # official 4p seating: owners 0..3, 10 ships each
                hb = int(w.get("home_base", -1))
                if 0 <= hb and hb + 3 < Ec:
                    for j in range(4):
                        po[b, hb + j] = float(j); ps[b, hb + j] = 10.0
""",
      "GpuEnv reset 4p seating")

# ============================================================================
# 4. env_encode: collapse EVERY non-ego owner to the single enemy code/aggregates
#    (v8 = the LEGACY-COMPATIBLE lineage: F_DIM stays 101, v7 weights drop in unchanged
#    and 4p states look exactly like the deployed everyone-vs-me adapter view.
#    The per-seat one-hot redesign lives in v9 -- scripts/make_v9_nb.py.)
# ============================================================================
ec = find_cell(lambda s: "def env_encode" in s, "encode cell")
patch(ec, """def env_encode(env, ego=0):
    B, Ec, Fc = env.B, env.Ec, env.Fc
    enemy = 1 - ego
    na = 2
""",
      """def env_encode(env, ego=0):
    \"\"\"v8: N-player aware. EVERY non-ego owner collapses to the single 'enemy' code and the
    enemy aggregates pool ALL opponents -- identical encodings in 2p, and exactly the deployed
    Kaggle 4p adapter view in FFA (the policy plays everyone-vs-me). v7 weights stay drop-in
    compatible (F_DIM 101); the per-seat one-hot redesign is v9.\"\"\"
    B, Ec, Fc = env.B, env.Ec, env.Fc
""",
      "encode prologue")
patch(ec, """    # ego-relative ownership code: 0 neutral, 1 ego, 2..na enemy
    m = owner - float(ego) + float(na)
    m = m - float(na) * torch.floor(m / float(na))
    own_code = torch.where(owner < 0.0, torch.zeros_like(owner), 1.0 + m)
""",
      """    # ownership code: 0 neutral, 1 ego, 2 ANY other player (2p-identical; 4p collapses)
    own_code = torch.where(owner < 0.0, torch.zeros_like(owner),
                           torch.where(owner == float(ego), torch.ones_like(owner),
                                       torch.full_like(owner, 2.0)))
""",
      "encode own_code")
patch(ec, "    en = (owner == float(enemy)) & alive\n",
      "    en = (owner >= 0.0) & (owner != float(ego)) & alive    # ALL opponents pooled\n",
      "encode enemy mask")
patch(ec, "    en_fleet = (env.f_ships * (env.f_owner == float(enemy)).to(DTYPE) * falive.to(DTYPE)).sum(1)\n",
      "    en_fleet = (env.f_ships * (env.f_owner != float(ego)).to(DTYPE) * falive.to(DTYPE)).sum(1)\n",
      "encode enemy fleet")

# ============================================================================
# 5. scripted opponents: parameterize the seat (pid)
# ============================================================================
oc = find_cell(lambda s: "def opponent_action" in s, "opponent cell")
patch(oc, "def opponent_action(env, opponent):\n    '''Scripted player-1 launches",
      "def opponent_action(env, opponent, pid=1):\n    '''Scripted launches for seat `pid` (v8: any seat in a 2p/4p game)",
      "opponent signature")
s = cell_src(oc)
n1 = s.count("env.p_owner == 1.0"); n2 = s.count("env.p_owner != 1.0")
n3 = s.count("env.f_owner == 1.0"); n4 = s.count("(env.f_owner == 0.0) & f_ok")
assert (n1, n2, n3, n4) == (3, 4, 1, 1), "opponent owner-literal counts changed: %r" % ((n1, n2, n3, n4),)
s = s.replace("env.p_owner == 1.0", "env.p_owner == float(pid)")
s = s.replace("env.p_owner != 1.0", "env.p_owner != float(pid)")
s = s.replace("env.f_owner == 1.0", "env.f_owner == float(pid)")
s = s.replace("(env.f_owner == 0.0) & f_ok", "(env.f_owner != float(pid)) & f_ok")
set_src(oc, s)

# ============================================================================
# 6. env_step: N-player rewrite (official top-vs-second combat; v7 2p calls unchanged)
# ============================================================================
NEW_ENV_STEP = r'''def env_step(env, ego_action, opponent=None, opp_action=None, step_idx=None, seats=None):
    """One tick, N-player (v8). ego_action (B,Ec,Ec+1)=[alloc | fire] for player 0.
    seats = list of opponent seat dicts, one per pid 1..n_players-1:
        {"pid": p, "script": code}    scripted bot (opponent_action codes), or
        {"pid": p, "action": tensor}  neural (B,Ec,Ec+1) gated-alloc action.
    v7 back-compat: seats=None -> a single pid-1 seat built from (opponent, opp_action)."""
    B, Ec, Fc = env.B, env.Ec, env.Fc
    N = int(getattr(env, "n_players", 2))
    if seats is None:
        seats = [{"pid": 1, "action": opp_action}] if opp_action is not None else [{"pid": 1, "script": opponent}]
    _sc = int(env.step_ct[0].item()) if step_idx is None else int(step_idx)   # avoid a per-step CUDA sync
    if COMETS_ENABLED and _sc in COMET_SPAWN_STEPS:   # hidden-schedule comet spawn
        if COMET_OFFICIAL and getattr(env, 'c_paths', None) is not None:
            spawn_comets_official(env, COMET_SPAWN_STEPS.index(_sc))
        else:
            spawn_comets(env)
    vmax = env.vmax
    dev = env.dev
    ego = 0

    ego_owned0 = (env.p_owner == float(ego)) & (env.p_alive > 0.5)
    comet0 = env.p_is_comet > 0.5   # comet status at step start (mask comet expiry out of capture/lost)

    # --- decode ego action (gated allocation: fire -> full garrison, routed) ---
    legal = (env.p_owner == float(ego)) & (env.p_alive > 0.5) & (env.p_ships > 0.0)
    e_ang, e_shp, e_can, valid, invalid, launches = _decode_target(env, ego_action, legal)

    # --- opponent seat launches (scripted: one launch/planet; neural: full alloc decode) ---
    slot_idx = torch.arange(Ec, dtype=torch.long, device=dev).unsqueeze(0).expand(B, Ec)
    src_mat = torch.arange(Ec, dtype=torch.long, device=dev).view(1, Ec, 1).expand(B, Ec, Ec).reshape(B, Ec * Ec)
    flatM = lambda x: x.reshape(B, Ec * Ec)                                   # (B,Ec,Ec) -> (B,Ec*Ec)
    own_blk = [torch.full((B, Ec * Ec), float(ego), dtype=DTYPE, device=dev)]
    slot_blk = [src_mat]
    ang_blk = [flatM(e_ang)]; shp_blk = [flatM(e_shp)]; can_blk = [flatM(e_can)]
    ded = e_shp.sum(2)                                                        # (B,Ec) ships deducted/source
    for st in seats:
        pid = int(st["pid"])
        if st.get("action") is not None:          # neural seat: decode exactly like the ego
            legal_p = (env.p_owner == float(pid)) & (env.p_alive > 0.5) & (env.p_ships > 0.0)
            o_ang, o_shp, o_can, _, _, _ = _decode_target(env, st["action"], legal_p)
            ded = ded + o_shp.sum(2)
            own_blk.append(torch.full((B, Ec * Ec), float(pid), dtype=DTYPE, device=dev))
            slot_blk.append(src_mat)
            ang_blk.append(flatM(o_ang)); shp_blk.append(flatM(o_shp)); can_blk.append(flatM(o_can))
        else:                                     # scripted seat: one launch per owned planet
            o_ang, o_shp, o_can = opponent_action(env, int(st["script"]), pid)
            ded = ded + o_shp
            own_blk.append(torch.full((B, Ec), float(pid), dtype=DTYPE, device=dev))
            slot_blk.append(slot_idx)
            ang_blk.append(o_ang); shp_blk.append(o_shp); can_blk.append(o_can)

    # --- deduct ships from origin planets ---
    env.p_ships = env.p_ships - ded

    # --- place fleets (ego first, then seats in pid order) ---
    owner = torch.cat(own_blk, 1)
    from_slot = torch.cat(slot_blk, 1)
    angle = torch.cat(ang_blk, 1)
    ships = torch.cat(shp_blk, 1)
    commit = torch.cat(can_blk, 1)
    Ltot = owner.shape[1]
    seq = (env.step_ct * float(Ltot + 1)).unsqueeze(1) + \
          torch.arange(Ltot, dtype=DTYPE, device=dev).unsqueeze(0)
    launch_fleets(env, owner, from_slot, angle, ships, commit, seq)

    # --- production ---
    env.p_ships = env.p_ships + env.p_prod * (env.p_owner != -1.0).to(DTYPE) * (env.p_alive > 0.5).to(DTYPE)

    # --- planet new positions (orbit) ---
    stepf = env.step_ct
    dxc = env.p_init_x - CENTER; dyc = env.p_init_y - CENTER
    r = torch.sqrt(dxc * dxc + dyc * dyc)
    ia = torch.atan2(dyc, dxc)
    ca = ia + env.ang_vel.unsqueeze(1) * stepf.unsqueeze(1)
    rot = env.p_rotates > 0.5
    cmt = env.p_is_comet > 0.5
    nx = torch.where(rot, CENTER + r * torch.cos(ca), torch.where(cmt, env.p_x + env.p_comet_vx, env.p_x))
    ny = torch.where(rot, CENTER + r * torch.sin(ca), torch.where(cmt, env.p_y + env.p_comet_vy, env.p_y))
    old_px, old_py = env.p_x, env.p_y
    _comet_expired = None
    if COMETS_ENABLED and COMET_OFFICIAL and getattr(env, 'c_paths', None) is not None:
        # waypoint playback: comet of event e at tick u sweeps path[u-s] -> path[u-s+1]
        _comet_expired = torch.zeros(B, Ec, dtype=torch.bool, device=dev)
        _ar = torch.arange(B, device=dev)
        for _e, _s_e in enumerate(COMET_SPAWN_STEPS):
            if not (_s_e <= _sc <= _s_e + COMET_MAX_LEN):   # at most one event is ever live
                continue
            k = int(_sc - _s_e + 1)                          # waypoint index this tick (same for all envs)
            for _m in range(4):
                sl = env.c_slot[:, _e, _m]                   # (B,) planet slot, -1 = none
                live = sl >= 0
                if not bool(live.any()):
                    continue
                slc = sl.clamp_min(0).unsqueeze(1)
                adv = live & (k < env.c_len[:, _e])          # still has waypoints -> advance
                exp = live & (k >= env.c_len[:, _e])         # path ended -> expire after combat
                wp = env.c_paths[_ar, _e, _m, min(k, COMET_MAX_LEN - 1)]   # (B,2)
                nx.scatter_(1, slc, torch.where(adv.unsqueeze(1), wp[:, 0:1], nx.gather(1, slc)))
                ny.scatter_(1, slc, torch.where(adv.unsqueeze(1), wp[:, 1:2], ny.gather(1, slc)))
                _comet_expired.scatter_(1, slc, exp.unsqueeze(1) | _comet_expired.gather(1, slc))
                env.c_slot[:, _e, _m] = torch.where(exp, torch.full_like(sl, -1), sl)

    # --- fleet movement + swept collision against planet paths ---
    falive = env.f_alive > 0.5
    speed = fleet_speed_t(env.f_ships, vmax)
    fox, foy = env.f_x, env.f_y
    fnx = fox + torch.cos(env.f_angle) * speed
    fny = foy + torch.sin(env.f_angle) * speed
    Ax = fox.unsqueeze(2); Ay = foy.unsqueeze(2)
    Bx = fnx.unsqueeze(2); By = fny.unsqueeze(2)
    P0x = old_px.unsqueeze(1); P0y = old_py.unsqueeze(1)
    P1x = nx.unsqueeze(1); P1y = ny.unsqueeze(1)
    rad = env.p_radius.unsqueeze(1)
    palive = (env.p_alive.unsqueeze(1) > 0.5)
    d0x = Ax - P0x; d0y = Ay - P0y
    dvx = (Bx - Ax) - (P1x - P0x); dvy = (By - Ay) - (P1y - P0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - rad * rad
    disc = b * b - 4.0 * a * c
    sq = torch.sqrt(disc.clamp_min(0.0))
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    hit_quad = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    hit_lin = (a < 1e-12) & (c <= 0.0)
    hit = torch.where(a < 1e-12, hit_lin, hit_quad)
    hit = hit & palive & falive.unsqueeze(2)
    slotf = torch.arange(Ec, dtype=DTYPE, device=dev).view(1, 1, Ec)
    order = torch.where(hit, slotf, torch.full_like(slotf, float(Ec)))
    fh_v, tgt_slot = order.min(2)
    has_hit = fh_v < float(Ec)

    # OOB / sun removal (point-to-segment to sun center)
    oob = (fnx < 0.0) | (fnx > BOARD_SIZE) | (fny < 0.0) | (fny > BOARD_SIZE)
    vx, vy, wx, wy = fox, foy, fnx, fny
    l2 = (vx - wx) ** 2 + (vy - wy) ** 2
    tt = ((CENTER - vx) * (wx - vx) + (CENTER - vy) * (wy - vy)) / l2.clamp_min(1e-12)
    tt = tt.clamp(0.0, 1.0)
    prx = vx + tt * (wx - vx); pry = vy + tt * (wy - vy)
    sundist = torch.sqrt((CENTER - prx) ** 2 + (CENTER - pry) ** 2)
    sun_hit = (l2 > 0.0) & (sundist < SUN_RADIUS)
    sun_pt = torch.sqrt((CENTER - vx) ** 2 + (CENTER - vy) ** 2) < SUN_RADIUS
    sun_hit = torch.where(l2 > 0.0, sun_hit, sun_pt)

    remove_fleet = falive & (has_hit | oob | sun_hit)
    contributes = falive & has_hit

    # --- combat: arrivals per PLAYER, official top-vs-second resolve (N-player) ---
    cf = contributes.to(DTYPE)
    tslot = tgt_slot.clamp(0, Ec - 1)
    arr_p = []
    for p in range(N):
        sp = env.f_ships * cf * (env.f_owner == float(p)).to(DTYPE)
        ap = torch.zeros(B, Ec, dtype=DTYPE, device=dev)
        ap.scatter_add_(1, tslot, sp)
        arr_p.append(ap)
    arr = torch.stack(arr_p, 2)                                  # (B,Ec,N)
    top2v, top2i = arr.topk(2, dim=2)                            # N >= 2 always
    surv_ships = top2v[..., 0] - top2v[..., 1]                   # exact tie -> 0 survivors
    surv_owner = torch.where(surv_ships > 0.0, top2i[..., 0].to(DTYPE),
                             torch.full((B, Ec), -1.0, dtype=DTYPE, device=dev))
    any_arr = top2v[..., 0] > 0.0
    apply = any_arr & (surv_ships > 0.0) & (env.p_alive > 0.5)
    same = (env.p_owner == surv_owner)
    reinforce = apply & same
    attack = apply & (~same)
    env.p_ships = torch.where(reinforce, env.p_ships + surv_ships, env.p_ships)
    after = env.p_ships - surv_ships
    flips = attack & (after < 0.0)
    env.p_ships = torch.where(attack, torch.where(after < 0.0, -after, after), env.p_ships)
    env.p_owner = torch.where(flips, surv_owner, env.p_owner)

    # --- apply planet positions; clear removed fleets ---
    env.p_x = nx; env.p_y = ny
    if COMETS_ENABLED:                                   # comet expiry: official = path end; legacy = swept past the inner region
        if _comet_expired is not None:
            gone = _comet_expired & (env.p_alive > 0.5)
        else:
            cdist = torch.sqrt((env.p_x - CENTER) ** 2 + (env.p_y - CENTER) ** 2)
            gone = (env.p_is_comet > 0.5) & (env.p_alive > 0.5) & (cdist > COMET_EXPIRE_RADIUS)
        keepc = (~gone).to(DTYPE)
        env.p_alive = env.p_alive * keepc; env.p_is_comet = env.p_is_comet * keepc
        env.p_comet_vx = env.p_comet_vx * keepc; env.p_comet_vy = env.p_comet_vy * keepc
        env.p_ships = env.p_ships * keepc
        env.p_owner = torch.where(gone, torch.full_like(env.p_owner, -1.0), env.p_owner)
    keep = falive & (~remove_fleet)
    keepf = keep.to(DTYPE)
    env.f_alive = keepf
    env.f_owner = env.f_owner * keepf
    env.f_x = fnx * keepf; env.f_y = fny * keepf
    env.f_angle = env.f_angle * keepf
    env.f_ships = env.f_ships * keepf

    env.step_ct = env.step_ct + 1.0

    ego_owned1 = (env.p_owner == float(ego)) & (env.p_alive > 0.5)

    out = StepOut()
    out.invalid = invalid
    out.valid = valid
    out.launches = launches
    out.launched_ships = e_shp.sum((1, 2))   # total ships launched this step (ship-weighted launch reward)
    out.owned_launch_ships = (e_shp * ego_owned0.unsqueeze(1).to(DTYPE)).sum((1, 2))   # ships launched to ALREADY-OWNED planets
    noncomet1 = env.p_is_comet < 0.5
    out.captured = (ego_owned1 & (~ego_owned0) & noncomet1).to(DTYPE).sum(1)
    out.lost = (ego_owned0 & (~ego_owned1) & (~comet0)).to(DTYPE).sum(1)
    out.captured_prod = (env.p_prod * (ego_owned1 & (~ego_owned0) & noncomet1).to(DTYPE)).sum(1)
    out.lost_prod = (env.p_prod * (ego_owned0 & (~ego_owned1) & (~comet0)).to(DTYPE)).sum(1)
    return out
'''
sc = find_cell(lambda s: s.startswith("class StepOut"), "step cell")
s = cell_src(sc)
marker = "def env_step(env, ego_action, opponent, opp_action=None, step_idx=None):"
assert marker in s, "v7 env_step signature not found"
set_src(sc, s[:s.index(marker)] + NEW_ENV_STEP)

# ============================================================================
# 7. rollout cell: settle_n + N-player potentials + unified collect_ppo (seats)
# ============================================================================
NEW_COLLECT = r'''def settle_n(env):
    """Per-player settle: ships (B,N) = planets+fleets per player; alive (B,N) bool."""
    N = int(getattr(env, "n_players", 2))
    pa = env.p_alive > 0.5; fa = env.f_alive > 0.5
    s, alive = [], []
    for p in range(N):
        op = (env.p_owner == float(p)) & pa
        gp = (env.f_owner == float(p)) & fa
        s.append((env.p_ships * op.to(DTYPE)).sum(1) + (env.f_ships * gp.to(DTYPE)).sum(1))
        alive.append(op.any(1) | gp.any(1))
    return torch.stack(s, 1), torch.stack(alive, 1)


def settle(env):
    """v7-compat 2p view: (s0, s1, side0_alive, side1_alive). BC/eval/probe cells use this."""
    s, alive = settle_n(env)
    return s[:, 0], s[:, 1], alive[:, 0], alive[:, 1]


def _potentials(env):
    """Policy-invariant shaping potentials (Ng et al.), N-player: ship-margin and production
    share vs the STRONGEST opponent (reduces exactly to the v7 pair when n_players == 2)."""
    s, _ = settle_n(env)
    phi_ship = ship_log_t(s[:, 0]) - ship_log_t(s[:, 1:].max(1).values)
    pa = env.p_alive > 0.5
    N = int(getattr(env, "n_players", 2))
    prods = [(env.p_prod * ((env.p_owner == float(p)) & pa).to(DTYPE)).sum(1) for p in range(N)]
    pe = prods[0]
    pen = torch.stack(prods[1:], 1).max(1).values
    tot = (env.p_prod * pa.to(DTYPE)).sum(1).clamp_min(1.0)
    return phi_ship, (pe - pen) / tot


_KIND2OPP = {"noop": 2, "random": 0, "starter": 1, "medium": 3, "greedy": 4, "intermediate": 5}


def collect_ppo(env, net, world_pool, cursor, rng, seats, n_players=2, n_envs=None):
    """One PPO iteration (learner = player 0) vs `seats` -- league members filling pids
    1..n_players-1. Scripted anchors run fully on-GPU; neural members act per step (no-grad).
    On-GPU buffers + GPU GAE; reward = potential shaping + outcome. n_players=2 is the v7 path.
    stats["seat_scores"][k] = the learner's mean pairwise score vs seat k (dual-Elo signal)."""
    Bn = int(n_envs or B)
    Ec, T = env.Ec, env.T
    dev = env.dev
    specs = []
    for i, m in enumerate(seats):
        pid = i + 1
        if m.get("net") is None:
            specs.append({"pid": pid, "script": _KIND2OPP[m["kind"]]})
        else:
            specs.append({"pid": pid, "net": m["net"]})

    worlds = [world_pool[(cursor + i) % len(world_pool)] for i in range(Bn)]
    cursor += Bn
    env.reset(worlds, n_players=n_players)

    # on-GPU rollout buffers (compute-bound: no host copies, no per-step D2H sync)
    ent_buf = torch.zeros(T, Bn, Ec, F_DIM, device=dev)
    em_buf = torch.zeros(T, Bn, Ec, device=dev); am_buf = torch.zeros(T, Bn, Ec, device=dev)
    gl_buf = torch.zeros(T, Bn, G_DIM, device=dev); act_buf = torch.zeros(T, Bn, Ec, Ec + 1, device=dev)
    oldlp_buf = torch.zeros(T, Bn, device=dev); valid_buf = torch.zeros(T, Bn, device=dev)
    rew_buf = torch.zeros(T, Bn, device=dev); val_buf = torch.zeros(T, Bn, device=dev); done_buf = torch.zeros(T, Bn, device=dev)

    active = torch.ones(Bn, device=dev); outcome = torch.zeros(Bn, device=dev)
    seat_sc = torch.zeros(len(specs), Bn, device=dev)   # pairwise score vs each seat, frozen at term
    inv_sum = torch.zeros(Bn, device=dev); lnch_sum = torch.zeros(Bn, device=dev)
    valid_sum = torch.zeros(Bn, device=dev); step_sum = torch.zeros(Bn, device=dev)
    launch_paid = torch.zeros(Bn, device=dev); milestone_prev = torch.zeros(Bn, device=dev)
    captureR = torch.zeros(Bn, device=dev); launchR = torch.zeros(Bn, device=dev); prodMR = torch.zeros(Bn, device=dev)

    phi_ship_prev, phi_prod_prev = _potentials(env)   # shaping potentials at s_0

    def _scores_outcome(s_all):
        """Pairwise scores vs every seat + the scalar outcome in [-1, 1]."""
        s0 = s_all[:, 0]
        scs = []
        for sp in specs:
            sk = s_all[:, sp["pid"]]
            scs.append(torch.where(s0 > sk, torch.ones_like(s0),
                                   torch.where(s0 < sk, torch.zeros_like(s0), torch.full_like(s0, 0.5))))
        if n_players == 2 or OUTCOME_4P == "winner":
            oc = torch.sign(s0 - s_all[:, 1:].max(1).values)    # official: top score only
        else:
            oc = 2.0 * torch.stack(scs, 0).mean(0) - 1.0        # placement-linear (beat-all = +1)
        return scs, oc

    last_t = 0
    for t in range(T):
        last_t = t
        ent, em, am, gl = env_encode(env, 0)
        a_t, logp, value = act(net, ent, em, am, gl, greedy=False)
        ent_buf[t] = ent; em_buf[t] = em; am_buf[t] = am; gl_buf[t] = gl
        act_buf[t] = a_t; oldlp_buf[t] = logp; valid_buf[t] = active; val_buf[t] = value

        step_seats = []
        for sp in specs:
            if "net" in sp:
                with torch.no_grad():
                    oe, om, oa, og = env_encode(env, sp["pid"])
                    o_act, _, _ = act(sp["net"], oe, om, oa, og, greedy=False)
                step_seats.append({"pid": sp["pid"], "action": o_act})
            else:
                step_seats.append({"pid": sp["pid"], "script": sp["script"]})
        out = env_step(env, a_t, seats=step_seats, step_idx=t)
        a = active

        if USE_POTENTIAL_SHAPING:
            phi_ship_now, phi_prod_now = _potentials(env)
            cap_t = a * (SHAPE_SHIP * (GAMMA * phi_ship_now - phi_ship_prev))   # ship-margin shaping
            mr_t = a * (SHAPE_PROD * (GAMMA * phi_prod_now - phi_prod_prev))    # production-share shaping
            launch_t = torch.zeros_like(cap_t)
            phi_ship_prev, phi_prod_prev = phi_ship_now, phi_prod_now
        else:
            s_all0, _ = settle_n(env)
            s0n = s_all0[:, 0]
            cap_t = a * (CAPTURE_REWARD * ((out.captured + CAPTURE_PROD_SCALE * out.captured_prod)
                                           - CAPTURE_LOSS_FRAC * (out.lost + CAPTURE_PROD_SCALE * out.lost_prod)))
            m_now = torch.where(s0n >= PROD_MILESTONE_BASE,
                                torch.floor(torch.log2(s0n.clamp_min(PROD_MILESTONE_BASE) / PROD_MILESTONE_BASE)) + 1.0,
                                torch.zeros_like(s0n))
            mr_t = a * (PROD_MILESTONE_REWARD * (m_now - milestone_prev).clamp_min(0.0))
            milestone_prev = torch.maximum(milestone_prev, m_now)
            if t < LAUNCH_WINDOW:
                _enemy_ships = out.launched_ships - out.owned_launch_ships
                _self_ships = torch.clamp(out.owned_launch_ships, max=SELF_LAUNCH_CAP)
                _disp_raw = LAUNCH_REWARD * _enemy_ships + SELF_LAUNCH_REWARD * _self_ships
                launch_t = a * torch.clamp(_disp_raw, max=LAUNCH_STEP_CAP)
            else:
                launch_t = torch.zeros_like(cap_t)
            launch_room = (LAUNCH_GAME_CAP - launch_paid).clamp_min(0.0)
            launch_t = torch.minimum(launch_t, launch_room)
            launch_paid = launch_paid + launch_t
        captureR = captureR + cap_t; prodMR = prodMR + mr_t; launchR = launchR + launch_t
        rew_buf[t] = (cap_t + mr_t + launch_t) / PPO_REWARD_SCALE

        inv_sum = inv_sum + a * out.invalid
        lnch_sum = lnch_sum + a * out.launches
        valid_sum = valid_sum + a * out.valid
        step_sum = step_sum + a

        s_all, alive_all = settle_n(env)
        step_now = env.step_ct
        n_alive = alive_all.to(DTYPE).sum(1)
        term = (step_now >= float(T - 2)) | (n_alive <= 1.0) | (~alive_all[:, 0])   # ego dead -> decided
        done_buf[t] = ((active > 0.5) & term).to(DTYPE)        # terminal on cap OR death (no spurious bootstrap)
        newly = (active > 0.5) & term
        scs, oc = _scores_outcome(s_all)
        for k in range(len(specs)):
            seat_sc[k] = torch.where(newly, scs[k], seat_sc[k])
        outcome = torch.where(newly, oc, outcome)
        active = torch.where(term, torch.zeros_like(active), active)
        # NOTE: no per-step active.sum().item() early-break -> that is a CUDA sync. Finished envs are masked.

    s_all, alive_all = settle_n(env)
    scs, oc = _scores_outcome(s_all)
    still = active > 0.5
    for k in range(len(specs)):
        seat_sc[k] = torch.where(still, scs[k], seat_sc[k])
    outcome = torch.where(still, oc, outcome)

    with torch.no_grad():
        fe, fm, fa_, fg = env_encode(env, 0)
        _, _, vT = act(net, fe, fm, fa_, fg, greedy=False)
        bootstrap = vT * active

    Tu = last_t + 1
    rew = rew_buf[:Tu].clone(); val = val_buf[:Tu]; done = done_buf[:Tu]; alive = valid_buf[:Tu]
    length = alive.sum(0)
    len_eff = (length - DECAY_START_STEP).clamp_min(0.0)
    win_val = torch.pow(torch.full_like(outcome, WIN_DECAY), len_eff) * WIN_BONUS
    loss_val = torch.pow(torch.full_like(outcome, LOSS_DECAY), len_eff) * LOSS_PENALTY
    # fractional outcomes (4p placement mode) scale the outcome magnitudes linearly;
    # in 2p outcome is exactly {-1, 0, +1} -> identical to the v7 payment
    Ot = torch.where(outcome > 0.0, outcome * win_val, outcome * loss_val)
    r_outcome_log = Ot.mean().item()
    if not USE_POTENTIAL_SHAPING:
        disp_cashout = torch.clamp(LAUNCH_WINDOW - length, min=0.0) * LAUNCH_STEP_CAP * (outcome > 0.0).to(outcome.dtype)
        disp_cashout = torch.minimum(disp_cashout, (LAUNCH_GAME_CAP - launch_paid).clamp_min(0.0))
        launchR = launchR + disp_cashout
        Ot = (Ot + disp_cashout) / PPO_REWARD_SCALE
    else:
        Ot = Ot / PPO_REWARD_SCALE
    last_idx = (length - 1.0).clamp_min(0.0).long().unsqueeze(0)
    rew.scatter_add_(0, last_idx, Ot.unsqueeze(0))

    # GAE on GPU (no .cpu(), no synchronize)
    adv = torch.zeros(Tu, Bn, device=dev)
    A = torch.zeros(Bn, device=dev); nextval = bootstrap.clone()
    for t in range(Tu - 1, -1, -1):
        al = alive[t]; notdone = 1.0 - done[t]
        delta = rew[t] + GAMMA * nextval * notdone - val[t]
        A = delta + GAMMA * GAE_LAMBDA * notdone * A
        adv[t] = A * al
        nextval = val[t]
        A = A * al
    ret_full = (adv + val).reshape(-1); adv_full = adv.reshape(-1)
    mean_return_log = (rew.sum(0).mean().item()) * PPO_REWARD_SCALE

    keep = valid_buf[:Tu].reshape(-1).nonzero().squeeze(-1)
    def sel(x): return x.index_select(0, keep)
    tb = {}
    tb["entities"] = sel(ent_buf[:Tu].reshape(Tu * Bn, Ec, F_DIM))
    tb["entity_mask"] = sel(em_buf[:Tu].reshape(Tu * Bn, Ec))
    tb["action_mask"] = sel(am_buf[:Tu].reshape(Tu * Bn, Ec))
    tb["globals"] = sel(gl_buf[:Tu].reshape(Tu * Bn, G_DIM))
    tb["action"] = sel(act_buf[:Tu].reshape(Tu * Bn, Ec, Ec + 1))
    tb["old_logp"] = sel(oldlp_buf[:Tu].reshape(Tu * Bn))
    adv_s = sel(adv_full)
    tb["advantage"] = (adv_s - adv_s.mean()) / (adv_s.std() + 1e-8)
    tb["returns"] = sel(ret_full)
    tb["n"] = keep.shape[0]
    del ent_buf, em_buf, am_buf, gl_buf, act_buf, oldlp_buf, val_buf, rew_buf, done_buf, valid_buf, adv, adv_full, ret_full

    step_total = step_sum.sum().item()
    stats = {
        "mean_return": mean_return_log,
        "mean_len": step_total / Bn,
        "win_rate": (seat_sc.min(0).values >= 0.999).to(DTYPE).mean().item(),   # beat EVERY opponent
        "score": seat_sc.mean().item(),
        "seat_scores": [float(seat_sc[k].mean().item()) for k in range(len(specs))],
        "fmt": n_players,
        "inv_per_step": (inv_sum.sum().item() / step_total) if step_total > 0 else 0.0,
        "lnch_per_step": (lnch_sum.sum().item() / step_total) if step_total > 0 else 0.0,
        "valid_per_step": (valid_sum.sum().item() / step_total) if step_total > 0 else 0.0,
        "transitions": tb["n"],
        "r_outcome": r_outcome_log,
        "r_capture": captureR.mean().item(),      # = ship-margin shaping when USE_POTENTIAL_SHAPING
        "r_launch": launchR.mean().item(),
        "r_milestone": prodMR.mean().item(),      # = production-share shaping when USE_POTENTIAL_SHAPING
    }
    return tb, stats, cursor
'''
cc = find_cell(lambda s: "def collect_ppo" in s, "collect cell")
set_src(cc, NEW_COLLECT)

# ============================================================================
# 8. league + train cell: dual Elo, invited members, bounded 2p/4p scheduler
# ============================================================================
NEW_LEAGUE = r'''import copy
import glob
import re

# ---------------- dual-Elo self-play league (PFSP pool + invited members) ------------------
def _freeze_snapshot(net):
    """Move to CPU, eval mode, no grad -> a fixed sparring partner."""
    net = net.cpu().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class LegacyObsAdapter(torch.nn.Module):
    """Wraps a league member saved with the older SCALAR-ownership obs (3 fewer body features:
    v7/v8 F_DIM=101) so it can play inside a one-hot build (v9, F_DIM=104): the 4 seat channels
    collapse back to the scalar code the member was trained on (1 = self, 2 = ANY enemy,
    0 = neutral). DORMANT whenever the member's F_DIM matches the current build's."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, ent, em, am, gl):
        b7 = ent[..., 7:8] + 2.0 * ent[..., 8:11].sum(-1, keepdim=True).clamp(max=1.0)
        return self.net(torch.cat([ent[..., :7], b7, ent[..., 11:]], -1), em, am, gl)


def _unwrap(net):
    return net.net if isinstance(net, LegacyObsAdapter) else net


def _wrap_if_legacy(net, cfg):
    """Current-format members pass through; v7-format (3 fewer body features) get the obs
    adapter; anything else is incompatible."""
    fd = int((cfg or {}).get("F_DIM", F_DIM))
    if fd == F_DIM:
        return net
    if fd == F_DIM - 3:
        return LegacyObsAdapter(net)
    raise ValueError("incompatible member F_DIM %d (current %d)" % (fd, F_DIM))


_POLICY_KW = ("HIDDEN", "D_G", "N_RES_BLOCKS", "USE_GLU", "USE_ATTENTION", "VALUE_RES_BLOCKS",
              "ARCH", "N_HEADS", "N_TX_LAYERS", "TX_MLP_RATIO", "N_STEM_RES", "N_HEAD_RES",
              "F_DIM")


def _cfg_from(blob_cfg, fallback=None):
    """Per-member PolicyNet dims: ckpt config > fallback (e.g. INVITE_CFG) > notebook globals."""
    out = {}
    for k in _POLICY_KW:
        if blob_cfg and k in blob_cfg:
            out[k] = blob_cfg[k]
        elif fallback and k in fallback:
            out[k] = fallback[k]
        else:
            out[k] = globals()[k]
    return out


def build_from_cfg(cfg):
    return PolicyNet(int(cfg.get("F_DIM", F_DIM)), G_DIM, cfg["HIDDEN"], cfg["D_G"], PLANET_CAP,
                     cfg["N_RES_BLOCKS"], cfg["USE_GLU"], INIT_GATE_BIAS,
                     use_attention=cfg["USE_ATTENTION"], value_res_blocks=cfg["VALUE_RES_BLOCKS"],
                     arch=cfg["ARCH"], n_heads=cfg["N_HEADS"], n_tx_layers=cfg["N_TX_LAYERS"],
                     tx_mlp_ratio=cfg["TX_MLP_RATIO"], n_stem_res=cfg["N_STEM_RES"],
                     n_head_res=cfg["N_HEAD_RES"])


def load_snapshot(path, fallback_cfg=None):
    """Load a save_ckpt() .pt into a frozen PolicyNet on CPU. Returns (net, cfg) -- the cfg is
    stored on the member so resumable states can rebuild differently-sized (invited) nets."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = _cfg_from(blob.get("config", {}), fallback_cfg)
    net = build_from_cfg(cfg)
    net.load_state_dict(blob["model"])
    return _freeze_snapshot(_wrap_if_legacy(net, cfg)), cfg


# ---------------- invited members: folder discovery + anchored Elo ------------------
_ELO_RE = re.compile(r"_elo(-?\d+)", re.IGNORECASE)


def _invited_elo(path):
    """Manually-anchored Elo of an invited ckpt: filename `_elo<NNNN>` > ckpt meta > default."""
    m = _ELO_RE.search(os.path.basename(path))
    if m:
        return float(m.group(1))
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("elo") is not None:
            return float(blob["elo"])
    except Exception:
        pass
    return float(INVITE_DEFAULT_ELO)


def discover_invited():
    """Scan the FIRST existing INVITE_DIRS folder for *.pt members; return the top INVITE_MAX
    [(path, anchored_elo)] sorted by Elo (desc)."""
    for d in INVITE_DIRS:
        if d and os.path.isdir(d):
            cands = sorted(glob.glob(os.path.join(d, "*.pt")))
            if cands:
                ranked = sorted(((p, _invited_elo(p)) for p in cands), key=lambda t: -t[1])
                print("[league] invite dir %s: %d candidate(s) -> inviting top %d"
                      % (d, len(ranked), min(INVITE_MAX, len(ranked))))
                return ranked[:INVITE_MAX]
    print("[league] no invite dir found (searched: %s)" % ", ".join(x for x in INVITE_DIRS if x))
    return []


class League:
    """Dual-Elo league. Every member carries a 2p rating (elo) AND a 4p rating (elo4); the
    learner keeps learner_elo / learner_elo4. Scripted anchors AND invited members are
    rating-ANCHORED in both formats (they pin the scales). Snapshots are PFSP-sampled with
    per-format ratings; invited members boost while unmastered, then their matchup rate DROPS
    to INVITE_MASTERED_W. share_4p() turns the (elo2 - elo4) gap into the bounded match mix."""
    def __init__(self, rng, invite=True):
        self.rng = rng
        self.learner_elo = ELO_INIT
        self.learner_elo4 = ELO_INIT
        self.advanced = False                                # curriculum: greedy/medium locked until the advance gate
        def anc(label, kind, elo):
            return {"label": label, "net": None, "kind": kind, "elo": elo, "elo4": elo,
                    "anchor": True, "pinned": True, "n": 0, "n4": 0}
        self.members = [
            anc("random", "random", ELO_RANDOM),
            anc("starter", "starter", ELO_STARTER),
            anc("greedy", "greedy", ELO_GREEDY),
            anc("medium", "medium", ELO_MEDIUM),
            anc("intermediate", "intermediate", ELO_INTERMEDIATE),
        ]
        if invite:
            for path, aelo in discover_invited():
                try:
                    self.add_invited(path, aelo)
                except Exception as e:
                    print("  !! invite failed %s -> %r" % (path, e))

    def add_invited(self, path, aelo):
        net, cfg = load_snapshot(path, fallback_cfg=INVITE_CFG)
        lab = os.path.splitext(os.path.basename(path))[0]
        m = {"label": "inv:" + lab, "net": net, "kind": "invited", "cfg": cfg,
             "elo": float(aelo), "elo4": float(aelo) + INVITE_ELO4_OFFSET,
             "anchor": True, "pinned": True, "n": 0, "n4": 0}
        self.members.append(m)
        print("    invited %-26s anchored elo2 %6.0f / elo4 %6.0f  (%s)"
              % (m["label"], m["elo"], m["elo4"], os.path.basename(path)))
        return m

    def snapshots(self):
        return [m for m in self.members if m["kind"] == "snapshot"]

    def anchor(self, kind):
        return next(m for m in self.members if m["kind"] == kind)

    def add_checkpoint(self, path, label=None, elo=None):
        net, cfg = load_snapshot(path)
        m = {"label": label or os.path.basename(path), "net": net, "kind": "snapshot", "cfg": cfg,
             "elo": ELO_INIT if elo is None else elo, "elo4": ELO_INIT if elo is None else elo,
             "anchor": False, "pinned": True, "n": 0, "n4": 0}
        self.members.append(m)
        return m

    def add_learner_snapshot(self, net, it):
        snap = _freeze_snapshot(copy.deepcopy(net))
        m = {"label": "it%d" % it, "net": snap, "kind": "snapshot",
             "elo": self.learner_elo, "elo4": self.learner_elo4,
             "anchor": False, "pinned": False, "n": 0, "n4": 0}
        self.members.append(m)
        self._prune()
        try:  # persist the weights as soon as the snapshot is generated (survives pruning)
            d = os.path.join(CKPT_DIR, "league_agents"); os.makedirs(d, exist_ok=True)
            save_ckpt(snap, os.path.join(d, m["label"] + ".pt"),
                      meta={"iter": it, "elo": float(m["elo"]), "elo4": float(m["elo4"])})
        except Exception as e:
            print("  !! league snapshot save failed: %r" % e)

    def _prune(self):
        autos = [m for m in self.members if m["kind"] == "snapshot" and not m["pinned"]]
        for m in autos[:max(0, len(autos) - LEAGUE_MAX_SNAPSHOTS)]:   # drop the oldest (FIFO)
            self.members.remove(m)

    # ---------------- dual-Elo machinery ----------------
    def _p_beat(self, m, fmt):
        me = self.learner_elo if fmt == 2 else self.learner_elo4
        oe = m["elo"] if fmt == 2 else m["elo4"]
        return 1.0 / (1.0 + 10.0 ** ((oe - me) / 400.0))

    def _pfsp_weight(self, p):
        if PFSP_MODE == "hard":
            return (1.0 - p) ** PFSP_POWER + PFSP_FLOOR   # focus on opponents you can't yet beat
        if PFSP_MODE == "even":
            return p * (1.0 - p) + PFSP_FLOOR             # focus on evenly-matched opponents
        return 1.0                                        # uniform

    def _mastered(self, m, fmt, thr):
        n = m["n"] if fmt == 2 else m.get("n4", 0)
        wr = m.get("wr") if fmt == 2 else m.get("wr4")
        return (n >= SCRIPT_BOOST_MIN_GAMES) and ((wr if wr is not None else 0.0) >= thr)

    def _weak_boost(self, m, fmt):
        """Weak-point matchmaking. If the learner's ACTUAL score EMA vs this member runs below
        the Elo-EXPECTED score (e.g. Elo says 0.90 but reality is 0.70), boost the matchup rate:
        1 + WEAK_BOOST_GAIN*(expected - actual), capped at WEAK_BOOST_MAX (x3 at the 0.90-vs-0.70
        example). ONLY targets weakness: overperforming opponents are never down-weighted, and
        the detector stays off below WEAK_BOOST_MIN_GAMES matches / inside the deadband."""
        if not WEAK_BOOST_ENABLED:
            return 1.0
        n = m["n"] if fmt == 2 else m.get("n4", 0)
        wr = m.get("wr") if fmt == 2 else m.get("wr4")
        if n < WEAK_BOOST_MIN_GAMES or wr is None:
            return 1.0
        gap = self._p_beat(m, fmt) - wr
        if gap < WEAK_BOOST_MIN_GAP:
            return 1.0
        return min(WEAK_BOOST_MAX, 1.0 + WEAK_BOOST_GAIN * gap)

    def _weights(self, fmt, net=None):
        """Per-member sampling weights for one format: PFSP(expected score) x footprint mix
        (2p only) x invited/scripted boosts x the advance lock."""
        members = self.members
        elo_w = [self._pfsp_weight(self._p_beat(m, fmt)) for m in members]
        ws = list(elo_w)
        if fmt == 2 and net is not None and MATCH_FP_W > 0.0:
            try:
                lfp = self._fingerprint(net)
                fd = []
                for m in members:
                    if m["net"] is None:
                        fd.append(1.0)                                  # scripted anchor: maximally distinct
                    else:
                        if m.get("fp") is None:
                            m["fp"] = self._fingerprint(m["net"])
                        fd.append(_js_div(lfp[0], m["fp"][0]) + _js_div(lfp[1], m["fp"][1]))
                mxe = max(elo_w) or 1.0; mxf = max(fd) or 1.0
                ws = [MATCH_ELO_W * (elo_w[i] / mxe) + MATCH_FP_W * (fd[i] / mxf) + PFSP_FLOOR
                      for i in range(len(members))]
            except Exception:
                ws = list(elo_w)
        for i, m in enumerate(members):
            if m["kind"] == "invited":
                # high-level sparring focus until mastered, then DROP the matchup rate
                ws[i] *= INVITE_MATCH_BOOST if not self._mastered(m, fmt, INVITE_MASTER_WR) else INVITE_MASTERED_W
            elif m["net"] is None and SCRIPT_MATCH_BOOST != 1.0:
                if not self._mastered(m, fmt, SCRIPT_BOOST_DROP_WR):
                    ws[i] *= SCRIPT_MATCH_BOOST              # spend compute on UNMASTERED scripted anchors
            ws[i] *= self._weak_boost(m, fmt)                # weak-point matchmaking (>= 1.0)
            if (not self.advanced) and m["kind"] in ("greedy", "medium"):
                ws[i] = 0.0                                  # curriculum lock until the advance gate
        return ws

    def sample(self, net=None, fmt=2):
        """PFSP-sample ONE opponent using the per-format ratings + boosts."""
        ws = self._weights(fmt, net)
        tot = sum(ws)
        if tot <= 0.0:
            return self.rng.choice(self.members)
        r = self.rng.random() * tot
        c = 0.0
        for m, w in zip(self.members, ws):
            c += w
            if r <= c:
                return m
        return self.members[-1]

    def sample_seats(self, net=None, k=3, fmt=4):
        """k seats for an FFA: weighted draws WITHOUT replacement while distinct members with
        positive weight remain (falls back to with-replacement on a tiny pool)."""
        ws = self._weights(fmt, net)
        out = []
        for _ in range(k):
            tot = sum(ws)
            if tot <= 0.0:
                out.append(self.rng.choice(self.members))
                continue
            r = self.rng.random() * tot
            c = 0.0; pick = len(self.members) - 1
            for i, w in enumerate(ws):
                c += w
                if r <= c:
                    pick = i
                    break
            out.append(self.members[pick])
            if sum(1 for w in ws if w > 0.0) > 1:
                ws[pick] = 0.0                                # without replacement while we can
        return out

    def update_elo(self, m, learner_score):
        """2p result: learner_score in [0,1] = win + 0.5*draw over the rollout. Anchors stay
        fixed. The 2p delta also drags BOTH 4p ratings by ELO_COUPLING (correlated ratings)."""
        exp = self._p_beat(m, 2)
        delta = ELO_K * (learner_score - exp)
        self.learner_elo += delta
        self.learner_elo4 += ELO_COUPLING * delta
        m["n"] += 1
        m["wr"] = learner_score if m.get("wr") is None else (1.0 - WR_EMA_BETA) * m["wr"] + WR_EMA_BETA * learner_score
        if not m["anchor"]:
            m["elo"] -= delta
            m["elo4"] -= ELO_COUPLING * delta

    def update_elo_4p(self, seats, seat_scores):
        """4p FFA result as len(seats) pairwise updates at ELO_K4/len each (standard multiplayer
        decomposition). Coupled back into the 2p ratings by ELO_COUPLING."""
        kf = ELO_K4 / max(1, len(seats))
        for m, sc in zip(seats, seat_scores):
            exp = self._p_beat(m, 4)
            delta = kf * (sc - exp)
            self.learner_elo4 += delta
            self.learner_elo += ELO_COUPLING * delta
            m["n4"] = m.get("n4", 0) + 1
            m["wr4"] = sc if m.get("wr4") is None else (1.0 - WR_EMA_BETA) * m["wr4"] + WR_EMA_BETA * sc
            if not m["anchor"]:
                m["elo4"] -= delta
                m["elo"] -= ELO_COUPLING * delta

    def share_4p(self):
        """Share of 4p iterations from the rating gap, clipped to the 1:1 .. 1:6 envelope.
        4p lagging (elo2 > elo4) -> more 4p; 4p ahead -> drift back toward 2p."""
        gap = (self.learner_elo - self.learner_elo4) / 400.0
        return min(S4_MAX, max(S4_MIN, S4_BASE + S4_GAIN * gap))

    def leaderboard(self):
        rows = sorted(self.members, key=lambda m: -m["elo"])
        s = "  league | learner elo2 %.0f elo4 %.0f | share4p %.2f | %d members\n" % (
            self.learner_elo, self.learner_elo4, self.share_4p(), len(self.members))
        for m in rows:
            tag = "[anchor]" if m["anchor"] else ("[pinned]" if m["pinned"] else "")
            if m["kind"] == "invited":
                tag = "[invited %s%s]" % ("M" if self._mastered(m, 2, INVITE_MASTER_WR) else "-",
                                          "M" if self._mastered(m, 4, INVITE_MASTER_WR) else "-")
            elif m["net"] is None:
                boosted = not self._mastered(m, 2, SCRIPT_BOOST_DROP_WR)
                tag += " x%.1f" % SCRIPT_MATCH_BOOST if boosted else " (mastered)"
            w2, w4 = self._weak_boost(m, 2), self._weak_boost(m, 4)
            if w2 > 1.0:
                tag += " weak2 x%.1f" % w2
            if w4 > 1.0:
                tag += " weak4 x%.1f" % w4
            wr = m.get("wr"); wr4 = m.get("wr4")
            s += "    %-22s elo2 %6.0f (n=%-3d wr %s)  elo4 %6.0f (n4=%-3d wr %s) %s\n" % (
                m["label"], m["elo"], m["n"], ("%.2f" % wr) if wr is not None else " -- ",
                m["elo4"], m.get("n4", 0), ("%.2f" % wr4) if wr4 is not None else " -- ", tag)
        return s


def save_ckpt(net, path, meta=None):
    blob = {"model": net.state_dict(),
            "config": {"HIDDEN": HIDDEN, "N_RES_BLOCKS": N_RES_BLOCKS, "D_G": D_G,
                       "PLANET_CAP": PLANET_CAP, "F_DIM": F_DIM, "G_DIM": G_DIM,
                       "ARCH": ARCH, "N_TX_LAYERS": N_TX_LAYERS, "N_HEADS": N_HEADS,
                       "TX_MLP_RATIO": TX_MLP_RATIO, "N_STEM_RES": N_STEM_RES, "N_HEAD_RES": N_HEAD_RES,
                       "USE_GLU": USE_GLU, "USE_ATTENTION": USE_ATTENTION, "VALUE_RES_BLOCKS": VALUE_RES_BLOCKS,
                       "target_actor": True}}
    if meta:
        blob.update(meta)
    # league snapshots can be SMALLER than the live config (invited dims): persist the true dims
    if hasattr(net, "trunk") and hasattr(net.trunk, "layers"):
        blob["config"]["HIDDEN"] = net.h
        blob["config"]["N_TX_LAYERS"] = len(net.trunk.layers)
        blob["config"]["N_HEADS"] = net.trunk.layers[0].nh if net.trunk.layers else N_HEADS
        blob["config"]["N_STEM_RES"] = len(net.trunk.stem_a)
        blob["config"]["N_HEAD_RES"] = len(net.trunk.head_a)
    torch.save(blob, path)


def anneal_ent_coef(it):
    if ENT_DECAY_ITERS <= 0:
        return ENT_COEF_END
    f = min(1.0, it / ENT_DECAY_ITERS)
    return ENT_COEF_START + (ENT_COEF_END - ENT_COEF_START) * f


def train(total_iters=TOTAL_ITERS, log_every=1, resume_from=None):
    net = build_policy()
    try:
        opt = torch.optim.Adam(net.parameters(), lr=LR, eps=ADAM_EPS, fused=(DEVICE.type == 'cuda'))
    except Exception:
        opt = torch.optim.Adam(net.parameters(), lr=LR, eps=ADAM_EPS)
    env = GpuEnv(PLANET_CAP, FLEET_CAP, EPISODE_STEPS, SHIP_SPEED, DEVICE)
    rng = random.Random(SEED * 2654435761 + 12345)
    start_it = 0
    best_wr = -1.0

    league = League(rng)
    for ck in LEAGUE_INIT_CKPTS:
        try:
            m = league.add_checkpoint(ck)
            print("league seeded with %s  <- %s" % (m["label"], ck))
        except Exception as e:
            print("  !! skipped league ckpt %s -> %r" % (ck, e))

    if resume_from:
        start_it, best_wr = load_train_state(net, opt, league, resume_from)
        print("resumed from %s -> global iter %d, best_wr %.2f, %d league members"
              % (resume_from, start_it, best_wr, len(league.members)))

    hist = {"iter": [], "return": [], "win_rate": [],
            "lnch_per_step": [], "approx_kl": [], "clipfrac": [], "sigma": [], "grad_norm": [],
            "r_outcome": [], "r_capture": [], "r_milestone": [], "r_launch": [],
            "learner_elo": [], "learner_elo4": [], "share_4p": [], "fmt": []}

    pool, pool_round = None, -1
    # held-out gauntlet worlds: FIXED seed range never touched by training rounds, so the
    # best-ckpt score measures generalization and stays comparable across the whole run.
    heldout_pool = make_world_pool(ELO_RECAL_ENVS, base_seed=SEED + 999_999_937)
    acc4 = 0.0       # error-diffusion accumulator -> the realized 2p:4p ratio equals share_4p
    for step in range(total_iters):
        it = start_it + step
        rnd = (it // WORLD_RESAMPLE_EVERY) if WORLD_RESAMPLE_EVERY > 0 else 0
        if rnd != pool_round:
            pool = make_world_pool(N_WORLDS, base_seed=SEED + rnd * N_WORLDS)
            cursor, pool_round = 0, rnd
            print("  [worlds] pool round %d @ it%d (base_seed %d)" % (rnd, it, SEED + rnd * N_WORLDS))
        if ELO_RECAL_EVERY > 0 and step > 0 and it % ELO_RECAL_EVERY == 0:  # re-ground BOTH rating scales
            league.recalibrate_elo(net, pool)
            if FOURP_ENABLED:
                league.recalibrate_elo4(net, pool)
        t0 = time.time()
        globals()['CUR_ENT_COEF'] = anneal_ent_coef(it)   # entropy anneal (anti collapse -> exploit)
        # v7 guardrails: linear LR warmup + critic-only warmup (both per train() CALL, not global iter)
        globals()['PPO_POLICY_COEF'] = 0.0 if step < VALUE_WARMUP_ITERS else 1.0
        if LR_WARMUP_ITERS > 0:
            wf = min(1.0, (step + 1) / float(LR_WARMUP_ITERS))
            for pg in opt.param_groups:
                pg['lr'] = LR * wf

        # grow the pool: snapshot the learner (after a short anchors-only warmup) every SELFPLAY_REFRESH iters
        if (it >= SNAPSHOT_WARMUP) and (it % max(1, SELFPLAY_REFRESH) == 0):
            league.add_learner_snapshot(net, it + 1)

        # ---- v8 match-type scheduler: dual-Elo gap -> bounded 2p/4p mix (1:1 .. 1:6) ----
        s4 = league.share_4p() if FOURP_ENABLED else 0.0
        acc4 += s4
        if acc4 >= 1.0:
            fmt = 4; acc4 -= 1.0
        else:
            fmt = 2

        # pick the opponent seat(s) for this rollout: PFSP over the whole league per format
        if fmt == 2:
            seats = [league.sample(net, fmt=2)]
        else:
            seats = league.sample_seats(net, k=3, fmt=4)
        for m in seats:
            if m["net"] is not None:
                m["net"].to(DEVICE)

        tb, rs, cursor = collect_ppo(env, net, pool, cursor, rng, seats,
                                     n_players=fmt, n_envs=(B if fmt == 2 else B_4P))
        us = ppo_update(net, opt, tb)

        # dual Elo: 2p = single update; 4p = pairwise vs every seat (cross-coupled inside)
        if fmt == 2:
            league.update_elo(seats[0], rs["score"])
        else:
            league.update_elo_4p(seats, rs["seat_scores"])
        if (not league.advanced) and league.learner_elo >= ADVANCE_TRIGGER_ELO:   # curriculum: confirm with an all-anchor recal (fresh seeds)
            league.recalibrate_elo(net, make_world_pool(ELO_RECAL_ENVS, base_seed=SEED + 104729 + 7919 * (it + 1)), all_anchors=True)
            if league.learner_elo >= ADVANCE_CONFIRM_ELO:
                league.advanced = True
                print("  [advance] learner hit %.0f ELO; all-anchor recal %.0f >= %d -> UNLOCK greedy/medium" % (ADVANCE_TRIGGER_ELO, league.learner_elo, ADVANCE_CONFIRM_ELO))
            else:
                print("  [advance] learner hit %.0f ELO but all-anchor recal %.0f < %d -> stay in base regime" % (ADVANCE_TRIGGER_ELO, league.learner_elo, ADVANCE_CONFIRM_ELO))
        for m in seats:
            if m["net"] is not None:
                m["net"].to("cpu")           # keep only the live policy resident on the GPU

        dt = time.time() - t0
        sps = rs["transitions"] / max(dt, 1e-9)

        hist["iter"].append(it + 1)
        hist["return"].append(rs["mean_return"]); hist["win_rate"].append(rs["win_rate"])
        hist["lnch_per_step"].append(rs["lnch_per_step"]); hist["approx_kl"].append(us["approx_kl"])
        hist["clipfrac"].append(us["clipfrac"]); hist["sigma"].append(us["sigma"]); hist["grad_norm"].append(us["grad_norm"])
        hist["r_outcome"].append(rs["r_outcome"]); hist["r_capture"].append(rs["r_capture"])
        hist["r_milestone"].append(rs["r_milestone"]); hist["r_launch"].append(rs["r_launch"])
        hist["learner_elo"].append(league.learner_elo)
        hist["learner_elo4"].append(league.learner_elo4)
        hist["share_4p"].append(s4); hist["fmt"].append(fmt)

        # v7 tripwire: gate-collapse detector (launch rate decaying under healthy-looking KL)
        _l = hist["lnch_per_step"]
        if len(_l) >= TRIPWIRE_BASE_ITERS and (it + 1) % 10 == 0:
            _base = sum(_l[:TRIPWIRE_BASE_ITERS]) / TRIPWIRE_BASE_ITERS
            _ema = sum(_l[-5:]) / len(_l[-5:])
            if _base > 0 and _ema < TRIPWIRE_FRAC * _base:
                print("  [TRIPWIRE] launch/st EMA %.2f < %.0f%% of early baseline %.2f -> passivity ratchet suspected; check gauntlet+elo before continuing" % (_ema, TRIPWIRE_FRAC * 100, _base))

        if (it + 1) % log_every == 0 or step == 0 or step == total_iters - 1:
            # heartbeat: R[o c p ln] = outcome, capture, prod-milestone, launch (real units)
            opp_lab = "+".join(m["label"] for m in seats)
            print("it%4d %dp | ret %8.2f wr %.2f sc %.2f | R[o %7.1f c %6.1f p %5.1f ln %5.1f] | "
                  "lnch/st %.2f | kl %.3f cf %.2f gn %.1f | loss %7.3f (pol %.3f vf %.3f) | "
                  "elo2 %5.0f elo4 %5.0f s4 %.2f vs %-22s | sps %5.0f"
                  % (it + 1, fmt, rs["mean_return"], rs["win_rate"], rs["score"],
                     rs["r_outcome"], rs["r_capture"], rs["r_milestone"], rs["r_launch"],
                     rs["lnch_per_step"], us["approx_kl"], us["clipfrac"], us["grad_norm"],
                     us["total"], us["policy"], us["vf"],
                     league.learner_elo, league.learner_elo4, s4, opp_lab[:22], sps))

        # checkpoint: periodic + best-by-MIXED-GAUNTLET (2p + 4p; fixed opponents, held-out worlds)
        if ((it + 1) % max(1, CKPT_EVERY) == 0) or (step == total_iters - 1):
            save_ckpt(net, CKPT_PATH, meta={"iter": it + 1, "win_rate": rs["win_rate"]})
            save_train_state(net, opt, league, it + 1, best_wr, TRAIN_STATE_PATH)
            g2 = eval_gauntlet(net, heldout_pool)
            g4 = eval_gauntlet4(net, heldout_pool) if FOURP_ENABLED else g2
            gscore = (1.0 - GAUNTLET_4P_W) * g2 + GAUNTLET_4P_W * g4
            hist.setdefault("gauntlet", []).append(gscore)
            hist.setdefault("gauntlet2", []).append(g2)
            hist.setdefault("gauntlet4", []).append(g4)
            if gscore > best_wr:
                best_wr = gscore
                save_ckpt(net, BEST_CKPT_PATH, meta={"iter": it + 1, "gauntlet": float(gscore),
                                                     "gauntlet2": float(g2), "gauntlet4": float(g4)})
                print("  [best] mixed gauntlet %.3f (2p %.3f / 4p %.3f) -> saved best" % (gscore, g2, g4))
            print(league.leaderboard(), end="")

    save_ckpt(net, CKPT_PATH, meta={"iter": start_it + total_iters, "win_rate": hist["win_rate"][-1]})
    save_train_state(net, opt, league, start_it + total_iters, best_wr, TRAIN_STATE_PATH)
    print("saved final checkpoint ->", CKPT_PATH, "| best mixed gauntlet %.2f ->" % best_wr, BEST_CKPT_PATH)
    print(league.leaderboard(), end="")
    import json as _json
    _json.dump({k: [float(x) for x in v] for k, v in hist.items()},
               open(os.path.join(CKPT_DIR, "metrics.json"), "w"))
    print("saved metrics ->", os.path.join(CKPT_DIR, "metrics.json"))
    globals()["LEAGUE"] = league      # also expose as a module global (robust if the caller drops it)
    return net, hist, league
'''
lc = find_cell(lambda s: "class League" in s and "def train" in s, "league/train cell")
set_src(lc, NEW_LEAGUE)

# ============================================================================
# 9. helpers cell: gauntlets (2p + 4p), dual-Elo state/recal, diversity prune
# ============================================================================
NEW_HELPERS = r'''# ---- resumable training state + fixed-opponent gauntlets (2p AND 4p) + Elo re-grounding ----
import math

def eval_gauntlet(net, worlds, anchors=("starter", "intermediate", "greedy"), n_envs=None):
    """Fixed-opponent 2p score for best-ckpt selection (greedy, deployment-faithful op set)."""
    n = n_envs or ELO_RECAL_ENVS
    return sum(_eval_score(net, a, worlds, n) for a in anchors) / len(anchors)


GAUNTLET4_SEATS = ("starter", "greedy", "intermediate")   # fixed FFA trio for the 4p gauntlet

_OPP_BY_KIND = {"random": 0, "starter": 1, "noop": 2, "medium": 3, "greedy": 4, "intermediate": 5}


def _eval_score(net, anchor_kind, worlds, n_envs):
    """No-grad mean learner-score (win + 0.5*draw) of `net` vs a scripted anchor on `worlds` (greedy, 2p)."""
    env = GpuEnv(PLANET_CAP, FLEET_CAP, EPISODE_STEPS, SHIP_SPEED, DEVICE)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)])
    active = torch.ones(n_envs, device=DEVICE); score = torch.zeros(n_envs, device=DEVICE)
    def _sc(s0, s1):
        return torch.where(s0 > s1, torch.ones_like(s0), torch.where(s0 < s1, torch.zeros_like(s0), torch.full_like(s0, 0.5)))
    with torch.no_grad():
        for _t in range(EPISODE_STEPS):
            ent, em, am, gl = env_encode(env, 0)
            a_t, _, _ = act(net, ent, em, am, gl, greedy=True)
            env_step(env, a_t, _OPP_BY_KIND[anchor_kind], None, step_idx=_t)
            s0, s1, a0, a1 = settle(env)
            term = (env.step_ct >= float(env.T - 2)) | (~(a0 & a1))
            newly = (active > 0.5) & term
            score = torch.where(newly, _sc(s0, s1), score)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s0, s1, _, _ = settle(env)
        score = torch.where(active > 0.5, _sc(s0, s1), score)
    return float(score.mean().item())


def _eval_score4(net, trio, worlds, n_envs):
    """No-grad mean PAIRWISE score of greedy `net` (seat 0) vs 3 scripted anchors in a 4p FFA."""
    env = GpuEnv(PLANET_CAP, FLEET_CAP, EPISODE_STEPS, SHIP_SPEED, DEVICE)
    env.reset([worlds[i % len(worlds)] for i in range(n_envs)], n_players=4)
    seats = [{"pid": p + 1, "script": _OPP_BY_KIND[k]} for p, k in enumerate(trio)]
    active = torch.ones(n_envs, device=DEVICE); score = torch.zeros(n_envs, device=DEVICE)
    def _pw(s_all):
        s0 = s_all[:, 0:1].expand_as(s_all[:, 1:]); rest = s_all[:, 1:]
        one = torch.ones_like(rest)
        return torch.where(s0 > rest, one, torch.where(s0 < rest, 0.0 * one, 0.5 * one)).mean(1)
    with torch.no_grad():
        for _t in range(EPISODE_STEPS):
            ent, em, am, gl = env_encode(env, 0)
            a_t, _, _ = act(net, ent, em, am, gl, greedy=True)
            env_step(env, a_t, seats=seats, step_idx=_t)
            s_all, alive_all = settle_n(env)
            n_alive = alive_all.to(DTYPE).sum(1)
            term = (env.step_ct >= float(env.T - 2)) | (n_alive <= 1.0) | (~alive_all[:, 0])
            newly = (active > 0.5) & term
            score = torch.where(newly, _pw(s_all), score)
            active = torch.where(term, torch.zeros_like(active), active)
            if active.sum().item() == 0:
                break
        s_all, _ = settle_n(env)
        score = torch.where(active > 0.5, _pw(s_all), score)
    return float(score.mean().item())


def eval_gauntlet4(net, worlds, n_envs=None):
    """Fixed-trio 4p FFA gauntlet (mean pairwise score vs starter/greedy/intermediate)."""
    return _eval_score4(net, GAUNTLET4_SEATS, worlds, n_envs or ELO_RECAL_ENVS)


# ---- league (de)serialization: dual ratings + per-member dims (invited nets are smaller) ----
def _league_state_dict(self):
    members = []
    for m in self.members:
        members.append({"label": m["label"], "kind": m["kind"], "elo": m["elo"],
                        "elo4": m.get("elo4", m["elo"]), "anchor": m["anchor"], "pinned": m["pinned"],
                        "n": m["n"], "n4": m.get("n4", 0), "wr": m.get("wr"), "wr4": m.get("wr4"),
                        "cfg": m.get("cfg"),
                        "model": (_unwrap(m["net"]).state_dict() if m["net"] is not None else None)})
    return {"learner_elo": self.learner_elo, "learner_elo4": self.learner_elo4,
            "advanced": self.advanced, "members": members}

def _league_load_state_dict(self, sd):
    self.learner_elo = sd["learner_elo"]
    self.learner_elo4 = sd.get("learner_elo4", sd["learner_elo"])
    self.advanced = sd.get("advanced", self.advanced)
    self.members = []
    for e in sd["members"]:
        net = None
        if e["model"] is not None:
            net = build_from_cfg(e["cfg"]) if e.get("cfg") else build_policy()
            net.load_state_dict(e["model"])
            net = _freeze_snapshot(_wrap_if_legacy(net, e.get("cfg")))
        m = {"label": e["label"], "kind": e["kind"], "net": net, "elo": e["elo"],
             "elo4": e.get("elo4", e["elo"]), "anchor": e["anchor"], "pinned": e["pinned"],
             "n": e["n"], "n4": e.get("n4", 0)}
        if e.get("cfg") is not None:
            m["cfg"] = e["cfg"]
        if e.get("wr") is not None:
            m["wr"] = e["wr"]
        if e.get("wr4") is not None:
            m["wr4"] = e["wr4"]
        self.members.append(m)

League.state_dict = _league_state_dict
League.load_state_dict = _league_load_state_dict


def save_train_state(net, opt, league, global_iter, best_wr, path):
    """Full resumable state: weights + Adam moments + global iter + best score + dual-Elo league."""
    pa = getattr(net, "popart", None)
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "league": league.state_dict(),
                "global_iter": int(global_iter), "best_wr": float(best_wr),
                "popart": ({"mu": pa.mu, "nu": pa.nu, "sigma": pa.sigma, "init": pa.init} if pa else None)}, path)


def load_train_state(net, opt, league, path):
    """Restore net/opt/league IN PLACE. Returns (start_iter, best_wr). Accepts a full train-state
    OR a plain save_ckpt weight (warm-start: weights only, fresh Adam/league). v7 states load too
    (missing elo4 fields default to the 2p values); invited members are re-discovered if absent."""
    blob = torch.load(path, map_location=DEVICE, weights_only=False)
    net.load_state_dict(blob["model"])
    if blob.get("popart") and getattr(net, "popart", None) is not None:
        d = blob["popart"]; net.popart.mu = d["mu"]; net.popart.nu = d["nu"]; net.popart.sigma = d["sigma"]; net.popart.init = d["init"]
    if "opt" in blob:
        opt.load_state_dict(blob["opt"])
    else:
        print("  [resume] %s has no optimizer state -> WARM-START (fresh Adam moments)" % os.path.basename(path))
    start_it = int(blob.get("global_iter", blob.get("iter", 0)))
    if "league" in blob:
        league.load_state_dict(blob["league"])
        if not any(m["kind"] == "invited" for m in league.members):   # v7 state: re-invite
            for p, aelo in discover_invited():
                try:
                    league.add_invited(p, aelo)
                except Exception as e:
                    print("  !! invite failed %s -> %r" % (p, e))
    else:  # warm-start: calibrate learner Elo vs anchors so the first snapshot is not stamped Elo 0
        rnd = (start_it // WORLD_RESAMPLE_EVERY) if WORLD_RESAMPLE_EVERY > 0 else 0
        print("  [resume] no league state -> calibrating learner Elo vs anchors")
        league.recalibrate_elo(net, make_world_pool(ELO_RECAL_ENVS, base_seed=SEED + rnd * N_WORLDS))
        league.learner_elo4 = league.learner_elo
    return start_it, float(blob.get("best_wr", -1.0))


def _league_recalibrate_elo(self, net, worlds, n_envs=None, all_anchors=False):
    """Re-ground every 2p rating from win-rate vs the FIXED anchors on `worlds` (drift removal)."""
    n_envs = n_envs or ELO_RECAL_ENVS
    _ADV = ("greedy", "medium")
    anchors = [m for m in self.members if m["anchor"] and m["net"] is None
               and (all_anchors or self.advanced or m["kind"] not in _ADV)]
    def est(scores):  # scores: [(anchor_elo, learner_score), ...] -> averaged ELO estimate
        vals = []
        for ae, sc in scores:
            sc = min(max(sc, 0.02), 0.98)
            vals.append(ae + 400.0 * math.log10(sc / (1.0 - sc)))
        return sum(vals) / len(vals)
    self.learner_elo = est([(a["elo"], _eval_score(net, a["kind"], worlds, n_envs)) for a in anchors])
    for m in self.members:
        if m["anchor"] or m["net"] is None:
            continue
        m["net"].to(DEVICE)
        m["elo"] = est([(a["elo"], _eval_score(m["net"], a["kind"], worlds, n_envs)) for a in anchors])
        m["net"].to("cpu"); m["n"] = 0
    print("    [elo recal] learner=%.0f | %s" % (self.learner_elo,
          " ".join("%s=%.0f" % (m["label"], m["elo"]) for m in self.members if not m["anchor"])))

League.recalibrate_elo = _league_recalibrate_elo


def _league_recalibrate_elo4(self, net, worlds, n_envs=None):
    """Re-ground the 4p ratings: learner + every snapshot eval'd in 4p FFA vs the fixed scripted
    trio; rating = mean trio Elo4 + the pairwise-score logit vs the field. Invited/scripted
    members stay anchored."""
    n_envs = n_envs or ELO_RECAL_ENVS
    base = sum(self.anchor(k)["elo4"] for k in GAUNTLET4_SEATS) / len(GAUNTLET4_SEATS)
    def est(s):
        s = min(max(s, 0.02), 0.98)
        return base + 400.0 * math.log10(s / (1.0 - s))
    self.learner_elo4 = est(_eval_score4(net, GAUNTLET4_SEATS, worlds, n_envs))
    for m in self.members:
        if m["anchor"] or m["net"] is None:
            continue
        m["net"].to(DEVICE)
        m["elo4"] = est(_eval_score4(m["net"], GAUNTLET4_SEATS, worlds, n_envs))
        m["net"].to("cpu"); m["n4"] = 0
    print("    [elo4 recal] learner4=%.0f | %s" % (self.learner_elo4,
          " ".join("%s=%.0f" % (m["label"], m["elo4"]) for m in self.members if not m["anchor"])))

League.recalibrate_elo4 = _league_recalibrate_elo4


# ---- league pruning: keep the most behaviorally DISTINCT snapshots (replaces FIFO) ----------
def _league_probe_obs(self):
    """Fingerprint probe = REAL encoded game states (on-manifold), cached once."""
    if getattr(self, '_probe', None) is None:
        env = GpuEnv(PLANET_CAP, FLEET_CAP, EPISODE_STEPS, SHIP_SPEED, DEVICE)
        env.reset(make_world_pool(16, base_seed=0xC0FFEE % 100000))
        with torch.no_grad():
            for _t in range(20):
                e, m, a, g = env_encode(env, 0)
                rnd = torch.softmax(torch.randn(env.B, PLANET_CAP, PLANET_CAP + 1, device=DEVICE), -1)
                env_step(env, rnd, 1, None, step_idx=_t)
        e, m, a, g = env_encode(env, 0)
        self._probe = (e.cpu(), m.cpu(), a.cpu(), g.cpu())
    return self._probe

def _league_fingerprint(self, net):
    """Behavioral fingerprint = mean (WHERE allocation, gate fire/hold) on the fixed probe batch."""
    ent, em, am, gl = self._probe_obs()
    dev = next(net.parameters()).device
    with torch.no_grad():
        dest_logits, gate_logits, _ = net(ent.to(dev), em.to(dev), am.to(dev), gl.to(dev))
        dest = torch.softmax(dest_logits, -1).reshape(-1, dest_logits.shape[-1]).mean(0)  # (E,)
        gp = torch.sigmoid(gate_logits).reshape(-1).float().mean()                         # mean fire-prob
        gate = torch.stack([gp, 1.0 - gp])                                                 # (2,) fire/hold
    return dest.cpu(), gate.cpu()

def _js_div(p, q):
    m = 0.5 * (p + q); eps = 1e-9
    kl = lambda a, b: (a * (torch.log(a + eps) - torch.log(b + eps))).sum()
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))

def _league_prune(self):
    """Keep LEAGUE_MAX_SNAPSHOTS most distinct auto-snapshots: repeatedly drop the one whose
    NEAREST neighbor (dest+gate Jensen-Shannon divergence) is closest = most redundant.
    Anchors / pinned ckpts and the NEWEST snapshot are always kept. FIFO fallback on error."""
    autos = [m for m in self.members if m['kind'] == 'snapshot' and not m['pinned']]
    if len(autos) <= LEAGUE_MAX_SNAPSHOTS:
        return
    try:
        fps = [self._fingerprint(m['net']) for m in autos]
    except Exception as e:
        print('  !! league diversity-prune -> FIFO fallback: %r' % e)
        for m in autos[:len(autos) - LEAGUE_MAX_SNAPSHOTS]:
            self.members.remove(m)
        return
    def d(i, j):
        return _js_div(fps[i][0], fps[j][0]) + _js_div(fps[i][1], fps[j][1])
    newest = len(autos) - 1                      # most recently appended -> always keep
    keep = list(range(len(autos)))
    while len(keep) > LEAGUE_MAX_SNAPSHOTS:
        drop, drop_nn = None, float('inf')
        for i in keep:
            if i == newest:
                continue
            nn = min(d(i, j) for j in keep if j != i)
            if nn < drop_nn:
                drop_nn, drop = nn, i
        if drop is None:
            break
        keep.remove(drop)
    kept = set(keep)
    for idx, m in enumerate(autos):
        if idx not in kept:
            self.members.remove(m)
    print('  league diversity-prune: kept %d/%d snapshots (max behavioral distinctness)'
          % (len(kept), len(autos)))

League._probe_obs = _league_probe_obs
League._fingerprint = _league_fingerprint
League._prune = _league_prune
'''
hc = find_cell(lambda s: s.startswith("# ---- resumable training state"), "helpers cell")
set_src(hc, NEW_HELPERS)

# ============================================================================
# 10. league-save cell: persist BOTH ratings in the exported weights
# ============================================================================
ls = find_cell(lambda s: "LEAGUE_OUT" in s, "league-save cell")
patch(ls, 'meta={"elo": m["elo"], "label": m["label"], "games": m["n"]}',
      'meta={"elo": m["elo"], "elo4": m.get("elo4"), "label": m["label"],'
      ' "games": m["n"], "games4": m.get("n4", 0)}',
      "league-save meta")

# ============================================================================
# 11. plot cell: dual-Elo + 4p-share panels
# ============================================================================
pl = find_cell(lambda s: "Training-health curves" in s, "plot cell")
patch(pl, '_panel(ax[0, 2], "learner_elo", "learner Elo")',
      '_panel(ax[0, 2], "learner_elo", "learner Elo (2p)")', "plot elo panel")
patch(pl, 'ax[1, 3].axis("off"); ax[2, 3].axis("off")',
      '_panel(ax[1, 3], "learner_elo4", "learner Elo (4p)")\n'
      '_panel(ax[2, 3], "share_4p", "share of 4p iterations", pct=True)', "plot v8 panels")
patch(pl, 'fig.suptitle("Orbit Wars v5 -- training health (faint = raw, bold = EMA)", fontsize=12)',
      'fig.suptitle("Orbit Wars v8 -- training health (faint = raw, bold = EMA)", fontsize=12)',
      "plot title")

# ============================================================================
# 12. shape-smoke cell: add a 4p FFA step check
# ============================================================================
sm = find_cell(lambda s: s.startswith("RUN_SMOKE"), "smoke cell")
patch(sm, '    print("self-play step OK | step_ct", senv.step_ct[0].item())\n    print("SMOKE OK")',
      '''    print("self-play step OK | step_ct", senv.step_ct[0].item())
    # v8: 4-player FFA step (scripted + neural seats)
    senv.reset(sworlds, n_players=4)
    ent, em, am, gl = env_encode(senv, 0)
    a_t, _, _ = act(snet, ent, em, am, gl)
    seats = [{"pid": 1, "script": 1}, {"pid": 2, "script": 4}, {"pid": 3, "action": a_t.clone()}]
    out = env_step(senv, a_t, seats=seats)
    s_all, alive_all = settle_n(senv)
    assert s_all.shape == (4, 4) and alive_all.shape == (4, 4), (s_all.shape, alive_all.shape)
    print("4p FFA step OK | p0..p3 ships", [round(float(x)) for x in s_all[0].tolist()])
    print("SMOKE OK")''',
      "smoke 4p block")

# ============================================================================
# 13. section markdown notes
# ============================================================================
MD_NOTES = {
    "## 6.": "\n\n**v8**: `reset(worlds, n_players)` -- for 4p FFA, owners 0..3 are seated on the "
             "official symmetric home group (`home_base`), 10 ships each, exactly like the engine's "
             "4-player branch.",
    "## 9.": "\n\n**v8**: `env_step(..., seats=[...])` takes any mix of scripted/neural seats for "
             "pids 1..N-1; combat is the official N-player top-vs-second resolve. v7 2p calls "
             "(`env_step(env, a, opp, opp_action)`) still work and are bit-exact.",
    "## 11.": "\n\n**v8**: `collect_ppo(env, net, pool, cursor, rng, seats, n_players)` -- the learner "
              "is always player 0; `stats['seat_scores']` carries the per-seat pairwise scores that "
              "feed the dual-Elo update. 4p outcome is placement-linear by default (`OUTCOME_4P`).",
    "## 13.": "\n\n**v8**: dual-Elo league (2p + 4p ratings, cross-coupled), invited members from the "
              "league folder (anchored, boost-then-drop), weak-point matchmaking (boost members the "
              "learner underperforms its Elo expectation against, up to x3), and the bounded "
              "error-diffusion 2p/4p scheduler (1:1 .. 1:6).",
}
for prefix, note in MD_NOTES.items():
    md = find_cell(lambda s, p=prefix: s.startswith(p), "markdown %s" % prefix)
    set_src(md, cell_src(md).rstrip("\n") + note + "\n")

json.dump(doc, open(DST, "w", encoding="utf-8"), indent=1)
print("wrote", DST, "(%d cells)" % len(doc["cells"]))
