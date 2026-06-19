"""End-to-end smoke of the packaged stack (orbit_wars_v12), the port of scripts/smoke_v12_nb.py.

Runs a short SMOKE train() and asserts the v12 invariants:
  - clean rebuild: NO monkeypatch cascade in the package source (_v9*/_orig_/OVERRIDES/weak-boost);
  - the recal path executes at runtime (chain-init + _reground + validate_watchdogs + measured
    eviction) without error;
  - measured eviction semantics via a live unit-test with stubbed eval funcs: an unmastered script
    survives, a mastered ordinary script is removed, a mastered keep-kind watchdog retires to DORMANT;
  - eviction applies the SAME standard to neural league players (snapshots + invited); the rolling
    self-anchor (Elo reference) and the newest snapshot (current self) are protected;
  - OR-gate: mastery in EITHER format alone (2p-only OR 4p-only) evicts; neither survives;
  - shared league invariants (one-hot obs F_DIM 104, mixed 2p/4p, dual Elo, gauntlets) + resume.

    python scripts/smoke_v12_pkg.py
"""
import glob
import os
import sys
import types

os.environ.setdefault("MPLBACKEND", "Agg")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ["OW_CKPT_DIR"] = os.path.join("runs", "v12_pkg_smoke")
os.makedirs(os.environ["OW_CKPT_DIR"], exist_ok=True)
# keep invited discovery out of the smoke (deterministic, fast); the package path is exercised
# separately by the parity test / a real run.
os.environ["OW_LEAGUE_DIR"] = os.path.join(os.environ["OW_CKPT_DIR"], "_noinvite")

import orbit_wars_v12 as ow
from orbit_wars_v12 import Config, constants, league as Lmod

PKG = os.path.join(REPO, "orbit_wars_v12")

# --- static guards: the clean rebuild must NOT carry any of the monkeypatch cascade ---
_allsrc = "\n".join(open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(PKG, "*.py")))
for bad in ["_v9", "_orig_", "League._", "OVERRIDES", "_weak_boost", "evict_dominated", "WEAK_BOOST"]:
    assert bad not in _allsrc, "clean rebuild must not contain %r" % bad
for need in ["def evict_mastered(", "def _eval_winrate4_net(", "def _reground(", "def _ground_ratings(",
             "def validate_watchdogs(", "[evict]", "[recal]"]:
    assert need in _allsrc, "missing expected v12 symbol %r" % need

# --- short SMOKE run (LEARNER_RECAL_EVERY>0 + ELO_RECAL_EVERY so BOTH recal paths run) ---
cfg = Config.create(
    smoke=True, TOTAL_ITERS=8, EPISODE_STEPS=96, N_WORLDS=48, B=8, B_4P=8, FLEET_CAP=256,
    ELO_RECAL_ENVS=8, CKPT_EVERY=3, SELFPLAY_REFRESH=2, LEAGUE_MAX_SNAPSHOTS=3,
    LEARNER_RECAL_EVERY=3, ELO_RECAL_EVERY=4,   # exercise BOTH lite (ground_members=False) + full recal
    VALUE_WARMUP_ITERS=1, LR_WARMUP_ITERS=2,
    INVITE_DIRS=[os.path.join(os.environ["OW_CKPT_DIR"], "_noinvite")],
)
print("[smoke] config: SMOKE B=%d EPISODE_STEPS=%d TOTAL_ITERS=%d" % (cfg.B, cfg.EPISODE_STEPS, cfg.TOTAL_ITERS))
net, hist, league = ow.Trainer(cfg).train()

# --- shared league invariants ---
assert constants.F_DIM == 104, "v12 keeps the one-hot obs (F_DIM 104), got %r" % constants.F_DIM
n2, n4 = hist["fmt"].count(2), hist["fmt"].count(4)
print("\n[smoke] iterations: %d x 2p, %d x 4p" % (n2, n4))
assert n4 >= 1 and n2 >= 1, "expected BOTH formats in the mix"
assert len(hist["learner_elo4"]) == len(hist["learner_elo"]) == cfg.TOTAL_ITERS
assert "gauntlet2" in hist and "gauntlet4" in hist and len(hist["gauntlet4"]) > 0

# the recal path must have actually run (chain-init froze a self-anchor)
assert any(m.get("kind") == Lmod.SELF_KIND for m in league.members), "recal never ran (no self-anchor frozen)"
print("[smoke] recal path ran: %d self-anchor(s) frozen" % len(league._self_anchors()))

# --- v12 measured-eviction semantics (stub the module eval funcs to force mastery) ---
_orig_s2, _orig_w4 = Lmod._eval_score, Lmod._eval_winrate4
_FORCE = {"starter": 0.99, "random": 0.99, "greedy": 0.99, "medium": 0.10}   # mastered except medium
Lmod._eval_score = lambda cfg, net, kind, worlds, n: _FORCE.get(kind, 0.5)
Lmod._eval_winrate4 = lambda cfg, net, kind, worlds, n: _FORCE.get(kind, 0.5)
saved = list(league.members)
m_un = dict(net=None, kind="medium", label="medium", anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0)
m_ma = dict(net=None, kind="starter", label="starter", anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0)
m_wd = dict(net=None, kind="greedy", label="greedy", anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0, wd_active=True)
league.members = [m_un, m_ma, m_wd]
league.evict_mastered(net, None, 8)
assert m_un in league.members, "unmastered script was wrongly evicted (eviction not mastery-gated)"
assert m_ma not in league.members, "mastered ordinary script was NOT removed (measured eviction broken)"
assert m_wd in league.members, "keep-kind watchdog must never be removed"
assert m_wd["wd_active"] is False, "re-mastered active watchdog must retire to DORMANT"
league.members = saved
Lmod._eval_score, Lmod._eval_winrate4 = _orig_s2, _orig_w4
print("[smoke] measured eviction: unmastered survives | mastered ordinary removed | watchdog -> dormant  OK")

# --- SAME standard for neural league players (snapshots + invited); protect self-anchor + newest ---
_orig_n2, _orig_n4 = Lmod._score2_vs, Lmod._eval_winrate4_net
Lmod._score2_vs = lambda cfg, net, opp, worlds, n: getattr(opp, "_evict", 0.5)
Lmod._eval_winrate4_net = lambda cfg, net, opp, worlds, n: getattr(opp, "_evict", 0.5)
_nnet = lambda v: types.SimpleNamespace(_evict=v, to=lambda *a, **k: None)
p_old = dict(net=_nnet(0.99), kind="snapshot", label="it_old", anchor=False, pinned=False, elo=1500.0, elo4=1500.0, n=0, n4=0)
p_new = dict(net=_nnet(0.99), kind="snapshot", label="it_new", anchor=False, pinned=False, elo=1500.0, elo4=1500.0, n=0, n4=0)
p_slf = dict(net=_nnet(0.99), kind=Lmod.SELF_KIND, label="selfA0", anchor=True, pinned=True, elo=1500.0, elo4=1500.0, n=0, n4=0)
p_inv = dict(net=_nnet(0.99), kind="invited", label="inv:x", anchor=True, pinned=True, elo=1500.0, elo4=1500.0, n=0, n4=0)
league.members = [p_old, p_new, p_slf, p_inv]
league.evict_mastered(net, None, 8, neural=True)
assert p_old not in league.members, "mastered league player (snapshot) must be evicted under the same standard"
assert p_new in league.members, "newest snapshot (current self) must be protected from eviction"
assert p_slf in league.members, "self-anchor (Elo reference) must never be evicted"
assert p_inv not in league.members, "mastered invited league player must be evicted"
# neural eviction is GATED OFF on a cheap learner recal (neural=False) -> nothing removed
league.members = [p_old, p_inv]
league.evict_mastered(net, None, 8, neural=False)
assert p_old in league.members and p_inv in league.members, "neural eviction must be skipped when neural=False"
league.members = saved
Lmod._score2_vs, Lmod._eval_winrate4_net = _orig_n2, _orig_n4
print("[smoke] league-player eviction: snapshot+invited removed | self-anchor & newest protected | gated off on lite recal  OK")

# --- v12.1 OR-gate: mastery in EITHER format ALONE must evict ---
_S2 = {"starter": 0.99, "medium": 0.10, "intermediate": 0.10}   # starter mastered 2p-only
_W4 = {"starter": 0.10, "medium": 0.99, "intermediate": 0.10}   # medium mastered 4p-only; intermediate neither
Lmod._eval_score = lambda cfg, net, kind, worlds, n: _S2[kind]
Lmod._eval_winrate4 = lambda cfg, net, kind, worlds, n: _W4[kind]
m_2p = dict(net=None, kind="starter", label="starter", anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0)
m_4p = dict(net=None, kind="medium", label="medium", anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0)
m_no = dict(net=None, kind="intermediate", label="intermediate", anchor=False, elo=1000.0, elo4=1000.0, n=0, n4=0)
league.members = [m_2p, m_4p, m_no]
league.evict_mastered(net, None, 8)
assert m_2p not in league.members, "OR-gate: 2p-only mastery must evict (gate reverted to AND?)"
assert m_4p not in league.members, "OR-gate: 4p-only mastery must evict (gate reverted to AND?)"
assert m_no in league.members, "OR-gate: neither-mastered script must survive"
league.members = saved
Lmod._eval_score, Lmod._eval_winrate4 = _orig_s2, _orig_w4
print("[smoke] OR-gate eviction: 2p-only evicts | 4p-only evicts | neither survives  OK")

# resume round-trip (league state reloads + first recal at resume runs eviction)
net2, hist2, lg2 = ow.Trainer(cfg).train(total_iters=2, resume_from=cfg.TRAIN_STATE_PATH)
assert any(m["kind"] == "snapshot" for m in lg2.members), "snapshots lost across resume"

print("\nSMOKE V12 PKG OK: clean rebuild, recal path runs, measured every-recal eviction, resume round-trip")
