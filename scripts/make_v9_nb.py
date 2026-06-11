"""Generate notebooks/setup1_v9_a100.ipynb -- v8 + per-seat ONE-HOT ownership (v9).

Layered on notebooks/setup1_v8_a100.ipynb (the LEGACY-COMPATIBLE mixed 2p/4p league build:
collapsed everyone-vs-me ownership, F_DIM 101, v7 weights drop in unchanged). v9 changes ONLY
the observation:

  - ownership scalar -> 4-channel EGO-RELATIVE seat ONE-HOT [self, enemy+1, enemy+2, enemy+3]
    (neutral = all-zero): the policy can tell WHICH opponent owns a body, with no ordinal
    fake-magnitude. In 2p only the first two channels ever fire, so 4p generalizes the same
    input space; a dead seat's channel simply stops firing.
  - N_BODY_FEATURES 11 -> 14, so F_DIM 101 -> 104 (fresh BC + training required; v7/v8-format
    weights do NOT load as the learner).
  - the league's LegacyObsAdapter (inherited dormant from v8) ACTIVATES here: invited/seeded
    v7/v8-format members (F_DIM 101) are auto-wrapped, their obs collapsed back to the scalar
    code they were trained on.

2p dynamics stay bit-exact to v8 (scripts/test_v9_parity.py). Run `make_v8_nb.py` first when
regenerating both.

    python scripts/make_v9_nb.py
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "notebooks", "setup1_v8_a100.ipynb")
DST = os.path.join(REPO, "notebooks", "setup1_v9_a100.ipynb")

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
hdr = find_cell(lambda s: s.startswith("# Orbit Wars"), "header markdown")
set_src(hdr, cell_src(hdr).replace("# Orbit Wars -- v8 A100 run (mixed 2p/4p league, dual Elo, invited members)",
                                   "# Orbit Wars -- v9 A100 run (v8 league + per-seat ONE-HOT ownership)", 1)
        + """

**v9 = v8 + the categorical ownership redesign.** Ownership is a 4-channel EGO-RELATIVE seat
one-hot `[self, enemy seat+1, seat+2, seat+3]` (neutral = all-zero) instead of the collapsed
scalar: the policy can tell WHICH opponent owns each body and learns per-enemy behavior; in 2p
the last two channels never fire, so the 4p input space is a strict superset (a dead seat's
channel just stops firing). F_DIM 101 -> 104: the learner needs FRESH BC + training; v7/v8-format
checkpoints still join the league as opponents through the auto-activating `LegacyObsAdapter`.
**Deploy note:** the Kaggle 4p adapter for v9 must PRESERVE seats (one owner id per opponent,
n_players=4) -- collapsing to a single enemy id would discard the trained seat-awareness.
""")

# ============================================================================
# 1. feature layout: ownership scalar -> 4-channel seat one-hot (F_DIM 101 -> 104)
# ============================================================================
fl = find_cell(lambda s: "N_BODY_FEATURES = " in s, "feature-layout cell")
patch(fl, "N_BODY_FEATURES = 11\n",
      "N_BODY_FEATURES = 14   # v9: ownership = 4-channel seat one-hot (was 1 scalar) -> +3\n",
      "feature layout N_BODY")

# ============================================================================
# 2. env_encode: collapsed scalar -> per-seat one-hot
# ============================================================================
ec = find_cell(lambda s: "def env_encode" in s, "encode cell")
patch(ec, '''def env_encode(env, ego=0):
    """v8: N-player aware. EVERY non-ego owner collapses to the single 'enemy' code and the
    enemy aggregates pool ALL opponents -- identical encodings in 2p, and exactly the deployed
    Kaggle 4p adapter view in FFA (the policy plays everyone-vs-me). v7 weights stay drop-in
    compatible (F_DIM 101); the per-seat one-hot redesign is v9."""
    B, Ec, Fc = env.B, env.Ec, env.Fc
''',
      '''def env_encode(env, ego=0):
    """v9: ownership is a 4-channel EGO-RELATIVE seat ONE-HOT [self, enemy seat+1, seat+2,
    seat+3] (neutral = all-zero), so the policy can tell WHICH opponent owns a body with no
    ordinal fake-magnitude. In 2p only the first two channels ever fire, so 4p generalizes the
    same input space; a dead seat's channel simply stops firing. The GLOBAL enemy aggregates
    still pool ALL opponents (everyone-vs-me totals)."""
    B, Ec, Fc = env.B, env.Ec, env.Fc
    na = float(getattr(env, "n_players", 2))
''',
      "encode prologue")
patch(ec, """    # ownership code: 0 neutral, 1 ego, 2 ANY other player (2p-identical; 4p collapses)
    own_code = torch.where(owner < 0.0, torch.zeros_like(owner),
                           torch.where(owner == float(ego), torch.ones_like(owner),
                                       torch.full_like(owner, 2.0)))
""",
      """    # ego-relative seat ONE-HOT: rel = (owner - ego) mod n_players -> channel rel fires;
    # neutral = all-zero. Channel 0 = self; channels 1..3 = the other seats in seat order
    # (rotation-equivariant: the same relative seat always lands in the same channel).
    rel = owner - float(ego)
    rel = rel - na * torch.floor(rel / na)
    _owned = owner >= 0.0
    own_oh = [(_owned & (rel == float(k))).to(DTYPE) for k in range(4)]
""",
      "encode own one-hot")
patch(ec, "    b7 = own_code\n",
      "    b7a, b7b, b7c, b7d = own_oh   # [self, enemy+1, enemy+2, enemy+3]; neutral all-zero\n",
      "encode b7 channels")
patch(ec, "    body = torch.stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10], 2)  # (B,Ec,11)\n",
      "    body = torch.stack([b0, b1, b2, b3, b4, b5, b6, b7a, b7b, b7c, b7d, b8, b9, b10], 2)  # (B,Ec,14)\n",
      "encode body stack")

# ============================================================================
# 3. cosmetics: plot title
# ============================================================================
pl = find_cell(lambda s: "Training-health curves" in s, "plot cell")
patch(pl, 'fig.suptitle("Orbit Wars v8 -- training health', 'fig.suptitle("Orbit Wars v9 -- training health',
      "plot title")

json.dump(doc, open(DST, "w", encoding="utf-8"), indent=1)
print("wrote", DST, "(%d cells)" % len(doc["cells"]))
