"""Fixed game/feature constants for the v12 stack.

These mirror the official Orbit Wars engine + the C++ core (`core/state.hpp`,
`core/encode.hpp`) and the feature layout the policy is trained on. They are PHYSICS, not
configuration: they never vary between runs. Run-tunable hyperparameters live in
:class:`orbit_wars_v12.config.Config`.

Ported verbatim from `notebooks/setup1_v12_a100.ipynb` cells 3 (DTYPE), 7 (board + feature
layout), and 11 (world-generator group bounds).
"""
import math

import torch

# ---- numeric convention --------------------------------------------------------------------
DTYPE = torch.float32  # integer-in-float32 convention (mirrors the native env)

# ---- board constants (core/state.hpp) ------------------------------------------------------
BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PI = math.pi

# ---- planet / ship structural dims (fixed across every version) ----------------------------
PLANET_CAP = 48   # E: planet slots/env == dest-categorical size (comets omitted)
SHIP_SPEED = 6.0

# ---- feature layout (core/encode.hpp) ------------------------------------------------------
# N_SOON/N_BIG are LEGACY names from the old "15 soonest + 15 biggest" threat split; the encoder now
# takes the TOP-30 inbound fleets BY SIZE (no soonest/biggest split), so these are just a 15+15
# decomposition of the 30 slots and carry no separate meaning.
N_SOON = 15
N_BIG = 15
N_THREAT_FLEETS = N_SOON + N_BIG            # 30 inbound-fleet slots per planet (top-30 by size)
# per-fleet threat descriptor: a 4-channel EGO-RELATIVE seat ONE-HOT owner [self, enemy+1, +2, +3]
# (v12: replaces the single self/enemy SIGN, which collapsed all opponents into one channel in 4p)
# + eta + raw ships (per-fleet ship-log compression removed; un-normalized integer count).
N_THREAT_FEATS = 6
N_BODY_FEATURES = 14   # v9: ownership = 4-channel seat one-hot (was 1 scalar) -> +3
N_ENTITY_FEATURES = N_BODY_FEATURES + N_THREAT_FEATS * N_THREAT_FLEETS   # 14 + 6*30 = 194
N_GLOBAL_FEATURES = 10
F_DIM = N_ENTITY_FEATURES
G_DIM = N_GLOBAL_FEATURES
# Older observation layouts kept playable as league members via checkpoint.LegacyObsAdapter, which
# down-converts the CURRENT obs to the member's layout each forward:
#   F_DIM_BINARY_THREAT (104): v9-v12 single self/enemy fleet SIGN (collapse the 4-ch owner one-hot).
#   F_DIM_SCALAR_OWNER  (101): v7/v8, additionally SCALAR body ownership (collapse the body one-hot).
F_DIM_BINARY_THREAT = N_BODY_FEATURES + 3 * N_THREAT_FLEETS           # 104: 14 body + 30x3 threat
F_DIM_SCALAR_OWNER = (N_BODY_FEATURES - 3) + 3 * N_THREAT_FLEETS      # 101: 11 body + 30x3 threat

SHIP_LOG_DENOM = math.log(1000.0)
DIAG_HALF = math.sqrt(BOARD_SIZE * BOARD_SIZE + BOARD_SIZE * BOARD_SIZE) / 2.0
THREAT_MAX_SPEED = 6.0      # == ship speed; fleet speed cap for the ETA model
THREAT_ETA_SCALE = 100.0
BIG = 1e18

# ---- world-generator group bounds (REFERENCE_orbit_wars.py::generate_planets) --------------
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
MIN_STATIC_GROUPS = 3
PLANET_CLEARANCE = 7


def ship_log_t(x):
    return torch.log1p(x.clamp_min(0.0)) / SHIP_LOG_DENOM


def fleet_speed_t(ships, vmax=SHIP_SPEED):
    # 1 + (vmax-1)*(log(ships)/log(1000))^1.5, capped, ships>=1  (core/encode.hpp)
    n = ships.clamp_min(1.0)
    v = 1.0 + (vmax - 1.0) * torch.pow(torch.log(n) / math.log(1000.0), 1.5)
    return v.clamp_max(vmax)
