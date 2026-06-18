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
N_SOON = 15
N_BIG = 15
N_THREAT_FLEETS = N_SOON + N_BIG            # 30 slots * 3 feats = 90
N_BODY_FEATURES = 14   # v9: ownership = 4-channel seat one-hot (was 1 scalar) -> +3
N_ENTITY_FEATURES = N_BODY_FEATURES + 3 * N_THREAT_FLEETS   # 104
N_GLOBAL_FEATURES = 10
F_DIM = N_ENTITY_FEATURES
G_DIM = N_GLOBAL_FEATURES

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
