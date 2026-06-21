"""Model (de)serialization, invited-member discovery, and resumable train state.

Ported from notebook cells 28 (snapshot helpers + invited discovery) and 30 (save_ckpt /
save_train_state / load_train_state). The checkpoint ``config`` blob keeps UPPER keys so existing
``.pt`` files round-trip; per-member dims are read back via :func:`build_from_cfg` so league
snapshots and (smaller) invited members rebuild at their own sizes. ``ACT_FN`` defaults to
``relu`` for pre-tanh checkpoints.
"""
import copy
import glob
import os
import re

import torch

from .constants import (F_DIM, F_DIM_BINARY_THREAT, F_DIM_NO_MOTION, F_DIM_SCALAR_OWNER, G_DIM,
                        N_BODY_FEATURES, N_BODY_FEATURES_V12, N_THREAT_FEATS, N_THREAT_FLEETS, PLANET_CAP)
from .policy import PolicyNet, build_policy
from .worldgen import make_world_pool


def _freeze_snapshot(net):
    """Move to CPU, eval mode, no grad -> a fixed sparring partner."""
    net = net.cpu().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class LegacyObsAdapter(torch.nn.Module):
    """Wraps a league member saved with an OLDER observation layout so it still plays inside the
    current build: each forward DOWN-CONVERTS the current obs to the member's layout.

    Current build (F_DIM=198): 18 body = 14 v12 channels (4-ch seat one-hot ownership, ...) + 4 v13
    MOTION channels (vx,vy,ax,ay), then 30 inbound fleets x 6 (4-ch fleet seat one-hot + eta + raw
    ships). EVERY supported older member first DROPS the 4 motion channels (they never had them):
      * F_DIM_NO_MOTION (194): pre-v13 v12 -> drop motion only; 6-feat threat passes through unchanged.
      * F_DIM_BINARY_THREAT (104): member used a single self/enemy SIGN per fleet -> collapse each
        fleet's 4-ch owner one-hot to that sign (+1 self / -1 enemy / 0 empty) -> 30 x 3.
      * F_DIM_SCALAR_OWNER (101): member also used SCALAR body ownership -> additionally collapse the
        body 4-ch one-hot to the scalar code (1 self, 2 ANY enemy, 0 neutral).
    DORMANT (the net is returned unwrapped) whenever the member's F_DIM matches the current build."""
    def __init__(self, net, member_fdim):
        super().__init__()
        self.net = net
        self.member_fdim = int(member_fdim)

    def forward(self, ent, em, am, gl):
        pre = ent.shape[:-1]
        body = ent[..., :N_BODY_FEATURES_V12]                                        # drop v13 motion -> (...,14)
        thr = ent[..., N_BODY_FEATURES:].reshape(*pre, N_THREAT_FLEETS, N_THREAT_FEATS)   # 6-feat threat (...,30,6)
        if self.member_fdim == F_DIM_NO_MOTION:                                      # pre-v13 v12: threat unchanged
            return self.net(torch.cat([body, thr.reshape(*pre, N_THREAT_FEATS * N_THREAT_FLEETS)], -1), em, am, gl)
        sign = thr[..., 0:1] - thr[..., 1:4].sum(-1, keepdim=True)                   # +1 self / -1 enemy / 0 empty
        thr_old = torch.cat([sign, thr[..., 4:6]], -1).reshape(*pre, 3 * N_THREAT_FLEETS)   # (...,90)
        if self.member_fdim == F_DIM_SCALAR_OWNER:                                   # collapse body ownership too
            b7 = body[..., 7:8] + 2.0 * body[..., 8:11].sum(-1, keepdim=True).clamp(max=1.0)
            body = torch.cat([body[..., :7], b7, body[..., 11:]], -1)               # (...,11)
        return self.net(torch.cat([body, thr_old], -1), em, am, gl)


def _unwrap(net):
    return net.net if isinstance(net, LegacyObsAdapter) else net


def _wrap_if_legacy(net, member_cfg):
    """Current-format members pass through; supported older obs layouts get the down-converting
    adapter; anything else is incompatible."""
    fd = int((member_cfg or {}).get("F_DIM", F_DIM))
    if fd == F_DIM:
        return net
    if fd in (F_DIM_NO_MOTION, F_DIM_BINARY_THREAT, F_DIM_SCALAR_OWNER):
        return LegacyObsAdapter(net, fd)
    raise ValueError("incompatible member F_DIM %d (current %d)" % (fd, F_DIM))


_POLICY_KW = ("HIDDEN", "D_G", "N_RES_BLOCKS", "USE_GLU", "USE_ATTENTION", "VALUE_RES_BLOCKS",
              "ARCH", "N_HEADS", "N_TX_LAYERS", "TX_MLP_RATIO", "N_STEM_RES", "N_HEAD_RES",
              "TRUNK_SPEC", "F_DIM")


def _live(cfg, k):
    """The current build's value for a _POLICY_KW key (F_DIM is a constant; the rest are on cfg)."""
    return F_DIM if k == "F_DIM" else getattr(cfg, k)


def _cfg_from(cfg, blob_cfg, fallback=None):
    """Per-member PolicyNet dims: ckpt config > fallback (e.g. INVITE_CFG) > current build (cfg)."""
    out = {}
    for k in _POLICY_KW:
        if blob_cfg and k in blob_cfg:
            out[k] = blob_cfg[k]
        elif fallback and k in fallback:
            out[k] = fallback[k]
        else:
            out[k] = _live(cfg, k)
    # activation: missing in OLD checkpoints -> they were trained with relu, so rebuild as relu
    out["ACT_FN"] = (blob_cfg or {}).get("ACT_FN", (fallback or {}).get("ACT_FN", "relu"))
    # aux reward head: an ARCHITECTURE-changing flag absent from pre-aux blobs. Unlike the dims above it
    # must NOT fall back to the live cfg -- that would build a head the member's .pt has no weights for and
    # break the strict load. Default OFF so pre-aux snapshots (and invited members) rebuild unchanged.
    out["AUX_REWARD_PRED"] = bool((blob_cfg or {}).get("AUX_REWARD_PRED",
                                                       (fallback or {}).get("AUX_REWARD_PRED", False)))
    out["AUX_WIN_BET"] = bool((blob_cfg or {}).get("AUX_WIN_BET",
                                                   (fallback or {}).get("AUX_WIN_BET", False)))
    return out


def build_from_cfg(cfg, member_cfg):
    return PolicyNet(cfg, int(member_cfg.get("F_DIM", F_DIM)), G_DIM, member_cfg["HIDDEN"],
                     member_cfg["D_G"], PLANET_CAP, member_cfg["N_RES_BLOCKS"], member_cfg["USE_GLU"],
                     cfg.INIT_GATE_BIAS, use_attention=member_cfg["USE_ATTENTION"],
                     value_res_blocks=member_cfg["VALUE_RES_BLOCKS"], arch=member_cfg["ARCH"],
                     n_heads=member_cfg["N_HEADS"], n_tx_layers=member_cfg["N_TX_LAYERS"],
                     tx_mlp_ratio=member_cfg["TX_MLP_RATIO"], n_stem_res=member_cfg["N_STEM_RES"],
                     n_head_res=member_cfg["N_HEAD_RES"], act=member_cfg.get("ACT_FN", "relu"),
                     aux_reward_pred=member_cfg.get("AUX_REWARD_PRED", False),
                     aux_win_bet=member_cfg.get("AUX_WIN_BET", False),
                     trunk_spec=member_cfg.get("TRUNK_SPEC", ""))


def load_snapshot(cfg, path, fallback_cfg=None):
    """Load a save_ckpt() .pt into a frozen PolicyNet on CPU. Returns (net, member_cfg) -- the
    member_cfg is stored on the league member so resumable states can rebuild differently-sized
    (invited) nets."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    member_cfg = _cfg_from(cfg, blob.get("config", {}), fallback_cfg)
    net = build_from_cfg(cfg, member_cfg)
    net.load_state_dict(blob["model"])
    return _freeze_snapshot(_wrap_if_legacy(net, member_cfg)), member_cfg


_ELO_RE = re.compile(r"_elo(-?\d+)", re.IGNORECASE)


def _invited_elo(cfg, path):
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
    return float(cfg.INVITE_DEFAULT_ELO)


def discover_invited(cfg):
    """Scan the FIRST existing INVITE_DIRS folder for *.pt members; return the top INVITE_MAX
    [(path, anchored_elo)] sorted by Elo (desc)."""
    for d in cfg.INVITE_DIRS:
        if d and os.path.isdir(d):
            cands = sorted(glob.glob(os.path.join(d, "*.pt")))
            if cands:
                ranked = sorted(((p, _invited_elo(cfg, p)) for p in cands), key=lambda t: -t[1])
                print("[league] invite dir %s: %d candidate(s) -> inviting top %d"
                      % (d, len(ranked), min(cfg.INVITE_MAX, len(ranked))))
                return ranked[:cfg.INVITE_MAX]
    print("[league] no invite dir found (searched: %s)" % ", ".join(x for x in cfg.INVITE_DIRS if x))
    return []


def save_ckpt(cfg, net, path, meta=None):
    blob = {"model": net.state_dict(),
            "config": {"HIDDEN": cfg.HIDDEN, "N_RES_BLOCKS": cfg.N_RES_BLOCKS, "D_G": cfg.D_G,
                       "PLANET_CAP": PLANET_CAP, "F_DIM": F_DIM, "G_DIM": G_DIM,
                       "ARCH": cfg.ARCH, "N_TX_LAYERS": cfg.N_TX_LAYERS, "N_HEADS": cfg.N_HEADS,
                       "TX_MLP_RATIO": cfg.TX_MLP_RATIO, "N_STEM_RES": cfg.N_STEM_RES, "N_HEAD_RES": cfg.N_HEAD_RES,
                       "TRUNK_SPEC": getattr(cfg, "TRUNK_SPEC", ""),   # [blockseq] rebuild the res/attn sequence
                       "USE_GLU": cfg.USE_GLU, "USE_ATTENTION": cfg.USE_ATTENTION, "VALUE_RES_BLOCKS": cfg.VALUE_RES_BLOCKS,
                       "ACT_FN": cfg.ACT_FN, "target_actor": True,
                       # persist the ACTUAL net's head state (not cfg) -- members built without the head
                       # (invited / legacy-wrapped) must round-trip as AUX_REWARD_PRED=False.
                       "AUX_REWARD_PRED": bool(getattr(net, "aux_reward_pred", False)),
                       "AUX_WIN_BET": bool(getattr(net, "aux_win_bet", False))}}
    if meta:
        blob.update(meta)
    # league snapshots can be SMALLER than the live config (invited dims): persist the true dims
    if hasattr(net, "trunk") and hasattr(net.trunk, "layers"):
        blob["config"]["HIDDEN"] = net.h
        blob["config"]["N_TX_LAYERS"] = len(net.trunk.layers)
        blob["config"]["N_HEADS"] = net.trunk.layers[0].nh if net.trunk.layers else cfg.N_HEADS
        blob["config"]["N_STEM_RES"] = len(net.trunk.stem_a)
        blob["config"]["N_HEAD_RES"] = len(net.trunk.head_a)
    torch.save(blob, path)


def save_train_state(net, opt, league, global_iter, best_wr, path):
    """Full resumable state: weights + Adam moments + global iter + best score + dual-Elo league."""
    pa = getattr(net, "popart", None)
    torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "league": league.state_dict(),
                "global_iter": int(global_iter), "best_wr": float(best_wr),
                "popart": ({"mu": pa.mu, "nu": pa.nu, "sigma": pa.sigma, "init": pa.init} if pa else None)}, path)


def load_train_state(cfg, net, opt, league, path):
    """Restore net/opt/league IN PLACE. Returns (start_iter, best_wr). Accepts a full train-state
    OR a plain save_ckpt weight (warm-start: weights only, fresh Adam/league). v7 states load too
    (missing elo4 fields default to the 2p values); invited members are re-discovered if absent."""
    blob = torch.load(path, map_location=cfg.device, weights_only=False)
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
            for p, aelo in discover_invited(cfg):
                try:
                    league.add_invited(p, aelo)
                except Exception as e:
                    print("  !! invite failed %s -> %r" % (p, e))
    else:  # warm-start: calibrate learner Elo vs anchors so the first snapshot is not stamped Elo 0
        rnd = (start_it // cfg.WORLD_RESAMPLE_EVERY) if cfg.WORLD_RESAMPLE_EVERY > 0 else 0
        print("  [resume] no league state -> calibrating learner Elo vs anchors")
        league.recalibrate_elo(net, make_world_pool(cfg, cfg.ELO_RECAL_ENVS, base_seed=cfg.SEED + rnd * cfg.N_WORLDS))
        league.learner_elo4 = league.learner_elo
    return start_it, float(blob.get("best_wr", -1.0))
