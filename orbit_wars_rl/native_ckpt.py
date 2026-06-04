"""Pure-Python reader/writer for the native .owc checkpoint format.

Mirrors native/io/serialize.hpp (magic 'OWC1'). Uses only stdlib + numpy so the Kaggle
submission -- which runs without the compiled extension -- can load a trained policy. Keep
the byte layout in sync with the C++ codec.

Layout (little-endian):
  magic 'OWC1'
  int32 n_entity_features, n_global_features, hidden, angle_bins, max_entities,
        num_fracs, episode_steps
  uint8 target_mode
  int32 num_fractions ; float64 fractions[num_fractions]
  int32 num_tensors ; per tensor: int32 name_len, name bytes, int32 ndim,
        int32 dims[ndim], float32 data[prod(dims)]  (row-major)

Helpers `owc_to_pt` / `pt_to_owc` convert between this and the legacy torch.save format
({"policy_state", "policy_config"}) used by the arena/search/submission tooling.
"""
from __future__ import annotations

import struct
import sys

import numpy as np

_MAGIC = b"OWC1"
_DEFAULT_FRACTIONS = [0.25, 0.5, 0.75, 1.0]
_MAX_ENTITIES = 64


def read_owc(path: str):
    """Return (state_dict: {name: np.ndarray float32}, meta: dict)."""
    with open(path, "rb") as fh:
        buf = fh.read()
    off = 0

    def take(fmt: str):
        nonlocal off
        vals = struct.unpack_from(fmt, buf, off)
        off += struct.calcsize(fmt)
        return vals

    if buf[off:off + 4] != _MAGIC:
        raise ValueError(f"bad magic {buf[:4]!r}, expected {_MAGIC!r}")
    off += 4
    nef, ngf, hidden, angle_bins, max_entities, num_fracs, episode_steps = take("<7i")
    (target_mode,) = take("<B")
    (num_fr,) = take("<i")
    fractions = list(take("<%dd" % num_fr)) if num_fr else []
    (num_tensors,) = take("<i")
    sd: dict[str, np.ndarray] = {}
    for _ in range(num_tensors):
        (nl,) = take("<i")
        name = buf[off:off + nl].decode("utf-8")
        off += nl
        (ndim,) = take("<i")
        dims = list(take("<%di" % ndim)) if ndim else []
        numel = int(np.prod(dims)) if dims else 1
        arr = np.frombuffer(buf, dtype="<f4", count=numel, offset=off).reshape(dims).copy()
        off += numel * 4
        sd[name] = arr
    meta = dict(n_entity_features=nef, n_global_features=ngf, hidden=hidden,
                angle_bins=angle_bins, max_entities=max_entities, num_fracs=num_fracs,
                episode_steps=episode_steps, target_mode=bool(target_mode), fractions=fractions)
    return sd, meta


def write_owc(path: str, state_dict, meta: dict) -> None:
    """state_dict: {name: array-like float32}. meta keys mirror read_owc's meta dict."""
    fr = list(meta.get("fractions") or _DEFAULT_FRACTIONS)
    with open(path, "wb") as fh:
        fh.write(_MAGIC)
        fh.write(struct.pack("<7i", meta["n_entity_features"], meta["n_global_features"],
                             meta["hidden"], meta["angle_bins"], meta["max_entities"],
                             meta["num_fracs"], meta.get("episode_steps", 500)))
        fh.write(struct.pack("<B", 1 if meta["target_mode"] else 0))
        fh.write(struct.pack("<i", len(fr)))
        if fr:
            fh.write(struct.pack("<%dd" % len(fr), *fr))
        items = list(state_dict.items())
        fh.write(struct.pack("<i", len(items)))
        for name, arr in items:
            arr = np.ascontiguousarray(arr, dtype="<f4")
            nb = name.encode("utf-8")
            fh.write(struct.pack("<i", len(nb)))
            fh.write(nb)
            fh.write(struct.pack("<i", arr.ndim))
            if arr.ndim:
                fh.write(struct.pack("<%di" % arr.ndim, *arr.shape))
            fh.write(arr.tobytes())


def _meta_to_policy_config(meta: dict) -> dict:
    choices = meta["max_entities"] if meta["target_mode"] else meta["angle_bins"]
    return dict(
        n_entity_features=meta["n_entity_features"],
        n_global_features=meta["n_global_features"],
        actions_per_entity=1 + choices * meta["num_fracs"],
        hidden=meta["hidden"],
        target_mode=meta["target_mode"],
        num_fracs=meta["num_fracs"],
    )


def owc_to_pt(owc_path: str, pt_path: str) -> str:
    """Convert a native checkpoint to the legacy torch.save format the Python tooling reads."""
    import torch
    from orbit_wars_rl.agents.ppo_policy import EntityPolicy

    sd, meta = read_owc(owc_path)
    cfg = _meta_to_policy_config(meta)
    policy = EntityPolicy(**cfg)
    ref = policy.state_dict()
    new_sd = {k: torch.from_numpy(np.ascontiguousarray(sd[k])).reshape(ref[k].shape) for k in ref}
    policy.load_state_dict(new_sd)
    torch.save({"policy_state": policy.state_dict(), "policy_config": cfg}, pt_path)
    return pt_path


def pt_to_owc(pt_path: str, owc_path: str, angle_bins: int = 16,
              episode_steps: int = 500, fractions=None) -> str:
    """Convert a legacy .pt checkpoint (e.g. a BC warm-start) to .owc for train.exe."""
    import torch

    ck = torch.load(pt_path, map_location="cpu")
    state = ck["policy_state"]
    cfg = ck["policy_config"]
    num_fracs = cfg["num_fracs"]
    target_mode = bool(cfg["target_mode"])
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state.items()}
    meta = dict(
        n_entity_features=cfg["n_entity_features"],
        n_global_features=cfg["n_global_features"],
        hidden=cfg["hidden"],
        # In angle mode the action dim encodes angle_bins; in target mode it encodes
        # max_entities (=64), so angle_bins is informational only.
        angle_bins=((cfg["actions_per_entity"] - 1) // num_fracs) if not target_mode else angle_bins,
        max_entities=_MAX_ENTITIES,
        num_fracs=num_fracs,
        episode_steps=episode_steps,
        target_mode=target_mode,
        fractions=list(fractions or _DEFAULT_FRACTIONS),
    )
    write_owc(owc_path, sd, meta)
    return owc_path


def _main(argv) -> int:
    usage = "usage: python -m orbit_wars_rl.native_ckpt {to-pt|to-owc} <in> <out>"
    if len(argv) != 4 or argv[1] not in ("to-pt", "to-owc"):
        print(usage)
        return 2
    if argv[1] == "to-pt":
        print("wrote", owc_to_pt(argv[2], argv[3]))
    else:
        print("wrote", pt_to_owc(argv[2], argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
