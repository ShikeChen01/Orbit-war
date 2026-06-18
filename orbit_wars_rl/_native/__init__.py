"""Loader for the compiled native core (orbitwars_native).

`import torch` must happen first: it puts torch's DLL dir on the search path so the
extension's torch_cpu.dll / torch_cuda.dll / c10*.dll dependencies resolve on Windows.
"""
import torch  # noqa: F401  (side effect: registers torch DLL dir)

from orbit_wars_rl._native.orbitwars_native import *  # noqa: F401,F403
from orbit_wars_rl._native import orbitwars_native as _ext

__all__ = [n for n in dir(_ext) if not n.startswith("_")]
