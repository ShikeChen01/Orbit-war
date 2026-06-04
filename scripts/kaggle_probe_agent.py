"""Hardware-probe agent for the Kaggle Orbit Wars eval box.

This is a throwaway *measurement* submission, not a playing agent. On its first
turn it characterises whatever machine the match-runner put it on — OS/platform,
schedulable CPU cores, GPU presence (``nvidia-smi`` + torch device enumeration),
an explicit "move a small net onto the GPU" attempt, achieved FLOP/s, and memory
bandwidth (benchmarked inside a ~0.9 s window so the numbers reflect what one real
inference turn can buy) — prints everything to **stderr**, and then no-ops for the
rest of the episode (returns ``[]`` every turn).

Why this exists: the Kaggle CLI exposes no hardware spec, and a notebook GPU
benchmark measures the *training* box, not the ladder match-runner. The only way
to see the inference hardware is to run code on it — i.e. submit an agent. After
submitting, open the submission's validation episode and read the agent logs; the
lines below are prefixed ``PROBE|`` so they are trivial to grep out.

Submit with:
    kaggle competitions submit -c orbit-wars -f scripts/kaggle_probe_agent.py -m "hw probe"

Self-contained on purpose (single file, stdlib + torch/numpy only) so it works as
a submission with no repo on the eval box.
"""
import os
import sys
import time

# Total wall-clock we let the probe spend on its first turn. The per-turn budget
# is actTimeout=1s plus a 60s episode overage bank; we stay well under 1s so the
# measurement reflects a *normal* turn (no overage spent) and never risks TIMEOUT.
PROBE_BUDGET_S = 0.9

_done = False


def _log(msg: str) -> None:
    print(f"PROBE| {msg}", file=sys.stderr, flush=True)


def _sh(cmd, timeout=4.0):
    """Run a command, return (rc, combined output); never raises (probe-safe)."""
    import subprocess
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def _sys_env() -> None:
    """Hardware/OS facts that need no torch: platform, the cores actually
    schedulable here, GPU-related env vars, and the canonical NVIDIA check
    (`nvidia-smi`). Absence/error from nvidia-smi is strong evidence of a
    genuinely CPU-only box, independent of whatever torch reports."""
    import platform
    _log(f"platform={platform.platform()} machine={platform.machine()} "
         f"python={platform.python_version()}")
    try:  # containers often cap the schedulable set well below cpu_count
        aff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "n/a"
    except Exception:  # noqa: BLE001
        aff = "n/a"
    _log(f"cpu_count={os.cpu_count()} schedulable_cores={aff}")
    gpu_keys = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
                "NVIDIA_DRIVER_CAPABILITIES", "CUDA_HOME", "CUDA_PATH",
                "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DOCKER_IMAGE")
    _log(f"gpu_env={ {k: os.environ[k] for k in gpu_keys if k in os.environ} }")
    rc, out = _sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                   "--format=csv,noheader"])
    _log(f"nvidia-smi(rc={rc}): {out[:300]!r}")
    try:  # Linux kernel driver node — present iff a real NVIDIA driver is loaded
        with open("/proc/driver/nvidia/version") as f:
            _log(f"/proc/driver/nvidia/version: {f.read().strip()[:200]!r}")
    except Exception:  # noqa: BLE001
        pass


def _try_gpu_net(torch) -> None:
    """Directly answer 'can we actually put a net on the GPU?' — independent of
    is_available(): build a tiny module, move it to cuda, run a forward, and
    report success or the *exact* exception (the most decisive evidence)."""
    import torch.nn as nn
    try:
        net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 8)).to("cuda")
        x = torch.randn(32, 64, device="cuda")
        y = net(x)
        torch.cuda.synchronize()
        _log(f"gpu_net_test: OK ran {tuple(x.shape)}->{tuple(y.shape)} on {y.device}")
    except Exception as e:  # noqa: BLE001
        _log(f"gpu_net_test: cuda UNUSABLE -> {type(e).__name__}: {e}")


def _bench_torch(budget_s: float) -> None:
    import torch

    _sys_env()
    cuda = torch.cuda.is_available()
    if cuda:
        # is_available() can be True on a box whose device is still unusable
        # (CUDA_VISIBLE_DEVICES="", driver/runtime mismatch). Verify with a real
        # allocation so the probe falls back to CPU instead of crashing out of
        # measurement entirely — it must always produce *some* FLOP number.
        try:
            torch.zeros(1, device="cuda")
        except Exception as e:  # noqa: BLE001
            _log(f"cuda is_available but UNUSABLE ({type(e).__name__}: {e}); using cpu")
            cuda = False
    dev = torch.device("cuda" if cuda else "cpu")
    _log(f"torch={torch.__version__} cuda_available={cuda} device={dev}")
    _log(f"cpu_count={os.cpu_count()} torch_threads={torch.get_num_threads()}")

    if not cuda:
        # Disambiguate "no GPU hardware" from "GPU present but torch can't use it".
        # device_count==0 -> genuinely CPU-only box; >0 -> a driver/runtime/visibility
        # issue; CUDA_VISIBLE_DEVICES=="" -> the runner hid the GPU.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        try:
            ndev = torch.cuda.device_count()
        except Exception as e:  # noqa: BLE001
            ndev = f"err:{type(e).__name__}:{e}"
        _log(f"no-cuda diag: device_count={ndev} CUDA_VISIBLE_DEVICES={cvd!r} "
             f"torch_cuda_build={torch.version.cuda}")

    if cuda:
        try:
            i = torch.cuda.current_device()
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            free, total = torch.cuda.mem_get_info(i)
            _log(f"gpu_name={name!r} capability={cap[0]}.{cap[1]} "
                 f"cuda_runtime={torch.version.cuda} "
                 f"mem_free_GB={free/1e9:.2f} mem_total_GB={total/1e9:.2f}")
        except Exception as e:  # noqa: BLE001 - metadata must not abort the benchmark
            _log(f"gpu info unavailable ({type(e).__name__}: {e})")

    _try_gpu_net(torch)  # explicit "send a net to the GPU" probe (runs either way)

    def sync():
        if cuda:
            torch.cuda.synchronize()

    # ---- FLOP/s via square matmul: a (NxN)@(NxN) is 2*N^3 FLOPs ----------
    # We calibrate one synced iteration, then run a *fixed* iter count sized to
    # the budget and sync once. Crucial under CUDA: a bare while-loop enqueues
    # kernels async (in-loop timer reads low, trailing sync balloons), which can
    # blow the deadline. Calibration bounds wall time on any device.
    def flops(n: int, dtype, label: str, allow_tf32: bool = False) -> None:
        try:
            if cuda:
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
                torch.backends.cudnn.allow_tf32 = allow_tf32
            a = torch.randn(n, n, device=dev, dtype=dtype)
            b = torch.randn(n, n, device=dev, dtype=dtype)
            for _ in range(3):  # warmup: alloc / cuInit / kernel-load excluded
                c = a @ b
            sync()
            t0 = time.perf_counter()  # calibrate: one synced matmul
            c = a @ b
            sync()
            t1 = max(time.perf_counter() - t0, 1e-6)
            iters = max(1, int((budget_s / 3.0) / t1))  # ~1/3 budget per dtype
            t0 = time.perf_counter()
            for _ in range(iters):
                c = a @ b
            sync()
            dt = time.perf_counter() - t0
            tflops = (2.0 * n**3 * iters) / dt / 1e12
            _log(f"matmul {label} N={n} iters={iters} dt={dt:.3f}s "
                 f"-> {tflops:.2f} TFLOP/s ({tflops*1e12:.3e} FLOP/s)")
            del a, b, c
            if cuda:
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 - probe must never crash the agent
            _log(f"matmul {label} FAILED: {type(e).__name__}: {e}")

    n = 4096 if cuda else 1024
    flops(n, torch.float32, "fp32")
    if cuda:
        flops(n, torch.float32, "tf32", allow_tf32=True)
        flops(n, torch.float16, "fp16")

    # ---- memory bandwidth via a read+write copy (2 bytes touched / elem) ---
    try:
        nbytes = 256 * 1024 * 1024  # 256 MB working set
        x = torch.empty(nbytes // 4, device=dev, dtype=torch.float32)
        y = torch.empty_like(x)
        y.copy_(x)
        sync()
        t0 = time.perf_counter()  # calibrate one synced copy, then fixed reps
        y.copy_(x)
        sync()
        t1 = max(time.perf_counter() - t0, 1e-6)
        reps = max(1, int((budget_s / 4.0) / t1))
        t0 = time.perf_counter()
        for _ in range(reps):
            y.copy_(x)
        sync()
        dt = time.perf_counter() - t0
        gbs = (x.numel() * 4 * 2 * reps) / dt / 1e9  # read + write
        _log(f"copy_bandwidth reps={reps} dt={dt:.3f}s -> {gbs:.1f} GB/s")
    except Exception as e:  # noqa: BLE001
        _log(f"bandwidth FAILED: {type(e).__name__}: {e}")


def _bench_numpy(budget_s: float) -> None:
    import numpy as np

    _sys_env()
    _log(f"torch=UNAVAILABLE numpy={np.__version__} cpu_count={os.cpu_count()}")
    n = 1024
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    _ = a @ b
    t0 = time.perf_counter()
    iters = 0
    while time.perf_counter() - t0 < budget_s / 2.0:
        _ = a @ b
        iters += 1
    dt = time.perf_counter() - t0
    tflops = (2.0 * n**3 * iters) / dt / 1e12
    _log(f"numpy matmul N={n} iters={iters} dt={dt:.3f}s -> {tflops:.3f} TFLOP/s")


def agent(obs, config):
    """kaggle_environments entrypoint: benchmark once, then no-op forever."""
    global _done
    if not _done:
        _done = True
        try:
            try:
                import torch  # noqa: F401
                _bench_torch(PROBE_BUDGET_S)
            except ImportError:
                _bench_numpy(PROBE_BUDGET_S)
        except Exception as e:  # noqa: BLE001 - never let the probe error the match
            _log(f"FATAL {type(e).__name__}: {e}")
        _log("done")
    return []  # no-op move (valid empty action)


if __name__ == "__main__":
    # Local dry-run: prints what the probe would report on THIS machine.
    agent({}, {})
