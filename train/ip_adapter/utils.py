"""
train/ip_adapter/utils.py — Shared utilities used across training scripts.

CPU core detection follows the approach in plans/ip-adapter-training.md §2.0:
  - Use sysctl to query performance core count on Apple Silicon
  - Never use os.cpu_count() alone (returns 10 on M1 Max, including E-cores)
  - Leave 1–2 P-cores free for OS + Metal command encoding when GPU is active
"""

import os
import subprocess


def get_performance_core_count() -> int:
    """
    Return the number of performance cores on Apple Silicon.
    Falls back to os.cpu_count() on non-Apple hardware.

    M1 Max: returns 8 (8 P-cores, 2 E-cores, 10 total).
    """
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            text=True,
        ).strip()
        return int(out)
    except Exception:
        return os.cpu_count() or 4


# Pre-computed at import time — used as defaults throughout the pipeline.
PERF_CORES = get_performance_core_count()

# Leave 2 cores free for OS and Metal command encoding when GPU is also active.
COMPUTE_WORKERS = max(1, PERF_CORES - 2)


def worker_counts():
    """
    Return recommended worker counts for each pipeline phase.
    Based on plans/ip-adapter-training.md §2.0 allocation table.

    Returns dict with keys:
      io_bound     — download, shard read (all P-cores)
      compute      — JPEG decode, encode, norm (P-cores minus 2)
      gpu_feed     — CPU workers feeding GPU batches (all P-cores)
      caption      — per-process workers for 2-process Moondream (P-cores // 2)
      io_threads   — img2dataset --thread_count (2× P-cores, I/O threads per worker)
      prefetch     — training prefetch threads (always 2 — GPU is bottleneck)
    """
    return {
        "io_bound":   PERF_CORES,
        "compute":    COMPUTE_WORKERS,
        "gpu_feed":   PERF_CORES,
        "caption":    max(1, PERF_CORES // 2),
        "io_threads": PERF_CORES * 2,
        "prefetch":   2,
    }
