#!/usr/bin/env python3
"""
train/train_ip_adapter.py — Main IP-Adapter training loop for Flux Klein 4B.

Usage:
    source train/.venv/bin/activate
    python train/train_ip_adapter.py --config train/configs/stage1_512px.yaml

Stage 2 (768px fine-tune):
    python train/train_ip_adapter.py --config train/configs/stage2_768px.yaml

Run under caffeinate for unattended multi-day training:
    caffeinate -i -d python train/train_ip_adapter.py --config ... \
      2>&1 | tee logs/train_stage1.log &

Architecture: plans/ip-adapter-training.md §3
Performance: plans/ip-adapter-training.md Metal Optimisations section
"""

import argparse
import glob
import math
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml

# ── MLX imports ──────────────────────────────────────────────────────────────
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils as mx_utils
    import numpy as np
except ImportError:
    print("Error: MLX not found. Run: source train/.venv/bin/activate", file=sys.stderr)
    sys.exit(1)

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from ip_adapter.model import IPAdapterKlein
from ip_adapter.loss import fused_flow_noise, get_schedule_values, gram_style_loss, reconstruct_x0
from ip_adapter.ema import update_ema, _flatten
from ip_adapter.dataset import make_prefetch_loader, augment_mlx, BUCKETS
from ip_adapter.constants import SIGLIP_IMAGE_SIZE

# ── mflux: Flux Klein 4B MLX inference ───────────────────────────────────────
try:
    from mflux.models.flux2 import Flux2Klein
    from mflux.models.flux.model.flux_transformer.common.attention_utils import (
        AttentionUtils as _FluxAttentionUtils,
    )
    _HAS_MFLUX = True
except ImportError:
    _HAS_MFLUX = False
    _FluxAttentionUtils = None
    print("Warning: mflux not installed. Run: pip install mflux", file=sys.stderr)

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False



# ─────────────────────────────────────────────────────────────────────────────
# LR schedule (plans/ip-adapter-training.md §3.11)
# ─────────────────────────────────────────────────────────────────────────────

def make_lr_schedule(lr_max: float, warmup_steps: int, total_steps: int,
                      start_step: int = 0):
    """
    Linear warmup then cosine decay.
    Uses optim.join_schedules matching plans/ip-adapter-training.md §3.11.

    start_step: if resuming, pass the loaded step so the schedule continues
    from the correct position rather than restarting warmup from scratch.

    eta_min is set proportionally to lr_max (1% of peak) so that chunks with
    lower lr (e.g. 1e-5) still have a meaningful decay range instead of
    collapsing to the hardcoded 1e-6 floor.
    """
    eta_min = max(1e-8, lr_max * 0.01)
    decay_steps = max(1, total_steps - warmup_steps)

    if start_step == 0:
        return optim.join_schedules(
            [
                optim.linear_schedule(1e-8, lr_max, steps=warmup_steps),
                optim.cosine_decay(lr_max, decay_steps=decay_steps, end=eta_min),
            ],
            [warmup_steps],
        )

    # Resume: fast-forward to the correct LR position
    if start_step < warmup_steps:
        current_lr = lr_max * start_step / max(1, warmup_steps)
        remaining_warmup = warmup_steps - start_step
        return optim.join_schedules(
            [
                optim.linear_schedule(current_lr, lr_max, steps=remaining_warmup),
                optim.cosine_decay(lr_max, decay_steps=decay_steps, end=eta_min),
            ],
            [remaining_warmup],
        )
    else:
        # Already in cosine decay phase.
        # Use a step-offset closure so the resumed curve is a continuation of the
        # original cosine rather than a fresh cosine starting flat from current_lr.
        # optim.cosine_decay(current_lr, ...) would treat current_lr as a new peak,
        # making the slope ~2× steeper than the original schedule at mid-decay.
        decay_done = start_step - warmup_steps
        decay_total = max(1, total_steps - warmup_steps)
        def _resumed_cosine(step: int) -> float:
            t = min(step + decay_done, decay_total)
            return eta_min + 0.5 * (lr_max - eta_min) * (1.0 + math.cos(math.pi * t / decay_total))
        return _resumed_cosine


# ─────────────────────────────────────────────────────────────────────────────
# Async checkpoint save (plans/ip-adapter-training.md §3.12)
# ─────────────────────────────────────────────────────────────────────────────

# Lock serialises _purge_old_checkpoints across concurrent async writer threads.
_ckpt_lock = threading.Lock()


def _git_sha() -> str:
    """Return the current HEAD git SHA (short), or 'unknown' if unavailable."""
    import subprocess
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


_SAFETENSORS_DTYPE = {
    mx.float32:  "F32",
    mx.float16:  "F16",
    mx.bfloat16: "BF16",
    mx.int8:     "I8",
    mx.uint8:    "U8",
    mx.int32:    "I32",
    mx.int64:    "I64",
}


def _save_safetensors_streaming(
    path: str,
    tensor_pairs: list,
) -> None:
    """
    Write safetensors without staging all tensors in memory at once, then
    validate the written file by re-reading it sequentially.

    Standard mx.save_safetensors serialises the full payload (e.g. 4 GB) into
    a single contiguous buffer before writing, doubling peak Metal usage.  This
    function writes one tensor at a time: the numpy chunk for each tensor is
    created, written to disk, and deleted before the next tensor is processed.

    On Apple Silicon (unified memory) np.array(mlx_arr) is often zero-copy —
    numpy receives a view into the same physical memory the Metal buffer uses.
    Even if a copy is made, at most one tensor (~50–200 MB) is held in CPU
    memory at a time, keeping peak overhead near zero.

    After writing, a streaming validation pass re-reads the data section
    sequentially, comparing CRC32 of each written chunk against the CRC32
    computed during the write.  Both passes stay within a single tensor's
    footprint at any moment.  Raises RuntimeError on mismatch.

    tensor_pairs: list of (name: str, array: mx.array) in write order.
    """
    import struct as _struct, zlib as _zlib

    # Pass 1: build header from tensor metadata only — no data movement.
    offset = 0
    header = {}
    for name, arr in tensor_pairs:
        n = arr.nbytes
        header[name] = {
            "dtype":        _SAFETENSORS_DTYPE[arr.dtype],
            "shape":        list(arr.shape),
            "data_offsets": [offset, offset + n],
        }
        offset += n

    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    # Safetensors spec: header length must be a multiple of 8 (pad with spaces).
    pad = (8 - len(header_json) % 8) % 8
    header_json += b' ' * pad
    header_len = len(header_json)

    # Pass 2: write header then tensors one at a time; accumulate CRC32 per tensor.
    crcs: list = []
    with open(path, 'wb') as _f:
        _f.write(_struct.pack('<Q', header_len))
        _f.write(header_json)
        for _name, arr in tensor_pairs:
            mx.eval(arr)  # no-op for concrete tensors; safety net
            if arr.dtype == mx.bfloat16:
                # numpy has no native bf16 — reinterpret as uint16 (same bytes)
                chunk = np.array(arr.view(mx.uint16))
            else:
                chunk = np.array(arr)
            raw = chunk.tobytes()
            crcs.append(_zlib.crc32(raw))
            _f.write(raw)
            del chunk, raw

    # Pass 3: validate — re-read the data section sequentially, one tensor at a
    # time.  The page cache is warm from the write so this is purely CPU-bound.
    with open(path, 'rb') as _f:
        _f.seek(8 + header_len)  # skip fixed header (8 bytes length + JSON)
        for i, (name, arr) in enumerate(tensor_pairs):
            n = arr.nbytes
            data = _f.read(n)
            if len(data) != n:
                raise RuntimeError(
                    f"Checkpoint validation: truncated data for '{name}' "
                    f"(expected {n} bytes, got {len(data)})"
                )
            actual = _zlib.crc32(data)
            if actual != crcs[i]:
                raise RuntimeError(
                    f"Checkpoint validation: CRC32 mismatch for '{name}' "
                    f"(written {crcs[i]:#010x}, read back {actual:#010x})"
                )
            del data


def _purge_file_page_cache(path: str) -> None:
    """
    Evict a file from the macOS unified buffer cache (page cache).

    mx.save_safetensors leaves ~4 GB of dirty pages in the kernel UBC after
    each checkpoint write. Two consecutive saves accumulate ~8 GB of page cache
    that jetsam does not evict before killing the training process.

    Approach: open the file read-write, mmap it, then call
    msync(MS_SYNC|MS_INVALIDATE). MS_SYNC flushes any dirty pages to the
    backing store; MS_INVALIDATE immediately evicts all pages from the UBC.
    A read-only mapping cannot issue MS_SYNC for dirty pages written by another
    fd — the read-write mapping is required so the kernel allows the flush.
    """
    try:
        import ctypes as _ctypes, mmap as _mmap
        _libc = _ctypes.CDLL("libc.dylib", use_errno=True)
        _fd = os.open(path, os.O_RDWR)
        try:
            _sz = os.fstat(_fd).st_size
            if _sz == 0:
                return
            _mm = _mmap.mmap(_fd, _sz)  # read-write mapping
            try:
                # 1-byte from_buffer gives the mmap base address via addressof();
                # must stay alive during msync so the buffer isn't unlocked.
                _anchor = (_ctypes.c_char * 1).from_buffer(_mm)
                _addr = _ctypes.c_void_p(_ctypes.addressof(_anchor))
                # MS_SYNC=1 | MS_INVALIDATE=2 = 3: flush dirty pages to disk,
                # then evict all pages from the UBC immediately.
                _libc.msync(_addr, _ctypes.c_size_t(_sz), _ctypes.c_int(3))
                del _anchor
            finally:
                _mm.close()
        finally:
            os.close(_fd)
    except Exception:
        pass  # non-critical; pages reclaim naturally under pressure


def save_checkpoint(
    adapter: IPAdapterKlein,
    ema_params: dict,
    step: int,
    output_dir: str,
    keep_last_n: int = 5,
    lineage: Optional[dict] = None,
) -> None:
    """
    Save adapter weights + EMA synchronously using streaming write + validation.

    Background saving was tried but caused OOM: the background thread held
    old adapter param tensors alive while the main thread created new ones
    (after optimizer.update), temporarily doubling adapter memory footprint
    and pushing unified memory over 32 GB. Synchronous saving adds ~2-4s
    every 500 steps (<0.2% overhead) and avoids the memory overlap entirely.

    Checkpoint format (single combined file):
      step_NNNNNNN.safetensors — adapter keys + ema.* keys

    _save_safetensors_streaming() writes one tensor at a time and validates
    each written chunk via CRC32 before returning, keeping peak Metal buffer
    overhead at ~one tensor (50–200 MB) rather than 2–8 GB for a bulk save.

    lineage: if provided, written as a sidecar step_NNNNNNN.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"step_{step:07d}.safetensors")
    ckpt_tmp  = ckpt_path + ".tmp"

    tensor_pairs = (
        list(_flatten(adapter.parameters()))
        + [(f"ema.{k}", v) for k, v in _flatten(ema_params)]
    )

    mx.clear_cache()
    _save_safetensors_streaming(ckpt_tmp, tensor_pairs)
    _purge_file_page_cache(ckpt_tmp)
    os.rename(ckpt_tmp, ckpt_path)
    mx.clear_cache()

    size_mb = os.path.getsize(ckpt_path) / 1e6
    print(f"  checkpoint saved: step_{step:07d}.safetensors ({size_mb:.0f} MB)")

    if lineage is not None:
        sidecar_path = os.path.join(output_dir, f"step_{step:07d}.json")
        sidecar_tmp  = sidecar_path + ".tmp"
        try:
            with open(sidecar_tmp, "w") as _f:
                json.dump(lineage, _f, indent=2)
            os.rename(sidecar_tmp, sidecar_path)
        except OSError as e:
            print(f"  WARNING: could not write lineage sidecar: {e}")

    with _ckpt_lock:
        _purge_old_checkpoints(output_dir, keep_last_n)


def _purge_old_checkpoints(directory: str, keep_last_n: int) -> None:
    """
    Keep the `keep_last_n` most recent checkpoints (by step number) plus up to
    3 checkpoints with the best validation loss recorded in val_loss.jsonl.
    The latest checkpoint is always preserved regardless of keep_last_n.

    Sorts by parsed step number (not lexically) so files with inconsistent
    zero-padding or step counts beyond the format width sort correctly.
    """
    all_paths = [
        f for f in glob.glob(os.path.join(directory, "step_*.safetensors"))
        if "ema." not in os.path.basename(f)
    ]
    if not all_paths:
        return

    # Sort numerically by step, not lexically.
    by_step = sorted(all_paths, key=step_from_checkpoint_path)

    if len(by_step) <= keep_last_n:
        return

    # Collect the set of paths to protect from deletion.
    protected: set = set()

    # Always keep the latest checkpoint.
    protected.add(by_step[-1])

    # Keep the keep_last_n most recent.
    protected.update(by_step[-keep_last_n:])

    # Also protect checkpoints that match the best 3 val_loss steps.
    val_log = os.path.join(directory, "val_loss.jsonl")
    if os.path.exists(val_log):
        try:
            records = []
            with open(val_log) as _vf:
                for line in _vf:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if records:
                records.sort(key=lambda r: r.get("val_loss", float("inf")))
                best_steps = {r["step"] for r in records[:3]}
                for p in by_step:
                    if step_from_checkpoint_path(p) in best_steps:
                        protected.add(p)
        except Exception:
            pass  # val_loss.jsonl unreadable — keep by recency only

    for f in by_step:
        if f in protected:
            continue
        try:
            os.remove(f)
        except FileNotFoundError:
            pass  # already removed by a concurrent thread
        sidecar = os.path.splitext(f)[0] + ".json"
        try:
            os.remove(sidecar)
        except FileNotFoundError:
            pass


def load_checkpoint(adapter: IPAdapterKlein, path: str) -> None:
    """Load adapter weights from a step_*.safetensors checkpoint (skips ema.* keys)."""
    from safetensors import safe_open
    params = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys():
            if not k.startswith("ema."):
                params[k] = mx.array(f.get_tensor(k))
    _nested_update(adapter, params)
    _purge_file_page_cache(path)  # reclaim ~4 GB OS page cache from the read
    print(f"Loaded checkpoint: {path}")


def load_ema_from_checkpoint(path: str) -> Optional[dict]:
    """
    Load EMA parameters from a checkpoint file. Handles two formats:
      - step_*.safetensors: keys prefixed with "ema." (written by save_checkpoint)
      - best.safetensors:   bare keys, no prefix (legacy format)
    Returns a nested dict matching adapter.parameters() structure, or None if
    the file contains no recognisable EMA or adapter keys.
    """
    from safetensors import safe_open
    flat: dict = {}
    with safe_open(path, framework="numpy") as f:
        keys = list(f.keys())
        has_ema_prefix = any(k.startswith("ema.") for k in keys)
        for k in keys:
            if has_ema_prefix:
                if k.startswith("ema."):
                    flat[k[4:]] = mx.array(f.get_tensor(k))
            else:
                # bare keys — legacy format (e.g. best.safetensors)
                flat[k] = mx.array(f.get_tensor(k))
    if not flat:
        return None
    _purge_file_page_cache(path)  # reclaim ~4 GB OS page cache from the read
    return _flat_to_nested(flat)


def step_from_checkpoint_path(path: str) -> int:
    """Parse the step number from a step_NNNNNNN.safetensors filename, or 0."""
    name = os.path.basename(path)
    if name.startswith("step_") and name.endswith(".safetensors"):
        try:
            return int(name[5:-len(".safetensors")])
        except ValueError:
            pass
    return 0


def _nested_update(model: nn.Module, flat: dict) -> None:
    nested = {}
    for key, val in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    model.update(nested)


def _flat_to_nested(flat: dict) -> dict:
    """Convert a flat dot-separated key dict to a nested dict."""
    nested: dict = {}
    for key, val in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return nested


# ─────────────────────────────────────────────────────────────────────────────
# Shard ↔ precompute-cache contract (the precompute → train handoff)
#
# These are the seam that silently failed flywheel iter 10 ("0/40 shards have
# qwen3+vae precomputed"): the trainer only trains on shards whose precomputed
# .npz files are present and findable under the configured cache dirs. The
# record-key naming convention ({shard_stem}_{index:04d}.npz) MUST match what
# precompute_all.py writes, or every shard is silently skipped.
#
# Extracted to module scope (from closures inside train()) so the contract is
# unit-testable — see train/tests/test_shard_cache_filter.py.
# ─────────────────────────────────────────────────────────────────────────────

# Records probed per shard to decide "is this shard substantially precomputed?".
# Checking only _0000 would pass even if precompute crashed after the first
# write, silently skipping 99%+ of that shard's batches.
_SHARD_CACHE_PROBE_INDICES = (0, 49)


def shard_internal_prefix(tar_path: str) -> str:
    """Return the npz record-key prefix for a shard tar.

    Production shards keep their staging filename (e.g. "000000.tar" → "000000",
    "250000.tar" → "250000"), so the tar stem IS the internal record prefix.
    precompute_all.py writes records as "{prefix}_{index:04d}.npz".
    """
    return os.path.splitext(os.path.basename(tar_path))[0]


def shard_has_cache(tar_path: str, qwen3_dir: str, vae_dir: str,
                    probe_indices=_SHARD_CACHE_PROBE_INDICES) -> bool:
    """True iff probe records exist in BOTH qwen3_dir and vae_dir for this shard."""
    pfx = shard_internal_prefix(tar_path)
    return all(
        os.path.exists(os.path.join(d, f"{pfx}_{i:04d}.npz"))
        for d in (qwen3_dir, vae_dir)
        for i in probe_indices
    )


def filter_shards_with_cache(shard_paths: list, qwen3_dir: str,
                             vae_dir: str) -> list:
    """Return the subset of shard_paths whose qwen3+vae precompute is present.

    Shards missing only siglip are kept (siglip falls back to zeros); that check
    lives separately in train(). This filter gates on qwen3+vae only.
    """
    return [p for p in shard_paths if shard_has_cache(p, qwen3_dir, vae_dir)]


def _resolve_versioned_cache_dirs(config: dict, data_root: str) -> None:
    """Follow the `current` symlink ONLY for cache dirs that are the flat default.

    A cache dir written as the bare `{data_root}/precomputed/{enc}` default is
    version-resolved to its `current` directory. An explicit cache_dir set to
    anything else — notably a per-iteration flywheel staging path
    (`.../iterNNNN/precomputed/{enc}`) — is respected as-is.

    Historically this unconditionally overwrote *every* cache dir with the global
    `current`, clobbering the per-iter staging path the orchestrator had set. When
    the global `current` was an in-progress, 0-record version, the shard-cache
    filter then matched 0 shards and every flywheel iteration aborted at startup.
    Mutates config in place; never raises.
    """
    data = config.get("data") or {}
    enc_for_key = {"qwen3_cache_dir": "qwen3",
                   "vae_cache_dir":   "vae",
                   "siglip_cache_dir": "siglip"}
    if not any(data.get(k) for k in enc_for_key):
        return
    try:
        _scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
        if _scripts not in sys.path:
            sys.path.insert(0, _scripts)
        from cache_manager import PrecomputeCache
    except Exception:
        return  # cache_manager unavailable — keep configured paths as-is
    precomp_root = os.path.join(data_root, "precomputed")
    for key, enc in enc_for_key.items():
        cur = data.get(key)
        if not cur:
            continue
        # Only the flat default gets version-resolved; an explicit dir is respected.
        if os.path.realpath(cur) != os.path.realpath(os.path.join(precomp_root, enc)):
            continue
        try:
            eff = PrecomputeCache.effective_dir(Path(precomp_root), enc)
        except Exception:
            eff = None
        if eff:
            data[key] = str(eff)


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict) -> None:
    mcfg = config["model"]
    acfg = config["adapter"]
    dcfg = config["data"]
    tcfg = config["training"]
    ocfg = config["output"]
    lcfg = config.get("logging") or {}
    ecfg = config.get("eval") or {}

    # ── Lineage base dict (MLX-10) ────────────────────────────────────────────
    # Written as a sidecar .json alongside each checkpoint for reproducibility.
    # step/loss are filled in at save time; everything else is static.
    _lineage_base = {
        "git_sha": _git_sha(),
        "argv": sys.argv[:],
        "config": config,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ── Optional W&B logging ──────────────────────────────────────────────────
    wandb_run = None
    if lcfg.get("wandb_project"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=lcfg["wandb_project"],
                name=lcfg.get("wandb_run_name"),
                config=config,
            )
        except ImportError:
            print("wandb not installed — logging to stdout only")

    # ── Pre-flight checks (MLX-11) ────────────────────────────────────────────
    _preflight_ok = True
    def _check(label: str, ok: bool, detail: str = "") -> bool:
        mark = "✅" if ok else "❌"
        suffix = f"  ({detail})" if detail else ""
        print(f"  {mark} {label}{suffix}")
        return ok
    print("Pre-flight checks:")
    # MLX version
    try:
        import mlx.core as _mlx_core
        _ver_str = _mlx_core.__version__
        _mlx_ver = tuple(int(x) for x in _ver_str.split(".")[:2])
        _preflight_ok &= _check("MLX >= 0.31.1", _mlx_ver >= (0, 31), _ver_str)
    except Exception as e:
        _preflight_ok &= _check("MLX version", False, str(e))
    # Required packages
    for _pkg in ["safetensors", "yaml", "turbojpeg"]:
        try:
            __import__(_pkg)
            _preflight_ok &= _check(f"import {_pkg}", True)
        except ImportError:
            _preflight_ok &= _check(f"import {_pkg}", False, "pip install " + _pkg)
    # Model weights directory
    _flux_dir = mcfg["flux_model_dir"]
    _flux_ok = os.path.isdir(_flux_dir)
    _preflight_ok &= _check(f"model dir: {_flux_dir}", _flux_ok,
                             "" if _flux_ok else "not found")

    # VAE-Q1: load the VAE BatchNorm stats used to convert raw precomputed
    # VAE-latents (mflux encode() space, std~1.7) into the transformer's packed/BN
    # convention that C iris_vae inference operates in (std~1). Without this the
    # model trains in a latent space that does not transfer to C inference. See
    # BUGS.md VAE-Q1 and debug/vae_parity.c (validated corr 0.9995).
    _bn_mean_r, _bn_std_r = _load_vae_bn_stats(_flux_dir)
    if _bn_mean_r is not None:
        print("VAE-Q1: VAE batch-norm stats loaded; latents will be BN-packed to "
              "the C-inference convention.", flush=True)
    # Shard path
    _shard_dir = dcfg["shard_path"]
    _shard_count = len(glob.glob(os.path.join(_shard_dir, "*.tar")))
    _preflight_ok &= _check(f"shards: {_shard_dir}", _shard_count > 0,
                             f"{_shard_count} tars" if _shard_count else "no .tar files found")
    # Checkpoint if resuming
    _warmstart = mcfg.get("warmstart_path")
    if _warmstart:
        _ws_ok = os.path.isfile(_warmstart) or os.path.isdir(_warmstart)
        _preflight_ok &= _check(f"resume checkpoint: {_warmstart}", _ws_ok,
                                 "" if _ws_ok else "not found")
    # Output dir writable
    _out_dir = ocfg["checkpoint_dir"]
    try:
        os.makedirs(_out_dir, exist_ok=True)
        _out_ok = os.access(_out_dir, os.W_OK)
    except OSError:
        _out_ok = False
    _preflight_ok &= _check(f"output dir writable: {_out_dir}", _out_ok)
    if not _preflight_ok:
        raise RuntimeError("Pre-flight checks failed — see ❌ items above.")
    print()

    # ── Seed RNGs ─────────────────────────────────────────────────────────────
    _seed = tcfg.get("seed")
    if _seed is not None:
        random.seed(_seed)
        mx.random.seed(_seed)
        try:
            import numpy as _np_seed
            _np_seed.random.seed(_seed)
        except ImportError:
            pass
        print(f"RNG seed: {_seed}")

    # ── Load frozen base models ───────────────────────────────────────────────
    if not _HAS_MFLUX:
        raise RuntimeError("pip install mflux")

    # Guard against post-sleep/wake Metal GPU instability. On macOS, mflux
    # loads weights lazily via mmap + ParallelFileReader and internally triggers
    # fast::LayerNorm GPU ops during init. Within ~2 min of a sleep/wake event
    # the Metal GPU server can be unstable, producing null MTLBuffer crashes.
    # sysctl kern.waketime gives seconds-since-last-wake on macOS.
    try:
        import re
        import subprocess
        r = subprocess.run(["sysctl", "-n", "kern.waketime"],
                           capture_output=True, text=True, timeout=2)
        m = re.search(r"sec = (\d+)", r.stdout)
        if m:
            wake_secs = time.time() - int(m.group(1))
            if wake_secs < 120:
                print(f"WARNING: system woke from sleep {wake_secs:.0f}s ago. "
                      "Metal GPU may be unstable. Waiting 30s for GPU to stabilise...")
                time.sleep(30)
    except Exception:
        pass  # non-macOS or sysctl unavailable — skip check

    # Cap MLX GPU allocation.  The fraction is configurable via training.mlx_memory_pct:
    #   0.44 (default) → 14 GB on 32 GB — calibrated for precomputed-cache mode
    #   0.55           → 17.6 GB on 32 GB — use for online-encoding mode where
    #                    SigLIP (PyTorch, ~1.4 GB) + 4 MLX models saturate 14 GB
    # Without a cap, jetsam kills the process during mx.compile's first backward-pass
    # compilation when transient buffer peaks exceed the system pressure threshold.
    # set_cache_limit(0) prevents MLX holding freed buffers in a cache, forcing
    # immediate return to the OS and reducing peak wired memory.
    _mem_pct = float(tcfg.get("mlx_memory_pct", 0.44))
    try:
        import subprocess as _sp
        _hw = _sp.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3)
        _ram_bytes = int(_hw.stdout.strip())
    except Exception:
        _ram_bytes = 32 * 1024 ** 3  # fallback for non-macOS or sysctl unavailable
    mx.set_memory_limit(int(_ram_bytes * _mem_pct))
    mx.set_cache_limit(int(_ram_bytes * 0.06))    # 6% → 2 GB on 32 GB
    print(f"MLX memory limit: {int(_ram_bytes * _mem_pct) // 1024**3} GB  "
          f"cache limit: {int(_ram_bytes * 0.06) // 1024**2} MB")

    # Write loading heartbeats every 60s while models initialise.
    # Without this, the orchestrator's 900s stale threshold kills the process
    # before model loading + graph compilation finish (~10-12 min total).
    _boot_chunk      = config.get("_chunk")
    _boot_run_name   = config.get("_run_name") if _boot_chunk is None else None
    _boot_hb_process = f"trainer_{_boot_run_name}" if _boot_run_name else "trainer"
    _boot_hb_stop    = __import__("threading").Event()
    def _boot_hb_thread():
        import time as _t
        from pipeline_lib import write_heartbeat as _wh
        while not _boot_hb_stop.wait(60):
            _wh(_boot_hb_process, _boot_chunk, status="loading", step=0)
    __import__("threading").Thread(target=_boot_hb_thread, daemon=True).start()

    print("Loading Flux Klein 4B (frozen) ...")
    flux = Flux2Klein(model_path=mcfg["flux_model_dir"], quantize=None)
    flux.freeze()

    use_siglip_live = dcfg.get("siglip_cache_dir") is None
    vae_cache_available = bool(dcfg.get("vae_cache_dir"))
    text_cache_available = bool(dcfg.get("qwen3_cache_dir"))

    # Online (live-encode) mode: do NOT eagerly load Qwen3/VAE/SigLIP here.
    # They are only needed for the ~100 ms encoding window at the *start* of each
    # training step. Keeping them resident for the full Flux+adapter fwd/bwd (~5-6 s)
    # wastes ~3.8 GB and causes jetsam OOM on 32 GB systems (flywheel warmup runs).
    # We capture the weight loader state once, then load the encoders on-demand
    # inside the step loop via _ensure_live_encoders() and drop them immediately
    # after encoding via _release_live_encoders() (see training loop below).
    _encoder_weights = None
    _encoder_quantize = None
    _encoder_model_path = mcfg["flux_model_dir"]

    if (not vae_cache_available) or (not text_cache_available):
        # Capture the LoadedWeights (mmap-backed) so we can re-attach just the
        # VAE or text_encoder submodule on demand without re-loading the whole Flux.
        try:
            from mflux.models.common.weights.loading.weight_loader import WeightLoader
            from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
            _encoder_weights = WeightLoader.load(
                weight_definition=Flux2KleinWeightDefinition,
                model_path=_encoder_model_path,
            )
            _encoder_quantize = None  # matches the quantize=None passed above
            print("Online mode: captured encoder weights for per-step load/unload.")
        except Exception as _cap_err:
            print(f"WARNING: failed to capture encoder weights for online reload: {_cap_err}")

        # Detach any submodules Flux2Klein.__init__ created so their weight buffers
        # are not resident during the rest of startup / graph warmup / training.
        # The transformer (the only part we need long-term) stays loaded.
        if not vae_cache_available and getattr(flux, "vae", None) is not None:
            flux.vae = None
        if not text_cache_available and getattr(flux, "text_encoder", None) is not None:
            flux.text_encoder = None
        mx.clear_cache()

    # SigLIP is always a separate load (never inside Flux).
    siglip = None
    if use_siglip_live:
        print("SigLIP: will load on-demand per step (online) and unload after encode to save ~1.4 GB.")
    else:
        print("SigLIP: using precomputed cache — skipping model load.")

    if vae_cache_available:
        vae = None  # precomputed latents — VAE not needed during training
        print("VAE: using precomputed cache — skipping model load.")
    else:
        print("VAE: will load on-demand per step (online) and unload after encode.")
        vae = None  # populated transiently by _ensure_live_encoders()

    if text_cache_available:
        text_encoder = None  # precomputed text embeddings — encoder not needed
        print("Text encoder: using precomputed cache — skipping model load.")
    else:
        print("Text encoder: will load on-demand per step (online) and unload after encode.")
        text_encoder = None  # populated transiently by _ensure_live_encoders()

    # One-time banner for the online path (the scenario that was OOMing flywheel warmups).
    _any_live = (not vae_cache_available) or (not text_cache_available) or use_siglip_live
    if _any_live:
        live_list = []
        if not vae_cache_available: live_list.append("VAE")
        if not text_cache_available: live_list.append("Qwen3")
        if use_siglip_live: live_list.append("SigLIP")
        print(f"\n[online-encode] LIVE ENCODER MODE ACTIVE for this run: {', '.join(live_list)}")
        print("[online-encode] These models will be attached only for the ~100-400 ms feature extraction")
        print("[online-encode] window at the TOP of each step, then fully unloaded (flux.*=None + gc +")
        print("[online-encode] clear_cache + torch.mps.empty_cache) BEFORE the Flux forward + adapter step.\n")

    # Start data loader threads now so they can prefill the batch queue during
    # the Flux pre-eval below (which takes several minutes). By the time training
    # starts the first several batches will already be decoded and waiting.
    # Campaign manifest support: when dcfg["shard_manifest"] is set, load shard
    # paths from the campaign's include-list instead of globbing a directory.
    # Falls back to the existing dcfg["shard_path"] glob when no manifest is set.
    _shard_manifest = dcfg.get("shard_manifest")
    if _shard_manifest:
        _m = json.loads(Path(_shard_manifest).read_text())
        shard_paths = sorted([
            e["path"] for e in _m.get("entries", [])
            if e.get("decision") == "include"
        ])
        if not shard_paths:
            raise RuntimeError(f"Campaign manifest has no included shards: {_shard_manifest}")
        print(f"Campaign '{_m.get('campaign', '?')}': {len(shard_paths):,} shards "
              f"(strategy={_m.get('strategy', '?')}, "
              f"avg_score={_m.get('avg_final_value_score', 0):.4f})")
    else:
        shard_paths = sorted(glob.glob(os.path.join(dcfg["shard_path"], "*.tar")))
        if not shard_paths:
            raise RuntimeError(f"No .tar shards found in {dcfg['shard_path']}")

    # Filter to only shards that have at least qwen3+vae precomputed.
    # Without this, the loader reads every shard (1.9 GB each) but the training
    # loop skips all their batches — 92% I/O waste and GPU data stalls.
    # Shards missing siglip cache are kept (siglip falls back to zeros).
    qwen3_dir = dcfg.get("qwen3_cache_dir")
    vae_dir   = dcfg.get("vae_cache_dir")

    if qwen3_dir and vae_dir:
        # See the "Shard ↔ precompute-cache contract" helpers above. This is the
        # seam that silently failed flywheel iter 10 (0/40 shards matched).
        all_shards = shard_paths
        shard_paths = filter_shards_with_cache(all_shards, qwen3_dir, vae_dir)
        print(f"Shard cache filter: {len(shard_paths)}/{len(all_shards)} shards "
              f"have qwen3+vae precomputed — training on {len(shard_paths)} shards only.")
        if not shard_paths:
            raise RuntimeError(
                "No shards with precomputed qwen3+vae cache found. "
                "Run train/scripts/precompute_all.py first."
            )

    # SigLIP coverage check (MLX-15): if siglip_cache_dir is set but incomplete,
    # batches silently use zero image features, degrading training signal.
    siglip_dir = dcfg.get("siglip_cache_dir")
    if siglip_dir and os.path.isdir(siglip_dir):
        _siglip_npz = glob.glob(os.path.join(siglip_dir, "*.npz"))
        # Precomputed files are keyed by shard stem (e.g. "000000" for chunk 1, "250000" for chunk 2).
        _covered = {os.path.basename(f).split("_")[0] for f in _siglip_npz}
        _shard_prefixes = {shard_internal_prefix(p) for p in shard_paths}
        _missing = _shard_prefixes - _covered
        _coverage = len(_covered & _shard_prefixes) / len(_shard_prefixes) if _shard_prefixes else 1.0
        if _missing:
            print(
                f"WARNING: SigLIP cache coverage {_coverage*100:.0f}% "
                f"({len(_covered & _shard_prefixes)}/{len(_shard_prefixes)} shards). "
                f"Missing: {sorted(_missing)[:5]}{'...' if len(_missing) > 5 else ''}. "
                f"Batches from uncovered shards will train with zero image features."
            )
        else:
            print(f"SigLIP cache: {_coverage*100:.0f}% coverage ({len(_shard_prefixes)} shards). OK.")

    # Precompute<->train resolution contract: VAE latents are precomputed at a
    # single fixed resolution (precompute_all.py resizes to image_size), so a
    # cached entry's spatial shape only matches one training bucket. With cached
    # VAE (no live encoder to fall back on), a non-matching bucket is a hard
    # cache miss -> the sample is skipped, and multi-bucket training skips ~100%
    # of batches. Pin training to `data.bucket` ([H, W]) matching the precompute
    # resolution. Leave unset only when VAE latents were precomputed per-bucket.
    _bucket_cfg = dcfg.get("bucket")
    _fixed_bucket = tuple(_bucket_cfg) if _bucket_cfg else None
    if _fixed_bucket:
        print(f"Training pinned to single bucket {_fixed_bucket} "
              f"(matches fixed-resolution VAE precompute).")
    loader = make_prefetch_loader(
        shard_paths=shard_paths,
        batch_size=dcfg["batch_size"],
        text_dropout_prob=tcfg["text_dropout_prob"],
        sample_buffer=dcfg.get("prefetch_batches", 6),
        qwen3_cache_dir=dcfg.get("qwen3_cache_dir"),
        vae_cache_dir=dcfg.get("vae_cache_dir"),
        siglip_cache_dir=dcfg.get("siglip_cache_dir"),
        anchor_shard_dir=dcfg.get("anchor_shard_dir"),
        anchor_mix_ratio=dcfg.get("anchor_mix_ratio", 0.20),
        hard_example_dir=dcfg.get("hard_example_dir"),
        hard_mix_ratio=dcfg.get("hard_mix_ratio", 0.05),
        bucket=_fixed_bucket,
        style_neighbors_db=dcfg.get("style_neighbors_db"),
    )

    # Force-materialize mmap'd Flux transformer weights into GPU memory before
    # training starts. Without this, the first training step lazily loads weights
    # while also computing activations; the mmap backing pages can be freed while
    # the GPU is still referencing them, causing a null-MTLBuffer SIGSEGV.
    print("Pre-evaluating Flux transformer weights ...")
    mx.eval(flux.transformer.parameters())

    # Pre-compile Metal graphs for all training bucket shapes.
    # _flux_forward_no_ip is a different graph from inference (collects Q at every
    # block), so MPSGraph must compile it separately. Without warmup the first real
    # training step stalls 10-30 min per bucket shape. We warm up all 6 buckets
    # now so no compilation happens mid-training.
    _txt_warmup = mx.zeros((1, 64, flux.transformer.context_embedder.weight.shape[1]),
                           dtype=mx.bfloat16)
    _t_warmup   = mx.array([500], dtype=mx.int32)
    print("Warming up Flux training graphs (all bucket shapes)...")
    for _bH, _bW in BUCKETS:
        _lat_H, _lat_W = _bH // 8, _bW // 8          # VAE latent spatial dims
        _dummy_lat = mx.zeros((1, 32, _lat_H, _lat_W), dtype=mx.bfloat16)
        print(f"  [{_bH}x{_bW}] compiling...", flush=True)
        _t0_wu = time.time()
        _fs = _flux_forward_no_ip(flux, _dummy_lat, _txt_warmup, _t_warmup)
        mx.eval(_fs["qs"], _fs["h_final"], _fs["temb"])
        print(f"  [{_bH}x{_bW}] done ({time.time() - _t0_wu:.1f}s)", flush=True)
        del _fs
        mx.clear_cache()
    del _dummy_lat, _txt_warmup, _t_warmup
    print("Flux training graphs ready.")

    ckpt_double = None
    ckpt_single = None
    # TRAIN-5 Stage 3: block gradient checkpointing infrastructure for TRAIN-6.
    # Wraps each Flux transformer block with mx.checkpoint so activations are freed
    # after forward and recomputed block-by-block during backward. Zero cost when
    # disabled; zero runtime cost with current _flux_forward_no_ip path even when
    # enabled. Only incurs recompute overhead when TRAIN-6 activates _flux_forward_with_ip.
    if acfg.get("block_gradient_checkpointing", False):
        _tr = flux.transformer
        ckpt_double = [mx.checkpoint(b) for b in _tr.transformer_blocks]
        ckpt_single = [mx.checkpoint(b) for b in _tr.single_transformer_blocks]
        print(f"  Block gradient checkpointing enabled: "
              f"{len(ckpt_double)} double + {len(ckpt_single)} single blocks")

    # ── Build IP-Adapter (trainable) ──────────────────────────────────────────
    print("Building IPAdapterKlein ...")
    warmstart = mcfg.get("warmstart_path")
    if warmstart and os.path.isdir(warmstart):
        # Warmstart Perceiver Resampler from InstantX Flux.1-dev weights
        adapter = IPAdapterKlein.from_pretrained_warmstart(
            instantx_path=warmstart,
            num_blocks=acfg["num_blocks"],
            num_double_blocks=acfg.get("num_double_blocks", 5),
            hidden_dim=acfg["hidden_dim"],
            num_image_tokens=acfg["num_image_tokens"],
            siglip_dim=acfg["siglip_dim"],
            perceiver_heads=acfg["perceiver_heads"],
        )
    else:
        adapter = IPAdapterKlein(
            num_blocks=acfg["num_blocks"],
            num_double_blocks=acfg.get("num_double_blocks", 5),
            hidden_dim=acfg["hidden_dim"],
            num_image_tokens=acfg["num_image_tokens"],
            siglip_dim=acfg["siglip_dim"],
            perceiver_heads=acfg["perceiver_heads"],
        )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    # Purge ALL checkpoint files from the OS page cache before loading anything.
    # Checkpoints written in previous sessions (before msync eviction was added,
    # or that were never loaded/purged) accumulate ~4 GB each in the unified
    # buffer cache. With keep_last_n=5, that's up to 20 GB of inactive pages
    # that psutil counts as "available" but Metal cannot use. Each crash-restart
    # cycle adds more. Evicting the whole directory at startup clears the slate.
    _ckpt_dir_for_purge = ocfg.get("checkpoint_dir", "")
    if _ckpt_dir_for_purge:
        for _cf in glob.glob(os.path.join(_ckpt_dir_for_purge, "*.safetensors")):
            _purge_file_page_cache(_cf)

    start_step = 0
    loaded_ema: Optional[dict] = None
    if warmstart and os.path.isfile(warmstart):
        load_checkpoint(adapter, warmstart)
        start_step = step_from_checkpoint_path(warmstart)
        loaded_ema = load_ema_from_checkpoint(warmstart)
        if start_step > 0:
            print(f"  Resuming from step {start_step:,}")
        if loaded_ema is not None:
            print(f"  Loaded EMA from checkpoint")

    # Weights-only warm-start (e.g. ablation arms from a campaign checkpoint):
    # load the adapter weights but DELIBERATELY keep start_step=0 — fresh warmup,
    # fresh optimizer, fresh step horizon — so the arm trains its full --max-steps
    # under its own config rather than inheriting the checkpoint's training horizon.
    _ws_weights_only = mcfg.get("warmstart_weights_only")
    if _ws_weights_only and os.path.isfile(_ws_weights_only):
        load_checkpoint(adapter, _ws_weights_only)
        print(f"  Warm-started adapter weights from "
              f"{os.path.basename(_ws_weights_only)} (fresh schedule, start_step=0)")

    if bool(tcfg.get("correct_forward_q", False)) and not bool(tcfg.get("use_block_injection", False)):
        _txt_warmup_cq = mx.zeros(
            (1, 64, flux.transformer.context_embedder.weight.shape[1]), dtype=mx.bfloat16)
        _t_warmup_cq = mx.array([500], dtype=mx.int32)
        _siglip_warmup_cq = mx.zeros((1, 729, acfg["siglip_dim"]), dtype=mx.bfloat16)
        _ip_embs_wu = adapter.get_image_embeds(_siglip_warmup_cq)
        _k_wu, _v_wu = adapter.get_kv_all(_ip_embs_wu)
        mx.eval(_k_wu, _v_wu, adapter.scale)
        print("Warming up correct-forward-Q graphs (all bucket shapes)...")
        for _bH, _bW in BUCKETS:
            _lat_H, _lat_W = _bH // 8, _bW // 8
            _dummy_lat_cq = mx.zeros((1, 32, _lat_H, _lat_W), dtype=mx.bfloat16)
            print(f"  [{_bH}x{_bW}] compiling...", flush=True)
            _t0_wu = time.time()
            _qs_wu = _flux_forward_with_ip_collect_q(
                flux, _dummy_lat_cq, _txt_warmup_cq, _t_warmup_cq,
                _k_wu, _v_wu, adapter.scale,
            )
            mx.eval(_qs_wu)
            print(f"  [{_bH}x{_bW}] done ({time.time() - _t0_wu:.1f}s)", flush=True)
            del _dummy_lat_cq, _qs_wu
            mx.clear_cache()
        del _txt_warmup_cq, _t_warmup_cq, _siglip_warmup_cq, _ip_embs_wu, _k_wu, _v_wu
        print("Correct-forward-Q graphs ready.")

    # chunk_base_step: the global step at which this chunk's training begins.
    # Equals sum of all prior chunks' steps.  Passed via --chunk-base-step.
    # Without it (standalone runs), assume start_step is the base (correct for
    # chunk-1 where base=0, and for single-chunk standalone training).
    chunk_base_step: int = config.get("_chunk_base_step", start_step)

    # _end_step: absolute global step at which this run terminates.
    # Using chunk_base_step + num_steps (not start_step + num_steps) so that
    # mid-chunk crash resumes stop at the same target as the original launch.
    _end_step = chunk_base_step + tcfg["num_steps"]

    # ── Optimizer with built-in cosine+warmup schedule ────────────────────────
    # MLX schedule object passed directly to AdamW — it advances each step.
    # _lr_start fast-forwards the schedule by how many steps into this chunk we
    # already are, so LR continues from where it left off on resume.
    # For a cross-chunk warmstart (start_step == chunk_base_step) _lr_start=0
    # starts the schedule fresh at the chunk's own LR.
    _lr_start = max(0, start_step - chunk_base_step)
    lr_schedule = make_lr_schedule(
        tcfg["learning_rate"],
        tcfg["warmup_steps"],
        tcfg["num_steps"],
        start_step=_lr_start,
    )
    optimizer = optim.AdamW(
        learning_rate=lr_schedule,
        betas=(0.9, 0.999),
        weight_decay=tcfg["weight_decay"],
    )

    # ── EMA: use loaded EMA weights if resuming, else start from adapter ──────
    if loaded_ema is not None:
        ema_params = loaded_ema
    else:
        ema_params = mx_utils.tree_map(lambda x: mx.array(x), adapter.parameters())

    # ── Pre-evaluate adapter parameters ───────────────────────────────────────
    # Flush all lazy init arrays (mx.random.normal * scale) before training.
    mx.eval(adapter.parameters())
    mx.eval(ema_params)

    # QUALITY-2: freeze double-stream IP scales throughout training.
    # Double-stream blocks (0..nd-1) primarily inject content/structure; keeping
    # them at zero forces the adapter to learn style via single-stream blocks only.
    _freeze_double_stream = acfg.get("freeze_double_stream_scales", False)
    _nd = adapter.num_double_blocks  # number of double-stream blocks (default 5)

    # QUALITY-2: zero double-stream scales after checkpoint load so content-injecting
    # blocks stay silent for the entire run (grad zeroing in compiled_step keeps them zero).
    if _freeze_double_stream:
        adapter.scale = mx.concatenate([
            mx.zeros((_nd,), dtype=adapter.scale.dtype),
            adapter.scale[_nd:],
        ])
        mx.eval(adapter.scale)

    _style_weight = float(tcfg.get("style_loss_weight", 0.0))
    _style_every  = int(tcfg.get("style_loss_every", 1))
    # Written by loss_fn on conditioned steps; read after eval for logging.
    _style_loss_accum: list = [mx.array(0.0)]

    # TRAIN-5 Stage 0: per-fence peak memory profiling.
    # Gate behind memory_profile: true in training config. Resets peak counter
    # after each eval fence to isolate: Flux fwd / adapter bwd+opt / param update / EMA.
    _mem_profile = bool(tcfg.get("memory_profile", False))
    _mem_fwd_peak = _mem_bwd_peak = _mem_ema_peak = 0

    # QUALITY-1 / QUALITY-3 aug probabilities
    _cross_ref_prob     = float(tcfg.get("cross_ref_prob", 0.0))
    _patch_shuffle_prob = float(tcfg.get("patch_shuffle_prob", 0.0))

    # Hoisted loop constants — avoid repeated dict lookups and mx.array() allocation
    # inside the hot path. Saves a sys.modules + dict hit on every step.
    _image_dropout_prob = float(tcfg["image_dropout_prob"])
    _text_dropout_prob  = float(tcfg["text_dropout_prob"])
    _grad_clip          = float(tcfg["grad_clip"])
    _ema_update_every   = int(tcfg["ema_update_every"])
    _ema_decay_factor   = float(tcfg["ema_decay"]) ** _ema_update_every
    # n_grad_steps_per_fwd>1 runs N optimizer updates on the SAME forward pass
    # result (same noise, latent, text embeds). This is NOT gradient accumulation —
    # effective LR is multiplied by N. Only useful without block-injection.
    _n_grad_steps       = int(tcfg.get("n_grad_steps_per_fwd", 1))
    _use_block_injection = bool(tcfg.get("use_block_injection", False))
    if _use_block_injection and _n_grad_steps > 1:
        print("  TRAIN-6: use_block_injection=true — n_grad_steps_per_fwd forced to 1")
        _n_grad_steps = 1
    if tcfg.get("correct_forward_q") and _n_grad_steps > 1:
        print("  WARNING: correct_forward_q=true with n_grad_steps_per_fwd>1 is unsupported "
              "(Q vectors go stale after step 1) — forcing n_grad_steps_per_fwd=1")
        _n_grad_steps = 1
    # Option C: correct-forward Q vectors (IP-injected forward for Q extraction,
    # end-sum adapter-only backward). Ignored when use_block_injection=true.
    _use_correct_forward_q = (
        bool(tcfg.get("correct_forward_q", False)) and not _use_block_injection
    )
    # Precomputed scalar MLX bools — avoids mx.array(bool) allocation every step.
    _MX_TRUE  = mx.array(True)
    _MX_FALSE = mx.array(False)

    def loss_fn(siglip_feats, use_null_image: mx.array,
                flux_state: dict, target: mx.array,
                x0_ref=None, noisy_in=None, alpha_in=None, sigma_in=None):
        """
        Differentiable adapter-only loss. Flux forward has already been run
        outside this function (no grad); only the tiny adapter graph is traced.

        flux_state contains stop_gradient'd tensors from _flux_forward_no_ip:
          qs:      list[num_blocks] of [B, H, seq_img, Hd] -- Q per injection point
          h_final: [B, seq_img, d_inner] -- Flux hidden state pre norm_out
          temb:    [B, d_inner] -- timestep embedding for norm_out
          shape metadata: B, C, Lh, Lw, pH, pW

        Gradient graph: adapter → k_ip/v_ip → ip_out → h_final+ip → norm_out → pred → loss
        No Flux block ops appear in the backward pass.
        """
        siglip_feats = mx.where(use_null_image, mx.zeros_like(siglip_feats), siglip_feats)
        ip_embeds = adapter.get_image_embeds(siglip_feats)

        k_ip_all, v_ip_all = adapter.get_kv_all(ip_embeds)

        qs      = flux_state["qs"]       # list[25] [B, H, seq_img, Hd], stop_gradient
        h_final = flux_state["h_final"]  # [B, seq_img, d_inner], stop_gradient
        temb    = flux_state["temb"]     # [B, d_inner], stop_gradient
        B       = flux_state["B"]
        C       = flux_state["C"]
        Lh      = flux_state["Lh"]
        Lw      = flux_state["Lw"]
        pH      = flux_state["pH"]
        pW      = flux_state["pW"]
        seq_img = flux_state["seq_img"]
        d_inner = h_final.shape[2]

        # Accumulate IP contributions from all injection points.
        # Approximation: each contribution is added to the final hidden state
        # rather than at its original position in the sequence. This matches
        # the standard IP-Adapter training approach (no grad through base model).
        ip_total = mx.zeros((B, seq_img, d_inner), dtype=h_final.dtype)
        for i, q_i in enumerate(qs):
            H_i  = q_i.shape[1]
            Hd_i = q_i.shape[3]
            k_i  = k_ip_all[:, i].reshape(B, -1, H_i, Hd_i).transpose(0, 2, 1, 3)
            v_i  = v_ip_all[:, i].reshape(B, -1, H_i, Hd_i).transpose(0, 2, 1, 3)
            ip_out = mx.fast.scaled_dot_product_attention(
                q_i, k_i, v_i, scale=Hd_i ** -0.5,
            )  # [B, H, seq_img, Hd]
            ip_out = ip_out.transpose(0, 2, 1, 3).reshape(B, seq_img, H_i * Hd_i)
            ip_total = ip_total + adapter.scale[i] * ip_out

        h_with_ip = h_final + ip_total

        # norm_out + proj_out: frozen Flux layers needed for correct gradient signal.
        # Jacobian-vector products flow through these 2 ops (negligible trace cost).
        tr = flux.transformer
        h_with_ip = tr.norm_out(h_with_ip, temb)
        pred_seq  = tr.proj_out(h_with_ip)  # [B, seq_img, 128]

        # Unpatchify → [B, 32, H/8, W/8]
        pred = pred_seq.transpose(0, 2, 1).reshape(B, C * 4, pH, pW)
        pred = pred.reshape(B, C, 2, 2, pH, pW)
        pred = pred.transpose(0, 1, 4, 2, 5, 3)
        pred = pred.reshape(B, C, Lh, Lw)

        flow_loss = mx.mean((pred - target) ** 2)
        if _style_weight > 0.0 and x0_ref is not None:
            x0_pred = reconstruct_x0(noisy_in, pred, alpha_in, sigma_in)
            style_term = gram_style_loss(x0_pred, x0_ref.astype(mx.float32))
            _style_loss_accum[0] = style_term
            return flow_loss + _style_weight * style_term
        return flow_loss

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    # TP-002 (infeasible): mx.compile requires all inputs to be explicit args or
    # declared captured state. flux_state contains dynamically-created arrays passed
    # via a dict — MLX cannot track these, raises "uncaptured inputs." Not worth
    # working around since the adapter step is already ~0.0s (lazy graph tracing
    # is negligible; all real work is in mx.eval dominated by Flux forward at 5-6s).

    def compiled_step(siglip_feats, use_null_image, flux_state, target,
                      x0_ref=None, noisy_in=None, alpha_in=None, sigma_in=None):
        loss_val, grads = loss_and_grad(
            siglip_feats, use_null_image, flux_state, target,
            x0_ref, noisy_in, alpha_in, sigma_in,
        )
        # QUALITY-2: zero double-stream scale gradients so the optimizer never
        # updates indices 0.._nd-1, keeping them pinned at zero.
        if _freeze_double_stream:
            grads["scale"] = mx.concatenate([
                mx.zeros((_nd,), dtype=grads["scale"].dtype),
                grads["scale"][_nd:],
            ])
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=_grad_clip)
        optimizer.update(adapter, grads)
        # Return all lazy. Eval is split into two fences in the caller AFTER
        # del flux_state — this keeps flux_state out of the peak window.
        return loss_val, grad_norm

    def loss_fn_with_ip(siglip_feats, use_null_image: mx.array,
                        noisy: mx.array, text_embeds: mx.array, t_int: mx.array,
                        target: mx.array,
                        x0_ref=None, noisy_in=None, alpha_in=None, sigma_in=None):
        """
        TRAIN-6: full forward+backward with block-by-block IP injection.

        Eliminates the train/inference mismatch of the end-sum approximation.
        Each block's Q is computed from IP-conditioned hidden states, exactly
        matching the inference path (_flux_forward_with_ip).

        Gradient graph: adapter → k_ip/v_ip at each block → ip_out[i] →
        hidden_states[i+1] → ... → pred → loss. Flux block weights are frozen;
        backward traces through them but accumulates no gradient for them.
        """
        siglip_feats = mx.where(use_null_image, mx.zeros_like(siglip_feats), siglip_feats)
        ip_embeds = adapter.get_image_embeds(siglip_feats)
        k_ip_all, v_ip_all = adapter.get_kv_all(ip_embeds)
        pred = _flux_forward_with_ip(
            flux, noisy, text_embeds, t_int,
            k_ip_all, v_ip_all, adapter.scale,
            ckpt_double, ckpt_single,
        )
        flow_loss = mx.mean((pred - target) ** 2)
        if _style_weight > 0.0 and x0_ref is not None:
            x0_pred = reconstruct_x0(noisy_in, pred, alpha_in, sigma_in)
            style_term = gram_style_loss(x0_pred, x0_ref.astype(mx.float32))
            _style_loss_accum[0] = style_term
            return flow_loss + _style_weight * style_term
        return flow_loss

    loss_and_grad_with_ip = nn.value_and_grad(adapter, loss_fn_with_ip)

    def compiled_step_with_ip(siglip_feats, use_null_image, noisy, text_embeds, t_int, target,
                               x0_ref=None, noisy_in=None, alpha_in=None, sigma_in=None):
        loss_val, grads = loss_and_grad_with_ip(
            siglip_feats, use_null_image, noisy, text_embeds, t_int, target,
            x0_ref, noisy_in, alpha_in, sigma_in,
        )
        if _freeze_double_stream:
            grads["scale"] = mx.concatenate([
                mx.zeros((_nd,), dtype=grads["scale"].dtype),
                grads["scale"][_nd:],
            ])
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=_grad_clip)
        optimizer.update(adapter, grads)
        return loss_val, grad_norm

    # ── T-05: validation loss on held-out set ─────────────────────────────────
    def _compute_val_loss(with_cond_gap: bool = False):
        """Run a no-grad forward on up to 64 held-out records. Returns mean loss or None.

        with_cond_gap=True additionally evaluates every SigLIP-bearing record TWICE
        with the SAME noise/timestep — conditioned and null — and returns a dict
        {loss, loss_cond, loss_null, cond_gap, n_pairs}. The paired design makes the
        held-out cond_gap low-variance; it is the metric the flywheel should rank by
        (the in-training gap is a train-batch statistic — see code review M1)."""
        if not _val_shards:
            return None
        from ip_adapter.dataset import _load_vae_latent, _load_qwen3_embed, _load_siglip_embed
        import pipeline_lib as _plib
        # Val precomputed embeddings: check ultrahot tier first (staged there when
        # data_prep_tier=ultrahot), fall back to hot DATA_ROOT, then to training caches.
        _uh_vbase  = _plib.ULTRAHOT_ROOT / "validation" / "precomputed"
        _hot_vbase = Path(str(_plib.DATA_ROOT)) / "validation" / "precomputed"
        _vbase     = _uh_vbase if _uh_vbase.is_dir() else _hot_vbase
        _val_qwen3  = str(_vbase / "qwen3")  if (_vbase / "qwen3").is_dir()  else dcfg.get("qwen3_cache_dir")
        _val_vae    = str(_vbase / "vae")    if (_vbase / "vae").is_dir()    else dcfg.get("vae_cache_dir")
        _val_siglip = str(_vbase / "siglip") if (_vbase / "siglip").is_dir() else None
        losses = []
        _pair_cond: list = []
        _pair_null: list = []

        def _one_loss(_siglip, _use_null, _noisy, _txt, _t, _target):
            if _use_block_injection:
                _l = loss_fn_with_ip(_siglip, _use_null, _noisy, _txt, _t, _target)
            else:
                _fs = _flux_forward_no_ip(flux, _noisy, _txt, _t)
                mx.eval(_fs["qs"], _fs["h_final"], _fs["temb"], _target)
                _l = loss_fn(_siglip, _use_null, _fs, _target)
                del _fs
            mx.eval(_l)
            return float(_l.item())

        for _shard in _val_shards[:2]:
            try:
                import tarfile as _tf
                with _tf.open(_shard) as _tar:
                    _member_names = [m.name for m in _tar.getmembers() if m.isfile()]
            except Exception:
                continue
            _keys: dict = {}
            for _name in _member_names:
                _stem, _, _ext = _name.rpartition(".")
                _keys.setdefault(_stem, {})[_ext.lower()] = _name
            for _stem in list(_keys)[:32]:
                _txt_key = _keys[_stem].get("txt") or _keys[_stem].get("caption")
                if not _txt_key:
                    continue
                _vae_np  = _load_vae_latent(_stem, _val_vae, expected_hw=None)
                _text_np = _load_qwen3_embed(_stem, _val_qwen3)
                if _vae_np is None or _text_np is None:
                    continue
                _siglip_np = _load_siglip_embed(_stem, _val_siglip)
                # VAE-Q1: same raw->packed/BN convention as the training path.
                _lat = _bn_pack_latents(mx.array(_vae_np[None], dtype=mx.bfloat16),
                                        _bn_mean_r, _bn_std_r)
                _txt = mx.array(_text_np[None], dtype=mx.bfloat16)
                # Sample a random timestep so val loss reflects the full noise distribution.
                _t     = mx.clip((mx.sigmoid(mx.random.normal(shape=(1,))) * 1000).astype(mx.int32), 0, 999)
                _noise = mx.random.normal(_lat.shape, dtype=_lat.dtype)
                _alpha, _sigma = get_schedule_values(_t)
                _noisy, _target = fused_flow_noise(_lat, _noise, _alpha, _sigma)
                if _use_block_injection:
                    mx.eval(_target)
                if _siglip_np is not None:
                    _siglip = mx.array(_siglip_np[None], dtype=mx.bfloat16)
                    _lc = _one_loss(_siglip, mx.array(False), _noisy, _txt, _t, _target)
                    losses.append(_lc)
                    if with_cond_gap:
                        # Paired null pass: SAME noise/timestep, conditioning off —
                        # the difference isolates what the adapter contributes.
                        _ln = _one_loss(_siglip, mx.array(True), _noisy, _txt, _t, _target)
                        _pair_cond.append(_lc)
                        _pair_null.append(_ln)
                else:
                    _siglip = mx.zeros((1, 729, acfg["siglip_dim"]), dtype=mx.bfloat16)
                    losses.append(_one_loss(_siglip, mx.array(True), _noisy, _txt, _t, _target))
                del _lat, _txt, _t, _noise, _alpha, _sigma, _siglip, _noisy, _target
                mx.clear_cache()
                if len(losses) >= 64:
                    break
            if len(losses) >= 64:
                break

        _mean = sum(losses) / len(losses) if losses else None
        if not with_cond_gap:
            return _mean
        if not _pair_cond:
            return {"loss": _mean, "loss_cond": None, "loss_null": None,
                    "cond_gap": None, "n_pairs": 0}
        _lc = sum(_pair_cond) / len(_pair_cond)
        _ln = sum(_pair_null) / len(_pair_null)
        return {"loss": _mean, "loss_cond": _lc, "loss_null": _ln,
                "cond_gap": _ln - _lc, "n_pairs": len(_pair_cond)}

    # ── Live encoder load/unload for online (no-cache) mode ───────────────────
    # These keep Qwen3/VAE/SigLIP resident only during the ~100 ms feature-
    # extraction window at the top of each step, then drop their ~3.8 GB of
    # GPU buffers so the Flux transformer forward + adapter backward fit in the
    # MLX memory limit.  Called from the main training loop only.
    def _ensure_live_encoders(step: int = 0) -> None:
        """(Re)load any live encoders needed for this step's batch.
        Debug prints are emitted only when we actually perform (re)loads so the
        flywheel logs show exactly when the transient ~3.8 GB spike occurs.
        """
        nonlocal vae, text_encoder, siglip

        loaded_this_call = []

        # VAE (lives inside flux.vae when attached)
        if (not vae_cache_available) and vae is None:
            if getattr(flux, "vae", None) is None and _encoder_weights is not None:
                from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
                from mflux.models.common.weights.loading.weight_applier import WeightApplier
                from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
                comps = {c.name: c for c in Flux2KleinWeightDefinition.get_components()}
                vae_mod = Flux2VAE()
                WeightApplier.apply_and_quantize_single(
                    weights=_encoder_weights,
                    model=vae_mod,
                    component=comps["vae"],
                    quantize_arg=_encoder_quantize,
                )
                flux.vae = vae_mod
                vae = vae_mod
                loaded_this_call.append("VAE(re-attach)")
            elif getattr(flux, "vae", None) is not None:
                vae = flux.vae
                loaded_this_call.append("VAE(existing)")
            if vae is not None and hasattr(vae, "freeze"):
                vae.freeze()

        # Qwen3 text encoder (lives inside flux.text_encoder)
        if (not text_cache_available) and text_encoder is None:
            if getattr(flux, "text_encoder", None) is None and _encoder_weights is not None:
                from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
                from mflux.models.common.weights.loading.weight_applier import WeightApplier
                from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
                comps = {c.name: c for c in Flux2KleinWeightDefinition.get_components()}
                te_mod = Qwen3TextEncoder()
                WeightApplier.apply_and_quantize_single(
                    weights=_encoder_weights,
                    model=te_mod,
                    component=comps["text_encoder"],
                    quantize_arg=_encoder_quantize,
                )
                flux.text_encoder = te_mod
                # Rebuild the bundle (tokenizer lives on flux, safe to keep)
                tok = flux.tokenizers.get("qwen3") if hasattr(flux, "tokenizers") else None
                text_encoder = _TextEncoderBundle(te_mod, tok) if tok is not None else _TextEncoderBundle(te_mod, None)
                loaded_this_call.append("Qwen3(re-attach)")
            elif getattr(flux, "text_encoder", None) is not None:
                tok = flux.tokenizers.get("qwen3") if hasattr(flux, "tokenizers") else None
                text_encoder = _TextEncoderBundle(flux.text_encoder, tok) if tok is not None else None
                loaded_this_call.append("Qwen3(existing)")
            if text_encoder is not None and hasattr(text_encoder, "freeze"):
                text_encoder.freeze()

        # SigLIP (standalone, PyTorch or MLX) — always a full load when live
        if use_siglip_live and siglip is None:
            siglip = _load_siglip(mcfg["siglip_model"])
            if hasattr(siglip, "freeze"):
                siglip.freeze()
            loaded_this_call.append("SigLIP(full)")

        if loaded_this_call:
            print(f"[online-encode] step {step}: loaded {', '.join(loaded_this_call)} "
                  f"(transient models now resident for encode window only)", flush=True)

    def _release_live_encoders(step: int = 0) -> None:
        """Drop live encoders (free their GPU weight buffers) after encoding.
        This is the critical step that prevents the ~3.8 GB from sitting through
        the entire Flux forward + adapter backward (the part that was causing jetsam).
        """
        nonlocal vae, text_encoder, siglip
        vae = None
        text_encoder = None
        siglip = None
        # Detach from flux so the weight arrays become unreferenced and clear_cache can release.
        if hasattr(flux, "vae"):
            flux.vae = None
        if hasattr(flux, "text_encoder"):
            flux.text_encoder = None
        import gc
        gc.collect()
        mx.clear_cache()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        if step > 0:
            print(f"[online-encode] step {step}: released all live encoders + cleared GPU caches "
                  f"(~3.8 GB transient now eligible for reclaim before Flux fwd)", flush=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    _steps_this_run = _end_step - start_step
    print(f"\nTraining: {tcfg['num_steps']:,} steps (steps {start_step:,}→{_end_step:,}), batch_size={dcfg['batch_size']}\n")

    step = start_step
    t0 = time.time()
    t_start = time.time()
    log_interval = ocfg["log_every"]
    loss_history: list[float] = []  # rolling window for smoothed loss
    loss_smooth = 0.0
    target: Optional[mx.array] = None  # J-5: init before loop so del target is safe if loop runs 0 iterations
    _pipeline_chunk    = config.get("_chunk")    # set by --chunk arg; None when run standalone
    _pipeline_run_name = config.get("_run_name") if _pipeline_chunk is None else None
    # Heartbeat process key: 'trainer_{name}' for direct runs, 'trainer' for pipeline runs.
    # Direct-run heartbeats land in .heartbeat/trainer_{name}.json — visible to
    # pipeline_status.py and pipeline_doctor.py without polluting chunk-scoped files.
    _hb_process = f"trainer_{_pipeline_run_name}" if _pipeline_run_name else "trainer"

    # Per-phase timing accumulators (TP-005). Printed at each log interval.
    # fwd = Flux forward + mx.eval(flux_state) (no grad, the heavy part)
    # step = adapter-only compiled_step (backward, should be tiny)
    _t_data = _t_prep = _t_fwd = _t_step = _t_eval = 0.0
    _t_eval_end = None  # end of last mx.eval; None on first iteration

    # T-01: grad norm EMA (decay=0.98 ≈ 50-step half-life)
    grad_norm_smooth = 0.0
    _grad_ema_decay = 0.98

    # T-02: per-bucket throughput (steps, loss, wall time)
    bucket_stats: dict = defaultdict(lambda: {"steps": 0, "loss_sum": 0.0, "time_sum": 0.0})

    # T-03: memory snapshot (read at log intervals to avoid overhead)
    mem_used_gb: Optional[float] = None
    mem_available_gb: Optional[float] = None



    # T-05: validation loss on held-out set
    _val_shards: list[str] = []
    val_loss_last: Optional[float] = None
    val_loss_log_path = os.path.join(ocfg["checkpoint_dir"], "val_loss.jsonl")
    _val_every = tcfg.get("val_every", 1000)
    try:
        from pathlib import Path as _P
        import pipeline_lib as _plib
        # Check ultrahot tier first (staged there when data_prep_tier=ultrahot),
        # fall back to hot DATA_ROOT.
        _uh_held_out  = _plib.ULTRAHOT_ROOT / "validation" / "held_out"
        _hot_held_out = _P(str(_plib.DATA_ROOT)) / "validation" / "held_out"
        _held_out = _uh_held_out if _uh_held_out.exists() else _hot_held_out
        if _held_out.exists():
            _val_shards = sorted(str(p) for p in _held_out.glob("*.tar"))
            print(f"Validation held-out: {len(_val_shards)} shards in {_held_out}")
        else:
            print("Validation held-out: not found — T-05 disabled")
    except Exception:
        pass

    # Eval hook (eval.py) — runs every _eval_every steps when enabled.
    # Reuses the loaded flux model and current EMA params; does not reload weights.
    _eval_enabled = ecfg.get("enabled", False)
    _eval_every   = ecfg.get("every_steps", 10000)
    _eval_prompts = os.path.join(
        config.get("_config_dir", "."),
        "eval_prompts.txt",
    )
    if _eval_enabled:
        print(f"Eval hook: every {_eval_every} steps  prompts={_eval_prompts}", flush=True)

    # T-06: EMA vs online weight divergence
    ema_drift: float = 0.0

    # T-10: SigLIP coverage per batch window
    _siglip_miss_steps = 0
    _siglip_first_miss_logged = False

    # T-11: gradient clipping events per log window
    _grad_clip_steps = 0

    # T-12: conditioned vs unconditioned loss split (QUALITY-4)
    # Tracks whether the adapter is actually helping reconstruction.
    # Healthy: loss_cond drops below loss_null as training progresses.
    # Flat gap throughout → adapter not learning.
    _cond_loss_sum = 0.0
    _cond_loss_count = 0
    _null_loss_sum = 0.0
    _null_loss_count = 0

    # Style loss tracking (Gram matrix term when style_loss_weight > 0)
    _style_loss_sum = 0.0
    _style_loss_count = 0

    # QUALITY-6: cross-ref vs self-ref loss split (populated by QUALITY-1 permutation)
    _self_ref_loss_sum   = 0.0
    _self_ref_loss_count = 0
    _cross_ref_loss_sum   = 0.0
    _cross_ref_loss_count = 0
    # SREF-1 telemetry: of the cross-ref steps in each log window, how many used a
    # style-paired neighbor (vs the legacy arbitrary buffer swap).
    _style_pair_steps      = 0
    _cross_ref_steps_total = 0

    # QUALITY-1: single-step feature buffer — stores siglip_feats from the previous
    # conditioned step so cross-ref works at batch_size=1 (no within-batch permutation).
    _cross_ref_buffer: Optional[mx.array] = None

    # ── training_warmup: compile Metal PSO graphs for all bucket shapes then exit ──
    # Triggered by the orchestrator's training_warmup pipeline step, or by running
    # train_ip_adapter.py --warmup-only directly.  Runs the full forward+backward+
    # optimizer-eval cycle once per bucket so the Metal compiler populates its PSO
    # cache.  The cache persists across process restarts so subsequent training
    # sessions (including all chunk 2+ runs) start without any compilation delay.
    if config.get("_warmup_only"):
        _txt_dim_wu = flux.transformer.context_embedder.weight.shape[1]
        _txt_wu   = mx.zeros((1, 64, _txt_dim_wu), dtype=mx.bfloat16)
        _t_wu     = mx.array([500], dtype=mx.int32)
        print("Warming up IP Adapter training graphs (all bucket shapes)...")
        for _bH, _bW in BUCKETS:
            _lat_H, _lat_W = _bH // 8, _bW // 8
            _dummy_lat    = mx.zeros((1, 32, _lat_H, _lat_W), dtype=mx.bfloat16)
            _dummy_tgt    = mx.zeros((1, 32, _lat_H, _lat_W), dtype=mx.bfloat16)
            _dummy_siglip = mx.zeros((1, 729, acfg["siglip_dim"]), dtype=mx.bfloat16)
            print(f"  [{_bH}x{_bW}] compiling...", flush=True)
            _t0_wu = time.time()
            if _use_block_injection:
                _loss_wu, _gnorm_wu = compiled_step_with_ip(
                    _dummy_siglip, mx.array(False), _dummy_lat, _txt_wu, _t_wu, _dummy_tgt)
            else:
                _fs = _flux_forward_no_ip(flux, _dummy_lat, _txt_wu, _t_wu)
                mx.eval(_fs["qs"], _fs["h_final"], _fs["temb"])
                _loss_wu, _gnorm_wu = compiled_step(_dummy_siglip, mx.array(False), _fs, _dummy_tgt)
            mx.eval(optimizer.state, _loss_wu, _gnorm_wu)
            mx.clear_cache()
            mx.eval(adapter.parameters())
            mx.clear_cache()
            _elapsed = time.time() - _t0_wu
            _label = "cold compile" if _elapsed > 5.0 else "warm cache"
            print(f"  [{_bH}x{_bW}] done ({_elapsed:.1f}s — {_label})", flush=True)
            if not _use_block_injection:
                del _fs
            del _dummy_lat, _dummy_tgt, _dummy_siglip, _loss_wu, _gnorm_wu
        print("IP Adapter training graphs ready.")
        sys.exit(0)

    _boot_hb_stop.set()  # training loop starting — boot heartbeat thread no longer needed

    # Hoist write_heartbeat once — avoids repeated sys.modules lookup inside the loop.
    try:
        from pipeline_lib import write_heartbeat as _write_hb
    except ImportError:
        _write_hb = None  # type: ignore[assignment]

    # Heartbeat is written every _heartbeat_every steps, independent of log_every.
    # Capped at 50: at 0.22 steps/s that is ~227 s, well inside the 900 s stale
    # threshold.  The old cap of 200 (≈909 s) was right at the limit — any small
    # slowdown triggered a spurious orchestrator restart.
    _heartbeat_every = min(log_interval, 50)
    _hb_t0 = time.time()
    _hb_loss = 0.0
    _hb_loss_smooth = 0.0
    _hb_sps = 0.0
    _hb_siglip_cov = 100.0   # last known from log block; 100% until first log fires
    _hb_loader_pct = 0.0     # last known from log block

    # PIPE-H-003: cache-miss skip tracking.
    # Consecutive skips: warns after 20; resets to 0 on any successful step.
    # Rolling window: if skip rate > 50% over 500 batches, exit with error so
    # the orchestrator can escalate instead of spinning indefinitely.
    _skip_consecutive = 0
    _skip_window_total = 0   # batches seen in the rolling window
    _skip_window_skips = 0   # cache-miss skips in the rolling window
    _SKIP_WARN_AFTER   = 20
    _SKIP_WINDOW       = 500
    _SKIP_RATE_LIMIT   = 0.50

    for images_np, captions, text_np, vae_np, siglip_np, style_ref_np, bucket_hw in loader:
        if step >= _end_step:
            break

        # How long the GPU was idle waiting for data (time from end of last eval
        # to receiving this batch). Zero on first step (loader pre-filled).
        _t_now = time.time()
        # On resume (start_step > chunk_base_step), discard the gap between the
        # previous session's last eval and this session's first batch — it includes
        # idle time between crash and restart.  Also reset t0 so the first interval's
        # steps_per_sec excludes startup/warmup overhead.
        if step == start_step and start_step > chunk_base_step:
            _t_eval_end = None
            t0 = _t_now
        if _t_eval_end is not None:
            _step_data_wait = _t_now - _t_eval_end
            # Skip outliers caused by machine sleep (caffeinate failure etc.):
            # a wait ≥ 300 s between steps is stale, not real loader latency.
            if _step_data_wait < 300.0:
                _t_data += _step_data_wait

        bH, bW = bucket_hw

        _t_bucket_start = _t0 = time.time()
        # Convert to MLX — images are padded (H+32, W+32) for GPU crop augmentation
        images = mx.array(images_np, dtype=mx.bfloat16)   # [B, C, H+32, W+32]

        # GPU augmentation: random horizontal flip + random crop (§3.10)
        # Done in MLX on GPU — no CPU→GPU copy penalty
        images = augment_mlx(images, bH, bW)               # [B, C, bH, bW]

        # Null conditioning flags — decided OUTSIDE compiled region (§3.7)
        # Keep as Python bools for use in Python conditionals (no GPU sync).
        # Wrap in mx.array only for passing into the compiled loss fn (mx.where).
        null_image = random.random() < _image_dropout_prob
        null_text  = random.random() < _text_dropout_prob
        use_null_image = _MX_TRUE if null_image else _MX_FALSE

        # Encode frozen models — use pre-computed cache if available (§2.7)
        # In online (live-encode) mode, bring the encoders up only for this window.
        _did_live_encode = False
        if (not vae_cache_available and vae_np is None) or \
           (not text_cache_available and text_np is None) or \
           (use_siglip_live and siglip_np is None):
            _ensure_live_encoders(step=step)
            _did_live_encode = True

        if vae_np is not None:
            latents = _bn_pack_latents(mx.array(vae_np, dtype=mx.bfloat16),
                                       _bn_mean_r, _bn_std_r)
        elif vae is not None:
            latents = _bn_pack_latents(_vae_encode(vae, images),
                                       _bn_mean_r, _bn_std_r)
        else:
            # VAE not loaded (cache-only mode) but this batch has no cached latents.
            # Skip rather than crash — incomplete precompute coverage.
            _skip_consecutive += 1
            _skip_window_skips += 1
            _skip_window_total += 1
            if _skip_consecutive == _SKIP_WARN_AFTER:
                print(f"  WARNING: {_skip_consecutive} consecutive cache-miss skips at step {step} "
                      f"— check VAE precompute coverage.", flush=True)
            if _skip_window_total >= _SKIP_WINDOW and _skip_window_skips / _skip_window_total > _SKIP_RATE_LIMIT:
                print(f"ERROR: cache-miss skip rate {_skip_window_skips}/{_skip_window_total} "
                      f"({100*_skip_window_skips/_skip_window_total:.0f}%) exceeds 50% over last "
                      f"{_SKIP_WINDOW} batches — precompute coverage too low. Exiting.", flush=True)
                sys.exit(1)
            continue

        if text_np is not None:
            text_embeds = mx.array(text_np, dtype=mx.bfloat16)
            if null_text:
                text_embeds = mx.zeros_like(text_embeds)
        elif text_encoder is not None:
            text_embeds = _encode_text(text_encoder, captions)
            if null_text:
                text_embeds = mx.zeros_like(text_embeds)
        else:
            # Text encoder not loaded (cache-only mode) but no cached embeddings.
            _skip_consecutive += 1
            _skip_window_skips += 1
            _skip_window_total += 1
            if _skip_consecutive == _SKIP_WARN_AFTER:
                print(f"  WARNING: {_skip_consecutive} consecutive cache-miss skips at step {step} "
                      f"— check text precompute coverage.", flush=True)
            if _skip_window_total >= _SKIP_WINDOW and _skip_window_skips / _skip_window_total > _SKIP_RATE_LIMIT:
                print(f"ERROR: cache-miss skip rate {_skip_window_skips}/{_skip_window_total} "
                      f"({100*_skip_window_skips/_skip_window_total:.0f}%) exceeds 50% over last "
                      f"{_SKIP_WINDOW} batches — precompute coverage too low. Exiting.", flush=True)
                sys.exit(1)
            continue

        if siglip_np is not None:
            siglip_feats = mx.array(siglip_np, dtype=mx.bfloat16)
        elif siglip is not None:
            siglip_feats = siglip(_resize_images_for_siglip(images))
        else:
            # Cache miss with no live encoder — force null image conditioning so
            # the adapter sees exact zeros after mx.where, not Perceiver(zeros).
            if not null_image:
                # Only warn and count when this was NOT already a dropout step;
                # otherwise we'd double-count and inflate the apparent miss rate.
                if not _siglip_first_miss_logged:
                    print(f"  WARNING: SigLIP cache miss at step {step} — "
                          f"forcing null-image conditioning. Check siglip precompute coverage.",
                          flush=True)
                    _siglip_first_miss_logged = True
                _siglip_miss_steps += 1  # T-10: counts forced-null only, not intended dropout
            B_miss = images.shape[0]
            siglip_feats = mx.zeros((B_miss, 729, acfg["siglip_dim"]), dtype=mx.bfloat16)
            null_image = True
            use_null_image = _MX_TRUE

        # Online mode: drop the encoders (~3.8 GB) *now*, before Flux forward + adapter step.
        # They are only required for the feature extraction at the top of the step.
        if _did_live_encode:
            _release_live_encoders(step=step)

        # PIPE-H-003: successful batch — reset consecutive-skip counter and count
        # this batch in the rolling window (slide window after _SKIP_WINDOW batches).
        _skip_consecutive = 0
        _skip_window_total += 1
        if _skip_window_total >= _SKIP_WINDOW:
            _skip_window_total = 0
            _skip_window_skips = 0

        # QUALITY-3: patch-shuffle — shuffle 729 SigLIP token positions to destroy
        # spatial layout while preserving per-patch texture/color statistics.
        # Applied before cross-ref permutation so the shuffled features are what
        # the model receives; only on conditioned (non-null) steps.
        if _patch_shuffle_prob > 0.0 and not null_image:
            if random.random() < _patch_shuffle_prob:
                _perm_sf = mx.argsort(mx.random.uniform(shape=(siglip_feats.shape[1],)))
                siglip_feats = siglip_feats[:, _perm_sf, :]

        # Cross-ref: replace current SigLIP features with a DIFFERENT image's
        # features, forcing style/content separation.
        #   SREF-1 (preferred): the loader supplies a SAME-STYLE different-content
        #   neighbor (style_ref_np) — the reference's style genuinely predicts the
        #   target, so the adapter is rewarded for style transfer, not for ignoring
        #   the reference.
        #   Legacy fallback (QUALITY-1): the previous conditioned step's features —
        #   an arbitrary pairing, kept only for when no neighbor is available.
        is_cross_ref = False
        is_style_pair = False
        if _cross_ref_prob > 0.0 and not null_image:
            if random.random() < _cross_ref_prob:
                if style_ref_np is not None:
                    siglip_feats = mx.array(style_ref_np, dtype=mx.bfloat16)
                    is_cross_ref = is_style_pair = True
                elif _cross_ref_buffer is not None:
                    siglip_feats, _cross_ref_buffer = _cross_ref_buffer, siglip_feats
                    is_cross_ref = True
                else:
                    _cross_ref_buffer = siglip_feats
            else:
                _cross_ref_buffer = siglip_feats
        if is_style_pair:
            _style_pair_steps += 1
        if is_cross_ref:
            _cross_ref_steps_total += 1

        _t_prep += time.time() - _t0

        # ── Forward pass + backward ───────────────────────────────────────────
        _t0 = time.time()
        B_lat = latents.shape[0]
        # Logit-normal timestep sampling: concentrates on mid-timesteps (t≈300–700)
        # where flow-matching loss has structure. Reduces loss variance ~40% vs uniform.
        t_int = mx.clip((mx.sigmoid(mx.random.normal(shape=(B_lat,))) * 1000).astype(mx.int32), 0, 999)
        noise  = mx.random.normal(latents.shape, dtype=latents.dtype)
        alpha_t, sigma_t = get_schedule_values(t_int)
        noisy, target = fused_flow_noise(latents, noise, alpha_t, sigma_t)

        if _use_block_injection:
            # TRAIN-6: fwd is inside loss_fn_with_ip; only target needs materialising now.
            mx.eval(target)
            flux_state = None
        else:
            if _mem_profile:
                mx.reset_peak_memory()
            flux_state = _flux_forward_no_ip(flux, noisy, text_embeds, t_int)
            # Force-materialize all Flux tensors before entering autodiff graph.
            # Without this, MLX defers Flux computation into the gradient trace,
            # negating the entire split-forward optimization.
            mx.eval(flux_state["qs"], flux_state["h_final"], flux_state["temb"], target)
            if _use_correct_forward_q and not null_image:
                # Option C: replace Q vectors with IP-influenced versions.
                # Skipped for null-image steps — IP is zeroed in the loss anyway,
                # so IP-free Q is already correct for that case.
                _ip_embs_cq = adapter.get_image_embeds(siglip_feats)
                _k_cq, _v_cq = adapter.get_kv_all(_ip_embs_cq)
                mx.eval(_k_cq, _v_cq, adapter.scale)
                flux_state["qs"] = _flux_forward_with_ip_collect_q(
                    flux, noisy, text_embeds, t_int, _k_cq, _v_cq, adapter.scale,
                )
                mx.eval(flux_state["qs"])
            if _mem_profile:
                _mem_fwd_peak = max(_mem_fwd_peak, mx.get_peak_memory())
        _t_fwd += time.time() - _t0

        # Backward + optimizer update. With block injection (TRAIN-6) the full Flux
        # forward is traced inside loss_fn_with_ip, so n_grad_steps_per_fwd=1.
        for _grad_i in range(_n_grad_steps):
            if step >= _end_step:
                break
            _t_inner_start = time.time()

            _t0 = time.time()
            _do_style = _style_weight > 0.0 and not null_image and step % _style_every == 0
            if _use_block_injection:
                if _do_style:
                    loss_val, grad_norm_val = compiled_step_with_ip(
                        siglip_feats, use_null_image, noisy, text_embeds, t_int, target,
                        latents, noisy, alpha_t, sigma_t,
                    )
                else:
                    loss_val, grad_norm_val = compiled_step_with_ip(
                        siglip_feats, use_null_image, noisy, text_embeds, t_int, target,
                    )
            elif _do_style:
                loss_val, grad_norm_val = compiled_step(
                    siglip_feats, use_null_image, flux_state, target,
                    latents, noisy, alpha_t, sigma_t,
                )
            else:
                loss_val, grad_norm_val = compiled_step(
                    siglip_feats, use_null_image, flux_state, target,
                )
            _t_step += time.time() - _t0

            # EMA update (every 10 steps saves ~23 minutes; plans §3.11)
            # Built lazily here so it references the same lazy adapter.parameters()
            # that Fence 2 will concretize — Fence 3 then evaluates only the new EMA.
            _do_ema = (step % _ema_update_every == 0)
            if _do_ema:
                ema_params = update_ema(ema_params, adapter, decay=_ema_decay_factor)

            # Synchronous eval in two fences to bound peak Metal allocation.
            #
            # Fence 1 — backward + optimizer state + new params (merged):
            #   Grads keep optimizer state and new params live simultaneously, so
            #   splitting into two evals gives the same peak (~20.4 GB measured).
            #   Merging saves one eval round-trip and one clear_cache call.
            #   Measured peak (step 108k, 512px): 20.44 GB.
            #
            # Fence 2 — EMA (if due): old_ema + new_ema, isolated from param update.
            _t0 = time.time()
            if _mem_profile:
                mx.reset_peak_memory()
            mx.eval(optimizer.state, loss_val, grad_norm_val, adapter.parameters())
            mx.clear_cache()
            if _mem_profile:
                _mem_bwd_peak = max(_mem_bwd_peak, mx.get_peak_memory())
                mx.reset_peak_memory()
            if _do_ema:
                mx.eval(ema_params)
                mx.clear_cache()
                if _mem_profile:
                    _mem_ema_peak = max(_mem_ema_peak, mx.get_peak_memory())
                    mx.reset_peak_memory()
            _t_eval_end = time.time()
            _t_eval += _t_eval_end - _t0

            step += 1

            # Style loss accumulation — _style_loss_accum[0] was set inside loss_fn
            # and is already materialized by Fence 1 above (it's in loss_val's graph).
            if _do_style:
                _style_loss_sum += float(_style_loss_accum[0].item())
                _style_loss_count += 1

            # T-01: grad norm EMA + spike alert
            _gn = float(grad_norm_val.item())
            if grad_norm_smooth == 0.0:
                grad_norm_smooth = _gn
            else:
                grad_norm_smooth = _grad_ema_decay * grad_norm_smooth + (1 - _grad_ema_decay) * _gn
            if grad_norm_smooth > 0 and _gn > 10 * grad_norm_smooth:
                print(f"  WARNING: grad norm spike step {step}: "
                      f"{_gn:.3f} vs smooth {grad_norm_smooth:.3f}", flush=True)
            # T-11: count steps where clipping actually fired
            if _gn > _grad_clip:
                _grad_clip_steps += 1

            # T-02: per-bucket throughput
            _bk = f"{bH}x{bW}"
            _step_loss_val = float(loss_val.item())
            bucket_stats[_bk]["steps"]    += 1
            bucket_stats[_bk]["loss_sum"] += _step_loss_val
            bucket_stats[_bk]["time_sum"] += _t_eval_end - _t_inner_start

            # T-12: conditioned vs unconditioned loss split
            if null_image:
                _null_loss_sum += _step_loss_val
                _null_loss_count += 1
            else:
                _cond_loss_sum += _step_loss_val
                _cond_loss_count += 1
                # QUALITY-6: cross-ref vs self-ref split within conditioned steps
                if is_cross_ref:
                    _cross_ref_loss_sum   += _step_loss_val
                    _cross_ref_loss_count += 1
                else:
                    _self_ref_loss_sum   += _step_loss_val
                    _self_ref_loss_count += 1

            # T-06: EMA drift (RMS diff between online and EMA weights, top-5 by size)
            # Sort by tensor element count descending so we sample weight matrices
            # rather than bias vectors — gives a more representative drift signal.
            if step % 500 == 0:
                try:
                    _flat_online = dict(_flatten(adapter.parameters()))
                    _flat_ema    = dict(_flatten(ema_params))
                    _drift_vals  = []
                    _sorted_keys = sorted(
                        _flat_online.keys(),
                        key=lambda _k: _flat_online[_k].size,
                        reverse=True,
                    )
                    for _k in _sorted_keys[:5]:
                        if _k in _flat_ema:
                            _a = _flat_online[_k].astype(mx.float32)
                            _b = _flat_ema[_k].astype(mx.float32)
                            if _a.shape == _b.shape:
                                _drift_vals.append(float(mx.mean((_a - _b) ** 2).item() ** 0.5))
                    if _drift_vals:
                        ema_drift = sum(_drift_vals) / len(_drift_vals)
                except Exception:
                    pass

            # T-05: validation loss on held-out set
            if _val_shards and step % _val_every == 0:
                val_loss_last = _compute_val_loss()
                if val_loss_last is not None:
                    try:
                        os.makedirs(os.path.dirname(val_loss_log_path), exist_ok=True)
                        with open(val_loss_log_path, "a") as _vf:
                            json.dump({"step": step, "val_loss": round(val_loss_last, 6),
                                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, _vf)
                            _vf.write("\n")
                    except OSError:
                        pass
                    print(f"  val_loss={val_loss_last:.4f} (step {step})", flush=True)

            # Logging
            if step % log_interval == 0:
                elapsed = time.time() - t0
                steps_per_sec = log_interval / elapsed
                loss_scalar = float(loss_val.item())
                lr_now = float(optimizer.learning_rate)
                loss_history.append(loss_scalar)
                if len(loss_history) > 20:
                    loss_history.pop(0)
                loss_smooth = sum(loss_history) / len(loss_history)
                steps_remaining = _end_step - step
                eta_s = steps_remaining / steps_per_sec if steps_per_sec > 0 else 0
                eta_h, eta_m = divmod(int(eta_s) // 60, 60)
                print(
                    f"step {step:>7,}/{_end_step:,}"
                    f"  loss {loss_scalar:.4f} (avg {loss_smooth:.4f})"
                    f"  lr {lr_now:.2e}"
                    f"  {steps_per_sec:.2f} steps/s"
                    f"  ETA {eta_h}h{eta_m:02d}m",
                    flush=True,
                )
                # TP-005: per-phase timing breakdown
                # fwd = Flux forward + flux_state eval (no grad)
                # step = compiled_step: backward + optimizer m/v eval (now includes bulk of eval time)
                # eval = outer Fence 1: just new_params (~2 GB), should be small
                total_phase = _t_data + _t_prep + _t_fwd + _t_step + _t_eval
                if total_phase > 0:
                    print(
                        f"  timing/{log_interval}steps:"
                        f"  data={_t_data:.1f}s ({100*_t_data/total_phase:.0f}%)"
                        f"  prep={_t_prep:.1f}s ({100*_t_prep/total_phase:.0f}%)"
                        f"  fwd={_t_fwd:.1f}s ({100*_t_fwd/total_phase:.0f}%)"
                        f"  step={_t_step:.1f}s ({100*_t_step/total_phase:.0f}%)"
                        f"  eval={_t_eval:.1f}s ({100*_t_eval/total_phase:.0f}%)",
                        flush=True,
                    )
                # T-07: loader wait fraction — computed before zeroing accumulators
                _loader_pct = round(100 * _t_data / total_phase, 1) if total_phase > 0 else 0.0
                _loader_wait_ms = round(_t_data * 1000 / log_interval, 1)
                _compute_ms = round((_t_prep + _t_fwd + _t_step + _t_eval) * 1000 / log_interval, 1)
                if _loader_pct > 20:
                    print(f"  WARNING: loader wait {_loader_pct:.0f}% — consider increasing prefetch_batches",
                          flush=True)
                _t_data = _t_prep = _t_fwd = _t_step = _t_eval = 0.0

                # T-01: grad norm at log interval
                print(f"  grad_norm {_gn:.3f}  (smooth {grad_norm_smooth:.3f})", flush=True)

                # T-02: per-bucket summary at log interval
                _bkt_summary = {}
                for _bk, _bv in sorted(bucket_stats.items()):
                    _n = _bv["steps"]
                    if _n > 0:
                        _bkt_summary[_bk] = {
                            "steps": _n,
                            "loss_avg": round(_bv["loss_sum"] / _n, 4),
                            "secs_avg": round(_bv["time_sum"] / _n, 2),
                        }
                if _bkt_summary:
                    _bkt_str = "  ".join(
                        f"{k}:{v['steps']}steps loss={v['loss_avg']:.4f} {v['secs_avg']:.1f}s/step"
                        for k, v in _bkt_summary.items()
                    )
                    print(f"  buckets: {_bkt_str}", flush=True)

                # T-03: memory pressure (peak read here; reset deferred until after
                # the checkpoint save so the interval peak captures both training
                # AND checkpoint serialization spikes in the same window).
                # When memory_profile=true, intra-step resets make the interval peak
                # meaningless; per-fence data below replaces it.
                mlx_active_gb  = round(mx.get_active_memory()  / 1e9, 2)
                mlx_peak_gb    = round(mx.get_peak_memory()    / 1e9, 2)
                if _mem_profile:
                    print(f"  mlx_mem: active={mlx_active_gb:.2f} GB  (per-fence peaks below)",
                          flush=True)
                else:
                    print(f"  mlx_mem: active={mlx_active_gb:.2f} GB  peak={mlx_peak_gb:.2f} GB",
                          flush=True)
                if _HAS_PSUTIL:
                    _vm = _psutil.virtual_memory()
                    mem_used_gb = round(_vm.used / 1e9, 1)
                    mem_available_gb = round(_vm.available / 1e9, 1)
                    print(f"  sys_mem: {mem_used_gb:.1f} GB used  {mem_available_gb:.1f} GB free",
                          flush=True)
                    if mem_available_gb < 6.0:
                        print(f"  WARNING: memory pressure — only {mem_available_gb:.1f} GB available",
                              flush=True)
                # TRAIN-5 Stage 0: per-fence peak breakdown (enabled by memory_profile: true).
                # fwd  = Flux forward (noisy+target+all Flux intermediates)
                # bwd  = adapter backward + optimizer m/v state
                # param = new adapter params from concrete m/v
                # ema  = EMA update (only on ema_update_every steps)
                if _mem_profile:
                    _gb = 1 / 2**30
                    print(
                        f"  [mem/fence] fwd={_mem_fwd_peak*_gb:.2f} GB"
                        f"  bwd+param={_mem_bwd_peak*_gb:.2f} GB"
                        f"  ema={_mem_ema_peak*_gb:.2f} GB"
                        f"  (peak-of-{log_interval}-steps per fence)",
                        flush=True,
                    )
                    _mem_fwd_peak = _mem_bwd_peak = _mem_ema_peak = 0

                # T-06: EMA drift
                if ema_drift > 0:
                    print(f"  ema_drift={ema_drift:.5f}", flush=True)

                # T-10: SigLIP coverage
                _siglip_cov = round(100 * (1 - _siglip_miss_steps / log_interval), 1)
                if _siglip_miss_steps > 0:
                    print(f"  siglip_coverage={_siglip_cov:.0f}% ({_siglip_miss_steps}/{log_interval} steps missing)",
                          flush=True)
                    if _siglip_cov < 90:
                        print(f"  WARNING: SigLIP coverage below 90%", flush=True)
                _siglip_miss_steps = 0
                # Keep inter-log heartbeats up-to-date with latest quality metrics.
                _hb_siglip_cov = _siglip_cov
                _hb_loader_pct = _loader_pct

                # T-11: gradient clipping report
                _grad_clip_pct_for_hb = round(100 * _grad_clip_steps / log_interval, 1)
                if _grad_clip_steps > 0:
                    _clip_pct = _grad_clip_pct_for_hb
                    print(f"  grad_clipped={_clip_pct:.0f}% ({_grad_clip_steps}/{log_interval} steps "
                          f"had norm > {_grad_clip})", flush=True)
                _grad_clip_steps = 0

                # T-12: conditioned vs unconditioned loss split (QUALITY-4)
                _loss_cond_avg = round(_cond_loss_sum / _cond_loss_count, 4) if _cond_loss_count > 0 else None
                _loss_null_avg = round(_null_loss_sum / _null_loss_count, 4) if _null_loss_count > 0 else None
                if _loss_cond_avg is not None and _loss_null_avg is not None:
                    _cond_gap = _loss_null_avg - _loss_cond_avg
                    _cond_gap_pct = round(100 * _cond_gap / _loss_null_avg, 1) if _loss_null_avg > 0 else 0
                    print(
                        f"  loss_cond={_loss_cond_avg:.4f}  loss_null={_loss_null_avg:.4f}"
                        f"  gap={_cond_gap:+.4f} ({_cond_gap_pct:+.1f}%)"
                        f"  [n={_cond_loss_count}/{_null_loss_count}]",
                        flush=True,
                    )
                    if _cond_gap_pct < 1.0 and step > 1000:
                        print("  WARNING: loss_cond ≈ loss_null — adapter may not be learning", flush=True)
                _cond_loss_sum = _cond_loss_count = 0
                _null_loss_sum = _null_loss_count = 0

                # SREF-1: style-paired share of cross-ref steps this window.
                if _cross_ref_steps_total > 0:
                    _sp_pct = round(100 * _style_pair_steps / _cross_ref_steps_total, 1)
                    print(f"  style_pair={_sp_pct:.0f}% "
                          f"({_style_pair_steps}/{_cross_ref_steps_total} cross-ref steps)",
                          flush=True)
                _style_pair_steps = 0
                _cross_ref_steps_total = 0

                # Style loss (Gram matrix term)
                if _style_weight > 0.0 and _style_loss_count > 0:
                    _style_loss_avg = round(_style_loss_sum / _style_loss_count, 6)
                    print(
                        f"  style_loss={_style_loss_avg:.6f}"
                        f"  (weight={_style_weight}, {_style_loss_count}/{log_interval} steps)",
                        flush=True,
                    )
                else:
                    _style_loss_avg = None
                _style_loss_sum = _style_loss_count = 0

                # QUALITY-6: cross-ref vs self-ref loss split
                _loss_self_ref_avg  = round(_self_ref_loss_sum  / _self_ref_loss_count,  4) if _self_ref_loss_count  > 0 else None
                _loss_cross_ref_avg = round(_cross_ref_loss_sum / _cross_ref_loss_count, 4) if _cross_ref_loss_count > 0 else None
                if _loss_self_ref_avg is not None or _loss_cross_ref_avg is not None:
                    _ref_parts = []
                    if _loss_self_ref_avg is not None:
                        _ref_parts.append(f"self={_loss_self_ref_avg:.4f} [n={_self_ref_loss_count}]")
                    if _loss_cross_ref_avg is not None:
                        _ref_parts.append(f"cross={_loss_cross_ref_avg:.4f} [n={_cross_ref_loss_count}]")
                    if _loss_self_ref_avg is not None and _loss_cross_ref_avg is not None:
                        _ref_gap = _loss_cross_ref_avg - _loss_self_ref_avg
                        _ref_parts.append(f"gap={_ref_gap:+.4f}")
                        if _ref_gap < 0 and step > 1000:
                            _ref_parts.append("WARNING: cross_ref < self_ref — unexpected, check augmentation")
                    print(f"  loss_ref: {'  '.join(_ref_parts)}", flush=True)
                _self_ref_loss_sum  = _self_ref_loss_count  = 0
                _cross_ref_loss_sum = _cross_ref_loss_count = 0

                # T-13: adapter scale magnitudes (QUALITY-5)
                _scale_all_mean = _scale_double_mean = _scale_single_mean = None
                try:
                    _scales = adapter.scale.astype(mx.float32).tolist()
                    _scale_double = _scales[:5]
                    _scale_single = _scales[5:]
                    _scale_all_mean   = round(sum(_scales)        / len(_scales),        4)
                    _scale_double_mean = round(sum(_scale_double) / len(_scale_double),  4)
                    _scale_single_mean = round(sum(_scale_single) / len(_scale_single),  4)
                    _scale_min = round(min(_scales), 4)
                    _scale_max = round(max(_scales), 4)
                    print(
                        f"  ip_scale: mean={_scale_all_mean:.4f}"
                        f"  double={_scale_double_mean:.4f}  single={_scale_single_mean:.4f}"
                        f"  range=[{_scale_min:.4f}, {_scale_max:.4f}]",
                        flush=True,
                    )
                    if _scale_max > 2.0:
                        print(f"  WARNING: ip_scale max {_scale_max:.2f} > 2.0 — content leakage risk",
                              flush=True)
                    if _scale_all_mean < 0.05 and step > 500:
                        print(f"  WARNING: ip_scale mean {_scale_all_mean:.4f} near zero — adapter not active",
                              flush=True)
                except Exception:
                    pass

                if wandb_run:
                    wandb_run.log(
                        {"loss": loss_scalar, "loss_smooth": loss_smooth,
                         "lr": lr_now, "steps_per_sec": steps_per_sec,
                         "grad_norm": _gn, "grad_norm_smooth": grad_norm_smooth,
                         "ema_drift": ema_drift, "siglip_coverage_pct": _siglip_cov,
                         "loader_wait_pct": _loader_pct,
                         "loss_cond": _loss_cond_avg, "loss_null": _loss_null_avg,
                         "style_loss": _style_loss_avg,
                         "ip_scale_mean": _scale_all_mean,
                         "ip_scale_double": _scale_double_mean,
                         "ip_scale_single": _scale_single_mean},
                        step=step,
                    )
                # Write machine-readable heartbeat to the pipeline location so the
                # orchestrator and pipeline_status.py can monitor liveness and anomalies.
                total_elapsed = time.time() - t_start
                try:
                    if _write_hb is not None:
                        _write_hb(
                            _hb_process, _pipeline_chunk,
                            step=step,
                            total_steps=_end_step,
                            loss=round(loss_scalar, 6),
                            loss_smooth=round(loss_smooth, 6),
                            lr=lr_now,
                            steps_per_sec=round(steps_per_sec, 4),
                            eta_sec=int(eta_s),
                            elapsed_seconds=int(total_elapsed),
                            grad_norm=round(_gn, 4),
                            grad_norm_smooth=round(grad_norm_smooth, 4),
                            grad_clip_pct=_grad_clip_pct_for_hb,
                            ema_drift=round(ema_drift, 5),
                            val_loss=round(val_loss_last, 6) if val_loss_last is not None else None,
                            siglip_coverage_pct=_siglip_cov,
                            loader_wait_ms_avg=_loader_wait_ms,
                            compute_ms_avg=_compute_ms,
                            buckets=_bkt_summary,
                            mem_used_gb=mem_used_gb,
                            mem_available_gb=mem_available_gb,
                            mlx_active_gb=mlx_active_gb,
                            mlx_peak_gb=mlx_peak_gb,
                            loss_cond=_loss_cond_avg,
                            loss_null=_loss_null_avg,
                            style_loss=_style_loss_avg,
                            loss_self_ref=_loss_self_ref_avg,
                            loss_cross_ref=_loss_cross_ref_avg,
                            ip_scale_mean=_scale_all_mean,
                            ip_scale_double=_scale_double_mean,
                            ip_scale_single=_scale_single_mean,
                        )
                except Exception:
                    pass
                # Reset rolling stats so per-interval averages don't drift over the
                # entire run as loss improves and bucket distribution shifts.
                bucket_stats.clear()
                # Release unused Metal buffer pool entries every log interval.
                # MLX pools buffers internally; without periodic clearing, pool
                # growth can exhaust unified memory after ~1200 steps (~12 evals).
                mx.clear_cache()
                import gc; gc.collect()
                t0 = time.time()

            # Heartbeat — decoupled from log_every so the orchestrator always sees
            # liveness even when log_every is large.  The full rich heartbeat is
            # written by the log block above at log_interval steps; this lightweight
            # one fires at _heartbeat_every (≤50) steps for the intervals between.
            if step % _heartbeat_every == 0 and step > 0:
                _hb_elapsed = time.time() - _hb_t0
                _hb_t0 = time.time()
                _hb_sps = _heartbeat_every / _hb_elapsed if _hb_elapsed > 0 else 0.0
                _hb_loss = float(loss_val.item())
                _hb_loss_smooth = 0.98 * _hb_loss_smooth + 0.02 * _hb_loss
                if step % log_interval != 0:
                    try:
                        if _write_hb is not None:
                            _write_hb(
                                _hb_process, _pipeline_chunk,
                                step=step,
                                total_steps=_end_step,
                                loss=round(_hb_loss, 6),
                                loss_smooth=round(_hb_loss_smooth, 6),
                                lr=float(optimizer.learning_rate),
                                steps_per_sec=round(_hb_sps, 4),
                                eta_sec=int((_end_step - step) / _hb_sps) if _hb_sps > 0 else 0,
                                elapsed_seconds=int(time.time() - t_start),
                                grad_norm=round(_gn, 4),
                                grad_norm_smooth=round(grad_norm_smooth, 4),
                                siglip_coverage_pct=_hb_siglip_cov,
                                loader_wait_pct=_hb_loader_pct,
                            )
                    except Exception:
                        pass

            # Checkpoint (async background write; plans §3.12)
            if step % ocfg["checkpoint_every"] == 0:
                _lineage = {**_lineage_base, "step": step,
                            "loss": round(loss_smooth, 6)}
                save_checkpoint(adapter, ema_params, step,
                                      ocfg["checkpoint_dir"], ocfg["keep_last_n"],
                                      lineage=_lineage)
                # Write a heartbeat after the potentially slow checkpoint save so
                # the orchestrator doesn't see a stale heartbeat and restart us.
                try:
                    if _write_hb is not None:
                        _write_hb(_hb_process, _pipeline_chunk, step=step,
                                  total_steps=_end_step,
                                  loss=round(_hb_loss, 6),
                                  steps_per_sec=round(_hb_sps, 4),
                                  eta_sec=int((_end_step - step) / _hb_sps) if _hb_sps > 0 else 0)
                except Exception:
                    pass

            # Eval hook: generate images + compute CLIP-I/T every _eval_every steps.
            # Runs after checkpoint save so the just-written checkpoint is coherent.
            # Uses the in-memory flux model and current EMA params (no reload).
            if _eval_enabled and step % _eval_every == 0 and step > 0:
                # Eval generation uses flux's high-level path (prompt encoding + VAE decode),
                # so the encoders must be attached for the duration of the eval.
                _eval_needed_encoders = (not vae_cache_available) or (not text_cache_available) or use_siglip_live
                if _eval_needed_encoders:
                    _ensure_live_encoders(step=step)
                    # Re-attach explicitly in case _ensure only set the python vars
                    if vae is not None:
                        flux.vae = vae
                    if text_encoder is not None:
                        flux.text_encoder = text_encoder.encoder if hasattr(text_encoder, "encoder") else text_encoder
                try:
                    from eval import run_eval as _run_eval
                    _eval_out = os.path.join(ocfg["checkpoint_dir"], "eval", f"step_{step:07d}")
                    _eval_summary = _run_eval(
                        flux=flux,
                        adapter_cfg=acfg,
                        adapter_params=dict(_flatten(ema_params)),
                        prompts_file=_eval_prompts,
                        output_dir=_eval_out,
                        step=step,
                        width=ecfg.get("width", 512),
                        height=ecfg.get("height", 512),
                        n_steps=ecfg.get("num_steps", 4),
                        seed=ecfg.get("seed", 42),
                        siglip_model_name=mcfg["siglip_model"],
                        vae=flux.vae,
                    )
                    if wandb_run and _eval_summary:
                        wandb_run.log(
                            {"eval/clip_i": _eval_summary.get("mean_clip_i"),
                             "eval/clip_t": _eval_summary.get("mean_clip_t")},
                            step=step,
                        )
                except Exception as _eval_err:
                    print(f"  WARNING: eval hook failed at step {step}: {_eval_err}", flush=True)
                if _eval_needed_encoders:
                    _release_live_encoders(step=step)
                mx.clear_cache()

            # Peak reset deferred until after checkpoint so the reported peak
            # covers the full interval (training + serialization spike).
            if step % log_interval == 0:
                mx.reset_peak_memory()

        # Release flux_state (25 Q tensors ≈ 300 MB) after all gradient steps.
        if flux_state is not None:
            del flux_state
        del target


    # Final checkpoint + EMA export
    # skip_checkpoint_save=true suppresses all final writes (used by ablation harness
    # to avoid writing ~8 GB per short run that would be deleted immediately).
    if not ocfg.get("skip_checkpoint_save", False):
        _lineage = {**_lineage_base, "step": step, "loss": round(loss_smooth, 6)}
        save_checkpoint(adapter, ema_params, step,
                              ocfg["checkpoint_dir"], keep_last_n=999,
                              lineage=_lineage)

        # Save best EMA as final export using the streaming writer so the bulk
        # mx.save_safetensors staging buffer (~2 GB) is avoided.
        mx.eval(ema_params)
        _best_path = os.path.join(ocfg["checkpoint_dir"], "best.safetensors")
        _best_tmp  = _best_path + ".tmp"
        _save_safetensors_streaming(_best_tmp, list(_flatten(ema_params)))
        _purge_file_page_cache(_best_tmp)
        os.rename(_best_tmp, _best_path)

        # Write lineage sidecar so best.safetensors is self-documenting.
        _best_meta = {**_lineage_base, "step": step, "loss": round(loss_smooth, 6),
                      "chunk": _pipeline_chunk}
        try:
            _best_json_path = os.path.splitext(_best_path)[0] + ".json"
            _best_json_tmp  = _best_json_path + ".tmp"
            with open(_best_json_tmp, "w") as _f:
                json.dump(_best_meta, _f, indent=2)
            os.rename(_best_json_tmp, _best_json_path)
        except OSError:
            pass
        print(f"\nTraining complete. EMA weights: {ocfg['checkpoint_dir']}/best.safetensors")
    else:
        print(f"\nTraining complete. (skip_checkpoint_save=true — no weights written)")

    # ── Held-out cond_gap (code review M1) ────────────────────────────────────
    # Paired conditioned/null eval on the held-out val set. This line is parsed
    # by flywheel_lib.collect_metrics_from_log (RE_VAL_COND) and, when present,
    # REPLACES the in-training cond_gap for champion selection and shard
    # attribution — the train-batch gap measures fit to the trained shards, not
    # generalization. ~2x N forwards; runs once, at the very end.
    try:
        _val_final = _compute_val_loss(with_cond_gap=True)
        if isinstance(_val_final, dict) and _val_final.get("cond_gap") is not None:
            print(f"VAL loss_cond={_val_final['loss_cond']:.4f} "
                  f"loss_null={_val_final['loss_null']:.4f} "
                  f"cond_gap={_val_final['cond_gap']:+.4f} "
                  f"[n={_val_final['n_pairs']}] (held-out)", flush=True)
        else:
            print("VAL held-out cond_gap unavailable (no val set or no SigLIP pairs)",
                  flush=True)
    except Exception as _val_exc:  # noqa: BLE001 — the checkpoint is already saved;
        # a val-eval failure must degrade to "no held-out signal" (downstream falls
        # back to the train-batch gap), never fail the whole iteration.
        print(f"VAL held-out eval failed ({_val_exc}) — falling back to train-batch gap",
              flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# SigLIP wrappers
# ─────────────────────────────────────────────────────────────────────────────

class _SigLIPMLXWrapper:
    """Thin wrapper around an mlx_vlm SigLIP vision model."""

    def __init__(self, vision_model):
        self._model = vision_model

    def freeze(self):
        if hasattr(self._model, "freeze"):
            self._model.freeze()

    def __call__(self, images: mx.array) -> mx.array:
        return self._model(images).astype(mx.bfloat16)  # [B, 729, 1152]


class _SigLIPTorchWrapper:
    """Wraps a transformers SigLIP vision model with numpy bridge for MLX."""

    def __init__(self, hf_model):
        self._model = hf_model

    def freeze(self):
        pass  # always frozen; no MLX-style freeze needed

    def __call__(self, images: mx.array) -> mx.array:
        import torch
        import numpy as np
        imgs_np = np.array(images.astype(mx.float32))
        imgs_t = torch.from_numpy(imgs_np)
        with torch.no_grad():
            out = self._model(pixel_values=imgs_t).last_hidden_state  # [B, 729, 1152]
        return mx.array(out.numpy()).astype(mx.bfloat16)


# ─────────────────────────────────────────────────────────────────────────────
# Text encoder bundle
# ─────────────────────────────────────────────────────────────────────────────

class _TextEncoderBundle:
    """Bundles Qwen3 encoder + tokenizer so _encode_text can tokenise and encode."""

    def __init__(self, encoder, tokenizer):
        self.encoder = encoder
        self.tokenizer = tokenizer

    def freeze(self):
        self.encoder.freeze()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resize_images_for_siglip(images: "mx.array") -> "mx.array":
    """Resize [B, 3, H, W] bf16 in [-1,1] → [B, 3, SIGLIP_IMAGE_SIZE, SIGLIP_IMAGE_SIZE] bf16 in [-1,1].

    SigLIP SO400M was pretrained on 384×384 inputs. This helper ensures the
    live-encode path matches the precompute path, which explicitly resizes to
    SIGLIP_IMAGE_SIZE before encoding. Only called when siglip_cache_dir is None.
    Normalisation: pixel/127.5 - 1.0  ==  (pixel/255 - SIGLIP_MEAN) / SIGLIP_STD.
    """
    from PIL import Image as _PilImage
    imgs_np = np.array(images.astype(mx.float32))  # [B, 3, H, W]
    out = np.empty((imgs_np.shape[0], 3, SIGLIP_IMAGE_SIZE, SIGLIP_IMAGE_SIZE), dtype=np.float32)
    for i in range(imgs_np.shape[0]):
        img = imgs_np[i].transpose(1, 2, 0)                              # [H, W, 3]
        img = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        img = np.array(_PilImage.fromarray(img).resize((SIGLIP_IMAGE_SIZE, SIGLIP_IMAGE_SIZE), _PilImage.LANCZOS))
        out[i] = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)  # HWC→CHW, [-1,1]
    return mx.array(out, dtype=mx.bfloat16)


def _load_siglip(model_name: str):
    """
    Load SigLIP SO400M vision encoder (frozen).
    Tries mlx_vlm first; falls back to transformers + torch.

    Input:  [B, 3, 384, 384] normalised to [-1, 1]
    Output: [B, 729, 1152]

    In practice SigLIP features are pre-computed by precompute_siglip.py.
    This path is only hit when siglip_cache_dir is None.
    """
    try:
        from mlx_vlm import load as vlm_load
        model, _ = vlm_load(model_name)
        model.eval()
        vm = model.vision_model if hasattr(model, "vision_model") else model
        return _SigLIPMLXWrapper(vm)
    except Exception:
        pass

    try:
        from transformers import AutoModel
        hf_model = AutoModel.from_pretrained(model_name).vision_model.eval()
        return _SigLIPTorchWrapper(hf_model)
    except Exception as e:
        raise RuntimeError(
            f"Cannot load SigLIP '{model_name}': {e}\n"
            "Install mlx-vlm: pip install mlx-vlm\n"
            "Or pre-compute features: python train/scripts/precompute_siglip.py "
            "--shards train/data/shards --output train/data/precomputed/siglip"
        ) from e


def _load_vae(flux):
    """Extract VAE encoder from the loaded Flux object."""
    return flux.vae


def _load_text_encoder(flux):
    """Return a _TextEncoderBundle (encoder + tokenizer) from the Flux object."""
    return _TextEncoderBundle(flux.text_encoder, flux.tokenizers["qwen3"])


# ─────────────────────────────────────────────────────────────────────────────
# Encode helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_vae_bn_stats(flux_model_dir: str):
    """Load the VAE BatchNorm running stats as [32,2,2] mean / std arrays (VAE-Q1).

    Returns (bn_mean_r, bn_std_r) or (None, None) if unavailable. The 128 BN
    features map to (latent_channel c, patch-row pi, patch-col pj) as
    feature = c*4 + pi*2 + pj — the same channel order iris_patchify / the trainer's
    pack step use — so reshaping to [32,2,2] lets a spatial tile apply the right
    per-feature stat at each (h%2, w%2). eps = 1e-4 (config batch_norm_eps)."""
    try:
        import os as _os
        path = _os.path.join(flux_model_dir, "vae", "diffusion_pytorch_model.safetensors")
        st = mx.load(path)
        bn_mean_r = st["bn.running_mean"].astype(mx.float32).reshape(32, 2, 2)
        bn_std_r = mx.sqrt(st["bn.running_var"].astype(mx.float32).reshape(32, 2, 2) + 1e-4)
        mx.eval(bn_mean_r, bn_std_r)
        return bn_mean_r, bn_std_r
    except Exception as e:
        print(f"  WARNING: VAE-Q1 bn stats unavailable ({e}); latents will NOT be "
              f"BN-packed — training space will not match C inference", flush=True)
        return None, None


def _bn_pack_latents(lat: mx.array, bn_mean_r, bn_std_r) -> mx.array:
    """Map a raw VAE-latent [B,32,Lh,Lw] (mflux encode() space, std~1.7) into the
    transformer's packed/BatchNorm convention that C inference operates in
    (std~1) — VAE-Q1. Per-128-feature BN with feature(c,h,w)=c*4+(h%2)*2+(w%2),
    applied in [32,Lh,Lw] space so the trainer's existing patchify pack produces
    exactly C iris_vae_encode's output (validated: corr 0.9995 on a real image).
    No-op if stats are unavailable."""
    if bn_mean_r is None:
        return lat
    _, _, Lh, Lw = lat.shape
    mean_full = mx.tile(bn_mean_r, (1, Lh // 2, Lw // 2))   # [32, Lh, Lw]
    std_full = mx.tile(bn_std_r, (1, Lh // 2, Lw // 2))
    return ((lat.astype(mx.float32) - mean_full) / std_full).astype(lat.dtype)


def _vae_encode(vae, images: mx.array) -> mx.array:
    """
    Encode images to VAE latents.
    images: [B, 3, H, W] BF16 in [-1, 1]
    Returns: [B, 32, H/8, W/8]
    """
    return vae.encode(images)


def _encode_text(text_encoder: _TextEncoderBundle, captions: list) -> mx.array:
    """
    Encode text captions using Qwen3 (frozen).
    Returns: [B, seq, text_dim] BF16
    """
    tokens = text_encoder.tokenizer.tokenize(captions)
    embeds = text_encoder.encoder.get_prompt_embeds(
        input_ids=tokens.input_ids,
        attention_mask=tokens.attention_mask,
        hidden_state_layers=(9, 18, 27),
    )
    return embeds.astype(mx.bfloat16)


# ─────────────────────────────────────────────────────────────────────────────
# Flux forward pass with IP-Adapter injection
# ─────────────────────────────────────────────────────────────────────────────

def _flux_forward_no_ip(
    flux,
    noisy_latents: mx.array,
    text_embeds: mx.array,
    t_int: mx.array,
) -> dict:
    """
    Flux Klein 4B forward pass WITHOUT IP injection.

    Runs all transformer blocks, collecting the Q vector at each injection
    point (5 double + 20 single = 25 total). All collected tensors are
    wrapped in mx.stop_gradient so they carry no gradient into the adapter's
    backward pass.

    Returns flux_state dict:
      qs:      list[25] of [B, H, seq_img, Hd] per-block image Q
      h_final: [B, seq_img, d_inner]  image hidden state before norm_out
      temb:    [B, d_inner]           timestep embedding for norm_out
      B, C, Lh, Lw, pH, pW, seq_img, seq_txt: shape metadata
    """
    tr = flux.transformer

    # ── Step 1: patchify + pack noisy latents ─────────────────────────────────
    B, C, Lh, Lw = noisy_latents.shape
    pH, pW = Lh // 2, Lw // 2

    h = noisy_latents.reshape(B, C, pH, 2, pW, 2)
    h = h.transpose(0, 1, 3, 5, 2, 4)
    h = h.reshape(B, C * 4, pH, pW)
    hidden_states = h.reshape(B, 128, pH * pW).transpose(0, 2, 1)  # [B, seq_img, 128]
    seq_img = pH * pW

    # ── Step 2: position IDs ──────────────────────────────────────────────────
    h_grid = mx.broadcast_to(
        mx.arange(pH, dtype=mx.int32)[:, None], (pH, pW)
    ).reshape(-1)
    w_grid = mx.broadcast_to(
        mx.arange(pW, dtype=mx.int32)[None, :], (pH, pW)
    ).reshape(-1)
    img_ids = mx.stack(
        [mx.zeros(seq_img, dtype=mx.int32), h_grid, w_grid,
         mx.zeros(seq_img, dtype=mx.int32)],
        axis=1,
    )

    seq_txt = text_embeds.shape[1]
    txt_ids = mx.stack(
        [mx.zeros(seq_txt, dtype=mx.int32),
         mx.zeros(seq_txt, dtype=mx.int32),
         mx.zeros(seq_txt, dtype=mx.int32),
         mx.arange(seq_txt, dtype=mx.int32)],
        axis=1,
    )

    # ── Step 3: timestep embedding ────────────────────────────────────────────
    if not isinstance(t_int, mx.array):
        timestep = mx.array(t_int, dtype=hidden_states.dtype)
    else:
        timestep = t_int.astype(hidden_states.dtype)
    if timestep.ndim == 0:
        timestep = mx.full((B,), float(timestep.item()), dtype=hidden_states.dtype)
    elif timestep.shape[0] == 1 and B > 1:
        timestep = mx.broadcast_to(timestep, (B,))

    temb = tr.time_guidance_embed(timestep, guidance=None)
    temb = temb.astype(mx.bfloat16)

    # ── Step 4: project inputs ────────────────────────────────────────────────
    hidden_states = tr.x_embedder(hidden_states)
    encoder_hidden_states = tr.context_embedder(text_embeds)

    # ── Step 5: RoPE ──────────────────────────────────────────────────────────
    image_rotary_emb = tr.pos_embed(img_ids)
    text_rotary_emb  = tr.pos_embed(txt_ids)
    concat_rotary_emb = (
        mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
        mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
    )

    # ── Step 6: double-stream modulation (shared across all double blocks) ────
    temb_mod_params_img = tr.double_stream_modulation_img(temb)
    temb_mod_params_txt = tr.double_stream_modulation_txt(temb)
    (shift_msa_img, scale_msa_img, _), _ = temb_mod_params_img

    qs: list[mx.array] = []

    # ── Step 7: double-stream blocks ──────────────────────────────────────────
    for block in tr.transformer_blocks:
        h_before = hidden_states

        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_params_img=temb_mod_params_img,
            temb_mod_params_txt=temb_mod_params_txt,
            image_rotary_emb=concat_rotary_emb,
        )

        # Collect Q for this injection point
        norm_h = block.norm1(h_before)
        norm_h = (1 + scale_msa_img) * norm_h + shift_msa_img
        q_ip = block.attn.to_q(norm_h)
        bsz, s_img, d_inner = q_ip.shape
        H  = block.attn.heads
        Hd = block.attn.dim_head
        q_ip = q_ip.reshape(bsz, s_img, H, Hd)
        q_ip = block.attn.norm_q(q_ip.astype(mx.float32)).astype(mx.bfloat16)
        qs.append(mx.stop_gradient(q_ip.transpose(0, 2, 1, 3)))  # [B, H, seq_img, Hd]

    # ── Step 8: merge streams ─────────────────────────────────────────────────
    hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

    # ── Step 9: single-stream modulation ─────────────────────────────────────
    temb_mod_params_single = tr.single_stream_modulation(temb)[0]
    mod_shift_s, mod_scale_s, mod_gate_s = temb_mod_params_single

    # ── Step 10: single-stream blocks (inlined to eliminate redundant GEMM) ──
    # Flux2SingleTransformerBlock.__call__ runs norm→modulate→to_qkv_mlp_proj→attn.
    # Previously we'd call block() and then re-run norm+to_qkv_mlp_proj on image
    # tokens to extract Q for IP injection — paying the big fused projection twice.
    # Inlining shares the single GEMM, saving ~116 GFLOPs × 20 blocks per step.
    for block in tr.single_transformer_blocks:
        bsz, seq_full, _ = hidden_states.shape
        H_s  = block.attn.heads
        Hd_s = block.attn.dim_head

        # Norm + modulation: block-specific LayerNorm, shared modulation params
        norm_h = block.norm(hidden_states)
        norm_h = (1 + mod_scale_s) * norm_h + mod_shift_s

        # Single fused GEMM — used for both block forward and Q extraction
        proj = block.attn.to_qkv_mlp_proj(norm_h)
        qkv, mlp_hidden = mx.split(proj, [block.attn.inner_dim * 3], axis=-1)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # [B, seq, H*Hd] → [B, H, seq, Hd]
        q = q.reshape(bsz, seq_full, H_s, Hd_s).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_full, H_s, Hd_s).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_full, H_s, Hd_s).transpose(0, 2, 1, 3)
        q = block.attn.norm_q(q.astype(mx.float32)).astype(mx.bfloat16)
        k = block.attn.norm_k(k.astype(mx.float32)).astype(mx.bfloat16)

        # Q for IP-adapter: image tokens only, already norm_q'd; RMSNorm is per-token
        # so slicing after norm_q is identical to norm_q on the slice alone
        qs.append(mx.stop_gradient(q[:, :, seq_txt:, :]))  # [B, H, seq_img, Hd]

        # Complete block forward: RoPE → attention → MLP → output proj → residual
        if concat_rotary_emb is not None:
            cos, sin = concat_rotary_emb
            q, k = _FluxAttentionUtils.apply_rope_bshd(q, k, cos, sin)

        attn_out = _FluxAttentionUtils.compute_attention(
            query=q, key=k, value=v,
            batch_size=bsz, num_heads=H_s, head_dim=Hd_s,
        )
        mlp_hidden = block.attn.mlp_act(mlp_hidden)
        attn_out = mx.concatenate([attn_out, mlp_hidden], axis=-1)
        attn_out = block.attn.to_out(attn_out)
        hidden_states = hidden_states + mod_gate_s * attn_out

    # ── Step 11: extract image tokens before norm_out/proj_out ───────────────
    h_final = hidden_states[:, seq_txt:, :]  # [B, seq_img, d_inner]

    return {
        "qs":      qs,
        "h_final": mx.stop_gradient(h_final),
        "temb":    mx.stop_gradient(temb),
        "B":  B,   "C":  C,
        "Lh": Lh,  "Lw": Lw,
        "pH": pH,  "pW": pW,
        "seq_img": seq_img,
        "seq_txt": seq_txt,
    }


def _flux_forward_with_ip_collect_q(
    flux,
    noisy_latents: mx.array,
    text_embeds: mx.array,
    t_int: mx.array,
    k_ip_all: mx.array,
    v_ip_all: mx.array,
    ip_scale: mx.array,
) -> list:
    """
    Flux forward WITH IP injection, returning only the corrected Q vectors.

    Identical to _flux_forward_no_ip except that after each block the IP
    contribution is injected into hidden_states, so each subsequent block
    computes its Q from activations that already carry the IP influence from
    earlier blocks.  This matches the inference-time Q distribution and
    eliminates the train/inference mismatch present in _flux_forward_no_ip.

    Called outside value_and_grad — k_ip_all / v_ip_all / ip_scale are
    pre-computed from the current adapter params (no gradient needed here).
    The returned Q vectors are stop_gradient'd and used to replace the
    flux_state["qs"] produced by _flux_forward_no_ip before the backward pass.

    Returns:
        qs: list[num_blocks] of [B, H, seq_img, Hd]
    """
    tr = flux.transformer

    B, C, Lh, Lw = noisy_latents.shape
    pH, pW = Lh // 2, Lw // 2
    h = noisy_latents.reshape(B, C, pH, 2, pW, 2)
    h = h.transpose(0, 1, 3, 5, 2, 4)
    h = h.reshape(B, C * 4, pH, pW)
    hidden_states = h.reshape(B, 128, pH * pW).transpose(0, 2, 1)
    seq_img = pH * pW

    h_grid = mx.broadcast_to(mx.arange(pH, dtype=mx.int32)[:, None], (pH, pW)).reshape(-1)
    w_grid = mx.broadcast_to(mx.arange(pW, dtype=mx.int32)[None, :], (pH, pW)).reshape(-1)
    img_ids = mx.stack(
        [mx.zeros(seq_img, dtype=mx.int32), h_grid, w_grid,
         mx.zeros(seq_img, dtype=mx.int32)], axis=1,
    )
    seq_txt = text_embeds.shape[1]
    txt_ids = mx.stack(
        [mx.zeros(seq_txt, dtype=mx.int32), mx.zeros(seq_txt, dtype=mx.int32),
         mx.zeros(seq_txt, dtype=mx.int32), mx.arange(seq_txt, dtype=mx.int32)],
        axis=1,
    )

    if not isinstance(t_int, mx.array):
        timestep = mx.array(t_int, dtype=hidden_states.dtype)
    else:
        timestep = t_int.astype(hidden_states.dtype)
    if timestep.ndim == 0:
        timestep = mx.full((B,), float(timestep.item()), dtype=hidden_states.dtype)
    elif timestep.shape[0] == 1 and B > 1:
        timestep = mx.broadcast_to(timestep, (B,))

    temb = tr.time_guidance_embed(timestep, guidance=None)
    temb = temb.astype(mx.bfloat16)

    hidden_states = tr.x_embedder(hidden_states)
    encoder_hidden_states = tr.context_embedder(text_embeds)

    image_rotary_emb = tr.pos_embed(img_ids)
    text_rotary_emb  = tr.pos_embed(txt_ids)
    concat_rotary_emb = (
        mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
        mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
    )

    temb_mod_params_img = tr.double_stream_modulation_img(temb)
    temb_mod_params_txt = tr.double_stream_modulation_txt(temb)
    (shift_msa_img, scale_msa_img, _), _ = temb_mod_params_img

    qs: list[mx.array] = []

    # Double-stream blocks: Q extracted from IP-influenced h_before, then IP injected
    for i, block in enumerate(tr.transformer_blocks):
        h_before = hidden_states  # includes IP contributions from blocks 0..i-1

        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_params_img=temb_mod_params_img,
            temb_mod_params_txt=temb_mod_params_txt,
            image_rotary_emb=concat_rotary_emb,
        )

        norm_h = block.norm1(h_before)
        norm_h = (1 + scale_msa_img) * norm_h + shift_msa_img
        q_ip = block.attn.to_q(norm_h)
        bsz, s_img, d_inner = q_ip.shape
        H  = block.attn.heads
        Hd = block.attn.dim_head
        q_ip = q_ip.reshape(bsz, s_img, H, Hd)
        q_ip = block.attn.norm_q(q_ip.astype(mx.float32)).astype(mx.bfloat16)
        q_ip_sg = mx.stop_gradient(q_ip.transpose(0, 2, 1, 3))  # [B, H, seq_img, Hd]
        qs.append(q_ip_sg)

        # Inject IP so next block's h_before carries this block's contribution
        k_i = k_ip_all[:, i, :, :].reshape(bsz, -1, H, Hd).transpose(0, 2, 1, 3)
        v_i = v_ip_all[:, i, :, :].reshape(bsz, -1, H, Hd).transpose(0, 2, 1, 3)
        ip_out = mx.fast.scaled_dot_product_attention(q_ip_sg, k_i, v_i, scale=Hd ** -0.5)
        ip_out = ip_out.transpose(0, 2, 1, 3).reshape(bsz, s_img, d_inner)
        hidden_states = hidden_states + ip_scale[i] * ip_out

    hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

    temb_mod_params_single = tr.single_stream_modulation(temb)[0]
    mod_shift_s, mod_scale_s, mod_gate_s = temb_mod_params_single

    n_double = len(tr.transformer_blocks)

    # Single-stream blocks (inlined GEMM sharing, same as _flux_forward_no_ip)
    for j, block in enumerate(tr.single_transformer_blocks):
        bsz, seq_full, _ = hidden_states.shape
        H_s  = block.attn.heads
        Hd_s = block.attn.dim_head
        block_ip_idx = n_double + j

        norm_h = block.norm(hidden_states)
        norm_h = (1 + mod_scale_s) * norm_h + mod_shift_s

        proj = block.attn.to_qkv_mlp_proj(norm_h)
        qkv, mlp_hidden = mx.split(proj, [block.attn.inner_dim * 3], axis=-1)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(bsz, seq_full, H_s, Hd_s).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seq_full, H_s, Hd_s).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seq_full, H_s, Hd_s).transpose(0, 2, 1, 3)
        q = block.attn.norm_q(q.astype(mx.float32)).astype(mx.bfloat16)
        k = block.attn.norm_k(k.astype(mx.float32)).astype(mx.bfloat16)

        # Approximation: Q is extracted before IP is injected into hidden_states for
        # this block, so single-stream Qs are conditioned only on double-stream IP
        # output, not on the previous single-stream block's IP contribution.
        # Full fix would require injecting from the previous block before computing Q
        # here. Acceptable for current training; document to avoid confusion.
        q_ip_sg = mx.stop_gradient(q[:, :, seq_txt:, :])  # [B, H, seq_img, Hd]
        qs.append(q_ip_sg)

        if concat_rotary_emb is not None:
            cos, sin = concat_rotary_emb
            q, k = _FluxAttentionUtils.apply_rope_bshd(q, k, cos, sin)

        attn_out = _FluxAttentionUtils.compute_attention(
            query=q, key=k, value=v,
            batch_size=bsz, num_heads=H_s, head_dim=Hd_s,
        )
        mlp_hidden = block.attn.mlp_act(mlp_hidden)
        attn_out = mx.concatenate([attn_out, mlp_hidden], axis=-1)
        attn_out = block.attn.to_out(attn_out)
        hidden_states = hidden_states + mod_gate_s * attn_out

        # Inject IP into image portion so next block sees IP-influenced state
        k_i = k_ip_all[:, block_ip_idx, :, :].reshape(bsz, -1, H_s, Hd_s).transpose(0, 2, 1, 3)
        v_i = v_ip_all[:, block_ip_idx, :, :].reshape(bsz, -1, H_s, Hd_s).transpose(0, 2, 1, 3)
        ip_out_s = mx.fast.scaled_dot_product_attention(q_ip_sg, k_i, v_i, scale=Hd_s ** -0.5)
        ip_out_s = ip_out_s.transpose(0, 2, 1, 3).reshape(bsz, seq_img, H_s * Hd_s)
        hidden_states = mx.concatenate([
            hidden_states[:, :seq_txt, :],
            hidden_states[:, seq_txt:, :] + ip_scale[block_ip_idx] * ip_out_s,
        ], axis=1)

    return qs


def _flux_forward_with_ip(
    flux,
    noisy_latents: mx.array,
    text_embeds: mx.array,
    t_int: mx.array,
    k_ip_all: mx.array,
    v_ip_all: mx.array,
    ip_scale: mx.array,
    ckpt_double=None,
    ckpt_single=None,
) -> mx.array:
    """
    Flux Klein 4B forward pass with IP-Adapter injection.

    Manually replays the transformer forward pass so that after each double-stream
    block's standard attention we add:
        ip_out = SDPA(img_Q, k_ip[i], v_ip[i])
        hidden_states += ip_scale[i] * ip_out

    For single-stream blocks the same injection is applied to the image portion of
    the concatenated [TEXT | IMAGE] sequence.

    Args:
        flux:           frozen Flux2Klein model
        noisy_latents:  [B, 32, H/8, W/8] noisy latents in VAE space
        text_embeds:    [B, seq_txt, 7680] Qwen3 embeddings
        t_int:          [B] or [1] integer timestep in [0, 1000)
        k_ip_all:       [B, num_blocks, 128, 3072] IP keys
        v_ip_all:       [B, num_blocks, 128, 3072] IP values
        ip_scale:       [num_blocks] per-block scale (from adapter.scale)

    Returns:
        pred: [B, 32, H/8, W/8] velocity prediction in VAE latent space
    """
    tr = flux.transformer

    # ── Step 1: patchify + pack noisy latents ─────────────────────────────────
    # VAE latents: [B, 32, Lh, Lw]  (Lh = H/8, Lw = W/8)
    # Patchify:    [B, 128, pH, pW]  (pH = Lh/2 = H/16)
    # Pack:        [B, seq_img, 128]
    B, C, Lh, Lw = noisy_latents.shape
    pH, pW = Lh // 2, Lw // 2

    h = noisy_latents.reshape(B, C, pH, 2, pW, 2)   # [B, 32, pH, 2, pW, 2]
    h = h.transpose(0, 1, 3, 5, 2, 4)               # [B, 32, 2, 2, pH, pW]
    h = h.reshape(B, C * 4, pH, pW)                  # [B, 128, pH, pW]
    hidden_states = h.reshape(B, 128, pH * pW).transpose(0, 2, 1)  # [B, seq_img, 128]
    seq_img = pH * pW

    # ── Step 2: position IDs ──────────────────────────────────────────────────
    h_grid = mx.broadcast_to(
        mx.arange(pH, dtype=mx.int32)[:, None], (pH, pW)
    ).reshape(-1)
    w_grid = mx.broadcast_to(
        mx.arange(pW, dtype=mx.int32)[None, :], (pH, pW)
    ).reshape(-1)
    img_ids = mx.stack(
        [mx.zeros(seq_img, dtype=mx.int32), h_grid, w_grid,
         mx.zeros(seq_img, dtype=mx.int32)],
        axis=1,
    )  # [seq_img, 4]

    seq_txt = text_embeds.shape[1]
    txt_ids = mx.stack(
        [mx.zeros(seq_txt, dtype=mx.int32),
         mx.zeros(seq_txt, dtype=mx.int32),
         mx.zeros(seq_txt, dtype=mx.int32),
         mx.arange(seq_txt, dtype=mx.int32)],
        axis=1,
    )  # [seq_txt, 4]

    # ── Step 3: timestep embedding ────────────────────────────────────────────
    # Klein 4B: guidance_embeds=False, so pass guidance=None
    if not isinstance(t_int, mx.array):
        timestep = mx.array(t_int, dtype=hidden_states.dtype)
    else:
        timestep = t_int.astype(hidden_states.dtype)
    if timestep.ndim == 0:
        timestep = mx.full((B,), float(timestep.item()), dtype=hidden_states.dtype)
    elif timestep.shape[0] == 1 and B > 1:
        timestep = mx.broadcast_to(timestep, (B,))

    temb = tr.time_guidance_embed(timestep, guidance=None)
    temb = temb.astype(mx.bfloat16)

    # ── Step 4: project inputs ────────────────────────────────────────────────
    hidden_states = tr.x_embedder(hidden_states)            # [B, seq_img, inner_dim]
    encoder_hidden_states = tr.context_embedder(text_embeds)  # [B, seq_txt, inner_dim]

    # ── Step 5: RoPE ──────────────────────────────────────────────────────────
    image_rotary_emb = tr.pos_embed(img_ids)
    text_rotary_emb = tr.pos_embed(txt_ids)
    concat_rotary_emb = (
        mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
        mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
    )

    # ── Step 6: double-stream modulation (shared across all double blocks) ────
    temb_mod_params_img = tr.double_stream_modulation_img(temb)
    temb_mod_params_txt = tr.double_stream_modulation_txt(temb)
    # Unpack the AdaLN-Zero shift/scale for IP Q extraction (same for all blocks)
    (shift_msa_img, scale_msa_img, _), _ = temb_mod_params_img

    # ── Step 7: double-stream blocks + IP injection ───────────────────────────
    for i, block in enumerate(tr.transformer_blocks):
        h_before = hidden_states  # save pre-block state to recompute Q

        call_block = ckpt_double[i] if ckpt_double is not None else block
        encoder_hidden_states, hidden_states = call_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_params_img=temb_mod_params_img,
            temb_mod_params_txt=temb_mod_params_txt,
            image_rotary_emb=concat_rotary_emb,
        )

        # Recompute normalised image hidden state to extract Q for IP attention
        norm_h = block.norm1(h_before)
        norm_h = (1 + scale_msa_img) * norm_h + shift_msa_img  # [B, seq_img, inner_dim]

        # Q projection + RMSNorm (matches double-block attention path)
        # stop_gradient: Q is from frozen Flux weights — gradient through Q is useless.
        # This removes Q from the backward graph without affecting k_ip/v_ip gradients.
        q_ip = block.attn.to_q(norm_h)
        bsz, s_img, d_inner = q_ip.shape
        H = block.attn.heads
        Hd = block.attn.dim_head
        q_ip = q_ip.reshape(bsz, s_img, H, Hd)
        q_ip = block.attn.norm_q(q_ip.astype(mx.float32)).astype(mx.bfloat16)
        q_ip = mx.stop_gradient(q_ip.transpose(0, 2, 1, 3))  # [B, heads, seq_img, head_dim]

        # IP K/V for this block: k_ip_all[:, i] is [B, 128, inner_dim]
        k_i = k_ip_all[:, i, :, :]  # [B, 128, inner_dim]
        v_i = v_ip_all[:, i, :, :]
        k_i = k_i.reshape(bsz, -1, H, Hd).transpose(0, 2, 1, 3)
        v_i = v_i.reshape(bsz, -1, H, Hd).transpose(0, 2, 1, 3)

        ip_out = mx.fast.scaled_dot_product_attention(
            q_ip, k_i, v_i, scale=Hd ** -0.5,
        )  # [B, heads, seq_img, head_dim]
        ip_out = ip_out.transpose(0, 2, 1, 3).reshape(bsz, s_img, d_inner)

        hidden_states = hidden_states + ip_scale[i] * ip_out

    # ── Step 8: merge streams ─────────────────────────────────────────────────
    hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

    # ── Step 9: single-stream modulation (shared across all single blocks) ────
    temb_mod_params_single = tr.single_stream_modulation(temb)[0]
    mod_shift_s, mod_scale_s, _ = temb_mod_params_single

    n_double = len(tr.transformer_blocks)  # 5 for Klein 4B

    # ── Step 10: single-stream blocks + IP injection ──────────────────────────
    for j, block in enumerate(tr.single_transformer_blocks):
        h_before_s = hidden_states

        call_block_s = ckpt_single[j] if ckpt_single is not None else block
        hidden_states = call_block_s(
            hidden_states=hidden_states,
            temb_mod_params=temb_mod_params_single,
            image_rotary_emb=concat_rotary_emb,
        )

        # IP injection for image tokens: recompute Q from image portion
        block_ip_idx = n_double + j

        norm_h_full = block.norm(h_before_s)
        norm_h_full = (1 + mod_scale_s) * norm_h_full + mod_shift_s
        img_norm_h = norm_h_full[:, seq_txt:, :]  # [B, seq_img, inner_dim]

        # Fused QKV+MLP projection — extract Q only
        proj_img = block.attn.to_qkv_mlp_proj(img_norm_h)
        qkv_img, _ = mx.split(proj_img, [block.attn.inner_dim * 3], axis=-1)
        q_ip_s, _, _ = mx.split(qkv_img, 3, axis=-1)

        bsz2, s2, d2 = q_ip_s.shape
        H_s = block.attn.heads
        Hd_s = block.attn.dim_head
        q_ip_s = q_ip_s.reshape(bsz2, s2, H_s, Hd_s)
        q_ip_s = block.attn.norm_q(q_ip_s.astype(mx.float32)).astype(mx.bfloat16)
        # stop_gradient: Q from frozen Flux weights, gradient not needed
        q_ip_s = mx.stop_gradient(q_ip_s.transpose(0, 2, 1, 3))  # [B, heads, seq_img, head_dim]

        k_i2 = k_ip_all[:, block_ip_idx, :, :]
        v_i2 = v_ip_all[:, block_ip_idx, :, :]
        k_i2 = k_i2.reshape(bsz2, -1, H_s, Hd_s).transpose(0, 2, 1, 3)
        v_i2 = v_i2.reshape(bsz2, -1, H_s, Hd_s).transpose(0, 2, 1, 3)

        ip_out_s = mx.fast.scaled_dot_product_attention(
            q_ip_s, k_i2, v_i2, scale=Hd_s ** -0.5,
        )
        ip_out_s = ip_out_s.transpose(0, 2, 1, 3).reshape(bsz2, s2, d2)

        # Add IP contribution to image portion of hidden_states
        hidden_states = mx.concatenate([
            hidden_states[:, :seq_txt, :],
            hidden_states[:, seq_txt:, :] + ip_scale[block_ip_idx] * ip_out_s,
        ], axis=1)

    # ── Step 11: extract image tokens, final norm + proj ─────────────────────
    hidden_states = hidden_states[:, seq_txt:, :]
    hidden_states = tr.norm_out(hidden_states, temb)
    hidden_states = tr.proj_out(hidden_states)  # [B, seq_img, 128]

    # ── Step 12: unpack → unatchify → latent space ────────────────────────────
    # [B, seq_img, 128] → [B, 128, pH, pW] → [B, 32, H/8, W/8]
    pred = hidden_states.transpose(0, 2, 1).reshape(B, C * 4, pH, pW)
    pred = pred.reshape(B, C, 2, 2, pH, pW)
    pred = pred.transpose(0, 1, 4, 2, 5, 3)  # [B, C, pH, 2, pW, 2]
    pred = pred.reshape(B, C, Lh, Lw)         # [B, 32, H/8, W/8]

    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train IP-Adapter for Flux Klein 4B")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint .safetensors path")
    parser.add_argument("--warmstart-weights", default=None,
                        help="Load adapter weights from this checkpoint but start a FRESH "
                             "schedule (start_step=0, fresh warmup/optimizer). Unlike --resume, "
                             "it does NOT continue the checkpoint's step count. For ablation "
                             "arms that warm-start from a campaign checkpoint without inheriting "
                             "its training horizon.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config (use lower LR for later chunks)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps from config")
    parser.add_argument("--anchor-shards", default=None,
                        help="Path to anchor shard directory mixed into every chunk (20%% of batches)")
    parser.add_argument("--hard-examples", default=None,
                        help="Path to hard example shard directory (from mine_hard_examples.py); "
                             "mixed at hard_mix_ratio (default 5%%)")
    parser.add_argument("--chunk", type=int, default=None,
                        help="Pipeline chunk number (1-4). Used to write the trainer "
                             "heartbeat to the correct pipeline location so the orchestrator "
                             "can monitor liveness and anomalies.")
    parser.add_argument("--run-name", default=None,
                        help="Name for this direct run (e.g. 'smoke', 'dev'). Writes a named "
                             "heartbeat (trainer_{name}.json) visible to pipeline_status.py "
                             "and pipeline_doctor.py. Defaults to the config filename stem "
                             "when running without --chunk.")
    parser.add_argument("--chunk-base-step", type=int, default=None,
                        help="Absolute global step at which this chunk's training begins "
                             "(sum of all previous chunks' steps). Used to compute the "
                             "correct end step and LR schedule fast-forward when resuming "
                             "from a cross-chunk warmstart checkpoint.")
    parser.add_argument("--data-root", default=None,
                        help="Root directory for shards and precomputed caches. "
                             "Relative paths in the config YAML are prefixed with this value. "
                             "Defaults to the current working directory.")
    parser.add_argument("--warmup-only", action="store_true",
                        help="Compile Metal PSO training graphs for all bucket shapes "
                             "and exit. Populates the Metal kernel cache so the first "
                             "real training step starts immediately. Run once per machine "
                             "after initial setup or after an OS/MLX version change.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without training")
    parser.add_argument("--log-every", type=int, default=None,
                        help="Override log_every from config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Prefix relative data paths with --data-root so the pipeline can point the
    # trainer at an external drive without editing the YAML.
    # Convention: YAML paths are written as "train/data/<subpath>"; we strip the
    # "train/data/" prefix and join the remainder onto data_root.
    if args.data_root:
        _dr = args.data_root
        _data_path_keys = [
            ("data", "shard_path"),
            ("data", "qwen3_cache_dir"),
            ("data", "vae_cache_dir"),
            ("data", "siglip_cache_dir"),
            ("data", "anchor_shard_dir"),
            ("data", "hard_example_dir"),
            ("output", "checkpoint_dir"),
        ]
        _strip_prefix = "train/data/"
        for section, key in _data_path_keys:
            val = config.get(section, {}).get(key)
            if val and not os.path.isabs(val):
                rel = val[len(_strip_prefix):] if val.startswith(_strip_prefix) else val
                config[section][key] = os.path.join(_dr, rel)

        # Version-resolve ONLY flat-default cache dirs to their `current` symlink;
        # never clobber an explicit (e.g. per-iter flywheel staging) cache dir.
        _resolve_versioned_cache_dirs(config, _dr)

    if args.resume:
        config["model"]["warmstart_path"] = args.resume
    if args.warmstart_weights:
        # Weights-only warm-start: load adapter weights but keep a fresh schedule.
        config["model"]["warmstart_weights_only"] = args.warmstart_weights
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.max_steps is not None:
        config["training"]["num_steps"] = args.max_steps
    if args.log_every is not None:
        config["output"]["log_every"] = args.log_every
    if args.anchor_shards is not None:
        config["data"]["anchor_shard_dir"] = args.anchor_shards
    if args.hard_examples is not None:
        config["data"]["hard_example_dir"] = args.hard_examples

    if args.chunk is not None:
        config["_chunk"] = args.chunk
    if args.chunk_base_step is not None:
        config["_chunk_base_step"] = args.chunk_base_step

    # Resolve run_name for direct (non-pipeline) runs.
    # Priority: --run-name flag > config run_name key > config filename stem.
    # Ignored when --chunk is set (pipeline runs use chunk-scoped heartbeats).
    if config.get("_chunk") is None:
        run_name = (args.run_name
                    or config.get("run_name")
                    or Path(args.config).stem)
        config["_run_name"] = run_name

    # Store config directory so train() can locate sibling files (e.g. eval_prompts.txt).
    config["_config_dir"] = str(Path(args.config).parent)
    config["_warmup_only"] = args.warmup_only

    if args.dry_run:
        print("Config OK:")
        print(yaml.dump(config, default_flow_style=False))
        return

    train(config)


if __name__ == "__main__":
    main()
