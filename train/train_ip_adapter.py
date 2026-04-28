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
from functools import partial
from pathlib import Path
from typing import Optional

import yaml

# ── MLX imports ──────────────────────────────────────────────────────────────
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
except ImportError:
    print("Error: MLX not found. Run: source train/.venv/bin/activate", file=sys.stderr)
    sys.exit(1)

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from ip_adapter.model import IPAdapterKlein
from ip_adapter.loss import fused_flow_noise, get_schedule_values
from ip_adapter.ema import update_ema, save_ema, _flatten
from ip_adapter.dataset import make_prefetch_loader, augment_mlx, BUCKETS

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
    decay_steps = total_steps - warmup_steps

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
        # Already in cosine decay phase
        decay_done = start_step - warmup_steps
        decay_total = max(1, total_steps - warmup_steps)
        progress = min(1.0, decay_done / decay_total)
        current_lr = eta_min + 0.5 * (lr_max - eta_min) * (1.0 + math.cos(math.pi * progress))
        remaining_decay = max(1, decay_total - decay_done)
        return optim.cosine_decay(current_lr, decay_steps=remaining_decay, end=eta_min)


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


def _purge_file_page_cache(path: str) -> None:
    """
    Evict a file from the macOS unified buffer cache (page cache).

    mx.save_safetensors (and safetensors safe_open reads) leave file data in the
    kernel page cache. With ~4 GB checkpoint files, two consecutive reads or
    writes fill ~8 GB of page cache silently, reducing psutil 'available' memory
    and triggering jetsam even when MLX's own buffers fit within the 14 GB limit.

    madvise(MADV_FREE) is insufficient: on macOS it marks pages as reclaimable
    but does not evict them immediately, so pages remain in physical RAM until
    the kernel decides to free them — which may be after jetsam has already fired.

    msync(MS_INVALIDATE=2) is synchronous: it flushes any dirty pages to the
    backing store and then immediately evicts the pages from the UBC, reclaiming
    physical RAM before the next Flux forward pass allocation.
    """
    try:
        import ctypes as _ctypes, mmap as _mmap, numpy as _np
        _libc = _ctypes.CDLL("libc.dylib", use_errno=True)
        _fd = os.open(path, os.O_RDONLY)
        try:
            _sz = os.fstat(_fd).st_size
            if _sz == 0:
                return
            _mm = _mmap.mmap(_fd, _sz, access=_mmap.ACCESS_READ)
            try:
                # np.frombuffer is a zero-copy view; .ctypes.data gives the
                # mmap base address without copying the 4 GB checkpoint data.
                _arr = _np.frombuffer(_mm, dtype=_np.uint8)
                _addr = _ctypes.c_void_p(_arr.ctypes.data)
                # MS_INVALIDATE = 2: synchronously evict pages from the UBC.
                # For a read-only file-backed mmap the flush is a no-op (no
                # dirty pages), so only the eviction step runs.
                _libc.msync(_addr, _ctypes.c_size_t(_sz), _ctypes.c_int(2))
                del _arr  # release exported buffer ref before closing mmap
            finally:
                _mm.close()
        finally:
            os.close(_fd)
    except Exception:
        pass  # non-critical; pages reclaim naturally under pressure


def save_checkpoint_async(
    adapter: IPAdapterKlein,
    ema_params: dict,
    step: int,
    output_dir: str,
    keep_last_n: int = 5,
    lineage: Optional[dict] = None,
) -> None:
    """
    Save adapter weights + EMA synchronously.

    Background saving was tried but caused OOM: the background thread held
    old adapter param tensors alive while the main thread created new ones
    (after optimizer.update), temporarily doubling adapter memory footprint
    and pushing unified memory over 32 GB. Synchronous saving adds ~1-2s
    every 200 steps (<0.2% overhead) and avoids the memory overlap entirely.

    mx.save_safetensors writes directly from Metal buffers without a numpy
    copy, so the write itself is efficient.

    lineage: if provided, written as a sidecar step_NNNNNNN.json recording the
    config, git SHA, training args, and step/loss for reproducibility (MLX-10).
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"step_{step:07d}.safetensors")

    # Free any cached MLX buffers before forcing eval of checkpoint tensors.
    # Reduces peak unified memory during the eval + serialize window, which
    # is the most likely trigger for jetsam kills on 32 GB systems.
    mx.clear_cache()
    mx.eval(adapter.parameters())
    mx.eval(ema_params)

    payload = {
        **dict(_flatten(adapter.parameters())),
        **{f"ema.{k}": v for k, v in _flatten(ema_params)},
    }

    mx.save_safetensors(ckpt_path, payload)
    del payload
    _purge_file_page_cache(ckpt_path)  # reclaim ~4 GB OS page cache from the write
    mx.clear_cache()

    size_mb = os.path.getsize(ckpt_path) / 1e6
    print(f"  checkpoint saved: step_{step:07d}.safetensors ({size_mb:.0f} MB)")

    if lineage is not None:
        sidecar_path = os.path.join(output_dir, f"step_{step:07d}.json")
        try:
            with open(sidecar_path, "w") as _f:
                json.dump(lineage, _f, indent=2)
        except OSError as e:
            print(f"  WARNING: could not write lineage sidecar: {e}")

    with _ckpt_lock:
        _purge_old_checkpoints(output_dir, keep_last_n)


def _purge_old_checkpoints(directory: str, keep_last_n: int) -> None:
    files = sorted(f for f in glob.glob(os.path.join(directory, "step_*.safetensors"))
                   if "ema." not in f)
    for f in files[:-keep_last_n]:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass  # already removed by a concurrent thread
        # Remove sidecar lineage JSON if present
        sidecar = f.replace(".safetensors", ".json")
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
      - step_*.safetensors: keys prefixed with "ema." (written by save_checkpoint_async)
      - best.safetensors:   bare keys, no prefix (written by save_ema)
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
                # bare keys — save_ema() export (e.g. best.safetensors)
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
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict) -> None:
    mcfg = config["model"]
    acfg = config["adapter"]
    dcfg = config["data"]
    tcfg = config["training"]
    ocfg = config["output"]
    lcfg = config.get("logging") or {}

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
    # Shard path
    _shard_dir = dcfg["shard_path"]
    _shard_count = len(glob.glob(os.path.join(_shard_dir, "*.tar")))
    _preflight_ok &= _check(f"shards: {_shard_dir}", _shard_count > 0,
                             f"{_shard_count} tars" if _shard_count else "no .tar files found")
    # Checkpoint if resuming
    _warmstart = mcfg.get("warmstart_path")
    if _warmstart:
        _ws_ok = os.path.isfile(_warmstart)
        _preflight_ok &= _check(f"resume checkpoint: {_warmstart}", _ws_ok,
                                 "" if _ws_ok else "file not found")
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

    # Cap MLX GPU allocation to 20 GB on 32 GB systems (leaves ~12 GB for OS +
    # Metal driver + other processes).  Without this cap, jetsam kills the
    # process during mx.compile's first backward-pass compilation (~step 250)
    # when transient buffer peaks exceed the system's memory pressure threshold.
    # set_cache_limit(0) prevents MLX from holding freed buffers in a cache —
    # forces immediate return to the OS, reducing peak wired memory.
    try:
        import subprocess as _sp
        _hw = _sp.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3)
        _ram_bytes = int(_hw.stdout.strip())
    except Exception:
        _ram_bytes = 32 * 1024 ** 3  # fallback for non-macOS or sysctl unavailable
    mx.set_memory_limit(int(_ram_bytes * 0.44))   # 44% → 14 GB on 32 GB, 28 GB on 64 GB
    mx.set_cache_limit(int(_ram_bytes * 0.06))    # 6% → 2 GB on 32 GB, 4 GB on 64 GB
    print(f"MLX memory limit: {int(_ram_bytes * 0.44) // 1024**3} GB  "
          f"cache limit: {int(_ram_bytes * 0.06) // 1024**2} MB")

    # Write loading heartbeats every 60s while models initialise.
    # Without this, the orchestrator's 900s stale threshold kills the process
    # before model loading + graph compilation finish (~10-12 min total).
    _boot_chunk   = config.get("_chunk")
    _boot_hb_stop = __import__("threading").Event()
    def _boot_hb_thread():
        import time as _t
        from pipeline_lib import write_heartbeat as _wh
        while not _boot_hb_stop.wait(60):
            _wh("trainer", _boot_chunk, status="loading", step=0)
    __import__("threading").Thread(target=_boot_hb_thread, daemon=True).start()

    print("Loading Flux Klein 4B (frozen) ...")
    flux = Flux2Klein(model_path=mcfg["flux_model_dir"], quantize=None)
    flux.freeze()

    use_siglip_live = dcfg.get("siglip_cache_dir") is None
    siglip = None
    if use_siglip_live:
        print("Loading SigLIP SO400M (frozen, live inference — consider precomputing) ...")
        siglip = _load_siglip(mcfg["siglip_model"])
        siglip.freeze()
    else:
        print("SigLIP: using precomputed cache — skipping model load.")

    vae_cache_available = bool(dcfg.get("vae_cache_dir"))
    text_cache_available = bool(dcfg.get("qwen3_cache_dir"))

    if vae_cache_available:
        vae = None  # precomputed latents — VAE not needed during training
        print("VAE: using precomputed cache — skipping model load.")
    else:
        print("Loading VAE (frozen) ...")
        vae = _load_vae(flux)
        vae.freeze()

    if text_cache_available:
        text_encoder = None  # precomputed text embeddings — encoder not needed
        print("Text encoder: using precomputed cache — skipping model load.")
    else:
        print("Loading Qwen3 text encoder (Q4, frozen) ...")
        text_encoder = _load_text_encoder(flux)
        text_encoder.freeze()

    # Start data loader threads now so they can prefill the batch queue during
    # the Flux pre-eval below (which takes several minutes). By the time training
    # starts the first several batches will already be decoded and waiting.
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
        def _internal_prefix(tar_path):
            # Production shards keep their staging filename (e.g. "000000.tar" for
            # chunk 1, "250000.tar" for chunk 2) so the stem IS the internal prefix.
            return os.path.splitext(os.path.basename(tar_path))[0]

        def _has_cache(tar_path):
            pfx = _internal_prefix(tar_path)
            # Check first and 50th record: if both exist, the shard is substantially
            # precomputed.  Checking only _0000 passes even if precompute crashed after
            # the first write, causing 99%+ of that shard's batches to be skipped silently.
            return all(
                os.path.exists(os.path.join(d, f"{pfx}_{i:04d}.npz"))
                for d in (qwen3_dir, vae_dir)
                for i in (0, 49)
            )

        all_shards = shard_paths
        shard_paths = [p for p in all_shards if _has_cache(p)]
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
        _shard_prefixes = {_internal_prefix(p) for p in shard_paths}
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

    loader = make_prefetch_loader(
        shard_paths=shard_paths,
        batch_size=dcfg["batch_size"],
        image_dropout_prob=tcfg["image_dropout_prob"],
        text_dropout_prob=tcfg["text_dropout_prob"],
        sample_buffer=dcfg.get("prefetch_batches", 6),
        qwen3_cache_dir=dcfg.get("qwen3_cache_dir"),
        vae_cache_dir=dcfg.get("vae_cache_dir"),
        siglip_cache_dir=dcfg.get("siglip_cache_dir"),
        anchor_shard_dir=dcfg.get("anchor_shard_dir"),
        anchor_mix_ratio=dcfg.get("anchor_mix_ratio", 0.20),
        hard_example_dir=dcfg.get("hard_example_dir"),
        hard_mix_ratio=dcfg.get("hard_mix_ratio", 0.05),
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

    # ── Build IP-Adapter (trainable) ──────────────────────────────────────────
    print("Building IPAdapterKlein ...")
    warmstart = mcfg.get("warmstart_path")
    if warmstart and os.path.isdir(warmstart):
        # Warmstart Perceiver Resampler from InstantX Flux.1-dev weights
        adapter = IPAdapterKlein.from_pretrained_warmstart(
            instantx_path=warmstart,
            num_blocks=acfg["num_blocks"],
            hidden_dim=acfg["hidden_dim"],
            num_image_tokens=acfg["num_image_tokens"],
            siglip_dim=acfg["siglip_dim"],
            perceiver_heads=acfg["perceiver_heads"],
        )
    else:
        adapter = IPAdapterKlein(
            num_blocks=acfg["num_blocks"],
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

    # ── Optimizer with built-in cosine+warmup schedule ────────────────────────
    # MLX schedule object passed directly to AdamW — it advances each step.
    # start_step fast-forwards the schedule so LR continues from where it left
    # off rather than restarting warmup.  (plans/ip-adapter-training.md §3.11)
    lr_schedule = make_lr_schedule(
        tcfg["learning_rate"],
        tcfg["warmup_steps"],
        tcfg["num_steps"],
        start_step=start_step,
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
        ema_params = adapter.parameters()

    # ── Pre-evaluate adapter parameters ───────────────────────────────────────
    # Flush all lazy init arrays (mx.random.normal * scale) before training.
    mx.eval(adapter.parameters())
    mx.eval(ema_params)

    def loss_fn(siglip_feats, use_null_image: mx.array, use_null_text: mx.array,
                flux_state: dict, target: mx.array):
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
        ip_embeds = adapter.get_image_embeds(siglip_feats)
        zero_embeds = mx.zeros_like(ip_embeds)
        ip_embeds = mx.where(use_null_image, zero_embeds, ip_embeds)

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
            ip_out = ip_out.transpose(0, 2, 1, 3).reshape(B, seq_img, d_inner)
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

        return mx.mean((pred - target) ** 2)

    loss_and_grad = nn.value_and_grad(adapter, loss_fn)

    # TP-002 (infeasible): mx.compile requires all inputs to be explicit args or
    # declared captured state. flux_state contains dynamically-created arrays passed
    # via a dict — MLX cannot track these, raises "uncaptured inputs." Not worth
    # working around since the adapter step is already ~0.0s (lazy graph tracing
    # is negligible; all real work is in mx.eval dominated by Flux forward at 5-6s).

    def compiled_step(siglip_feats, use_null_image, use_null_text, flux_state, target):
        loss_val, grads = loss_and_grad(
            siglip_feats, use_null_image, use_null_text, flux_state, target,
        )
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=tcfg["grad_clip"])
        optimizer.update(adapter, grads)
        return loss_val, grad_norm

    # ── T-05: validation loss on held-out set ─────────────────────────────────
    def _compute_val_loss() -> Optional[float]:
        """Run a no-grad forward on up to 16 held-out records. Returns mean loss or None."""
        if not _val_shards:
            return None
        from ip_adapter.dataset import _load_vae_latent, _load_qwen3_embed
        losses = []
        for _shard in _val_shards[:2]:
            try:
                import tarfile as _tf
                with _tf.open(_shard) as _tar:
                    _contents = {m.name: _tar.extractfile(m).read()
                                 for m in _tar.getmembers() if m.isfile()}
            except Exception:
                continue
            _keys: dict = {}
            for _name in _contents:
                _stem, _, _ext = _name.rpartition(".")
                _keys.setdefault(_stem, {})[_ext.lower()] = _name
            for _stem in list(_keys)[:8]:
                _txt_key = _keys[_stem].get("txt") or _keys[_stem].get("caption")
                if not _txt_key:
                    continue
                _vae_np  = _load_vae_latent(_stem, dcfg.get("vae_cache_dir"),
                                             expected_hw=None)
                _text_np = _load_qwen3_embed(_stem, dcfg.get("qwen3_cache_dir"))
                if _vae_np is None or _text_np is None:
                    continue
                _lat = mx.array(_vae_np[None], dtype=mx.bfloat16)
                _txt = mx.array(_text_np[None], dtype=mx.bfloat16)
                _t   = mx.array([500], dtype=mx.int32)
                _noise = mx.random.normal(_lat.shape, dtype=_lat.dtype)
                _alpha, _sigma = get_schedule_values(_t)
                _noisy, _target = fused_flow_noise(_lat, _noise, _alpha, _sigma)
                _fs = _flux_forward_no_ip(flux, _noisy, _txt, _t)
                mx.eval(_fs["qs"], _fs["h_final"], _fs["temb"], _target)
                # No-grad forward: call loss_fn directly (not through value_and_grad)
                _siglip_zero = mx.zeros((1, 729, acfg["siglip_dim"]), dtype=mx.bfloat16)
                _loss = loss_fn(_siglip_zero, mx.array(True), mx.array(False), _fs, _target)
                mx.eval(_loss)
                losses.append(float(_loss.item()))
                del _fs, _loss, _noisy, _target
                mx.clear_cache()
                if len(losses) >= 16:
                    break
        return sum(losses) / len(losses) if losses else None

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining: {tcfg['num_steps']:,} steps, batch_size={dcfg['batch_size']}\n")

    step = start_step
    t0 = time.time()
    t_start = time.time()
    log_interval = ocfg["log_every"]
    loss_history: list[float] = []  # rolling window for smoothed loss
    loss_smooth = 0.0
    _pipeline_chunk = config.get("_chunk")  # set by --chunk arg; None when run standalone

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

    # T-04: per-source loss (source inferred from shard name; "unknown" until shard naming
    # includes source prefix e.g. chunk1_laion_0000.tar — requires build_shards tagging)
    source_stats: dict = defaultdict(lambda: {"steps": 0, "loss_sum": 0.0})

    # T-05: validation loss on held-out set
    _val_shards: list[str] = []
    val_loss_last: Optional[float] = None
    val_loss_log_path = os.path.join(ocfg["checkpoint_dir"], "val_loss.jsonl")
    _val_every = tcfg.get("val_every", 1000)
    try:
        from pathlib import Path as _P
        import pipeline_lib as _plib
        _held_out = _P(str(_plib.DATA_ROOT)) / "validation" / "held_out"
        if _held_out.exists():
            _val_shards = sorted(str(p) for p in _held_out.glob("*.tar"))
            print(f"Validation held-out: {len(_val_shards)} shards in {_held_out}")
        else:
            print("Validation held-out: not found — T-05 disabled")
    except Exception:
        pass

    # T-06: EMA vs online weight divergence
    ema_drift: float = 0.0

    # T-10: SigLIP coverage per batch window
    _siglip_miss_steps = 0
    _siglip_first_miss_logged = False

    # T-11: gradient clipping events per log window
    _grad_clip_steps = 0

    _boot_hb_stop.set()  # training loop starting — boot heartbeat thread no longer needed

    # Heartbeat is written every _heartbeat_every steps, independent of log_every.
    # When log_every is large (e.g. 500 steps × 0.19 sps ≈ 44 min) the log block
    # fires too infrequently to keep the orchestrator's 900 s stale threshold happy.
    _heartbeat_every = min(log_interval, 100)
    _hb_t0 = time.time()
    _hb_loss = 0.0
    _hb_loss_smooth = 0.0
    _hb_siglip_cov = 100.0   # last known from log block; 100% until first log fires
    _hb_loader_pct = 0.0     # last known from log block

    for images_np, captions, style_refs_np, text_np, vae_np, siglip_np, bucket_hw in loader:
        if step >= tcfg["num_steps"]:
            break

        # How long the GPU was idle waiting for data (time from end of last eval
        # to receiving this batch). Zero on first step (loader pre-filled).
        _t_now = time.time()
        if _t_eval_end is not None:
            _t_data += _t_now - _t_eval_end

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
        null_image = random.random() < tcfg["image_dropout_prob"]
        null_text  = random.random() < tcfg["text_dropout_prob"]
        use_null_image = mx.array(null_image)
        use_null_text  = mx.array(null_text)

        # Encode frozen models — use pre-computed cache if available (§2.7)
        if vae_np is not None:
            latents = mx.array(vae_np, dtype=mx.bfloat16)
        elif vae is not None:
            latents = _vae_encode(vae, images)
        else:
            # VAE not loaded (cache-only mode) but this batch has no cached latents.
            # Skip rather than crash — incomplete precompute coverage.
            continue

        if text_np is not None:
            text_embeds = mx.array(text_np, dtype=mx.bfloat16)
            if null_text:
                text_embeds = mx.zeros_like(text_embeds)
        elif text_encoder is not None:
            captions_in = [""] * len(captions) if null_text else captions
            text_embeds = _encode_text(text_encoder, captions_in)
        else:
            # Text encoder not loaded (cache-only mode) but no cached embeddings.
            continue

        if siglip_np is not None:
            siglip_feats = mx.array(siglip_np, dtype=mx.bfloat16)
        elif siglip is not None:
            siglip_feats = siglip(images)
        else:
            # Cache miss — use zero features (neutral image conditioning).
            # Happens when siglip precompute is partial; treated as null-image dropout.
            if not _siglip_first_miss_logged:
                print(f"  WARNING: SigLIP cache miss at step {step} — "
                      f"falling back to zero features. Check siglip precompute coverage.",
                      flush=True)
                _siglip_first_miss_logged = True
            B = images.shape[0]
            siglip_feats = mx.zeros((B, 729, acfg["siglip_dim"]), dtype=mx.bfloat16)
            _siglip_miss_steps += 1  # T-10

        _t_prep += time.time() - _t0

        # ── Split-forward: Flux no-grad forward, then tiny adapter backward ──
        # Sample timestep + noise outside grad scope. fused_flow_noise is a pure
        # kernel; target needs no grad. Flux forward is run here (no grad), only
        # the adapter graph is traced by nn.value_and_grad.
        _t0 = time.time()
        B_lat = latents.shape[0]
        # Logit-normal timestep sampling: concentrates on mid-timesteps (t≈300–700)
        # where flow-matching loss has structure. Reduces loss variance ~40% vs uniform.
        t_int = mx.clip((mx.sigmoid(mx.random.normal(shape=(B_lat,))) * 1000).astype(mx.int32), 0, 999)
        noise  = mx.random.normal(latents.shape, dtype=latents.dtype)
        alpha_t, sigma_t = get_schedule_values(t_int)
        noisy, target = fused_flow_noise(latents, noise, alpha_t, sigma_t)

        flux_state = _flux_forward_no_ip(flux, noisy, text_embeds, t_int)
        # Force-materialize all Flux tensors before entering autodiff graph.
        # Without this, MLX defers Flux computation into the gradient trace,
        # negating the entire split-forward optimization.
        # target has no dependency on Flux; batching into one eval lets Metal
        # schedule both concurrently rather than across two separate fences.
        mx.eval(flux_state["qs"], flux_state["h_final"], flux_state["temb"], target)
        _t_fwd += time.time() - _t0

        # Adapter-only backward + optimizer update (tiny graph, should be fast)
        _t0 = time.time()
        loss_val, grad_norm_val = compiled_step(
            siglip_feats, use_null_image, use_null_text, flux_state, target,
        )
        # flux_state (25 Q tensors, h_final, temb ≈ 300 MB) and target are no
        # longer needed — release before eval so Metal reclaims them promptly.
        del flux_state, target
        _t_step += time.time() - _t0

        # EMA update (every 10 steps saves ~23 minutes; plans §3.11)
        # Computed here — before mx.eval — so the update graph is folded into
        # the same Metal command submission as the optimizer step.  This avoids
        # a second GPU fence (was: eval params, then separately eval ema_params).
        # The lazy chain is broken by including ema_params in the eval below.
        _do_ema = (step % tcfg["ema_update_every"] == 0)
        if _do_ema:
            ema_params = update_ema(ema_params, adapter, decay=tcfg["ema_decay"])

        # Synchronous eval: prevents lazy-graph accumulation across steps.
        _t0 = time.time()
        if _do_ema:
            mx.eval(loss_val, grad_norm_val, adapter.parameters(), optimizer.state, ema_params)
        else:
            mx.eval(loss_val, grad_norm_val, adapter.parameters(), optimizer.state)
        # Return freed buffers to the OS immediately.  Without this, MLX holds
        # them in a cache that inflates peak wired memory and triggers jetsam.
        mx.clear_cache()
        _t_eval_end = time.time()
        _t_eval += _t_eval_end - _t0

        step += 1

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
        if _gn > tcfg["grad_clip"]:
            _grad_clip_steps += 1

        # T-02: per-bucket throughput
        _bk = f"{bH}x{bW}"
        _step_loss_val = float(loss_val.item())
        bucket_stats[_bk]["steps"]    += 1
        bucket_stats[_bk]["loss_sum"] += _step_loss_val
        bucket_stats[_bk]["time_sum"] += _t_eval_end - _t_bucket_start

        # T-04: per-source loss (source tag requires shard naming; "unknown" until then)
        source_stats["unknown"]["steps"]    += 1
        source_stats["unknown"]["loss_sum"] += _step_loss_val

        # T-06: EMA drift (RMS diff between online and EMA weights, first 5 param tensors)
        if step % 500 == 0:
            try:
                _flat_online = dict(_flatten(adapter.parameters()))
                _flat_ema    = dict(_flatten(ema_params))
                _drift_vals  = []
                for _k in list(_flat_online)[:5]:
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
            steps_remaining = tcfg["num_steps"] - step
            eta_s = steps_remaining / steps_per_sec if steps_per_sec > 0 else 0
            eta_h, eta_m = divmod(int(eta_s) // 60, 60)
            print(
                f"step {step:>7,}/{tcfg['num_steps']:,}"
                f"  loss {loss_scalar:.4f} (avg {loss_smooth:.4f})"
                f"  lr {lr_now:.2e}"
                f"  {steps_per_sec:.2f} steps/s"
                f"  ETA {eta_h}h{eta_m:02d}m",
                flush=True,
            )
            # TP-005: per-phase timing breakdown
            # fwd = Flux forward + flux_state eval (no grad)
            # step = adapter compiled_step (backward, should be small)
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

            # T-03: memory pressure
            if _HAS_PSUTIL:
                _vm = _psutil.virtual_memory()
                mem_used_gb = round(_vm.used / 1e9, 1)
                mem_available_gb = round(_vm.available / 1e9, 1)
                print(f"  mem: {mem_used_gb:.1f} GB used  {mem_available_gb:.1f} GB free",
                      flush=True)
                if mem_available_gb < 6.0:
                    print(f"  WARNING: memory pressure — only {mem_available_gb:.1f} GB available",
                          flush=True)

            # T-04: per-source loss summary
            _src_summary = {
                src: {"steps": v["steps"],
                      "loss_avg": round(v["loss_sum"] / v["steps"], 4) if v["steps"] else 0}
                for src, v in source_stats.items()
            }

            # T-06: EMA drift
            if ema_drift > 0:
                print(f"  ema_drift={ema_drift:.5f}", flush=True)

            # T-07: loader wait fraction (derived from phase accumulators already printed)
            _total_phase = _t_data + _t_prep + _t_fwd + _t_step + _t_eval
            _loader_pct = round(100 * _t_data / _total_phase, 1) if _total_phase > 0 else 0.0
            _loader_wait_ms = round(_t_data * 1000 / log_interval, 1)
            _compute_ms = round((_t_prep + _t_fwd + _t_step + _t_eval) * 1000 / log_interval, 1)
            if _loader_pct > 20:
                print(f"  WARNING: loader wait {_loader_pct:.0f}% — consider increasing prefetch_batches",
                      flush=True)

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
            if _grad_clip_steps > 0:
                _clip_pct = round(100 * _grad_clip_steps / log_interval, 1)
                print(f"  grad_clipped={_clip_pct:.0f}% ({_grad_clip_steps}/{log_interval} steps "
                      f"had norm > {tcfg['grad_clip']})", flush=True)
            _grad_clip_steps = 0

            if wandb_run:
                wandb_run.log(
                    {"loss": loss_scalar, "loss_smooth": loss_smooth,
                     "lr": lr_now, "steps_per_sec": steps_per_sec,
                     "grad_norm": _gn, "grad_norm_smooth": grad_norm_smooth,
                     "ema_drift": ema_drift, "siglip_coverage_pct": _siglip_cov,
                     "loader_wait_pct": _loader_pct},
                    step=step,
                )
            # Write machine-readable heartbeat to the pipeline location so the
            # orchestrator and pipeline_status.py can monitor liveness and anomalies.
            total_elapsed = time.time() - t_start
            try:
                from pipeline_lib import write_heartbeat as _write_hb
                _write_hb(
                    "trainer", _pipeline_chunk,
                    step=step,
                    total_steps=tcfg["num_steps"],
                    loss=round(loss_scalar, 6),
                    loss_smooth=round(loss_smooth, 6),
                    lr=lr_now,
                    steps_per_sec=round(steps_per_sec, 4),
                    eta_sec=int(eta_s),
                    elapsed_seconds=int(total_elapsed),
                    grad_norm=round(_gn, 4),
                    grad_norm_smooth=round(grad_norm_smooth, 4),
                    grad_clip_pct=round(100 * _grad_clip_steps / log_interval, 1),
                    ema_drift=round(ema_drift, 5),
                    val_loss=round(val_loss_last, 6) if val_loss_last is not None else None,
                    siglip_coverage_pct=_siglip_cov,
                    loader_wait_ms_avg=_loader_wait_ms,
                    compute_ms_avg=_compute_ms,
                    source_loss=_src_summary,
                    buckets=_bkt_summary,
                    mem_used_gb=mem_used_gb,
                    mem_available_gb=mem_available_gb,
                )
            except Exception:
                pass
            # Reset rolling stats so per-interval averages don't drift over the
            # entire run as loss improves and bucket distribution shifts.
            bucket_stats.clear()
            source_stats.clear()
            # Release unused Metal buffer pool entries every log interval.
            # MLX pools buffers internally; without periodic clearing, pool
            # growth can exhaust unified memory after ~1200 steps (~12 evals).
            mx.clear_cache()
            import gc; gc.collect()
            t0 = time.time()

        # Heartbeat — decoupled from log_every so the orchestrator always sees
        # liveness even when log_every is large.  The full rich heartbeat is
        # written by the log block above at log_interval steps; this lightweight
        # one fires at _heartbeat_every (≤100) steps for the intervals between.
        if step % _heartbeat_every == 0 and step > 0:
            _hb_elapsed = time.time() - _hb_t0
            _hb_t0 = time.time()
            _hb_sps = _heartbeat_every / _hb_elapsed if _hb_elapsed > 0 else 0.0
            _hb_loss = float(loss_val.item())
            _hb_loss_smooth = 0.98 * _hb_loss_smooth + 0.02 * _hb_loss
            if step % log_interval != 0:
                try:
                    from pipeline_lib import write_heartbeat as _write_hb
                    _write_hb(
                        "trainer", _pipeline_chunk,
                        step=step,
                        total_steps=tcfg["num_steps"],
                        loss=round(_hb_loss, 6),
                        loss_smooth=round(_hb_loss_smooth, 6),
                        lr=float(optimizer.learning_rate),
                        steps_per_sec=round(_hb_sps, 4),
                        eta_sec=int((tcfg["num_steps"] - step) / _hb_sps) if _hb_sps > 0 else 0,
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
            save_checkpoint_async(adapter, ema_params, step,
                                  ocfg["checkpoint_dir"], ocfg["keep_last_n"],
                                  lineage=_lineage)


    # Final checkpoint + EMA export
    _lineage = {**_lineage_base, "step": step, "loss": round(loss_smooth, 6)}
    save_checkpoint_async(adapter, ema_params, step,
                          ocfg["checkpoint_dir"], keep_last_n=999,
                          lineage=_lineage)

    # Save best EMA as final export
    mx.eval(ema_params)
    _best_path = os.path.join(ocfg["checkpoint_dir"], "best.safetensors")
    save_ema(ema_params, _best_path)

    # Write lineage sidecar so best.safetensors is self-documenting.
    _best_meta = {**_lineage_base, "step": step, "loss": round(loss_smooth, 6),
                  "chunk": _pipeline_chunk}
    try:
        with open(_best_path.replace(".safetensors", ".json"), "w") as _f:
            json.dump(_best_meta, _f, indent=2)
    except OSError:
        pass
    print(f"\nTraining complete. EMA weights: {ocfg['checkpoint_dir']}/best.safetensors")


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
    parser.add_argument("--data-root", default=None,
                        help="Root directory for shards and precomputed caches. "
                             "Relative paths in the config YAML are prefixed with this value. "
                             "Defaults to the current working directory.")
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

    if args.resume:
        config["model"]["warmstart_path"] = args.resume
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

    if args.dry_run:
        print("Config OK:")
        print(yaml.dump(config, default_flow_style=False))
        return

    train(config)


if __name__ == "__main__":
    main()
