"""
train/scripts/mine_hard_examples.py — Mine hard training examples after a chunk.

Runs a lightweight eval pass over a random sample of precomputed shards.
Computes per-sample flow-matching loss using the trained EMA checkpoint.
Extracts the top-K highest-loss records into hard_examples/ as WDS .tar files.

The hard_examples/ directory is persistent — never deleted between chunks.
It is mixed into subsequent chunks' training at hard_mix_ratio (default 5%).

Usage:
    python train/scripts/mine_hard_examples.py \\
        --checkpoint checkpoints/stage1/best.safetensors \\
        --shards     train/data/shards \\
        --qwen3-cache train/data/precomputed/qwen3 \\
        --vae-cache   train/data/precomputed/vae \\
        --flux-model  flux-klein-model \\
        --output      train/data/hard_examples \\
        [--siglip-cache train/data/precomputed/siglip] \\
        [--eval-records 5000] \\
        [--top-k 2000]

Wall clock: ~15-20 min on M1 Max at eval-batch=8 (default).

Records that already exist in the output directory are not re-extracted.
Re-running after a crash resumes safely.
"""

import argparse
import heapq
import io
import math
import os
import random
import sys
import tarfile
import threading
import time
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_TRAIN_DIR  = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_TRAIN_DIR))

from pipeline_lib import write_heartbeat, log_event, log_orch

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    print("Error: MLX not found. Run: source train/.venv/bin/activate", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Embed loaders (same dequant as dataset.py)
# ---------------------------------------------------------------------------

def _dequant_4bit(d) -> np.ndarray:
    """Unpack nibble-packed 4-bit quantised array → float16. d is a loaded npz dict."""
    q, scale = d["q"], d["scale"]
    lo = np.empty(q.shape, dtype=np.int8)
    hi = np.empty(q.shape, dtype=np.int8)
    np.bitwise_and(q, np.int8(0x0F), out=lo)
    np.right_shift(q, 4, out=hi)
    np.bitwise_and(hi, np.int8(0x0F), out=hi)
    full = np.empty((*q.shape[:-1], q.shape[-1] * 2), dtype=np.int8)
    full[..., 0::2] = lo
    full[..., 1::2] = hi
    return (full.astype(np.float32) * scale.astype(np.float32)).astype(np.float16)


def _load_qwen3(rec_id: str, cache_dir: str):
    path = os.path.join(cache_dir, f"{rec_id}.npz")
    if not os.path.exists(path):
        return None
    try:
        return _dequant_4bit(np.load(path))
    except Exception:
        return None


def _load_vae(rec_id: str, cache_dir: str):
    path = os.path.join(cache_dir, f"{rec_id}.npz")
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path)
        return (d["q"].astype(np.float32) * d["scale"].astype(np.float32)).astype(np.float16)
    except Exception:
        return None


def _load_siglip(rec_id: str, cache_dir: str):
    path = os.path.join(cache_dir, f"{rec_id}.npz")
    if not os.path.exists(path):
        return None
    try:
        return _dequant_4bit(np.load(path))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SigLIP encoding (when cache not available)
# ---------------------------------------------------------------------------

_siglip_model = None
_NULL_SIGLIP = np.zeros((729, 1152), dtype=np.float16)

def _ensure_siglip(model_name: str):
    global _siglip_model
    if _siglip_model is not None:
        return
    print(f"  Loading SigLIP '{model_name}' for on-the-fly encoding...")
    # Prefer open_clip (already in venv, MPS-accelerated) over transformers fallback.
    try:
        import open_clip, torch
        oc_model, _, oc_preprocess = open_clip.create_model_and_transforms(
            'hf-hub:timm/ViT-SO400M-14-SigLIP-384', device='mps'
        )
        oc_model.eval()
        _siglip_model = ("open_clip", oc_model, oc_preprocess)
        return
    except Exception:
        pass
    try:
        from transformers import AutoModel
        hf = AutoModel.from_pretrained(model_name).vision_model.eval()
        _siglip_model = ("torch", hf, None)
        return
    except Exception as e:
        raise RuntimeError(f"Cannot load SigLIP '{model_name}': {e}") from e


def _encode_siglip_jpg(jpg_bytes: bytes, model_name: str) -> np.ndarray:
    """Decode JPEG, run SigLIP. Returns [1, 729, 1152] float16."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    except Exception:
        return None

    _ensure_siglip(model_name)
    backend, model, preprocess = _siglip_model

    if backend == "open_clip":
        import torch
        t = preprocess(img).unsqueeze(0).to('mps')
        with torch.no_grad():
            out = model.encode_image(t, normalize=False)
            # encode_image returns pooled [1, dim]; for patch tokens use visual.forward
            out = model.visual.trunk.forward_features(t)  # [1, 729, 1152]
        return out[0].cpu().numpy().astype(np.float16)    # strip batch → (729, 1152)
    else:
        img_np = np.array(img.resize((384, 384), Image.LANCZOS), dtype=np.float32)
        img_np = (img_np / 127.5 - 1.0).transpose(2, 0, 1)[None]
        import torch
        with torch.no_grad():
            t = torch.from_numpy(img_np)
            out = model(pixel_values=t).last_hidden_state  # [1, 729, 1152]
        return out[0].numpy().astype(np.float16)           # strip batch → (729, 1152)


# ---------------------------------------------------------------------------
# Per-sample loss (single forward pass, no gradients)
# ---------------------------------------------------------------------------

def _eval_loss(adapter, flux, text_np, vae_np, siglip_np) -> float:
    """Compute flow-matching loss for one sample. Random t for unbiased ranking."""
    from ip_adapter.loss import fused_flow_noise, get_schedule_values
    from train_ip_adapter import _flux_forward_with_ip

    text_embeds = mx.array(text_np[None], dtype=mx.bfloat16)   # [1, seq, 7680]
    latents     = mx.array(vae_np[None],  dtype=mx.bfloat16)   # [1, 32, H, W]
    siglip_feats = mx.array(siglip_np[None], dtype=mx.bfloat16) # [1, 729, 1152]

    ip_embeds = adapter.get_image_embeds(siglip_feats)
    k_ip_all, v_ip_all = adapter.get_kv_all(ip_embeds)

    t_int = mx.clip(
        (mx.sigmoid(mx.random.normal(shape=(1,))) * 1000).astype(mx.int32), 0, 999
    )
    alpha_t, sigma_t = get_schedule_values(t_int)

    noise = mx.random.normal(latents.shape, dtype=latents.dtype)
    noisy, target = fused_flow_noise(latents, noise, alpha_t, sigma_t)

    pred = _flux_forward_with_ip(
        flux, noisy, text_embeds, t_int,
        k_ip_all=k_ip_all,
        v_ip_all=v_ip_all,
        ip_scale=adapter.scale,
    )

    loss = float(mx.mean((pred - target) ** 2).item())
    mx.clear_cache()
    return loss


def _eval_loss_batch(adapter, flux, text_list, vae_list, siglip_list, null_kv=None) -> list:
    """
    Batch eval: compute flow-matching loss for N samples in one Flux forward pass.

    All VAE latents must be the same spatial shape (standard: all 512×512 → [32,64,64]).
    Text embeddings are zero-padded to max_seq within the batch; this slightly biases
    absolute loss values but preserves relative ranking, which is all mining needs.

    null_kv: precomputed (k_ip [1,B,128,3072], v_ip [1,B,128,3072]) for null-siglip
             mode. Pass None when siglip_list carries real per-sample features.

    Returns a list of N float losses (same order as inputs).
    """
    from ip_adapter.loss import fused_flow_noise, get_schedule_values
    from train_ip_adapter import _flux_forward_with_ip

    B = len(text_list)
    latents = mx.array(np.stack(vae_list, axis=0), dtype=mx.bfloat16)  # [B, 32, H, W]

    max_seq = max(t.shape[0] for t in text_list)
    txt_dim = text_list[0].shape[1]
    text_pad = np.zeros((B, max_seq, txt_dim), dtype=np.float16)
    for i, t in enumerate(text_list):
        text_pad[i, :t.shape[0]] = t
    text_embeds = mx.array(text_pad, dtype=mx.bfloat16)  # [B, max_seq, txt_dim]

    if null_kv is not None:
        k_ip_all = mx.repeat(null_kv[0], B, axis=0)
        v_ip_all = mx.repeat(null_kv[1], B, axis=0)
    else:
        siglip_feats = mx.array(np.stack(siglip_list, axis=0), dtype=mx.bfloat16)
        ip_embeds = adapter.get_image_embeds(siglip_feats)
        k_ip_all, v_ip_all = adapter.get_kv_all(ip_embeds)

    t_int = mx.clip(
        (mx.sigmoid(mx.random.normal(shape=(B,))) * 1000).astype(mx.int32), 0, 999
    )
    alpha_t, sigma_t = get_schedule_values(t_int)
    noise = mx.random.normal(latents.shape, dtype=latents.dtype)
    noisy, target = fused_flow_noise(latents, noise, alpha_t, sigma_t)

    pred = _flux_forward_with_ip(
        flux, noisy, text_embeds, t_int,
        k_ip_all=k_ip_all,
        v_ip_all=v_ip_all,
        ip_scale=adapter.scale,
    )
    # Per-sample MSE: flatten spatial dims, mean over [C*H*W]
    losses = mx.mean(((pred - target) ** 2).reshape(B, -1), axis=1).tolist()
    mx.clear_cache()
    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mine hard training examples by per-sample loss ranking."
    )
    _repo = _TRAIN_DIR.parent
    _flux_default = next(
        (str(p) for p in [
            _repo / "flux-klein-model",
            _repo / "flux-klein-4b",
        ] if (p / "vae").is_dir()),
        "black-forest-labs/FLUX.2-Klein",
    )
    parser.add_argument("--checkpoint",    required=True,
                        help="Checkpoint path (best.safetensors or a step_*.safetensors)")
    parser.add_argument("--use-ema",       dest="use_ema", action="store_true",
                        help="Require EMA weights for scoring. Also checks for a companion "
                             "{checkpoint}.ema.safetensors file. Exits if EMA unavailable.")
    parser.add_argument("--shards",        required=True,
                        help="Directory of unified .tar shards")
    parser.add_argument("--qwen3-cache",   required=True,
                        help="Precomputed Qwen3 .npz cache dir")
    parser.add_argument("--vae-cache",     required=True,
                        help="Precomputed VAE .npz cache dir")
    parser.add_argument("--flux-model",    default=_flux_default,
                        help="Flux Klein model path")
    parser.add_argument("--siglip-model",  default="google/siglip-so400m-patch14-384")
    parser.add_argument("--siglip-cache",  default=None,
                        help="Precomputed SigLIP .npz cache dir (optional — fast path)")
    parser.add_argument("--null-siglip",   action="store_true",
                        help="Use zero SigLIP features instead of live inference. "
                             "Use when training was run without SigLIP precompute "
                             "so loss rankings match training conditions.")
    parser.add_argument("--output",        required=True,
                        help="Output dir for hard example .tar files")
    parser.add_argument("--chunk",          type=int, default=None,
                        help="Pipeline chunk number (for heartbeat naming)")
    parser.add_argument("--eval-records",  type=int, default=5000,
                        help="Max records to evaluate for loss (default 5000)")
    parser.add_argument("--top-k",         type=int, default=2000,
                        help="Number of hard examples to extract (default 2000)")
    parser.add_argument("--shard-size",    type=int, default=500,
                        help="Max records per output .tar (default 500)")
    parser.add_argument("--eval-batch",   type=int, default=8,
                        help="Flux forward batch size during eval (default 8; use 4 or 1 if OOM)")
    parser.add_argument("--blocklist",     default=None, metavar="PATH",
                        help="File of duplicate record IDs to exclude from mining (one per line)")
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument("--ai",            action="store_true",
                        help="Emit compact JSON summary to stdout at completion; progress to stderr")
    args = parser.parse_args()

    # Block manual runs when GPU is already in use by training or the pipeline.
    # Orchestrated runs (PIPELINE_ORCHESTRATED=1) skip this — the orchestrator
    # manages GPU serialisation via GPU_TOKEN and the file lock itself.
    if not os.environ.get("PIPELINE_ORCHESTRATED"):
        import atexit
        from pipeline_lib import (
            gpu_is_free, tmux_window_exists, TMUX_PREP_WIN,
            acquire_gpu_lock, release_gpu_lock, gpu_lock_holder,
        )
        if not gpu_is_free():
            print("ERROR: iris-train is running. GPU is in use by training.", file=sys.stderr)
            sys.exit(1)
        if tmux_window_exists(TMUX_PREP_WIN):
            print("ERROR: iris-prep is running. GPU is in use by the pipeline.", file=sys.stderr)
            sys.exit(1)
        _lock_info = gpu_lock_holder()
        if _lock_info is not None:
            print(
                f"ERROR: GPU lock held by '{_lock_info.get('label', '?')}' "
                f"(PID {_lock_info.get('pid', '?')}). Exiting.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not acquire_gpu_lock("mine (manual)"):
            print("ERROR: GPU lock acquire race — try again.", file=sys.stderr)
            sys.exit(1)
        atexit.register(release_gpu_lock)

    # Resolve flux model: accept bare name or full path
    fm = Path(args.flux_model)
    if not fm.is_dir():
        fm = _TRAIN_DIR.parent / args.flux_model
    if not fm.is_dir():
        print(f"Error: flux model not found: {args.flux_model}", file=sys.stderr)
        sys.exit(1)
    args.flux_model = str(fm)

    rng = random.Random(args.seed)
    os.makedirs(args.output, exist_ok=True)
    log_event("mine_hard_examples", "start", output=args.output,
              eval_records=args.eval_records, top_k=args.top_k)

    # ── Find which record IDs already exist in the output dir ─────────────────
    # Use .existing_ids.txt manifest when present (fast); fall back to tar scan
    # only on the first run (or if the manifest was deleted).
    manifest_path = os.path.join(args.output, ".existing_ids.txt")
    existing_ids: set = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as _mf:
            existing_ids = {line.strip() for line in _mf if line.strip()}
        print(f"  Resuming: {len(existing_ids)} records in manifest — skipping those")
    else:
        for fn in os.listdir(args.output):
            if not fn.endswith(".tar"):
                continue
            try:
                with tarfile.open(os.path.join(args.output, fn)) as t:
                    for m in t.getmembers():
                        stem, _, ext = m.name.rpartition(".")
                        if ext.lower() in ("jpg", "jpeg", "txt"):
                            existing_ids.add(stem)
            except Exception:
                pass
        if existing_ids:
            print(f"  Resuming: {len(existing_ids)} records already extracted — skipping those")

    # ── Collect candidate records (those with qwen3 + vae cache) ──────────────
    import glob as _glob
    shard_paths = sorted(_glob.glob(os.path.join(args.shards, "*.tar")))
    if not shard_paths:
        print(f"No shards found in {args.shards}", file=sys.stderr)
        sys.exit(1)

    # Build shard path map: stem → full path ("000006" → ".../000006.tar")
    shard_map = {Path(p).stem: p for p in shard_paths}

    # Scan cache dirs once; intersect to find records with both embeddings present.
    # Record IDs encode the shard stem as a prefix (e.g. "000006_0001"), so we can
    # derive the shard path directly without opening any tar files.
    # This replaces opening N tar files + N×5000×2 stat() calls with 2 directory scans.
    print(f"Scanning cache dirs for precomputed candidates ({len(shard_paths)} shards)...")
    cached_q  = {f[:-4] for f in os.listdir(args.qwen3_cache) if f.endswith(".npz")}
    cached_v  = {f[:-4] for f in os.listdir(args.vae_cache)   if f.endswith(".npz")}
    blocklist = {l.strip() for l in Path(args.blocklist).read_text().splitlines() if l.strip()} if args.blocklist else set()
    eligible  = (cached_q & cached_v) - existing_ids - blocklist

    candidates = []   # [(rec_id, shard_path)]
    for rec_id in eligible:
        shard_stem = rec_id.split("_")[0]
        if shard_stem in shard_map:
            candidates.append((rec_id, shard_map[shard_stem]))

    if not candidates:
        print("No precomputed candidates found — nothing to mine.", file=sys.stderr)
        sys.exit(0)

    print(f"  {len(candidates):,} eligible records across {len(shard_paths)} shards")

    # ── Sample eval budget ────────────────────────────────────────────────────
    n_eval = min(args.eval_records, len(candidates))
    sample = rng.sample(candidates, n_eval)
    print(f"  Evaluating {n_eval:,} records (budget={args.eval_records})")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading models...")
    from mflux.models.flux2 import Flux2Klein
    from ip_adapter.model import IPAdapterKlein
    from train_ip_adapter import load_ema_from_checkpoint

    print("  Flux Klein 4B (frozen)...")
    flux = Flux2Klein(model_path=args.flux_model, quantize=None)
    flux.freeze()

    print(f"  Adapter ({'EMA required' if args.use_ema else 'EMA preferred'}: {args.checkpoint})...")
    # Build adapter with same default dims as stage1 config
    adapter = IPAdapterKlein(
        num_blocks=25, hidden_dim=3072,
        num_image_tokens=128, siglip_dim=1152,
        perceiver_heads=16,
    )
    # When --use-ema: also try a companion .ema.safetensors file (bare EMA keys,
    # load with load_checkpoint) before falling back to load_ema_from_checkpoint.
    ema_params = None
    if args.use_ema:
        companion = Path(args.checkpoint).with_suffix(".ema.safetensors")
        if companion.exists():
            from train_ip_adapter import load_checkpoint as _lc, _nested_update
            _lc(adapter, str(companion))
            print(f"    EMA weights loaded from companion: {companion.name}")
            companion_loaded = True
        else:
            companion_loaded = False
            ema_params = load_ema_from_checkpoint(args.checkpoint)
    else:
        companion_loaded = False
        ema_params = load_ema_from_checkpoint(args.checkpoint)

    if not companion_loaded:
        if ema_params:
            from train_ip_adapter import _nested_update
            _nested_update(adapter, ema_params)
            print("    EMA weights loaded")
        elif args.use_ema:
            print(f"ERROR: --use-ema set but no EMA weights found in {args.checkpoint} "
                  f"and no companion .ema.safetensors exists.", file=sys.stderr)
            sys.exit(1)
        else:
            from train_ip_adapter import load_checkpoint
            load_checkpoint(adapter, args.checkpoint)
            print("    WARNING: no EMA weights found — loaded raw adapter weights")
    adapter.freeze()
    mx.eval(adapter.parameters())

    # ── Precompute null IP KV once (null-siglip: same zeros for every sample) ───
    null_kv = None
    if args.null_siglip and args.eval_batch > 1:
        _sg = mx.zeros((1, 729, 1152), dtype=mx.bfloat16)
        _ip = adapter.get_image_embeds(_sg)
        _nk, _nv = adapter.get_kv_all(_ip)
        mx.eval(_nk, _nv)
        null_kv = (_nk, _nv)
        print(f"  Null IP KV precomputed — reused for all batches (eval_batch={args.eval_batch}).")

    # ── Eval pass — pass 1: compute loss per record ───────────────────────────
    print(f"\nPass 1: computing loss for {n_eval:,} records...")
    heap = []   # (-loss, rec_id, shard_path)  — max-heap of top-K
    done = 0
    skipped = 0
    eval_done_event = threading.Event()
    _threshold = [0.0]   # shared with heartbeat thread via mutable container
    _start_time = [time.time()]

    def _heartbeat_loop():
        while not eval_done_event.is_set():
            elapsed = time.time() - _start_time[0]
            rate = done / elapsed if elapsed > 0 and done > 0 else 0
            eta = int((n_eval - done) / rate) if rate > 0 and done < n_eval else None
            write_heartbeat("mine_hard_examples", args.chunk,
                            done=done, total=n_eval,
                            pct=round(done / n_eval * 100, 1) if n_eval else 100,
                            threshold_loss=round(_threshold[0], 4) if _threshold[0] else None,
                            eta_sec=eta)
            time.sleep(30)

    hb = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb.start()

    _buf_text: list = []
    _buf_vae:  list = []
    _buf_sg:   list = []
    _buf_meta: list = []

    def _flush():
        nonlocal done, skipped
        if not _buf_text:
            return
        try:
            losses = _eval_loss_batch(adapter, flux, _buf_text, _buf_vae, _buf_sg, null_kv)
        except Exception as e:
            import traceback
            print(f"  Batch ({len(_buf_text)}) failed ({e}), retrying serial", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            losses = []
            for t, v, s in zip(_buf_text, _buf_vae, _buf_sg):
                try:
                    losses.append(_eval_loss(adapter, flux, t, v, s))
                except Exception:
                    losses.append(math.nan)

        for loss, (r_id, s_path) in zip(losses, _buf_meta):
            if math.isnan(loss):
                skipped += 1
                continue
            entry = (-loss, r_id, s_path)
            if len(heap) < args.top_k:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
            done += 1
            if done % 100 == 0:
                threshold = -heap[0][0] if heap else 0.0
                _threshold[0] = threshold
                print(f"  [{done}/{n_eval}]  skipped={skipped}  "
                      f"top-{len(heap)} threshold loss={threshold:.4f}", flush=True)

        _buf_text.clear(); _buf_vae.clear(); _buf_sg.clear(); _buf_meta.clear()

    for rec_id, shard_path in sample:
        text_np   = _load_qwen3(rec_id, args.qwen3_cache)
        vae_np    = _load_vae(rec_id, args.vae_cache)
        siglip_np = None

        if args.null_siglip:
            siglip_np = _NULL_SIGLIP
        elif args.siglip_cache:
            siglip_np = _load_siglip(rec_id, args.siglip_cache)

        if siglip_np is None:
            try:
                with tarfile.open(shard_path) as t:
                    members = {m.name: m for m in t.getmembers() if m.isfile()}
                    for name, m in members.items():
                        stem2, _, ext = name.rpartition(".")
                        if stem2 == rec_id and ext.lower() in ("jpg", "jpeg"):
                            jpg_bytes = t.extractfile(m).read()
                            siglip_np = _encode_siglip_jpg(jpg_bytes, args.siglip_model)
                            break
            except Exception:
                pass

        if text_np is None or vae_np is None or siglip_np is None:
            skipped += 1
            continue

        _buf_text.append(text_np)
        _buf_vae.append(vae_np)
        _buf_sg.append(siglip_np)
        _buf_meta.append((rec_id, shard_path))

        if len(_buf_text) >= args.eval_batch:
            _flush()

    _flush()  # drain any remaining partial batch

    eval_done_event.set()
    if done % 100 != 0 and done > 0:
        threshold = -heap[0][0] if heap else 0.0
        _threshold[0] = threshold
        print(f"  [{done}/{n_eval}]  skipped={skipped}  "
              f"top-{len(heap)} threshold loss={threshold:.4f}", flush=True)
    write_heartbeat("mine_hard_examples", args.chunk, done=done, total=n_eval, pct=100,
                    threshold_loss=round(_threshold[0], 4) if _threshold[0] else None,
                    eta_sec=0)

    print(f"\nEval complete: {done} evaluated, {skipped} skipped, "
          f"{len(heap)} hard examples selected")
    log_event("mine_hard_examples", "eval_done",
              evaluated=done, skipped=skipped, hard_count=len(heap))
    if not heap:
        print("No hard examples found — check that precomputed caches are populated.")
        return

    # ── Pass 2: extract JPEG + txt for top-K from source shards ──────────────
    print(f"\nPass 2: extracting {len(heap)} hard examples from source shards...")
    top_k = [(rec_id, shard_path) for (_, rec_id, shard_path) in heap]

    # Group by shard to minimise shard opens
    by_shard: dict = {}
    for rec_id, shard_path in top_k:
        by_shard.setdefault(shard_path, []).append(rec_id)

    records_to_write = []   # [(rec_id, jpg_bytes, txt)]
    for shard_path, ids in by_shard.items():
        id_set = set(ids)
        try:
            with tarfile.open(shard_path) as t:
                members = {m.name: m for m in t.getmembers() if m.isfile()}
                # Build stem → {ext: member} index
                by_stem: dict = {}
                for name, m in members.items():
                    stem2, _, ext = name.rpartition(".")
                    if stem2 in id_set:
                        by_stem.setdefault(stem2, {})[ext.lower()] = m

                for rec_id in ids:
                    exts = by_stem.get(rec_id, {})
                    jpg_key = exts.get("jpg") or exts.get("jpeg")
                    txt_key = exts.get("txt") or exts.get("caption")
                    if not jpg_key or not txt_key:
                        continue
                    jpg_bytes = t.extractfile(jpg_key).read()
                    txt = t.extractfile(txt_key).read()
                    records_to_write.append((rec_id, jpg_bytes, txt))
        except Exception as e:
            print(f"  Warning: failed to read {shard_path}: {e}", file=sys.stderr)

    print(f"  Extracted {len(records_to_write)} records — writing to {args.output}/")

    # Write in output shard files
    out_idx = 0
    # Find next available shard index
    existing_tars = sorted((f for f in os.listdir(args.output) if f.endswith(".tar")),
                           key=lambda f: int(f[:-4]))
    if existing_tars:
        out_idx = int(existing_tars[-1][:-4]) + 1

    i = 0
    while i < len(records_to_write):
        batch = records_to_write[i:i + args.shard_size]
        out_path = os.path.join(args.output, f"{out_idx:06d}.tar")
        tmp_path = out_path + ".tmp"
        with tarfile.open(tmp_path, "w") as out_tar:
            for rec_id, jpg_bytes, txt_bytes in batch:
                for data, ext in [(jpg_bytes, "jpg"), (txt_bytes, "txt")]:
                    info = tarfile.TarInfo(name=f"{rec_id}.{ext}")
                    info.size = len(data)
                    out_tar.addfile(info, io.BytesIO(data))
        os.replace(tmp_path, out_path)
        print(f"  Wrote {out_path} ({len(batch)} records)")
        i += args.shard_size
        out_idx += 1

    total = len(records_to_write)
    _out = sys.stderr if args.ai else sys.stdout
    print(f"\nDone: {total} hard examples in {args.output}/", file=_out)
    print(f"  Mix into training with:  hard_example_dir: {args.output}", file=_out)
    print(f"                           hard_mix_ratio: 0.05", file=_out)
    log_event("mine_hard_examples", "done", total=total, output=args.output)

    # Update manifest: append new IDs so future resumes skip re-scanning tars.
    new_ids = sorted({rec_id for rec_id, _, _ in records_to_write})
    if new_ids:
        with open(manifest_path, "a") as _mf:
            _mf.write("\n".join(new_ids) + "\n")

    if args.ai:
        import json as _json
        top_k_loss_mean = None
        if heap:
            # heap entries are (-loss, rec_id, shard_path) — negate to get positive loss
            top_k_loss_mean = round(sum(-neg_loss for neg_loss, *_ in heap) / len(heap), 6)
        ai_out = {
            "ok": True,
            "done": done,
            "total": n_eval,
            "pct": round(100 * done / n_eval, 1) if n_eval > 0 else 0,
            "top_k_loss_mean": top_k_loss_mean,
            "extracted": total,
            "output": args.output,
        }
        print(_json.dumps(ai_out))


if __name__ == "__main__":
    main()
