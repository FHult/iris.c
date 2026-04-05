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

Wall clock: ~15-20 min on M1 Max (Flux forward × 2500 steps).

Records that already exist in the output directory are not re-extracted.
Re-running after a crash resumes safely.
"""

import argparse
import heapq
import io
import os
import random
import sys
import tarfile
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_TRAIN_DIR  = _SCRIPT_DIR.parent
sys.path.insert(0, str(_TRAIN_DIR))

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    print("Error: MLX not found. Run: source train/.venv/bin/activate", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Embed loaders (same dequant as dataset.py)
# ---------------------------------------------------------------------------

def _load_qwen3(rec_id: str, cache_dir: str):
    path = os.path.join(cache_dir, f"{rec_id}.npz")
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path)
        q, scale = d["q"], d["scale"]
        lo = np.empty(q.shape, dtype=np.int8)
        hi = np.empty(q.shape, dtype=np.int8)
        np.bitwise_and(q, np.int8(0x0F), out=lo)
        np.right_shift(q, 4, out=hi)
        np.bitwise_and(hi, np.int8(0x0F), out=hi)
        full = np.empty((q.shape[0], q.shape[1] * 2), dtype=np.int8)
        full[:, 0::2] = lo
        full[:, 1::2] = hi
        return (full.astype(np.float32) * scale.astype(np.float32)).astype(np.float16)
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
        d = np.load(path)
        q, scale = d["q"], d["scale"]
        lo = np.empty(q.shape, dtype=np.int8)
        hi = np.empty(q.shape, dtype=np.int8)
        np.bitwise_and(q, np.int8(0x0F), out=lo)
        np.right_shift(q, 4, out=hi)
        np.bitwise_and(hi, np.int8(0x0F), out=hi)
        full = np.empty((q.shape[0], q.shape[1] * 2), dtype=np.int8)
        full[:, 0::2] = lo
        full[:, 1::2] = hi
        return (full.astype(np.float32) * scale.astype(np.float32)).astype(np.float16)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SigLIP encoding (when cache not available)
# ---------------------------------------------------------------------------

_siglip_model = None

def _ensure_siglip(model_name: str):
    global _siglip_model
    if _siglip_model is not None:
        return
    print(f"  Loading SigLIP '{model_name}' for on-the-fly encoding...")
    try:
        from mlx_vlm import load as vlm_load
        model, _ = vlm_load(model_name)
        model.eval()
        _siglip_model = ("mlx", model.vision_model if hasattr(model, "vision_model") else model)
        return
    except Exception:
        pass
    try:
        from transformers import AutoModel
        hf = AutoModel.from_pretrained(model_name).vision_model.eval()
        _siglip_model = ("torch", hf)
        return
    except Exception as e:
        raise RuntimeError(f"Cannot load SigLIP '{model_name}': {e}") from e


def _encode_siglip_jpg(jpg_bytes: bytes, model_name: str) -> np.ndarray:
    """Decode JPEG, resize to 384×384, run SigLIP. Returns [1, 729, 1152] float16."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
        img = img.resize((384, 384), Image.LANCZOS)
        img_np = np.array(img, dtype=np.float32) / 127.5 - 1.0   # [-1, 1]
        img_np = img_np.transpose(2, 0, 1)[None]                   # [1, 3, 384, 384]
    except Exception:
        return None

    _ensure_siglip(model_name)
    backend, model = _siglip_model
    if backend == "mlx":
        out = model(mx.array(img_np, dtype=mx.bfloat16))
        return np.array(out.astype(mx.float16))          # [1, 729, 1152]
    else:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(img_np)
            out = model(pixel_values=t).last_hidden_state  # [1, 729, 1152]
        return out.numpy().astype(np.float16)


# ---------------------------------------------------------------------------
# Per-sample loss (single forward pass, no gradients)
# ---------------------------------------------------------------------------

def _eval_loss(adapter, flux, text_np, vae_np, siglip_np) -> float:
    """Compute flow-matching loss for one sample. Fixed t=500 for stable ranking."""
    from ip_adapter.loss import fused_flow_noise, get_schedule_values
    from train_ip_adapter import _flux_forward_with_ip

    text_embeds = mx.array(text_np[None], dtype=mx.bfloat16)   # [1, seq, 7680]
    latents     = mx.array(vae_np[None],  dtype=mx.bfloat16)   # [1, 32, H, W]
    siglip_feats = mx.array(siglip_np[None], dtype=mx.bfloat16) # [1, 729, 1152]

    ip_embeds = adapter.get_image_embeds(siglip_feats)
    k_ip_all, v_ip_all = adapter.get_kv_all(ip_embeds)

    t_int = mx.array([500], dtype=mx.int32)
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
                        help="EMA checkpoint (best.safetensors from training)")
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
    parser.add_argument("--output",        required=True,
                        help="Output dir for hard example .tar files")
    parser.add_argument("--eval-records",  type=int, default=5000,
                        help="Max records to evaluate for loss (default 5000)")
    parser.add_argument("--top-k",         type=int, default=2000,
                        help="Number of hard examples to extract (default 2000)")
    parser.add_argument("--shard-size",    type=int, default=500,
                        help="Max records per output .tar (default 500)")
    parser.add_argument("--seed",          type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output, exist_ok=True)

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

    print(f"Scanning {len(shard_paths)} shards for precomputed candidates...")
    candidates = []   # [(rec_id, shard_path)]
    for shard_path in shard_paths:
        try:
            with tarfile.open(shard_path) as t:
                stems = set()
                for m in t.getmembers():
                    stem, _, ext = m.name.rpartition(".")
                    if ext.lower() in ("jpg", "jpeg"):
                        stems.add(stem)
            # Only include records with both caches present and not already extracted
            for stem in stems:
                if stem in existing_ids:
                    continue
                q_ok = os.path.exists(os.path.join(args.qwen3_cache, f"{stem}.npz"))
                v_ok = os.path.exists(os.path.join(args.vae_cache,   f"{stem}.npz"))
                if q_ok and v_ok:
                    candidates.append((stem, shard_path))
        except Exception as e:
            print(f"  Warning: failed to read {shard_path}: {e}", file=sys.stderr)

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

    print(f"  Adapter (EMA from {args.checkpoint})...")
    # Build adapter with same default dims as stage1 config
    adapter = IPAdapterKlein(
        num_blocks=25, hidden_dim=3072,
        num_image_tokens=128, siglip_dim=1152,
        perceiver_depth=4, perceiver_heads=16,
    )
    ema_params = load_ema_from_checkpoint(args.checkpoint)
    if ema_params:
        from train_ip_adapter import _nested_update
        _nested_update(adapter, ema_params)
        print("    EMA weights loaded")
    else:
        from train_ip_adapter import load_checkpoint
        load_checkpoint(adapter, args.checkpoint)
        print("    Raw weights loaded (no EMA found)")
    adapter.freeze()
    mx.eval(adapter.parameters())

    # ── Eval pass — pass 1: compute loss per record ───────────────────────────
    print(f"\nPass 1: computing loss for {n_eval:,} records...")
    # Min-heap of size top_k: (loss, rec_id, shard_path)
    # We keep a MAX-heap via negation so we can efficiently drop the lowest.
    heap = []   # (-loss, rec_id, shard_path)  — max-heap of top-K
    done = 0
    skipped = 0

    for rec_id, shard_path in sample:
        text_np   = _load_qwen3(rec_id, args.qwen3_cache)
        vae_np    = _load_vae(rec_id, args.vae_cache)
        siglip_np = None

        if args.siglip_cache:
            siglip_np = _load_siglip(rec_id, args.siglip_cache)

        if siglip_np is None:
            # Need to decode the JPEG and encode with SigLIP
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

        try:
            loss = _eval_loss(adapter, flux, text_np, vae_np, siglip_np)
        except Exception as e:
            print(f"  Warning: eval failed for {rec_id}: {e}", file=sys.stderr)
            skipped += 1
            continue

        entry = (-loss, rec_id, shard_path)
        if len(heap) < args.top_k:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:   # loss > current min in heap
            heapq.heapreplace(heap, entry)

        done += 1
        if done % 500 == 0 or done == n_eval:
            threshold = -heap[0][0] if heap else 0.0
            print(f"  [{done}/{n_eval}]  skipped={skipped}  "
                  f"top-{len(heap)} threshold loss={threshold:.4f}", flush=True)

    print(f"\nEval complete: {done} evaluated, {skipped} skipped, "
          f"{len(heap)} hard examples selected")
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
    existing_tars = sorted(f for f in os.listdir(args.output) if f.endswith(".tar"))
    if existing_tars:
        out_idx = int(existing_tars[-1].replace(".tar", "")) + 1

    i = 0
    while i < len(records_to_write):
        batch = records_to_write[i:i + args.shard_size]
        out_path = os.path.join(args.output, f"{out_idx:06d}.tar")
        with tarfile.open(out_path, "w") as out_tar:
            for rec_id, jpg_bytes, txt_bytes in batch:
                for data, ext in [(jpg_bytes, "jpg"), (txt_bytes, "txt")]:
                    info = tarfile.TarInfo(name=f"{rec_id}.{ext}")
                    info.size = len(data)
                    out_tar.addfile(info, io.BytesIO(data))
        print(f"  Wrote {out_path} ({len(batch)} records)")
        i += args.shard_size
        out_idx += 1

    total = len(records_to_write)
    print(f"\nDone: {total} hard examples in {args.output}/")
    print(f"  Mix into training with:  hard_example_dir: {args.output}")
    print(f"                           hard_mix_ratio: 0.05")

    # Update manifest: append new IDs so future resumes skip re-scanning tars.
    new_ids = sorted({rec_id for rec_id, _, _ in records_to_write})
    if new_ids:
        with open(manifest_path, "a") as _mf:
            _mf.write("\n".join(new_ids) + "\n")


if __name__ == "__main__":
    main()
