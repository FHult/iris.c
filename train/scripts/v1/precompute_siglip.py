"""
train/scripts/precompute_siglip.py — Pre-compute and 4-bit quantise SigLIP features.

Saves ~50ms per training step = ~1.7 hours over Stage 1 (105K steps).
Storage: ~420 GB at 4-bit quantised (3.2 TB at BF16 — impractical at full precision).
Only pre-compute if 420 GB remains after Qwen3 (~143 GB) + VAE (~198 GB).

Run ONCE after build_shards.py + filter_shards.py complete.

Output: one .npz file per sample under data/siglip_q4/{id}.npz
  - q:     uint8 [729, 576]    — packed 4-bit values (1152 dim / 2 = 576 pairs)
  - scale: float16 [729, 1]    — per-token absmax scale

Dequantisation format: identical to Qwen3 (nibble-packed uint8 + per-token scale).
Use 2 processes only — GPU is shared between them; more processes causes contention.

Reference: plans/ip-adapter-training.md §2.7
"""

import argparse
import glob
import io
import multiprocessing
import os
import sys
import tarfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import subprocess
    _PERF_CORES = int(subprocess.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Quantisation helpers (same nibble-pack format as Qwen3)
# ---------------------------------------------------------------------------

def quantize_4bit(arr: np.ndarray):
    """
    Per-token absmax 4-bit quantisation.
    arr: float32 [729, 1152] — SigLIP patch tokens
    Returns:
      q_packed: uint8 [729, 576]  — nibble-packed (pairs of 4-bit values)
      scale:    float16 [729, 1]  — per-token absmax / 7
    """
    scale = np.abs(arr).max(axis=-1, keepdims=True) / 7.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.round(arr / scale), -8, 7).astype(np.int8)
    q_packed = ((q[:, 0::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)).astype(np.uint8)
    return q_packed, scale.astype(np.float16)


def load_siglip_embed(npz_path: str) -> np.ndarray:
    """
    Dequantise a saved SigLIP embedding.
    Returns float16 [729, 1152].
    Called from training prefetch thread.
    """
    d = np.load(npz_path)
    q = d["q"]  # uint8 [729, 576]
    scale = d["scale"]  # float16 [729, 1]
    lo = (q & 0x0F).astype(np.int8)
    hi = ((q >> 4) & 0x0F).astype(np.int8)
    full = np.empty((q.shape[0], q.shape[1] * 2), dtype=np.int8)
    full[:, 0::2] = lo
    full[:, 1::2] = hi
    return (full.astype(np.float32) * scale.astype(np.float32)).astype(np.float16)


# ---------------------------------------------------------------------------
# Image preprocessing for SigLIP
# ---------------------------------------------------------------------------

_SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_SIGLIP_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess_siglip(jpg_bytes: bytes, image_size: int = 384, tj=None) -> np.ndarray:
    """
    Decode and resize to 384×384, normalise to SigLIP expected range.
    Pass a TurboJPEG instance via `tj` to reuse across calls.
    Returns float32 [1, 3, 384, 384].
    """
    if tj is not None:
        from turbojpeg import TJPF_RGB
        img = tj.decode(jpg_bytes, pixel_format=TJPF_RGB)
    else:
        try:
            from turbojpeg import TurboJPEG, TJPF_RGB
            img = TurboJPEG().decode(jpg_bytes, pixel_format=TJPF_RGB)
        except ImportError:
            from PIL import Image as PilImage
            img = np.array(
                PilImage.open(io.BytesIO(jpg_bytes)).convert("RGB"), dtype=np.uint8
            )

    # Resize to 384×384
    if img.shape[0] != image_size or img.shape[1] != image_size:
        from PIL import Image as PilImage
        img = np.array(
            PilImage.fromarray(img).resize((image_size, image_size), PilImage.BICUBIC),
            dtype=np.uint8
        )

    # Normalise: uint8 → float32, then (x/255 - mean) / std
    img_f = img.astype(np.float32) / 255.0
    img_f = (img_f - _SIGLIP_MEAN) / _SIGLIP_STD
    # HWC → NCHW
    return img_f.transpose(2, 0, 1)[np.newaxis]  # [1, 3, 384, 384]


# ---------------------------------------------------------------------------
# Shard iteration
# ---------------------------------------------------------------------------

def iter_shard(shard_path: str):
    """Yield (id, jpg_bytes) from a .tar shard."""
    try:
        with tarfile.open(shard_path) as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            keys = {}
            for name in members:
                stem, _, ext = name.rpartition(".")
                if stem not in keys:
                    keys[stem] = {}
                keys[stem][ext.lower()] = name

            for stem, exts in keys.items():
                jpg_key = exts.get("jpg") or exts.get("jpeg") or exts.get("png")
                if not jpg_key:
                    continue
                jpg = tar.extractfile(members[jpg_key]).read()
                yield stem, jpg
    except Exception as e:
        print(f"Warning: {shard_path}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Per-worker model state (populated once via Pool initializer)
# ---------------------------------------------------------------------------

_W: dict = {}


def _worker_init() -> None:
    """Load SigLIP model once per worker process."""
    global _W
    try:
        import mlx.core as mx  # noqa: F401
        try:
            from mlx_vlm import load as vlm_load
            model, _processor = vlm_load("google/siglip-so400m-patch14-384")
            model.eval()
            _W["model_state"] = (True, model)
        except Exception:
            from transformers import AutoModel
            import torch
            hf_model = AutoModel.from_pretrained(
                "google/siglip-so400m-patch14-384"
            ).vision_model.eval()
            _W["model_state"] = (False, hf_model)
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Run: pip install mlx-vlm transformers", file=sys.stderr)
        raise
    try:
        from turbojpeg import TurboJPEG
        _W["tj"] = TurboJPEG()
    except ImportError:
        _W["tj"] = None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _encode_batch_siglip(model_state, rec_ids, imgs_np, output_dir):
    """
    Encode a batch of pre-processed images through SigLIP in one GPU call.
    model_state: (_use_mlx_vlm, model_or_hf_model)
    Falls back to single-image on error.
    Returns number of successfully saved files.
    """
    import mlx.core as mx
    _use_mlx_vlm, model_obj = model_state
    try:
        stacked = np.concatenate(imgs_np, axis=0)  # [B, 3, 384, 384]
        if _use_mlx_vlm:
            feats = model_obj.vision_model(mx.array(stacked))  # [B, 729, 1152]
            mx.eval(feats)
            feats_np = np.array(feats)  # [B, 729, 1152]
        else:
            import torch
            with torch.no_grad():
                out = model_obj(pixel_values=torch.from_numpy(stacked))
            feats_np = out.last_hidden_state.float().numpy()  # [B, 729, 1152]
        for k, rec_id in enumerate(rec_ids):
            q_packed, scale = quantize_4bit(feats_np[k].astype(np.float32))
            np.savez(os.path.join(output_dir, f"{rec_id}.npz"), q=q_packed, scale=scale)
        return len(rec_ids)
    except Exception as e:
        print(f"  Batch encode failed ({e}), retrying single-image", file=sys.stderr)
        saved = 0
        for rec_id, img_np in zip(rec_ids, imgs_np):
            try:
                if _use_mlx_vlm:
                    feats = model_obj.vision_model(mx.array(img_np))  # [1, 729, 1152]
                    mx.eval(feats)
                    feat_np = np.array(feats[0])
                else:
                    import torch
                    with torch.no_grad():
                        out = model_obj(pixel_values=torch.from_numpy(img_np))
                    feat_np = out.last_hidden_state[0].float().numpy()
                q_packed, scale = quantize_4bit(feat_np.astype(np.float32))
                np.savez(os.path.join(output_dir, f"{rec_id}.npz"), q=q_packed, scale=scale)
                saved += 1
            except Exception as e2:
                print(f"  Skipping {rec_id}: {e2}", file=sys.stderr)
        return saved


def process_shard(args) -> dict:
    """
    Worker: encode all images in one shard with SigLIP and save quantised features.

    Two-phase design within each shard:
      Phase 1: sequential tar read → list of (rec_id, jpg_bytes)
      Phase 2: 1-ahead prefetch — while GPU encodes batch N, P-core threads
               decode+preprocess batch N+1 in parallel, hiding CPU latency.
    """
    shard_path, output_dir, batch_size = args
    os.makedirs(output_dir, exist_ok=True)

    model_state = _W["model_state"]
    tj = _W["tj"]

    # Phase 1: sequential tar read — collect pending (id, bytes) pairs
    written = 0
    raw_items = []
    for rec_id, jpg_bytes in iter_shard(shard_path):
        if os.path.exists(os.path.join(output_dir, f"{rec_id}.npz")):
            written += 1
        else:
            raw_items.append((rec_id, jpg_bytes))

    if not raw_items:
        return {"shard": shard_path, "written": written, "error": False}

    # Split into batches upfront
    batches = [raw_items[i:i + batch_size] for i in range(0, len(raw_items), batch_size)]

    def _preprocess_batch(items):
        """Decode+preprocess a list of (rec_id, jpg_bytes) in parallel."""
        def _one(item):
            rec_id, jpg_bytes = item
            try:
                return rec_id, preprocess_siglip(jpg_bytes, tj=tj)
            except Exception as e:
                print(f"  Skipping {rec_id}: {e}", file=sys.stderr)
                return rec_id, None
        n = min(_PERF_CORES, len(items))
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(_one, items))

    # Phase 2: 1-ahead prefetch — decode next batch while GPU encodes current
    prefetch_q: deque = deque()
    with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
        if batches:
            prefetch_q.append(prefetch_pool.submit(_preprocess_batch, batches[0]))

        for batch_idx in range(len(batches)):
            preprocessed = prefetch_q.popleft().result()

            # Submit next batch decode immediately (runs during GPU encode below)
            next_idx = batch_idx + 1
            if next_idx < len(batches):
                prefetch_q.append(prefetch_pool.submit(_preprocess_batch, batches[next_idx]))

            batch_ids = [r for r, img in preprocessed if img is not None]
            batch_imgs = [img for r, img in preprocessed if img is not None]
            if batch_imgs:
                written += _encode_batch_siglip(model_state, batch_ids, batch_imgs, output_dir)

    return {"shard": shard_path, "written": written, "error": False}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute 4-bit quantised SigLIP features"
    )
    parser.add_argument(
        "--shards", required=True,
        help="Directory containing .tar shards"
    )
    parser.add_argument(
        "--output", default="data/siglip_q4",
        help="Output directory for .npz files (default: data/siglip_q4)"
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Parallel processes (default 2; GPU shared so >2 contends)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Images per SigLIP forward pass (default 8)"
    )
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(args.shards, "*.tar")))
    if not shards:
        print(f"No .tar files in {args.shards}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Storage estimate: 1.55M × (729×576 + 729×2) bytes ≈ 420 GB
    bytes_per = 729 * 576 + 729 * 2
    total_gb = len(shards) * 5000 * bytes_per / 1e9

    print(f"Pre-computing SigLIP features for {len(shards)} shards")
    print(f"  Output:     {args.output}")
    print(f"  Workers:    {args.workers}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Storage estimate: ~{total_gb:.0f} GB")
    print(f"  NOTE: Only pre-compute if {total_gb:.0f} GB free after Qwen3+VAE storage.")
    print()

    work_items = [(s, args.output, args.batch_size) for s in shards]

    import time as _time
    results = []
    t_start = _time.time()
    t_last_hb = t_start
    interval_rates = []
    with multiprocessing.Pool(
        processes=args.workers,
        initializer=_worker_init,
    ) as pool:
        for done, result in enumerate(
            pool.imap_unordered(process_shard, work_items, chunksize=1), 1
        ):
            results.append(result)
            written_so_far = sum(r["written"] for r in results)
            errs_so_far = sum(1 for r in results if r["error"])
            t_now = _time.time()
            interval_time = t_now - t_last_hb
            if interval_time > 0:
                interval_rates.append(1.0 / interval_time)
            avg_rate = sum(interval_rates) / len(interval_rates) if interval_rates else 0
            eta = (len(work_items) - done) / avg_rate if avg_rate > 0 else 0
            t_last_hb = t_now
            err_str = f"  errors={errs_so_far}" if errs_so_far else ""
            print(
                f"  [{done}/{len(work_items)}] {written_so_far:,} features"
                f"{err_str}  {1/avg_rate:.1f} s/shard  ETA {eta/60:.0f}m",
                flush=True,
            )

    total = sum(r["written"] for r in results)
    errors = sum(1 for r in results if r["error"])
    print(f"\nDone. {total:,} features saved to {args.output}/")
    if errors:
        print(f"  {errors} shards had errors (check stderr)")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
