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


def preprocess_siglip(jpg_bytes: bytes, image_size: int = 384) -> np.ndarray:
    """
    Decode and resize to 384×384, normalise to SigLIP expected range.
    Returns float32 [1, 3, 384, 384].
    """
    try:
        from turbojpeg import TurboJPEG
        tj = TurboJPEG()
        img = tj.decode(jpg_bytes)  # HWC uint8 RGB
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
# Worker
# ---------------------------------------------------------------------------

def process_shard(args) -> dict:
    """
    Worker: encode all images in one shard with SigLIP and save quantised features.
    """
    shard_path, output_dir = args
    os.makedirs(output_dir, exist_ok=True)

    try:
        import mlx.core as mx
        # Load SigLIP via mlx_vlm or transformers
        # SigLIP SO400M: google/siglip-so400m-patch14-384
        try:
            from mlx_vlm import load as vlm_load
            model, processor = vlm_load("google/siglip-so400m-patch14-384")
            model.eval()
            _use_mlx_vlm = True
        except Exception:
            # Fallback: load via transformers + convert to MLX manually
            _use_mlx_vlm = False
            from transformers import AutoModel, AutoProcessor
            import torch
            hf_model = AutoModel.from_pretrained(
                "google/siglip-so400m-patch14-384"
            ).vision_model.eval()
            processor = AutoProcessor.from_pretrained(
                "google/siglip-so400m-patch14-384"
            )
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Run: pip install mlx-vlm transformers", file=sys.stderr)
        return {"shard": shard_path, "written": 0, "error": True}

    written = 0
    for rec_id, jpg_bytes in iter_shard(shard_path):
        out_path = os.path.join(output_dir, f"{rec_id}.npz")
        if os.path.exists(out_path):
            written += 1
            continue

        try:
            img_np = preprocess_siglip(jpg_bytes)  # [1, 3, 384, 384]

            if _use_mlx_vlm:
                img_mx = mx.array(img_np)
                feats = model.vision_model(img_mx)  # [1, 729, 1152]
                mx.eval(feats)
                feats_np = np.array(feats[0])  # [729, 1152]
            else:
                import torch
                img_t = torch.from_numpy(img_np)
                with torch.no_grad():
                    out = hf_model(pixel_values=img_t)
                feats_np = out.last_hidden_state[0].float().numpy()  # [729, 1152]

            q_packed, scale = quantize_4bit(feats_np.astype(np.float32))
            np.savez_compressed(out_path, q=q_packed, scale=scale)
            written += 1

        except Exception as e:
            print(f"  Skipping {rec_id}: {e}", file=sys.stderr)

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
    print(f"  Output:  {args.output}")
    print(f"  Workers: {args.workers}")
    print(f"  Storage estimate: ~{total_gb:.0f} GB")
    print(f"  NOTE: Only pre-compute if {total_gb:.0f} GB free after Qwen3+VAE storage.")
    print()

    work_items = [(s, args.output) for s in shards]

    with multiprocessing.Pool(processes=args.workers) as pool:
        results = pool.map(process_shard, work_items)

    total = sum(r["written"] for r in results)
    errors = sum(1 for r in results if r["error"])
    print(f"\nDone. {total:,} features saved to {args.output}/")
    if errors:
        print(f"  {errors} shards had errors (check stderr)")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
