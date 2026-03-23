"""
train/scripts/precompute_vae.py — Pre-compute and int8-quantise VAE latents.

Saves ~180ms per training step = ~6.0 hours over Stage 1 (105K steps).
Storage: ~198 GB at int8 (vs ~1.6 TB at float32).

Run ONCE after build_shards.py + filter_shards.py complete.
Takes ~6 hours on M1 Max (1.55M images × ~14ms encode).

Output: one .npz file per sample under data/vae_int8/{id}.npz
  - q:     int8 [32, H/8, W/8]   — per-channel absmax quantised latent
  - scale: float16 [32, 1, 1]    — per-channel absmax / 127

At training time, load with load_vae_latent() (see bottom of this file).
Dequantization in the prefetch thread is CPU-trivial.

Quantisation format:
  int8 per channel, scale = abs(arr).max(axis=(1,2)) / 127.0

CPU allocation:
  VAE encode is GPU-bound; outer shard loop runs 2 parallel processes.
  Each process initialises its own MLX Metal context.

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
# Quantisation helpers
# ---------------------------------------------------------------------------

def quantize_int8(arr: np.ndarray):
    """
    Per-channel absmax int8 quantisation.
    arr: float32 [32, H, W] (VAE latent channels)
    Returns:
      q:     int8 [32, H, W]
      scale: float16 [32, 1, 1]
    """
    scale = np.abs(arr).max(axis=(1, 2), keepdims=True) / 127.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return q, scale.astype(np.float16)


def load_vae_latent(npz_path: str) -> np.ndarray:
    """
    Dequantise a saved VAE latent.
    Returns float16 [32, H/8, W/8].
    Called from the training prefetch thread.
    """
    d = np.load(npz_path)
    return (d["q"].astype(np.float32) * d["scale"].astype(np.float32)).astype(np.float16)


# ---------------------------------------------------------------------------
# Image preprocessing for VAE
# ---------------------------------------------------------------------------

def preprocess_vae(jpg_bytes: bytes, image_size: int = 512, tj=None) -> np.ndarray:
    """
    Decode JPEG and prepare for VAE encoder.
    Returns float32 [1, 3, H, W] in [-1, 1].
    Pass a TurboJPEG instance via `tj` to reuse across calls (avoids per-image init cost).
    """
    if tj is not None:
        from turbojpeg import TJPF_RGB
        img = tj.decode(jpg_bytes, pixel_format=TJPF_RGB)  # HWC uint8 RGB
    else:
        try:
            from turbojpeg import TurboJPEG, TJPF_RGB
            img = TurboJPEG().decode(jpg_bytes, pixel_format=TJPF_RGB)
        except ImportError:
            from PIL import Image as PilImage
            img = np.array(
                PilImage.open(io.BytesIO(jpg_bytes)).convert("RGB")
                   .resize((image_size, image_size), PilImage.LANCZOS),
                dtype=np.uint8
            )

    if img.shape[0] != image_size or img.shape[1] != image_size:
        from PIL import Image as PilImage
        img = np.array(
            PilImage.fromarray(img).resize((image_size, image_size), PilImage.LANCZOS),
            dtype=np.uint8
        )

    # HWC → CHW, uint8 → float32 in [-1, 1]
    img_f = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
    return img_f[np.newaxis]  # [1, 3, H, W]


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
# Batch encode helper
# ---------------------------------------------------------------------------

def _encode_batch(vae, rec_ids: list, imgs_np: list, output_dir: str) -> int:
    """
    Encode a list of pre-processed images through the VAE in one batched GPU call.
    Falls back to single-image mode if the batch encode raises (e.g. OOM).
    Returns the number of successfully saved images.
    """
    import mlx.core as mx
    try:
        stacked = np.concatenate(imgs_np, axis=0)       # [B, 3, H, W]
        latents = vae.encode(mx.array(stacked))          # [B, 32, H/8, W/8]
        mx.eval(latents)
        latents_np = np.array(latents)                   # [B, 32, H/8, W/8]
        for k, rec_id in enumerate(rec_ids):
            q, scale = quantize_int8(latents_np[k].astype(np.float32))
            np.savez(os.path.join(output_dir, f"{rec_id}.npz"), q=q, scale=scale)
        return len(rec_ids)
    except Exception as e:
        print(f"  Batch encode failed ({e}), retrying single-image", file=sys.stderr)
        saved = 0
        for rec_id, img_np in zip(rec_ids, imgs_np):
            try:
                latent = vae.encode(mx.array(img_np))
                mx.eval(latent)
                q, scale = quantize_int8(np.array(latent[0]).astype(np.float32))
                np.savez(os.path.join(output_dir, f"{rec_id}.npz"), q=q, scale=scale)
                saved += 1
            except Exception as e2:
                print(f"  Skipping {rec_id}: {e2}", file=sys.stderr)
        return saved


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def process_shard(args) -> dict:
    """
    Worker: encode all images in one shard through VAE and save quantised latents.
    Images are processed in batches (batch_size) for better GPU utilisation.
    """
    shard_path, output_dir, model_path, image_size, batch_size = args
    os.makedirs(output_dir, exist_ok=True)

    try:
        import mlx.core as mx
        from mflux.models.flux2 import Flux2Klein
    except ImportError:
        print("mflux not available. Run: pip install mflux", file=sys.stderr)
        return {"shard": shard_path, "written": 0, "error": True}

    try:
        flux = Flux2Klein(model_path=model_path, quantize=None)
        vae = flux.vae
        vae.freeze()
    except Exception as e:
        print(f"Failed to load VAE from {model_path}: {e}", file=sys.stderr)
        return {"shard": shard_path, "written": 0, "error": True}

    try:
        from turbojpeg import TurboJPEG
        tj = TurboJPEG()
    except ImportError:
        tj = None

    written = 0
    batch_ids: list = []
    batch_imgs: list = []

    for rec_id, jpg_bytes in iter_shard(shard_path):
        out_path = os.path.join(output_dir, f"{rec_id}.npz")
        if os.path.exists(out_path):
            written += 1
            continue

        try:
            batch_imgs.append(preprocess_vae(jpg_bytes, image_size, tj=tj))
            batch_ids.append(rec_id)
        except Exception as e:
            print(f"  Skipping {rec_id}: {e}", file=sys.stderr)

        if len(batch_imgs) >= batch_size:
            written += _encode_batch(vae, batch_ids, batch_imgs, output_dir)
            batch_ids.clear()
            batch_imgs.clear()

    if batch_imgs:
        written += _encode_batch(vae, batch_ids, batch_imgs, output_dir)

    return {"shard": shard_path, "written": written, "error": False}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute int8 quantised VAE latents"
    )
    parser.add_argument(
        "--shards", required=True,
        help="Directory containing .tar shards"
    )
    parser.add_argument(
        "--output", default="data/vae_int8",
        help="Output directory for .npz files (default: data/vae_int8)"
    )
    parser.add_argument(
        "--model", default="flux-klein-4b",
        help="Model ID or path (default: flux-klein-4b)"
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Image size fed to VAE (default 512)"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel processes (default 1; GPU-bound — 2+ processes contend for Metal)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Images per VAE encode call (default 8; batching amortises GPU launch overhead)"
    )
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(args.shards, "*.tar")))
    if not shards:
        print(f"No .tar files in {args.shards}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print(f"Pre-computing VAE latents for {len(shards)} shards")
    print(f"  Model:   {args.model}")
    print(f"  Output:  {args.output}")
    print(f"  Workers:    {args.workers}")
    print(f"  Batch size: {args.batch_size}")
    latent_h = args.image_size // 8
    bytes_per = 32 * latent_h * latent_h  # int8
    total_gb = len(shards) * 5000 * bytes_per / 1e9
    print(f"  Storage estimate: ~{total_gb:.0f} GB")
    print()

    work_items = [(s, args.output, args.model, args.image_size, args.batch_size) for s in shards]

    import time as _time
    results = []
    t_start = _time.time()
    t_last_hb = t_start
    interval_rates = []
    with multiprocessing.Pool(processes=args.workers) as pool:
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
                f"  [{done}/{len(work_items)}] {written_so_far:,} latents"
                f"{err_str}  {avg_rate:.2f} shards/s  ETA {eta/60:.0f}m",
                flush=True,
            )

    total = sum(r["written"] for r in results)
    errors = sum(1 for r in results if r["error"])
    print(f"\nDone. {total:,} latents saved to {args.output}/")
    if errors:
        print(f"  {errors} shards had errors (check stderr)")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
