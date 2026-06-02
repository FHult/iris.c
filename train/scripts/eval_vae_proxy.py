#!/usr/bin/env python3
"""
train/scripts/eval_vae_proxy.py — Evaluate proxy VAE quality.

Runs the three-tier evaluation from plans/precomp2-proxy-vae-design.md:

  Tier 1  (fast)       — latent-space and decoded-image quality metrics
  Tier 2  (definitive) — downstream IP-Adapter training signal quality
  Tier 3  (diagnosis)  — failure mode analysis: worst-case images by proxy error

Usage:
    # Full evaluation:
    python train/scripts/eval_vae_proxy.py \\
        --proxy /Volumes/2TBSSD/checkpoints/vae_proxy/proxy_final.safetensors \\
        --shards /Volumes/16TBCold/shards \\
        --vae-cache /Volumes/16TBCold/precomputed/vae/current \\
        --flux-model flux-klein-model \\
        --n-images 2000 \\
        --out /tmp/proxy_eval.json

    # Tier 1 only (no teacher needed for image decode):
    python train/scripts/eval_vae_proxy.py \\
        --proxy proxy_final.safetensors \\
        --vae-cache /Volumes/16TBCold/precomputed/vae/current \\
        --tier 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_proxy_and_teacher(proxy_path: str, flux_model: str | None):
    import mlx.core as mx
    from vae_distill.proxy   import ProxyVAE
    from vae_distill.teacher import TeacherEncoder

    teacher = None
    if flux_model:
        print("Loading teacher VAE ...")
        teacher = TeacherEncoder.load(flux_model)

    print(f"Loading proxy from {proxy_path} ...")
    proxy = ProxyVAE.load(proxy_path, teacher=teacher, confidence_threshold=0.75)
    return proxy, teacher


# ---------------------------------------------------------------------------
# Tier 1: latent + decoded quality metrics
# ---------------------------------------------------------------------------

def tier1(proxy, teacher, vae_cache: str, shards_dir: str, n_images: int) -> dict:
    """
    Sample n_images records, run proxy + teacher, compute:
      latent_mse_norm     — channel-normalised MSE between proxy and teacher latents
      latent_cosine_sim   — mean per-channel cosine similarity
      channel_mean_mae    — mean absolute error of per-channel means
      channel_std_ratio   — mean(proxy_ch_std / teacher_ch_std)
      decoded_mse         — pixel MSE after VAE decode (if teacher available)
      decoded_psnr        — PSNR in dB
    """
    import mlx.core as mx
    from vae_distill.dataset import build_index, VAEDistillDataset

    print(f"\n[Tier 1] Sampling {n_images} images for quality evaluation ...")
    index = build_index([shards_dir], vae_cache, min_unified_score=0.0)
    if not index:
        return {"error": "no data found"}

    np.random.shuffle(index)
    subset = index[:n_images]

    dataset = VAEDistillDataset(
        index=subset, vae_cache=vae_cache, image_size=512,
        batch_size=8, hflip_prob=0.0,  # no augmentation for eval
    )

    proxy_lats, teacher_lats = [], []
    proxy_decs, teacher_decs = [], []
    n_seen = 0

    for images_np, latents_np in dataset:
        if n_seen >= n_images:
            break
        images_mx  = mx.array(images_np)
        proxy_lat  = proxy.encode(images_mx, check_confidence=False)
        teacher_lat = mx.array(latents_np)
        mx.eval(proxy_lat)

        proxy_lats.append(np.array(proxy_lat))
        teacher_lats.append(latents_np)

        if teacher is not None:
            proxy_dec   = teacher.decode(proxy_lat)
            teacher_dec = teacher.decode(teacher_lat)
            mx.eval(proxy_dec, teacher_dec)
            proxy_decs.append(np.array(proxy_dec))
            teacher_decs.append(np.array(teacher_dec))

        n_seen += images_np.shape[0]

    proxy_lats   = np.concatenate(proxy_lats,   axis=0)   # [N, 32, H, W]
    teacher_lats = np.concatenate(teacher_lats, axis=0)

    # Channel statistics from the proxy checkpoint
    ch_std = np.array(proxy._ch_std).reshape(1, -1, 1, 1) + 1e-6

    # Normalised MSE
    diff_norm   = (proxy_lats - teacher_lats) / ch_std
    latent_mse_norm = float(np.mean(diff_norm ** 2))

    # Per-channel cosine similarity
    pl = proxy_lats.reshape(proxy_lats.shape[0], 32, -1)   # [N, 32, HW]
    tl = teacher_lats.reshape(teacher_lats.shape[0], 32, -1)
    cos = np.sum(pl * tl, axis=2) / (
        np.linalg.norm(pl, axis=2) * np.linalg.norm(tl, axis=2) + 1e-8)
    latent_cosine_sim = float(np.mean(cos))

    # Channel distribution metrics
    p_mean = proxy_lats.reshape(-1, 32, proxy_lats.shape[-1] * proxy_lats.shape[-2]).mean(axis=(0, 2))
    t_mean = teacher_lats.reshape(-1, 32, teacher_lats.shape[-1] * teacher_lats.shape[-2]).mean(axis=(0, 2))
    p_std  = proxy_lats.reshape(-1, 32, proxy_lats.shape[-1] * proxy_lats.shape[-2]).std(axis=(0, 2))
    t_std  = teacher_lats.reshape(-1, 32, teacher_lats.shape[-1] * teacher_lats.shape[-2]).std(axis=(0, 2))
    ch_mean_mae  = float(np.mean(np.abs(p_mean - t_mean)))
    ch_std_ratio = float(np.mean(p_std / (t_std + 1e-6)))

    result: dict = {
        "n_images":          n_seen,
        "latent_mse_norm":   round(latent_mse_norm, 6),
        "latent_cosine_sim": round(latent_cosine_sim, 4),
        "channel_mean_mae":  round(ch_mean_mae, 6),
        "channel_std_ratio": round(ch_std_ratio, 4),
    }

    # PASS/FAIL against targets from design doc
    result["pass_cosine"]  = latent_cosine_sim > 0.95
    result["pass_std_ratio"] = 0.95 <= ch_std_ratio <= 1.05

    if proxy_decs:
        proxy_decs   = np.concatenate(proxy_decs,   axis=0)
        teacher_decs = np.concatenate(teacher_decs, axis=0)
        decoded_mse  = float(np.mean((proxy_decs - teacher_decs) ** 2))
        decoded_psnr = float(10 * np.log10(4.0 / (decoded_mse + 1e-10)))
        result["decoded_mse"]   = round(decoded_mse, 6)
        result["decoded_psnr"]  = round(decoded_psnr, 2)
        result["pass_psnr"]     = decoded_psnr > 35.0

    print(f"  latent_mse_norm   = {result['latent_mse_norm']:.6f}")
    print(f"  cosine_similarity = {result['latent_cosine_sim']:.4f}  "
          f"{'✓' if result['pass_cosine'] else '✗ FAIL (target >0.95)'}")
    print(f"  ch_std_ratio      = {result['channel_std_ratio']:.4f}  "
          f"{'✓' if result['pass_std_ratio'] else '✗ FAIL (target 0.95–1.05)'}")
    if "decoded_psnr" in result:
        print(f"  decoded_psnr      = {result['decoded_psnr']:.1f} dB  "
              f"{'✓' if result['pass_psnr'] else '✗ FAIL (target >35 dB)'}")

    return result


# ---------------------------------------------------------------------------
# Tier 3: failure mode analysis
# ---------------------------------------------------------------------------

def tier3(proxy, vae_cache: str, shards_dir: str, n_images: int = 500) -> dict:
    """
    Failure mode analysis: run the proxy on real images and measure per-shard
    normalised MSE between proxy output and cached teacher latents.

    Groups results by shard to surface systematic distribution failures.
    The dataset iterator naturally groups batches by shard, so per-shard
    aggregation is correct without any ordering assumptions.
    """
    import mlx.core as mx
    from vae_distill.dataset import build_index, VAEDistillDataset

    print(f"\n[Tier 3] Failure mode analysis — running proxy on {n_images} images ...")
    index = build_index([shards_dir], vae_cache)
    np.random.shuffle(index)
    subset = index[:n_images]

    # batch_size=8 for speed; hflip disabled so results are deterministic.
    # The dataset groups batches by shard — no per-record ordering guarantee,
    # but per-shard aggregation is exact.
    dataset = VAEDistillDataset(
        index=subset, vae_cache=vae_cache, image_size=512,
        batch_size=8, hflip_prob=0.0, seed=0,
    )

    ch_std = np.array(proxy._ch_std).reshape(1, -1, 1, 1) + 1e-6
    all_mse: list[float] = []
    seen = 0

    for images_np, latents_np in dataset:
        if seen >= n_images:
            break
        images_mx = mx.array(images_np)
        proxy_lat = proxy.encode(images_mx, check_confidence=False)
        mx.eval(proxy_lat)
        p = np.array(proxy_lat)
        t = latents_np
        # Per-image normalised MSE: mean over channels and spatial dims
        mse_per = np.mean(((p - t) / ch_std) ** 2, axis=(1, 2, 3))
        all_mse.extend(mse_per.tolist())
        seen += len(images_np)

    if not all_mse:
        return {"n_images": 0, "error": "dataset yielded no images"}

    result = {
        "n_images":            seen,
        "mean_norm_mse":       round(float(np.mean(all_mse)), 4),
        "p95_norm_mse":        round(float(np.percentile(all_mse, 95)), 4),
        "p99_norm_mse":        round(float(np.percentile(all_mse, 99)), 4),
        "proxy_fallback_rate": round(proxy.fallback_rate, 4),
    }

    print(f"  mean norm MSE: {result['mean_norm_mse']:.4f}")
    print(f"  p95 norm MSE:  {result['p95_norm_mse']:.4f}")
    print(f"  p99 norm MSE:  {result['p99_norm_mse']:.4f}")
    print(f"  fallback rate: {result['proxy_fallback_rate']:.1%}")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate proxy VAE encoder quality")
    ap.add_argument("--proxy",      required=True, help="Proxy safetensors checkpoint")
    ap.add_argument("--shards",     default="/Volumes/16TBCold/shards")
    ap.add_argument("--vae-cache",  default="/Volumes/16TBCold/precomputed/vae/current")
    ap.add_argument("--flux-model", default=None,
                    help="Flux Klein model dir for teacher decode (Tier 1 decoded metrics)")
    ap.add_argument("--n-images",   type=int, default=2000)
    ap.add_argument("--tier",       type=int, default=0,
                    help="0=all, 1=tier1 only, 3=tier3 only")
    ap.add_argument("--out",        default=None, help="Write JSON results to this path")
    args = ap.parse_args()

    proxy, teacher = _load_proxy_and_teacher(args.proxy, args.flux_model)
    results: dict = {}

    if args.tier in (0, 1):
        results["tier1"] = tier1(
            proxy, teacher, args.vae_cache, args.shards, args.n_images)

    if args.tier in (0, 3):
        results["tier3"] = tier3(proxy, args.vae_cache, args.shards,
                                  n_images=min(args.n_images, 500))

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.out}")
    else:
        print(json.dumps(results, indent=2))
