# PRECOMP-4 — Aspect-ratio bucketing, end-to-end (implementation plan)

Status: design. Targets the M5 Max quality stage; the GPU-free pieces are
front-loadable on the M1 Max. Supersedes the BACKLOG PRECOMP-4 sketch with a
concrete approach + build order.

## Problem (recap)
Today the data path squashes every image to a single **square** resolution
(`precompute_all._resize` → `(image_size, image_size)`) and the loader does not
honour aspect ratio (`make_prefetch_loader(bucket=None)` picks `rng.choice(BUCKETS)`
per shard). So the multi-resolution machinery is non-functional: `_load_vae_latent`
rejects any latent whose shape ≠ the bucket's `(H//8, W//8)` as a cache miss, and in
cached mode a miss is an unrecoverable skip. The shipped workaround pins training to
`data.bucket: [512,512]` — correct but geometrically distorted (square squash) and
single-aspect. PRECOMP-4 removes the distortion.

## Goal / non-goals
- **Goal:** distortion-free, multi-aspect training — images encoded at an aspect that
  preserves their proportions, training batches homogeneous in shape, output not
  square-biased.
- **Non-goal (this doc):** changing the model/objective; high-res (1024px) is a
  *parameter* of this (the bucket set), not a separate effort — VAE tiling (PRECOMP-1)
  already handles the large-bucket encode.

## Core decision: per-image-bucket latents (NOT re-shard)

Two ways to make multi-aspect work:

| | **A: per-image-bucket** (recommended) | **B: aspect-sorted shards** |
|---|---|---|
| precompute | encode each image at *its* aspect bucket; latent shape varies per image, keyed by `rec_id` | one bucket per shard; all its images share a shape |
| sharding | **unchanged** | **re-shard the whole 1280-pool by aspect** |
| knowledge base | **preserved** — shard IDs, `shard_scores.db`, the encoder-identity precompute cache all keyed on existing shard stems | **disrupted** — new shard IDs invalidate scores + cache; needs a migration/mapping |
| loader | minimal: leverage the *existing* shape-rejection | per-shard-single-bucket becomes correct as-is |
| batch efficiency | read-and-skip: for a chosen bucket B, images not in B return `None` and are skipped (~5/6 I/O waste if uniform) | efficient — a shard *is* one bucket |

**Recommendation: A (per-image-bucket).** The decisive factor is **knowledge-base
preservation** — the M1 Max bootstrap's shard scores, coverage, and (PRECOMP-3
content-addressed) precompute cache must carry forward to the M5 Max untouched.
Re-sharding (B) throws those away or needs a fragile remap. A also exploits an
elegant property: **the existing `_load_vae_latent(expected_hw=B)` shape check already
filters images to bucket B for free** — store each image's latent at its own bucket
shape, and a B-batch naturally loads only the B-aspect images (others return `None`).
So A needs *less* new loader logic, not more.

Revisit B (or a hybrid that physically groups same-aspect records within a shard at
build time) only if the read-and-skip I/O proves to be the throughput limiter on the
M5 Max. Note: the read-skip cost is mostly **metadata** (npz that don't match are not
fully read), and SSD-resident hot staging makes it cheap.

## End-to-end design (Approach A)

1. **Bucket assignment** — one canonical function `aspect_bucket(w, h, buckets)`
   (today's `dataset._select_bucket`, lifted to a shared, pure, tested module so
   precompute and the loader agree exactly). Aspect-closest bucket; tie → first.
2. **Precompute (multi-bucket mode, opt-in)** — for each image: pick its bucket,
   resize **aspect-aware** to that bucket (replace the square `_resize`; resize to the
   bucket H×W — still a resize, but to the correct aspect, so no squash), VAE-encode at
   that resolution (single-pass ≤512², `_encode_vae_tiled` for larger), store
   `{rec_id}.npz` at the bucket's latent shape. SigLIP/qwen3 unchanged in key; SigLIP
   resizes the **same** bucket-shaped image to 384² (conditioning stays geometrically
   consistent with the target — see PRECOMP-4 SigLIP note).
3. **Cache versioning** — multi-bucket latents are a *different encoder output* than
   the single-512² ones, so they must land in a distinct version dir. Add the bucket
   regime to the VAE `encoder_config_subset` (e.g. `"buckets": sorted(BUCKETS)` instead
   of a scalar `image_size`, + bump `ENCODER_CODE_VERSION["vae"]`). This keeps the two
   regimes' caches separate and non-colliding (no silent shape mismatch).
4. **Loader (multi-bucket mode, opt-in, default off)** — per batch: choose bucket B
   (round-robin or aspect-frequency-weighted), iterate records, load latent with
   `expected_hw=B`; accumulate only the records that match (others skipped via the
   existing `None` path), resize their images to B for conditioning. Falls back to
   today's fixed-`data.bucket` path when multi-bucket is off.
5. **Teacher + proxy** — re-precompute uses the **teacher VAE** as ground truth, the
   distilled **proxy (PRECOMP-2)** for speed (confidence-gated, teacher fallback;
   teacher authoritative for golden/validation). **Retrain the proxy** on multi-bucket
   pairs (the current proxy only saw square 512²).
6. **Refine the adapter** — warm-start from the 512² bootstrap weights, fine-tune on
   the multi-bucket cache, drop the `data.bucket` pin.

## Component changes (files)
- `train/ip_adapter/dataset.py` — lift `_select_bucket` to the shared module; add the
  multi-bucket loader mode (default off); replace square resize on the precompute side
  with aspect resize (precompute uses `precompute_all`, loader uses `_resize_to_bucket`
  which already takes H,W — it's the *precompute* `_resize` that squashes).
- `train/scripts/precompute_all.py` — `_preprocess_vae` aspect-resize; per-image bucket
  assignment; multi-bucket output; `image_size>512` already tiles.
- `train/scripts/cache_manager.py` — VAE `encoder_config_subset` carries the bucket
  regime; bump `ENCODER_CODE_VERSION["vae"]`.
- `train/vae_distill/` — retrain proxy on multi-bucket pairs.
- new `train/scripts/aspect_bucket.py` (or `train/ip_adapter/bucketing.py`) — the one
  canonical, pure, tested `aspect_bucket()` + `bucket_latent_hw()` both sides import.

## Build order
**GPU-free (front-loadable on M1 Max, testable now):**
1. Canonical `aspect_bucket` module + tests (the shared primitive). *Pure.*
2. Cache-versioning change for the bucket regime + tests. *Pure (hash math).*
3. Precompute multi-bucket **assignment + output-naming + version-key** path (the CPU
   scaffolding; the GPU encode reuses existing `_encode`/`_encode_vae_tiled`). Default
   off. Unit-test the assignment + naming; the encode runs on the M5 Max.
4. Loader multi-bucket mode behind a flag (default off). Test the bucket-selection +
   record-filtering logic with fake npz (reuses the `test_dataset_bucketing` pattern).
   **Risk note:** the loader is imported by live training — keep the new path strictly
   behind a default-off flag so iter N+1 is unaffected.

**GPU-bound (M5 Max — execution + validation):**
5. Run the multi-bucket re-precompute (teacher + retrained proxy).
6. Refine the adapter; validate with `quality_gate.py` (golden-set clip_i/lpips/fid)
   that multi-aspect actually beats the 512² baseline before adopting it.

## Validation
The win is *claimed* (less distortion) but must be *measured*: run the refined
multi-bucket adapter through `quality_gate.py` against the 512² bootstrap champion on
the fixed golden set. Adopt only on IMPROVED/NEUTRAL; a REGRESSION means the
read-skip/efficiency or a bucketing bug outweighs the distortion win.

## Open questions / risks
- **Read-skip I/O** at scale (Approach A) — measure on M5 Max; fall back to within-shard
  aspect grouping at build time if it bottlenecks.
- **Bucket set** — start with today's 6; add 1024px buckets (the commented-out ones)
  only after the M5 Max memory profile (TRAIN-7) confirms headroom.
- **Mixed-bucket batches** — batches must stay homogeneous in shape for the Metal graph
  cache; the loader must never mix buckets within a batch.
- **Proxy staleness** — the proxy must be retrained per regime; gate its use behind the
  confidence/teacher-fallback that PRECOMP-2 already provides.
