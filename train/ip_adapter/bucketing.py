"""
train/ip_adapter/bucketing.py — canonical aspect-ratio bucketing.

THE single source of truth for "which resolution does an image belong to," shared
by precompute (which encoder resolution to encode each image at) and the training
loader (which bucket to assign / which latent shape to expect). Keeping ONE
definition is what prevents the precompute↔train shape mismatch that pinned
training to 512² (PRECOMP-4): if the two sides ever compute buckets differently,
every cross-aspect latent is a cache miss.

Pure (no deps) so both `train/scripts/precompute_all.py` and
`train/ip_adapter/dataset.py` can import it, and it's trivially testable.
"""

from __future__ import annotations

from typing import List, Tuple

Bucket = Tuple[int, int]  # (H, W)

# (H, W) training buckets. 512px set. The 1024px buckets are added only after the
# M5 Max memory profile (TRAIN-7) confirms headroom — bucket set is a scale param.
DEFAULT_BUCKETS: List[Bucket] = [
    (512, 512),
    (512, 768),
    (768, 512),
    (640, 640),
    (512, 896),
    (896, 512),
]


def aspect_bucket(w: int, h: int, buckets: List[Bucket] = DEFAULT_BUCKETS) -> Bucket:
    """Return the bucket (H, W) whose aspect (W/H) is closest to the source (w/h).

    - Degenerate source (w or h <= 0) → the first bucket (a safe square default).
    - Ties resolve to the first matching bucket, so a square source picks (512,512)
      rather than (640,640). This matches dataset._select_bucket exactly — both must
      agree, which is the whole point of this module.
    """
    if w <= 0 or h <= 0:
        return buckets[0]
    aspect = w / h
    return min(buckets, key=lambda b: abs(b[1] / b[0] - aspect))


def bucket_latent_hw(bucket: Bucket, vae_downscale: int = 8) -> Tuple[int, int]:
    """VAE latent spatial shape (H//ds, W//ds) for a bucket.

    This is exactly the `expected_hw` the loader checks a cached latent against
    (dataset._load_vae_latent). Encoding an image at `bucket` produces a latent of
    this shape; the loader assigning that image to `bucket` expects this shape, so
    they match — that's how per-image-bucket precompute makes multi-aspect work.
    """
    bH, bW = bucket
    return (bH // vae_downscale, bW // vae_downscale)
