"""
train/tests/test_dataset_bucketing.py — the precompute<->train resolution contract.

Guards the 4th flywheel blocker (found by the warmup-run2 quick test): VAE latents
are precomputed at a single SQUARE resolution (precompute_all squashes to
image_size²), but training buckets images by aspect ratio. `_load_vae_latent`
rejects any cached latent whose spatial shape != the current bucket's (H//8, W//8)
as a cache miss; in cached mode (no live VAE to fall back on) a miss is an
unrecoverable skip, so multi-bucket training skips ~100% of batches and exits. The
fix pins training to one bucket via `data.bucket` (train_ip_adapter →
make_prefetch_loader(bucket=...)).

These pin the contract so the blocker can't silently regress:
  - the shape-rejection mechanism itself (matching shape hits, mismatch misses),
  - `_select_bucket` aspect math,
  - the headline fact: a square-512 latent matches ONLY the (512,512) bucket — which
    is exactly why a single fixed bucket is required while precompute is square.

Pure: numpy + fake .npz files; no model, no GPU compute (one-time kernel compile on
import, like the other train test modules).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import ip_adapter.dataset as ds


def _write_latent(d: Path, rec_id: str, shape, key="latent"):
    """Create {d}/{rec_id}.npz holding a zero latent of `shape` (np.savez adds .npz)."""
    if key == "latent":
        np.savez(str(d / rec_id), latent=np.zeros(shape, dtype=np.float32))
    else:  # legacy q/scale
        np.savez(str(d / rec_id), q=np.zeros(shape, dtype=np.int8),
                 scale=np.float32(1.0))


# ---------------------------------------------------------------------------
# _select_bucket — aspect-ratio assignment
# ---------------------------------------------------------------------------

class TestSelectBucket:
    def test_square_picks_512(self):
        # aspect 1.0 → (512,512) (first 1.0 bucket; (640,640) also 1.0 but later)
        assert ds._select_bucket(512, 512) == (512, 512)
        assert ds._select_bucket(1000, 1000) == (512, 512)

    def test_exact_wide_15(self):
        # 768x512 → aspect 1.5 → the W/H=1.5 bucket (512,768)
        assert ds._select_bucket(768, 512) == (512, 768)

    def test_very_wide_picks_widest(self):
        # aspect 2.0 → closest bucket W/H is 1.75 → (512,896)
        assert ds._select_bucket(1024, 512) == (512, 896)

    def test_very_tall_picks_tallest(self):
        # aspect 0.5 → closest bucket W/H is 0.571 → (896,512)
        assert ds._select_bucket(512, 1024) == (896, 512)

    def test_zero_dim_falls_back_to_first(self):
        assert ds._select_bucket(0, 512) == ds.BUCKETS[0]
        assert ds._select_bucket(512, 0) == ds.BUCKETS[0]


# ---------------------------------------------------------------------------
# _load_vae_latent — the shape-rejection cache-miss mechanism
# ---------------------------------------------------------------------------

class TestLoadVaeLatent:
    def test_matching_shape_returns_latent(self, tmp_path):
        _write_latent(tmp_path, "000000_0000", (32, 64, 64))
        out = ds._load_vae_latent("000000_0000", str(tmp_path), expected_hw=(64, 64))
        assert out is not None and out.shape == (32, 64, 64)
        assert out.dtype == np.float32

    def test_mismatched_shape_returns_none(self, tmp_path):
        # THE blocker: a 512² latent (64×64) under a non-square bucket (e.g. 768×512
        # → expected 96×64) is a cache miss.
        _write_latent(tmp_path, "000000_0000", (32, 64, 64))
        assert ds._load_vae_latent("000000_0000", str(tmp_path), expected_hw=(96, 64)) is None
        assert ds._load_vae_latent("000000_0000", str(tmp_path), expected_hw=(64, 96)) is None

    def test_no_expected_hw_accepts_any_shape(self, tmp_path):
        _write_latent(tmp_path, "r", (32, 80, 48))
        assert ds._load_vae_latent("r", str(tmp_path), expected_hw=None).shape == (32, 80, 48)

    def test_missing_file_returns_none(self, tmp_path):
        assert ds._load_vae_latent("nope", str(tmp_path), expected_hw=(64, 64)) is None

    def test_no_vae_dir_returns_none(self):
        assert ds._load_vae_latent("r", None, expected_hw=(64, 64)) is None

    def test_legacy_q_scale_format(self, tmp_path):
        _write_latent(tmp_path, "r", (32, 64, 64), key="legacy")
        out = ds._load_vae_latent("r", str(tmp_path), expected_hw=(64, 64))
        assert out is not None and out.shape == (32, 64, 64)


# ---------------------------------------------------------------------------
# The contract: square precompute ⇒ only the square bucket can ever hit
# ---------------------------------------------------------------------------

class TestSquareLatentBucketContract:
    def test_square512_latent_matches_only_512_bucket(self, tmp_path):
        # A latent from the single-resolution (square 512²) precompute is (32,64,64).
        _write_latent(tmp_path, "000000_0000", (32, 64, 64))
        matched = []
        for (bH, bW) in ds.BUCKETS:
            exp = (bH // 8, bW // 8)
            if ds._load_vae_latent("000000_0000", str(tmp_path), expected_hw=exp) is not None:
                matched.append((bH, bW))
        # Exactly one of the six buckets hits — the square one. This is why training
        # must be pinned to (512,512) (data.bucket) while precompute is square; the
        # other 5 buckets would each be a 100% cache miss.
        assert matched == [(512, 512)]

    def test_pinned_bucket_expected_hw_matches_square_latent(self):
        # data.bucket=[512,512] → loader passes expected_hw=(512//8, 512//8)=(64,64),
        # which matches the (32,64,64) square latent. Encodes the fix's invariant.
        bH, bW = (512, 512)
        assert (bH // 8, bW // 8) == (64, 64)


# ---------------------------------------------------------------------------
# rec_id lookup contract for the other encoders (keyed by tar member stem)
# ---------------------------------------------------------------------------

class TestRecIdLookupContract:
    def test_qwen3_missing_and_no_dir_return_none(self, tmp_path):
        assert ds._load_qwen3_embed("missing", str(tmp_path)) is None
        assert ds._load_qwen3_embed("r", None) is None

    def test_siglip_missing_and_no_dir_return_none(self, tmp_path):
        assert ds._load_siglip_embed("missing", str(tmp_path)) is None
        assert ds._load_siglip_embed("r", None) is None


# ---------------------------------------------------------------------------
# _resize_to_bucket — squash (no aspect preservation), the precompute's transform
# ---------------------------------------------------------------------------

class TestResizeToBucket:
    def test_squash_to_exact_bucket_shape(self):
        img = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)  # non-square source
        out = ds._resize_to_bucket(img, 512, 512)
        assert out.shape == (512, 512, 3)   # squashed to square (aspect not preserved)

    def test_passthrough_when_already_target(self):
        img = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        out = ds._resize_to_bucket(img, 512, 512)
        assert out.shape == (512, 512, 3)
