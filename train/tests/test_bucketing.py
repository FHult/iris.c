"""
train/tests/test_bucketing.py — the canonical aspect-bucket primitive (PRECOMP-4).

The single source of truth precompute and the loader will share. These pin its
behaviour AND assert it agrees byte-for-byte with the existing dataset._select_bucket
(so lifting that logic here later is provably behaviour-preserving). Pure.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from ip_adapter import bucketing as bk


class TestAspectBucket:
    def test_square_picks_512(self):
        assert bk.aspect_bucket(512, 512) == (512, 512)
        assert bk.aspect_bucket(1000, 1000) == (512, 512)   # tie → first square

    def test_exact_wide_15(self):
        assert bk.aspect_bucket(768, 512) == (512, 768)     # W/H = 1.5

    def test_very_wide(self):
        assert bk.aspect_bucket(1024, 512) == (512, 896)    # closest W/H = 1.75

    def test_very_tall(self):
        assert bk.aspect_bucket(512, 1024) == (896, 512)    # closest W/H = 0.571

    def test_degenerate_dims_first_bucket(self):
        assert bk.aspect_bucket(0, 512) == bk.DEFAULT_BUCKETS[0]
        assert bk.aspect_bucket(512, 0) == bk.DEFAULT_BUCKETS[0]
        assert bk.aspect_bucket(-1, 5) == bk.DEFAULT_BUCKETS[0]

    def test_custom_bucket_set(self):
        buckets = [(256, 256), (256, 512)]
        assert bk.aspect_bucket(1000, 500, buckets) == (256, 512)
        assert bk.aspect_bucket(500, 500, buckets) == (256, 256)


class TestBucketLatentHw:
    def test_square(self):
        assert bk.bucket_latent_hw((512, 512)) == (64, 64)

    def test_non_square(self):
        assert bk.bucket_latent_hw((768, 512)) == (96, 64)
        assert bk.bucket_latent_hw((512, 896)) == (64, 112)

    def test_custom_downscale(self):
        assert bk.bucket_latent_hw((512, 512), vae_downscale=16) == (32, 32)


class TestAgreesWithDatasetSelectBucket:
    """The canonical function must match the existing loader's _select_bucket exactly,
    so the future lift is behaviour-preserving and precompute/loader never disagree."""

    def _load_dataset(self):
        spec = importlib.util.spec_from_file_location(
            "ip_adapter.dataset",
            Path(__file__).parent.parent / "ip_adapter" / "dataset.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_matches_across_aspects(self):
        ds = self._load_dataset()
        # same bucket set in both
        assert list(bk.DEFAULT_BUCKETS) == list(ds.BUCKETS)
        for (w, h) in [(512, 512), (1000, 1000), (768, 512), (1024, 512),
                       (512, 1024), (1920, 1080), (1080, 1920), (640, 640),
                       (300, 400), (400, 300), (1, 1)]:
            assert bk.aspect_bucket(w, h) == ds._select_bucket(w, h), (w, h)
