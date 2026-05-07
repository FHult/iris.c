"""
train/tests/test_dataset.py — Unit tests for dataset utilities.

Tests:
  - _select_bucket: aspect ratio matching
  - _normalize: range, dtype, shape
  - augment_mlx: output shape and dtype
  - _iter_shard_contents: parsing synthetic shard dicts
  - _load_qwen3_embed: 4-bit unpack round-trip
  - _load_vae_latent: int8 dequant round-trip, shape mismatch rejection
  - _load_siglip_embed: 4-bit unpack round-trip
  - make_prefetch_loader: yields correct batch structure with synthetic shards
"""

import io
import os
import sys
import tarfile
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ip_adapter.dataset import (
    BUCKETS,
    _select_bucket,
    _normalize,
    augment_mlx,
    _iter_shard_contents,
    _load_qwen3_embed,
    _load_vae_latent,
    _load_siglip_embed,
    make_prefetch_loader,
)


# ---------------------------------------------------------------------------
# _select_bucket
# ---------------------------------------------------------------------------

class TestSelectBucket:
    def test_square_image_selects_square_bucket(self):
        bH, bW = _select_bucket(512, 512)
        assert bH == 512 and bW == 512

    def test_wide_landscape_selects_wide_bucket(self):
        # 896×512 aspect ≈ 1.75 → (512, 896) bucket (W=896, H=512)
        bH, bW = _select_bucket(896, 512)
        assert bW > bH  # landscape bucket

    def test_tall_portrait_selects_tall_bucket(self):
        # 512×896 aspect ≈ 0.57 → (896, 512) bucket (H=896, W=512)
        bH, bW = _select_bucket(512, 896)
        assert bH > bW  # portrait bucket

    def test_zero_dimension_returns_first_bucket(self):
        bH, bW = _select_bucket(0, 512)
        assert (bH, bW) == BUCKETS[0]
        bH, bW = _select_bucket(512, 0)
        assert (bH, bW) == BUCKETS[0]

    def test_known_aspect_ratio_mappings(self):
        """Verify specific aspect ratios map to expected buckets."""
        # 3:2 landscape → (512, 768) bucket
        assert _select_bucket(768, 512) == (512, 768)
        # 2:3 portrait → (768, 512) bucket
        assert _select_bucket(512, 768) == (768, 512)
        # wide 7:4 → (512, 896) bucket
        assert _select_bucket(896, 512) == (512, 896)
        # tall 4:7 → (896, 512) bucket
        assert _select_bucket(512, 896) == (896, 512)

    def test_640_bucket_note(self):
        """The (640,640) bucket ties with (512,512) for square images.
        Both have W/H ratio = 1.0; min() picks (512,512) as it appears first
        in the BUCKETS list. This is expected: (640,640) is used as an override
        via the bucket= parameter directly, not through aspect-ratio selection."""
        # Square images always route to (512, 512), not (640, 640)
        assert _select_bucket(640, 640) == (512, 512)
        assert _select_bucket(512, 512) == (512, 512)

    def test_return_type_is_tuple(self):
        result = _select_bucket(512, 512)
        assert isinstance(result, tuple) and len(result) == 2

    def test_minimum_bucket_size_is_512(self):
        """No bucket smaller than 512 on either axis (protects patchification)."""
        for bH, bW in BUCKETS:
            assert bH >= 512 and bW >= 512


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_range_minus_one_to_one(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = _normalize(img)
        assert float(result.min()) == pytest.approx(-1.0, abs=1e-5)

        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = _normalize(img)
        assert float(result.max()) == pytest.approx(1.0, abs=1e-3)

    def test_midpoint_near_zero(self):
        img = np.full((4, 4, 3), 127, dtype=np.uint8)
        result = _normalize(img)
        assert abs(float(result.mean())) < 0.01

    def test_output_dtype_float32(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        result = _normalize(img)
        assert result.dtype == np.float32

    def test_hwc_to_chw_transpose(self):
        img = np.zeros((32, 48, 3), dtype=np.uint8)  # H=32, W=48, C=3
        result = _normalize(img)
        assert result.shape == (3, 32, 48)  # CHW

    def test_random_image_stays_in_range(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        result = _normalize(img)
        assert result.min() >= -1.0 - 1e-5
        assert result.max() <=  1.0 + 1e-5


# ---------------------------------------------------------------------------
# augment_mlx
# ---------------------------------------------------------------------------

class TestAugmentMlx:
    def test_output_shape_4d(self):
        import mlx.core as mx
        B, C, H, W = 2, 3, 544, 544  # bucket 512 + 32 pad
        img = mx.zeros((B, C, H, W))
        result = augment_mlx(img, bucket_h=512, bucket_w=512)
        mx.eval(result)
        assert result.shape == (B, C, 512, 512)

    def test_output_shape_3d(self):
        import mlx.core as mx
        C, H, W = 3, 544, 544
        img = mx.zeros((C, H, W))
        result = augment_mlx(img, bucket_h=512, bucket_w=512)
        mx.eval(result)
        assert result.shape == (C, 512, 512)

    def test_output_shape_non_square_bucket(self):
        import mlx.core as mx
        B, C = 2, 3
        bH, bW = 768, 512
        img = mx.zeros((B, C, bH + 32, bW + 32))
        result = augment_mlx(img, bucket_h=bH, bucket_w=bW)
        mx.eval(result)
        assert result.shape == (B, C, bH, bW)

    def test_values_are_subset_of_input(self):
        """Crop must not produce values outside the padded input."""
        import mlx.core as mx
        rng = np.random.default_rng(5)
        arr = rng.standard_normal((1, 3, 544, 544)).astype(np.float32)
        img = mx.array(arr)
        result = augment_mlx(img, bucket_h=512, bucket_w=512)
        mx.eval(result)
        # After optional flip, values still come from the original array
        # We can verify the range matches
        assert float(mx.min(result).item()) >= arr.min() - 1e-5
        assert float(mx.max(result).item()) <= arr.max() + 1e-5


# ---------------------------------------------------------------------------
# _iter_shard_contents
# ---------------------------------------------------------------------------

class TestIterShardContents:
    def _make_contents(self, records):
        """Build {filename: bytes} dict like a decompressed tar shard."""
        contents = {}
        for stem, jpg, txt in records:
            contents[f"{stem}.jpg"] = jpg
            contents[f"{stem}.txt"] = txt.encode()
        return contents

    def test_basic_parsing(self):
        contents = self._make_contents([
            ("000001_0001", b"\xff\xd8\xff" + b"\x00" * 100, "a red cat"),
            ("000001_0002", b"\xff\xd8\xff" + b"\x00" * 100, "a blue dog"),
        ])
        records = _iter_shard_contents(contents)
        assert len(records) == 2
        ids = {r["id"] for r in records}
        assert ids == {"000001_0001", "000001_0002"}

    def test_caption_decoded_correctly(self):
        contents = self._make_contents([
            ("key1", b"\xff\xd8\xff" + b"\x00" * 50, "hello world"),
        ])
        records = _iter_shard_contents(contents)
        assert records[0]["txt"] == "hello world"

    def test_missing_jpg_skipped(self):
        contents = {
            "key1.txt": b"caption",
            # no key1.jpg
        }
        records = _iter_shard_contents(contents)
        assert len(records) == 0

    def test_missing_txt_skipped(self):
        contents = {
            "key1.jpg": b"\xff\xd8\xff" + b"\x00" * 50,
            # no key1.txt
        }
        records = _iter_shard_contents(contents)
        assert len(records) == 0

    def test_png_extension_accepted(self):
        contents = {
            "key1.png": b"\x89PNG" + b"\x00" * 50,
            "key1.txt": b"a picture",
        }
        records = _iter_shard_contents(contents)
        assert len(records) == 1
        assert records[0]["jpg"] == b"\x89PNG" + b"\x00" * 50

    def test_record_has_required_keys(self):
        contents = self._make_contents([("id0", b"\xff\xd8" * 30, "test")])
        records = _iter_shard_contents(contents)
        assert set(records[0].keys()) == {"id", "jpg", "txt"}


# ---------------------------------------------------------------------------
# _load_qwen3_embed (4-bit nibble unpack round-trip)
# ---------------------------------------------------------------------------

class TestLoadQwen3Embed:
    def _save_q4(self, tmpdir, rec_id, shape_seq, shape_d):
        """Save a synthetic 4-bit packed Qwen3 embed NPZ file."""
        # shape_d must be even (packed 2 nibbles per byte)
        half_d = shape_d // 2
        # All nibble values = 2 → q byte = 0x22
        q = np.full((shape_seq, half_d), 0x22, dtype=np.int8)
        scale = np.ones((shape_seq, 1), dtype=np.float32)
        path = os.path.join(tmpdir, f"{rec_id}.npz")
        np.savez(path, q=q, scale=scale)
        return path

    def test_round_trip_values(self, tmp_path):
        tmpdir = str(tmp_path)
        self._save_q4(tmpdir, "test_id", shape_seq=4, shape_d=8)
        result = _load_qwen3_embed("test_id", tmpdir)
        assert result is not None
        # Nibble value 2 * scale 1.0 = 2.0
        assert result.dtype == np.float16
        assert result.shape == (4, 8)
        assert np.allclose(result, 2.0, atol=0.01)

    def test_correct_shape_7680(self, tmp_path):
        tmpdir = str(tmp_path)
        self._save_q4(tmpdir, "emb_a", shape_seq=16, shape_d=7680)
        result = _load_qwen3_embed("emb_a", tmpdir)
        assert result is not None
        assert result.shape == (16, 7680)
        assert result.dtype == np.float16

    def test_missing_file_returns_none(self, tmp_path):
        result = _load_qwen3_embed("nonexistent_id", str(tmp_path))
        assert result is None

    def test_none_dir_returns_none(self):
        result = _load_qwen3_embed("any_id", None)
        assert result is None


# ---------------------------------------------------------------------------
# _load_vae_latent (int8 dequant round-trip)
# ---------------------------------------------------------------------------

class TestLoadVaeLatent:
    def _save_int8(self, tmpdir, rec_id, channels=32, lh=64, lw=64):
        """Save a synthetic int8 VAE latent NPZ."""
        q = np.full((channels, lh, lw), 10, dtype=np.int8)
        scale = np.float32(0.5)
        path = os.path.join(tmpdir, f"{rec_id}.npz")
        np.savez(path, q=q, scale=scale)

    def test_round_trip_values(self, tmp_path):
        tmpdir = str(tmp_path)
        self._save_int8(tmpdir, "lat_0", channels=32, lh=64, lw=64)
        result = _load_vae_latent("lat_0", tmpdir)
        assert result is not None
        assert result.dtype == np.float16
        assert result.shape == (32, 64, 64)
        # 10 * 0.5 = 5.0
        assert np.allclose(result, 5.0, atol=0.01)

    def test_shape_mismatch_returns_none(self, tmp_path):
        tmpdir = str(tmp_path)
        self._save_int8(tmpdir, "lat_mismatch", channels=32, lh=64, lw=64)
        # Expect (32, 32) but stored is (64, 64)
        result = _load_vae_latent("lat_mismatch", tmpdir, expected_hw=(32, 32))
        assert result is None

    def test_shape_match_accepted(self, tmp_path):
        tmpdir = str(tmp_path)
        self._save_int8(tmpdir, "lat_match", channels=32, lh=64, lw=64)
        result = _load_vae_latent("lat_match", tmpdir, expected_hw=(64, 64))
        assert result is not None
        assert result.shape == (32, 64, 64)

    def test_missing_file_returns_none(self, tmp_path):
        result = _load_vae_latent("no_such_id", str(tmp_path))
        assert result is None

    def test_none_dir_returns_none(self):
        result = _load_vae_latent("any", None)
        assert result is None


# ---------------------------------------------------------------------------
# _load_siglip_embed (4-bit nibble unpack — same format as qwen3)
# ---------------------------------------------------------------------------

class TestLoadSiglipEmbed:
    def _save_q4(self, tmpdir, rec_id, seq=729, dim=1152):
        half_d = dim // 2
        q = np.full((seq, half_d), 0x22, dtype=np.int8)  # nibble=2 (lo=2, hi=2)
        scale = np.ones((seq, 1), dtype=np.float32)
        path = os.path.join(tmpdir, f"{rec_id}.npz")
        np.savez(path, q=q, scale=scale)

    def test_round_trip_shape_and_values(self, tmp_path):
        tmpdir = str(tmp_path)
        self._save_q4(tmpdir, "sig_0")
        result = _load_siglip_embed("sig_0", tmpdir)
        assert result is not None
        assert result.shape == (729, 1152)
        assert result.dtype == np.float16
        assert np.allclose(result, 2.0, atol=0.01)

    def test_missing_file_returns_none(self, tmp_path):
        result = _load_siglip_embed("nope", str(tmp_path))
        assert result is None

    def test_none_dir_returns_none(self):
        result = _load_siglip_embed("x", None)
        assert result is None


# ---------------------------------------------------------------------------
# make_prefetch_loader (integration: synthetic shards)
# ---------------------------------------------------------------------------

def _make_synthetic_shard(path: str, n_records: int = 5):
    """Write a minimal WebDataset tar shard for testing."""
    from PIL import Image as PILImage
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(n_records):
            stem = f"rec_{i:04d}"
            # Minimal 64×64 JPEG
            img_arr = np.zeros((64, 64, 3), dtype=np.uint8)
            img_arr[:, :, 0] = 128  # red channel
            pil = PILImage.fromarray(img_arr, "RGB")
            jpg_buf = io.BytesIO()
            pil.save(jpg_buf, format="JPEG", quality=80)
            jpg_bytes = jpg_buf.getvalue()

            for ext, data in [(".jpg", jpg_bytes), (".txt", b"a test caption")]:
                info = tarfile.TarInfo(name=stem + ext)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    buf.seek(0)
    with open(path, "wb") as f:
        f.write(buf.read())


class TestMakePrefetchLoader:
    def test_yields_6_tuple(self, tmp_path):
        shard = str(tmp_path / "shard_000.tar")
        _make_synthetic_shard(shard, n_records=10)

        loader = make_prefetch_loader([shard], batch_size=2, bucket=(512, 512))
        batch = next(iter(loader))
        assert len(batch) == 6

    def test_image_batch_shape(self, tmp_path):
        shard = str(tmp_path / "shard_000.tar")
        _make_synthetic_shard(shard, n_records=10)

        bH, bW = 512, 512
        loader = make_prefetch_loader([shard], batch_size=2, bucket=(bH, bW))
        images, captions, text_embs, vae_lats, siglip, bucket_hw = next(iter(loader))

        assert images.shape == (2, 3, bH + 32, bW + 32)
        assert images.dtype == np.float32

    def test_captions_list_of_str(self, tmp_path):
        shard = str(tmp_path / "shard_000.tar")
        _make_synthetic_shard(shard, n_records=10)

        loader = make_prefetch_loader([shard], batch_size=2, bucket=(512, 512))
        _, captions, *_ = next(iter(loader))

        assert isinstance(captions, list)
        assert len(captions) == 2
        assert all(isinstance(c, str) for c in captions)

    def test_bucket_hw_matches_requested(self, tmp_path):
        shard = str(tmp_path / "shard_000.tar")
        _make_synthetic_shard(shard, n_records=10)

        loader = make_prefetch_loader([shard], batch_size=2, bucket=(640, 640))
        *_, bucket_hw = next(iter(loader))
        assert bucket_hw == (640, 640)

    def test_no_precomputed_caches_returns_none(self, tmp_path):
        shard = str(tmp_path / "shard_000.tar")
        _make_synthetic_shard(shard, n_records=10)

        loader = make_prefetch_loader([shard], batch_size=2, bucket=(512, 512))
        _, _, text_embs, vae_lats, siglip, _ = next(iter(loader))
        assert text_embs is None
        assert vae_lats is None
        assert siglip is None

    def test_image_values_in_range(self, tmp_path):
        """Images normalised to [-1, 1]."""
        shard = str(tmp_path / "shard_000.tar")
        _make_synthetic_shard(shard, n_records=10)

        loader = make_prefetch_loader([shard], batch_size=2, bucket=(512, 512))
        images, *_ = next(iter(loader))
        assert images.min() >= -1.0 - 1e-4
        assert images.max() <=  1.0 + 1e-4
