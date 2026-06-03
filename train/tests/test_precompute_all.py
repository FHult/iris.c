"""
train/tests/test_precompute_all.py — precompute building blocks (GROK-TEST-6).

Covers the pure helpers that implement the "read each shard once + skip records
already done" contract — the logic that makes per-iter precompute incremental
and that the trainer's shard-cache filter depends on downstream:

  _scan_existing   — the O(1) "already computed" set (skip vs recompute)
  iter_shard       — tar record pairing (jpg+txt required; png accepted)
  _save_npz_atomic — crash-safe writes (no torn .npz mistaken for done)

Pure: synthetic in-memory tars + tempdirs. No GPU, no encoders, no real data.
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))
import precompute_all as pa


# ---------------------------------------------------------------------------
# Synthetic shard tar builder
# ---------------------------------------------------------------------------

def _make_tar(path: Path, records: dict) -> None:
    """records: {stem: {"img": ("jpg"|"png"|None), "txt": str|None}}.

    Writes a tar with the requested members. img value is the extension to use
    (or None to omit the image); txt is the caption text (or None to omit).
    """
    with tarfile.open(path, "w") as tar:
        for stem, spec in records.items():
            ext = spec.get("img")
            if ext is not None:
                data = b"\xff\xd8fakejpeg" if ext in ("jpg", "jpeg") else b"\x89PNGfake"
                _add(tar, f"{stem}.{ext}", data)
            txt = spec.get("txt")
            if txt is not None:
                _add(tar, f"{stem}.txt", txt.encode())


def _add(tar, name, data):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


# ---------------------------------------------------------------------------
# _scan_existing  — the "already done" set
# ---------------------------------------------------------------------------

class TestScanExisting:
    def test_missing_dir_is_empty(self, tmp_path):
        assert pa._scan_existing(str(tmp_path / "nope")) == set()

    def test_none_dir_is_empty(self):
        assert pa._scan_existing(None) == set()

    def test_empty_dir_is_empty(self, tmp_path):
        (tmp_path / "vae").mkdir()
        assert pa._scan_existing(str(tmp_path / "vae")) == set()

    def test_returns_npz_stems(self, tmp_path):
        d = tmp_path / "vae"
        d.mkdir()
        for n in ("000000_0000", "000000_0001", "000042_0049"):
            (d / f"{n}.npz").touch()
        assert pa._scan_existing(str(d)) == {"000000_0000", "000000_0001", "000042_0049"}

    def test_ignores_non_npz(self, tmp_path):
        d = tmp_path / "vae"
        d.mkdir()
        (d / "000000_0000.npz").touch()
        (d / "000000_0001.txt").touch()       # not an npz
        (d / "manifest.json").touch()
        assert pa._scan_existing(str(d)) == {"000000_0000"}


# ---------------------------------------------------------------------------
# iter_shard  — record pairing contract
# ---------------------------------------------------------------------------

class TestIterShard:
    def test_complete_records_yielded(self, tmp_path):
        tar = tmp_path / "000000.tar"
        _make_tar(tar, {
            "000000_0000": {"img": "jpg", "txt": "a cat"},
            "000000_0001": {"img": "jpg", "txt": "a dog"},
        })
        got = {stem: caption for stem, _jpg, caption in pa.iter_shard(str(tar))}
        assert got == {"000000_0000": "a cat", "000000_0001": "a dog"}

    def test_missing_txt_skipped(self, tmp_path):
        tar = tmp_path / "000000.tar"
        _make_tar(tar, {
            "000000_0000": {"img": "jpg", "txt": "ok"},
            "000000_0001": {"img": "jpg", "txt": None},     # image only → skipped
        })
        stems = [s for s, _, _ in pa.iter_shard(str(tar))]
        assert stems == ["000000_0000"]

    def test_missing_image_skipped(self, tmp_path):
        tar = tmp_path / "000000.tar"
        _make_tar(tar, {
            "000000_0000": {"img": None, "txt": "caption only"},  # → skipped
            "000000_0001": {"img": "jpg", "txt": "ok"},
        })
        stems = [s for s, _, _ in pa.iter_shard(str(tar))]
        assert stems == ["000000_0001"]

    def test_png_image_accepted(self, tmp_path):
        tar = tmp_path / "000000.tar"
        _make_tar(tar, {"000000_0000": {"img": "png", "txt": "p"}})
        stems = [s for s, _, _ in pa.iter_shard(str(tar))]
        assert stems == ["000000_0000"]

    def test_caption_whitespace_stripped(self, tmp_path):
        tar = tmp_path / "000000.tar"
        _make_tar(tar, {"000000_0000": {"img": "jpg", "txt": "  spaced  \n"}})
        _, _, caption = next(iter(pa.iter_shard(str(tar))))
        assert caption == "spaced"

    def test_corrupt_tar_warns_not_raises(self, tmp_path, capsys):
        bad = tmp_path / "000000.tar"
        bad.write_bytes(b"not a tar file at all")
        # iter_shard catches the exception, warns, and yields nothing.
        assert list(pa.iter_shard(str(bad))) == []


# ---------------------------------------------------------------------------
# _save_npz_atomic  + round-trip into _scan_existing (the done→skip cycle)
# ---------------------------------------------------------------------------

class TestSaveNpzAtomic:
    def test_writes_final_file_no_tmp_leftover(self, tmp_path):
        out = tmp_path / "000000_0000.npz"
        pa._save_npz_atomic(str(out), latent=np.zeros((2, 2), dtype=np.float32))
        assert out.exists()
        # No transient .tmp.npz left behind after a successful write.
        leftovers = list(tmp_path.glob("*.tmp.npz"))
        assert leftovers == []

    def test_roundtrip_values(self, tmp_path):
        out = tmp_path / "000000_0000.npz"
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        pa._save_npz_atomic(str(out), latent=arr)
        loaded = np.load(str(out))["latent"]
        assert np.array_equal(loaded, arr)

    def test_saved_file_is_seen_by_scan_existing(self, tmp_path):
        # The done→skip cycle: a freshly saved record must be reported as present
        # so the next precompute pass skips it.
        d = tmp_path / "vae"
        d.mkdir()
        pa._save_npz_atomic(str(d / "000123_0007.npz"),
                            latent=np.zeros((1,), dtype=np.float32))
        assert "000123_0007" in pa._scan_existing(str(d))
