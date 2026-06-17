"""
train/tests/test_scripts.py — Tests for data processing scripts.

Tests:
  - convert_journeydb: tgz range filtering, caption key priority order,
    annotation loading from synthetic JSONL
  - clip_dedup: device detection logic
"""

import importlib.util
import io
import json
import os
import sys
import tarfile
import tempfile

import numpy as np
import pytest


def _load_script(name: str):
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    path = os.path.join(scripts_dir, f"{name}.py")
    if not os.path.exists(path):
        # V1 scripts were archived to scripts/v1/ — their logic tests still run
        # against the archive so a resurrection starts from a green baseline.
        path = os.path.join(scripts_dir, "v1", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# convert_journeydb — annotation loading
# ---------------------------------------------------------------------------

class TestConvertJourneydbAnnotations:
    def _make_anno_tgz(self, tmpdir: str, lines: list) -> str:
        """Write a synthetic train_anno_realease_repath.jsonl.tgz."""
        jsonl_bytes = b"\n".join(json.dumps(l).encode() for l in lines)
        tgz_path = os.path.join(tmpdir, "train_anno_realease_repath.jsonl.tgz")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            info = tarfile.TarInfo("train_anno.jsonl")
            info.size = len(jsonl_bytes)
            tf.addfile(info, io.BytesIO(jsonl_bytes))
        buf.seek(0)
        with open(tgz_path, "wb") as f:
            f.write(buf.read())
        return tgz_path

    def test_loads_prompt_key(self, tmp_path):
        mod = _load_script("convert_journeydb")
        lines = [
            {"img_path": "imgs/abc.jpg", "prompt": "a golden retriever"},
        ]
        tgz = self._make_anno_tgz(str(tmp_path), lines)
        captions = mod.load_annotations(tgz)
        assert "abc" in captions
        assert captions["abc"] == "a golden retriever"

    def test_loads_task2_caption_preferred_over_prompt(self, tmp_path):
        """Task2.Caption should take priority over raw prompt."""
        mod = _load_script("convert_journeydb")
        lines = [
            {
                "img_path": "imgs/xyz.jpg",
                "prompt": "raw diffusion prompt here",
                "Task2": {"Caption": "clean natural language description"},
            }
        ]
        tgz = self._make_anno_tgz(str(tmp_path), lines)
        captions = mod.load_annotations(tgz)
        assert captions["xyz"] == "clean natural language description"

    def test_caption_key_also_accepted(self, tmp_path):
        mod = _load_script("convert_journeydb")
        lines = [
            {"img_path": "imgs/img1.jpg", "caption": "explicit caption field"},
        ]
        tgz = self._make_anno_tgz(str(tmp_path), lines)
        captions = mod.load_annotations(tgz)
        assert captions["img1"] == "explicit caption field"

    def test_caption_priority_order(self, tmp_path):
        """caption > Task2.Caption > prompt > Prompt."""
        mod = _load_script("convert_journeydb")
        lines = [
            {
                "img_path": "imgs/p1.jpg",
                "caption": "highest priority",
                "Task2": {"Caption": "second"},
                "prompt": "third",
                "Prompt": "fourth",
            }
        ]
        tgz = self._make_anno_tgz(str(tmp_path), lines)
        captions = mod.load_annotations(tgz)
        assert captions["p1"] == "highest priority"

    def test_skips_missing_img_path(self, tmp_path):
        mod = _load_script("convert_journeydb")
        lines = [
            {"caption": "no image path here"},
            {"img_path": "imgs/good.jpg", "prompt": "has path"},
        ]
        tgz = self._make_anno_tgz(str(tmp_path), lines)
        captions = mod.load_annotations(tgz)
        assert len(captions) == 1
        assert "good" in captions

    def test_skips_empty_caption(self, tmp_path):
        mod = _load_script("convert_journeydb")
        lines = [
            {"img_path": "imgs/empty.jpg", "prompt": ""},
            {"img_path": "imgs/ok.jpg", "prompt": "real caption"},
        ]
        tgz = self._make_anno_tgz(str(tmp_path), lines)
        captions = mod.load_annotations(tgz)
        assert "empty" not in captions
        assert "ok" in captions

    def test_skips_invalid_json(self, tmp_path):
        """Corrupt lines should not crash — just skip."""
        mod = _load_script("convert_journeydb")
        jsonl_bytes = (
            b'{"img_path": "imgs/ok.jpg", "prompt": "valid"}\n'
            b'not valid json at all\n'
            b'{"img_path": "imgs/ok2.jpg", "prompt": "also valid"}\n'
        )
        tgz_path = os.path.join(str(tmp_path), "train_anno_realease_repath.jsonl.tgz")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            info = tarfile.TarInfo("train_anno.jsonl")
            info.size = len(jsonl_bytes)
            tf.addfile(info, io.BytesIO(jsonl_bytes))
        buf.seek(0)
        with open(tgz_path, "wb") as f:
            f.write(buf.read())

        captions = mod.load_annotations(tgz_path)
        assert len(captions) == 2


# ---------------------------------------------------------------------------
# convert_journeydb — tgz range filtering
# ---------------------------------------------------------------------------

class TestConvertJourneydbRangeFilter:
    """
    Test that _tgz_num and the filter logic select the correct archives
    without running the full conversion (which would require real JourneyDB data).
    """

    def _make_fake_tgzs(self, tmpdir: str, indices: list) -> list:
        """Create zero-byte placeholder .tgz files with numeric names."""
        paths = []
        for i in indices:
            p = os.path.join(tmpdir, f"{i:03d}.tgz")
            with open(p, "wb") as f:
                f.write(b"")
            paths.append(p)
        return sorted(paths)

    def _filter_tgzs(self, mod, tgz_files: list, start=None, end=None) -> list:
        """Replicate the filtering logic from convert_journeydb.convert()."""
        def _tgz_num(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                return int(stem)
            except ValueError:
                return -1

        return [
            f for f in tgz_files
            if _tgz_num(f) >= 0
            and (start is None or _tgz_num(f) >= start)
            and (end   is None or _tgz_num(f) <= end)
        ]

    def test_no_filter_includes_all(self, tmp_path):
        mod = _load_script("convert_journeydb")
        tgzs = self._make_fake_tgzs(str(tmp_path), range(50))
        filtered = self._filter_tgzs(mod, tgzs)
        assert len(filtered) == 50

    def test_start_filter(self, tmp_path):
        mod = _load_script("convert_journeydb")
        tgzs = self._make_fake_tgzs(str(tmp_path), range(50))
        filtered = self._filter_tgzs(mod, tgzs, start=25)
        nums = sorted(int(os.path.splitext(os.path.basename(f))[0]) for f in filtered)
        assert min(nums) == 25
        assert max(nums) == 49

    def test_end_filter(self, tmp_path):
        mod = _load_script("convert_journeydb")
        tgzs = self._make_fake_tgzs(str(tmp_path), range(50))
        filtered = self._filter_tgzs(mod, tgzs, end=24)
        nums = sorted(int(os.path.splitext(os.path.basename(f))[0]) for f in filtered)
        assert min(nums) == 0
        assert max(nums) == 24

    def test_start_and_end_filter(self, tmp_path):
        """Chunk 2 (tgz 50–99) from a 200-file set."""
        mod = _load_script("convert_journeydb")
        tgzs = self._make_fake_tgzs(str(tmp_path), range(200))
        filtered = self._filter_tgzs(mod, tgzs, start=50, end=99)
        assert len(filtered) == 50
        nums = sorted(int(os.path.splitext(os.path.basename(f))[0]) for f in filtered)
        assert nums == list(range(50, 100))

    def test_non_numeric_tgz_excluded(self, tmp_path):
        """Non-numeric files like 'train_anno.tgz' are excluded."""
        mod = _load_script("convert_journeydb")
        tgzs = self._make_fake_tgzs(str(tmp_path), range(5))
        # Add a non-numeric entry
        anno = os.path.join(str(tmp_path), "train_anno.tgz")
        with open(anno, "wb") as f:
            f.write(b"")
        all_files = tgzs + [anno]

        def _tgz_num(p):
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                return int(stem)
            except ValueError:
                return -1

        filtered = [f for f in all_files if _tgz_num(f) >= 0]
        assert len(filtered) == 5
        assert anno not in filtered


# ---------------------------------------------------------------------------
# clip_dedup — device detection
# ---------------------------------------------------------------------------

class TestClipDedupDevice:
    def test_device_detection_returns_valid_string(self):
        mod = _load_script("clip_dedup")
        device = mod._clip_device()
        assert device in ("mps", "cuda", "cpu")

    def test_mps_detected_on_apple_silicon(self):
        """On the dev machine (M1 Max), MPS should be detected."""
        import torch
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this machine")
        mod = _load_script("clip_dedup")
        assert mod._clip_device() == "mps"

    def test_command_list_completeness(self):
        """All expected CLI entry points are present (current cmd_* API)."""
        mod = _load_script("clip_dedup")
        assert callable(mod.cmd_embed)
        assert callable(mod.cmd_build_index)
        assert callable(mod.cmd_find_dups)
        assert callable(mod.dedup_wds_tar)


# ---------------------------------------------------------------------------
# clean_wds_pool — DEDUP-3 interrupted-restart rollback helpers
# ---------------------------------------------------------------------------

class TestCleanWdsPoolRollback:
    """The .processing-marker recovery rolls a partially-indexed tar back out of
    the cumulative FAISS index + .ids sidecar so a restart does not deduplicate a
    tar against its own vectors (false-positive self-dupes)."""

    def _index(self, n: int, dim: int = 8):
        import faiss
        rng = np.random.default_rng(0)
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        idx = faiss.IndexFlatIP(dim)
        idx.add(vecs)
        return idx, vecs

    def test_truncate_index_preserves_prefix_in_order(self):
        mod = _load_script("clean_wds_pool")
        idx, vecs = self._index(10)
        out = mod._truncate_index(idx, 7)
        assert out.ntotal == 7
        kept = out.reconstruct_n(0, 7)
        # Insertion order of the first 7 vectors is preserved exactly.
        np.testing.assert_allclose(kept, vecs[:7], rtol=0, atol=1e-6)

    def test_truncate_index_noop_when_count_ge_total(self):
        mod = _load_script("clean_wds_pool")
        idx, _ = self._index(5)
        assert mod._truncate_index(idx, 5) is idx
        assert mod._truncate_index(idx, 9) is idx
        assert mod._truncate_index(None, 3) is None

    def test_truncate_index_to_zero_is_empty(self):
        mod = _load_script("clean_wds_pool")
        idx, _ = self._index(4)
        out = mod._truncate_index(idx, 0)
        assert out.ntotal == 0
        assert out.d == idx.d

    def test_truncate_ids_keeps_first_n_lines(self, tmp_path):
        mod = _load_script("clean_wds_pool")
        ids_path = tmp_path / "dedup_index.ids"
        ids_path.write_text("".join(f"id{i}\n" for i in range(10)))
        mod._truncate_ids(ids_path, 7)
        lines = ids_path.read_text().splitlines()
        assert lines == [f"id{i}" for i in range(7)]

    def test_truncate_ids_noop_when_short_or_missing(self, tmp_path):
        mod = _load_script("clean_wds_pool")
        ids_path = tmp_path / "dedup_index.ids"
        ids_path.write_text("a\nb\n")
        mod._truncate_ids(ids_path, 5)
        assert ids_path.read_text().splitlines() == ["a", "b"]
        # Missing sidecar must not raise.
        mod._truncate_ids(tmp_path / "absent.ids", 3)

    def test_rollback_clears_self_dupe_for_restart(self):
        """End-to-end: a tar's vectors flushed to disk by an interrupt-time flush
        must be removed so the reprocessed tar finds no self-matches."""
        mod = _load_script("clean_wds_pool")
        idx, vecs = self._index(12)
        prior_n = 8  # 4 vectors (indices 8..11) belong to the interrupted tar
        # Before rollback: those 4 vectors self-match at score 1.0 ≥ 0.95.
        tar_batch = vecs[prior_n:]
        D, _ = idx.search(tar_batch, 1)
        assert (D[:, 0] >= 0.95).all()
        # After rollback they are gone, so a reprocess sees no self-dupes.
        rolled = mod._truncate_index(idx, prior_n)
        assert rolled.ntotal == prior_n
        D2, _ = rolled.search(tar_batch, 1)
        assert (D2[:, 0] < 0.95).all()
