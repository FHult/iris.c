"""
train/tests/test_shard_cache_filter.py — the precompute → train handoff contract.

Guards the seam that silently failed flywheel iter 10 ("Shard cache filter: 0/40
shards have qwen3+vae precomputed"): the trainer only trains on shards whose
precomputed .npz files are present and findable under the configured cache dirs,
keyed as "{shard_stem}_{index:04d}.npz". If precompute writes a different key, or
a shard is only partially precomputed, every affected shard is silently skipped.

These tests exercise the extracted helpers in train_ip_adapter:
  shard_internal_prefix, shard_has_cache, filter_shards_with_cache
using fake tar paths (the filter reads only the basename) + empty .npz touch-files
(the filter only os.path.exists, never opens them). No mflux/Metal/real data.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# conftest.py puts train/ on sys.path; import the real module under test.
import train_ip_adapter as tia


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cache(tmp: Path, shard_stems, present_indices=(0, 49),
                encoders=("qwen3", "vae")):
    """Create qwen3/ + vae/ dirs and touch {stem}_{i:04d}.npz for each shard.

    Returns (qwen3_dir, vae_dir). `present_indices` controls which record indices
    are written (default the two the filter probes). `encoders` controls which
    encoder dirs receive the files.
    """
    dirs = {}
    for enc in ("qwen3", "vae"):
        d = tmp / enc
        d.mkdir(parents=True, exist_ok=True)
        dirs[enc] = d
    for stem in shard_stems:
        for enc in encoders:
            for i in present_indices:
                (dirs[enc] / f"{stem}_{i:04d}.npz").touch()
    return str(dirs["qwen3"]), str(dirs["vae"])


def _tars(stems):
    """Fake tar paths — the filter only reads the basename stem."""
    return [f"/fake/staging/{s}.tar" for s in stems]


# ---------------------------------------------------------------------------
# shard_internal_prefix
# ---------------------------------------------------------------------------

class TestInternalPrefix:
    def test_chunk1_stem(self):
        assert tia.shard_internal_prefix("/x/000000.tar") == "000000"

    def test_chunk2_stem(self):
        # Chunk 2+ shards keep their staging filename as the record prefix.
        assert tia.shard_internal_prefix("/y/250000.tar") == "250000"

    def test_strips_dir_and_ext(self):
        assert tia.shard_internal_prefix("a/b/c/000042.tar") == "000042"


# ---------------------------------------------------------------------------
# shard_has_cache  — the per-shard predicate
# ---------------------------------------------------------------------------

class TestHasCache:
    def test_complete_cache_passes(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            q, v = _make_cache(tmp, ["000000"])
            assert tia.shard_has_cache("/s/000000.tar", q, v) is True

    def test_missing_probe_record_excluded(self):
        # Only _0000 written (precompute crashed after first record) → excluded.
        # This is exactly why the filter probes _0049 and not only _0000.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            q, v = _make_cache(tmp, ["000000"], present_indices=(0,))
            assert tia.shard_has_cache("/s/000000.tar", q, v) is False

    def test_vae_missing_excluded(self):
        # qwen3 present, vae absent → excluded (filter requires BOTH encoders).
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            q, v = _make_cache(tmp, ["000000"], encoders=("qwen3",))
            assert tia.shard_has_cache("/s/000000.tar", q, v) is False

    def test_qwen3_missing_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            q, v = _make_cache(tmp, ["000000"], encoders=("vae",))
            assert tia.shard_has_cache("/s/000000.tar", q, v) is False

    def test_naming_mismatch_excluded(self):
        # Regression for the iter-10 failure CLASS: tar filename stem differs from
        # the npz record-key prefix. Tar is "005381.tar" but npz are keyed
        # "200000_*". The filter looks for "005381_0000.npz" → not found → skipped.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            q, v = _make_cache(tmp, ["200000"])          # npz keyed 200000_*
            assert tia.shard_has_cache("/s/005381.tar", q, v) is False  # tar stem 005381


# ---------------------------------------------------------------------------
# filter_shards_with_cache  — the list filter used by train()
# ---------------------------------------------------------------------------

class TestFilterShards:
    def test_all_present(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            stems = ["000000", "000001", "000002"]
            q, v = _make_cache(tmp, stems)
            got = tia.filter_shards_with_cache(_tars(stems), q, v)
            assert len(got) == 3

    def test_partial_pool_filters_uncached(self):
        # 5 shards selected, only 3 precomputed → filter returns those 3.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            cached = ["000000", "000001", "000002"]
            q, v = _make_cache(tmp, cached)
            selected = cached + ["000468", "001253"]      # last 2 not precomputed
            got = tia.filter_shards_with_cache(_tars(selected), q, v)
            assert sorted(os.path.basename(p) for p in got) == \
                ["000000.tar", "000001.tar", "000002.tar"]

    def test_empty_cache_yields_zero(self):
        # The iter-10 failure: cache dirs exist but contain no matching npz.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "qwen3").mkdir()
            (tmp / "vae").mkdir()
            got = tia.filter_shards_with_cache(
                _tars(["000000", "000001"]), str(tmp / "qwen3"), str(tmp / "vae"))
            assert got == []

    def test_ordering_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            stems = ["000005", "000003", "000009"]
            q, v = _make_cache(tmp, stems)
            tars = _tars(stems)
            got = tia.filter_shards_with_cache(tars, q, v)
            assert got == tars   # filter must not reorder
