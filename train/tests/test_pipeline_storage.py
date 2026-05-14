"""
train/tests/test_pipeline_storage.py — Unit tests for PIPELINE-26/27/28/29.

Covers:
  - pipeline_lib COLD_* constants
  - build_shards provenance helpers (_classify_source, _write_provenance_sidecars)
  - shard_scorer.compute_tgz_scores join logic
  - download_convert._prioritised_tgz_range (sequential fallback + scored ordering)
  - data_stager.DataStager init + update_best_symlinks (relative symlink, idempotency)

All tests use only tempdir fixtures and in-process logic — no real I/O to cold volume,
no DB connections, no subprocess launches.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Script loader (mirrors test_scripts.py pattern)
# ---------------------------------------------------------------------------

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# PIPELINE-26 — pipeline_lib COLD_* constants
# ---------------------------------------------------------------------------

class TestColdConstants:
    def test_importable(self):
        lib = _load("pipeline_lib")
        assert lib.COLD_ROOT           == Path("/Volumes/16TBCold")
        assert lib.COLD_PRECOMPUTE_DIR == Path("/Volumes/16TBCold/precomputed")
        assert lib.COLD_WEIGHTS_DIR    == Path("/Volumes/16TBCold/weights")
        assert lib.COLD_METADATA_DIR   == Path("/Volumes/16TBCold/metadata")

    def test_derived_from_cold_root(self):
        lib = _load("pipeline_lib")
        assert lib.COLD_PRECOMPUTE_DIR == lib.COLD_ROOT / "precomputed"
        assert lib.COLD_WEIGHTS_DIR    == lib.COLD_ROOT / "weights"
        assert lib.COLD_METADATA_DIR   == lib.COLD_ROOT / "metadata"

    def test_shard_scores_db_and_ablation_db_on_data_root(self):
        """DB paths must sit on DATA_ROOT, not COLD_ROOT — stager uses these constants."""
        lib = _load("pipeline_lib")
        assert lib.SHARD_SCORES_DB_PATH.parent == lib.DATA_ROOT
        assert lib.ABLATION_DB_PATH.parent     == lib.DATA_ROOT


# ---------------------------------------------------------------------------
# PIPELINE-27 — build_shards provenance helpers
# ---------------------------------------------------------------------------

class TestClassifySource:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _load("build_shards")

    def test_jdb_named_by_index(self):
        r = self.mod._classify_source("/Volumes/cold/raw/journeydb/007.tar")
        assert r == {"type": "jdb", "tgz": 7, "path": "/Volumes/cold/raw/journeydb/007.tar"}

    def test_jdb_stem_zero_padded(self):
        r = self.mod._classify_source("/data/journeydb/042.tar")
        assert r["tgz"] == 42

    def test_jdb_non_numeric_stem(self):
        r = self.mod._classify_source("/data/jdb/annotations.tar")
        assert r["type"] == "jdb"
        assert "tgz" not in r

    def test_wikiart(self):
        r = self.mod._classify_source("/data/wikiart/paintings.tar")
        assert r["type"] == "wikiart"

    def test_laion(self):
        r = self.mod._classify_source("/data/laion/subset.tar")
        assert r["type"] == "laion"

    def test_coyo(self):
        r = self.mod._classify_source("/data/coyo/chunk.tar")
        assert r["type"] == "coyo"

    def test_unknown(self):
        r = self.mod._classify_source("/data/something/else.tar")
        assert r["type"] == "unknown"

    def test_case_insensitive_jdb(self):
        r = self.mod._classify_source("/data/JourneyDB/003.tar")
        assert r["type"] == "jdb"


class TestWriteProvenanceSidecars:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _load("build_shards")

    def test_writes_json_alongside_shard(self, tmp_path):
        (tmp_path / "000042.tar").write_bytes(b"")
        prov = {42: ["/data/journeydb/003.tar", "/data/wikiart/w.tar"]}
        self.mod._write_provenance_sidecars(str(tmp_path), prov)
        pf = tmp_path / "000042.provenance.json"
        assert pf.exists()
        data = json.loads(pf.read_text())
        assert data["shard_id"] == "shard-000042"
        types = {s["type"] for s in data["sources"]}
        assert types == {"jdb", "wikiart"}

    def test_skips_if_tar_missing(self, tmp_path):
        # shard 99 has no .tar — sidecar should NOT be created
        prov = {99: ["/data/journeydb/003.tar"]}
        self.mod._write_provenance_sidecars(str(tmp_path), prov)
        assert not (tmp_path / "000099.provenance.json").exists()

    def test_deduplicates_source_paths(self, tmp_path):
        (tmp_path / "000001.tar").write_bytes(b"")
        # Same path listed twice — should appear once in sources
        prov = {1: ["/data/journeydb/000.tar", "/data/journeydb/000.tar"]}
        self.mod._write_provenance_sidecars(str(tmp_path), prov)
        data = json.loads((tmp_path / "000001.provenance.json").read_text())
        assert len(data["sources"]) == 1

    def test_multiple_shards(self, tmp_path):
        (tmp_path / "000000.tar").write_bytes(b"")
        (tmp_path / "000001.tar").write_bytes(b"")
        prov = {
            0: ["/data/journeydb/000.tar"],
            1: ["/data/journeydb/001.tar"],
        }
        self.mod._write_provenance_sidecars(str(tmp_path), prov)
        assert (tmp_path / "000000.provenance.json").exists()
        assert (tmp_path / "000001.provenance.json").exists()


# ---------------------------------------------------------------------------
# PIPELINE-27 — shard_scorer.compute_tgz_scores
# ---------------------------------------------------------------------------

class TestComputeTgzScores:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _load("shard_scorer")

    def _score(self, shard_scores, provenance):
        return self.mod.compute_tgz_scores(shard_scores, provenance)

    def test_basic_join(self):
        scores = {"000042": 0.72, "000007": 0.80}
        prov = {
            "000042": [{"type": "jdb", "tgz": 3}],
            "000007": [{"type": "jdb", "tgz": 3}],
        }
        r = self._score(scores, prov)
        assert 3 in r
        assert abs(r[3]["score"] - 0.76) < 1e-5
        assert r[3]["n_shards"] == 2

    def test_multiple_tgzs_per_shard(self):
        # Shard 000042 got records from tgz 3 and tgz 5
        scores = {"000042": 0.60}
        prov = {"000042": [{"type": "jdb", "tgz": 3}, {"type": "jdb", "tgz": 5}]}
        r = self._score(scores, prov)
        assert r[3]["score"] == pytest.approx(0.60)
        assert r[5]["score"] == pytest.approx(0.60)

    def test_unscored_shard_skipped(self):
        scores = {"000042": 0.80}        # 000001 has no score
        prov = {
            "000042": [{"type": "jdb", "tgz": 3}],
            "000001": [{"type": "jdb", "tgz": 9}],
        }
        r = self._score(scores, prov)
        assert 9 not in r, "tgz with only unscored shards should be absent"

    def test_non_jdb_sources_ignored(self):
        scores = {"000042": 0.80}
        prov = {"000042": [
            {"type": "jdb", "tgz": 3},
            {"type": "wikiart", "path": "/x"},  # no tgz field
            {"type": "laion",   "path": "/y"},
        ]}
        r = self._score(scores, prov)
        assert list(r.keys()) == [3]

    def test_empty_inputs(self):
        assert self._score({}, {}) == {}
        assert self._score({"000042": 0.5}, {}) == {}
        assert self._score({}, {"000042": [{"type": "jdb", "tgz": 0}]}) == {}

    def test_score_mean_accuracy(self):
        # tgz 7: shards with scores 0.40, 0.60, 0.80 → mean 0.60
        scores = {"a": 0.40, "b": 0.60, "c": 0.80}
        prov = {k: [{"type": "jdb", "tgz": 7}] for k in ("a", "b", "c")}
        r = self._score(scores, prov)
        assert r[7]["score"] == pytest.approx(0.60, abs=1e-5)
        assert r[7]["n_shards"] == 3


# ---------------------------------------------------------------------------
# PIPELINE-27 — download_convert._prioritised_tgz_range
# ---------------------------------------------------------------------------

class TestPrioritisedTgzRange:
    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _load("download_convert")

    def _pri(self, start, end, config=None):
        return self.mod._prioritised_tgz_range(start, end, config or {})

    def test_sequential_when_no_scores_file(self):
        result = self._pri(0, 4)
        assert result == [0, 1, 2, 3, 4]

    def test_sequential_when_cold_root_missing(self, tmp_path):
        # cold_root exists but no metadata/tgz_scores.json inside
        config = {"storage": {"cold_root": str(tmp_path)}}
        result = self._pri(0, 4, config)
        assert result == [0, 1, 2, 3, 4]

    def test_scored_ordering(self, tmp_path):
        scores = {"tgz_scores": {
            "3": {"score": 0.90},
            "1": {"score": 0.75},
            "7": {"score": 0.50},
        }}
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "tgz_scores.json").write_text(json.dumps(scores))
        config = {"storage": {"cold_root": str(tmp_path)}}
        result = self._pri(0, 9, config)
        # Scored tgzs first (descending score), then unscored sequentially
        assert result[0] == 3
        assert result[1] == 1
        assert result[2] == 7
        unscored = result[3:]
        assert set(unscored) == {0, 2, 4, 5, 6, 8, 9}
        assert unscored == sorted(unscored)

    def test_scored_outside_range_ignored(self, tmp_path):
        # tgz 99 is scored but outside the [0,4] range
        scores = {"tgz_scores": {"99": {"score": 0.99}, "2": {"score": 0.80}}}
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "tgz_scores.json").write_text(json.dumps(scores))
        config = {"storage": {"cold_root": str(tmp_path)}}
        result = self._pri(0, 4, config)
        assert 99 not in result
        assert result[0] == 2   # highest scoring tgz in range

    def test_all_unscored_returns_sequential(self, tmp_path):
        # scores file exists but none of range [0,3] appear in it
        scores = {"tgz_scores": {"99": {"score": 0.99}}}
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "tgz_scores.json").write_text(json.dumps(scores))
        config = {"storage": {"cold_root": str(tmp_path)}}
        result = self._pri(0, 3, config)
        assert result == [0, 1, 2, 3]

    def test_corrupt_scores_file_falls_back(self, tmp_path):
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "tgz_scores.json").write_text("not json {{")
        config = {"storage": {"cold_root": str(tmp_path)}}
        result = self._pri(0, 2, config)
        assert result == [0, 1, 2]


# ---------------------------------------------------------------------------
# PIPELINE-29 — DataStager init + update_best_symlinks
# ---------------------------------------------------------------------------

class TestDataStager:
    @pytest.fixture
    def stager(self, tmp_path, monkeypatch):
        # Redirect DATA_ROOT so SHARD_SCORES_DB_PATH / ABLATION_DB_PATH don't
        # point to real files; they won't be accessed in these tests.
        lib = _load("pipeline_lib")
        monkeypatch.setattr(lib, "DATA_ROOT", tmp_path)
        mod = _load("data_stager")
        cold = tmp_path / "cold"
        hot  = tmp_path / "hot"
        cold.mkdir(); hot.mkdir()
        cfg  = {"storage": {"cold_root": str(cold), "hot_root": str(hot)}}
        return mod.DataStager(cfg)

    def test_enabled_when_cold_ne_hot(self, stager):
        assert stager.enabled is True

    def test_derived_cold_paths(self, stager):
        assert stager._cold_weights  == stager.cold_root / "weights"
        assert stager._cold_metadata == stager.cold_root / "metadata"

    def test_update_best_symlinks_creates_relative_symlink(self, stager, tmp_path):
        # Create a fake checkpoint
        campaign = stager.cold_root / "weights" / "flywheel-20260514"
        campaign.mkdir(parents=True)
        ckpt = campaign / "chunk1_final.safetensors"
        ckpt.write_bytes(b"weights")

        stager.update_best_symlinks({"cond_gap": 0.42}, ckpt)
        best_dir = stager.cold_root / "weights" / "best"
        link = best_dir / "cond_gap.safetensors"
        assert link.is_symlink()
        assert link.resolve() == ckpt.resolve()
        # Must be a relative path (survives volume remounts)
        raw_target = os.readlink(str(link))
        assert not os.path.isabs(raw_target), f"symlink must be relative, got: {raw_target}"

    def test_update_best_symlinks_json_sidecar(self, stager):
        campaign = stager.cold_root / "weights" / "flywheel-20260514"
        campaign.mkdir(parents=True)
        ckpt = campaign / "chunk1_final.safetensors"
        ckpt.write_bytes(b"w")
        stager.update_best_symlinks({"cond_gap": 0.42, "loss": 0.18}, ckpt)
        best_dir = stager.cold_root / "weights" / "best"
        meta = json.loads((best_dir / "cond_gap.json").read_text())
        assert meta["value"] == 0.42

    def test_no_update_on_worse_metric(self, stager):
        campaign = stager.cold_root / "weights" / "flywheel-20260514"
        campaign.mkdir(parents=True)
        ckpt = campaign / "chunk1_final.safetensors"
        ckpt.write_bytes(b"w")
        stager.update_best_symlinks({"cond_gap": 0.42}, ckpt)
        stager.update_best_symlinks({"cond_gap": 0.10}, ckpt)  # worse — should not update
        best_dir = stager.cold_root / "weights" / "best"
        meta = json.loads((best_dir / "cond_gap.json").read_text())
        assert meta["value"] == 0.42, "best should stay at 0.42, not regress to 0.10"

    def test_update_on_better_metric(self, stager):
        campaign = stager.cold_root / "weights" / "flywheel-20260514"
        campaign.mkdir(parents=True)
        ckpt1 = campaign / "chunk1_final.safetensors"
        ckpt2 = campaign / "chunk2_final.safetensors"
        ckpt1.write_bytes(b"v1"); ckpt2.write_bytes(b"v2")
        stager.update_best_symlinks({"cond_gap": 0.42}, ckpt1)
        stager.update_best_symlinks({"cond_gap": 0.55}, ckpt2)
        best_dir = stager.cold_root / "weights" / "best"
        assert (best_dir / "cond_gap.safetensors").resolve() == ckpt2.resolve()
        meta = json.loads((best_dir / "cond_gap.json").read_text())
        assert meta["value"] == 0.55

    def test_lower_is_better_for_loss(self, stager):
        campaign = stager.cold_root / "weights" / "flywheel-20260514"
        campaign.mkdir(parents=True)
        ckpt1 = campaign / "chunk1_final.safetensors"
        ckpt2 = campaign / "chunk2_final.safetensors"
        ckpt1.write_bytes(b"v1"); ckpt2.write_bytes(b"v2")
        stager.update_best_symlinks({"loss": 0.30}, ckpt1)
        stager.update_best_symlinks({"loss": 0.20}, ckpt2)  # lower loss = better
        best_dir = stager.cold_root / "weights" / "best"
        assert (best_dir / "loss.safetensors").resolve() == ckpt2.resolve()

    def test_non_numeric_metric_values_skipped(self, stager):
        campaign = stager.cold_root / "weights" / "flywheel-20260514"
        campaign.mkdir(parents=True)
        ckpt = campaign / "chunk1_final.safetensors"
        ckpt.write_bytes(b"w")
        # Should not raise or create a symlink for non-numeric value
        stager.update_best_symlinks({"cond_gap": "not-a-number"}, ckpt)
        best_dir = stager.cold_root / "weights" / "best"
        assert not (best_dir / "cond_gap.safetensors").exists()
