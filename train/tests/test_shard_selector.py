"""
train/tests/test_shard_selector.py — flywheel shard-selection logic.

The selector decides which n shards each iteration trains on (per-source floor,
diversity, exploration schedule, recency) — the campaign's exploration policy, and
previously untested. Two layers:

  - _resolve_exploration_rate: the pure schedule + adaptive-boost decision (extracted
    behaviour-preserving from select_shards), where off-by-one boundaries on
    through_iteration/after_iteration would silently mis-pace exploration.
  - select_shards: integration against a synthetic ShardScoreDB (tempdir; never the
    live shard_scores.db) — count, per_source_min guarantee, subset/no-dup invariants.

Flywheel-safe: explicit tempdir db_path; pure stdlib sqlite + numpy. No GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import shard_selector as ss
from shard_selector import ShardScoreDB, select_shards


# ---------------------------------------------------------------------------
# _resolve_exploration_rate — schedule + adaptive boost
# ---------------------------------------------------------------------------

class TestResolveExplorationRate:
    def test_base_rate_no_schedule(self):
        assert ss._resolve_exploration_rate({"exploration_rate": 0.4}, 1, 0.0) == 0.4

    def test_default_when_absent(self):
        assert ss._resolve_exploration_rate({}, 1, 0.0) == 0.15

    def test_through_iteration_match(self):
        cfg = {"exploration_rate": 0.2,
               "exploration_schedule": [{"through_iteration": 8, "rate": 0.45},
                                        {"through_iteration": 15, "rate": 0.35}]}
        assert ss._resolve_exploration_rate(cfg, 8, 0.0) == 0.45   # boundary inclusive
        assert ss._resolve_exploration_rate(cfg, 9, 0.0) == 0.35   # next bracket

    def test_first_match_wins(self):
        cfg = {"exploration_schedule": [{"through_iteration": 5, "rate": 0.5},
                                        {"through_iteration": 20, "rate": 0.2}]}
        assert ss._resolve_exploration_rate(cfg, 3, 0.0) == 0.5

    def test_after_iteration(self):
        cfg = {"exploration_rate": 0.3,
               "exploration_schedule": [{"after_iteration": 10, "rate": 0.22}]}
        assert ss._resolve_exploration_rate(cfg, 11, 0.0) == 0.22
        assert ss._resolve_exploration_rate(cfg, 10, 0.0) == 0.3   # not yet 'after'

    def test_iteration_zero_ignores_schedule(self):
        cfg = {"exploration_rate": 0.3,
               "exploration_schedule": [{"through_iteration": 8, "rate": 0.45}]}
        assert ss._resolve_exploration_rate(cfg, 0, 0.0) == 0.3

    def test_adaptive_boost_floors_rate(self):
        # unscored_frac above EXPLORE_THRESH lifts the rate to EXPLORE_BOOSTED...
        assert ss._resolve_exploration_rate({"exploration_rate": 0.1}, 1,
                                            ss.EXPLORE_THRESH + 0.01) == ss.EXPLORE_BOOSTED
        # ...but never LOWERS an already-higher rate
        assert ss._resolve_exploration_rate({"exploration_rate": 0.6}, 1, 0.9) == 0.6

    def test_boost_applies_on_top_of_schedule(self):
        cfg = {"exploration_schedule": [{"through_iteration": 8, "rate": 0.10}]}
        # schedule gives 0.10, boost lifts to EXPLORE_BOOSTED (0.30)
        assert ss._resolve_exploration_rate(cfg, 3, 0.9) == ss.EXPLORE_BOOSTED


# ---------------------------------------------------------------------------
# select_shards — integration on a synthetic DB
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    d = ShardScoreDB(db_path=tmp_path / "sc.db")
    yield d


def _populate(db, counts: dict):
    """counts: {source: n}. Creates sequential shard ids tagged with each source."""
    sid = 0
    for src, n in counts.items():
        for _ in range(n):
            db.upsert_shard(f"{sid:06d}", f"/shards/{sid:06d}.tar", source=src)
            sid += 1
    return sid


class TestSelectShards:
    def test_small_pool_returns_all(self, db):
        _populate(db, {"journeydb": 6, "coyo": 4})
        out = select_shards(db, 20, {}, "run", 1)
        assert len(out) == 10                      # pool <= n_shards → all
        assert all(p.endswith(".tar") for p in out)

    def test_returns_exactly_n(self, db):
        _populate(db, {"journeydb": 60, "coyo": 30, "laion": 10})
        out = select_shards(db, 40, {"performance_weight": 0.3}, "run", 1)
        assert len(out) == 40
        assert len(set(out)) == 40                 # no duplicates

    def test_per_source_min_honoured(self, db):
        _populate(db, {"journeydb": 80, "coyo": 20})
        cfg = {"performance_weight": 0.3, "per_source_min": {"journeydb": 6, "coyo": 6}}
        out = select_shards(db, 40, cfg, "run", 1)
        # map selected paths back to their source
        by_id = {s["shard_id"]: s["source"] for s in db.get_all_shards()}
        srcs = [by_id[Path(p).stem] for p in out]
        assert srcs.count("coyo") >= 6             # the fix this guards: coyo floor met
        assert srcs.count("journeydb") >= 6

    def test_selection_is_subset_of_pool(self, db):
        n = _populate(db, {"journeydb": 30, "coyo": 30})
        out = select_shards(db, 25, {}, "run", 1)
        pool = {s["path"] for s in db.get_all_shards()}
        assert set(out) <= pool and len(out) == 25

    def test_empty_pool_returns_empty(self, db):
        assert select_shards(db, 10, {}, "run", 1) == []
