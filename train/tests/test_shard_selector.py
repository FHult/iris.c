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


class TestSourceAttribution:
    """Per-source rollup must be campaign-scoped, single-convention, and honest
    about confidence/ubiquity (debug/source_attribution.py + doctor consume this)."""

    def _campaign(self, db, name, n_iters, sources):
        """sources: {src: n_shards}. Every shard included every iter (ubiquity 1),
        with iteration-level cond_gap (same for all shards in an iter)."""
        _populate(db, sources)
        ids_by_src = {}
        sid = 0
        for src, n in sources.items():
            ids_by_src[src] = [f"{sid+i:06d}" for i in range(n)]
            sid += n
        for it in range(1, n_iters + 1):
            cg = 0.05 + 0.001 * it
            for src, ids in ids_by_src.items():
                for shard in ids:
                    db.update_scores(shard, ref_gap=0.01, cond_gap=cg, loss=0.5,
                                     flywheel_name=name, iteration=it)
        return ids_by_src

    def test_campaign_scoped_and_ubiquitous(self, db):
        self._campaign(db, "c1", 6, {"coyo": 4, "journeydb": 6})
        roll = db.source_attribution("c1")
        srcs = {r["source"]: r for r in roll}
        assert set(srcs) == {"coyo", "journeydb"}
        # every shard included every iteration, never excluded -> ubiquity 1.0,
        # and NO attributed shards (needs >=3 excluded obs too).
        assert srcs["coyo"]["ubiquity"] == 1.0
        assert srcs["journeydb"]["n_attributed"] == 0
        assert srcs["journeydb"]["attr_cond_gap_mean"] is None

    def test_excluded_observations_enable_attribution(self, db):
        ids = self._campaign(db, "c2", 4, {"coyo": 4, "journeydb": 4})
        all_ids = ids["coyo"] + ids["journeydb"]
        keep = set(ids["journeydb"])   # exclude all coyo
        # Exclude every coyo shard in 3 extra iterations at a LOWER cond_gap, so
        # coyo's contrast (incl ~0.05 vs excl 0.02) is positive & separable.
        for it in range(5, 8):
            db.update_excluded_scores(all_ids, keep, ref_gap=0.0, cond_gap=0.02,
                                      flywheel_name="c2", iteration=it)
        roll = {r["source"]: r for r in db.source_attribution("c2")}
        assert roll["coyo"]["n_attributed"] == 4          # 4 incl & 3 excl each
        assert roll["coyo"]["attr_cond_gap_mean"] > 0.02  # ~0.05 - 0.02
        assert roll["coyo"]["ubiquity"] < 1.0             # excluded in some iters
        # journeydb still always-included -> not attributed
        assert roll["journeydb"]["n_attributed"] == 0

    def test_other_campaign_isolated(self, db):
        # One shared pool scored under two campaigns; the rollup must count only
        # the requested campaign's updates (cross-campaign convention isolation).
        self._campaign(db, "c3", 4, {"coyo": 3})
        for it in range(1, 5):
            for shard in ("000000", "000001", "000002"):   # same shards
                db.update_scores(shard, ref_gap=0.01, cond_gap=0.9, loss=0.5,
                                 flywheel_name="c4", iteration=it)
        c3 = db.source_attribution("c3")[0]
        c4 = db.source_attribution("c4")[0]
        assert c3["incl_cond_gap_mean"] < 0.1    # c3's ~0.05x, not c4's 0.9
        assert c4["incl_cond_gap_mean"] > 0.8    # c4's 0.9, isolated

    def test_iteration_mix_reconstructs(self, db):
        self._campaign(db, "c5", 3, {"coyo": 2, "journeydb": 5})
        mix = db.source_iteration_mix("c5")
        it1 = {m["source"]: m["n"] for m in mix if m["it"] == 1}
        assert it1 == {"coyo": 2, "journeydb": 5}


class TestSourceHoldout:
    """resolve_source_holdout: pure schedule fn; select_shards(exclude_sources)
    drops a source from the candidate pool (SRC-ATTR-1 follow-up)."""

    def test_no_config_holds_nothing(self):
        from shard_selector import resolve_source_holdout
        assert resolve_source_holdout(None, 5) == set()
        assert resolve_source_holdout({"sources": []}, 5) == set()

    def test_rotation_cycles_sources(self):
        from shard_selector import resolve_source_holdout
        cfg = {"sources": ["coyo", "journeydb", "wikiart"]}
        got = [next(iter(resolve_source_holdout(cfg, it))) for it in range(1, 7)]
        assert got == ["coyo", "journeydb", "wikiart", "coyo", "journeydb", "wikiart"]

    def test_every_and_start(self):
        from shard_selector import resolve_source_holdout
        cfg = {"sources": ["coyo", "journeydb"], "every": 2, "start": 3}
        # active on iters 3,5,7…; off on 1,2,4,6
        assert resolve_source_holdout(cfg, 1) == set()
        assert resolve_source_holdout(cfg, 2) == set()
        assert resolve_source_holdout(cfg, 3) == {"coyo"}
        assert resolve_source_holdout(cfg, 4) == set()
        assert resolve_source_holdout(cfg, 5) == {"journeydb"}
        assert resolve_source_holdout(cfg, 7) == {"coyo"}

    def test_excluded_source_never_selected(self, db):
        _populate(db, {"coyo": 10, "journeydb": 30})
        sel = select_shards(db, 20, {"performance_weight": 0.3,
                                     "exploration_rate": 0.5}, "c", 1,
                            exclude_sources={"coyo"})
        srcs = {db._conn.execute("SELECT source FROM shards WHERE path=?",
                                 (p,)).fetchone()[0] for p in sel}
        assert "coyo" not in srcs
        assert srcs == {"journeydb"}

    def test_no_exclusion_includes_all_sources(self, db):
        _populate(db, {"coyo": 10, "journeydb": 30})
        sel = select_shards(db, 20, {"performance_weight": 0.3}, "c", 1)
        srcs = {db._conn.execute("SELECT source FROM shards WHERE path=?",
                                 (p,)).fetchone()[0] for p in sel}
        assert srcs == {"coyo", "journeydb"}   # both present when nothing excluded
