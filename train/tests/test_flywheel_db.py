"""
train/tests/test_flywheel_db.py — FlywheelDB persistence roundtrip (GROK-TEST-7).

The flywheel's campaign telemetry + best-checkpoint selection runs entirely
through this SQLite DB; it had no test coverage. These exercise insert/update/
query/best-selection against a tempdir DB.

Fully isolated and flywheel-safe: FlywheelDB takes an explicit db_path, so each
test gets a fresh DB in a tempdir — it never opens the live flywheel_history.db.
Pure stdlib sqlite3; no GPU, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from flywheel_lib import FlywheelDB


@pytest.fixture
def db(tmp_path):
    return FlywheelDB(db_path=tmp_path / "fw.db")


def _insert(db, name="run-x", iteration=1, shards=("000000", "000001"),
            steps=1000):
    return db.insert_iteration(
        name=name, iteration=iteration, n_shards=len(shards),
        shard_ids=list(shards), hyperparams={"lr": 1e-4}, steps=steps,
        git_commit="abc1234", checkpoint_hash="h0",
    )


# ---------------------------------------------------------------------------
# insert / get roundtrip
# ---------------------------------------------------------------------------

class TestInsertGet:
    def test_insert_returns_rowid_and_is_queryable(self, db):
        rid = _insert(db, iteration=1)
        assert isinstance(rid, int)
        rows = db.get_iterations("run-x")
        assert len(rows) == 1
        assert rows[0]["iteration"] == 1
        assert rows[0]["n_shards"] == 2
        assert rows[0]["steps"] == 1000

    def test_selected_shards_roundtrips_as_json(self, db):
        import json
        _insert(db, shards=("000042", "000043", "000044"))
        row = db.get_iterations("run-x")[0]
        # Stored as a JSON string in the selected_shards column.
        assert json.loads(row["selected_shards"]) == ["000042", "000043", "000044"]

    def test_iterations_ordered_by_iteration(self, db):
        _insert(db, iteration=3)
        _insert(db, iteration=1)
        _insert(db, iteration=2)
        order = [r["iteration"] for r in db.get_iterations("run-x")]
        assert order == [1, 2, 3]


# ---------------------------------------------------------------------------
# update_iteration
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_sets_status_and_metrics(self, db):
        rid = _insert(db, iteration=1)
        db.update_iteration(
            row_id=rid, status="done", exit_code=0, elapsed_secs=120,
            train_loss=0.5, ref_gap=0.1, cond_gap=0.25, checkpoint="step_1000.safetensors",
            checkpoint_hash="h1",
        )
        row = db.get_iterations("run-x")[0]
        assert row["status"] == "done"
        assert row["exit_code"] == 0
        assert row["cond_gap"] == 0.25
        assert row["checkpoint"] == "step_1000.safetensors"

    def test_failed_iteration_persists_exit_code(self, db):
        rid = _insert(db, iteration=1)
        db.update_iteration(
            row_id=rid, status="failed", exit_code=139, elapsed_secs=60,
            train_loss=None, ref_gap=None, cond_gap=None,
        )
        row = db.get_iterations("run-x")[0]
        assert row["status"] == "failed"
        assert row["exit_code"] == 139
        assert row["cond_gap"] is None


# ---------------------------------------------------------------------------
# get_best — cond_gap is the selection criterion
# ---------------------------------------------------------------------------

class TestGetBest:
    def test_none_when_no_cond_gap(self, db):
        rid = _insert(db, iteration=1)
        db.update_iteration(row_id=rid, status="failed", exit_code=1,
                            elapsed_secs=10, train_loss=None, ref_gap=None,
                            cond_gap=None)
        assert db.get_best("run-x") is None

    def test_highest_cond_gap_wins(self, db):
        for it, cg in [(1, 0.10), (2, 0.30), (3, 0.20)]:
            rid = _insert(db, iteration=it)
            db.update_iteration(row_id=rid, status="done", exit_code=0,
                                elapsed_secs=100, train_loss=0.4, ref_gap=0.0,
                                cond_gap=cg)
        best = db.get_best("run-x")
        assert best is not None
        assert best["iteration"] == 2          # cond_gap 0.30 is highest
        assert best["cond_gap"] == 0.30

    def test_null_cond_gap_excluded_from_best(self, db):
        rid1 = _insert(db, iteration=1)
        db.update_iteration(row_id=rid1, status="done", exit_code=0,
                            elapsed_secs=100, train_loss=0.4, ref_gap=0.0,
                            cond_gap=0.15)
        rid2 = _insert(db, iteration=2)         # later iter but no cond_gap
        db.update_iteration(row_id=rid2, status="failed", exit_code=1,
                            elapsed_secs=10, train_loss=None, ref_gap=None,
                            cond_gap=None)
        best = db.get_best("run-x")
        assert best["iteration"] == 1           # the only scored iteration


# ---------------------------------------------------------------------------
# checkpoint log + isolation
# ---------------------------------------------------------------------------

class TestCheckpointLogAndIsolation:
    def test_checkpoint_log_roundtrip_and_best(self, db):
        for it in (1, 2):
            _insert(db, iteration=it)
            db.upsert_checkpoint(name="run-x", iteration=it,
                                 checkpoint_path=f"step_{it}.safetensors",
                                 checkpoint_hash=f"h{it}",
                                 ref_gap=0.0, cond_gap=0.1 * it, train_loss=0.3)
        db.mark_best_checkpoint("run-x", 2)
        hist = db.get_checkpoint_history("run-x")
        assert len(hist) == 2
        best = [h for h in hist if h["is_best"]]
        assert len(best) == 1 and best[0]["iteration"] == 2

    def test_campaigns_are_isolated(self, db):
        _insert(db, name="alpha", iteration=1)
        _insert(db, name="beta", iteration=1)
        _insert(db, name="beta", iteration=2)
        assert len(db.get_iterations("alpha")) == 1
        assert len(db.get_iterations("beta")) == 2
        assert set(db.get_all_run_names()) == {"alpha", "beta"}
