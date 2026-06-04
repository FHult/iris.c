"""
train/tests/test_ablation_db.py — AblationDB persistence roundtrip (GROK-TEST-7 remainder).

The ablation harness's experiment history + best/Pareto selection runs entirely
through this SQLite DB and had no test coverage. Useful before re-enabling ablation
in a post-warmup campaign: a silent schema/decode regression would corrupt the
hyperparameter search that adopts "best" arms.

Fully isolated and flywheel-safe: AblationDB takes an explicit db_path, so each test
uses a fresh tempdir DB — it never opens the live ablation_history.db. Pure stdlib
sqlite3; no GPU, no network (git_commit is read locally and not asserted on).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from ablation_harness import AblationDB


@pytest.fixture
def db(tmp_path):
    d = AblationDB(tmp_path / "abl.db")
    yield d
    d.close()


def _exp(db, run="run-x", params=None, strategy="grid", steps=500):
    params = params or {"lr": 1e-4, "cross_ref_prob": 0.35}
    return db.insert_experiment(run, params, strategy, steps)


def _score(db, exp_id, score=0.5, ref_gap=0.2, cond_gap=0.3, final_loss=0.4,
           verdict="OK", snapshots=None, exit_code=0):
    db.update_experiment(
        exp_id, score=score, verdict=verdict, ref_gap=ref_gap, cond_gap=cond_gap,
        final_loss=final_loss, elapsed_secs=120, n_snapshots=len(snapshots or []),
        exit_code=exit_code, snapshots=snapshots or [], grad_norm_final=0.5,
        ip_scale_final=0.8, stopped_early=False, stop_step=None)


# ---------------------------------------------------------------------------
# insert / get roundtrip
# ---------------------------------------------------------------------------

class TestInsertGet:
    def test_insert_returns_id_and_roundtrips(self, db):
        eid = _exp(db, params={"lr": 2e-4})
        assert isinstance(eid, int) and eid > 0
        rows = db.get_experiments("run-x")
        assert len(rows) == 1
        r = rows[0]
        assert r["id"] == eid
        assert r["params"] == {"lr": 2e-4}        # JSON-decoded back to dict
        assert r["strategy"] == "grid" and r["steps"] == 500
        assert r["combo_id"] == f"exp_{eid:04d}"  # _decode_row synthetic field

    def test_unscored_experiment_has_null_score(self, db):
        _exp(db)
        assert db.get_experiments("run-x")[0]["score"] is None

    def test_ordering_by_id(self, db):
        a = _exp(db, params={"lr": 1})
        b = _exp(db, params={"lr": 2})
        c = _exp(db, params={"lr": 3})
        assert [r["id"] for r in db.get_experiments("run-x")] == [a, b, c]


# ---------------------------------------------------------------------------
# params_hash / is_duplicate
# ---------------------------------------------------------------------------

class TestDuplicate:
    def test_params_hash_deterministic_and_order_independent(self):
        a = AblationDB.params_hash({"lr": 1e-4, "p": 0.3})
        b = AblationDB.params_hash({"p": 0.3, "lr": 1e-4})
        assert a == b and len(a) == 16

    def test_is_duplicate(self, db):
        p = {"lr": 1e-4, "cross_ref_prob": 0.35}
        assert db.is_duplicate("run-x", p) is False
        _exp(db, params=p)
        assert db.is_duplicate("run-x", p) is True
        assert db.is_duplicate("run-x", {"lr": 9e-9}) is False
        assert db.is_duplicate("other-run", p) is False   # scoped per run_name


# ---------------------------------------------------------------------------
# update_experiment roundtrip
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_persists_metrics_and_snapshots(self, db):
        eid = _exp(db)
        snaps = [{"step": 100, "loss": 0.9}, {"step": 200, "loss": 0.7}]
        _score(db, eid, score=0.77, ref_gap=0.21, cond_gap=0.34,
               final_loss=0.45, verdict="GOOD", snapshots=snaps)
        r = db.get_experiments("run-x")[0]
        assert r["score"] == 0.77 and r["verdict"] == "GOOD"
        assert r["ref_gap"] == 0.21 and r["cond_gap"] == 0.34
        assert r["snapshots"] == snaps                  # JSON list roundtrip
        assert r["mean_ref_gap"] == 0.21 and r["mean_cond_gap"] == 0.34  # normalised fields
        assert r["grad_norm_final"] == 0.5 and r["ip_scale_final"] == 0.8

    def test_failed_run_records_exit_code(self, db):
        eid = _exp(db)
        _score(db, eid, score=None, verdict="FAIL", exit_code=1)
        r = db.get_experiments("run-x")[0]
        assert r["exit_code"] == 1 and r["verdict"] == "FAIL" and r["score"] is None


# ---------------------------------------------------------------------------
# get_best / scored_only
# ---------------------------------------------------------------------------

class TestBest:
    def test_get_best_orders_by_score_desc_and_excludes_unscored(self, db):
        lo = _exp(db, params={"lr": 1}); _score(db, lo, score=0.2)
        hi = _exp(db, params={"lr": 2}); _score(db, hi, score=0.9)
        mid = _exp(db, params={"lr": 3}); _score(db, mid, score=0.5)
        _exp(db, params={"lr": 4})  # unscored — must be excluded
        best = db.get_best("run-x")
        assert [r["id"] for r in best] == [hi, mid, lo]

    def test_get_best_limit(self, db):
        for i in range(4):
            e = _exp(db, params={"lr": i}); _score(db, e, score=float(i))
        assert len(db.get_best("run-x", n=2)) == 2

    def test_scored_only_filter(self, db):
        a = _exp(db, params={"lr": 1}); _score(db, a, score=0.3)
        _exp(db, params={"lr": 2})  # unscored
        assert len(db.get_experiments("run-x")) == 2
        assert len(db.get_experiments("run-x", scored_only=True)) == 1


# ---------------------------------------------------------------------------
# run isolation / run names
# ---------------------------------------------------------------------------

class TestRunIsolation:
    def test_runs_are_isolated(self, db):
        _exp(db, run="run-a", params={"lr": 1})
        _exp(db, run="run-b", params={"lr": 2})
        assert len(db.get_experiments("run-a")) == 1
        assert len(db.get_experiments("run-b")) == 1
        assert db.get_all_run_names() == ["run-a", "run-b"]   # DISTINCT, sorted


# ---------------------------------------------------------------------------
# Pareto front (3-objective: max ref_gap, max cond_gap, min final_loss)
# ---------------------------------------------------------------------------

class TestParetoFront:
    def test_dominated_experiment_excluded(self, db):
        a = _exp(db, params={"lr": 1}); _score(db, a, ref_gap=0.5, cond_gap=0.5, final_loss=0.1)
        b = _exp(db, params={"lr": 2}); _score(db, b, ref_gap=0.3, cond_gap=0.3, final_loss=0.2)
        # b is dominated by a on all three objectives → front = {a}
        n = db.update_pareto_front("run-x")
        assert n == 1
        flags = {r["id"]: r["is_pareto"] for r in db.get_experiments("run-x")}
        assert flags[a] == 1 and flags[b] == 0

    def test_non_dominated_both_on_front(self, db):
        a = _exp(db, params={"lr": 1}); _score(db, a, ref_gap=0.5, cond_gap=0.2, final_loss=0.3)
        c = _exp(db, params={"lr": 2}); _score(db, c, ref_gap=0.2, cond_gap=0.9, final_loss=0.3)
        # neither dominates the other (a wins ref_gap, c wins cond_gap) → both on front
        assert db.update_pareto_front("run-x") == 2


# ---------------------------------------------------------------------------
# post_train_validation roundtrip
# ---------------------------------------------------------------------------

class TestValidation:
    def test_insert_and_get_validation(self, db):
        eid = _exp(db)
        db.insert_validation(eid, "run-x", {
            "checkpoint_path": "/ckpt/x.safetensors", "weight_ok": True,
            "weight_errors": ["minor"], "n_params": 1234, "clip_i": 0.71,
            "adapter_delta": 0.05, "clip_skipped": False, "verdict": "PASS",
            "elapsed_secs": 12.5,
        })
        vals = db.get_validations("run-x")
        assert set(vals.keys()) == {eid}
        v = vals[eid]
        assert v["weight_ok"] == 1 and v["verdict"] == "PASS"
        assert v["weight_errors"] == ["minor"]          # JSON list roundtrip
        assert v["checkpoint_path"] == "/ckpt/x.safetensors"

    def test_validation_defaults_skip(self, db):
        eid = _exp(db)
        db.insert_validation(eid, "run-x", {"clip_skipped": True, "skip_reason": "no clip"})
        v = db.get_validations("run-x")[eid]
        assert v["verdict"] == "SKIP" and v["clip_skipped"] == 1
        assert v["weight_errors"] == []
