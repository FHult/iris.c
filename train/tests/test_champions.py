"""
test_champions.py — cross-campaign champion read-out (debug/champions.py).

Guards the three honesty rails: era tagging across the cond_gap convention
boundary, ephemeral (smoke/test) filtering, and champion = max-cond_gap
done-iteration with source-mix reconstruction.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "champions", Path(__file__).parent.parent.parent / "debug" / "champions.py")
champions_mod = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
_spec.loader.exec_module(champions_mod)


class TestEra:
    def test_boundary(self):
        assert champions_mod.era_of("2026-06-11T00:00:00") == "held-out-EMA"
        assert champions_mod.era_of("2026-06-10T23:59:59") == "train-batch"
        assert champions_mod.era_of(None) == "train-batch"

    def test_ephemeral(self):
        assert champions_mod.is_ephemeral("smoke-style2")
        assert champions_mod.is_ephemeral("test-x")
        assert not champions_mod.is_ephemeral("warmup-run5")


@pytest.fixture
def fh(tmp_path, monkeypatch):
    # synthetic shard_scores.db (sources only) + flywheel_history.db
    sdb = tmp_path / "shard_scores.db"
    s = sqlite3.connect(sdb)
    s.execute("CREATE TABLE shards (shard_id TEXT, source TEXT)")
    s.executemany("INSERT INTO shards VALUES (?,?)",
                  [("a", "coyo"), ("b", "journeydb"), ("c", "journeydb")])
    s.commit(); s.close()
    monkeypatch.setattr(champions_mod, "SHARD_DB", sdb)

    fdb = tmp_path / "flywheel_history.db"
    f = sqlite3.connect(fdb)
    f.execute("""CREATE TABLE iterations (id INTEGER PRIMARY KEY, flywheel_name TEXT,
        iteration INT, selected_shards TEXT, hyperparams TEXT, train_loss REAL,
        ref_gap REAL, cond_gap REAL, status TEXT, checkpoint TEXT,
        checkpoint_hash TEXT, git_commit TEXT, n_first_contact INT, ts_start TEXT)""")
    f.execute("""CREATE TABLE campaign_summary (flywheel_name TEXT PRIMARY KEY,
        ts_first TEXT, config_path TEXT, status TEXT)""")
    def it(name, i, cg, shards, status="done"):
        f.execute("INSERT INTO iterations (flywheel_name,iteration,selected_shards,"
                  "hyperparams,cond_gap,ref_gap,status,ts_start,n_first_contact) "
                  "VALUES (?,?,?,?,?,?,?,?,?)",
                  (name, i, json.dumps(shards), json.dumps({"cross_ref_prob": 0.5}),
                   cg, 0.0, status, "2026-06-12T00:00:00", 0))
    # run5: champion is iter2 (cg 0.07 > iter1 0.05); also a 'running' iter ignored
    it("warmup-run5", 1, 0.05, ["a", "b"]); it("warmup-run5", 2, 0.07, ["a", "b", "c"])
    it("warmup-run5", 3, 0.9, ["a"], status="running")
    # old-era campaign
    f.execute("INSERT INTO campaign_summary VALUES ('warmup-run2','2026-06-05T00:00:00',null,'completed')")
    f.execute("INSERT INTO campaign_summary VALUES ('warmup-run5','2026-06-11T00:00:00',null,'active')")
    it("warmup-run2", 1, 0.03, ["b", "c"])
    # ephemeral
    it("smoke-x", 1, -5.0, ["a"])
    f.execute("INSERT INTO campaign_summary VALUES ('smoke-x','2026-06-11T00:00:00',null,'completed')")
    f.commit()
    return f


class TestChampions:
    def test_champion_is_max_cond_gap_done(self, fh):
        champs = {c["campaign"]: c for c in champions_mod.champions(fh)}
        assert champs["warmup-run5"]["iteration"] == 2          # 0.07 > 0.05; running ignored
        assert champs["warmup-run5"]["cond_gap"] == 0.07

    def test_source_mix_reconstructed(self, fh):
        c = next(x for x in champions_mod.champions(fh) if x["campaign"] == "warmup-run5")
        assert c["source_mix"] == {"coyo": 1, "journeydb": 2}   # a=coyo, b+c=journeydb

    def test_ephemeral_filtered_by_default(self, fh):
        names = {c["campaign"] for c in champions_mod.champions(fh)}
        assert "smoke-x" not in names
        assert names == {"warmup-run5", "warmup-run2"}

    def test_include_ephemeral(self, fh):
        names = {c["campaign"] for c in champions_mod.champions(fh, include_ephemeral=True)}
        assert "smoke-x" in names

    def test_era_tagging(self, fh):
        champs = {c["campaign"]: c for c in champions_mod.champions(fh)}
        assert champs["warmup-run5"]["era"] == "held-out-EMA"
        assert champs["warmup-run2"]["era"] == "train-batch"
