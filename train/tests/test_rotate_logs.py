"""train/tests/test_rotate_logs.py — unit tests for the GROK-T-13 log rotator.

Hermetic: operates entirely inside tmp_path, injects a fixed `now` so mtime-based
cutoffs are deterministic, and never touches the real LOG_DIR.
"""

from __future__ import annotations

import gzip
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import rotate_logs as rl


def _write(p: Path, text: str, age_days: float, now: float) -> Path:
    p.write_text(text)
    mt = now - age_days * 86400.0
    os.utime(p, (mt, mt))
    return p


def test_archives_only_old_logs(tmp_path):
    now = 1_000_000_000.0
    old = _write(tmp_path / "train_chunk1.log", "old\n", age_days=10, now=now)
    recent = _write(tmp_path / "train_chunk2.log", "recent\n", age_days=1, now=now)
    old_jsonl = _write(tmp_path / "orchestrator.jsonl", "{}\n", age_days=30, now=now)

    res = rl.rotate_logs(tmp_path, days=7, now=now)

    assert set(res["archived"]) == {"train_chunk1.log", "orchestrator.jsonl"}
    assert not old.exists()           # moved
    assert not old_jsonl.exists()
    assert recent.exists()            # left live
    arch = tmp_path / "archive"
    assert (arch / "train_chunk1.log.gz").exists()
    with gzip.open(arch / "train_chunk1.log.gz", "rt") as fh:
        assert fh.read() == "old\n"   # content preserved


def test_ignores_unrelated_suffixes_and_archive_dir(tmp_path):
    now = 1_000_000_000.0
    _write(tmp_path / "notes.txt", "x", age_days=99, now=now)        # wrong suffix
    (tmp_path / "archive").mkdir()
    _write(tmp_path / "archive" / "old.log.gz", "y", age_days=99, now=now)  # already archived

    res = rl.rotate_logs(tmp_path, days=7, now=now)

    assert res["archived"] == []
    assert (tmp_path / "notes.txt").exists()


def test_dry_run_changes_nothing(tmp_path):
    now = 1_000_000_000.0
    old = _write(tmp_path / "flywheel.log", "data\n", age_days=20, now=now)

    res = rl.rotate_logs(tmp_path, days=7, dry_run=True, now=now)

    assert res["archived"] == ["flywheel.log"]
    assert old.exists()                          # untouched
    assert not (tmp_path / "archive").exists()   # not even created


def test_max_archive_days_prunes_old_archives(tmp_path):
    now = 1_000_000_000.0
    arch = tmp_path / "archive"
    arch.mkdir()
    stale = _write(arch / "ancient.log.gz", "z", age_days=120, now=now)
    fresh = _write(arch / "newish.log.gz", "z", age_days=5, now=now)

    res = rl.rotate_logs(tmp_path, days=7, max_archive_days=90, now=now)

    assert res["pruned"] == ["ancient.log.gz"]
    assert not stale.exists()
    assert fresh.exists()


def test_missing_log_dir_is_safe(tmp_path):
    res = rl.rotate_logs(tmp_path / "does_not_exist", days=7, now=1_000_000_000.0)
    assert res["archived"] == [] and res["pruned"] == []


def test_name_collision_disambiguates(tmp_path):
    now = 1_000_000_000.0
    arch = tmp_path / "archive"
    arch.mkdir()
    (arch / "train_chunk1.log.gz").write_text("pre-existing")   # collides with the gz name
    _write(tmp_path / "train_chunk1.log", "new\n", age_days=10, now=now)

    res = rl.rotate_logs(tmp_path, days=7, now=now)

    assert res["archived"] == ["train_chunk1.log"]
    assert (arch / "train_chunk1.log.gz").read_text() == "pre-existing"  # untouched
    assert (arch / "train_chunk1.log.1.gz").exists()                     # new one beside it
