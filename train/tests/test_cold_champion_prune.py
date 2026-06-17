"""
train/tests/test_cold_champion_prune.py — cold champion archive keep-N pruning.

COLD-CHAMPION-PRUNE-1: _archive_flywheel_champion copies the champion into a
date-keyed cold dir (flywheel-{name}-{YYYYMMDD}). Same-day runs overwrite, but
across days these ~2-4 GB dirs accumulated unbounded over a long campaign. After
a successful archive the function now keeps only the newest `keep_cold_champions`
dirs for that flywheel name. These guard: dirs beyond the limit are pruned, dirs
within the limit are preserved, and dirs for other flywheel names are untouched.

Pure I/O — orchestrator imports without the MLX/Metal runtime, so this runs in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# conftest.py puts train/ on sys.path; the orchestrator lives under scripts/.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import orchestrator as orch


class _FakeDB:
    """Minimal stand-in for FlywheelDB.get_best()."""

    def __init__(self, checkpoint: str, iteration: int = 5, cond_gap: float = 0.1):
        self._best = {"checkpoint": checkpoint, "iteration": iteration,
                      "cond_gap": cond_gap}

    def get_best(self, name):  # noqa: ARG002 — signature parity
        return self._best


def _setup(tmp_path, name="run-x", keep=3):
    """Create a source checkpoint + cold_root, and return (fw_db, fw_cfg, weights_dir)."""
    step_src = tmp_path / "step.safetensors"
    step_src.write_text("weights")

    cold_root = tmp_path / "cold"
    weights = cold_root / "weights"
    weights.mkdir(parents=True)

    fw_cfg = {
        "storage": {"cold_root": str(cold_root)},
        "keep_cold_champions": keep,
    }
    return _FakeDB(str(step_src)), fw_cfg, weights


def test_prunes_dirs_beyond_keep_limit(tmp_path):
    name = "run-x"
    fw_db, fw_cfg, weights = _setup(tmp_path, name=name, keep=3)

    # Five pre-existing date-keyed champion dirs (YYYYMMDD sorts chronologically).
    for day in ("20200101", "20200102", "20200103", "20200104", "20200105"):
        (weights / f"flywheel-{name}-{day}").mkdir()

    cold_dir = orch._archive_flywheel_champion(name, fw_db, fw_cfg)
    assert cold_dir is not None  # archive succeeded

    remaining = sorted(d.name for d in weights.glob(f"flywheel-{name}-*"))
    # 5 old + 1 freshly-archived (today, > 2020) = 6 → keep newest 3.
    assert len(remaining) == 3
    # The two oldest within-2020 dirs survive alongside today's; 01/02/03 pruned.
    assert "flywheel-run-x-20200101" not in remaining
    assert "flywheel-run-x-20200102" not in remaining
    assert "flywheel-run-x-20200103" not in remaining
    assert "flywheel-run-x-20200104" in remaining
    assert "flywheel-run-x-20200105" in remaining
    assert Path(cold_dir).name in remaining  # today's dir is newest, kept


def test_preserves_dirs_within_limit(tmp_path):
    name = "run-x"
    fw_db, fw_cfg, weights = _setup(tmp_path, name=name, keep=5)

    for day in ("20200101", "20200102"):
        (weights / f"flywheel-{name}-{day}").mkdir()

    orch._archive_flywheel_champion(name, fw_db, fw_cfg)

    remaining = sorted(d.name for d in weights.glob(f"flywheel-{name}-*"))
    # 2 old + 1 new = 3 ≤ keep(5) → nothing pruned.
    assert "flywheel-run-x-20200101" in remaining
    assert "flywheel-run-x-20200102" in remaining
    assert len(remaining) == 3


def test_other_flywheel_names_untouched(tmp_path):
    name = "run-x"
    fw_db, fw_cfg, weights = _setup(tmp_path, name=name, keep=1)

    # Old dirs for THIS campaign (will be pruned down to keep=1)...
    for day in ("20200101", "20200102", "20200103"):
        (weights / f"flywheel-{name}-{day}").mkdir()
    # ...and dirs for an unrelated campaign that must never be touched.
    for day in ("20200101", "20200102", "20200103"):
        (weights / f"flywheel-other-run-{day}").mkdir()

    orch._archive_flywheel_champion(name, fw_db, fw_cfg)

    other = sorted(d.name for d in weights.glob("flywheel-other-run-*"))
    assert other == ["flywheel-other-run-20200101",
                     "flywheel-other-run-20200102",
                     "flywheel-other-run-20200103"]
    # run-x pruned to keep=1 (today's dir only).
    assert len(list(weights.glob(f"flywheel-{name}-*"))) == 1
