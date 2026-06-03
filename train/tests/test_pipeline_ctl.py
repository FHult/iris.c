"""train/tests/test_pipeline_ctl.py — pure cores in pipeline_ctl (GROK-TEST-2).

_restart_plan is the resume-from-N invariant: reset exactly chunks N..total,
never a chunk before N (those are complete — an off-by-one re-runs finished
chunks / wastes days of compute). Pure; no side effects.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import pipeline_ctl as ctl


def test_mid_resume_resets_n_to_total():
    p = ctl._restart_plan(2, 4)
    assert p["reset_chunks"] == [2, 3, 4]
    assert p["hard_ex_chunks"] == [2, 3, 4]
    assert p["restore_predecessor"] == 1


def test_resume_from_one_resets_all_and_no_restore():
    p = ctl._restart_plan(1, 4)
    assert p["reset_chunks"] == [1, 2, 3, 4]
    assert p["restore_predecessor"] is None


def test_resume_last_chunk_resets_only_it():
    p = ctl._restart_plan(4, 4)
    assert p["reset_chunks"] == [4]
    assert p["restore_predecessor"] == 3


def test_never_resets_a_chunk_before_n():
    # the safety invariant: nothing < N is ever in the reset set.
    p = ctl._restart_plan(3, 5)
    assert min(p["reset_chunks"]) == 3 and max(p["reset_chunks"]) == 5
