"""
train/tests/test_ablation_safety.py — ablation safety nets (ABL-1, ABL-2).

The TrialTimer (wallclock kill for a hung trainer) and the multi-signal
EarlyStopper (NaN/loss explosion, grad explosion, dead style signal, cond_gap
floor) are the only things standing between a stalled/diverging trial and a
frozen or GPU-wasting campaign — and they had zero test coverage (the ablation
harness was flagged at 0 tests by the testing-suite review).

Pure: the decision logic runs without a real subprocess. SIGTERM delivery is
verified against a tiny fake-process stub. No GPU, no real trainer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from ablation_harness import TrialTimer, EarlyStopper, CampaignPlateau


# ---------------------------------------------------------------------------
# Fake subprocess: records send_signal calls; poll() controls aliveness.
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, alive=True):
        self._alive = alive
        self.signals = []

    def poll(self):
        return None if self._alive else 0

    def send_signal(self, sig):
        self.signals.append(sig)
        self._alive = False        # SIGTERM ends it


# ---------------------------------------------------------------------------
# TrialTimer (ABL-1)
# ---------------------------------------------------------------------------

class TestTrialTimer:
    def test_fires_after_timeout(self):
        proc = _FakeProc(alive=True)
        t = TrialTimer(timeout_secs=0.05)
        t.start(proc)
        time.sleep(0.2)            # let the timer thread fire
        assert t.timed_out is True
        assert len(proc.signals) == 1     # exactly one SIGTERM

    def test_cancel_before_timeout_is_noop(self):
        proc = _FakeProc(alive=True)
        t = TrialTimer(timeout_secs=0.3)
        t.start(proc)
        t.cancel()                 # process finished naturally first
        time.sleep(0.4)
        assert t.timed_out is False
        assert proc.signals == []

    def test_no_signal_if_proc_already_done(self):
        proc = _FakeProc(alive=False)   # poll() != None → not alive
        t = TrialTimer(timeout_secs=0.05)
        t.start(proc)
        time.sleep(0.2)
        # Timer still "timed out", but must not signal a finished process.
        assert proc.signals == []


# ---------------------------------------------------------------------------
# EarlyStopper (ABL-2)
# ---------------------------------------------------------------------------

def _stopper(**kw):
    base = dict(min_cond_gap=-0.3, patience=4, min_snapshots=3)
    base.update(kw)
    return EarlyStopper(**base)


class TestEarlyStopperLossExplosion:
    def test_constant_high_loss_kills_on_first_snapshot(self):
        # The backlog's own ABL-2 success criterion: constant loss=9.0 → instant kill.
        es = _stopper(nan_loss_threshold=5.0)
        assert es.feed_snapshot({"loss_smooth": 9.0}) is True
        assert "loss=9" in es.trigger_reason

    def test_loss_below_threshold_does_not_kill(self):
        es = _stopper(nan_loss_threshold=5.0)
        assert es.feed_snapshot({"loss_smooth": 1.2}) is False

    def test_loss_explosion_bypasses_warmup(self):
        # No min_snapshots warmup for the instant-kill signal.
        es = _stopper(min_snapshots=10, nan_loss_threshold=5.0)
        assert es.feed_snapshot({"loss_smooth": 7.0}) is True


class TestEarlyStopperGradExplosion:
    def test_three_consecutive_high_grad_kills(self):
        es = _stopper(grad_norm_threshold=50.0)
        assert es.feed_snapshot({"grad_norm": 80.0, "loss_smooth": 1.0}) is False
        assert es.feed_snapshot({"grad_norm": 80.0, "loss_smooth": 1.0}) is False
        assert es.feed_snapshot({"grad_norm": 80.0, "loss_smooth": 1.0}) is True
        assert "grad_norm" in es.trigger_reason

    def test_grad_streak_resets_on_good_snapshot(self):
        es = _stopper(grad_norm_threshold=50.0)
        es.feed_snapshot({"grad_norm": 80.0, "loss_smooth": 1.0})
        es.feed_snapshot({"grad_norm": 80.0, "loss_smooth": 1.0})
        es.feed_snapshot({"grad_norm": 10.0, "loss_smooth": 1.0})   # resets streak
        assert es.feed_snapshot({"grad_norm": 80.0, "loss_smooth": 1.0}) is False

    def test_grad_disabled_when_threshold_none(self):
        es = _stopper(grad_norm_threshold=None)
        for _ in range(5):
            assert es.feed_snapshot({"grad_norm": 999.0, "loss_smooth": 1.0}) is False


class TestEarlyStopperCondGapFloor:
    def test_kills_after_patience_past_warmup(self):
        es = _stopper(min_cond_gap=-0.3, patience=4, min_snapshots=3)
        # First 3 snapshots are warmup (cond_gap floor not evaluated).
        for _ in range(3):
            assert es.feed_snapshot({"cond_gap": -0.9, "loss_smooth": 1.0}) is False
        # Now 4 consecutive bad snapshots past warmup → kill on the 4th.
        assert es.feed_snapshot({"cond_gap": -0.9, "loss_smooth": 1.0}) is False  # streak 1
        assert es.feed_snapshot({"cond_gap": -0.9, "loss_smooth": 1.0}) is False  # 2
        assert es.feed_snapshot({"cond_gap": -0.9, "loss_smooth": 1.0}) is False  # 3
        assert es.feed_snapshot({"cond_gap": -0.9, "loss_smooth": 1.0}) is True   # 4

    def test_good_cond_gap_never_triggers(self):
        es = _stopper(min_cond_gap=-0.3)
        for _ in range(20):
            assert es.feed_snapshot({"cond_gap": 0.2, "loss_smooth": 0.5}) is False


class TestEarlyStopperRefGap:
    def test_dead_style_signal_kills_after_patience(self):
        es = _stopper(ref_gap_min=-0.5, ref_gap_patience=3, min_snapshots=2)
        for _ in range(2):                 # warmup
            es.feed_snapshot({"ref_gap": -0.9, "loss_smooth": 1.0})
        assert es.feed_snapshot({"ref_gap": -0.9, "loss_smooth": 1.0}) is False  # 1
        assert es.feed_snapshot({"ref_gap": -0.9, "loss_smooth": 1.0}) is False  # 2
        assert es.feed_snapshot({"ref_gap": -0.9, "loss_smooth": 1.0}) is True   # 3
        assert "ref_gap" in es.trigger_reason


class TestEarlyStopperGeneral:
    def test_idempotent_once_triggered(self):
        es = _stopper(nan_loss_threshold=5.0)
        assert es.feed_snapshot({"loss_smooth": 9.0}) is True
        # Subsequent calls keep returning True without changing the reason.
        reason = es.trigger_reason
        assert es.feed_snapshot({"loss_smooth": 0.1}) is True
        assert es.trigger_reason == reason

    def test_sends_sigterm_when_attached(self):
        import signal
        proc = _FakeProc(alive=True)
        es = _stopper(nan_loss_threshold=5.0)
        es.attach(proc)
        es.feed_snapshot({"loss_smooth": 9.0})
        assert signal.SIGTERM in proc.signals

    def test_no_proc_attached_is_safe(self):
        # Decision logic must work even without a process to signal.
        es = _stopper(nan_loss_threshold=5.0)
        assert es.feed_snapshot({"loss_smooth": 9.0}) is True   # no crash


# ---------------------------------------------------------------------------
# CampaignPlateau — campaign-level "this search is played out" detector
# ---------------------------------------------------------------------------

class TestCampaignPlateau:
    def test_warmup_suppresses_early_trigger(self):
        # patience=2 would fire after 2 stale runs, but min_runs=5 holds it off.
        cp = CampaignPlateau(patience=2, min_delta=0.01, min_runs=5)
        # 4 flat runs: stale climbs but we're still under min_runs → no trigger.
        results = [cp.update(0.30) for _ in range(4)]
        assert results == [False, False, False, False]

    def test_plateau_fires_when_stale_reaches_patience_past_warmup(self):
        cp = CampaignPlateau(patience=3, min_delta=0.01, min_runs=3)
        assert cp.update(0.30) is False       # run 1: first best, stale 0
        assert cp.update(0.30) is False       # run 2: stale 1 (n_runs<min_runs)
        assert cp.update(0.30) is False       # run 3: stale 2 (n_runs==min_runs, 2<3)
        assert cp.update(0.30) is True        # run 4: stale 3 >= patience 3 → plateau
        assert cp.stale_count == 3

    def test_real_improvement_resets_stale(self):
        cp = CampaignPlateau(patience=3, min_delta=0.01, min_runs=2)
        cp.update(0.30)
        cp.update(0.30)        # stale 1
        cp.update(0.30)        # stale 2
        cp.update(0.50)        # clear improvement → stale resets to 0
        assert cp.stale_count == 0
        assert cp.best_score == 0.50

    def test_sub_min_delta_change_is_not_improvement(self):
        # An increase smaller than min_delta does NOT reset the stale counter.
        cp = CampaignPlateau(patience=5, min_delta=0.05, min_runs=2)
        cp.update(0.300)
        cp.update(0.310)       # +0.010 < min_delta 0.05 → stale, not improvement
        cp.update(0.320)       # +0.010 again → stale
        assert cp.stale_count == 2
        assert cp.best_score == 0.300

    def test_triggers_after_patience_consecutive_stale(self):
        cp = CampaignPlateau(patience=3, min_delta=0.01, min_runs=2)
        cp.update(0.50)                       # best
        assert cp.update(0.40) is False       # stale 1
        assert cp.update(0.40) is False       # stale 2
        assert cp.update(0.40) is True        # stale 3 == patience → plateau

    def test_none_scores_dont_corrupt_state(self):
        cp = CampaignPlateau(patience=2, min_delta=0.01, min_runs=1)
        cp.update(0.50)                       # best
        assert cp.update(None) is False       # crashed run — no effect on best/stale
        assert cp.best_score == 0.50
        assert cp.stale_count == 0

    def test_status_string(self):
        cp = CampaignPlateau(patience=4, min_delta=0.01, min_runs=1)
        assert cp.status() == "no data"
        cp.update(0.42)
        assert cp.status() == "best=0.420  stale=0/4"
