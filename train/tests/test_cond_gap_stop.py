"""
train/tests/test_cond_gap_stop.py — held-out cond_gap selection + early-stop logic (PROD-2).

Pins the pure decision logic the trainer's T-05 eval loop will use: select on cond_gap
(never train_loss), early-stop on a cond_gap plateau, and detect the over-training
signature (cond_gap down while train_loss down). Hermetic, no I/O.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ip_adapter import cond_gap_stop as cg
from ip_adapter.cond_gap_stop import CondGapPoint as P


class TestBestPoint:
    def test_empty(self):
        assert cg.best_point([]) is None

    def test_picks_max_cond_gap(self):
        h = [P(1000, 0.01), P(2000, 0.03), P(3000, 0.02)]
        assert cg.best_point(h).step == 2000

    def test_earliest_on_tie(self):
        # equal cond_gap → prefer the earlier (less-trained) checkpoint
        h = [P(1000, 0.03), P(2000, 0.03)]
        assert cg.best_point(h).step == 1000

    def test_evals_since_best(self):
        h = [P(1000, 0.05), P(2000, 0.04), P(3000, 0.03)]
        assert cg.evals_since_best(h) == 2
        assert cg.evals_since_best([P(1, 0.0), P(2, 0.1)]) == 0


class TestShouldStop:
    def test_improving_does_not_stop(self):
        h = [P(i * 1000, 0.01 * i) for i in range(1, 6)]   # strictly rising
        assert cg.should_stop(h, patience=3) is False

    def test_too_few_evals(self):
        h = [P(1000, 0.05), P(2000, 0.04), P(3000, 0.03)]  # 3 < patience(3)+1
        assert cg.should_stop(h, patience=3) is False

    def test_plateau_stops(self):
        # peak at eval 1, then 3 evals with no new high → stop
        h = [P(1000, 0.05), P(2000, 0.04), P(3000, 0.03), P(4000, 0.02)]
        assert cg.should_stop(h, patience=3) is True

    def test_recent_high_resets(self):
        # a new high at the last eval → not stopped
        h = [P(1000, 0.05), P(2000, 0.04), P(3000, 0.03), P(4000, 0.06)]
        assert cg.should_stop(h, patience=3) is False

    def test_min_delta_ignores_tiny_gains(self):
        # tiny improvements below min_delta don't count as new highs → stop
        h = [P(1000, 0.050), P(2000, 0.0501), P(3000, 0.0502), P(4000, 0.0503)]
        assert cg.should_stop(h, patience=3, min_delta=0.01) is True
        assert cg.should_stop(h, patience=3, min_delta=0.0) is False  # they do count


class TestIsOvertraining:
    def test_decline_with_falling_loss(self):
        # the warmup-run2 numbers: cond_gap down, train_loss down → over-training
        h = [P(1000, 0.0273, 1.0043), P(2000, -0.0054, 0.5328), P(3000, -0.0275, 0.3877)]
        assert cg.is_overtraining(h) is True

    def test_decline_but_loss_rising_is_not(self):
        h = [P(1000, 0.03, 0.4), P(2000, -0.005, 0.5), P(3000, -0.027, 0.6)]
        assert cg.is_overtraining(h) is False

    def test_needs_train_loss(self):
        h = [P(1000, 0.03), P(2000, -0.005), P(3000, -0.027)]  # no train_loss
        assert cg.is_overtraining(h) is False

    def test_too_few_points(self):
        assert cg.is_overtraining([P(1, 0.03, 1.0), P(2, 0.01, 0.5)], window=3) is False

    def test_rising_cond_gap_is_not(self):
        h = [P(1000, 0.01, 0.5), P(2000, 0.02, 0.4), P(3000, 0.03, 0.3)]
        assert cg.is_overtraining(h) is False
