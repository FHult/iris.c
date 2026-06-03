"""
train/tests/test_monitoring.py — trend store + alerts (v3.21.0 phase 2).

TrendStore (durable metric time-series) and the rule-driven alert evaluation
that surfaces in `pipeline_doctor.py --monitor`. These guard the signals that
make a long flywheel campaign trustworthy: fallback-rate creep, quality drops,
disk/memory.

Isolated: explicit db_path → tempdir; controlled timestamps for window queries.
Pure stdlib sqlite3, no GPU.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from monitoring.trends import TrendStore
from monitoring.alerts import AlertRule, evaluate, rules_from_config, DEFAULT_RULES


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(timespec="seconds")


@pytest.fixture
def store(tmp_path):
    return TrendStore(db_path=tmp_path / "mon.db")


# ---------------------------------------------------------------------------
# TrendStore
# ---------------------------------------------------------------------------

class TestTrendStore:
    def test_record_and_latest(self, store):
        store.record("proxy_fallback_rate", 0.03, ts=_iso(2))
        store.record("proxy_fallback_rate", 0.05, ts=_iso(1))
        latest = store.latest("proxy_fallback_rate")
        assert latest["value"] == 0.05

    def test_history_window_filters_old(self, store):
        store.record("m", 1.0, ts=_iso(40))     # outside 30d window
        store.record("m", 2.0, ts=_iso(10))     # inside
        store.record("m", 3.0, ts=_iso(1))      # inside
        vals = [r["value"] for r in store.history("m", since_days=30)]
        assert vals == [2.0, 3.0]

    def test_history_full_when_since_none(self, store):
        store.record("m", 1.0, ts=_iso(100))
        store.record("m", 2.0, ts=_iso(1))
        assert len(store.history("m", since_days=None)) == 2

    def test_campaign_filter(self, store):
        store.record("m", 1.0, campaign="a", ts=_iso(1))
        store.record("m", 2.0, campaign="b", ts=_iso(1))
        assert [r["value"] for r in store.history("m", campaign="a")] == [1.0]
        assert len(store.history("m")) == 2          # None = all campaigns

    def test_summary_stats_and_slope(self, store):
        for i, v in enumerate([0.1, 0.2, 0.3, 0.4]):
            store.record("m", v, ts=_iso(10 - i))    # ascending over time
        s = store.summary("m", since_days=30)
        assert s["n"] == 4
        assert s["min"] == 0.1 and s["max"] == 0.4
        assert s["first"] == 0.1 and s["last"] == 0.4
        assert s["slope"] > 0                         # rising trend

    def test_summary_empty_window(self, store):
        store.record("m", 1.0, ts=_iso(100))
        s = store.summary("m", since_days=30)
        assert s["n"] == 0 and s["mean"] is None and s["slope"] is None

    def test_metrics_list(self, store):
        store.record("a", 1.0); store.record("b", 2.0); store.record("a", 3.0)
        assert store.metrics() == ["a", "b"]

    def test_none_value_excluded_from_summary(self, store):
        store.record("m", None, ts=_iso(2))
        store.record("m", 0.5, ts=_iso(1))
        s = store.summary("m", since_days=30)
        assert s["n"] == 1 and s["mean"] == 0.5


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class TestAlerts:
    def test_threshold_high_fires(self, store):
        store.record("proxy_fallback_rate", 0.40, ts=_iso(0))   # > 0.25 default
        alerts = evaluate(store, [AlertRule("proxy_fallback_rate", "threshold_high",
                                            threshold=0.25)])
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "WARNING"
        assert alerts[0]["metric"] == "proxy_fallback_rate"

    def test_threshold_high_silent_when_ok(self, store):
        store.record("proxy_fallback_rate", 0.05, ts=_iso(0))
        assert evaluate(store, [AlertRule("proxy_fallback_rate", "threshold_high",
                                          threshold=0.25)]) == []

    def test_threshold_low_fires(self, store):
        store.record("disk_free_gb", 20, ts=_iso(0))
        alerts = evaluate(store, [AlertRule("disk_free_gb", "threshold_low",
                                            threshold=40, severity="CRITICAL")])
        assert alerts and alerts[0]["severity"] == "CRITICAL"

    def test_spike_against_baseline(self, store):
        # Baseline ~0.05; latest 0.20 > mean*(1+1.0) and > abs floor 0.10.
        for d in (14, 12, 10, 8):
            store.record("proxy_fallback_rate", 0.05, ts=_iso(d))
        store.record("proxy_fallback_rate", 0.20, ts=_iso(0))
        alerts = evaluate(store, [AlertRule("proxy_fallback_rate", "spike",
                                            rel=1.0, threshold=0.10, window_days=14)])
        assert len(alerts) == 1 and alerts[0]["kind"] == "spike"

    def test_spike_needs_min_history(self, store):
        # Only 2 points → spike rule needs ≥3, stays silent.
        store.record("m", 0.05, ts=_iso(5))
        store.record("m", 0.50, ts=_iso(0))
        assert evaluate(store, [AlertRule("m", "spike", rel=1.0, threshold=0.1)]) == []

    def test_quality_drop_fires(self, store):
        # Recent best 0.74; latest 0.66 < 0.74*(1-0.05)=0.703 → drop.
        for v, d in [(0.70, 20), (0.74, 10), (0.72, 5)]:
            store.record("champion_clip_i", v, ts=_iso(d))
        store.record("champion_clip_i", 0.66, ts=_iso(0))
        alerts = evaluate(store, [AlertRule("champion_clip_i", "drop", rel=0.05,
                                            window_days=60)])
        assert len(alerts) == 1 and alerts[0]["kind"] == "drop"

    def test_quality_drop_silent_when_stable(self, store):
        for v, d in [(0.70, 20), (0.72, 10), (0.71, 5), (0.72, 0)]:
            store.record("champion_clip_i", v, ts=_iso(d))
        assert evaluate(store, [AlertRule("champion_clip_i", "drop", rel=0.05)]) == []

    def test_no_data_no_alert(self, store):
        assert evaluate(store, DEFAULT_RULES) == []

    def test_severity_ordering(self, store):
        store.record("disk_free_gb", 10, ts=_iso(0))            # CRITICAL
        store.record("proxy_fallback_rate", 0.40, ts=_iso(0))   # WARNING
        alerts = evaluate(store)                                # default rules
        assert alerts[0]["severity"] == "CRITICAL"

    def test_rules_from_config_override(self):
        cfg = {"monitoring": {"alerts": [
            {"metric": "x", "kind": "threshold_high", "threshold": 1.5,
             "severity": "CRITICAL"}]}}
        rules = rules_from_config(cfg)
        assert len(rules) == 1
        assert rules[0].metric == "x" and rules[0].severity == "CRITICAL"

    def test_rules_from_config_defaults_when_empty(self):
        assert rules_from_config({}) is DEFAULT_RULES
