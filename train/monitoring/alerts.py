"""
train/monitoring/alerts.py — threshold + trend alerts over the trend store (v3.21.0).

Rule-driven evaluation of the metric history. Each rule inspects a metric's
recent window and emits an alert when a condition is met. Today alerts are
console-only (returned as dicts the doctor renders); the structure is ready for
email/Slack sinks later without changing the rules.

Rule kinds:
  threshold_high  — latest value > threshold
  threshold_low   — latest value < threshold
  spike           — latest value > window mean * (1 + rel) AND > abs_floor
  drop            — latest value < recent best * (1 - rel)  [quality regression]

Pure: reads a TrendStore, returns a list of alert dicts. No GPU, no I/O sinks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .trends import TrendStore


@dataclass
class AlertRule:
    metric: str
    kind: str                       # threshold_high | threshold_low | spike | drop
    threshold: float = 0.0          # for threshold_* ; abs_floor for spike
    rel: float = 0.5                # relative fraction for spike/drop
    window_days: float = 30
    severity: str = "WARNING"       # WARNING | CRITICAL | INFO
    campaign: Optional[str] = None  # None = global series
    message: str = ""               # optional override


# Sensible defaults for the v3.21.0 signals. Operators override via config.
DEFAULT_RULES = [
    AlertRule("proxy_fallback_rate", "threshold_high", threshold=0.25,
              severity="WARNING",
              message="Proxy VAE fallback rate is high — the proxy is being "
                      "bypassed often; check the latest golden eval / consider "
                      "retraining or high_fidelity mode."),
    AlertRule("proxy_fallback_rate", "spike", rel=1.0, threshold=0.10,
              window_days=14, severity="WARNING",
              message="Proxy VAE fallback rate spiked vs its recent baseline."),
    AlertRule("champion_clip_i", "drop", rel=0.05, window_days=60,
              severity="WARNING",
              message="Champion golden CLIP-I dropped >5% below its recent best "
                      "— a campaign may be regressing quality."),
    AlertRule("disk_free_gb", "threshold_low", threshold=40,
              severity="CRITICAL",
              message="Hot-volume free space below 40 GB."),
    AlertRule("mem_available_gb", "threshold_low", threshold=3.0,
              severity="WARNING",
              message="Available memory below 3 GB — OOM/jetsam risk."),
]


def _eval_rule(store: TrendStore, rule: AlertRule) -> Optional[dict]:
    latest = store.latest(rule.metric, rule.campaign)
    if not latest or latest.get("value") is None:
        return None
    val = latest["value"]

    triggered = False
    detail = ""
    if rule.kind == "threshold_high":
        triggered = val > rule.threshold
        detail = f"{val:.4g} > {rule.threshold:g}"
    elif rule.kind == "threshold_low":
        triggered = val < rule.threshold
        detail = f"{val:.4g} < {rule.threshold:g}"
    elif rule.kind == "spike":
        s = store.summary(rule.metric, rule.window_days, rule.campaign)
        if s["n"] >= 3 and s["mean"] is not None:
            baseline = s["mean"] * (1 + rule.rel)
            triggered = val > baseline and val > rule.threshold
            detail = f"{val:.4g} > baseline {baseline:.4g} (mean {s['mean']:.4g}×{1+rule.rel:g})"
    elif rule.kind == "drop":
        s = store.summary(rule.metric, rule.window_days, rule.campaign)
        if s["n"] >= 3 and s["max"] is not None:
            floor = s["max"] * (1 - rule.rel)
            triggered = val < floor
            detail = f"{val:.4g} < {floor:.4g} (recent best {s['max']:.4g} ×{1-rule.rel:g})"

    if not triggered:
        return None
    return {
        "severity": rule.severity,
        "metric": rule.metric,
        "kind": rule.kind,
        "campaign": rule.campaign,
        "value": val,
        "detail": detail,
        "message": rule.message or f"{rule.metric} {rule.kind} ({detail})",
    }


def evaluate(store: TrendStore, rules: Optional[list] = None) -> list[dict]:
    """Evaluate all rules; return the list of triggered alerts (most severe first)."""
    rules = rules if rules is not None else DEFAULT_RULES
    alerts = [a for a in (_eval_rule(store, r) for r in rules) if a is not None]
    order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    alerts.sort(key=lambda a: order.get(a["severity"], 9))
    return alerts


def rules_from_config(cfg: dict) -> list:
    """Build AlertRule list from a pipeline-config monitoring.alerts block.

    Falls back to DEFAULT_RULES when no alerts are configured. Each entry:
        {metric, kind, threshold?, rel?, window_days?, severity?, campaign?, message?}
    """
    mon = (cfg or {}).get("monitoring", {}) or {}
    entries = mon.get("alerts")
    if not entries:
        return DEFAULT_RULES
    out = []
    for e in entries:
        out.append(AlertRule(
            metric=e["metric"], kind=e["kind"],
            threshold=float(e.get("threshold", 0.0)),
            rel=float(e.get("rel", 0.5)),
            window_days=float(e.get("window_days", 30)),
            severity=e.get("severity", "WARNING"),
            campaign=e.get("campaign"),
            message=e.get("message", ""),
        ))
    return out
