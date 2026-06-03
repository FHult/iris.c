"""
train/monitoring — historical trend tracking + alerting for long campaigns (v3.21.0).

TrendStore: durable SQLite time-series (proxy fallback rate, unified-score
distribution, precompute speed, training loss, champion quality, disk/memory).
alerts: rule-driven evaluation (threshold/spike/drop) → console alerts the doctor
renders. Surfaced via `pipeline_doctor.py --monitor --history N`.
"""

from .trends import TrendStore
from .alerts import AlertRule, evaluate, rules_from_config, DEFAULT_RULES
from .sinks import slack_payload, dispatch_slack, post_to_slack

__all__ = ["TrendStore", "AlertRule", "evaluate", "rules_from_config",
           "DEFAULT_RULES", "slack_payload", "dispatch_slack", "post_to_slack"]
