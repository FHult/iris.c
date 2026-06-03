"""
train/monitoring/sinks.py — alert delivery sinks (v3.21.0 QL-5).

Today alerts are returned as dicts and rendered to the console by the doctor.
This adds a Slack incoming-webhook sink so long unattended campaigns can page you.

Security: the webhook URL is a secret and is NEVER stored in the repo or config.
The config holds only the *name of the environment variable* that carries it
(default SLACK_WEBHOOK_URL); the operator exports it:

    export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/T…/B…/…'

Dependency-free: stdlib urllib only (no requests). The payload formatting is pure
and tested; the network POST is mockable and never hit in tests.
"""

from __future__ import annotations

import json
import os
from typing import Optional

_SEVERITY_EMOJI = {"CRITICAL": "🔴", "WARNING": "🟠", "INFO": "🔵"}
_SEVERITY_RANK = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}


# ---------------------------------------------------------------------------
# Payload formatting (pure)
# ---------------------------------------------------------------------------

def slack_payload(alerts: list[dict], context: str = "") -> dict:
    """Build a Slack incoming-webhook payload from a list of alert dicts.

    Returns {text, blocks}: `text` is the notification fallback; `blocks` render
    a header + one section per alert (severity emoji, message, detail).
    """
    n = len(alerts)
    worst = min((_SEVERITY_RANK.get(a.get("severity"), 9) for a in alerts), default=9)
    worst_emoji = next((e for s, e in _SEVERITY_EMOJI.items()
                        if _SEVERITY_RANK[s] == worst), "⚪")
    ctx = f" — {context}" if context else ""
    header = f"{worst_emoji} iris pipeline: {n} alert{'s' if n != 1 else ''}{ctx}"

    blocks = [{"type": "header",
               "text": {"type": "plain_text", "text": header[:150]}}]
    for a in alerts:
        emoji = _SEVERITY_EMOJI.get(a.get("severity"), "⚪")
        camp = f"  ·  campaign `{a['campaign']}`" if a.get("campaign") else ""
        line = (f"{emoji} *[{a.get('severity','?')}]* {a.get('message','')}"
                f"\n`{a.get('metric','?')}`  {a.get('detail','')}{camp}")
        blocks.append({"type": "section",
                       "text": {"type": "mrkdwn", "text": line[:2900]}})

    # Plain-text fallback for clients that don't render blocks.
    text_lines = [header] + [
        f"[{a.get('severity')}] {a.get('message')} ({a.get('detail')})"
        for a in alerts]
    return {"text": "\n".join(text_lines)[:3000], "blocks": blocks}


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

def post_to_slack(webhook_url: str, payload: dict, timeout: float = 10.0) -> bool:
    """POST a payload to a Slack incoming webhook. Returns True on HTTP 200.

    Network I/O is isolated here so callers/tests can mock it.
    """
    import urllib.request
    import urllib.error
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=data, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _resolve_webhook(cfg: dict) -> Optional[str]:
    """Resolve the webhook URL from the env var named in the config (never stored)."""
    slack = ((cfg or {}).get("monitoring", {}) or {}).get("slack", {}) or {}
    env_name = slack.get("webhook_env", "SLACK_WEBHOOK_URL")
    return os.environ.get(env_name) or None


def dispatch_slack(
    alerts: list[dict],
    cfg: Optional[dict] = None,
    webhook_url: Optional[str] = None,
    min_severity: str = "WARNING",
    context: str = "",
    poster=post_to_slack,
) -> dict:
    """Filter alerts by severity and send them to Slack.

    webhook_url overrides the config/env resolution (mainly for tests). Returns a
    result dict: {sent, n_sent, n_total, skipped}. A no-op (sent=False) when there
    are no qualifying alerts or no webhook is configured — never raises.
    """
    cfg = cfg or {}
    threshold = _SEVERITY_RANK.get(min_severity, 1)
    qualifying = [a for a in alerts
                  if _SEVERITY_RANK.get(a.get("severity"), 9) <= threshold]

    if not qualifying:
        return {"sent": False, "n_sent": 0, "n_total": len(alerts),
                "skipped": "no alerts at/above min_severity"}

    url = webhook_url or _resolve_webhook(cfg)
    if not url:
        return {"sent": False, "n_sent": 0, "n_total": len(alerts),
                "skipped": "no webhook URL (set the env var named in "
                           "monitoring.slack.webhook_env)"}

    ok = poster(url, slack_payload(qualifying, context))
    return {"sent": bool(ok), "n_sent": len(qualifying) if ok else 0,
            "n_total": len(alerts),
            "skipped": None if ok else "POST failed"}
