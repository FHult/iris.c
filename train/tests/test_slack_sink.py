"""
train/tests/test_slack_sink.py — Slack alert sink (v3.21.0 QL-5).

Covers the pure payload formatting and the dispatch logic (severity filtering,
webhook resolution from an env var, no-op paths). The network POST is mocked via
the `poster` injection — no real webhook is ever hit, and no secret is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from monitoring.sinks import slack_payload, dispatch_slack, _resolve_webhook


def _alert(severity="WARNING", metric="proxy_fallback_rate",
           message="fallback high", detail="0.4 > 0.25", campaign=None):
    return {"severity": severity, "metric": metric, "message": message,
            "detail": detail, "campaign": campaign}


# ---------------------------------------------------------------------------
# Payload formatting
# ---------------------------------------------------------------------------

class TestSlackPayload:
    def test_has_text_fallback_and_blocks(self):
        p = slack_payload([_alert()])
        assert "text" in p and "blocks" in p
        assert p["blocks"][0]["type"] == "header"
        assert "1 alert" in p["blocks"][0]["text"]["text"]

    def test_header_reflects_worst_severity(self):
        p = slack_payload([_alert(severity="WARNING"), _alert(severity="CRITICAL")])
        # 🔴 = CRITICAL emoji must lead the header.
        assert p["blocks"][0]["text"]["text"].startswith("🔴")

    def test_one_section_per_alert(self):
        p = slack_payload([_alert(), _alert(), _alert()])
        sections = [b for b in p["blocks"] if b["type"] == "section"]
        assert len(sections) == 3

    def test_message_and_detail_in_section(self):
        p = slack_payload([_alert(message="disk low", detail="20 < 40")])
        body = p["blocks"][1]["text"]["text"]
        assert "disk low" in body and "20 < 40" in body

    def test_campaign_included_when_present(self):
        p = slack_payload([_alert(campaign="warmup-run1")])
        assert "warmup-run1" in p["blocks"][1]["text"]["text"]

    def test_context_in_header(self):
        p = slack_payload([_alert()], context="nightly check")
        assert "nightly check" in p["blocks"][0]["text"]["text"]

    def test_plural_grammar(self):
        one = slack_payload([_alert()])["blocks"][0]["text"]["text"]
        two = slack_payload([_alert(), _alert()])["blocks"][0]["text"]["text"]
        assert "1 alert" in one and "1 alerts" not in one
        assert "2 alerts" in two


# ---------------------------------------------------------------------------
# Dispatch (mocked poster — no network)
# ---------------------------------------------------------------------------

class _Poster:
    """Records calls; returns a configurable success value."""
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def __call__(self, url, payload, timeout=10.0):
        self.calls.append((url, payload))
        return self.ok


class TestDispatch:
    def test_sends_when_webhook_and_alerts(self):
        poster = _Poster(ok=True)
        res = dispatch_slack([_alert(severity="CRITICAL")],
                             webhook_url="https://hooks.slack.test/x", poster=poster)
        assert res["sent"] is True and res["n_sent"] == 1
        assert len(poster.calls) == 1
        assert poster.calls[0][0] == "https://hooks.slack.test/x"

    def test_no_webhook_is_noop(self):
        poster = _Poster()
        res = dispatch_slack([_alert()], cfg={}, poster=poster)   # no env, no url
        assert res["sent"] is False
        assert "webhook" in res["skipped"]
        assert poster.calls == []

    def test_min_severity_filters(self):
        poster = _Poster(ok=True)
        # min_severity WARNING → an INFO-only alert set sends nothing.
        res = dispatch_slack([_alert(severity="INFO")],
                             webhook_url="https://x", min_severity="WARNING",
                             poster=poster)
        assert res["sent"] is False
        assert "min_severity" in res["skipped"]
        assert poster.calls == []

    def test_critical_passes_warning_threshold(self):
        poster = _Poster(ok=True)
        res = dispatch_slack([_alert(severity="INFO"), _alert(severity="CRITICAL")],
                             webhook_url="https://x", min_severity="WARNING",
                             poster=poster)
        assert res["sent"] is True
        # Only the CRITICAL one qualifies (INFO filtered out).
        sent_payload = poster.calls[0][1]
        sections = [b for b in sent_payload["blocks"] if b["type"] == "section"]
        assert len(sections) == 1

    def test_post_failure_reported(self):
        poster = _Poster(ok=False)
        res = dispatch_slack([_alert(severity="CRITICAL")],
                             webhook_url="https://x", poster=poster)
        assert res["sent"] is False and res["skipped"] == "POST failed"

    def test_empty_alerts_noop(self):
        poster = _Poster()
        res = dispatch_slack([], webhook_url="https://x", poster=poster)
        assert res["sent"] is False and poster.calls == []


# ---------------------------------------------------------------------------
# Webhook resolution (env var, never config)
# ---------------------------------------------------------------------------

class TestWebhookResolution:
    def test_resolves_from_named_env(self, monkeypatch):
        monkeypatch.setenv("MY_HOOK", "https://hooks.slack.test/abc")
        cfg = {"monitoring": {"slack": {"webhook_env": "MY_HOOK"}}}
        assert _resolve_webhook(cfg) == "https://hooks.slack.test/abc"

    def test_defaults_to_slack_webhook_url(self, monkeypatch):
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/default")
        assert _resolve_webhook({}) == "https://hooks.slack.test/default"

    def test_missing_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        assert _resolve_webhook({}) is None
