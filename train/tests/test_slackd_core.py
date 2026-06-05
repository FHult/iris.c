"""
train/tests/test_slackd_core.py — Slack command daemon policy core (QL-7).

Covers the entire security surface with zero network and zero Slack SDK:
  - authorize: channel + user allow-list, fail-closed.
  - parse: mention stripping, keyword lowercasing, confirm-token extraction.
  - resolve: read-only RUN, armed gate (both directions), destructive
    confirm-token flow (issue / confirm / wrong / expired / replay / unarmed),
    unknown + help.
  - RateLimiter: token bucket trips and refills (injected clock).
  - audit_record shape.
  - Property: no message text can ever produce an argv outside COMMANDS verbatim.

Flywheel-safe: pure stdlib; no GPU, no live DB, no SDK import.
"""

from __future__ import annotations

import random
import string
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import slackd_core as sc
from slackd_core import (
    COMMANDS, ConfirmState, RateLimiter, authorize, audit_record, parse, resolve,
)

CH = "C123"
USER = "Ume"
USERS = {"Ume", "Uother"}


def _resolve(text, *, armed=False, pending=None, now=0.0, token="abcdef"):
    return resolve(parse(text), armed=armed, pending_confirm=pending, now=now,
                   make_token=lambda: token)


# ---------------------------------------------------------------------------
# authorize
# ---------------------------------------------------------------------------

class TestAuthorize:
    def test_ok(self):
        a = authorize(CH, USER, allow_channel=CH, allow_users=USERS)
        assert a.ok and a.reason == "ok"

    def test_wrong_channel(self):
        a = authorize("Cnope", USER, allow_channel=CH, allow_users=USERS)
        assert not a.ok and a.reason == "wrong_channel"

    def test_wrong_user(self):
        a = authorize(CH, "Uintruder", allow_channel=CH, allow_users=USERS)
        assert not a.ok and a.reason == "wrong_user"

    def test_fail_closed_no_channel(self):
        assert not authorize(CH, USER, allow_channel="", allow_users=USERS).ok
        assert not authorize(CH, USER, allow_channel=None, allow_users=USERS).ok

    def test_fail_closed_empty_userlist(self):
        assert not authorize(CH, USER, allow_channel=CH, allow_users=set()).ok
        assert not authorize(CH, USER, allow_channel=CH, allow_users=None).ok


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

class TestParse:
    def test_basic_keyword(self):
        p = parse("status")
        assert p.keyword == "status" and p.rest == "" and p.confirm_token is None

    def test_lowercases_keyword(self):
        assert parse("STATUS").keyword == "status"
        assert parse("Doctor").keyword == "doctor"

    def test_strips_bot_mention(self):
        assert parse("<@U0BOT> status").keyword == "status"

    def test_strips_multiple_angle_tokens(self):
        p = parse("<@U0BOT> <!here> doctor")
        assert p.keyword == "doctor"

    def test_confirm_token_extracted(self):
        p = parse("confirm a1b2c3")
        assert p.keyword == "confirm" and p.confirm_token == "a1b2c3"

    def test_confirm_without_token(self):
        p = parse("confirm")
        assert p.keyword == "confirm" and p.confirm_token is None

    def test_empty(self):
        p = parse("   ")
        assert p.keyword == "" and p.confirm_token is None

    def test_non_confirm_has_no_token(self):
        assert parse("stop now").confirm_token is None


# ---------------------------------------------------------------------------
# resolve — read-only
# ---------------------------------------------------------------------------

class TestResolveReadOnly:
    @pytest.mark.parametrize("kw", ["status", "doctor", "quality", "flywheel"])
    def test_read_only_runs_unarmed(self, kw):
        d = _resolve(kw, armed=False)
        assert d.action == sc.RUN
        assert d.argv == COMMANDS[kw].argv          # exact compiled argv
        assert d.audit_reason == f"run:{kw}"

    def test_unknown(self):
        d = _resolve("deploy --prod")
        assert d.action == sc.UNKNOWN and d.argv is None
        assert d.audit_reason == "rejected: unknown_command"

    def test_help(self):
        d = _resolve("help")
        assert d.action == sc.HELP and d.argv is None
        assert "status" in d.reply and "stop" in d.reply

    def test_empty_is_help(self):
        assert _resolve("").action == sc.HELP


# ---------------------------------------------------------------------------
# resolve — armed gate (pause / resume)
# ---------------------------------------------------------------------------

class TestResolveArmed:
    @pytest.mark.parametrize("kw", ["pause", "resume"])
    def test_refused_when_unarmed(self, kw):
        d = _resolve(kw, armed=False)
        assert d.action == sc.REFUSE_UNARMED and d.argv is None
        assert d.audit_reason == f"rejected: unarmed:{kw}"

    @pytest.mark.parametrize("kw", ["pause", "resume"])
    def test_runs_when_armed(self, kw):
        d = _resolve(kw, armed=True)
        assert d.action == sc.RUN and d.argv == COMMANDS[kw].argv
        assert d.audit_reason == f"run:{kw}"


# ---------------------------------------------------------------------------
# resolve — destructive confirm-token flow (stop)
# ---------------------------------------------------------------------------

class TestResolveConfirm:
    def test_stop_unarmed_refused(self):
        d = _resolve("stop", armed=False)
        assert d.action == sc.REFUSE_UNARMED and d.new_pending is None

    def test_stop_armed_issues_token(self):
        d = _resolve("stop", armed=True, now=100.0, token="deadbe")
        assert d.action == sc.NEED_CONFIRM and d.argv is None
        assert d.new_pending == ConfirmState("deadbe", "stop", 100.0 + sc.CONFIRM_TTL_SEC)
        assert "deadbe" in d.reply

    def test_confirm_happy_path(self):
        pend = ConfirmState("deadbe", "stop", expires_at=200.0)
        d = _resolve("confirm deadbe", armed=True, pending=pend, now=150.0)
        assert d.action == sc.CONFIRMED
        assert d.argv == COMMANDS["stop"].argv
        assert d.new_pending is None                # consumed

    def test_confirm_wrong_token(self):
        pend = ConfirmState("deadbe", "stop", expires_at=200.0)
        d = _resolve("confirm zzzzzz", armed=True, pending=pend, now=150.0)
        assert d.action == sc.REFUSE_CONFIRM
        assert d.new_pending is None                # consumed (one-shot)
        assert d.audit_reason == "rejected: confirm_bad_token"

    def test_confirm_expired(self):
        pend = ConfirmState("deadbe", "stop", expires_at=200.0)
        d = _resolve("confirm deadbe", armed=True, pending=pend, now=250.0)
        assert d.action == sc.REFUSE_CONFIRM
        assert d.audit_reason == "rejected: confirm_expired"

    def test_confirm_no_pending(self):
        d = _resolve("confirm deadbe", armed=True, pending=None, now=10.0)
        assert d.action == sc.REFUSE_CONFIRM
        assert d.audit_reason == "rejected: no_pending_confirm"

    def test_confirm_replay_after_consume(self):
        # first confirm consumes; daemon carries new_pending (None) forward
        pend = ConfirmState("deadbe", "stop", expires_at=200.0)
        first = _resolve("confirm deadbe", armed=True, pending=pend, now=150.0)
        assert first.action == sc.CONFIRMED
        second = _resolve("confirm deadbe", armed=True, pending=first.new_pending,
                          now=151.0)
        assert second.action == sc.REFUSE_CONFIRM   # token already gone

    def test_confirm_when_disarmed_after_issue(self):
        pend = ConfirmState("deadbe", "stop", expires_at=200.0)
        d = _resolve("confirm deadbe", armed=False, pending=pend, now=150.0)
        assert d.action == sc.REFUSE_UNARMED


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_allows_up_to_capacity(self):
        rl = RateLimiter(capacity=3, window=60.0)
        assert [rl.allow("u", 0.0) for _ in range(3)] == [True, True, True]
        assert rl.allow("u", 0.0) is False          # 4th in same instant denied

    def test_refills_over_time(self):
        rl = RateLimiter(capacity=2, window=60.0)
        assert rl.allow("u", 0.0) and rl.allow("u", 0.0)
        assert rl.allow("u", 0.0) is False
        # one token refills after window/capacity = 30s
        assert rl.allow("u", 30.0) is True

    def test_per_user_isolation(self):
        rl = RateLimiter(capacity=1, window=60.0)
        assert rl.allow("a", 0.0) is True
        assert rl.allow("a", 0.0) is False
        assert rl.allow("b", 0.0) is True           # b unaffected


# ---------------------------------------------------------------------------
# audit_record
# ---------------------------------------------------------------------------

class TestAuditRecord:
    def test_shape(self):
        r = audit_record("2026-06-05T00:00:00Z", USER, CH, "status", "run:status", 0, 12)
        assert set(r) == {"ts", "user", "channel", "raw", "matched",
                          "exit_code", "latency_ms"}
        assert r["matched"] == "run:status" and r["exit_code"] == 0

    def test_raw_truncated(self):
        r = audit_record("t", USER, CH, "x" * 9000, "rejected: unknown_command", None)
        assert len(r["raw"]) <= 500


# ---------------------------------------------------------------------------
# Property: no message can synthesise an out-of-table argv
# ---------------------------------------------------------------------------

class TestArgvContainment:
    def test_fuzz_argv_always_in_table_or_none(self):
        allowed = {c.argv for c in COMMANDS.values()}
        rng = random.Random(1234)
        alphabet = string.ascii_letters + string.digits + " -_/<>@|;&"
        for _ in range(2000):
            text = "".join(rng.choice(alphabet)
                           for _ in range(rng.randint(0, 24)))
            for armed in (False, True):
                d = resolve(parse(text), armed=armed, pending_confirm=None,
                            now=0.0, make_token=lambda: "tok123")
                assert d.argv is None or d.argv in allowed

    def test_confirm_only_yields_pending_keyword_argv(self):
        # Even with an injected pending state, a confirm can only run the exact
        # argv of the keyword that was pending — never anything else.
        allowed = {c.argv for c in COMMANDS.values()}
        pend = ConfirmState("tok123", "stop", expires_at=1e9)
        d = resolve(parse("confirm tok123"), armed=True, pending_confirm=pend,
                    now=0.0, make_token=lambda: "tok123")
        assert d.argv == COMMANDS["stop"].argv and d.argv in allowed
