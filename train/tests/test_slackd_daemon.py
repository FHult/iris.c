"""
train/tests/test_slackd_daemon.py — iris_slackd.Daemon transport wiring (QL-7).

The policy is exhaustively tested in test_slackd_core.py. Here we verify the thin
daemon shell honours it: a rejected message NEVER spawns; an accepted one spawns
the EXACT compiled argv (shell=False, fixed list); the armed gate and confirm
flow thread per-user state correctly; audit rows are written. The Slack SDK is
never imported (run_socket_mode is not exercised) — a fake runner stands in for
subprocess.

Flywheel-safe: tempdir audit log, injected clock + runner; no GPU, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import slackd_core as core
import iris_slackd as sd


def _cfg(tmp_path, *, armed=False, users="Uok", channel="C1"):
    env = {"SLACK_APP_TOKEN": "x", "SLACK_BOT_TOKEN": "x",
           "SLACK_CMD_CHANNEL": channel, "SLACK_CMD_USERS": users,
           "IRIS_SLACKD_LOG": str(tmp_path / "slackd.jsonl")}
    if armed:
        env["IRIS_SLACKD_ARMED"] = "1"
    return sd.Config(env)


class _Recorder:
    def __init__(self, exit_code=0, output="out"):
        self.calls = []
        self.exit_code, self.output = exit_code, output

    def __call__(self, argv):
        self.calls.append(argv)
        return self.exit_code, self.output


def _daemon(tmp_path, *, armed=False, t=0.0):
    rec = _Recorder()
    clock = {"t": t}
    d = sd.Daemon(_cfg(tmp_path, armed=armed), runner=rec,
                  clock=lambda: clock["t"])
    return d, rec, clock


def _audit_rows(tmp_path):
    p = tmp_path / "slackd.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Authorization gate
# ---------------------------------------------------------------------------

class TestAuthGate:
    def test_wrong_channel_ignored_no_spawn(self, tmp_path):
        d, rec, _ = _daemon(tmp_path)
        assert d.handle("Cwrong", "Uok", "status") is None
        assert rec.calls == []
        assert _audit_rows(tmp_path)[-1]["matched"] == "rejected: wrong_channel"

    def test_wrong_user_ignored_no_spawn(self, tmp_path):
        d, rec, _ = _daemon(tmp_path)
        assert d.handle("C1", "Uintruder", "status") is None
        assert rec.calls == []
        assert _audit_rows(tmp_path)[-1]["matched"] == "rejected: wrong_user"


# ---------------------------------------------------------------------------
# Read-only spawn
# ---------------------------------------------------------------------------

class TestReadOnly:
    @pytest.mark.parametrize("kw", ["status", "doctor", "quality", "flywheel"])
    def test_spawns_exact_argv(self, tmp_path, kw):
        d, rec, _ = _daemon(tmp_path)
        reply = d.handle("C1", "Uok", kw)
        assert rec.calls == [core.COMMANDS[kw].argv]
        assert "✅" in reply
        assert _audit_rows(tmp_path)[-1]["matched"] == f"run:{kw}"

    def test_unknown_no_spawn(self, tmp_path):
        d, rec, _ = _daemon(tmp_path)
        reply = d.handle("C1", "Uok", "rm -rf /")
        assert rec.calls == []
        assert "Unknown" in reply

    def test_nonzero_exit_surfaced(self, tmp_path):
        rec = _Recorder(exit_code=1, output="boom")
        d = sd.Daemon(_cfg(tmp_path), runner=rec, clock=lambda: 0.0)
        reply = d.handle("C1", "Uok", "status")
        assert "exit 1" in reply


# ---------------------------------------------------------------------------
# Armed gate
# ---------------------------------------------------------------------------

class TestArmedGate:
    def test_pause_unarmed_refused_no_spawn(self, tmp_path):
        d, rec, _ = _daemon(tmp_path, armed=False)
        reply = d.handle("C1", "Uok", "pause")
        assert rec.calls == []
        assert "armed" in reply.lower()
        assert _audit_rows(tmp_path)[-1]["matched"] == "rejected: unarmed:pause"

    def test_pause_armed_spawns(self, tmp_path):
        d, rec, _ = _daemon(tmp_path, armed=True)
        d.handle("C1", "Uok", "pause")
        assert rec.calls == [core.COMMANDS["pause"].argv]


# ---------------------------------------------------------------------------
# Confirm flow (stateful, per user)
# ---------------------------------------------------------------------------

class TestConfirmFlow:
    def test_stop_then_confirm_spawns(self, tmp_path):
        d, rec, clock = _daemon(tmp_path, armed=True)
        r1 = d.handle("C1", "Uok", "stop")
        assert rec.calls == []                       # no spawn yet
        # extract the issued token from the reply
        token = r1.split("confirm ")[1].split("`")[0].strip()
        r2 = d.handle("C1", "Uok", f"confirm {token}")
        assert rec.calls == [core.COMMANDS["stop"].argv]
        assert "✅" in r2                              # spawned -> command output

    def test_confirm_wrong_token_no_spawn(self, tmp_path):
        d, rec, _ = _daemon(tmp_path, armed=True)
        d.handle("C1", "Uok", "stop")
        r = d.handle("C1", "Uok", "confirm deadbeef")
        assert rec.calls == []
        assert "Invalid" in r

    def test_confirm_expired_no_spawn(self, tmp_path):
        d, rec, clock = _daemon(tmp_path, armed=True)
        r1 = d.handle("C1", "Uok", "stop")
        token = r1.split("confirm ")[1].split("`")[0].strip()
        clock["t"] = core.CONFIRM_TTL_SEC + 1.0       # advance past TTL
        r = d.handle("C1", "Uok", f"confirm {token}")
        assert rec.calls == []
        assert "expired" in r.lower()

    def test_stop_unarmed_no_token_issued(self, tmp_path):
        d, rec, _ = _daemon(tmp_path, armed=False)
        r = d.handle("C1", "Uok", "stop")
        assert rec.calls == []
        # a later confirm finds nothing pending
        r2 = d.handle("C1", "Uok", "confirm whatever")
        assert "Nothing to confirm" in r2


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    def test_trips_after_capacity(self, tmp_path):
        d, rec, _ = _daemon(tmp_path)            # default capacity 5
        for _ in range(5):
            assert d.handle("C1", "Uok", "status") is not None
        reply = d.handle("C1", "Uok", "status")
        assert "Rate limit" in reply
        assert len(rec.calls) == 5               # 6th never spawned
