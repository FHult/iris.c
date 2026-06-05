#!/usr/bin/env python3
"""
train/scripts/iris_slackd.py — Slack command daemon (QL-7).

A thin Socket-Mode shell around slackd_core. ALL security policy lives in
slackd_core (pure, unit-tested); this file only wires the WebSocket transport to
`Daemon.handle()` and spawns the compiled-in argv. See
plans/slack-command-daemon.md and BACKLOG.md QL-7.

Security posture (enforced here + in slackd_core):
  - Socket Mode only: outbound WebSocket, no inbound port. Tokens from env.
  - Single channel + user allow-list (fail closed).
  - Fixed command table; no message text ever reaches a shell or argv slot.
  - Read-only by default; pause/resume need IRIS_SLACKD_ARMED=1; stop needs
    armed + a one-shot confirm token.
  - Per-user rate limit; every inbound message is audit-logged to slackd.jsonl.
  - subprocess.run([...], shell=False) only; the daemon never mutates pipeline
    state itself — it spawns existing locking-aware scripts.

Run (after exporting tokens — never commit them):
    export SLACK_APP_TOKEN=xapp-...     # connections:write
    export SLACK_BOT_TOKEN=xoxb-...     # chat:write, files:write, channels:history
    export SLACK_CMD_CHANNEL=C0XXXXXXX
    export SLACK_CMD_USERS=U0AAA,U0BBB
    # export IRIS_SLACKD_ARMED=1        # only if you want pause/resume/stop enabled
    train/.venv/bin/python train/scripts/iris_slackd.py

Self-test the full handle() pipeline with no SDK and no network:
    train/.venv/bin/python train/scripts/iris_slackd.py --self-test
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slackd_core as core

REPO_ROOT = Path(__file__).resolve().parents[2]

# Output caps for the reply posted back to the channel.
MAX_REPLY_LINES = 40
MAX_REPLY_CHARS = 3500
SPAWN_TIMEOUT_SEC = 180


# ---------------------------------------------------------------------------
# Config (env only; never repo/config)
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, env: Optional[dict] = None):
        env = env if env is not None else os.environ
        self.app_token = env.get("SLACK_APP_TOKEN", "")
        self.bot_token = env.get("SLACK_BOT_TOKEN", "")
        self.channel = env.get("SLACK_CMD_CHANNEL", "")
        users = env.get("SLACK_CMD_USERS", "")
        self.users = {u.strip() for u in users.split(",") if u.strip()}
        self.armed = env.get("IRIS_SLACKD_ARMED", "") == "1"
        self.audit_path = Path(
            env.get("IRIS_SLACKD_LOG", str(REPO_ROOT / "logs" / "slackd.jsonl")))
        self.rate_capacity = int(env.get("IRIS_SLACKD_RATE_N", "5"))
        self.rate_window = float(env.get("IRIS_SLACKD_RATE_WINDOW", "60"))

    def missing(self) -> list[str]:
        miss = []
        if not self.app_token:
            miss.append("SLACK_APP_TOKEN")
        if not self.bot_token:
            miss.append("SLACK_BOT_TOKEN")
        if not self.channel:
            miss.append("SLACK_CMD_CHANNEL")
        if not self.users:
            miss.append("SLACK_CMD_USERS")
        return miss


def _default_runner(argv: tuple[str, ...]) -> tuple[Optional[int], str]:
    """Spawn the compiled argv with no shell. Returns (exit_code, output_tail)."""
    try:
        p = subprocess.run(list(argv), shell=False, cwd=str(REPO_ROOT),
                           capture_output=True, text=True, timeout=SPAWN_TIMEOUT_SEC)
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, out
    except subprocess.TimeoutExpired:
        return None, f"(timed out after {SPAWN_TIMEOUT_SEC}s)"
    except Exception as e:  # spawn failure must never crash the daemon
        return None, f"(spawn error: {e})"


def _tail(text: str, max_lines=MAX_REPLY_LINES, max_chars=MAX_REPLY_CHARS) -> str:
    lines = (text or "").rstrip().splitlines()
    clipped = lines[-max_lines:]
    body = "\n".join(clipped)
    if len(body) > max_chars:
        body = body[-max_chars:]
    prefix = "…(truncated)…\n" if (len(lines) > max_lines or len(body) < len(text or "")) else ""
    return prefix + body


# ---------------------------------------------------------------------------
# Daemon (transport-free policy wiring; tested without the SDK)
# ---------------------------------------------------------------------------

class Daemon:
    def __init__(self, cfg: Config, *,
                 runner: Callable[[tuple[str, ...]], tuple[Optional[int], str]] = _default_runner,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self._runner = runner
        self._clock = clock
        self._rl = core.RateLimiter(cfg.rate_capacity, cfg.rate_window)
        self._pending: dict[str, core.ConfirmState] = {}   # user -> pending confirm
        cfg.audit_path.parent.mkdir(parents=True, exist_ok=True)

    # -- audit ----------------------------------------------------------------
    def _audit(self, user, channel, raw, matched, exit_code, latency_ms=None):
        rec = core.audit_record(_dt.datetime.now(_dt.timezone.utc).isoformat(),
                                user, channel, raw, matched, exit_code, latency_ms)
        try:
            with open(self.cfg.audit_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass  # auditing must never crash the daemon
        return rec

    # -- the one entry point the transport calls ------------------------------
    def handle(self, channel: str, user: str, text: str) -> Optional[str]:
        """Process one inbound message. Returns the reply string to post, or
        None if the message is silently ignored (wrong channel/user — we do not
        reply to unauthorized contexts, only audit them)."""
        now = self._clock()
        t0 = time.monotonic()

        authz = core.authorize(channel, user, allow_channel=self.cfg.channel,
                               allow_users=self.cfg.users)
        if not authz.ok:
            self._audit(user, channel, text, f"rejected: {authz.reason}", None)
            return None  # do not engage with unauthorized channel/user

        if not self._rl.allow(user, now):
            self._audit(user, channel, text, "rejected: rate_limited", None)
            return "Rate limit exceeded — slow down."

        parsed = core.parse(text)
        decision = core.resolve(parsed, armed=self.cfg.armed,
                                pending_confirm=self._pending.get(user), now=now)

        # carry the per-user confirm state forward
        if decision.new_pending is None:
            self._pending.pop(user, None)
        else:
            self._pending[user] = decision.new_pending

        if decision.argv is None:
            # no spawn: help / unknown / refusals / need-confirm
            self._audit(user, channel, text, decision.audit_reason, None,
                        int((time.monotonic() - t0) * 1000))
            return decision.reply

        exit_code, output = self._runner(decision.argv)
        self._audit(user, channel, text, decision.audit_reason, exit_code,
                    int((time.monotonic() - t0) * 1000))
        tail = _tail(output)
        status = "✅" if exit_code == 0 else f"⚠️ exit {exit_code}"
        return f"{status}\n```\n{tail}\n```"


# ---------------------------------------------------------------------------
# Socket-Mode transport (lazy SDK import; holds no policy)
# ---------------------------------------------------------------------------

def run_socket_mode(cfg: Config) -> int:
    missing = cfg.missing()
    if missing:
        print(f"iris-slackd: missing env: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        from slack_sdk import WebClient
        from slack_sdk.socket_mode import SocketModeClient
        from slack_sdk.socket_mode.request import SocketModeRequest
        from slack_sdk.socket_mode.response import SocketModeResponse
    except ImportError:
        print("iris-slackd: slack_sdk not installed. "
              "train/.venv/bin/pip install slack_sdk", file=sys.stderr)
        return 3

    web = WebClient(token=cfg.bot_token)
    sm = SocketModeClient(app_token=cfg.app_token, web_client=web)
    daemon = Daemon(cfg)
    armed = "ARMED" if cfg.armed else "read-only"
    print(f"iris-slackd: connecting (Socket Mode, {armed}); "
          f"channel={cfg.channel} users={len(cfg.users)}", file=sys.stderr)

    def _on_request(client: "SocketModeClient", req: "SocketModeRequest"):
        # ack immediately so Slack doesn't retry
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
        if req.type != "events_api":
            return
        event = (req.payload or {}).get("event", {})
        if event.get("type") not in ("app_mention", "message"):
            return
        if event.get("bot_id") or event.get("subtype"):
            return  # ignore bot echoes and edits/joins
        channel = event.get("channel", "")
        user = event.get("user", "")
        text = event.get("text", "")
        try:
            reply = daemon.handle(channel, user, text)
        except Exception as e:  # never let one message kill the loop
            reply = f"internal error: {e}"
        if reply:
            try:
                web.chat_postMessage(channel=channel, text=reply)
            except Exception as e:
                print(f"iris-slackd: post failed: {e}", file=sys.stderr)

    sm.socket_mode_request_listeners.append(_on_request)
    while True:
        try:
            sm.connect()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("iris-slackd: shutting down", file=sys.stderr)
            return 0
        except Exception as e:
            print(f"iris-slackd: socket error, reconnecting in 5s: {e}",
                  file=sys.stderr)
            time.sleep(5)


# ---------------------------------------------------------------------------
# Self-test: exercise handle() end to end with a fake runner, no SDK/network
# ---------------------------------------------------------------------------

def _self_test() -> int:
    env = {"SLACK_APP_TOKEN": "x", "SLACK_BOT_TOKEN": "x",
           "SLACK_CMD_CHANNEL": "C1", "SLACK_CMD_USERS": "Uok",
           "IRIS_SLACKD_LOG": "/tmp/iris_slackd_selftest.jsonl"}
    cfg = Config(env)
    spawned = []

    def fake_runner(argv):
        spawned.append(argv)
        return 0, "ok output"

    d = Daemon(cfg, runner=fake_runner, clock=lambda: 0.0)

    # wrong channel -> ignored, no spawn
    assert d.handle("Cwrong", "Uok", "status") is None
    # wrong user -> ignored
    assert d.handle("C1", "Uintruder", "status") is None
    # read-only -> spawns exact argv
    r = d.handle("C1", "Uok", "status")
    assert spawned and spawned[-1] == core.COMMANDS["status"].argv, r
    # unarmed pause -> refused, no new spawn
    n = len(spawned)
    d.handle("C1", "Uok", "pause")
    assert len(spawned) == n, "unarmed pause must not spawn"
    print("iris-slackd self-test OK:", len(spawned), "spawn(s),",
          "argv containment + auth + armed-gate verified")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="iris Slack command daemon (QL-7)")
    ap.add_argument("--self-test", action="store_true",
                    help="run handle() pipeline with a fake runner (no SDK/network)")
    args = ap.parse_args(argv)
    if args.self_test:
        return _self_test()
    return run_socket_mode(Config())


if __name__ == "__main__":
    raise SystemExit(main())
