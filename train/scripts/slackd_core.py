"""
train/scripts/slackd_core.py — pure, transport-free core for the Slack command
daemon (QL-7). See plans/slack-command-daemon.md.

ALL security logic lives here so it is unit-testable with zero network and zero
Slack SDK. The daemon (iris_slackd.py) is a thin Socket-Mode shell that calls:

    authorize → parse → ratelimit → resolve → (spawn argv) → reply → audit

Every branch returns a structured value the tests assert on. The only unmockable
part (the WebSocket loop) holds no policy.

Hard rules realised here (BACKLOG.md QL-7):
  - Fixed command table: keyword → explicit, compiled-in argv. No message text
    ever reaches a shell or an argv slot. `resolve()` can only ever emit an argv
    that is one of COMMANDS[*].argv verbatim (property-tested).
  - Default read-only; pause/resume need armed; stop needs armed + confirm-token.
  - Channel + user allow-list (fail closed).
  - Per-user rate limit; audit record for every inbound message.

Pure: no slack_sdk import, no network, no global mutable policy state. Randomness
(confirm token) and time (rate-limit / TTL) are injected so tests are deterministic.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Compiled-in paths (NOT from any message)
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent          # train/scripts
VENV_PY = str(SCRIPTS_DIR.parent / ".venv" / "bin" / "python")

# Command classes
READ_ONLY = "read-only"
ARMED = "armed"
ARMED_CONFIRM = "armed+confirm"

CONFIRM_TTL_SEC = 60.0


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]   # full compiled argv, venv_py first; never message-derived
    klass: str              # READ_ONLY | ARMED | ARMED_CONFIRM


# keyword -> (script-relative argv tail, class). The tail's first element is a
# script name resolved under SCRIPTS_DIR; the rest are fixed literal flags.
_TABLE: dict[str, tuple[list[str], str]] = {
    "status":   (["pipeline_status.py"],                  READ_ONLY),
    "doctor":   (["pipeline_doctor.py", "--ai"],          READ_ONLY),
    "quality":  (["pipeline_doctor.py", "--quality-report"], READ_ONLY),
    "flywheel": (["pipeline_ctl.py", "flywheel-status"],  READ_ONLY),
    "pause":    (["pipeline_ctl.py", "pause-flywheel"],   ARMED),
    "resume":   (["pipeline_ctl.py", "resume-flywheel"],  ARMED),
    "stop":     (["pipeline_ctl.py", "stop-flywheel"],    ARMED_CONFIRM),
}

COMMANDS: dict[str, Command] = {
    kw: Command(argv=(VENV_PY, str(SCRIPTS_DIR / parts[0]), *parts[1:]), klass=klass)
    for kw, (parts, klass) in _TABLE.items()
}


# ---------------------------------------------------------------------------
# Authorization (fail closed)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Authz:
    ok: bool
    reason: str   # "ok" | "wrong_channel" | "wrong_user" | "no_allow_channel"


def authorize(channel: str, user: str, *, allow_channel: str,
              allow_users) -> Authz:
    """Allow only a single channel id and an explicit user allow-list.

    Fails closed: an empty/None allow_channel or empty allow_users rejects all.
    """
    if not allow_channel:
        return Authz(False, "no_allow_channel")
    if channel != allow_channel:
        return Authz(False, "wrong_channel")
    if user not in set(allow_users or ()):
        return Authz(False, "wrong_user")
    return Authz(True, "ok")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Parsed:
    keyword: str               # lowercased first token ("" if empty)
    rest: str                  # remainder after the keyword, stripped
    confirm_token: Optional[str]   # first token of rest iff keyword == "confirm"


def _strip_mentions(text: str) -> str:
    """Remove Slack <@Uxxxx> / <!subteam...> style mentions and angle wrappers."""
    out = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "<":
            j = text.find(">", i)
            if j != -1:
                i = j + 1
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def parse(text: str) -> Parsed:
    cleaned = _strip_mentions(text or "").strip()
    if not cleaned:
        return Parsed("", "", None)
    parts = cleaned.split()
    keyword = parts[0].lower()
    rest = cleaned[len(parts[0]):].strip()
    confirm_token = None
    if keyword == "confirm" and len(parts) >= 2:
        confirm_token = parts[1]
    return Parsed(keyword, rest, confirm_token)


# ---------------------------------------------------------------------------
# Confirm-token state (per user, in-memory in the daemon; passed in/out here)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfirmState:
    token: str
    keyword: str       # the destructive command awaiting confirmation (e.g. "stop")
    expires_at: float  # monotonic seconds


# ---------------------------------------------------------------------------
# Resolution (the decision core)
# ---------------------------------------------------------------------------

# Actions
RUN = "RUN"                       # read-only or armed command cleared to spawn
REFUSE_UNARMED = "REFUSE_UNARMED" # armed command but daemon not armed
NEED_CONFIRM = "NEED_CONFIRM"     # destructive: token issued, awaiting confirm
CONFIRMED = "CONFIRMED"           # valid confirm: spawn the destructive argv
REFUSE_CONFIRM = "REFUSE_CONFIRM" # confirm with no/expired/bad token
UNKNOWN = "UNKNOWN"               # keyword not in table
HELP = "HELP"                     # local responder, never spawns


@dataclass(frozen=True)
class Decision:
    action: str
    argv: Optional[tuple[str, ...]]
    reply: str
    audit_reason: str
    new_pending: Optional[ConfirmState]   # confirm state to carry forward for this user


def _default_token() -> str:
    return secrets.token_hex(3)   # 6 hex chars, short nonce


def help_text() -> str:
    lines = ["*iris pipeline commands*"]
    for kw, cmd in COMMANDS.items():
        tag = {"read-only": "", "armed": "  _(armed)_",
               "armed+confirm": "  _(armed + confirm)_"}[cmd.klass]
        lines.append(f"• `{kw}`{tag}")
    lines.append("• `help` — this message")
    return "\n".join(lines)


def resolve(parsed: Parsed, *, armed: bool,
            pending_confirm: Optional[ConfirmState],
            now: float,
            make_token: Callable[[], str] = _default_token) -> Decision:
    """Map a parsed message to a structured Decision.

    Pure: takes the caller's pending-confirm state for THIS user in, returns the
    new state out via Decision.new_pending. Never mutates globals. The returned
    argv, when not None, is always COMMANDS[k].argv for some k — verbatim.
    """
    kw = parsed.keyword

    if kw in ("help", ""):
        return Decision(HELP, None, help_text(), "help", pending_confirm)

    if kw == "confirm":
        if pending_confirm is None:
            return Decision(REFUSE_CONFIRM, None,
                            "Nothing to confirm.", "rejected: no_pending_confirm",
                            None)
        if now > pending_confirm.expires_at:
            return Decision(REFUSE_CONFIRM, None,
                            "Confirmation expired — re-issue the command.",
                            "rejected: confirm_expired", None)
        if not parsed.confirm_token or parsed.confirm_token != pending_confirm.token:
            # consume on bad token (one-shot) to prevent brute-force/replay
            return Decision(REFUSE_CONFIRM, None,
                            "Invalid confirmation token.",
                            "rejected: confirm_bad_token", None)
        if not armed:
            return Decision(REFUSE_UNARMED, None,
                            "Daemon not armed; destructive command refused.",
                            "rejected: unarmed_confirm", None)
        cmd = COMMANDS[pending_confirm.keyword]
        return Decision(CONFIRMED, cmd.argv,
                        f"Confirmed — running `{pending_confirm.keyword}`.",
                        f"confirmed:{pending_confirm.keyword}", None)

    cmd = COMMANDS.get(kw)
    if cmd is None:
        return Decision(UNKNOWN, None,
                        f"Unknown command `{kw}`. Try `help`.",
                        "rejected: unknown_command", pending_confirm)

    if cmd.klass == READ_ONLY:
        return Decision(RUN, cmd.argv, "", f"run:{kw}", pending_confirm)

    if cmd.klass == ARMED:
        if not armed:
            return Decision(REFUSE_UNARMED, None,
                            f"`{kw}` needs an armed daemon (IRIS_SLACKD_ARMED=1).",
                            f"rejected: unarmed:{kw}", pending_confirm)
        return Decision(RUN, cmd.argv, "", f"run:{kw}", pending_confirm)

    # ARMED_CONFIRM (destructive)
    if not armed:
        return Decision(REFUSE_UNARMED, None,
                        f"`{kw}` needs an armed daemon (IRIS_SLACKD_ARMED=1).",
                        f"rejected: unarmed:{kw}", pending_confirm)
    token = make_token()
    new_pending = ConfirmState(token=token, keyword=kw, expires_at=now + CONFIRM_TTL_SEC)
    return Decision(NEED_CONFIRM, None,
                    f"⚠️ `{kw}` is destructive. Confirm within "
                    f"{int(CONFIRM_TTL_SEC)}s with: `confirm {token}`",
                    f"need_confirm:{kw}", new_pending)


# ---------------------------------------------------------------------------
# Rate limiting (per-user token bucket, injected clock)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Per-user token bucket. capacity tokens, refilled over `window` seconds.

    The clock is injected via `now` so tests are deterministic. allow() returns
    True and consumes a token when available, else False (no spawn).
    """

    def __init__(self, capacity: int = 5, window: float = 60.0):
        self.capacity = float(capacity)
        self.rate = capacity / window          # tokens per second
        self._state: dict[str, tuple[float, float]] = {}   # user -> (tokens, last_ts)

    def allow(self, user: str, now: float) -> bool:
        tokens, last = self._state.get(user, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        if tokens < 1.0:
            self._state[user] = (tokens, now)
            return False
        self._state[user] = (tokens - 1.0, now)
        return True


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_record(ts: str, user: str, channel: str, raw: str, matched: str,
                 exit_code: Optional[int], latency_ms: Optional[int] = None) -> dict:
    """One JSON-serialisable audit row. `matched` is the command keyword for an
    accepted message or "rejected: <reason>" for a refused one. Never log secrets
    (the confirm token is short-lived and not security-critical, but we still
    avoid logging raw tokens by recording only the matched reason)."""
    return {
        "ts": ts,
        "user": user,
        "channel": channel,
        "raw": (raw or "")[:500],
        "matched": matched,
        "exit_code": exit_code,
        "latency_ms": latency_ms,
    }
