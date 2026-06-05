# Plan: Slack Command Daemon (QL-7) + Bidirectional Slack (QL-6)

Status: design / not started. Earmarked for a dedicated, security-first session.
Prereq QL-5 (outbound alert sink) is DONE in `train/monitoring/sinks.py`.

## Decision: ship QL-7 first, QL-6 is the optional later superset

QL-7 is the *minimum viable, maximally hardened* realisation of QL-6: a closed,
compiled-in command set that can only launch already-existing pipeline scripts.
We build QL-7 in full; QL-6's richer surface (interactive buttons, broad action
space, arbitrary-arg commands) is explicitly deferred and may never be needed.

## Threat model (drives every decision)

- The command channel may be visible to more people than intended.
- Slack message text is attacker-influenceable.
- Therefore: the daemon must be **incapable** of running anything outside its
  compiled-in allow-list, regardless of message content. No string ever flows
  from a Slack message into a shell or an argv slot in v1.

## Hard rules (non-negotiable — copied from BACKLOG.md QL-7)

1. **Socket Mode only.** Outbound WebSocket; no inbound port, no public endpoint,
   no tunnel. `SLACK_APP_TOKEN` (xapp-) + `SLACK_BOT_TOKEN` (xoxb-) from env.
   Never in repo/config.
2. **Single channel allow-list** (`SLACK_CMD_CHANNEL` id) + **user allow-list**
   (`SLACK_CMD_USERS`, comma-sep ids). Anything else → ignored + audit-logged.
3. **Fixed command table**: short keyword → explicit `argv list`. No
   user-supplied arguments in v1. Never `shell=True`. Never interpolate message
   text into a command.
4. **Default read-only.** State-changing commands (`pause`/`resume`) require
   `IRIS_SLACKD_ARMED=1` in the daemon env; otherwise acknowledged-but-refused.
   Destructive (`stop`, `force-next`) always require an explicit `confirm <token>`
   reply (and still need armed).
5. **No GPU, no direct pipeline mutation.** The daemon only spawns existing
   scripts (which own their own locking/sentinels). It never touches state itself.
6. **Audit everything** to `logs/slackd.jsonl`: ts, user, channel, raw text,
   matched command (or `rejected: <reason>`), exit code. Rate-limit per user.
7. **Output back to channel** = script stdout tail, truncated. Long output → a
   file snippet (`files_upload_v2`), not a wall of text.

## Files

- `train/scripts/iris_slackd.py` — the daemon (new). Single file, no package.
- `train/scripts/slackd_core.py` — pure, transport-free core (new): command
  table, auth/allow-list predicate, message parser, argv resolver, rate limiter,
  audit-record builder. **All security logic lives here so it is unit-testable
  with zero network and zero Slack SDK.** The daemon is a thin Socket-Mode shell
  around it.
- `train/scripts/tests/test_slackd_core.py` — tests (new), transport mocked.
- `train/.venv` — add `slack_sdk` (pinned), the one new train-only dev dep.
- `train/DISPATCH.md` — add an "iris-slackd" operational section.
- `train/scripts/start_pipeline.sh` (or a small `start_slackd.sh`) — optional
  supervised launch in an `iris-slackd` tmux window under caffeinate.

Splitting core from transport mirrors the QL-5 pattern (`sinks.py` keeps the
mockable poster separate) and the orchestrator/ctl pure-core extractions
(`_retry_policy`, `_ready_gate`) — behaviour is testable without the live edge.

## Command table (v1)

| keyword   | argv (all prefixed with venv_py)                         | class        |
|-----------|----------------------------------------------------------|--------------|
| `status`  | `pipeline_status.py`                                     | read-only    |
| `doctor`  | `pipeline_doctor.py --ai`                                | read-only    |
| `quality` | `pipeline_doctor.py --quality-report`                    | read-only    |
| `flywheel`| `pipeline_ctl.py flywheel-status`                        | read-only    |
| `pause`   | `pipeline_ctl.py pause-flywheel`                         | armed        |
| `resume`  | `pipeline_ctl.py resume-flywheel`                        | armed        |
| `stop`    | `pipeline_ctl.py stop-flywheel`                          | armed+confirm|

- `venv_py` = absolute path to `train/.venv/bin/python`, compiled in (not from a
  message). argv is a fixed Python list; no shell, `subprocess.run([...],
  shell=False, timeout=N, cwd=repo_root)`.
- `help` is a local responder (lists the table); never spawns anything.
- Unknown keyword → `rejected: unknown command`, audit-logged, friendly reply.

## Core API (transport-free, the test surface)

```python
# slackd_core.py
COMMANDS: dict[str, Command]          # keyword -> Command(argv, klass)

def authorize(channel, user, *, allow_channel, allow_users) -> Authz
    # -> Authz(ok: bool, reason: str)   # wrong channel / wrong user / ok

def parse(text) -> Parsed
    # strips bot mention, lowercases keyword; -> Parsed(keyword, rest, confirm_token)

def resolve(parsed, *, armed, pending_confirm) -> Decision
    # -> Decision(action: RUN|REFUSE_UNARMED|NEED_CONFIRM|CONFIRMED|UNKNOWN|HELP,
    #             argv: list[str] | None, reply: str, audit_reason: str)

class RateLimiter:    # per-user token bucket, monotonic clock injected for tests
    def allow(self, user, now) -> bool

def audit_record(ts, user, channel, raw, matched, exit_code) -> dict
```

The daemon calls: `authorize → parse → ratelimit → resolve → (spawn) → reply →
audit`. Every branch returns a structured value the tests assert on; the only
unmockable part (the WebSocket loop) holds no policy.

## Confirm-token flow (destructive: `stop`)

- On `stop` (armed), daemon replies: `⚠️ confirm with: confirm <token>` where
  `<token>` is a short random nonce stored in-memory keyed by user, TTL ~60s.
- A subsequent `confirm <token>` from the **same user in the same channel**
  within TTL resolves to `CONFIRMED` → runs the argv. Wrong/expired/replayed
  token → refused + audit-logged. One-shot (consumed on use).
- Pure: `resolve()` takes `pending_confirm` state in, returns the new state out;
  no global mutation in the core. Tested with injected clock.

## Audit + rate limit

- `logs/slackd.jsonl`, one JSON object per inbound message (accepted or not):
  `{ts, user, channel, raw, matched, exit_code, latency_ms}`. Rejections carry
  `matched: "rejected: <reason>"`. Never log token secrets.
- Rate limit: per-user token bucket (e.g. 5 cmds / 60s). Over-limit →
  refused + audit-logged, no spawn.

## Output handling

- Capture `stdout`+`stderr`, cap at N lines / M KB. Reply with a fenced tail.
- If over cap, `files_upload_v2` a snippet and reply with a one-line summary +
  exit code. (This uses the Socket-Mode WebClient/bot token, distinct from the
  QL-5 webhook sink, which stays for unsolicited alerts. Note the two are
  different transports — don't conflate.)

## Process model

- `iris-slackd` tmux window, `caffeinate -dim`, supervised like the other
  pipeline processes. Clean shutdown on SIGTERM; auto-reconnect on socket drop
  (slack_sdk `SocketModeHandler` handles reconnect; we add a supervising retry
  loop with backoff for hard failures).
- Single instance guard (pidfile/lock) so two daemons can't both act on a command.

## Build order (and what "done" means at each step)

1. **Read-only core + listener + auth + audit.** `status` / `doctor` / `quality`
   / `flywheel`, allow-list enforcement, audit log, rate limit. Tests:
   wrong-channel/user rejected+logged, keyword→exact-argv mapping, unknown
   rejected, rate-limit trips, audit-record shape. Deliverable: can ask the
   pipeline for status from Slack; cannot change anything.
2. **Armed controls.** `pause` / `resume` behind `IRIS_SLACKD_ARMED=1`;
   acknowledged-but-refused when unarmed. Tests: armed gate both directions.
3. **Confirm-gated destructive.** `stop` with `confirm <token>`. Tests: happy
   path, wrong token, expired token, replay, cross-user token rejection.

Effort: ~1.5–2 days for (1)+(2); (3) ~half a day on top.

## Testing (transport mocked, no network — mirror test_slack_sink.py)

- `slackd_core` is pure → direct unit tests, no SDK import needed for the policy
  tests. Inject clock for rate-limit + confirm-TTL.
- For the thin daemon, mock the WebClient: assert it never calls `subprocess`
  for a rejected message, and calls it with the **exact** compiled argv for an
  accepted one (`shell=False`, fixed list).
- A "no message can produce an argv outside the table" property test: fuzz
  `parse → resolve` over random text, assert `Decision.argv` is always either
  `None` or an element of `COMMANDS[*].argv` verbatim.

## Secrets / config (never in repo)

- Env only: `SLACK_APP_TOKEN`, `SLACK_BOT_TOKEN`, `SLACK_CMD_CHANNEL`,
  `SLACK_CMD_USERS`, `IRIS_SLACKD_ARMED`. Config (if any) holds only the env-var
  *names*, like QL-5 holds only `SLACK_WEBHOOK_URL` the name. Document the
  required Slack app scopes (`connections:write`, `chat:write`,
  `files:write`, `app_mentions:read` / `channels:history` for the one channel).

## QL-6 (deferred superset — not built now)

Only revisit if QL-7 proves insufficient. Adds: interactive buttons/modals,
per-command arg validators (e.g. `restart-from-chunk N` with `^[1-9][0-9]?$`),
`promote-champion` / `golden-eval` launches, richer routing. Same Socket-Mode
transport, same auth/allow-list/audit foundation — QL-7's `slackd_core` is the
substrate it would extend. Keep the closed command table as the default even
then; any arg-accepting command gets a strict regex validator, never raw text.

## Explicitly out of scope for v1

- HTTP / Slash-Command transport (needs a public endpoint; Socket Mode avoids it).
- Any command that takes free-form arguments.
- Any direct state mutation by the daemon (it only spawns locking-aware scripts).
- Posting anything other than the spawned script's captured output.
