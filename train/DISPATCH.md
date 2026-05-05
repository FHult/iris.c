# IP-Adapter Training Pipeline — CoWork Dispatch Reference

**This is your primary reference file. Read it before taking any action on the training pipeline.**

> **V2 pipeline is live as of 2026-04-20.**
> V1 scripts have been archived to `train/scripts/v1/` — do not use them.
> See `plans/pipeline-v2-architecture.md` for design and `plans/pipeline-mlops-backlog.md`
> for implementation plan and phase status.
>
> **Future:** V3 (container-native, Kubernetes, web GUI) and V4 (PyTorch/CUDA + cloud GPU)
> are documented in `plans/pipeline-v3-architecture.md`. V3/V4 are pre-design; V2 is the
> active pipeline.

---

## What This Is

Training an IP-Adapter for **Flux Klein 4B** on an M1 Max Mac with a 2 TB NVMe SSD (`/Volumes/2TBSSD`). The adapter adds `--sref` (style reference) conditioning: given a reference image, the model generates images matching its visual style. Training is MLX-based (Apple Silicon GPU, no CUDA).

Hardware: M1 Max (8 P-cores / 2 E-cores, 64 GB unified memory), 2 TB NVMe SSD.

---

## V2 Script Inventory

Active scripts (all in `train/scripts/`):

| Script | Role |
|--------|------|
| `orchestrator.py` | State machine — drives all pipeline steps end-to-end |
| `download_convert.py` | Download + convert JDB tgzs (launched by orchestrator) |
| `downloader.py` | Per-source download worker (imported by download_convert.py) |
| `build_shards.py` | Build unified WebDataset shards |
| `filter_shards.py` | Quality filter shards (min size, caption length) |
| `clip_dedup.py` | CLIP embedding + FAISS dedup |
| `precompute_all.py` | Qwen3 + VAE precompute in a single pass |
| `mine_hard_examples.py` | Extract highest-loss records after each chunk |
| `validator.py` | Post-chunk validation checks |
| `pipeline_lib.py` | Shared primitives: state I/O, sentinels, heartbeats, tmux helpers |
| `pipeline_status.py` | Live pipeline status (reads state file + heartbeats) |
| `pipeline_ctl.py` | Control interface: pause / resume / abort / retry |
| `pipeline_doctor.py` | Deep diagnostic tool — cross-checks sentinel claims against actual artifacts, logs, and heartbeats; produces remediation commands |

**V1 scripts** (archived, not for active use): `train/scripts/v1/`

---

## Pipeline Flow

The orchestrator drives each chunk through these steps in sequence:

```
IDLE → DOWNLOADING → CONVERTING → BUILDING → FILTERING
     → CLIP_EMBED → CLIP_INDEX → CLIP_DUPS
     → PRECOMPUTING → READY → TRAINING → MINING → VALIDATING → DONE
```

Each step writes a `{step}.done` sentinel under `{DATA_ROOT}/pipeline/chunk{N}/` when complete.
On error, a `{step}.error` file is written instead. State is persisted to `pipeline_state.json`.

Training is **chunked** across up to 4 JourneyDB data batches:

| Chunk | JourneyDB range | LR | Steps (medium) | Hard-example mix |
|-------|----------------|----|----------------|-----------------|
| 1 | 000–049 | 1e-4 | 105,000 | — (mined after) |
| 2 | 050–099 | 3e-5 | 40,000 | 5% |
| 3 | 100–149 | 1e-5 | 40,000 | 5% |
| 4 | 150–201 | 1e-5 | 40,000 | 5% |

---

## Session Start Protocol

**Do this at the start of every new session, in order:**

**Step 1 — Run the doctor (AI-optimised, replaces most manual file reads):**
```bash
train/.venv/bin/python train/scripts/pipeline_doctor.py --ai
```
Returns a single compact JSON blob with: `summary` (disk, active training step/loss/ETA, prep progress), `top_action` (the single most important thing to do), `issue_counts`, and `issues` with machine-readable context and remediation commands. **Always run this before reading any individual log, heartbeat, or sentinel file.** The doctor cross-checks sentinel claims against actual artifacts and detects phantom completions, 0-step exits, stale hard examples, stale processes, and ordering violations that `pipeline_status.py` cannot see.

**Step 2 — Check pipeline state (human-readable view):**
```bash
train/.venv/bin/python train/scripts/pipeline_status.py
```
Complements the doctor: shows the live progress line (step within chunk, loss, ETA, log tail). Use this when you need to read the current training log tail or watch live progress; use the doctor when you need to investigate anomalies or verify that completed steps are genuine.

**Step 3 — Check tmux:**
```bash
tmux list-windows -t iris 2>/dev/null || echo "not running"
```
Active runs use the `iris` tmux session. `iris-orch` is the orchestrator window; `iris-prep` is the current prep step; `iris-train` is the trainer.

---

## Calls to Action

### `status` — Pipeline snapshot
```bash
PIPELINE_DATA_ROOT=/Volumes/2TBSSD \
  train/.venv/bin/python train/scripts/pipeline_status.py
```
Shows current chunk, active step, staging shard count, heartbeat age, loss, ETA, and disk usage.

For a smoke test run:
```bash
PIPELINE_DATA_ROOT=/Volumes/2TBSSD/smoke \
  train/.venv/bin/python train/scripts/pipeline_status.py
```

---

### `start` — Start or resume the orchestrator

**Preferred — use `pipeline_ctl.py` (wraps with `caffeinate -dim`, manages the tmux window):**
```bash
# Create the tmux session if it doesn't exist yet, then start:
tmux new-session -d -s iris -n iris-orch 2>/dev/null || true
PIPELINE_DATA_ROOT=/Volumes/2TBSSD \
  train/.venv/bin/python train/scripts/pipeline_ctl.py restart-orchestrator
```

**Smoke test (full pipeline at tiny scale):**
```bash
tmux new-session -d -s iris -n iris-orch 2>/dev/null || true
tmux send-keys -t iris:iris-orch \
  "PIPELINE_DATA_ROOT=/Volumes/2TBSSD/smoke caffeinate -dim \
   train/.venv/bin/python train/scripts/orchestrator.py --resume \
   --config train/configs/v2_pipeline_smoke.yaml" \
  Enter
```

The orchestrator is fully resumable — it reads `pipeline_state.json` and sentinels on startup and picks up from where it left off. You do not need to pass a resume checkpoint; the trainer discovers the latest checkpoint automatically.

Flags:
- `--dry-run` — print decisions without launching anything
- `--skip-dedup` — skip CLIP embedding, indexing, and dedup steps

---

### `pause` / `resume` — Pause and continue

Pause (stops orchestrator and trainer gracefully; trainer saves checkpoint before exit):
```bash
PIPELINE_DATA_ROOT=/Volumes/2TBSSD \
  train/.venv/bin/python train/scripts/pipeline_ctl.py pause
```

Resume (restart the orchestrator — it picks up from state):
```bash
PIPELINE_DATA_ROOT=/Volumes/2TBSSD \
  train/.venv/bin/python train/scripts/pipeline_ctl.py restart-orchestrator
```

---

### `abort` — Stop everything immediately
```bash
PIPELINE_DATA_ROOT=/Volumes/2TBSSD \
  train/.venv/bin/python train/scripts/pipeline_ctl.py abort
# Or hard-kill:
tmux kill-session -t iris
```

---

### `retry` — Unblock training after hitting retry limit

Use this when `pipeline_status.py` shows `Chunk N: ERROR` with "Training exited 137" and `pipeline_ctl.py clear-error` alone has not worked. The orchestrator allows up to 5 auto-retries for jetsam kills; after that it sets `train.error` and stops. The `retry` command resets the in-memory counter, clears the sentinel, and **archives the stale log** — all three are required. Simply deleting `train.error` manually does not work because the orchestrator immediately re-creates it by reading `EXIT_CODE=137` from the stale log.

```bash
train/.venv/bin/python train/scripts/pipeline_ctl.py retry 1 train
```

The orchestrator picks this up on the next poll (~60 s) and re-launches training. To watch recovery:
```bash
train/.venv/bin/python train/scripts/pipeline_status.py
```

For other steps, substitute the step name: `retry 1 precompute`, `retry 2 train`, etc.

---

### `logs` — View active logs
```bash
# Orchestrator:
tail -f /Volumes/2TBSSD/logs/orchestrator.log

# Current prep step (download, build, filter, precompute, etc.):
tail -f /Volumes/2TBSSD/logs/<step>_chunk1.log

# Training:
tail -f /Volumes/2TBSSD/logs/train_chunk1.log

# All log files:
ls /Volumes/2TBSSD/logs/
```

---

### `doctor` — Deep pipeline diagnostic

`pipeline_doctor.py` actively investigates the pipeline state rather than passively reading it. Unlike `pipeline_status.py` (which reports what sentinels say), the doctor cross-checks each sentinel claim against actual artifacts, log content, heartbeat freshness, and ordering invariants.

**Default (human report):**
```bash
train/.venv/bin/python train/scripts/pipeline_doctor.py
train/.venv/bin/python train/scripts/pipeline_doctor.py --chunk 2   # single chunk
```

**AI-optimised compact JSON** (replaces multiple file reads with one structured report):
```bash
train/.venv/bin/python train/scripts/pipeline_doctor.py --ai
```
Output includes: `summary` (disk, active training step/loss/ETA, prep progress, orch poll age), `top_action` (highest-priority action in one sentence), `issue_counts`, and `issues` with machine-readable `context` dicts.

**Interactive fix mode:**
```bash
train/.venv/bin/python train/scripts/pipeline_doctor.py --fix
```
Prints each issue with its remediation command and prompts `[y/N]` before running it.

**Quality mode** (controls how stale-hard-examples issues are handled mid-training):
```bash
# strict (default): stop training, re-mine, restart chunk from clean checkpoint
train/.venv/bin/python train/scripts/pipeline_doctor.py --quality strict

# fast: let current training finish; re-mine before next chunk starts
train/.venv/bin/python train/scripts/pipeline_doctor.py --quality fast
```

**Exit code**: 0 if no CRITICALs, 1 if any CRITICALs found. Useful in shell scripts.

**What it checks:**

| Category | What it detects |
|----------|----------------|
| `phantom` | `.done` sentinel exists but underlying artifacts are missing or empty (no shards, no NPZ files, no checkpoint at expected step, no hard-example tars) |
| `training` | 0-step exit (resumed past end step), NaN loss in logs, very short log for a completed run, non-zero exit code |
| `precompute` | Low NPZ coverage vs shard count, orphaned `.tmp.npz` crash artifacts, `.npz.tmp.npz` double-extension artifacts from pre-fix atomic write bug |
| `checkpoint` | No checkpoint file near expected end step, orphaned future checkpoints |
| `liveness` | Stale heartbeat with tmux window alive (zombie process); distinguishes window-serving-other-chunk from true zombie |
| `code` | Missing `--chunk-base-step` arg in training log (0-step exit risk on cross-chunk warmstarts) |
| `ordering` | Hard examples are older than the training checkpoint they should have been mined from; detects whether dependent chunk is mid-training and adjusts fix severity/commands accordingly |
| `dispatch` | Open issues in `dispatch_queue.jsonl` not yet resolved |
| `error_sentinel` | `.error` sentinels without a corresponding `.done` |
| `environment` | Disk < 80/40 GB, venv missing, `pipeline_state.json` stale or corrupt |

**Orchestrator startup check** (separate, lightweight):
```bash
PIPELINE_DATA_ROOT=/Volumes/2TBSSD \
  train/.venv/bin/python train/scripts/orchestrator.py --dry-run
```
Checks: tmux available · DATA_ROOT exists and writable · venv python · disk ≥ 40 GB · numpy · yaml

---

## Data Layout (`/Volumes/2TBSSD`)

```
pipeline_state.json         Authoritative state (chunk, step, timestamps, config)
pipeline_control.json       Control signals written by pipeline_ctl.py
pipeline/
  chunk1/                   Sentinel files: {step}.done, {step}.error
staging/
  chunk1/
    raw/                    Downloaded raw tgzs (deleted after convert)
    converted/              Converted WDS tars, per source
    shards/                 Chunk-local WebDataset shards (000000–000004 etc.)
    embeddings/             CLIP embeddings for this chunk
shards/                     Promoted production shards (globally unique IDs)
precomputed/
  qwen3/                    Qwen3 text embeddings (.npz, named by record ID)
  vae/                      VAE latents (.npz, named by record ID)
dedup_ids/                  FAISS dedup index and duplicate blocklist
hard_examples/              *** PERSISTENT *** Highest-loss records (WDS .tar)
                            Grows across chunks. Never delete.
checkpoints/stage1/         Training checkpoints (.safetensors)
logs/                       All pipeline logs (orchestrator.log, *_chunk*.log)
.heartbeat/                 Heartbeat files: trainer_chunk1.json, prep_*.json
```

**Shard ID space**: each chunk owns a 200,000-ID block (`chunk1: 0–199999`, `chunk2: 200000–399999`, etc.). IDs are globally unique so shards from different chunks can safely coexist in `shards/` and `precomputed/`.

---

## Telemetry Reference

This section is the definitive map of every file the pipeline reads and writes for state, progress, and alerting. Understanding it prevents the recurring "why is status wrong?" and "what is the orchestrator actually doing?" confusion.

### Source of Truth Hierarchy

```
Sentinel files (.done / .error)   ← authoritative; derive_chunk_state() reads only these
Heartbeat files (.json)           ← live progress; written by workers, read by status + orchestrator
Log files (*.log)                 ← human evidence; status shows tail; orchestrator reads EXIT_CODE only
pipeline_state.json               ← convenience mirror of sentinel state; NOT authoritative
dispatch_queue.jsonl              ← escalated alerts; shown at the bottom of pipeline_status.py output
```

### Sentinel Files

**Path**: `/Volumes/2TBSSD/pipeline/chunk{N}/{step}.done` and `{step}.error`

**What writes them**: Orchestrator — writes `.done` when a step exits 0, `.error` when it fails. Coverage/overflow checks in `_promote_chunk()` can also write `promoted.error` directly.

**What reads them**: `derive_chunk_state()` in `orchestrator.py` and `pipeline_status.py`. This function is stateless — it walks `CHUNK_STEPS` in order and returns the state implied by the last `.done` sentinel. Called on every orchestrator poll and every `pipeline_status.py` invocation.

**Resilience**: Sentinel files survive orchestrator restarts, reboots, and process crashes. They are the only persistent state that the orchestrator trusts. If you need to re-run a step, delete its `.done` and any `.error` sentinels for that step.

**Steps and their sentinel names** (in pipeline order):
```
download, convert, build_shards, filter_shards,
clip_embed, clip_index, clip_dups, precompute,
promoted, train, mine, validate
```

To list all sentinels for a chunk:
```bash
ls /Volumes/2TBSSD/pipeline/chunk1/
```

### Heartbeat Files

**Path**: `/Volumes/2TBSSD/.heartbeat/{process}_chunk{N}.json`

**Format**: JSON with at minimum `{"ts": "<ISO UTC>", "process": "<name>", "chunk": N, ...}`. Additional fields are process-specific (see below).

**Written by**: Each worker process via a background thread calling `write_heartbeat()` from `pipeline_lib.py`. Written atomically (tmp + rename) every ~60s.

**Read by**:
- `pipeline_status.py` — reads all heartbeats to populate the progress line for the active step
- `orchestrator.py` — reads **only** the `trainer_chunk{N}.json` heartbeat for anomaly detection (NaN loss, grad norm, staleness-triggered restart). Prep worker heartbeats are NOT read by the orchestrator.

**Staleness thresholds**:
- Status script marks trainer heartbeat stale at **120s** (custom threshold in `_trainer_heartbeat`)
- Status script marks prep worker heartbeats stale at **300s** (`_worker_heartbeat`)
- Orchestrator triggers trainer restart at **900s** (`HEARTBEAT_STALE_SECS`)

**Heartbeats by process**:

| File | Written by | Key fields |
|------|-----------|------------|
| `orchestrator.json` | orchestrator each poll | `step="poll"` |
| `precompute_chunk{N}.json` | `precompute_all.py` | `done`, `total`, `pct`, `eta_sec`, `current_shard`, `current_phase` |
| `prep_precompute_chunk{N}.json` | `precompute_all.py` | `status` (failed/running), `exit_code` |
| `trainer_chunk{N}.json` | `train_ip_adapter.py` | `step`, `total_steps`, `loss`, `grad_norm`, `eta_sec`, `siglip_coverage_pct` |
| `build_shards_chunk{N}.json` | `build_shards.py` | `done`, `total`, `pct` |
| `download_convert_chunk{N}.json` | `download_convert.py` | `done`, `total`, `pct`, `phase`, `current_tgz`, `dl_speed_mbps` |
| `filter_shards_chunk{N}.json` | `filter_shards.py` | `done`, `total`, `pct` |
| `clip_dedup_chunk{N}.json` | `clip_dedup.py` | `done`, `total`, `pct` |
| `mine_hard_examples_chunk{N}.json` | `mine_hard_examples.py` | `done`, `total`, `pct` |

To inspect a heartbeat directly:
```bash
cat /Volumes/2TBSSD/.heartbeat/precompute_chunk1.json | python3 -m json.tool
cat /Volumes/2TBSSD/.heartbeat/trainer_chunk1.json   | python3 -m json.tool
```

To check heartbeat age:
```bash
python3 -c "
import json, datetime, pathlib
p = pathlib.Path('/Volumes/2TBSSD/.heartbeat/precompute_chunk1.json')
d = json.loads(p.read_text())
ts = datetime.datetime.fromisoformat(d['ts'])
age = (datetime.datetime.now(datetime.timezone.utc) - ts).total_seconds()
print(f'{age:.0f}s ago — {d}')
"
```

### Log Files

**Path**: `/Volumes/2TBSSD/logs/{step}_chunk{N}.log`

**What writes them**: The bash wrapper in `tmux_new_window()`: `({cmd}) >> '{log}' 2>&1; echo EXIT_CODE=$? >> '{log}'`. Every line of stdout and stderr goes here, plus `EXIT_CODE=N` as the final line.

**What reads them**:
- `last_exit_code()` in `pipeline_lib.py` — scans the file in reverse for `EXIT_CODE=N`. This is how the orchestrator knows if a step succeeded.
- `_log_tail()` in `pipeline_status.py` — shows the last 4–10 lines (excluding `EXIT_CODE=`).

**Critical rule: log files are fixed-name and never automatically cleaned between sessions.** Old logs from a prior run will appear current to the status script. The orchestrator rotates prep step logs on error-auto-retry (renames to `{step}_chunk{N}.{timestamp}.log`) but does NOT rotate training logs. Manual cleaning is required after a pipeline reset.

**Standard log names**:
```
orchestrator.log              orchestrator main output
download_chunk{N}.log         download + convert step
build_chunk{N}.log            build_shards step
filter_chunk{N}.log           filter_shards step
clip_embed_chunk{N}.log       clip_dedup embed step
clip_index_chunk{N}.log       clip_dedup build-index step
clip_dups_chunk{N}.log        clip_dedup find-dups step
precompute_chunk{N}.log       precompute_all step (active log; old ones get .{ts}.log suffix)
train_chunk{N}.log            training (never auto-rotated)
mine_chunk{N}.log             hard example mining
validate_chunk{N}.log         validation
```

**How to clean stale logs before a fresh run**:
```bash
# Remove ALL logs (do this when resetting pipeline state):
rm -f /Volumes/2TBSSD/logs/*.log /Volumes/2TBSSD/logs/*.jsonl

# Remove stale heartbeats (do this when resetting pipeline state):
rm -f /Volumes/2TBSSD/.heartbeat/*.json

# Keep only the orchestrator and active step log if pipeline is mid-run:
# (identify current step from sentinels first, keep its log)
```

### `pipeline_state.json`

**Path**: `/Volumes/2TBSSD/pipeline_state.json`

**Written by**: Orchestrator — updated each poll whenever chunk state changes. Also updated at training start with `steps`, `lr`, `started_at`.

**Read by**: `pipeline_status.py` for `run_id`, `scale`, and chunk state snapshots.

**NOT authoritative**: The orchestrator re-derives chunk state from sentinel files on every poll. `pipeline_state.json` is a snapshot for tooling; if it disagrees with sentinels, sentinels win.

### `dispatch_queue.jsonl`

**Path**: `/Volumes/2TBSSD/logs/dispatch_queue.jsonl`

**Written by**: `dispatch_issue()` in `pipeline_lib.py` — called by the orchestrator for problems it cannot auto-resolve: step failed too many times, trainer restart limit exceeded, loss NaN/Inf, sustained high grad norm, disk critical.

**Read by**: `pipeline_status.py` via `_read_dispatch_issues()` — open (unresolved) issues appear at the bottom of status output. The status script filters by `resolved: false`.

To inspect directly:
```bash
cat /Volumes/2TBSSD/logs/dispatch_queue.jsonl | python3 -m json.tool
```

To resolve an issue after intervention (removes it from status output):
```bash
train/.venv/bin/python train/scripts/pipeline_ctl.py dispatch-resolve <issue_id>
```

### `orchestrator.jsonl`

**Path**: `/Volumes/2TBSSD/logs/orchestrator.jsonl`

Written by `log_orch()` — a structured event log of every orchestrator decision. Useful for post-mortem debugging. Not read by any tooling currently.

```bash
# Recent orchestrator decisions:
tail -20 /Volumes/2TBSSD/logs/orchestrator.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(d['ts'], d.get('level','info').upper(), d.get('message',''))
"
```

---

## Known Telemetry Gaps

Confirmed issues found in code review. Document here so they are not rediscovered. Resolved items are kept for historical context.

**Gap 1 — ~~Escalated alerts are invisible~~ RESOLVED (2026-04-25)**
`pipeline_status.py` now reads `dispatch_queue.jsonl` directly via `_read_dispatch_issues()` and shows open issues at the bottom of status output. Escalated alerts (retry limit exceeded, NaN loss, disk critical) are visible in normal `pipeline_status.py` output.

**Gap 2 — Orchestrator restart orphans any in-flight prep step** *(still open)*
`_active_prep` is in-memory only. If the orchestrator is killed while iris-prep is running, on restart `_active_prep` is None. `_poll_prep_window()` returns immediately. When the prep window eventually finishes (or was already finished), the exit code is never read and the step is never marked done. The chunk is stuck indefinitely. **Workaround**: after orchestrator restart, check if iris-prep window exists. If it does, wait for it to finish then manually mark the step done or check its EXIT_CODE in the log. If the step log already contains `EXIT_CODE=0`, mark it done manually:
```bash
# Example: precompute completed but orchestrator missed it
touch /Volumes/2TBSSD/pipeline/chunk1/precompute.done
```

**Gap 3 — ~~Hung prep workers undetected~~ RESOLVED (2026-04-25)**
The orchestrator now dispatches a warning after `PREP_HUNG_HOURS` (6h) of continuous prep window activity. It also monitors prep worker heartbeats: if a worker's heartbeat goes stale for >30 min, the orchestrator dispatches an alert (visible in `pipeline_status.py`). If you see a prep-hung alert, check the log and kill the window manually if confirmed stuck.

**Gap 4 — ~~SigLIP coverage not enforced at promotion~~ RESOLVED (2026-04-25)**
`_promote_chunk()` now enforces ≥90% siglip coverage when `training.siglip: true` in the pipeline config. A partial siglip cache is rejected at promotion with a `promoted.error` sentinel.

---

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `Chunk N: ERROR` — "Training exited 137" — retry limit exceeded | macOS jetsam OOM kill (unified memory pressure) | `pipeline_ctl.py retry N train` — resets counter, clears sentinel, archives stale log. Do NOT just delete `train.error`; orchestrator immediately recreates it from stale log EXIT_CODE |
| Orchestrator exits immediately | Doctor check failed | Read startup output; fix the flagged issue |
| Prep window creation fails with "index N in use" | tmux session/window index conflict | Fixed in pipeline_lib.py — update if you see this again |
| Training never starts (READY state loops) | Precompute coverage < 90% | Check precompute log; re-run precompute_all.py manually if stuck |
| Trainer heartbeat stale alarm fires immediately after restart | Old heartbeat file not cleared | Fixed: orchestrator clears heartbeat on restart |
| Training loss NaN | Bad batch or LR too high | Orchestrator will restart trainer once; if persistent, abort and investigate |
| Precompute .npz collision between chunks | Shard IDs overlapping | Fixed: --start-idx ensures disjoint ID space; canary in _promote_chunk catches overflow |
| SSD at >90% capacity | staging/raw not cleaned | Orchestrator deletes raw dir after convert; if stuck, delete manually |
| Tmux session missing on status check | Orchestrator crashed | Check orchestrator.log tail; restart orchestrator |
| `No shards with precomputed cache found` | Precompute sentinel set but coverage is partial | Delete precompute sentinels; re-run precompute step |

---

## Decision Flow

**Starting a new session / anything seems off**
→ `pipeline_doctor.py --ai` **first**. Read `top_action` and `issues`. Only reach for individual logs, heartbeats, or sentinel files if the doctor output is insufficient for the specific question.

**"Is anything running?"**
→ Doctor `summary.training.hb_age_s` + `summary.prep` + `tmux list-windows -t iris`

**"Something is running — check live progress"**
→ `pipeline_status.py` (step within chunk, loss, ETA, log tail)

**"Nothing is running — continue where we left off"**
→ Restart orchestrator (it resumes from state automatically)

**"Something looks wrong / unexpected state"**
→ `pipeline_doctor.py --ai`. Do not start reading individual log files or sentinel dirs until the doctor has been run — it cross-checks all of those in one pass and surfaces the root cause directly. Only read raw logs if the doctor `issues` list points to a specific log file for further detail.

**"Training completed but seems suspicious"**
→ `pipeline_doctor.py --ai` — the doctor checks checkpoint step vs expected end, log length, and whether `--chunk-base-step` was passed. These are the common phantom-completion patterns.

**"Want to free disk space"**
→ `staging/chunk{N}/raw/` (deleted automatically after convert); `staging/chunk{N}/converted/` (safe after shards are built); **never delete `hard_examples/`** while the dependent chunk is still training.

**"Training finished chunk N, start chunk N+1"**
→ Orchestrator advances automatically. Before chunk N+1 training starts, run `pipeline_doctor.py --ai` to confirm hard examples are fresh and checkpoint step matches expected end.
