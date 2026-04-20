# IP-Adapter Training Pipeline — CoWork Dispatch Reference

**This is your primary reference file. Read it before taking any action on the training pipeline.**

> **V2 pipeline is live as of 2026-04-20.**
> V1 scripts have been archived to `train/scripts/v1/` — do not use them.
> See `plans/pipeline-v2-architecture.md` for design and `plans/pipeline-mlops-backlog.md`
> for implementation plan and phase status.

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
| `pipeline_ctl.py` | Control interface: pause / resume / abort |

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

**Step 1 — Check pipeline state:**
```bash
PIPELINE_DATA_ROOT=/Volumes/2TBSSD \
  train/.venv/bin/python train/scripts/pipeline_status.py
```
This reads `pipeline_state.json`, sentinel files, and heartbeats. It is the authoritative source of truth — do not infer state from logs or checkpoint names.

**Step 2 — Check tmux:**
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
```bash
# Production run (starts or resumes):
tmux new-session -d -s iris -n iris-orch
tmux send-keys -t iris:iris-orch \
  "PIPELINE_DATA_ROOT=/Volumes/2TBSSD train/.venv/bin/python train/scripts/orchestrator.py \
   --config train/configs/v2_pipeline.yaml 2>&1 | tee /Volumes/2TBSSD/logs/orchestrator.log" \
  Enter

# Smoke test (full pipeline at tiny scale):
tmux new-session -d -s iris -n iris-orch
tmux send-keys -t iris:iris-orch \
  "PIPELINE_DATA_ROOT=/Volumes/2TBSSD/smoke train/.venv/bin/python train/scripts/orchestrator.py \
   --config train/configs/v2_pipeline_smoke.yaml 2>&1 | tee /Volumes/2TBSSD/smoke/logs/orchestrator.log" \
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

Resume (simply restart the orchestrator — it picks up from state):
```bash
# Same as start above
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

### `doctor` — Startup health check

The orchestrator runs a doctor check at startup and refuses to proceed if any check fails. To run it manually or inspect results:
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

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
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

**"Is anything running?"**
→ `pipeline_status.py` + `tmux list-windows -t iris`

**"Something is running — check progress"**
→ `pipeline_status.py` (shows active step, heartbeat, loss, ETA)

**"Nothing is running — continue where we left off"**
→ Restart orchestrator (it resumes from state automatically)

**"Something looks wrong"**
→ Check `logs/<step>_chunk1.log` for the failing step; check `pipeline/chunk1/*.error` for error sentinels

**"Want to free disk space"**
→ `staging/chunk1/raw/` (deleted automatically after convert); `staging/chunk1/converted/` (safe after shards are built); never delete `hard_examples/`

**"Training finished chunk 1, start chunk 2"**
→ Orchestrator starts mining automatically after training completes; then advances to chunk 2 when config specifies multiple chunks
