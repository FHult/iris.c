# DISPATCH — AI Agent Operational Guide

This document is written for AI agents operating on this repository. It covers
how to check training status, manage the pipeline, interpret output, and act
safely without disrupting ongoing training.

---

## 1. What Is Running

This repo trains an IP-Adapter on top of Flux Klein 4B (image synthesis model).
Training runs in a tmux session on an M1 Max MacBook (32GB RAM).
All training data lives on an external SSD at `/Volumes/2TBSSD`.
Checkpoints are written to `/Volumes/2TBSSD/checkpoints/stage1/`.

The pipeline is a multi-step process orchestrated by `run_training_pipeline.sh`.
Active training is the most common long-running state. Do not stop or restart
the pipeline without explicit user instruction.

---

## 2. Checking Status

### Quick machine-readable check (preferred for agents)

```bash
bash train/scripts/pipeline_status.sh --json
```

Returns a single JSON object. Key fields:

```json
{
  "timestamp": "2026-04-05T19:14:27Z",
  "pipeline": {
    "running": true,
    "pid": 13584,
    "chunk": "1",
    "scale": "small"
  },
  "active_step": "train",
  "steps": {
    "downloads": "done",
    "dedup": "done",
    "build_shards": "done",
    "filter": "done",
    "precompute": "done",
    "train": "running",
    "mine_hard": "pending"
  },
  "training": {
    "step": 1000,
    "total_steps": 50000,
    "pct": 2.0,
    "loss": 1.125,
    "loss_smooth": 0.899,
    "lr": 9.99e-05,
    "steps_per_sec": 0.19,
    "eta_seconds": 289477,
    "eta_hours": 80.4,
    "heartbeat_age_s": 120,
    "heartbeat_stale": false,
    "running": true,
    "checkpoints": 5,
    "latest_checkpoint": "step_0001000"
  },
  "data": {
    "shard_count": 432,
    "qwen3_count": 166829,
    "vae_count": 166829,
    "siglip_count": 166829,
    "checkpoints": 5
  },
  "disk": {
    "device": "/dev/disk5s1",
    "used_gib": 1307.4,
    "avail_gib": 555.2
  }
}
```

**Interpreting training health:**
- `training.heartbeat_stale: false` — training is alive and writing heartbeats
- `training.heartbeat_stale: true` — heartbeat >5 min old; process may be stuck or dead
- `training.heartbeat_age_s` — seconds since last heartbeat (written every 100 steps)
- `training.steps_per_sec` — expected ~0.19 on M1 Max at 512px batch_size=1
- `pipeline.running: false` with `steps.train: "running"` — pipeline crashed; needs restart

**Step states:** `"done"` | `"running"` | `"pending"`

### Human-readable check

```bash
bash train/scripts/pipeline_status.sh
```

### Check heartbeat directly

```bash
cat /Volumes/2TBSSD/checkpoints/stage1/heartbeat.json
```

Written every 100 steps. Contains step, loss, lr, steps_per_sec, eta_seconds, timestamp.

### Check latest log tail

```bash
tail -20 $(ls -t /Volumes/2TBSSD/logs/pipeline_chunk1_*.log | head -1)
```

---

## 3. Safe Actions (no user approval needed)

These are read-only and cause no side effects:

```bash
bash train/scripts/pipeline_status.sh --json    # full JSON status
bash train/scripts/pipeline_status.sh           # human-readable status
cat /Volumes/2TBSSD/checkpoints/stage1/heartbeat.json
ls -lh /Volumes/2TBSSD/checkpoints/stage1/
tail -30 $(ls -t /Volumes/2TBSSD/logs/pipeline_chunk1_*.log | head -1)
df -h /Volumes/2TBSSD
```

---

## 4. Actions Requiring User Approval

**Always ask before:**

- Stopping the pipeline (`pipeline_stop.sh`) — training is irreversible wall-clock time
- Restarting the pipeline (`pipeline_start.sh`) — will retrain from step 0 unless `--resume` is passed
- Changing config files (`train/configs/stage1_512px.yaml`)
- Deleting any checkpoint or data file
- Running `git push`

---

## 5. Pipeline Control Scripts

### Start

```bash
bash train/scripts/pipeline_start.sh [OPTIONS]
```

Options:
| Flag | Description |
|------|-------------|
| `--scale small` | 50K steps, 21 shards (~3 days) |
| `--scale medium` | 105K steps, 43 shards (~6 days) |
| `--scale large` | 200K steps, 81 shards (~12 days) |
| `--resume PATH` | Resume from checkpoint (required for chunks 2–4) |
| `--chunk N` | Training chunk 1–4 (default: 1) |
| `--skip-train` | Run data prep only, skip training |
| `--data-root PATH` | Override SSD mount path |

Auto-detects DATA_ROOT from `/Volumes/2TBSSD` → `/Volumes/IrisData` → `train/data`.

### Stop

```bash
bash train/scripts/pipeline_stop.sh
```

Sends SIGTERM, waits up to 120s for graceful shutdown (checkpoint write takes ~60s).
Safe to run — training saves a checkpoint on SIGTERM before exiting.

### Resume after stop

```bash
# Find the latest checkpoint
ls -lh /Volumes/2TBSSD/checkpoints/stage1/step_*.safetensors | sort -k6 | tail -3

# Resume from it
bash train/scripts/pipeline_start.sh --scale small \
  --resume /Volumes/2TBSSD/checkpoints/stage1/step_XXXXXXX.safetensors
```

### Watchdog

Started automatically by the pipeline. Monitors heartbeat and sends macOS
notifications on stall or crash. Do not start manually unless pipeline is
running outside tmux.

---

## 6. Key Paths

| Path | Contents |
|------|----------|
| `/Volumes/2TBSSD/` | External SSD — all training data |
| `/Volumes/2TBSSD/shards/` | 432 WebDataset tar shards (801 GB) |
| `/Volumes/2TBSSD/precomputed/qwen3/` | Qwen3 text embeddings (21 GB) |
| `/Volumes/2TBSSD/precomputed/vae/` | VAE latents (21 GB) |
| `/Volumes/2TBSSD/precomputed/siglip/` | SigLIP image features (66 GB) |
| `/Volumes/2TBSSD/checkpoints/stage1/` | Saved adapter weights + heartbeat |
| `/Volumes/2TBSSD/anchor_shards/` | 10 anchor shards — never delete |
| `/Volumes/2TBSSD/hard_examples/` | Hard example shards — never delete |
| `/Volumes/2TBSSD/logs/` | All pipeline logs |
| `train/configs/stage1_512px.yaml` | Active training config |
| `train/train_ip_adapter.py` | Main training script |
| `plans/` | Backlogs, experiments, design docs |

---

## 7. Training Config (`train/configs/stage1_512px.yaml`)

Key parameters an agent may need to read or report:

```yaml
data:
  batch_size: 1             # 1 = optimal for M1 Max (bs=2 tested, no gain)
  qwen3_cache_dir: ...      # precomputed text embeddings
  vae_cache_dir: ...        # precomputed VAE latents

training:
  num_steps: 50000          # set by --scale; small=50K, medium=105K
  learning_rate: 1.0e-4
  ema_update_every: 10      # EMA updated every 10 steps; eval'd immediately

output:
  checkpoint_dir: "/Volumes/2TBSSD/checkpoints/stage1"
  checkpoint_every: 200     # checkpoint every 200 steps (~17 min)
  log_every: 100            # heartbeat written every 100 steps
```

---

## 8. Expected Performance

| Metric | Value |
|--------|-------|
| Steps/sec | ~0.19 (M1 Max, batch_size=1, 512px) |
| Fwd pass | ~4.1s (80% of step time) |
| mx.eval | ~1.0s (20% of step time) |
| Checkpoint write | ~60s (3.9 GB safetensors) |
| Heartbeat interval | every 100 steps (~8.5 min) |
| small run (50K steps) | ~3 days |
| medium run (105K steps) | ~6.4 days |

---

## 9. Known Issues / Gotchas

- **Heartbeat stale during Metal compilation**: first ~20 min of each new run,
  the Metal GPU compiles training graph shapes. No heartbeat is written during
  this phase. `heartbeat_stale: true` is expected and not an error at startup.

- **scale not shown in JSON when started via pipeline_start.sh without --scale**:
  `pipeline.scale` will be `null` if pipeline was launched without `--scale`.
  Check `training.total_steps` instead: 50000=small, 105000=medium, 200000=large.

- **Checkpoint dir resolves from config**: `pipeline_status.sh` reads
  `checkpoint_dir` from `train/configs/stage1_512px.yaml`. If the config
  changes, re-run status to pick up the new path.

- **Stop script reports "No checkpoint found"**: cosmetic bug (A3-20) — the stop
  works correctly; the message searches the wrong path. Ignore it.

- **A3-03 (cache suffix mismatch)**: at `--scale medium` or larger, the shard
  cache filter has a naming bug that limits training to 34 shards instead of 432.
  Must be fixed before any medium/large run. See `plans/pipeline-audit2-backlog.md`.

---

## 10. Backlogs and Plans

| File | Contents |
|------|----------|
| `plans/pipeline-audit2-backlog.md` | All known pipeline bugs and improvements (audits 2 & 3) |
| `plans/training-perf-backlog.md` | Performance experiments, timing data, batch_size=2 result |
| `plans/pipeline-mlops-backlog.md` | MLOps and observability improvements |
| `plans/ip-adapter-training.md` | Full training architecture design doc |
| `plans/roadmap.md` | High-level project roadmap |
