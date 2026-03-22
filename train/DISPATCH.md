# IP-Adapter Training Pipeline — CoWork Dispatch Reference

**This is your primary reference file. Read it before taking any action on the training pipeline.**

---

## What This Is

Training an IP-Adapter for **Flux Klein 4B** on an M1 Max Mac with a 2 TB NVMe SSD (`/Volumes/2TBSSD`). The adapter adds `--sref` (style reference) conditioning: given a reference image, the model generates images matching its visual style. Training is MLX-based (Apple Silicon GPU, no CUDA).

Hardware: M1 Max (8 P-cores / 2 E-cores, 64 GB unified memory), 2 TB NVMe SSD.

---

## Pipeline Overview

The pipeline has **9 sequential steps** across ~3 weeks of compute:

| # | Step | Duration | Output |
|---|------|----------|--------|
| 1 | Verify downloads | 1 min | — |
| 2a | WikiArt → WebDataset | 5 min | `raw/wikiart_wds/` (104 shards) |
| 2b | JourneyDB → WebDataset | 30 min | `raw/journeydb_wds/` (210 shards) |
| 3 | CLIP deduplication | 2 h | `dedup_ids/duplicate_ids.txt` |
| 4 | Build unified shards | 2–4 h | `shards/*.tar` (5000 img/shard) |
| 5 | Filter shards | 30 min | shards rewritten in place |
| 6 | Cross-chunk dedup index | 1 h | `dedup_ids/dedup_index.faiss` |
| 7 | Anchor set | 5 min | `anchor_shards/` |
| 8a | Precompute Qwen3 embeddings | 8 h | `precomputed/qwen3/` (~143 GB) |
| 8b | Precompute VAE latents | 6 h | `precomputed/vae/` (~198 GB) |
| 9 | Train IP-Adapter | 3–7 days | `checkpoints/step_*.safetensors` |

Training is **chunked** across 4 JourneyDB data batches. Chunk 1 is the initial full run. Chunks 2–4 fine-tune on additional data with decaying LR.

| Chunk | JourneyDB range | LR | Steps |
|-------|----------------|----|-------|
| 1 | 000–049 | 1e-4 | 105,000 |
| 2 | 050–099 | 3e-5 | 40,000 |
| 3 | 100–149 | 1e-5 | 40,000 |
| 4 | 150–201 | 1e-5 | 40,000 |

---

## Calls to Action

All scripts work from any working directory. All long-running scripts launch in `tmux` automatically and survive shell disconnects.

### `status` — Full pipeline snapshot
```bash
bash train/scripts/pipeline_status.sh
```
Shows all 9 steps (✅/⏳/⬜), active process names, tmux sessions, last 15 lines of the active log, and disk layout. **This is usually the first command to run.** Auto-detects `/Volumes/2TBSSD` so no flags needed.

For running steps, each line shows the **most recent heartbeat** parsed from the step's log — no separate progress query needed:
- `build_shards`: `[worker N] src X/Y | written N records | shards A/B full`
- `filter_shards`: `[X/Y] kept=N  dropped=N  X.X shards/s  ETA Xm`
- `precompute_qwen3/vae/siglip`: `[X/Y] N,NNN embeddings/latents/features  X.XX shards/s  ETA Xm`
- `clip_dedup`: `[X/N] N duplicates found`
- `train_ip_adapter`: `step X,XXX/105,000  loss X.XXXX (avg X.XXXX)  lr X  X.XX steps/s  ETA Xh XXm`

---

### `start` — Start the pipeline
```bash
# Chunk 1 (initial full run):
bash train/scripts/pipeline_start.sh

# Specific chunk with optional overrides:
bash train/scripts/pipeline_start.sh --chunk 2 --resume /path/to/step_105000.safetensors
bash train/scripts/pipeline_start.sh --chunk 1 --skip-dedup --skip-train
bash train/scripts/pipeline_start.sh --data-root /Volumes/2TBSSD --siglip
```
Guards against double-starting: exits if a `pipeline` tmux session already exists. Launches `run_training_pipeline.sh` under `caffeinate` in a tmux session named `pipeline`.

Options: `--chunk 1–4`, `--data-root PATH`, `--resume CKPT`, `--config YAML`, `--steps N`, `--lr RATE`, `--siglip`, `--skip-dedup`, `--skip-train`

---

### `stop` — Graceful stop
```bash
bash train/scripts/pipeline_stop.sh
```
Sends SIGTERM to all pipeline processes (training saves its last periodic checkpoint before exiting), then kills tmux sessions `pipeline`, `build_shards`, `precompute`. Reports the latest checkpoint found so you know where to resume from.

---

### `pause` — Pause and checkpoint
```bash
bash train/scripts/pipeline_pause.sh
```
Same as `stop` with explicit messaging that the run is paused and can be resumed. The training loop saves a checkpoint at the next periodic interval (every 1000 steps by default) before exiting. Use `resume` to continue.

---

### `resume` — Resume from latest checkpoint
```bash
bash train/scripts/pipeline_resume.sh
bash train/scripts/pipeline_resume.sh --chunk 2   # force chunk
```
Auto-detects the latest checkpoint in `checkpoints/`, infers the current chunk from the step count, and re-launches the pipeline with `--resume`. If the chunk is ambiguous, pass `--chunk` explicitly.

---

### `logs` — View the active log
```bash
bash train/scripts/pipeline_logs.sh              # last 60 lines
bash train/scripts/pipeline_logs.sh --lines 200  # more context
bash train/scripts/pipeline_logs.sh --follow     # stream live (Ctrl-C to exit)
bash train/scripts/pipeline_logs.sh --progress   # heartbeat lines only (best for dispatch)
bash train/scripts/pipeline_logs.sh --all        # list all log files
```
Auto-selects the most relevant log for what is currently running: build_shards → `/tmp/build_shards.log`, precompute → `DATA_ROOT/logs/precompute*.log`, training → `DATA_ROOT/logs/pipeline_chunk*.log`.

`--progress` filters to only heartbeat lines (worker updates, shard counts, step/loss lines), removing startup noise and model output. This is the recommended mode for Claude CoWork Dispatch progress checks.

---

### `sysmon` — System observability snapshot
```bash
bash train/scripts/pipeline_sysmon.sh
```
One-shot snapshot of: CPU utilisation by core class (P/E), memory pressure and swap, GPU memory allocation, disk I/O throughput, network bytes, and per-process breakdown for all active pipeline processes. No flags needed.

---

### `precompute` — Run precompute steps only
```bash
bash train/scripts/pipeline_precompute.sh
bash train/scripts/pipeline_precompute.sh --siglip   # include SigLIP (420 GB extra)
```
Runs steps 8a (Qwen3, ~8 h) and 8b (VAE, ~6 h) in sequence. Use after `build_shards` and `filter_shards` complete if you want to decouple precompute from training. Launches in tmux session `precompute`.

---

### `test` — Smoke tests across the full pipeline
```bash
bash train/scripts/pipeline_test.sh           # full suite (9 tests, < 5 min)
bash train/scripts/pipeline_test.sh --fast    # skip dataset loader test
```
Verifies every pipeline component is functional before committing to a long run: Python environment and packages, shard format and readability, filter logic, precompute cache validity, model importability, and checkpoint integrity. All tests are **read-only** — they do not modify data. Run this after setup or after any code change before starting a full pipeline run.

Tests: T1 Python env · T2 Shard format · T3 Caption filter logic · T4 Metadata collection · T5 Qwen3 cache · T6 VAE cache · T7 Dataset loader · T8 Model import · T9 Checkpoint readability

---

### `doctor` — Health check of pipeline outputs
```bash
bash train/scripts/pipeline_doctor.sh           # warnings-only output
bash train/scripts/pipeline_doctor.sh --verbose  # show passing checks too
```
Reviews all completed outputs for correctness and integrity: source counts, dedup blocklist, shard content sampling, filter quality, precompute coverage and dtype consistency, ID alignment between shards and caches, and checkpoint step progression. **Use this when something looks wrong** — e.g. training loss is unexpectedly high, a step seemed to complete too fast, or after a crash recovery. Reports ⚠️ warnings (anomalies worth reviewing) and ❌ errors (things that need to be fixed before proceeding).

Checks: D1 Source completeness · D2 Dedup blocklist · D3 Shard content · D4 Filter quality · D5 Qwen3 integrity · D6 VAE integrity · D7 ID alignment · D8 Checkpoint progression

---

## Proposed Calls to Action (not yet implemented)

These scripts do not exist yet but are well-defined and ready to build:

### `pipeline_validate.sh` — Check training quality
Generate N sample images using the current checkpoint and the fixed eval prompts in `train/configs/eval_prompts.txt`. Saves to `/tmp/iris_eval/`. Useful to spot degradation or confirm a chunk improved quality.
```bash
bash train/scripts/pipeline_validate.sh --steps 2 --n 4
```

### `pipeline_benchmark.sh` — Throughput and ETA
Parse the training log to compute current steps/hour, report progress (steps done / total budget), and estimate time remaining for the current chunk and all remaining chunks.
```bash
bash train/scripts/pipeline_benchmark.sh
```

### `pipeline_export.sh` — Package adapter for deployment
Copy the best EMA checkpoint + configs to a deployment bundle; optionally quantise to int4. Output is a directory loadable by `iris.c --ip-adapter`.
```bash
bash train/scripts/pipeline_export.sh --checkpoint checkpoints/step_105000_ema.safetensors
```

### `pipeline_recaption.sh` — Re-caption short captions (parallelised)
Launch two tmux panes running `recaption.py` over split shard ranges. Moondream-based style-focused re-captioning improves training signal for ~30% of records with captions under 10 words. Takes ~2 days.
```bash
bash train/scripts/pipeline_recaption.sh --data-root /Volumes/2TBSSD
```

---

## Data Layout (`/Volumes/2TBSSD`)

```
raw/
  laion/            LAION-Aesthetics-v2 (~150 shards, pre-downloaded WebDataset)
  coyo/             COYO-700M subset (~50 shards)
  journeydb_wds/    JourneyDB converted to WDS (210 shards)
  wikiart_wds/      WikiArt converted to WDS (104 shards)
shards/             Merged unified shards (5000 images/shard)
anchor_shards/      Fixed anchor set (~10K images, mixed in at training time)
dedup_ids/
  duplicate_ids.txt Blocklist from CLIP dedup (124,340 IDs)
  dedup_index.faiss Cross-chunk FAISS index (chunk 2+ only)
embeddings/         CLIP embeddings from dedup step (~3 GB)
precomputed/
  qwen3/            Qwen3 text embeddings, 4-bit quantised (.npz, ~143 GB)
  vae/              VAE latents, int8 quantised (.npz, ~198 GB)
  siglip/           SigLIP features, 4-bit quantised (.npz, ~420 GB, optional)
logs/               Pipeline run logs (pipeline_chunk*.log, precompute*.log)
```

Checkpoints: `train/checkpoints/` (on the internal SSD, not the external).

---

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `status` shows 0/N shards | Wrong DATA_ROOT detected | `pipeline_status.sh --data-root /Volumes/2TBSSD` |
| build_shards exits with NameError | Old run with unfixed code | Pull latest, re-run `pipeline_start.sh` |
| Kernel watchdog panic / system crash | Swap exhaustion in build_shards | Restart; code is now fixed (streaming I/O) |
| Training hangs, no log output | sample_q timeout (dataset crash) | `pipeline_stop.sh`, check logs, `pipeline_resume.sh` |
| SSD at >90% capacity | raw/ not cleaned up after conversion | Raw dirs safe to delete after shards are built |
| FAISS SIGSEGV | OpenMP on Apple Silicon | Already fixed (`omp_set_num_threads(1)`) |
| tmux session not found on `resume` | Previous run already finished or crashed | Check `pipeline_status.sh`, then `pipeline_start.sh` |

---

## Decision Flow for Common Situations

**"Is anything running?"** → `pipeline_status.sh` → look at Active processes and tmux sections.

**"Something is running but I want to check on it"** → `pipeline_logs.sh --follow`

**"Nothing is running, I want to continue where we left off"** → `pipeline_resume.sh`

**"Nothing is running, I want to start fresh (chunk 1)"** → `pipeline_start.sh`

**"System seems slow / SSD is hot"** → `pipeline_sysmon.sh`

**"Training finished chunk 1, start chunk 2"** → `pipeline_start.sh --chunk 2` (auto-detects latest checkpoint)

**"I need to free up disk space"** → `pipeline_sysmon.sh` shows disk; raw/ dirs safe to delete once shards are built; embeddings/ safe to delete once dedup is done.

**"Something looks wrong — training loss is high or a step finished suspiciously fast"** → `pipeline_doctor.sh` — audits all outputs for correctness.

**"I want to verify everything is set up correctly before starting"** → `pipeline_test.sh` — smoke tests all components in < 5 min.
