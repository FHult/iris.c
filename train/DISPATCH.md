# IP-Adapter Training Pipeline — CoWork Dispatch Reference

**This is your primary reference file. Read it before taking any action on the training pipeline.**

> **V2 TRANSITION IN PROGRESS (2026-04-18)**
> All V1 pipeline data has been cleared. V2 architecture is being built from scratch.
> See `plans/pipeline-v2-architecture.md` for design and `plans/pipeline-mlops-backlog.md`
> for implementation plan and phase status.
> V2 Phase 1 (orchestrator + all-source chunked pipeline) is the current work item.
> This document describes V1 for historical reference; update it as V2 scripts are built.

---

## What This Is

Training an IP-Adapter for **Flux Klein 4B** on an M1 Max Mac with a 2 TB NVMe SSD (`/Volumes/2TBSSD`). The adapter adds `--sref` (style reference) conditioning: given a reference image, the model generates images matching its visual style. Training is MLX-based (Apple Silicon GPU, no CUDA).

Hardware: M1 Max (8 P-cores / 2 E-cores, 64 GB unified memory), 2 TB NVMe SSD.

---

## Pipeline Overview

The pipeline has **10 steps** (some parallel) across ~3 weeks of compute (at `medium` scale):

| # | Step | Duration | Output | Parallelism |
|---|------|----------|--------|-------------|
| 1 | Verify LAION/COYO on disk | 1 min | — | — |
| 2a | WikiArt download + WDS convert | 2 min | `raw/wikiart_wds/` (80 shards) | ↕ parallel with 2b, 3 |
| 2b | JourneyDB download + WDS convert | 5 min | `raw/journeydb_wds/` (210 shards) | ↕ parallel with 2a, 3 |
| 3 | CLIP deduplication | 2 h | `dedup_ids/duplicate_ids.txt` | ↕ parallel with 2a, 2b |
| 4 | Build unified shards | 2–4 h | `shards/*.tar` (5000 img/shard) | filter runs in background |
| 5 | Filter shards (final pass) | 30 min | shards rewritten in place | background during step 4 |
| 7 | Anchor set | 5 min | `anchor_shards/` (10K images) | after step 5 |
| 8 | Precompute Qwen3 + VAE | 2–14 h | `precomputed/qwen3/` + `vae/` | selective per-chunk |
| 9a | Train IP-Adapter (chunk N) | 1–7 days | `checkpoints/step_*.safetensors` | chunked across 4 phases |
| 9b | Mine hard examples | 15–30 min | `hard_examples/*.tar` | after each chunk |

**Key topology**: Steps 2a, 2b, and 3 all launch simultaneously at the start. Step 4 runs a concurrent background filter loop so shards are validated as they are written. Step 8 reads each shard once and writes both Qwen3 embeddings and VAE latents in a single pass (2× faster than running them sequentially). The precompute step now uses **selective shard sampling** — only the shards needed for each chunk are precomputed (VAE dominates at ~0.28 s/image; see `train/TRAINING.md`).

Training is **chunked** across 4 JourneyDB data batches. Chunk 1 is the initial full run. Chunks 2–4 fine-tune on additional data with decaying LR. **Step counts and shard budget are controlled by `--scale`** (see `start` section). The table below shows `medium` defaults.

| Chunk | JourneyDB range | LR | Steps (medium) | Hard-example mix |
|-------|----------------|----|----------------|-----------------|
| 1 | 000–049 | 1e-4 | 105,000 | — (mined after) |
| 2 | 050–099 | 3e-5 | 40,000 | 5% |
| 3 | 100–149 | 1e-5 | 40,000 | 5% |
| 4 | 150–201 | 1e-5 | 40,000 | 5% |

**Scale presets** (`--scale` option):

| Preset | Chunk 1 steps | Chunks 2–4 steps | Chunk 1 shards | Chunks 2–4 shards | Wall clock |
|--------|---------------|------------------|----------------|-------------------|------------|
| `small` | 50,000 | 15,000 | 21 | 7 | ~3 days |
| `medium` | 105,000 | 40,000 | 43 | 17 | ~7 days |
| `large` | 200,000 | 60,000 | 81 | 25 | ~11 days |
| `god-like` | 400,000 | 120,000 | 162 | 50 | ~18 days |
| `all-in` | 540,000 | 200,000 | all | all | ~26 days |
| `N` (numeric) | N | — | ceil(N×2/4970) | — | varies |

Shard counts are sized for ~1 pass through each chunk's selected images. Each chunk uses a different random seed so the full 4-chunk pipeline samples ~4× as many unique images.

---

## Session Start Protocol

**Do this at the start of every new session, in order:**

**Step 1 — Check V2 build status** (if V2 Phase 1 is in progress):
```bash
cat /Volumes/2TBSSD/pipeline_state.json
```
This file is the authoritative source of chunk, scale, config, and per-chunk step state.
Do **not** infer these values from logs, config files, or checkpoint directory names.

> **If `pipeline_state.json` is missing:** V2 orchestrator has not started yet.
> Check `plans/pipeline-v2-architecture.md` Section 12 for current phase and next steps.

**Step 2 — Get live telemetry** (once V2 orchestrator is running):
```bash
python train/scripts/pipeline_status.py --json
```
Returns step count, loss, ETA, heartbeat age, disk usage, and full state embedded under `"state"`.

**Step 2 (V1 fallback, if V2 not yet running):**
```bash
bash train/scripts/pipeline_status.sh --json
```

---

## Calls to Action

All scripts work from any working directory.

### `status` — Full pipeline snapshot
```bash
bash train/scripts/pipeline_status.sh
```
Shows all steps (✅/⏳/⬜), active process names, tmux sessions, last 15 lines of the active log, and disk layout. **This is always the first command to run.** Auto-detects `/Volumes/2TBSSD` so no flags needed.

For running steps, each line shows the **most recent heartbeat** parsed from the step's log:
- `build_shards`: `[worker N] src X/Y | written N records | shards A/B full`
- `filter_shards`: `[X/Y] kept=N  dropped=N  X.X s/shard  ETA Xm`
- `precompute_all`: `[X/Y] PCT%  X.X s/shard  ETA Xh Xm`
- `clip_dedup`: `[X/N] N duplicates found`
- `train_ip_adapter`: `step X,XXX/105,000  loss X.XXXX (avg X.XXXX)  lr X  X.XX steps/s  ETA Xh XXm`

---

### `start` — Start the pipeline
```bash
# Chunk 1 (initial full run, medium scale):
bash train/scripts/pipeline_start.sh

# Larger run:
bash train/scripts/pipeline_start.sh --scale large
bash train/scripts/pipeline_start.sh --scale god-like
bash train/scripts/pipeline_start.sh --scale all-in   # uses every shard

# Custom step count (shards sized automatically):
bash train/scripts/pipeline_start.sh --scale 200000

# Specific chunk with optional overrides:
bash train/scripts/pipeline_start.sh --chunk 2 --resume /path/to/step_105000.safetensors
bash train/scripts/pipeline_start.sh --chunk 1 --skip-dedup --skip-train
bash train/scripts/pipeline_start.sh --data-root /Volumes/2TBSSD --siglip
```
Guards against double-starting: exits if a `pipeline` tmux session already exists. Launches `run_training_pipeline.sh` under `caffeinate` in a tmux session named `pipeline`.

Options: `--chunk 1–4`, `--scale PRESET_OR_N`, `--data-root PATH`, `--resume CKPT`, `--config YAML`, `--steps N`, `--lr RATE`, `--siglip`, `--recaption`, `--skip-dedup`, `--skip-train`

The `--scale` preset controls both step count and how many shards to precompute for each chunk (see table in Pipeline Overview). When `--steps N` is also provided it overrides the step count from the preset while keeping the shard sizing.

> **Operational requirements before leaving a training run unattended:**
> - Always launch training inside tmux: `tmux new-session -s training` (if `pipeline_start.sh` didn't already create a session).
> - Never run pipeline jobs (WDS conversion, precompute) on 2TBSSD while training is active — competing I/O caused a 2.6h stall during chunk 1 (see BACKLOG PIPELINE-3).
> - Check for `WARNING: SigLIP cache coverage` in training startup output. If present, some shards lack siglip features and image conditioning will silently degrade for those batches.
> - For chunks 2–4: confirm `--anchor-shard-dir` is set in the launch command, otherwise the 20% anchor slot silently receives nothing (see BACKLOG PIPELINE-1).

---

### `reset` — Delete all pipeline outputs, keep downloads
```bash
bash train/scripts/pipeline_reset.sh              # standard: keeps raw/wikiart, raw/journeydb
bash train/scripts/pipeline_reset.sh --full       # also deletes downloaded datasets
bash train/scripts/pipeline_reset.sh --dry-run    # preview what would be deleted
bash train/scripts/pipeline_reset.sh --yes        # skip confirmation prompt
```
Deletes all intermediate and final pipeline outputs: converted WDS dirs, unified shards, dedup blocklist, precomputed caches, embeddings, and logs. Refuses to run if the pipeline lock is held by a live process — run `stop` first. `--full` additionally removes the downloaded raw datasets (wikiart, journeydb); LAION and COYO are never touched (pre-existing). Training checkpoints are never deleted.

**NEVER delete `anchor_shards/` or `hard_examples/`.** These are persistent across all chunks — the anchor set is always mixed in, and the hard example store grows through all 4 chunks. Reset intentionally leaves them in place.

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
Auto-selects the most relevant log for what is currently running. `--progress` filters to only heartbeat lines (worker updates, shard counts, step/loss lines), removing startup noise. This is the recommended mode for Claude CoWork Dispatch progress checks.

---

### `sysmon` — System observability snapshot
```bash
bash train/scripts/pipeline_sysmon.sh
```
One-shot snapshot of: CPU utilisation by core class (P/E), memory pressure and swap, GPU memory allocation, disk I/O throughput, network bytes, and per-process breakdown for all active pipeline processes. No flags needed.

---

### `precompute` — Run precompute step only
```bash
bash train/scripts/pipeline_precompute.sh
bash train/scripts/pipeline_precompute.sh --siglip   # include SigLIP (~420 GB extra)
```
Runs step 8 (unified Qwen3 + VAE precompute) via `precompute_all.py`. Reads each shard once and writes both caches in a single pass. Use after build/filter complete if you want to decouple precompute from training. Launches in tmux session `precompute`. Each .npz file is idempotent — already-computed records are skipped on restart.

**Selective precompute**: the full pipeline only precomputes the shards needed per chunk (sized by `--scale`). Subsequent chunks bias toward new/unprocessed shards via `--new-shards-first` (70% new, 30% existing) for data diversity. Pass `--max-shards N` to `precompute_all.py` directly for manual control. VAE is the dominant cost at ~0.28 s/image (~6.6 h per 43 shards); see `train/TRAINING.md` for full measurements.

---

### `test` — Smoke tests across the full pipeline
```bash
bash train/scripts/pipeline_test.sh           # full suite (9 tests, < 5 min)
bash train/scripts/pipeline_test.sh --fast    # skip dataset loader test
```
Verifies every pipeline component is functional before committing to a long run: Python environment and packages, shard format and readability, filter logic, precompute cache validity, model importability, and checkpoint integrity. All tests are **read-only** — they do not modify data.

Tests: T1 Python env · T2 Shard format · T3 Caption filter logic · T4 Metadata collection · T5 Qwen3 cache · T6 VAE cache · T7 Dataset loader · T8 Model import · T9 Checkpoint readability

---

### `doctor` — Health check of pipeline outputs
```bash
bash train/scripts/pipeline_doctor.sh           # warnings-only output
bash train/scripts/pipeline_doctor.sh --verbose  # show passing checks too
```
Reviews all completed outputs for correctness and integrity: source counts, dedup blocklist, shard content sampling, filter quality, precompute coverage and dtype consistency, ID alignment between shards and caches, and checkpoint step progression. **Use this when something looks wrong** — e.g. training loss is unexpectedly high, a step seemed to complete too fast, or after a crash recovery.

Checks: D1 Source completeness · D2 Dedup blocklist · D3 Shard content · D4 Filter quality · D5 Qwen3 integrity · D6 VAE integrity · D7 ID alignment · D8 Checkpoint progression

---

## Proposed Calls to Action (not yet implemented)

These scripts do not exist yet but are well-defined and ready to build:

### `pipeline_validate.sh` — Check training quality
Generate N sample images using the current checkpoint and fixed eval prompts in `train/configs/eval_prompts.txt`. Saves to `/tmp/iris_eval/`. Useful to spot degradation or confirm a chunk improved quality.
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
  journeydb_wds/    JourneyDB chunk 1 converted to WDS (210 shards)
  wikiart_wds/      WikiArt converted to WDS (80 shards)
shards/             Merged unified shards (~430+ shards, 5000 images each)
anchor_shards/      *** PERSISTENT *** Fixed anchor set (~10K images, LAION+WikiArt only)
                    Always mixed into training (20% by default). Never delete.
hard_examples/      *** PERSISTENT *** Highest-loss records extracted after each chunk
                    WDS .tar files written by mine_hard_examples.py; grows across chunks
                    Mixed at 5% into subsequent chunk training. Never delete.
dedup_ids/
  duplicate_ids.txt Blocklist from CLIP dedup (124,340 IDs)
  dedup_index.faiss Cross-chunk FAISS index (chunk 2+ only)
precomputed/
  qwen3/            Qwen3 text embeddings, 4-bit quantised (.npz, ~143 GB)
  vae/              VAE latents, int8 quantised (.npz, ~198 GB)
  siglip/           SigLIP features, 4-bit quantised (.npz, ~420 GB, optional)
  .done             Sentinel: precompute step 8 complete
logs/               Pipeline run logs (pipeline_chunk*.log)
```

Checkpoints: `train/checkpoints/` (on the internal SSD, not the external).

**Storage note**: Only the shards selected by `--scale` are precomputed per chunk — you do not need to precompute all ~430 shards upfront. At `medium` scale, chunk 1 uses ~43 shards (~341 GB of precomputed data), chunks 2–4 use ~17 each.

---

## Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `status` shows 0/N shards | Wrong DATA_ROOT detected | `pipeline_status.sh --data-root /Volumes/2TBSSD` |
| `status` shows filter ✅ but 0 shards | Stale `.filtered_chunk1` sentinel from failed run | Safe to ignore — pipeline re-runs step 4 when count_tars=0 |
| build_shards reports N shards but dir empty | Old bug: background filter deleted .tar.tmp files | Fixed in `filter_shards.py` (only deletes files >5 min old) |
| Training hangs, no log output | sample_q timeout (dataset crash) | `pipeline_stop.sh`, check logs, `pipeline_resume.sh` |
| SSD at >90% capacity | raw/ not cleaned up after conversion | Raw dirs safe to delete after shards are built |
| FAISS SIGSEGV | OpenMP on Apple Silicon | Already fixed (`omp_set_num_threads(1)`) |
| tmux session not found on `resume` | Previous run already finished or crashed | Check `pipeline_status.sh`, then `pipeline_start.sh` |
| Pipeline killed when shell exits | Not using tmux | Use `pipeline_start.sh` (launches in tmux automatically) |
| Chunk 2+ training loss spikes | hard_examples/ deleted between chunks | Restore from backup or skip `--hard-examples` for that chunk |
| Mining step fails: no EMA checkpoint | Chunk 1 ended before first ema_update_every | Check checkpoints/; mining needs at least one EMA .safetensors |
| Precompute much slower than expected | All shards already done (--new-shards-first had nothing new) | Normal — idempotent skip is fast; actual precompute only on new shards |
| Training only sees anchor + hard data | remaining_ratio near zero (anchor + hard ratios sum to ≥1.0) | Reduce anchor_mix_ratio or hard_mix_ratio in stage config |

---

## Decision Flow for Common Situations

**"Is anything running?"** → `pipeline_status.sh` → look at Active processes and tmux sections.

**"Something is running but I want to check on it"** → `pipeline_logs.sh --follow`

**"Nothing is running, I want to continue where we left off"** → `pipeline_resume.sh`

**"Nothing is running, I want to start fresh (chunk 1)"** → `pipeline_start.sh`

**"System seems slow / SSD is hot"** → `pipeline_sysmon.sh`

**"Training finished chunk 1, start chunk 2"** → mining runs automatically; then `pipeline_start.sh --chunk 2` (auto-detects latest checkpoint, mixes hard examples in).

**"I need to free up disk space"** → `pipeline_sysmon.sh` shows disk; raw/ dirs safe to delete once shards are built; embeddings/ safe to delete once dedup is done. Never delete `anchor_shards/` or `hard_examples/`.

**"I want to increase the training budget"** → re-run with `--scale large` or `--scale god-like`; precompute is idempotent so only new shards are processed.

**"Something looks wrong — training loss is high or a step finished suspiciously fast"** → `pipeline_doctor.sh` — audits all outputs for correctness.

**"I want to verify everything is set up correctly before starting"** → `pipeline_test.sh` — smoke tests all components in < 5 min.
