# Pipeline V2 Architecture

**Status:** Design — not yet implemented  
**Scope:** --sref IP-Adapter training pipeline, extensible to other training recipes  
**Target hardware:** Apple M1 Max 32 GB, 2 TB external SSD  
**Date:** 2026-04-18

---

## 1. Design Goals

| Goal | Constraint |
|------|-----------|
| Autonomous end-to-end execution | Escalate to operator on irrecoverable errors only |
| Resilient: survives process crash, reboot, disk full | All state external to processes; resumable at any stage |
| Resource-safe: never OOM the GPU or saturate disk simultaneously | Orchestrator holds all scheduling decisions |
| Disk-efficient: free stage output when downstream stage is complete | Explicit lifecycle rules per artefact type |
| Maximum parallelism: download, preprocess, and train overlap where safe | Resource conflict model is explicit in this doc |
| Extensible: support IP-Adapter, LoRA, full finetune, future recipes | Recipe registry decouples pipeline from training objective |
| Observable: all events are structured JSON; Dispatch/CLI/web can consume | Single telemetry bus, no ad hoc log scraping |

---

## 2. Data Strategy

### 2.1 Current (V1) approach — what we're replacing

| Source | Volume | V1 treatment |
|--------|--------|-------------|
| LAION-subset | ~350 K | Downloaded and sharded once; all shards loaded across all chunks |
| COYO-subset | ~200 K | Same as LAION |
| WikiArt | ~90 K | Same |
| JourneyDB | ~200 K × 4 chunks | Only source chunked across 4 runs |

Problem: LAION/COYO/WikiArt front-load network and disk I/O, then sit idle. Early chunks see all non-JDB data. The training distribution shifts chunk-to-chunk as JDB grows but other sources don't.

### 2.2 V2 approach — all sources chunked equally

Every data source is split into 4 equal-volume slices; each training chunk draws from slice N of every source.

```
Chunk 1:  LAION[0..25%]  + COYO[0..25%]  + WikiArt[0..25%]  + JDB[000-049]
Chunk 2:  LAION[25..50%] + COYO[25..50%] + WikiArt[25..50%] + JDB[050-099]
Chunk 3:  LAION[50..75%] + COYO[50..75%] + WikiArt[50..75%] + JDB[100-149]
Chunk 4:  LAION[75..100%]+ COYO[75..100%]+ WikiArt[75..100%]+ JDB[150-201]
```

Benefits:
- Consistent data mix across training: no chunk N sees exclusively more JDB than chunk 1
- Download, dedup, and sharding for chunk N+1 can proceed while chunk N trains (no blocking dependency on all data being ready upfront)
- Disk usage is bounded: only 1-chunk-worth of raw data needs to live on disk at once

### 2.3 Shard target sizes

| Source | Total images | Per-chunk images | Shards/chunk (5 K/shard) |
|--------|-------------|-----------------|--------------------------|
| LAION | ~350 K | ~87 K | ~17 |
| COYO | ~200 K | ~50 K | ~10 |
| WikiArt | ~90 K | ~22 K | ~4 |
| JourneyDB | ~200 K | ~50 K | ~10 |
| **Total/chunk** | | **~209 K** | **~41 shards** |

Target: ~40 shards per chunk. Steps per chunk: 40 × 5000 / batch_size = to be calibrated.

---

## 3. Process Architecture

Eight independent worker processes, each owning exactly one concern. No process reaches into another's responsibility domain.

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  downloader │   │  converter  │   │ deduplicator│   │   builder   │
│  (net I/O)  │   │  (cpu/disk) │   │  (cpu/FAISS)│   │  (cpu/disk) │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │                  │
       └─────────────────┴─────────────────┴──────────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │     orchestrator    │
                          │  (state machine +   │
                          │   resource model)   │
                          └─────────┬──────────┘
       ┌─────────────────┬──────────┴──────────┬─────────────────┐
       │                 │                     │                 │
┌──────▼──────┐   ┌──────▼──────┐   ┌─────────▼────┐   ┌───────▼─────┐
│ precomputer │   │   trainer   │   │    miner     │   │  validator  │
│  (GPU+disk) │   │  (GPU only) │   │  (GPU+disk)  │   │  (cpu/GPU)  │
└─────────────┘   └─────────────┘   └──────────────┘   └─────────────┘
```

### 3.1 Process responsibilities

**downloader**
- Input: source manifest (URLs, expected checksums)
- Output: raw files in `raw/{source}/chunk{N}/`
- Telemetry: bytes/s, file count, HTTP errors, checksum failures
- Resource profile: network-heavy, minimal CPU, minimal disk IOPS (sequential write)
- Can run: always; does not conflict with any other process

**converter**
- Input: raw files from downloader
- Output: normalized JPEG images in `converted/{source}/chunk{N}/`
- Telemetry: images/s, decode errors, corrupt file count, disk written
- Resource profile: CPU-heavy (image decode/encode), no GPU, moderate disk
- Can run: alongside downloader; avoid overlap with precomputer or trainer on M1 Max

**deduplicator**
- Input: converted images, existing FAISS index from previous chunks
- Output: updated FAISS index, per-image duplicate flags in `dedup_ids/`
- Telemetry: embeddings/s, duplicate rate, index size, FAISS errors
- Resource profile: CPU-heavy (FAISS), GPU embed step via MPS (brief, <5 min)
- **Serial constraint**: FAISS index update must be single-process; cannot run two dedup workers on same index simultaneously
- Can run: parallel with downloader; avoid GPU overlap with trainer/precomputer

**builder**
- Input: converted + deduped images
- Output: WebDataset `.tar` shards in `shards/`
- Telemetry: shards written, images/shard, skip rate, disk written
- Resource profile: disk-intensive (random read, sequential write), CPU moderate
- Can run: after dedup; can overlap with download/convert for next chunk

**precomputer**
- Input: shards
- Output: per-record Qwen3 embeddings + VAE latents in `precomputed/{chunk}/`
- Telemetry: records/s, GPU utilisation, cache hits, skipped (already computed), errors
- Resource profile: GPU-dominant (Qwen3 + VAE both on GPU), high disk write
- **Serial constraint with trainer**: both use GPU; must not overlap
- Can run: after builder; strictly separate from trainer time-wise

**trainer**
- Input: shards + precomputed cache + hard examples
- Output: checkpoints in `checkpoints/stage1/`
- Telemetry: step, loss, grad norm, learning rate, steps/s, ETA
- Resource profile: GPU-exclusive, high memory (adapter + backbone frozen)
- **Serial constraint**: nothing else may use GPU while trainer runs
- Can run: after precomputer; hard examples injected when miner has produced them

**miner**
- Input: shards + precomputed cache + latest checkpoint
- Output: hard examples in `hard_examples/`
- Telemetry: records evaluated, skip rate, threshold loss, examples written
- Resource profile: GPU-heavy (inference pass), disk read-heavy
- **Serial constraint**: must not overlap with trainer (GPU conflict)
- Can run: after chunk N training; feed into chunk N+1 training

**validator**
- Input: checkpoints, shard sample, hard examples
- Output: validation report JSON + sample images
- Telemetry: FID proxy, per-class loss, hard-example loss trend, embedding coverage
- Resource profile: GPU moderate, CPU light
- Can run: after miner completes; does not block next chunk start

### 3.2 Process interface contract

Every process:
1. Accepts `--config config.yaml` and `--chunk N`
2. Writes a heartbeat file every 30 s: `{work_dir}/.heartbeat/{process_name}.json`
3. Writes structured events to `{log_dir}/{process_name}_chunk{N}.jsonl`
4. Exits 0 on success, non-zero on unrecoverable error
5. Is idempotent: running twice produces the same output (sentinel files gate re-execution)
6. Exposes a `--status` flag that prints current progress as JSON and exits

---

## 4. Orchestrator

The orchestrator is the only process that knows about other processes. Workers do not talk to each other.

### 4.1 State machine

States per chunk:

```
IDLE → DOWNLOADING → CONVERTING → DEDUPLICATING → BUILDING
     → PRECOMPUTING → TRAINING → MINING → VALIDATING → DONE
```

The orchestrator maintains `pipeline_state.json` (atomic write via temp file + rename):

```json
{
  "schema_version": 2,
  "run_id": "run_20260418_001",
  "recipe": "ip_adapter_flux4b",
  "chunks": {
    "1": {
      "state": "DONE",
      "started_at": "2026-04-10T10:00:00Z",
      "completed_at": "2026-04-15T08:00:00Z",
      "checkpoint": "checkpoints/stage1/step_065000.safetensors"
    },
    "2": {
      "state": "TRAINING",
      "started_at": "2026-04-18T12:00:00Z",
      "trainer_pid": 12345,
      "last_step": 42000
    }
  },
  "issues": [
    {
      "id": "I-001",
      "severity": "warning",
      "chunk": 2,
      "process": "builder",
      "message": "Shard 000017 has only 4312 records (< 5000 target)",
      "resolved": false,
      "ts": "2026-04-18T14:23:11Z"
    }
  ]
}
```

### 4.2 Resource model

The orchestrator tracks a resource token system. Each process declares its resource requirements; the orchestrator grants tokens before launch:

```
GPU_TOKEN       — exclusive; only one holder at a time
GPU_SHARE       — concurrent low-intensity GPU (embed step in dedup)
DISK_WRITE_HIGH — high sequential write (builder, precomputer)
DISK_WRITE_LOW  — low write (trainer checkpoints)
NET_TOKEN       — network download
CPU_HEAVY       — converter, dedup FAISS
```

| Process | GPU_TOKEN | GPU_SHARE | DISK_WRITE_HIGH | NET_TOKEN | CPU_HEAVY |
|---------|-----------|-----------|-----------------|-----------|-----------|
| downloader | — | — | — | ✓ | — |
| converter | — | — | — | — | ✓ |
| deduplicator | — | ✓ (brief) | — | — | ✓ |
| builder | — | — | ✓ | — | — |
| precomputer | ✓ | — | ✓ | — | — |
| trainer | ✓ | — | ✓ (ckpt only) | — | — |
| miner | ✓ | — | ✓ | — | — |
| validator | ✓ | — | — | — | — |

Rules:
- `GPU_TOKEN` is mutually exclusive with itself and `GPU_SHARE`
- `DISK_WRITE_HIGH` is mutually exclusive with itself (both precomputer and builder writing full speed simultaneously would saturate the SSD)
- All other combinations are safe to run in parallel

### 4.3 Time estimation

Each process reports its throughput metric (images/s, records/s, steps/s) in its heartbeat. The orchestrator maintains a rolling ETA per stage:

```
ETA(stage) = remaining_work / observed_throughput
```

When scheduling whether to start chunk N+1's download while chunk N trains, the orchestrator computes:
- `ETA(train_N)`: remaining training steps × observed steps/s
- `ETA(download_N+1)`: total images × observed download rate

If `ETA(download_N+1) < ETA(train_N)`, start download immediately (it will finish before training ends, no conflict).

### 4.4 Interfaces

**Dispatch interface**: The orchestrator writes `dispatch_queue.jsonl` — append-only log of issues requiring escalation. Claude Dispatch reads this file and responds with action records in `dispatch_responses.jsonl`. Orchestrator polls for responses every 60 s.

```jsonl
{"id":"E-003","severity":"error","ts":"...","process":"trainer","message":"Loss NaN at step 1200","context":{...},"suggested_action":"reduce_lr"}
```

**CLI interface**: `pipeline_status.py --json` reads `pipeline_state.json` and prints a structured summary. `pipeline_ctl.py pause|resume|abort|skip-chunk N` writes control signals to `pipeline_control.json` which the orchestrator polls.

**Web interface** (future): REST API wrapping `pipeline_state.json` and `dispatch_queue.jsonl`. No orchestrator changes required — the file-based protocol is the stable interface.

### 4.5 Resilience

- Orchestrator is a long-running process, run inside a `tmux` session
- On crash: restart with `pipeline_ctl.py restart-orchestrator` — state is fully in `pipeline_state.json`; orchestrator picks up from current chunk state
- Process crash detection: if a process heartbeat is >90 s old and process is in `RUNNING` state, orchestrator marks it `CRASHED`, logs the issue, attempts one automatic restart; if second attempt fails, escalates to dispatch
- Disk-full detection: orchestrator monitors available space every 60 s; if <20 GB, pauses new stage launches and logs a `disk_low` warning; if <5 GB, emergency pause + escalate

---

## 5. Telemetry Schema

All processes emit structured events to `{log_dir}/{process}_chunk{N}.jsonl`:

```json
{"ts": "2026-04-18T14:23:11.123Z", "process": "trainer", "chunk": 2, "event": "step",
 "step": 42000, "loss": 0.3241, "grad_norm": 0.8812, "lr": 0.0001, "steps_per_sec": 2.3,
 "eta_sec": 9870}

{"ts": "2026-04-18T14:23:11.123Z", "process": "precomputer", "chunk": 2, "event": "progress",
 "records_done": 12000, "records_total": 41000, "skipped": 320, "records_per_sec": 4.1}

{"ts": "2026-04-18T14:23:11.123Z", "process": "orchestrator", "event": "issue",
 "id": "I-007", "severity": "warning", "message": "Dedup rate 42% (expected <20%)",
 "auto_resolved": false}
```

### 5.1 Anomaly detection (orchestrator)

| Metric | Anomaly condition | Action |
|--------|------------------|--------|
| Training loss | NaN or >2.0 for 100 steps | Pause + escalate |
| Grad norm | >10.0 for 50 steps | Log warning; if >50.0 for 10 steps, pause |
| Dedup rate | >50% (indicates wrong source slice or index corruption) | Warn + log |
| Shard record count | <4000 or >6000 (target is 5000) | Log warning |
| Precompute skip rate | >50% (all records already cached → stale config?) | Warn |
| SigLIP coverage | <90% of shard records have siglip precomputed | Block training start |
| Hard example loss | Not decreasing across chunks (adapter not learning) | Log warning after chunk 2 |
| Process heartbeat | Missing >90 s | Crash detection → restart |

---

## 6. Storage Lifecycle

Artefact lifecycle — what to keep and when to free:

```
raw/{source}/chunk{N}/          → delete after converter completes chunk N
converted/{source}/chunk{N}/    → delete after builder completes chunk N
shards/chunk{N}/                → keep until end of pipeline (trainer + miner need it)
precomputed/chunk{N}/           → keep until end of pipeline
hard_examples/chunk{N}/         → keep until end of pipeline
checkpoints/stage1/step_*.sft   → keep last 3 only; delete older on new save
checkpoints/stage1/best.sft     → keep always
dedup_ids/                      → keep across all chunks (cumulative index)
logs/                           → keep always
```

### 6.1 Space budget (per chunk, worst case)

| Artefact | Size estimate |
|----------|--------------|
| raw/chunk{N} | ~20 GB (images compressed) |
| converted/chunk{N} | ~15 GB (normalised JPEG) |
| shards/chunk{N} | ~10 GB (WebDataset tars) |
| precomputed/chunk{N} | ~8 GB (embeddings + latents) |
| hard_examples/chunk{N} | ~1 GB |
| checkpoints (rolling 3) | ~6 GB |
| **Total peak** | **~60 GB** |

Peak disk usage occurs during builder phase when raw + converted + shards all coexist.  
After builder: raw + converted can be freed → peak drops to ~25 GB ongoing.

---

## 7. Parallelism Map

Explicit map of what can and cannot run simultaneously.

### 7.1 Safe parallel combinations

```
downloader(chunk N+1)  +  trainer(chunk N)          ← net vs GPU, no conflict
downloader(chunk N+1)  +  validator(chunk N)         ← net vs GPU light
converter(chunk N+1)   +  trainer(chunk N)           ← CPU vs GPU, no conflict
converter(chunk N+1)   +  validator(chunk N)         ← CPU vs GPU light
downloader             +  converter                  ← pipeline the two CPU steps
```

### 7.2 Serial constraints (cannot overlap)

```
precomputer  ✗  trainer      — both hold GPU_TOKEN exclusively
precomputer  ✗  miner        — both hold GPU_TOKEN
trainer      ✗  miner        — both hold GPU_TOKEN
builder      ✗  precomputer  — both hold DISK_WRITE_HIGH
dedup (embed)✗  trainer      — GPU_SHARE conflicts with GPU_TOKEN
```

### 7.3 Recommended chunk timeline

```
          CHUNK 1                     CHUNK 2                    CHUNK 3
download  [====]
convert        [==]
dedup            [=]
build              [=]
precompute          [====]
train                    [==========]
mine                                  [==]
validate                                [=]
                         download  [====]
                         convert        [==]
                         dedup            [=]
                         build              [=]
                         precompute          [====]
                         train                    [==========]
                         mine                                 [==]
                         validate                               [=]
```

Download, convert, dedup, and build for chunk N+1 are all complete before training chunk N ends — zero training dead time between chunks.

---

## 8. Recipe Registry

The pipeline is parameterised by a recipe YAML. The orchestrator loads the recipe and instantiates the correct trainer, precomputer arguments, and validation checks.

```yaml
# recipes/ip_adapter_flux4b.yaml
name: ip_adapter_flux4b
model: flux-klein-4b
adapter_type: ip_adapter
precompute:
  siglip: true
  qwen3: true
  vae: true
training:
  batch_size: 4
  steps_per_chunk: 65000
  hard_mix_ratio: 0.05
  lr: 1e-4
  grad_clip: 1.0
data:
  sources: [laion, coyo, wikiart, journeydb]
  chunks: 4
  shard_size: 5000
validation:
  fid_proxy: true
  hard_example_trend: true
  siglip_coverage_min: 0.90
```

```yaml
# recipes/lora_flux4b.yaml  (future)
name: lora_flux4b
model: flux-klein-4b
adapter_type: lora
precompute:
  siglip: false
  qwen3: true
  vae: true
training:
  batch_size: 8
  steps_per_chunk: 20000
  hard_mix_ratio: 0.0
  lr: 5e-5
```

Adding a new training objective requires only a new recipe YAML. No orchestrator or worker code changes needed for standard precompute/train/validate flow.

---

## 9. Directory Structure

```
/Volumes/2TBSSD/
├── raw/
│   ├── laion/chunk1/        ← deleted after convert
│   ├── coyo/chunk1/
│   ├── wikiart/chunk1/
│   └── journeydb/chunk1/
├── converted/
│   ├── laion/chunk1/        ← deleted after build
│   └── ...
├── shards/
│   ├── chunk1/000000.tar ... 000040.tar
│   └── chunk2/
├── precomputed/
│   ├── chunk1/
│   └── chunk2/
├── hard_examples/
│   ├── chunk1/
│   └── chunk2/
├── checkpoints/
│   └── stage1/
│       ├── step_065000.safetensors
│       ├── step_065000.json        ← lineage sidecar
│       └── best.safetensors
├── dedup_ids/                      ← cumulative cross-chunk FAISS index
├── logs/
│   ├── orchestrator.jsonl
│   ├── trainer_chunk1.jsonl
│   └── ...
├── pipeline_state.json             ← orchestrator state (atomic writes)
├── dispatch_queue.jsonl            ← issues for Claude Dispatch
├── dispatch_responses.jsonl        ← responses from Claude Dispatch
└── pipeline_control.json           ← CLI control signals
```

```
/Users/fredrikhult/src/iris.c/train/
├── scripts/
│   ├── orchestrator.py             ← NEW: orchestrator state machine
│   ├── downloader.py               ← NEW: per-source download worker
│   ├── converter.py                ← NEW: raw→JPEG normaliser
│   ├── deduplicator.py             ← refactored from embed + FAISS steps
│   ├── build_shards.py             ← refactored (fix incremental bug)
│   ├── precompute.py               ← refactored
│   ├── train.py                    ← refactored (emit JSONL events)
│   ├── mine_hard_examples.py       ← refactored
│   ├── validator.py                ← NEW: post-chunk validation
│   ├── pipeline_status.py          ← refactored (JSON-first output)
│   └── pipeline_ctl.py             ← NEW: CLI control interface
├── recipes/
│   └── ip_adapter_flux4b.yaml
└── config/
    └── stage1.yaml
```

---

## 10. Pre-flight Doctor

Before any chunk starts the orchestrator runs a doctor check. All checks must pass before resources are allocated:

| Check | Failure action |
|-------|---------------|
| Python ≥ 3.12 | Abort |
| MLX ≥ 0.31.1 | Abort |
| Required packages importable (turbojpeg, safetensors, faiss, open_clip) | Abort |
| Model weights directory exists and non-empty | Abort |
| DATA_ROOT writable | Abort |
| DATA_ROOT free space ≥ 60 GB | Warn if <80 GB, abort if <40 GB |
| Recipe YAML valid and all referenced sources defined | Abort |
| Previous chunk checkpoint exists (if resuming) | Abort |
| tmux available | Abort |
| No stale process heartbeats from previous crashed run | Warn + offer cleanup |

---

## 11. Serial Processing: Explicit Callouts

The following steps are serial by design and cannot be parallelised without architectural changes noted:

| Step | Why serial | Cost | Mitigation |
|------|-----------|------|-----------|
| FAISS index update (dedup) | Single writer; concurrent writes corrupt index | ~30 min/chunk | Acceptable; run during training of previous chunk |
| GPU inference (precompute/train/mine/validate) | M1 Max 32 GB shared GPU memory; two GPU processes thrash each other | Dominant cost | Schedule strictly: train→mine→validate→train next chunk |
| Shard build from dedup output | Must read dedup_ids which is finalized only after dedup completes | ~15 min/chunk | Run immediately after dedup; no wait |
| Checkpoint save (trainer) | Atomic file write; no parallel saves | Seconds | Non-issue |

---

## 12. Implementation Plan

### Phase 0 — Clean slate (before any code)

1. Confirm data to delete with user: all `/Volumes/2TBSSD/` pipeline artefacts except `checkpoints/`
2. Create V2 directory structure (Section 9) on the SSD
3. Verify `best.safetensors` checkpoint integrity (weight sanity check)

### Phase 1 — Core infrastructure

Goal: working orchestrator that can drive a single chunk end-to-end for all 4 sources.

| Order | Script | What it does |
|-------|--------|-------------|
| 1 | `pipeline_state.json` schema | Write `write_state()` function used by all scripts |
| 2 | `orchestrator.py` | State machine + resource tokens + heartbeat monitor + dispatch interface |
| 3 | `downloader.py` | Per-source download; JDB uses producer-consumer (MLX-19); LAION/COYO/WikiArt batch |
| 4 | `converter.py` | Consumer thread; deletes raw after convert |
| 5 | `build_shards.py` refactor | All-source support; incremental fix already committed |
| 6 | `pipeline_status.py` refactor | JSON-first; reads state file + heartbeats |
| 7 | `pipeline_ctl.py` | pause/resume/abort/restart-orchestrator |
| 8 | Doctor check | Run at orchestrator startup; fail fast on bad env |
| 9 | Storage lifecycle | Orchestrator deletes `raw/` and `converted/` after `build_shards.done` |

Milestone: run chunk 1 of all 4 sources through orchestrator to `precompute.done`.

### Phase 2 — Quality + observability

Goal: autonomous quality gates and sufficient telemetry for error detection.

| Order | Item | What it does |
|-------|------|-------------|
| 1 | T-01/02/03 | Grad norm, per-bucket stats, memory pressure in trainer |
| 2 | MLX-22 first pass | V-01 weight integrity + V-03/04 CLIP scoring + V-07 visual grid |
| 3 | MLX-22 second pass | V-05 no-adapter delta + V-08 regression check |
| 4 | Orchestrator anomaly rules | Section 5.1 thresholds wired to dispatch escalation |
| 5 | MLX-26 | A/B comparison (reuses MLX-22 scoring) |
| 6 | T-04–T-10 | Remaining telemetry metrics |

Milestone: chunk N training completes → validator runs → orchestrator advances or escalates autonomously.

### Phase 3 — End product

| Order | Item | What it does |
|-------|------|-------------|
| 1 | `extract_sref.py` | SigLIP feature extraction to `.sref` binary |
| 2 | `iris_ip_adapter.c` | Perceiver resampler + weight loading |
| 3 | IP K/V injection | Double-stream blocks (5), validate parity with Python |
| 4 | IP K/V injection | Single-stream blocks (20) |
| 5 | Metal GPU path | Cached projections, non-cached resampler |
| 6 | CLI wiring | `--sref`, `--ip-adapter`, `--sref-scale` |

MLX-23 (Phase 3) can start in parallel with Phase 1 as it is independent of the pipeline refactor.

---

## 13. Open Questions / Deferred Decisions

1. **LAION/COYO manifest**: Do we have pre-generated per-chunk URL lists, or does the downloader need to slice them itself? If sliced at download time, need a deterministic split (e.g. sorted by URL hash).

2. **WikiArt chunking**: WikiArt is a fixed dataset (~90 K images). Splitting into 4 equal slices of ~22 K is fine. Need to verify the download source and whether it can be re-downloaded cleanly.

3. **Dedup index across sources**: Current dedup indexes all sources together (correct — avoids near-duplicates across sources). V2 must maintain this: the FAISS index is cumulative across all sources AND all previous chunks.

4. **Hard example mixing**: In V2, hard examples come from the previous chunk's miner pass. Chunk 1 has no hard examples (acceptable). Chunks 2-4 mix in chunk N-1 hard examples at `hard_mix_ratio`.

5. **Steps-per-chunk calibration**: With ~41 shards/chunk × 5000 images = 205 K images, at batch_size=4 and 3 epochs equivalent = ~150 K steps. This is 2× the V1 per-chunk budget. Need to re-calibrate or reduce shard count.

6. **SigLIP precompute for non-JDB sources**: LAION/COYO/WikiArt records need SigLIP features too (IP-Adapter training). V1 may have only precomputed these for JDB. Confirm before rebuild.
