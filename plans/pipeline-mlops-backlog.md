# Pipeline MLOps Backlog

Updated 2026-04-19 after Phase 2 completion.
Dispositions: ✅ DONE | ➡ ABSORBED INTO V2 | 🗑 RETIRED | ✳ INCLUDE

See `plans/pipeline-v2-architecture.md` for the design that governs open items.

---

## Status Summary

| Item | Title | Disposition |
|------|-------|------------|
| MLX-10 | Model lineage in checkpoints | ✅ DONE |
| MLX-11 | Pre-flight doctor check | ✅ DONE |
| MLX-12 | Mining failure silent in pipeline log | ✅ DONE |
| MLX-13 | Training stall alerting | ✅ DONE |
| MLX-14 | build_shards filter window | ✅ DONE |
| MLX-15 | SigLIP coverage gate | ✅ DONE |
| MLX-16 | Front-run JDB conversion | 🗑 RETIRED — orchestrator does this for all sources |
| MLX-17 | Full parallel data pipeline | 🗑 RETIRED — superseded by V2 resource model |
| MLX-18 | Pipeline orchestrator | ➡ ABSORBED — V2 Section 4 (Python, not bash) |
| MLX-19 | Producer-consumer download | ✅ DONE — Phase 1 (download_convert.py) |
| MLX-20 | Pipeline state file | ➡ ABSORBED — V2 Section 4.1 (extended schema) |
| MLX-21 T-01/02/03 | Telemetry: grad norm, bucket, memory | ✅ DONE — Phase 2 |
| MLX-21 T-04–T-10 | Telemetry: remaining 7 metrics | ✅ DONE — Phase 2 |
| MLX-22 | Post-training validation suite | ✅ DONE — Phase 2 |
| MLX-23 | `--sref` in iris.c | ✳ INCLUDE — Phase 3 |
| MLX-24 | LAION/WikiArt resampling | 🗑 RETIRED — equal chunking in V2 solves root cause |
| MLX-25 | Staging architecture | ➡ ABSORBED — V2 Section 9 (directory + promotion logic) |
| MLX-26 | A/B weight comparison | ✳ INCLUDE — Phase 3 (deferred until after pipeline test run) |
| MLX-27 | pipeline_status.py intelligence uplift | ✳ INCLUDE — Phase 3 |

---

## Implementation Phases

### Phase 0 — Clean slate
- Delete all `/Volumes/2TBSSD/` pipeline data
- Verify: shards, precomputed, hard_examples, raw, converted, embeddings, dedup_ids all cleared
- Keep: checkpoints (best.safetensors is the trained artifact), logs (history)
- Create new directory structure per V2 Section 9

### Phase 1 — Core infrastructure
Goal: working orchestrator + data pipeline for all 4 sources in chunked mode.

1. `orchestrator.py` — Python state machine (V2 Section 4); reads/writes `pipeline_state.json`
2. `downloader.py` — per-source download worker with `.ready` sentinels (MLX-19)
3. `converter.py` — consumer thread; deletes raw after convert to save disk (MLX-19)
4. `build_shards.py` — refactor: fix incremental bug (committed), all-source support
5. `pipeline_state.json` — schema per V2 Section 4.1; written by orchestrator
6. `pipeline_status.py` — JSON-first; reads state file + heartbeats
7. `pipeline_ctl.py` — pause/resume/abort/restart-orchestrator control signals
8. Doctor check — V2 Section 10 checks run at orchestrator startup
9. Storage lifecycle — orchestrator deletes raw/converted after build (V2 Section 6)

### Phase 2 — Quality + observability ✅ COMPLETE
Goal: autonomous quality gates and sufficient telemetry for error detection.

1. ✅ MLX-21 T-01/02/03 — grad norm (EMA + spike alert), per-bucket throughput, memory pressure
2. ✅ MLX-22 — validation suite: weight integrity (V-01), CLIP-I/CLIP-T (V-03/04), visual grid (V-07)
3. ✅ MLX-21 T-04–T-10 — per-source loss (T-04), val loss held-out (T-05), EMA drift (T-06),
   loader wait % (T-07), filter rejection by source (T-08), caption length distribution (T-09),
   SigLIP coverage per batch (T-10)
4. ✅ Anomaly detection — orchestrator: heartbeat staleness→restart, loss NaN/high→pause,
   grad spike→pause; SigLIP coverage warning

### Phase 3 — End product + post-test tools
1. MLX-23 — `--sref` in iris.c binary
2. MLX-26 — A/B weight comparison (deferred until after pipeline test run)

---

## Retired items (collapsed)

### 🗑 MLX-16 — Front-run JDB conversion
Superseded. V2 orchestrator downloads, converts, and builds shards for all 4 sources during the prior training window automatically. JDB is no longer special-cased.

### 🗑 MLX-17 — Full parallel data pipeline
Superseded. V2 resource model (Section 3.2, 7) formalises exactly which steps can run alongside training. The specific bash pseudocode is replaced by the Python orchestrator.

### 🗑 MLX-24 — LAION/WikiArt resampling at promotion
Superseded. V2 equal-chunk data strategy (Section 2.2) ensures all sources contribute a fresh 25% slice per chunk. The diversity drift problem that MLX-24 addressed is prevented by design. Revisit after first V2 training run if per-source loss (T-04) shows drift.

---

## Absorbed items (collapsed)

### ➡ MLX-18 — Pipeline orchestrator
Core design absorbed into V2 Section 4. Key decisions carried forward:
- tmux session layout: `iris-orch`, `iris-train`, `iris-prep`, `iris-watchdog`
- GPU-free detection: `iris-train` window absence
- Per-step sentinel files in `pipeline/chunk{N}/{step}.done`
- `notify_error()` via `osascript`

Key changes from MLX-18:
- Implementation language: Python (not bash) for structured telemetry and error handling
- State stored in `pipeline_state.json` (not sentinel-only)
- Dispatch interface added (Dispatch/CLI/web)
- Autonomous restart on crash (one retry before escalate)
- Resource token system replaces manual GPU checks

Sentinel schema from MLX-18 is adopted verbatim in V2 Section 9.

### ➡ MLX-20 — Pipeline state file
Schema absorbed into V2 Section 4.1 with extensions:
- Added `issues[]` array for orchestrator-level anomaly log
- Added `run_id` for multi-run disambiguation
- `current_run.training_pid` for crash detection
- `schema_version: 2` (V2)

The `notes` free-text field is preserved — valuable for cross-session context.
`DISPATCH.md` must be updated to read state file first in every session.

### ➡ MLX-25 — Staging architecture
Directory structure absorbed into V2 Section 9. Promotion logic (`promote_chunk()`) is implemented in `orchestrator.py`. Sentinel refactor table (old paths → new paths) still needed for migration — document in `DISPATCH.md`.

Key decisions carried forward:
- `staging/chunk{N}/` isolation before promotion
- `chunk{N}_NNNN.tar` prefix naming enables per-chunk rollback
- Promotion is atomic (temp + rename or short serial move)
- SigLIP precompute required before mining; null-siglip flag when not precomputed

---

## Open items (full spec)

---

### ✳ MLX-19 — Producer-consumer download+conversion (Phase 1)

**Problem:** Downloading all N tgzs before starting conversion serialises download+convert wall-clock. For chunk 4 (52 tgzs × ~16 GB), holding all tgzs simultaneously requires ~832 GB — more than the SSD.

**Fix:** Producer-consumer pattern. Applies to JourneyDB (large tgzs). LAION/COYO/WikiArt can use simpler sequential download if files are smaller.

**Disk benefit:** Peak raw storage is 1-2 tgzs in-flight (~32 GB) rather than all N simultaneously.

**Wall-clock saving:** `(N-1) × convert_time_per_tgz`. For 52 tgzs at 15 min each: ~12h saved for chunk 4.

**Implementation:**

```python
# train/scripts/downloader.py  (JDB path)
import threading, time
from pathlib import Path
from huggingface_hub import hf_hub_download

def downloader_thread(data_root, tgz_range, sentinel_dir):
    for i in tgz_range:
        converted = sentinel_dir / f"{i:03d}.converted"
        ready     = sentinel_dir / f"{i:03d}.ready"
        if converted.exists():
            continue
        if not ready.exists():
            hf_hub_download(
                repo_id="JourneyDB/JourneyDB", repo_type="dataset",
                filename=f"data/train/imgs/{i:03d}.tgz",
                local_dir=str(data_root / "raw" / "journeydb"),
            )
            ready.touch()

def converter_thread(data_root, tgz_range, sentinel_dir, output_dir, poll=30):
    pending = set(tgz_range)
    while pending:
        for i in sorted(pending):
            ready     = sentinel_dir / f"{i:03d}.ready"
            converted = sentinel_dir / f"{i:03d}.converted"
            tgz_path  = data_root / "raw" / "journeydb" / "data" / "train" / "imgs" / f"{i:03d}.tgz"
            if ready.exists() and not converted.exists():
                run_convert(data_root, output_dir, tgz_index=i)
                tgz_path.unlink(missing_ok=True)   # free raw tgz immediately
                converted.touch()
            if converted.exists():
                pending.discard(i)
        if pending:
            time.sleep(poll)
```

**Resume behaviour:** Sentinel files make this fully resumable. `hf_hub_download` handles interrupted downloads via ETag cache.

**Integration:** `downloader.py` is launched by the orchestrator when the `DOWNLOADING` state is entered for a chunk. The orchestrator transitions to `CONVERTING` state when all `.ready` sentinels exist; `BUILDING` when all `.converted` sentinels exist.

**Also download annotation file first** (`train_anno_realease_repath.jsonl.tgz`, ~200 MB) — converter needs it.

---

### ✳ MLX-21 — Extended telemetry (Phase 2)

Ten metrics for training quality and performance. See V2 Section 5 for the JSONL event schema.

#### T-01 — Gradient norm (HIGH, implement first)

```python
# in train_ip_adapter.py, after mx.eval(grads):
grad_norm = float(mx.sqrt(sum(mx.sum(g**2) for g in mx.utils.tree_flatten(grads)[0])))
# emit to heartbeat: "grad_norm": float, "grad_norm_smooth": float (EMA over 50 steps)
# alert: if grad_norm > 10 × grad_norm_smooth → log WARNING
```

#### T-02 — Per-bucket throughput + loss (HIGH, implement first)

```python
bucket_key = f"{H}x{W}"
bucket_stats[bucket_key]["steps"] += 1
bucket_stats[bucket_key]["loss"]  += loss_val
bucket_stats[bucket_key]["time"]  += step_time
# emit to heartbeat: "buckets": {"512x512": {"steps": N, "loss_avg": X, "secs_avg": X}}
```

#### T-03 — Memory pressure (HIGH, implement first)

```python
import psutil
mem = psutil.virtual_memory()
# emit: "mem_used_gb": float, "mem_available_gb": float
# alert: if mem_available_gb < 6.0 → log WARNING: memory pressure high
```

#### T-04 — Per-dataset loss breakdown (MEDIUM)

```python
source = batch.get("source", "unknown")  # shard must tag source in __key__ or metadata
source_loss[source].append(float(loss_val))
# emit rolling 100-step average per source
```

Note: shards must carry a `__source__` field or source must be inferrable from shard name prefix (`chunk{N}_laion_*`).

#### T-05 — Validation loss on held-out set (MEDIUM)

Requires: curate ~200-image held-out set from LAION+WikiArt (one-time, ~1h effort).
Store in `$DATA_ROOT/validation/held_out/`.

```python
if step % cfg.get("val_every", 1000) == 0:
    val_loss = compute_val_loss(model, val_loader)  # no-grad forward pass
    # emit to heartbeat and append to $CKPT_DIR/val_loss.jsonl
```

#### T-06 — EMA vs online weight divergence (MEDIUM)

```python
if step % 500 == 0:
    online_w = adapter.ip_proj[0].weight
    ema_w    = ema_params["ip_proj"][0]["weight"]
    ema_drift = float(mx.mean((online_w - ema_w)**2)**0.5)
    # emit: "ema_drift": float
```

#### T-07 — Data loader wait time (MEDIUM)

```python
t0 = time.perf_counter()
batch = next(data_iter)
loader_wait_ms = (time.perf_counter() - t0) * 1000
# emit: "loader_wait_ms": float, "compute_ms": float, "loader_pct": float
# if loader_pct > 20% → investigate prefetch depth
```

#### T-08 — Filter rejection rate per source (MEDIUM)

Implement in `filter_shards.py`. Add source tag parsing from shard filename.
Emit to `$DATA_ROOT/logs/filter_stats_chunk{N}.json` on completion.

Important in V2: with all sources chunked, per-source rejection rates inform download budget calculations. If JDB chunk 3 rejects 30%, download 43% more tgzs.

#### T-09 — Caption length distribution (LOW)

Sample 10% of shard records in `filter_shards.py` or `pipeline_ctl.py doctor`.
Emit to `$DATA_ROOT/logs/caption_stats_chunk{N}.json`.

#### T-10 — SigLIP coverage per batch (LOW)

```python
siglip_miss_count += int(siglip_was_zeros)
# emit rolling: "siglip_coverage_pct": float
# alert if < 90%
```

#### Implementation order

Phase 2 first pass: T-01, T-02, T-03 (all in `train_ip_adapter.py`, low risk, high signal)
Phase 2 second pass: T-04, T-05, T-06, T-07, T-08, T-09, T-10

---

### ✳ MLX-22 — Post-training validation suite (Phase 2)

Run after each chunk's training completes, before launching next chunk. The validator is a separate process (V2 Section 3.1) that writes `pipeline/chunk{N}/validation.done` or `validation.fail` for the orchestrator.

**Components (implement in order):**

| ID | Check | Time | Priority |
|----|-------|------|----------|
| V-01 | Weight integrity (shapes, NaN/Inf, ip_scale range) | ~10s | First |
| V-03 | CLIP-I style similarity (output vs reference) | ~2 min | First |
| V-04 | CLIP-T prompt alignment (output vs prompt) | ~1 min | First |
| V-07 | Visual grid (reference + with/without adapter + prev chunk) | ~1 min | First |
| V-05 | No-adapter delta (net adapter contribution) | ~3 min | Second |
| V-08 | Regression vs prev chunk checkpoint | ~3 min | Second |
| V-02 | Smoke test (no crash, no blank image) | ~3 min | First (implicit in V-03) |
| V-06 | EMA vs online weight comparison | ~3 min | Second |

**Verdict logic:**
- `PASS`: mean_clip_i > 0.20 AND mean_adapter_delta > 0.05 AND no weight integrity errors
- `WARN`: mean_clip_i > 0.15 AND mean_adapter_delta > 0.0
- `FAIL`: integrity error OR mean_clip_i < 0.15 OR mean_adapter_delta ≤ 0

**Orchestrator integration:** Orchestrator reads `validation.done` before advancing to next chunk. On `FAIL`, writes `pipeline/chunk{N}/validation.error` and escalates to dispatch. Does not block the operator from force-advancing with `pipeline_ctl.py force-next-chunk N`.

**Files to create:**

| File | Purpose |
|------|---------|
| `train/scripts/validator.py` | Orchestrator entry point; runs checks, writes sentinel |
| `train/scripts/validate_weights.py` | V-01 weight integrity |
| `train/scripts/run_inference.py` | Python MLX inference with IP-Adapter (for V-02/03/04/05) |
| `train/scripts/score_validation.py` | CLIP-I/CLIP-T scoring, regression delta |
| `train/scripts/render_validation_grid.py` | V-07 visual grid |
| `train/configs/eval_prompts.txt` | Already exists — expand to 10 (prompt, sref_path) pairs |
| `train/eval_refs/` | 10 reference images for eval pairs (curate manually) |

**Note on `run_inference.py`:** This is the Python-side `--sref` inference path, enabling validation before MLX-23 (`--sref` in iris.c) is complete. Reuses the forward pass from `train_ip_adapter.py` in no-grad eval mode, with `ip_scale` loaded from checkpoint unchanged.

---

### ✳ MLX-26 — A/B weight comparison (Phase 2, after MLX-22)

Compares two checkpoints on identical (prompt, reference) pairs. Reuses `run_inference.py` and `score_validation.py` from MLX-22.

**Interface:**
```bash
# Compare two checkpoints:
python train/scripts/ab_compare.py \
    --a checkpoints/stage1/step_050000_ema.safetensors --label-a chunk1 \
    --b checkpoints/stage1/step_065000_ema.safetensors --label-b chunk2

# Batch: evolution curve across all step_*.safetensors vs baseline:
python train/scripts/ab_compare.py --baseline best \
    --candidates "checkpoints/stage1/step_*.safetensors"
```

**Output:**
- `checkpoints/stage1/ab_results/{timestamp}.json` — per-pair CLIP-I/CLIP-T scores + summary
- `checkpoints/stage1/ab_results/{timestamp}_grid.png` — visual side-by-side with winner labels
- Batch mode: CLIP-I curve across steps (detect diminishing returns, find best checkpoint)

**Confidence scoring:**
- `high`: B wins ≥4/5 pairs AND mean CLIP-I delta > 0.04
- `medium`: ≥3/5 pairs AND delta > 0.02
- `low`: split result or delta < 0.02 — models comparable

**Seed stability test:** Generate same (prompt, ref) pair with N=5 seeds per checkpoint. `std(CLIP-I)` measures sensitivity to initialisation noise. Lower is better.

**Files to create:**

| File | Purpose |
|------|---------|
| `train/scripts/ab_compare.py` | Core: inference × 2, CLIP scoring, JSON output |
| `train/scripts/render_ab_grid.py` | Visual grid with per-pair winner labels |

---

### ✳ MLX-27 — pipeline_status.py intelligence uplift (Phase 3)

**Problem:** The status script is too simplistic — it shows step names and sentinel flags but not what's actually happening. Claude can diagnose more from log files, heartbeats, and sentinel bodies than the script surfaces to the operator.

**Goal:** Status output should tell the operator everything they need without having to grep logs manually.

**Required improvements:**

- **Error text from sentinels** — Read `.error` sentinel file body and print the message inline (not just a flag)
- **Per-worker heartbeats** — Show heartbeat age and staleness for all workers (clip_dedup, precompute), not just trainer
- **Active step log tail** — For the running step, show last 5 lines of its log file; helps spot Python tracebacks, hung processes, etc.
- **clip_embed progress** — clip_dedup.py already writes a heartbeat with `done/total/pct`; surface it
- **precompute progress** — If precompute_all.py writes a heartbeat, surface `done/total/pct`
- **ETA estimates** — Where progress % and elapsed time are known, compute and show ETA
- **Download/convert progress** — Count `.converted` sentinels vs total expected tgzs for the chunk
- **Per-chunk staging detail** — Shard count and precomputed file count per staging chunk dir, not just aggregate
- **Error recovery hint** — When in ERROR state, print the sentinel path the operator must delete to retry
- **`--verbose` mode** — Longer log tail (20 lines), full heartbeat JSON dump, all sentinel ages

**Implementation notes:**
- `pipeline_lib.py` already has `read_heartbeat()` — extend to all worker names
- Error sentinel body is written by `mark_error(chunk, step, msg)` in `pipeline_lib.py` — read it back
- Log tail: use `pathlib.Path.read_text().splitlines()[-N:]` — no subprocess needed
- Keep default output concise; gate verbosity behind `--verbose`

---

### ✳ MLX-23 — Implement `--sref` in iris.c (parallel track)

The end-product goal. Can be developed in parallel with pipeline V2 work.

**Architecture:** IP-Adapter adds cross-attention to 25 Flux blocks (5 double + 20 single).
SigLIP features extracted by Python sidecar; iris.c reads `.sref` binary file.

**User-facing API:**
```bash
./iris -d flux-klein-4b --sref portrait.jpg --ip-adapter weights.safetensors \
    -p "a cat" --sref-scale 0.7 -o out.png
```

**SigLIP sidecar pattern:**
1. iris.c checks for `portrait.jpg.sref` (float16 binary, shape [729, 1152])
2. If absent: executes `python3 train/scripts/extract_sref.py portrait.jpg portrait.jpg.sref`
3. Reads `.sref` → promotes to float32 internally

Multiple `--sref` flags: average features across up to 4 reference images.

**Adapter weight structure (from `train/ip_adapter/model.py`):**
```
image_proj.*       — Perceiver resampler: siglip [729,1152] → ip_embeds [128,3072]
to_k_ip_stacked    — [25, 3072, 3072]  stacked K projections
to_v_ip_stacked    — [25, 3072, 3072]  stacked V projections
ip_scale           — [25]  per-block attention scale (trained, ~0.5–1.5)
```

**C implementation steps:**

1. `extract_sref.py` — SigLIP feature extraction to `.sref` binary format
2. `iris_ip_adapter.c/.h` — `iris_ip_adapter_t` struct + weight loading from safetensors
3. Perceiver resampler forward: siglip [729,1152] → [128,3072] via cross-attention (reuse `flux_attention()`)
4. IP K/V precompute: project ip_embeds through all 25 `to_k_ip`/`to_v_ip` matrices once before denoising loop
5. Inject into double-stream blocks first (5 blocks) — validate output matches Python
6. Extend to single-stream blocks (20 blocks)
7. Metal GPU path: `flux_metal_sgemm_cached()` for static `to_k_ip`/`to_v_ip`; `flux_metal_sgemm()` for dynamic resampler cross-attention
8. CLI wiring: `--sref PATH`, `--ip-adapter PATH`, `--sref-scale N`, `--sref-precompute`

**Validation criteria (links to MLX-22):**
- V-01: weight integrity PASS
- V-03: iris.c CLIP-I within ±0.02 of Python `run_inference.py` output (confirms parity)
- V-07: visual grid shows matching style between Python and C paths

**Implementation note:** `--ip-adapter` weights are bfloat16 in the checkpoint. Convert to float32 at load time using existing `iris_safetensors.c` infrastructure. Static weight projections use the `_cached()` Metal path. Resampler cross-attention uses non-cached path (dynamic per image).
