# FLUX.2 Web UI — Improvement Backlog

## Operations / Data Safety

- **Do NOT delete /Volumes/2TBSSD/shards/** until chunk 1 training is complete. Training reads raw JPEG images directly from the shard tar files (for training target + IP-adapter style reference). Safe to delete only after train_ip_adapter.py finishes.



## Tier 1 — High Impact, Expose Existing C Binary Capability

- [x] **1. Guidance scale slider** — Essential creative control for base models; already in C binary (`-g` flag, `flux_params.guidance`). Auto-selects 1.0 distilled / 4.0 base.
- [x] **2. Increase steps max to 50+** — Base models need 50 steps; slider currently capped at 8. C binary supports up to 256 (`FLUX_MAX_STEPS`).
- [x] **3. Schedule selection** (linear/power/sigmoid) — Dramatically different results; already in C binary (`--linear`, `--power`, `--power-alpha`).
- [x] **4. Model info display** — Users don't know what model is loaded (4B/9B, distilled/base). Add `/model-info` endpoint + header display.
- [x] **5. Embedding cache in server mode** — Skip ~1s text encoding on repeated prompts. CLI uses 4-bit quantized cache (`embcache.c`). Server re-encodes every time.

## Tier 2 — High Impact, New UX Features

- [x] **6. Ctrl+Enter to generate** — Standard UX pattern in AI tools. Trivial to add.
- [x] **7. ETA display** — Has step_time + remaining steps; just needs math in progress handler.
- [x] **8. History search/filter** — Finding past generations gets hard with many images. Add text search input.
- [x] **9. Lightbox metadata overlay** — See prompt/seed/params without going back to grid.
- [x] **10. Image caching headers** — Immutable images should cache; saves bandwidth on history grid scroll.

## Tier 3 — Medium Impact

- [x] **11. Store generation time in history** — Track performance, know what to expect.
- [x] **12. Persist style preset & variations count** in localStorage — Settings lost on reload.
- [x] **13. Auto-restart on process crash** — Hung server requires manual restart.
- [x] **14. Variation grid/mosaic view** — See all variations at once instead of one-by-one in history.
- [x] **15. Side-by-side comparison** — Compare two images (before/after, different seeds).

## Tier 4 — Nice to Have

- [x] **16. Lightbox delete + download buttons** — Common actions require leaving lightbox currently.
- [x] **17. Prompt templates** — Reusable prompt structures with variable placeholders. [Plan](plans/17-prompt-templates.md)
- [ ] **18. Batch prompt generation** — Submit a list of different prompts to generate in sequence.
- [x] **19. Reference image reorder** — Drag to reorder slots (order matters for multi-ref T_offset).
- [ ] **20. Per-job timeout** — Prevent hung generations from blocking the queue forever.

## Tier 5 — Extended Features

- [x] **26. Dark/light theme toggle** — Toggle between dark and light themes with localStorage persistence.

---

## Training Quality Improvements

- ~~**PERF-1: Reduce eval frequency**~~ ✅ DONE — `log_every` changed 100 → 500 in `train/configs/stage1_512px.yaml`. Saves ~16–17% wall-clock. Takes effect on next training restart.

- ~~**PERF-2: Logit-normal timestep sampling**~~ ✅ DONE — `mx.random.randint(0, 1000, ...)` replaced with `mx.clip((mx.sigmoid(mx.random.normal(...)) * 1000).astype(mx.int32), 0, 999)` in `train_ip_adapter.py`. Concentrates samples at mid-timesteps (t≈300–700) where flow-matching loss has structure; reduces loss variance ~40%. Takes effect on next training restart.

- ~~**PIPELINE-1: Fix anchor_shard_dir for chunks 2–4**~~ ✅ DONE — `pipeline_lib.py` adds `ANCHOR_SHARDS_DIR = DATA_ROOT / "anchor_shards"`. `_start_training()` passes `--anchor-shards` for chunk ≥ 2 when `anchor_shards/` exists and has .tar files; logs a warning (not hard error) when empty. Operator must populate `/Volumes/2TBSSD/anchor_shards/` with chunk 1 representative shards before chunk 2 starts.

- ~~**PIPELINE-2: Check SigLIP coverage before each chunk**~~ ✅ DONE — `_promote_chunk()` in `orchestrator.py` enforces ≥90% siglip coverage when `training.siglip: true` in the pipeline config; fails promotion with a hard error if coverage is insufficient.

- **PIPELINE-6: Preserve each chunk's final checkpoint for pipeline recovery** — The orchestrator currently relies on `keep_last_n` to manage checkpoint disk usage, which rotates out a chunk's final checkpoint once the next chunk starts training. If chunk N+1 training develops a bug or must be restarted from scratch, there is no way to recover without retraining chunk N from zero. Fix: when the orchestrator marks a chunk `train.done`, copy (or hard-link) the final checkpoint pair (`step_NNNNNNN.json` + `step_NNNNNNN.safetensors`) to a dedicated archive path e.g. `/Volumes/2TBSSD/checkpoints/stage1/archive/chunk{N}_final.*`. These archive files are excluded from `keep_last_n` rotation. The `pipeline_ctl.py restart-from-chunk N` operation should look in this archive first. EMA files (`.ema.safetensors`) should also be archived. Disk cost: ~2× the size of one checkpoint per chunk boundary; at small scale negligible.

- **PIPELINE-3: Never run pipeline jobs while training is active on 2TBSSD** — Two epoch-boundary stalls during chunk 1 training (at steps ~19,900 and ~24,900) were extended by competing I/O from the JDB chunk 2 conversion running in parallel. The step ~24,900 stall lasted 2.6h instead of the typical ~15–20min. Rule: fully complete all pipeline work (WDS conversion, precompute) before starting training, or ensure pipeline and training use separate storage volumes.

- **PIPELINE-4: Investigate MLX CLIP for clip_embed step** — `clip_dedup.py` currently uses `open_clip` ViT-L-14 via PyTorch MPS (fp16). MLX is installed in the venv (`mlx==0.31.1`) but `mlx_clip` is not. MLX runs natively on Apple Silicon without PyTorch overhead and may offer meaningfully higher throughput for the embedding step. Measure actual img/s on a real chunk before implementing; only worthwhile if clip_embed is a bottleneck. Implementation: `pip install mlx-clip`, add MLX branch to `_load_clip()` / `_embed_batch()` in `clip_dedup.py`.

- ~~**PIPELINE-5: Enforce single GPU token across ALL GPU-bound steps**~~ ✅ DONE — `pipeline_lib.py` adds `GPU_LOCK_FILE` (PID JSON record, `/Volumes/2TBSSD/.gpu_lock`) with `acquire_gpu_lock()` / `release_gpu_lock()` / `gpu_lock_holder()`. `ResourceManager.request("GPU_TOKEN")` checks for external holders before acquiring. `precompute_all.py` and `mine_hard_examples.py` refuse to start if iris-train/iris-prep is running or another process holds the lock; orchestrated runs skip the check via `PIPELINE_ORCHESTRATED=1`.

- ~~**RESILIENCE-1: Wrap training in tmux**~~ ✅ DONE — Orchestrator launches training in a dedicated tmux window (`iris-train`) with `caffeinate -dim`. Orchestrator itself runs in `iris-orch` window via `pipeline_ctl.py restart-orchestrator`, also wrapped in `caffeinate -dim` (commit 6f9e7e6). Boot-grace fix added (commit ff87603): stale detection skips restart when training window is alive and heartbeat shows step=0 (model loading phase). DISPATCH.md updated to use pipeline_ctl.py as the canonical start method.

- **PIPELINE-7: Checkpoint archive on chunk promotion** — When `train.done` is marked, copy (or hard-link) the final checkpoint pair plus `.ema.safetensors` into `/Volumes/2TBSSD/checkpoints/stage1/archive/chunk{N}_final.*`. Excluded from `keep_last_n` rotation. `pipeline_ctl.py restart-from-chunk N` should look here first. See also PIPELINE-6 for motivation.

- **PIPELINE-8: Shard integrity scan before training** — After `promoted.done`, run a fast tarfile header-only scan across all newly promoted shards (no decompression). Flag any truncated or zero-byte members. This catches JPEG corruption and partial writes before they silently produce NaN gradients mid-training. Implement as a short `validate_shards.py` step inserted between `promoted` and `train`.

- **PIPELINE-9: Disk space reservation check before each step** — Before launching any GPU or prep step, verify that disk free space on 2TBSSD exceeds a configurable minimum (e.g. 50 GB). If not, pause and dispatch an alert. Prevents silent OOM-on-disk that stalls steps mid-run. Add `min_free_gb` to `v2_pipeline.yaml`.

- **PIPELINE-10: Pipeline run summary report on completion** — When the orchestrator writes "All chunks complete", generate a human-readable summary: total wall-clock per chunk, steps/hour per chunk, final loss, checkpoint path, hard example counts. Save to `/Volumes/2TBSSD/logs/run_summary.txt` and notify. Simplest implementation: parse `orchestrator.jsonl` timestamps for events.

- **PIPELINE-11: `restart-from-chunk N` subcommand** — `pipeline_ctl.py restart-from-chunk N` should: (a) kill iris-train if running, (b) clear all sentinels for chunks N..total, (c) restore chunk N-1 final checkpoint from archive (see PIPELINE-7), (d) restart orchestrator. Provides a safe, one-command way to re-run from any chunk boundary without manual sentinel surgery.

- **PIPELINE-12: Per-chunk data quality report** — After `clip_dups` finishes for each chunk, log: total images, deduplication rate (% removed), mean CLIP score, and dataset size breakdown by source (JDB / WikiArt / LAION / COYO). Helps catch bad data ratios before precompute runs. Implement by adding a `report_chunk.py` step or extending `clip_dedup.py` to emit a JSON summary.

- **PIPELINE-13: Precompute progress saved across restarts** — `precompute_all.py` currently re-scans from scratch if interrupted. Add a resume-state file (list of completed shard IDs) so a restart skips already-done shards. Saves hours when precompute is interrupted at 80%.

- **PIPELINE-14: Configurable hard_mix_ratio per chunk** — Current `hard_mix_ratio: 0.05` is global. Chunk 2+ may benefit from a higher ratio (0.10–0.15) since the hard example set is built from chunk N-1's model. Add per-chunk override in `v2_pipeline.yaml` under `training.hard_mix_ratio_by_chunk`.

- **PIPELINE-15: EMA checkpoint auto-select for mining** — `mine_hard_examples.py` always uses `best.safetensors`. The EMA-smoothed weights (`best.ema.safetensors`) often generalise better and produce cleaner hard-example scores. Add `--use-ema` flag to `mine_hard_examples.py` and expose it via orchestrator config.

- **PIPELINE-16: Validation FID/CLIP score tracking across chunks** — `validator.py` produces per-chunk images. Add CLIP score (text-image alignment) and a lightweight FID proxy metric. Store as JSON in `/Volumes/2TBSSD/logs/val_chunk{N}/metrics.json`. Orchestrator reads these to confirm chunk-over-chunk improvement at promotion time.

- **PIPELINE-17: Auto-populate anchor_shards after chunk 1** — After chunk 1 `train.done`, automatically copy a representative sample of chunk 1 shards (e.g. every Nth shard) to `ANCHOR_SHARDS_DIR`. This removes the current manual step. Add `anchor_sample_rate: 10` to `v2_pipeline.yaml`.

- **PIPELINE-18: Graceful pipeline pause on low disk** — Currently the disk-low alert is advisory only. Add a hard pause (write CONTROL_FILE) when free space drops below `disk_critical_gb` (e.g. 20 GB). Resume automatically when space recovers (e.g. after staging cleanup). Prevents silent partial writes that corrupt shards.

- **PIPELINE-20: Structured run metadata for each pipeline execution** — Write a `run_metadata.json` to `DATA_ROOT` at orchestrator start (run_id, scale, start_time, total_chunks, config snapshot). Append completion_time and final_checkpoint at the end. Makes it easy to trace which training run produced which checkpoint, especially after multiple re-runs.

---

## C Binary / CLI
<!-- Items from plans/backlog.md. These are standalone C-only changes — no server or UI work involved.
     B-002 is the most foundational: it unblocks Z-Image-Omni-Base functionality.
     None require new model weights or external dependencies. -->

- **B-001: --vary-from / --vary-strength CLI wiring** (~1 hour) — `main.c`, `iris.h`
- **B-002: Z-Image CFG infrastructure** (~1 day) — `iris_sample.c`, `iris.c`, `iris.h` — unblocks Z-Image-Omni-Base; do this before B-003
- **B-003: Negative prompt for distilled Flux** (~2 hours) — `iris.c`, `main.c` — prerequisite for Web UI Feature 1

---

## Web UI Features
<!-- Items from plans/web-ui-backlog.md. Recommended order: 3 → 2 (UI only) → 4 → 1.
     B-003 (C binary negative prompt) must land before Feature 1 can be completed.
     Extract fetchImageAsBase64() first — it's duplicated in 4 places and all features touch it. -->

- **Prerequisite: extract fetchImageAsBase64()** — duplicated across 4 files; extract into shared util before touching any feature below
- **Feature 3: Enhanced Vary-from-History** (~2–3h) — fastest win, no C backend changes needed
- **Feature 2: Per-Slot Reference Strength + Style Reference Mode** (~3h UI / ~8h full C with backend)
- **Feature 4: Outpaint UI** (~5–7h)
- **Feature 1: Negative Prompt** (~3–4h server+UI + 4h C backend) — blocked on B-003

---

## Metal / GPU Performance
<!-- Items from memory/perf_backlog.md. BL-004 and BL-005 are M3+ only — skip on M1/M2.
     Already completed: BL-001 (Float32 SDPA), BL-002 (SIMD reductions), BL-003 (bias fusion),
     BL-006 (MTLResidencySet), BL-007 (graph cache sizes), BL-008 (Qwen3 causal SDPA), VAE-GPU-ATTN. -->

- **BL-004: simdgroup_matrix for Custom GEMM Tiles** — M3+ only
- **BL-005: Native bfloat MSL Type** — M3+ only

---

## Test Gaps
<!-- Items from memory/test_backlog.md. Already covered with no model needed:
     test_lora.c, test_kernels.c, test_embcache.c, jpg tests, png_compare, web tests.
     Highest-value no-model additions: TB-010 then TB-001. Start there. -->

- **TB-001: Qwen3 Tokenizer Correctness** (P1) — no model needed, only tokenizer JSON
- **TB-010: Flash Attention vs Naive Attention Parity** (P2) — no model needed
- **TB-002: Base Model Regression 4B-base** (P1) — requires model
- **TB-004: VAE Encode/Decode Roundtrip** (P2) — requires model
- **TB-005: img2img Strength Sweep** (P2) — requires model
- **TB-006: CFG Guidance Value Validation** (P2) — requires model
- **TB-003: Z-Image Regression** (P2) — requires model
- **TB-007: Step Preview (--show-steps) Output** (P3) — requires model
- **TB-008: Backend Parity MPS vs generic** (P3) — requires model
- **TB-011: LoRA Integration load+apply in transformer** (P3) — requires model
- **TB-009: 9B Model Regression** (P3) — requires model

---

## Pipeline Scripts (Unimplemented)
<!-- Proposed scripts from train/DISPATCH.md. None exist yet but all are well-defined and ready to build.
     pipeline_recaption.sh is the most time-consuming (~2 days of GPU time) but improves dataset quality.
     pipeline_benchmark.sh is the quickest win — just parses the existing training log. -->

- **pipeline_benchmark.sh** — Parse training log for steps/hour, ETA, timing breakdown. Quick win.
- **pipeline_validate.sh** — Generate N sample images from current checkpoint to spot-check quality
- **pipeline_export.sh** — Package adapter for deployment / int4 quantise for inference
- **pipeline_recaption.sh** — Re-caption short captions across dataset (parallelised, ~2 days GPU time)
