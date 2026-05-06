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

- ~~**PIPELINE-6: Preserve each chunk's final checkpoint for pipeline recovery**~~ ✅ DONE — Superseded by PIPELINE-7 (implemented). See PIPELINE-7 entry above.

- **PIPELINE-3: Never run pipeline jobs while training is active on 2TBSSD** — Two epoch-boundary stalls during chunk 1 training (at steps ~19,900 and ~24,900) were extended by competing I/O from the JDB chunk 2 conversion running in parallel. The step ~24,900 stall lasted 2.6h instead of the typical ~15–20min. Rule: fully complete all pipeline work (WDS conversion, precompute) before starting training, or ensure pipeline and training use separate storage volumes.

- **PIPELINE-4: Investigate MLX CLIP for clip_embed step** — `clip_dedup.py` currently uses `open_clip` ViT-L-14 via PyTorch MPS (fp16). MLX is installed in the venv (`mlx==0.31.1`) but `mlx_clip` is not. MLX runs natively on Apple Silicon without PyTorch overhead and may offer meaningfully higher throughput for the embedding step. Measure actual img/s on a real chunk before implementing; only worthwhile if clip_embed is a bottleneck. Implementation: `pip install mlx-clip`, add MLX branch to `_load_clip()` / `_embed_batch()` in `clip_dedup.py`.

- ~~**PIPELINE-5: Enforce single GPU token across ALL GPU-bound steps**~~ ✅ DONE — `pipeline_lib.py` adds `GPU_LOCK_FILE` (PID JSON record, `/Volumes/2TBSSD/.gpu_lock`) with `acquire_gpu_lock()` / `release_gpu_lock()` / `gpu_lock_holder()`. `ResourceManager.request("GPU_TOKEN")` checks for external holders before acquiring. `precompute_all.py` and `mine_hard_examples.py` refuse to start if iris-train/iris-prep is running or another process holds the lock; orchestrated runs skip the check via `PIPELINE_ORCHESTRATED=1`.

- ~~**RESILIENCE-1: Wrap training in tmux**~~ ✅ DONE — Orchestrator launches training in a dedicated tmux window (`iris-train`) with `caffeinate -dim`. Orchestrator itself runs in `iris-orch` window via `pipeline_ctl.py restart-orchestrator`, also wrapped in `caffeinate -dim` (commit 6f9e7e6). Boot-grace fix added (commit ff87603): stale detection skips restart when training window is alive and heartbeat shows step=0 (model loading phase). DISPATCH.md updated to use pipeline_ctl.py as the canonical start method.

- ~~**PIPELINE-7: Checkpoint archive on chunk promotion**~~ ✅ DONE — Orchestrator calls `_archive_chunk_checkpoint(chunk)` after `train.done`; copies `step_NNNNNNN.{safetensors,json,.ema.safetensors}` to `checkpoints/stage1/archive/chunk{N}_final.*`. `pipeline_ctl.py restart-from-chunk N` restores from there.

- ~~**PIPELINE-8: Shard integrity scan before training**~~ ✅ DONE — `validate_shards.py` added as a new pipeline step between `promoted` and `train`. Opens each .tar header-only (no JPEG decompression); checks for truncated archives, zero-byte members, and missing image/caption pairs. Orchestrator runs it via `_start_shard_validation()`, writes a JSON report to `LOG_DIR/validate_shards_chunk{N}.json`. Exit 1 on CRITICAL errors; exit 0 on warnings. `pipeline_doctor.py` reads the step log and the report is surfaced in `_check_stale_logs`.

- ~~**PIPELINE-9: Disk space reservation check before each step**~~ ✅ DONE — `_disk_guard()` in orchestrator checks `min_free_gb` (config key, defaults to `DISK_ABORT_GB`) before `_launch_prep()` and `_start_training()`. Returns False and logs if below threshold.

- ~~**PIPELINE-10: Pipeline run summary report on completion**~~ ✅ DONE — `_write_run_summary()` parses all `orchestrator*.jsonl` files on completion and writes `logs/run_summary.txt` with per-chunk wall-clock, final checkpoint, hard example counts. Called from `_all_done()` block.

- ~~**PIPELINE-11: `restart-from-chunk N` subcommand**~~ ✅ DONE — `pipeline_ctl.py restart-from-chunk N`: kills iris-train, clears sentinels for chunks N..total, restores chunk N-1 final checkpoint from archive, restarts orchestrator. Requires confirmation before destructive operations.

- ~~**PIPELINE-12: Per-chunk data quality report**~~ ✅ DONE — `clip_dedup.py find-dups` gains `--report-out PATH` (writes `n_total`, `n_dups_total`, `n_kept`, `dedup_rate_pct`). Orchestrator passes `LOG_DIR/clip_dups_report_chunk{N}.json`. `pipeline_doctor.py` reads the report in `_check_data_quality()`: WARN when dedup rate >20%, INFO otherwise.

- ~~**PIPELINE-13: Precompute progress saved across restarts**~~ ✅ DONE — `precompute_all.py` maintains `qwen3_output/.precompute_done.json` (list of fully-done shard basenames). On restart, done shards are skipped before `work_items` is built, avoiding tar-open and IO prefetch for already-complete shards.

- ~~**PIPELINE-14: Configurable hard_mix_ratio per chunk**~~ ✅ DONE — `v2_pipeline.yaml` gains `training.hard_mix_ratio_by_chunk` (defaults: chunk 2→0.10, 3→0.12, 4→0.15). Orchestrator generates a per-chunk temp YAML override at `/tmp/iris_train_chunk{N}_config.yaml` when an override exists; passes it as `--config` to the training script. No changes to `train_ip_adapter.py`.

- ~~**PIPELINE-15: EMA checkpoint auto-select for mining**~~ ✅ DONE — `mine_hard_examples.py` gains `--use-ema` flag. When set: checks for a companion `{checkpoint}.ema.safetensors` file first; falls back to `ema.*` keys inside the checkpoint; exits with error if neither EMA source is available (instead of silently using raw weights). Exposed via `training.mine_use_ema: true` in `v2_pipeline.yaml` (default enabled).

- ~~**PIPELINE-16: Validation FID/CLIP score tracking across chunks**~~ ✅ DONE — `validator.py` `run_inference_and_score()` now calls `run_inference.py` + `score_validation.py` as subprocesses (skips gracefully if CLIP deps missing). Saves `val_dir/metrics.json` with `mean_clip_i`, `mean_adapter_delta`, and `clip_i_delta_vs_prev` when a prior chunk's metrics exist. Logs a WARNING when CLIP-I drops >0.05 vs the previous chunk.

- ~~**PIPELINE-17: Auto-populate anchor_shards after chunk 1**~~ ✅ DONE — `_auto_populate_anchor_shards()` copies every Nth production shard to `ANCHOR_SHARDS_DIR` after chunk 1 `train.done`. Rate controlled by `training.anchor_sample_rate` in config (default 10). Skips if anchor dir already populated.

- ~~**PIPELINE-18: Graceful pipeline pause on low disk**~~ ✅ DONE — `_check_disk()` now calls `_write_pause()` + `notify()` + dispatches a cooldown alert when free GB drops below `DISK_ABORT_GB`. Operator can free disk and then run `pipeline_ctl.py resume`.

- ~~**PIPELINE-20: Structured run metadata for each pipeline execution**~~ ✅ DONE — Orchestrator writes `run_metadata.json` at start (run_id, scale, total_chunks, config snapshot) and appends `completed_at` + `final_checkpoint` + `run_summary` path on pipeline completion.

- ~~**PIPELINE-19: Use precomputed SigLIP features in hard-example mining**~~ ✅ DONE — `_start_mining()` in `orchestrator.py` now passes `--siglip-cache '{PRECOMP_DIR}/siglip'` when `training.siglip: true` and the cache dir exists, replacing the hardcoded `--null-siglip` flag. Mining loss now evaluates with real IP-adapter visual conditioning, matching the training distribution. Falls back to `--null-siglip` when siglip is disabled or the cache is absent.

- ~~**PIPELINE-21: Configurable mining eval budget per scale**~~ ✅ DONE — `v2_pipeline.yaml` gains `training.mine_eval_records` and `training.mine_top_k` dicts keyed by scale (smoke: 500/200, small: 10000/2000, medium: 20000/3000, large: 30000/5000, all-in: 50000/8000). Orchestrator reads these with scale lookup and passes `--eval-records` / `--top-k` to `mine_hard_examples.py`. Falls back to script defaults if keys absent.

- ~~**PIPELINE-22: Dev/integration run scale**~~ ✅ DONE — `v2_pipeline_dev.yaml` added: 1 chunk, 200 steps, `skip_dedup: true`, `siglip: false`, `mine: false`. Produces a valid checkpoint in <2h from scratch for iris.c binary integration testing. `v2_pipeline_smoke.yaml` updated to enable all quality flags (`siglip: true`, `mine_use_ema: true`, `mine_eval_records: 500`, `mine_top_k: 200`). `pipeline_ctl.py populate-anchor-shards` command added for manual anchor shard population; auto-populated 6 anchor shards for the current production run.

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
