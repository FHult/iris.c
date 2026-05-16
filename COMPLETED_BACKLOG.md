# Completed Backlog

Archived completed items from BACKLOG.md. Each entry is summarised; full implementation notes
are in git history.

---

## Web UI — Completed

- **1. Guidance scale slider** — Auto-selects 1.0 distilled / 4.0 base.
- **2. Increase steps max to 50+** — Slider now capped at 256.
- **3. Schedule selection** (linear/power/sigmoid)
- **4. Model info display** — `/model-info` endpoint + header.
- **5. Embedding cache in server mode** — 4-bit quantized `embcache.c`.
- **6. Ctrl+Enter to generate**
- **7. ETA display** — step_time × remaining steps.
- **8. History search/filter**
- **9. Lightbox metadata overlay**
- **10. Image caching headers** — Immutable cache on history images.
- **11. Store generation time in history**
- **12. Persist style preset & variations count** in localStorage
- **13. Auto-restart on process crash**
- **14. Variation grid/mosaic view**
- **15. Side-by-side comparison**
- **16. Lightbox delete + download buttons**
- **17. Prompt templates** — Reusable prompt structures with variable placeholders.
- **19. Reference image reorder** — Drag to reorder slots.
- **26. Dark/light theme toggle**

---

## Training Quality — Completed

- **TRAIN-4** — On resume, reset `_t_eval_end = None` + `t0 = time.time()` at first step; cap per-step data wait to 300 s to discard sleep/wake stale timestamps.

- **QUALITY-1** — Cross-ref permutation: `cross_ref_prob: 0.5` in training config; permutes `siglip_feats` along batch dim on 50% of conditioned steps, forcing style/content separation.
- **QUALITY-2** — Freeze double-stream scales: `freeze_double_stream_scales: true` in adapter config; zeros `adapter.scale[:nd]` at init and zeroes those grad slices in `compiled_step`.
- **QUALITY-3** — Patch-shuffle: `patch_shuffle_prob: 0.5`; shuffles 729-token SigLIP sequence before passing to adapter on 50% of conditioned steps.
- **QUALITY-6** — Cross-ref loss split: `_self_ref_loss_*` / `_cross_ref_loss_*` accumulators; printed at log interval and included in heartbeat.
- **QUALITY-8** — Style loss enabled: `style_loss_weight: 0.05` in `stage1_512px.yaml` (was 0.0).
- **QUALITY-9** — `train/scripts/quality_tracker.py`: HTML report + `--ai` JSON mode; reads eval_results.json, val_loss.jsonl, trainer heartbeat.

- **PERF-1** — `log_every` 100→500 in `stage1_512px.yaml`. ~16–17% wall-clock saving.
- **PERF-2** — Logit-normal timestep sampling in `train_ip_adapter.py`.
- **TRAINING-1** — Removed dead `style_ref` path. SigLIP cache-miss now forces `use_null_image=True` with exact zeros.
- **TRAIN-3** — Worker threads wrapped in `_guarded()` helper; crash traceback surfaced at 120 s timeout and `None`-sentinel path.

---

## Pipeline — Completed

- **PRECOMP-1** — Versioned, content-addressable precompute cache: hash-based versioned dirs, atomic `current` symlink, `cache_manager.py`, `cache_inspect.py`, backwards-compatible legacy migration. Shards with unchanged source data reuse cached VAE latents across chunks.

- **PRECOMP-4** (NOT VIABLE) — Multi-worker precompute with per-process memory caps. Investigated 2026-05-11: on Apple Silicon, Metal serialises GPU submissions across processes — two workers do not run in parallel on the GPU. The bottleneck is Conv2d at 512×512 spatial resolution (38% of encode time per profiling), which is memory-bandwidth-bound; two workers splitting the 400 GB/s bus each get 200 GB/s, yielding the same total throughput at higher contention. CPU–GPU overlap is already near-optimal via the single-worker prefetch architecture (`_encode_with_prefetch`); JPEG decode (~15 ms) is 10× faster than GPU encode (~148 ms), so there is nothing left to hide. Closed as not viable on Apple Silicon for GPU-bound stages.

- **PRECOMP-2** (NOT VIABLE) — Adopt distilled/tiny VAE encoder (TAEF1) for precompute speed. Investigated 2026-05-10: TAEF1 creates train/inference distribution mismatch because training would use TAEF1 latents but inference uses the full VAE decoder. Correct path is PRECOMP-3 (VAE batch tuning) instead.

- **PRECOMP-3** — VAE batch size optimised: profiled on M1 Max at 512px, B=4 is the throughput sweet spot (145.7 ms/img vs 174.6 ms/img at prior default B=16, 20% faster). Default changed in `precompute_all.py`; `precompute.vae_batch: 4` added to `v2_pipeline.yaml`; orchestrator wires it through. Mid-block attention tiling already handled by MLX Flash Attention.

- **PIPELINE-4** — Native MLX ViT-L-14 CLIP backend in `clip_dedup.py`. `mlx_clip_embed.py` implements the full ViT-L-14 architecture in MLX (NHWC Conv2d, fused QKV, `mx.fast.scaled_dot_product_attention`). Loads weights from the timm safetensors already cached by open_clip — no extra download. `clip_dedup.py embed` gains `--clip-backend auto|mlx|open_clip|transformers` and `--benchmark`. Parity vs open_clip: cosine ≥ 0.9999. Benchmark on M1 Max: open_clip ~61 img/s, MLX ~30 img/s — `auto` mode prefers open_clip for throughput; MLX is fallback when PyTorch is absent.

- **PIPELINE-3** — Replaced hard block with `taskpolicy -d throttle` + `nice -n 10` wrappers on download/build/filter; these front-run next-chunk prep during training with kernel-level I/O yield. Precompute/clip_embed still gated on GPU.

- **RESILIENCE-1** — Training runs in dedicated tmux window (`iris-train`) with `caffeinate -dim`. Orchestrator in `iris-orch` via `pipeline_ctl.py restart-orchestrator`.
- **PIPELINE-1** — `anchor_shard_dir` wired for chunks 2–4 via `--anchor-shards` in `_start_training()`.
- **PIPELINE-2** — `_promote_chunk()` enforces ≥90% SigLIP coverage when `training.siglip: true`.
- **PIPELINE-5** — GPU token lock (`GPU_LOCK_FILE`) across all GPU-bound steps.
- **PIPELINE-6** — Superseded by PIPELINE-7.
- **PIPELINE-7** — Checkpoint archive on chunk promotion; `pipeline_ctl.py restart-from-chunk N` restores from archive.
- **PIPELINE-8** — `validate_shards.py` pipeline step: header-only tar scan for corruption before training.
- **PIPELINE-9** — `_disk_guard()` in orchestrator checks free GB before `_launch_prep()` and `_start_training()`.
- **PIPELINE-10** — `_write_run_summary()` writes `logs/run_summary.txt` on completion (wall-clock, checkpoints, hard-example counts). Fixed: correct `message` field and `"starting training"` pattern lookup.
- **PIPELINE-11** — `pipeline_ctl.py restart-from-chunk N` subcommand.
- **PIPELINE-12** — Per-chunk clip-dedup quality report (`clip_dups_report_chunk{N}.json`); doctor warns if dedup rate >20%.
- **PIPELINE-13** — Precompute resume via `.precompute_done.json` list of completed shard basenames.
- **PIPELINE-14** — Configurable `hard_mix_ratio_by_chunk` in `v2_pipeline.yaml`; per-chunk temp YAML override.
- **PIPELINE-15** — `mine_hard_examples.py --use-ema`; `training.mine_use_ema: true` default in config.
- **PIPELINE-16** — Validation FID/CLIP score tracking; `val_dir/metrics.json` with CLIP-I delta vs prior chunk.
- **PIPELINE-17** — `_auto_populate_anchor_shards()` after chunk 1 `train.done`.
- **PIPELINE-18** — Graceful pipeline pause on low disk via `_write_pause()` + `notify()`.
- **PIPELINE-19** — Mining uses precomputed SigLIP features (`--siglip-cache`) when available.
- **PIPELINE-20** — Structured `run_metadata.json` at start; `completed_at` + `run_summary` path appended on finish.
- **PIPELINE-21** — Configurable mining eval budget per scale in `v2_pipeline.yaml`.
- **PIPELINE-22** — `v2_pipeline_dev.yaml` (1 chunk, 200 steps, no dedup/siglip/mine); `pipeline_ctl.py populate-anchor-shards`.
- **QUALITY-4** — Conditioned vs unconditioned loss split (`loss_cond`, `loss_null`, gap) in log + heartbeat + doctor.
- **QUALITY-5** — Adapter scale magnitude logging per log interval; double/single-stream breakdown; doctor anomaly checks.
- **QUALITY-7** — `train/eval.py` with CLIP-T and CLIP-I against `configs/eval_prompts.txt`; JSON + HTML grid report. Gram-matrix style distance not implemented (CLIP-I sufficient for current stage).
- **PIPE-27** — `_dispatch_once` doubles cooldown to 7200 s after 5 fires of the same key.
- **PIPE-28** — `build_shards.py` heartbeat thread: `os.scandir()` wrapped in `try/except OSError` to survive sleep-wake SIGCONT.
- **PIPE-29** — Post-precompute per-shard `.npz` coverage verification at `precompute_all.py` exit; exits 1 with gap report.
- **PIPE-30** — `train/eval_refs/` populated with 5 public-domain MET Museum artworks (512×512 JPEG, 350 KB total).
- **PIPELINE-23** — `build_shards.py` `--max-shards` arg + cap after shuffle; `_start_build()` in orchestrator passes cap from `precompute.max_shards` config key.
- **PIPELINE-24** — `pipeline_setup.py` clean-slate/selective-purge wizard: `_interactive_reset_wizard` (resume/partial/full), `_find_checkpoints`, `_archive_checkpoints`, `_purge_pipeline_state`. `--reset` CLI flag, `--ai --reset` executes non-interactively. Stale log/heartbeat cleanup automated in both reset modes.
- **PIPE-26** — Memory watchdog daemon thread already starts in `Orchestrator.__init__` unconditionally. Added `orchestrator_pid` and `memory_watchdog_log` to `run_metadata.json`.

- **PIPELINE-26 (--ai mode)** — `--ai` mode added to all 7 remaining pipeline scripts: `pipeline_status.py`, `orchestrator.py` (read-only snapshot), `validator.py`, `validate_shards.py`, `validate_weights.py`, `mine_hard_examples.py`, `precompute_all.py`. Contract: JSON to stdout, all prose/progress to stderr, `ok`/`passed` as primary signal.

- **PIPELINE-26 (profiler)** — `pipeline_profile.py`: per-stage wall-clock from orchestrator JSONL launch events + sentinel mtimes; cross-chunk summary; bottleneck flag; VAE note when precompute is slowest. `pipeline_status.py`: timing footer in human output; `stage_mean_hours` + `bottleneck_stage` in `--ai` JSON.

---

## Bugs — Completed (commit `ad2ca02`, 2026-05-15)

30 of 33 bugs from the 2026-05-15 full code review fixed in one batch. 3 remain open (BUG-C-002, BUG-M-004, BUG-T-005) — see BACKLOG.md.

**C Core (8 fixed):** BUG-C-001 — NULL-check transformer forward in all samplers (crash). BUG-C-003 — NULL-check malloc in `load_double_block_weights` f32 path; free `ff_in` on failure (crash). BUG-C-004 — `ensure_work_buffers` frees partial allocations before returning -1 (leak). BUG-C-005 — Remove `static int block_idx` shadow variable in `DEBUG_DOUBLE_BLOCK` path (wrong result, debug only). BUG-C-006 — Cast `calloc` args to `size_t` in `iris_image_create` (integer overflow / heap corruption). BUG-C-007 — `stderr` warning in `iris_linear_nobias_bf16` on malloc failure (silent bad output). BUG-C-008 — NULL-check `normed` alloc in `zi_final_forward` (crash). BUG-C-009 — `ftell()` stored as `long` in `iris_img2img_debug_py` (integer truncation).

**Metal/GPU (8 fixed):** BUG-M-001 — nil check + NSLog on MTLBuffer alloc at 4 OOM-crash sites. BUG-M-002 — MTLBuffer ARC release in `iris_gpu_tensor_alloc_persistent` on struct malloc fail (leak). BUG-M-003 — `bufferB` pool slot released on `bufferC` alloc failure in `sgemm_impl` (pool leak). BUG-M-005 — `attention_fused_bf16`: `head_dim > 128` guard added in kernel (latent corruption). BUG-M-006 — SDPA graph cache key includes `scale` field; prevents silent graph reuse on scale change. BUG-M-007 — `iris_gpu_linear` bias path replaced CPU loop with GPU `g_bias_add_pipeline` kernel; removes unconditional batch commit. BUG-M-008 — `causal_attention_fused` / `_bf16`: `seq > 512` guard added inside kernel (latent corruption). BUG-M-009 — Removed `__strong` from `cBuffers` calloc array (ARC violation).

**Pipeline Scripts (6 fixed):** BUG-P-001 — `_link_or_copy`: relative symlinks via `os.path.relpath` (absolute symlinks dangle on remount). BUG-P-002 — `_atomic_symlink` in `data_stager` + `cache_manager`: uuid-based temp names (concurrent-caller race). BUG-P-003 — `pipeline_doctor` stager fix commands changed to subcommand form (`archive --chunk N`). BUG-P-004 — `_recover_prep_window` reads heartbeat mtime for `started_at`; hung-prep timer survives restart. BUG-P-005 — `data_explorer --fix`: `shlex.split` instead of `cmd.split` (paths with spaces). BUG-P-006 — `_render_status_html`: removed dead self-import block (always raised `ModuleNotFoundError`).

**Trainer (4 fixed):** BUG-T-001 — `_internal_prefix` moved outside `if qwen3_dir and vae_dir` block; fixes `NameError` on siglip-only mode. BUG-T-002 — `grad_clip_pct` captured before `_grad_clip_steps` reset; heartbeat now correct. BUG-T-003 — `_compute_val_loss`: `use_null_image=False`; real SigLIP features loaded from val precompute dirs (TRAIN-8). BUG-T-004 — Checkpoint and JSON sidecars written via `.tmp` + `os.rename` (atomic on crash).

**Additional fixes (2026-05-15):** BUG-C-002 — `iris_sample.c`: all 9 sampler functions now guard `num_steps > IRIS_MAX_STEPS` with `fprintf` + `return NULL` before first malloc; covers 7 functions with `step_times[IRIS_MAX_STEPS]` plus `iris_sample_euler_ancestral` and `iris_sample_heun`. BUG-T-005 — `train_ip_adapter.py` line 987: `ip_out.reshape(B, seq_img, d_inner)` → `reshape(B, seq_img, H_i * Hd_i)`; makes the block's own head geometry explicit, converts silent wrong-result to loud shape error if dims ever diverge.

---

## Training Quality — Completed (continued)

- **QUALITY-10** — Automated style feature ablation harness: `train/scripts/ablation_harness.py`. Bayesian optimisation (GP-UCB, Matern-5/2 kernel) over `cross_ref_prob`, `patch_shuffle_prob`, `freeze_double_stream_scales`, `style_loss_weight`. Pareto front tracking (3-objective), EarlyStopper (SIGTERM on persistent bad cond_gap), HTML report with trend/Pareto/heatmap charts, SQLite DB with schema v2. Subsequently upgraded (commit `e87b8d3`): Optuna TPE backend, Pareto best-compromise marker (⊕), campaign plateau detection, warm-start from prior campaign, stats bar + YAML download button in HTML report.

- **TRAIN-8** — Held-out validation set (all four parts complete). Part A: tgz 0 reserved in `v2_pipeline.yaml` (`validation_tgzs: [0, 0]`; all training ranges start at 1). Part B: `pipeline_ctl.py create-val-set` — idempotent one-time command; downloads/converts tgz 0, builds 2 shards via `build_shards.py`, runs `precompute_all.py` for qwen3+vae+siglip, writes sentinel. Part C: `pipeline_setup.py _stage_val_set()` symlinks val shards to `data_root/validation/held_out/` and copies NPZ files to `data_root/validation/precomputed/` before each training session. Part D: `_compute_val_loss()` loads real SigLIP features via `_load_siglip_embed`; falls back to zeros+`use_null_image=True` only when siglip NPZ is absent; cap raised to 64 records. Val loss is now comparable across all campaigns (same cold-resident held-out shards). See also BUG-T-003 in the bugs section.

- **TRAIN-5** — Memory profiling + gradient checkpointing infrastructure (2026-05-13). Key finding: `mx.checkpoint` only affects backward passes; Flux is frozen so checkpointing saves nothing on Flux blocks — the original framing was wrong. Per-fence profiling at step 108k: fwd 16.97 GB, bwd+param 20.44 GB (peak), ema 18.58 GB, steady-state 17.96 GB → **11.5 GB headroom on 32 GB**. Stage 3 delivered `block_gradient_checkpointing` flag: wraps each Flux block with `mx.checkpoint(block)` at startup; zero cost when disabled. Stages 1+2 parked (headroom sufficient).

- **TRAIN-6** — Block-by-block IP injection matching inference (2026-05-13). Implemented `loss_fn_with_ip` / `compiled_step_with_ip` calling `_flux_forward_with_ip` inside `nn.value_and_grad`. Requires `block_gradient_checkpointing: true` on 32 GB (45 GB OOM without it; 21.54 GB with it). Profiled cost: 8.38 s/step clean, 14.2 s/step smoke — **4.7× slower** than old path. Strategy comparison from step 108,500 (500 steps): old path mean cond_gap +0.334 / 82% positive; TRAIN-6 +0.076 / 56% — ~20× worse wall-clock efficiency at this warmstart (distribution shift artifact). **Decision: gated off (`use_block_injection: false`); re-evaluate for from-scratch runs.** Key architectural constraint: the 5.97s backward is irreducible — the only gradient path to `k_ip[i]` is through Flux block Jacobians.

- **Option C (correct_forward_q)** — IP-injected forward for correct Q-vector extraction (2026-05-14). Two-pass approach: (1) `_flux_forward_no_ip` (no grad, fast); (2) `_flux_forward_with_ip_collect_q` (no grad, collects Q vectors from IP-influenced hidden states at each block). Adds ~1× Flux forward time vs 5.97s TRAIN-6 backward. Gated by `training.correct_forward_q: true`. 100-step smoke from step 108,500: fwd 18.52 GB / bwd+param 20.51 GB / ema 18.66 GB (negligible overhead vs old path); ~1.1 s/step uncontested; cond_gap 8/10 windows positive, mean +0.149. **Decision: enabled in production.**

---

## Flywheel Performance — Completed

- **FLYWHEEL-BUG-1** — Restart used best-by-ref_gap checkpoint instead of latest. Fixed in `orchestrator.py`: prefer chronologically latest `step_*.safetensors` from CKPT_DIR; fall back to `get_best()` only if CKPT_DIR is empty, then to `base_checkpoint` from config.

- **FLYWHEEL-TRIAL2** — Pre-trial-2 warm-start checklist. Resolved by choosing extend-not-restart: `max_iterations: 21 → 42`. Warm-start is automatic via checkpoint auto-discovery (FLYWHEEL-BUG-1 fix). DB history continuous; iteration numbering continues from 22.

- **FLYWHEEL-METRIC-1** — Composite score and best-checkpoint criterion used wrong primary metric. Trial 1 showed `ref_gap` is consistently negative and noisy at 1000-step budgets. Fixed: `shard_selector.py` weights changed to `cond_gap=0.65 / ref_gap=0.20 / loss=0.15`; cond_gap normalisation tightened to `[-0.5, +0.5]`; `get_best` and mark-best trigger switched from `ref_gap` to `cond_gap`.

- **FLYWHEEL-ATTR-1** — Attribution convergence too slow. `min_attribution_obs=3` meant only 2 of 42 shards activated after 12 iterations. Fixed: `min_attribution_obs: 3 → 2` in `flywheel_sref_v1.yaml`. Activates attribution ~2× faster while still requiring evidence on both sides.

- **FLYWHEEL-RECENCY-1** — Performance slots had no recency penalty. Top-scoring shard selected every iteration despite `recency_penalty: 0.30` because the penalty only applied to random-fill slots. Fixed in `shard_selector.py`: step 1 (performance slots) now sorts by `_score_penalised()` using the same discount formula as step 4.

- **FLYWHEEL-ABL-1** — Ablation harness integration broken for trial 2. Fixed: `steps_per_run: 1000` (was 12000, ~12.8h/run), ablation objective updated to `cond_gap=0.70` primary, `ablation_every_n: 0 → 5` in `flywheel_sref_v1.yaml`. Will fire at iters 25, 30, 35, 40 (~4h overhead per burst).

- **FLYWHEEL-PERF-1** — Multiple adapter gradient steps per Flux forward. Implemented `n_grad_steps_per_fwd` in `train_ip_adapter.py`. Reuses frozen `flux_state` Q vectors across N adapter backward steps. At N=2: ~1.47× throughput (2.6 s/step vs 3.8 s/step baseline). Config: `stage1_512px.yaml` → `n_grad_steps_per_fwd: 1` default.

---

## Cold Storage — Completed (2026-05-14)

- **PIPELINE-25** — Persistent raw + converted JDB pool on cold storage. `download_convert.py` checks cold pool before HuggingFace fetch; downloads to `raw_pool_root / data/train/imgs/{N:03d}.tgz`; after conversion writes to `converted_pool_root / {N:03d}.tar` with `.converted/` sentinel; staging symlink removed on completion, pool file kept. `pipeline_setup.py` `_setup_pool_dirs()` creates pool dirs with sentinel tracking. `pipeline_lib.py` gains `RAW_POOL_DIR` + `CONVERTED_POOL_DIR` constants. HF cache moved to cold. Symlinks used when hot + cold share a device; atomic copies otherwise.

- **PIPELINE-26** — Versioned precompute cold migration. `pipeline_lib.py` gains `COLD_ROOT`, `COLD_PRECOMPUTE_DIR`, `COLD_WEIGHTS_DIR`, `COLD_METADATA_DIR` constants. `pipeline_doctor.py` reports cold precompute version per encoder + warns if cold `current` lags hot. `cache_manager.migrate_legacy()` handles flat→versioned migration (renames .npz files into `v_legacy/`, creates `current` symlink, writes `manifest.json` with `record_count`). Two-pass review discipline identified and fixed 7 bugs before first production use (see BACKLOG.md lessons 6–9).

- **PIPELINE-27** — Tgz-level shard quality scoring via provenance tracking. `build_shards.py` writes `shard-NNNNNN.provenance.json` sidecars mapping each output shard to its source tgz indices and type (jdb/wikiart/laion/coyo). `shard_scorer.py` (new): reads `shard_scores.db` + provenance sidecars → writes `cold/metadata/tgz_scores.json` (per-tgz mean quality score, n_shards, shard_ids). `pipeline_setup.py` reads `tgz_scores.json` to order tgz downloads by quality; falls back to sequential on cold start. `orchestrator.py` calls `shard_scorer.py` after `mine.done`. `test_pipeline_storage.py` covers all new logic with 35 pytest tests.

- **PIPELINE-28** — `train/scripts/data_explorer.py` (new). 7 subcommands: `status` (full cold storage overview: disk, pool, precompute, weights, DBs), `shards` (browse `shard_scores.db` with score history and trend), `weights` (browse campaign weight archive with best-ever markers), `suggest-warmstart` (emit exact `--warmstart` + `--precompute-version` flags), `ablation` (Pareto front across all campaigns), `compare` (side-by-side campaign comparison), `maintenance` (symlink audit, orphan detection; `--prune`/`--export` gated behind `--confirm`). Bug found and fixed: `list_versions()` on flat precompute layout causes 400K stat syscalls → minutes of hang; fixed by checking `current` symlink presence as a single stat before iterating.

- **PIPE-SMOKE-1** — Trainer heartbeat wired for direct runs. `--run-name` flag writes `trainer_{name}.json` at log cadence with same fields as production heartbeat (step, loss, cond_gap, mem peaks, eta_sec). `pipeline_status.py` scans all `trainer_*.json` heartbeats and shows each named run as a separate row. `pipeline_doctor.py` includes named-run heartbeats in trainer health check and warns on stale heartbeat with live tmux window.

- **DATAMGMT-1** — `data_explorer.py diagnose` subcommand. Surfaces stale hot data, redundant staging copies, misplaced source-of-truth (hot-only DBs/weights), cross-tier duplicates, orphaned precompute versions, and training coverage gaps. Read-only by default; prints `rm -rf` commands as suggestions. `--fix` executes with per-item confirmation; `--json` for scripting with `reclaimable_gb` field.

- **PIPELINE-29** — Hot→Cold archiving loop closed. `data_stager.py`: `_archive_checkpoints()` writes campaign-structured layout (`cold/weights/flywheel-YYYYMMDD/chunk{N}_final.*` and `final.*` from `best.*`); `update_best_symlinks()` maintains per-metric relative symlinks under `cold/weights/best/` (relative to survive volume remounts); `_archive_dbs()` copies `shard_scores.db` + `ablation_history.db` from `DATA_ROOT` to `cold/metadata/`. `pipeline_doctor.py` warns if cold weights stale (>48h behind hot). First production archive: 2026-05-14, chunk 4, flywheel-20260507.

- **PIPELINE-30** — Ultrahot tier (internal NVMe as lowest-latency serving path). `pipeline_lib.py` gains `ULTRAHOT_ROOT`, `ULTRAHOT_WEIGHTS`, `ULTRAHOT_PRECOMP`, `ULTRAHOT_CURATED`, `ULTRAHOT_PREP_DIR`, `ULTRAHOT_MANIFEST` constants. `v2_pipeline.yaml` gains `storage.ultrahot_root`. `data_stager.py` gains `promote_to_ultrahot(chunk)`: atomically copies best checkpoint + active precompute versions to `ultrahot_root` via `.staging/` dir rename; writes `manifest.json` with checkpoint path, chunk, precompute versions, and `promoted_at` timestamp. `promote` CLI subcommand added. `data_explorer.py status` shows UHOT card with disk usage, active checkpoint, and precompute versions. `pipeline_doctor.py` warns if `cold/weights/best/` is newer than `ultrahot/manifest.json` (promotion needed).

- **PIPELINE-31** — Ultrahot-tier data prep and ablation routing. `v2_pipeline.yaml` gains `storage.data_prep_tier: hot|ultrahot`. `data_stager.py` derives `_hot_shards/_hot_precomp/_hot_ckpts` from `_prep_root` (hot_root or ultrahot_root per config); space guard checks prep_root. `orchestrator.py` computes `self.prep_root` from config and threads it through `PIPELINE_DATA_ROOT` export and `--data-root` arg in all prep, training, warmup, and stager launch sites. Flywheel loop and ablation burst read `data_root` from `fw_cfg` (optional key, defaults to `DATA_ROOT`) for opt-in per-flywheel routing. `pipeline_doctor.py` warns CRITICAL/WARNING when `data_prep_tier=ultrahot` and ultrahot_root is missing or low on space. Use: set `data_prep_tier: ultrahot` in storage config for dev/smoke/ablation runs; `data_root: /Users/fredrikhult/ultrahot` in flywheel config for flywheel routing.

---

## Inference/Training Cross-Reference Review — Completed (2026-05-16)

- **INFER-H-001** — Strict aliasing UB in `f16_to_f32()`: `return *(float *)&f32_bits` replaced with `memcpy(&result, &f32_bits, sizeof(result)); return result` in `iris_metal.m`. The `f32_to_f16` direction was already fixed (memcpy). Both conversion directions are now UB-free.

- **INFER-M-001** — Pad token embedding mismatch (Option B). After `qwen3_forward` returns in `qwen3_encode_text_ex`, positions `num_tokens..511` are zeroed via `memset`. Matches the training convention where precompute saves only real-token rows and the dataloader zero-pads to 512.

- **INFER-M-002** — `iris_metal_sgemm_batch()` malloc NULL-check added for both `cBuffers` and `cPtrs`; returns with error message on OOM.

- **INFER-L-001** — Dead `len` mutation statements removed from `parse_json_string` second pass in `iris_qwen3_tokenizer.c`.

- **INFER-L-002** — Dead f32 `apply_rope_2d` kernel and `g_rope_2d_pipeline` removed from `iris_shaders.metal` and `iris_metal.m`.

- **INFER-L-003** — Dead `iris_apply_rope` and `iris_compute_rope_freqs` removed from `iris_kernels.c` and `iris_kernels.h`.

- **INFER-L-004** — Dead `iris_bf16_qk_rms_norm`, `iris_bf16_silu`, `iris_bf16_silu_mul`, `iris_metal_qk_rms_norm` removed from `iris_metal.m` and `iris_metal.h`.

- **INFER-L-005** — `iris_qwen3.h` doc comment updated from "layers 9, 18, 27" to "loop iterations 8, 17, 26 (0-indexed)".
