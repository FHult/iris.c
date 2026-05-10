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
