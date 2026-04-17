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

- **PERF-1: Reduce eval frequency** — Change `log_every: 100` to `log_every: 500` in `train/configs/stage1_512px.yaml` (and any other config files). Eval consumes 20% of compute time (104s per 100 steps vs 416s forward). At log_every=500 this saves ~16–17% wall-clock time — roughly 8h on a full 105K-step run, or ~1 full day across the 4-chunk pipeline.

- **PERF-2: Logit-normal timestep sampling** — Currently using uniform timestep sampling. High-noise timesteps (t→1) produce inherently large flow-matching loss, causing raw loss to swing 0.30–2.84 with no readable trend. Logit-normal weighting (standard in Flux-family training) concentrates samples at middle timesteps where the model learns structure, reducing variance ~40% and making loss curves interpretable. One-line change to the timestep sampler in `ip_adapter/loss.py`.

- **PIPELINE-1: Verify anchor_shard_dir is set for chunks 2–4** — Config has `anchor_mix_ratio: 0.20` but `anchor_shard_dir: null` for chunk 1. For chunks 2–4, the pipeline script must pass the anchor shard directory when launching training, otherwise the 20% anchor slot silently falls through to nothing. Verify `run_training_pipeline.sh` passes `--anchor-shard-dir` correctly for subsequent chunks.

- **PIPELINE-2: Check SigLIP coverage before each chunk** — Training script prints `WARNING: SigLIP cache coverage X%` at startup if any shards are missing siglip precompute. Shards missing siglip fall back to zero image features, silently degrading image conditioning for those batches. Before starting each chunk's precompute, grep the startup log for this warning. Add a coverage check to the pipeline script (fail loudly if coverage < 100%).

- **PIPELINE-3: Never run pipeline jobs while training is active on 2TBSSD** — Two epoch-boundary stalls during chunk 1 training (at steps ~19,900 and ~24,900) were extended by competing I/O from the JDB chunk 2 conversion running in parallel. The step ~24,900 stall lasted 2.6h instead of the typical ~15–20min. Rule: fully complete all pipeline work (WDS conversion, precompute) before starting training, or ensure pipeline and training use separate storage volumes.

- **RESILIENCE-1: Wrap training in tmux** — Training runs in a foreground terminal under `caffeinate`. If the terminal closes, the run dies silently. All multi-day training runs must be launched inside a tmux session. Add to DISPATCH.md launch instructions.

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
     BL-002 and BL-003 apply to all Apple Silicon and are the highest-value open items.
     Already completed: BL-001 (Float32 SDPA), BL-006 (MTLResidencySet), BL-007 (graph cache sizes), VAE-GPU-ATTN. -->

- **BL-002: SIMD Group Reductions** — all Apple Silicon
- **BL-003: GPU Bias Fusion** — all Apple Silicon
- **BL-004: simdgroup_matrix for Custom GEMM Tiles** — M3+ only
- **BL-005: Native bfloat MSL Type** — M3+ only
- **BL-008: Qwen3 Causal Attention MPSGraph fallback** — all Apple Silicon

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
