# FLUX.2 / iris.c — Improvement Backlog

Completed items are archived in [COMPLETED_BACKLOG.md](COMPLETED_BACKLOG.md).

---

## Training & Model Quality

**TRAIN-5: Memory reduction + gradient checkpointing infrastructure** (Medium priority, prerequisite for TRAIN-6)

**Background:** The original framing ("gradient checkpointing saves memory on frozen Flux blocks") is wrong. `mx.checkpoint` only affects backward passes, and Flux is frozen — there is no backward through it. The real value of TRAIN-5 is (a) modest near-term savings via structural fixes, and (b) building the infrastructure that TRAIN-6 needs.

**Stage 0 — Per-fence profiling ✓ Done (2026-05-13):**
Measured at step 108k, 512px, 4× stable intervals of 100 steps each:
- fwd  (Flux forward + noisy/target): **16.97 GB** — below steady-state active; Flux streams blocks cleanly
- bwd  (adapter backward + optimizer m/v state): **20.44 GB** — peak; +3.47 GB above steady state
- param (adapter params from concrete m/v): **20.44 GB** — same as bwd; `mx.clear_cache()` between fences not releasing optimizer state before Fence 2
- ema  (EMA update): **18.58 GB** — +1.61 GB above steady state
- Steady-state active: 17.96 GB; system peak: **20.44 GB** (not the ~25.93 GB estimated earlier — 11.5 GB headroom on 32 GB)

**Revised memory breakdown (measured):**
- Steady-state active: ~17.96 GB (Flux weights + adapter + optimizer state + MLX cache pool)
- Flux forward transient: negligible (streams below steady state at 16.97 GB)
- Optimizer m/v alloc during backward: +~2.5 GB (brings bwd to 20.44 GB)
- EMA update: +~0.6 GB above steady state

**Stage 1 — Parked** (low urgency given 11.5 GB headroom; revisit if memory becomes tight).

**Stage 2 — Parked** (low urgency given 11.5 GB headroom; revisit if memory becomes tight).

**Stage 3 — Block checkpoint infrastructure ✓ Done (2026-05-13):**
- `block_gradient_checkpointing: false` flag added to `adapter:` section of `stage1_512px.yaml`.
- When enabled: pre-builds `ckpt_double` / `ckpt_single` by wrapping each Flux block with `mx.checkpoint(block)`. These are passed into `_flux_forward_with_ip` (already had the lookup wired).
- Zero cost when disabled. Zero runtime cost with current `_flux_forward_no_ip` path even when enabled (lists built once at startup, never referenced). Recompute overhead only activates when TRAIN-6 switches to `_flux_forward_with_ip`.
- Revised memory estimate for TRAIN-6 (based on measured 20.44 GB peak): 25 blocks × ~75 MB ≈ +1.9 GB → expected TRAIN-6 peak ~22–23 GB. Checkpointing likely not required on 32 GB but available as a fallback if actual measurement exceeds estimate.

**TRAIN-5 complete. Next: TRAIN-6.**

**TRAIN-6: Retrain IP-adapter with block-by-block injection** ✓ Done (2026-05-13)
- Implemented `loss_fn_with_ip` / `compiled_step_with_ip` calling `_flux_forward_with_ip` inside `nn.value_and_grad`. Eliminates the train/inference mismatch of the end-sum approximation.
- Gated by `training.use_block_injection: true` in `stage1_512px.yaml` (enabled). Original split-forward path preserved under `false`.
- `n_grad_steps_per_fwd` forced to 1 when enabled (Q vectors not reusable across steps).
- Memory estimate: ~22–23 GB peak; `block_gradient_checkpointing: true` available as fallback.
- **Next**: run 100-step smoke test to verify no NaN, peak memory in range, loss_cond falls. Monitor cond_gap for improvement toward 0.7–0.85 CLIP-I range.

**TRAIN-7: IP-Adapter production quality roadmap** (High priority, next major release)

Proof-of-concept validated (2026-05-11, `train/reports/ip_adapter_v1/`): the adapter architecture and training signal are sound. The model responds to the style reference with coherent, stable output (CLIP-I 0.53, no NaN, correct image structure). The gap to production quality is entirely a matter of scale and refinement — no architectural rethink is required. The following improvements are known to provide benefit, roughly in priority order:

1. **Larger resolution + more training steps** — the v1 run was a `--small` configuration. Higher resolution (1024px) exposes finer style features to the SigLIP encoder; more steps allow the PerceiverResampler and K/V projections to build a richer style space. Expected CLIP-I gain: +0.05–0.10. **Note (32 GB):** 1024px training on 4B Flux currently peaks at ~26 GB at 512px; 1024px roughly doubles the sequence length (256→1024 image tokens), which increases attention memory. Feasibility needs a short profiling run before committing to a full 1024px flywheel.

2. **Block-by-block injection (TRAIN-6)** — the highest-leverage architectural fix. Currently all style influence arrives at `h_final` after the transformer has committed to its content structure; earlier blocks that govern texture and composition never see the style signal. Moving to block-by-block injection lets every layer adapt to the style. Expected CLIP-I gain: +0.15–0.30, bringing scores into the 0.7–0.85 range typical of production IP-Adapters. Memory: expected peak ~22–23 GB (9–10 GB headroom); `block_gradient_checkpointing` available as fallback if needed (TRAIN-5 Stage 3 done).

3. **Source data curation (PIPELINE-27)** ⛔ blocked on storage — over time, bias shard selection toward high-signal style examples (diverse, distinctive styles; high self/cross-ref gap; low redundancy). Requires PIPELINE-25 (persistent raw pool) + a large-capacity storage volume (~37 TB for full JDB pool). **Note:** the flywheel's `shard_selector.py` already does performance-attributed selection within the fixed precomputed pool — this item is about upstream control of which raw data gets downloaded and precomputed, not which precomputed shards to train on. That distinction matters: the flywheel is already doing the tractable part of curation.

4. **QUALITY-10 ablation harness** ✓ Done — running as part of flywheel trial 2 (iters 25, 30, 35, 40).

**References:** TRAIN-6 (block-by-block injection), PIPELINE-27 (data curation), PIPELINE-25 (raw pool prerequisite).
**Dependency summary:** TRAIN-7 items 1–2 blocked on TRAIN-5 Stage 0 profiling (~1–2h to unblock). Item 3 blocked on storage hardware. Item 4 done.

---

## Pipeline Improvements

**PIPE-ORCH-1: Orchestrator coverage gaps — paths not exercised by smoke run** (Low priority, code bugs fixed)

Smoke run 3 (2026-05-11) validated the happy path across all 14 steps × 2 chunks. Three code bugs found in audit were fixed (commit `cdd9fb0`, 2026-05-13):

- ~~`_check_hot_space()` dead code~~ — now called in `_stage_shards()` and `_stage_precomputed()` with pre-scanned transfer size before any copies begin. `staging_margin_gb: 50` is now enforced.
- ~~GPU_TOKEN race~~ — `_start_training()` returns early when window is gone but `EXIT_CODE` not yet written.
- ~~Duplicate dispatch on restart~~ — `_stager_dispatched_errors` pre-seeded from `dispatch_queue.jsonl` via `_load_open_dispatch_ids()`.

**Remaining: validation gaps only (no known code bugs)**
- LAION/COYO/WikiArt download paths and chunk 3+ sequencing — code generalises correctly; untested at scale.
- Real two-device stager (copy path) — `_check_hot_space()` now wired; needs a real cold→hot run to verify.
- `stage.done` gate blocking training, `_poll_stager` retry after error, training crash one-retry + escalate — all coded correctly; never exercised end-to-end.
- GPU_TOKEN contention at production timing — documented; code fix applied; no observed failure.
- Download throttle stall false-positive — documented in DISPATCH.md Gap 6 as a known operator issue.
- `dispatch-resolve` UI-only clarification — documented in DISPATCH.md.

**PIPELINE-25: Persistent raw-data pool — decouple download from chunk staging** ⛔ blocked on storage

Currently `download_convert.py` downloads each JDB tgz directly into `staging/chunk{N}/raw/journeydb/` and deletes it immediately after conversion. There is no persistent raw pool. Consequences: re-running any scale re-downloads all tgzs even if they were downloaded before; scale changes cause confusion about which tgzs belong to which chunk.

**Storage constraint:** the full JDB dataset is ~25,000 tgzs × ~1.5 GB ≈ ~37 TB. The current 2 TB SSD has ~833 GB free — nowhere near sufficient for a persistent pool of meaningful scale. Implementation requires either a dedicated large-capacity volume (spinning disk, NAS, or additional SSD) or accepting a partial pool (e.g. keep the last N chunks' raw data). The `--pool-dir` override in the design is specifically to allow the pool to live on a separate volume. **This item should not be started until additional storage is available.**

**Proposed layout:**
```
data_root/
  raw/
    journeydb/
      000.tgz   ← persistent pool; never auto-deleted
      001.tgz
      …
    journeydb_anno/
      train_anno_realease_repath.jsonl.tgz  ← downloaded once, kept
  staging/
    chunk1/
      raw/journeydb/
        000.tgz → symlink to raw/journeydb/000.tgz
```

**Behaviour:**
- If `raw/journeydb/{idx:03d}.tgz` already exists, skip HuggingFace fetch entirely.
- After conversion: delete only the staging symlink, not the pool copy.
- `pipeline_setup.py` populates `staging/chunk{N}/raw/journeydb/` with symlinks to the pool subset for the selected scale + chunk.
- `PIPELINE-24` purge logic: `full` reset removes staging but not pool; only an explicit `--purge-pool` flag clears the pool.

**`--pool-dir` override:** allow the raw pool to live on a different volume from `data_root` (e.g. spinning disk or NAS) via `download.pool_dir` config key or CLI flag.

**Implementation scope:**
- `downloader.py`: `_hf_download_file_guarded()` — check pool first; download to pool, not staging.
- `download_convert.py`: `run_jdb_download_convert()` — create staging symlinks before producer loop; remove symlink (not pool file) after conversion.
- `pipeline_setup.py`: add "populate staging symlinks" step; report pool coverage vs. scale requirement.
- `pipeline_lib.py`: add `RAW_POOL_DIR = DATA_ROOT / "raw" / "journeydb"` constant.


**PIPELINE-27: Smart precompute shard selection v2** (Low-Medium priority) ⛔ blocked on PIPELINE-25 + storage

- Build a performance-aware shard selector that uses eval metrics (CLIP-I, self/cross-ref gap, style loss) to dynamically bias the next chunk toward high-value shards.
- **Hard prerequisite: PIPELINE-25 (persistent raw-data pool) + additional storage.** Without a persistent pool there is no stable candidate set to select from — each chunk's raw data is currently ephemeral. Both the engineering work (PIPELINE-25) and the physical storage it requires must be in place first.
- **Partial mitigation already in place:** the flywheel's `shard_selector.py` already does performance-attributed shard selection within the fixed precomputed shard pool (scoring shards by cond_gap contribution, applying recency penalty, diversity slots). This operates on shards that have already been precomputed — it biases *which precomputed shards to train on*, not which raw data to download. PIPELINE-27 is the upstream complement: controlling which raw JDB tgzs are downloaded and precomputed in the first place.
- **Realistic impact:** we are currently training on ~320 of ~50,000 JDB shards. Random selection from that pool already provides wide style diversity; the marginal gain from smarter upstream selection is modest until coverage increases significantly.
- Output: `shard_scores.json` + weighted sampling logic with configurable quality vs diversity trade-off.

---

## C Binary / CLI

- **B-001: --vary-from / --vary-strength CLI wiring** (~1 hour) — `main.c`, `iris.h`
- **B-002: Z-Image CFG infrastructure** (~1 day) — `iris_sample.c`, `iris.c`, `iris.h` — unblocks Z-Image-Omni-Base; do this before B-003
- **B-003: Negative prompt for distilled Flux** (~2 hours) — `iris.c`, `main.c` — prerequisite for Web UI Feature 1

---

## Web UI Features

- [ ] **18. Batch prompt generation** — Submit a list of different prompts to generate in sequence.
- [ ] **20. Per-job timeout** — Prevent hung generations from blocking the queue forever.

**Advanced (prerequisite: extract `fetchImageAsBase64()` — duplicated across 4 files)**
- **Feature 3: Enhanced Vary-from-History** (~2–3h) — fastest win, no C backend changes needed
- **Feature 2: Per-Slot Reference Strength + Style Reference Mode** (~3h UI / ~8h full C with backend)
- **Feature 4: Outpaint UI** (~5–7h)
- **Feature 1: Negative Prompt** (~3–4h server+UI + 4h C backend) — blocked on B-003

---

---

## Metal / GPU Performance

- **BL-004: simdgroup_matrix for Custom GEMM Tiles** — M3+ only
- **BL-005: Native bfloat MSL Type** — M3+ only

---

## Test Gaps

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
