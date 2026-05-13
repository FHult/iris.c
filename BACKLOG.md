# FLUX.2 / iris.c — Improvement Backlog

Completed items are archived in [COMPLETED_BACKLOG.md](COMPLETED_BACKLOG.md).

---

## Training & Model Quality

**TRAIN-5: Memory reduction + gradient checkpointing infrastructure** (Medium priority, prerequisite for TRAIN-6)

**Background:** The original framing ("gradient checkpointing saves memory on frozen Flux blocks") is wrong. `mx.checkpoint` only affects backward passes, and Flux is frozen — there is no backward through it. The ~25.93 GB peak is dominated by model weights, optimizer state, and transient Flux forward allocations, not adapter activations. Checkpointing the small adapter graph saves only ~200–400 MB (~1.5% of peak). The real value of TRAIN-5 is (a) modest near-term savings via structural fixes, and (b) building the infrastructure that TRAIN-6 needs.

**Memory breakdown (32 GB system, 512px training):**
- Flux 4B weights (frozen): ~8.0 GB
- Adapter weights + EMA + AdamW state: ~4.2 GB
- MLX cache pool (6% × 32 GB): ~2.0 GB
- Metal/OS/framework: ~3.5 GB
- Transient Flux forward allocations (single-block fused projections): ~2–5 GB peak window
- ~5 GB unexplained — requires per-fence profiling to confirm source

**Stage 0 — Per-fence profiling (1–2h, prerequisite):**
Add `mx.get_peak_memory()` / `mx.reset_peak_memory()` instrumentation after each eval fence inside the training step (Fence 0: Flux forward, Fence 1: adapter backward + optimizer update, Fence 2: weight update). Gate behind `memory_profile: true` config flag. Run 500 steps to confirm where the actual peak occurs before writing any checkpointing code.

**Stage 1 — Structural fixes (~2–4h, ~1.2 GB saving):**
- Delete `flux_state` before the adapter backward when `n_grad_steps_per_fwd=1` (current default). The Q tensors + h_final (~150–280 MB) are currently alive in Python during Fence 1 despite not being needed. For N>1 keep the existing behaviour.
- Lower `cache_limit_pct` from `0.06` to `0.03` in `stage1_512px.yaml` (~1 GB saving, +1–3% slowdown from less buffer pooling). Gate behind `memory.cache_limit_pct` config key.

**Stage 2 — Adapter graph checkpointing (~2–4h, ~200–400 MB saving):**
- Add `gradient_checkpointing: false` config flag to `training:` section.
- When enabled, wrap `adapter.get_image_embeds` with `mx.checkpoint(...)` inside `loss_fn`. Recomputes PerceiverResampler during backward instead of storing intermediates. Overhead ~5–10ms/step (<0.2% at current 5–6s/step).
- Combined with Stage 1: expected peak reduction ~1.4–1.7 GB → new peak ~24.2–24.5 GB.

**Stage 3 — Block checkpoint infrastructure for TRAIN-6 (~4–8h, 3–5 GB saving when TRAIN-6 activates):**
- Add `block_gradient_checkpointing: false` config flag.
- When enabled, pre-build `ckpt_double` / `ckpt_single` lists by wrapping each Flux transformer block with `mx.checkpoint(block)`. Pass these into `_flux_forward_with_ip` (already has `ckpt_double`/`ckpt_single` parameters).
- This does nothing in the current `_flux_forward_no_ip` path (no backward). It is the TRAIN-6 activation switch: when TRAIN-6 switches to `_flux_forward_with_ip`, enabling this flag keeps peak memory viable on 32 GB by recomputing one block at a time during backward (storing ~200–400 MB per block instead of ~3–5 GB for all 25 simultaneously).
- Disable by default, test in isolation with a short `_flux_forward_with_ip` smoke run.

**Files to modify:** `train/train_ip_adapter.py`, `train/configs/stage1_512px.yaml`

**Dependency:** Stage 3 is a hard prerequisite for TRAIN-6 on 32 GB. Stages 0–2 are independent improvements.

**Current blocker:** Stage 0 profiling run (~1–2h) is the only thing needed to unblock Stages 1–3. This is not a large unknown — it is a short instrumentation task. All subsequent TRAIN-5/6/7 quality work gates on those numbers.

**TRAIN-6: Retrain IP-adapter with block-by-block injection** (Medium priority, next major release)
- Current training uses `_flux_forward_no_ip` + end-sum approximation: all IP contributions are summed and added to `h_final` after all 25 blocks. Q vectors are collected from a clean (no-IP) Flux forward, so earlier blocks cannot adapt their computation to the style signal. This limits quality; CLIP-I ~0.53 vs ~0.7–0.85 for canonical IP-Adapter.
- Replace with `_flux_forward_with_ip` as the actual training forward pass (block-by-block injection matching inference). Each block's Q is computed from IP-conditioned hidden states, matching the canonical IP-Adapter approach.
- **Warm-start**: current checkpoint (`best.safetensors`, step 95000) gives a good init for perceiver and ip_scale; `to_k_ip_stacked`/`to_v_ip_stacked` will re-learn at the correct injection points.
- **Memory cost (32 GB system)**: the backward through 25 Flux blocks would store intermediate activations for all blocks simultaneously: ~200–300 MB per double block + ~100–200 MB per single block = estimated +3–5 GB above the current 25.93 GB peak, pushing to ~29–31 GB. This exceeds safe headroom without mitigation. The fix is TRAIN-5 Stage 3 (`block_gradient_checkpointing`), which applies `mx.checkpoint` to each Flux block so only one block's activations are live at a time during backward (~200–400 MB total instead of 3–5 GB). **TRAIN-5 Stage 3 is a hard prerequisite before attempting TRAIN-6 on 32 GB.**

**TRAIN-7: IP-Adapter production quality roadmap** (High priority, next major release)

Proof-of-concept validated (2026-05-11, `train/reports/ip_adapter_v1/`): the adapter architecture and training signal are sound. The model responds to the style reference with coherent, stable output (CLIP-I 0.53, no NaN, correct image structure). The gap to production quality is entirely a matter of scale and refinement — no architectural rethink is required. The following improvements are known to provide benefit, roughly in priority order:

1. **Larger resolution + more training steps** — the v1 run was a `--small` configuration. Higher resolution (1024px) exposes finer style features to the SigLIP encoder; more steps allow the PerceiverResampler and K/V projections to build a richer style space. Expected CLIP-I gain: +0.05–0.10. **Note (32 GB):** 1024px training on 4B Flux currently peaks at ~26 GB at 512px; 1024px roughly doubles the sequence length (256→1024 image tokens), which increases attention memory. Feasibility needs a short profiling run before committing to a full 1024px flywheel.

2. **Block-by-block injection (TRAIN-6)** — the highest-leverage architectural fix. Currently all style influence arrives at `h_final` after the transformer has committed to its content structure; earlier blocks that govern texture and composition never see the style signal. Moving to block-by-block injection lets every layer adapt to the style. Expected CLIP-I gain: +0.15–0.30, bringing scores into the 0.7–0.85 range typical of production IP-Adapters. **Prerequisite on 32 GB: implement gradient checkpointing (TRAIN-5) first** to bring peak memory under control before enabling gradient flow through 25 blocks.

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
