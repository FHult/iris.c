# FLUX.2 / iris.c — Improvement Backlog

Completed items are archived in [COMPLETED_BACKLOG.md](COMPLETED_BACKLOG.md).

---

## Training Development Lessons

Lessons crystallised from the TRAIN-6 / Option C development cycle (2026-05-13).

**1. Always smoke before a long run.**
Run 100 steps from the current checkpoint before enabling any new training code path. Validates: Metal graph warmup (catches 10–30 min compilation stalls), step timing, memory peak, NaN propagation, and style-loss / EMA / cross-ref interactions. The TRAIN-6 smoke discovered a 45 GB OOM peak (2.25× over budget) that was invisible in profiler benchmarks. Without the smoke, the first production run would have swapped and stalled indefinitely.

**2. Profiler step time ≠ production step time.**
The TRAIN-6 profiler gave 8.38 s/step; the smoke measured 14.2 s/step (1.69× gap). The gap comes from style loss, EMA update, data prefetch, cross-ref permutation, and checkpoint I/O — none of which appear in a synthetic-batch profiler. Always derive production timeline estimates from smoke-measured step time, not profiler numbers. Scale the profiler number by 1.6–1.7× for realistic planning.

**3. Memory peaks are only visible in smoke, not profiler.**
The synthetic profiler batch skips style loss (no reference latent), EMA (no ema_params state), and data prefetch (no concurrent threads). These add ~2–4 GB above the profiler peak. `memory_profile: true` in a real 100-step smoke is the only reliable way to confirm a new backward path fits in 32 GB.

**4. Warmstart comparison understates gradient path quality differences.**
The TRAIN-6 vs old-path 500-step comparison from step 108500 showed mean cond_gap +0.076 (TRAIN-6) vs +0.334 (old path). This gap is largely a warmstart artifact: switching gradient direction after 108K steps on the old path creates a distribution mismatch — the adapter's K/V weights are optimised for a different loss landscape, so the first ~250 steps are spent adjusting (first-half mean: +0.008, near noise). Only a from-scratch comparison or a multi-thousand-step continued run is a fair quality-ceiling test. Treat short warmstart comparisons as efficiency signals, not quality-ceiling signals.

**5. New gradient paths need end-to-end smoke validation, not just unit tests.**
`_flux_forward_with_ip_collect_q` was unit-tested (correct shapes, non-zero Q delta at later blocks). But the full training loop — warmup compilation for all 6 bucket shapes, `adapter.get_image_embeds` called outside `value_and_grad` with real weights, memory peak during the extra forward pass — was not validated until smoke. Unit tests confirm the function is correct; smoke tests confirm it integrates correctly with the rest of the training stack.

---

## Platform Vision & Long-term Architecture

**Goal:** evolve iris.c from a fast inference engine into a fully autonomous, self-improving `--sref` optimization platform — running continuous flywheel campaigns (days/weeks/months) that automatically improve both training data and hyperparameters, culminating in open-weight release of a high-quality IP-Adapter.

### Dual Flywheel System

```
┌─────────────────────────────────────────────────────────┐
│  Meta / Optimization Flywheel  (slower cadence)         │
│  Smart Shard Selection  +  Ablation Harness             │
│  → What data to train on  +  How to train               │
└───────────────┬─────────────────────────────────────────┘
                │ curated shards + best config
                ▼
┌─────────────────────────────────────────────────────────┐
│  Main Training Flywheel  (frequent)                     │
│  IP-Adapter training  →  eval metrics  →  shard scores  │
│  → cond_gap / CLIP-I / style loss feed back to meta     │
└─────────────────────────────────────────────────────────┘
```

The meta flywheel (shard selector + ablation harness) decides what to train on and with which hyperparameters. The main flywheel executes and feeds metrics back. Both layers must support warm-starts from historical checkpoints and precompute caches.

### Storage Architecture

Two-tier hot/cold split:

- **Cold storage** (`/Volumes/16TBCold`, 16 TB spinning disk) — source of truth and long-term archive. Contains all raw data, every historical precompute version, all archived weights, and persistent metadata/telemetry. Never auto-deleted by pipeline operations.
- **Hot storage** (`/Volumes/2TBSSD`, 2 TB TB5 SSD) — fast working area for the active + next compute window only. Populated by the JIT stager from cold; archived back to cold after each successful run.

**JIT Data Stager** is the bidirectional intelligence layer between them: stages cold→hot before a compute window, archives hot→cold after. Uses symlinks on the same filesystem (near-instant), atomic copies across filesystems. `_check_hot_space()` enforces the `staging_margin_gb` headroom budget before any transfer begins.

### Proposed cold storage layout

```
/Volumes/16TBCold/
├── raw/
│   ├── journeydb/          # persistent tgz pool — never auto-deleted
│   └── journeydb_anno/     # annotation index — downloaded once, kept
├── precompute/             # versioned encoder caches (managed by cache_manager.py)
│   ├── v1/
│   ├── v2/ …
│   ├── current/            # symlink → active version
│   └── manifests/          # per-version coverage manifests
├── weights/                # archived IP-Adapter weights + checkpoints
│   ├── flywheel-YYYYMMDD/  # one dir per campaign
│   └── best/               # symlinks → current best weights per metric
├── metadata/               # persistent telemetry — never reset between campaigns
│   ├── shard_scores.db     # scored shard history (feeds meta flywheel)
│   ├── ablation_history.db # all ablation runs ever (feeds warm-start)
│   └── flywheel_logs/      # structured per-campaign logs
├── reports/                # all HTML reports (flywheel, ablation, shard selection)
├── temp/                   # staging area for in-progress transfers
└── logs/                   # operational logs (pipeline, orchestrator)
```

This layout is the target state. Current hot-storage paths under `/Volumes/2TBSSD/` remain unchanged during the transition; the stager will progressively migrate source-of-truth data to cold as PIPELINE-25/26 land.

### Hardware scaling roadmap

Current: M1 Max, 32 GB unified memory, 2 TB hot + 16 TB cold.
Future: M5 Max Mac Studio (projected ~128–192 GB unified memory, dramatically higher compute). The dual-flywheel architecture, cold storage layout, and versioned precompute design are all intended to scale without structural changes — only config and scale parameters change.

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
- `block_gradient_checkpointing: true` required on 32 GB (measured peak 45 GB without it; 21.54 GB with it).

**TRAIN-6 smoke + profiling results (2026-05-13):**

Memory (with `block_gradient_checkpointing: true`, 100-step smoke from step 108500):
- `bwd+param` peak: **21.54 GB** — 10 GB headroom on 32 GB. Clean run, no NaN.
- `fwd` peak: 0 GB (grad-free fence not used in block injection path).

Step timing profiled via `/tmp/profile_train6.py` (512×512, synthetic batch, 5 reps):

| Component | Time | % of step |
|-----------|------|-----------|
| Flux forward (no IP, old path) | 1.035s | — |
| Old adapter backward (end-sum) | 0.372s | — |
| **Old full step** | **1.78s** | baseline |
| Flux forward (with IP) | 2.033s | 24% |
| Backward (checkpoint recompute + Jacobians) | 5.973s | 71% |
| Optimizer (AdamW) | 0.376s | 5% |
| **TRAIN-6 full step (clean profiler)** | **8.38s** | **4.7× slower** |
| Smoke measured (with style loss + EMA + overhead) | 14.2s | — |

Root cause of 5.97s backward: Jacobian-vector products propagated backward through all 25 frozen Flux blocks (5 double + 20 single). With `mx.checkpoint`, each block is recomputed once during backward (+~2s recompute), then its Jacobian is applied (+~4s). Increasing to K blocks adds K/25 × 5.97s to the backward.

**Speed at production scale:**
- 200K steps (chunk 1) at profiler rate: **~466h ≈ 19.4 days**
- 200K steps at smoke rate (14.2s/step): **~789h ≈ 32.9 days**
- Old path (use_block_injection=false) at profiler rate: **~99h ≈ 4.1 days**

**Key constraint**: the 5.97s backward is irreducible given the current architecture. Any approach that breaks the hidden_states chain (stop_gradient between blocks) reduces gradient to zero for all but the last block, because Q is already stop_gradient'd — the only path from loss to k_ip[i] is through the Flux block Jacobians.

**TRAIN-6 gradient strategy comparison** ✓ Done (2026-05-13):
- 500-step warmstart comparison from step 108500. Metric: cond_gap (loss_null − loss_cond) per 10-step window.

| Path | n | mean_gap | positive% | first-half | last-half |
|------|---|----------|-----------|------------|-----------|
| Old (end-sum, `use_block_injection=false`) | 49 | **+0.334** | **82%** | +0.266 | +0.401 |
| Full TRAIN-6 (`use_block_injection=true`) | 50 | +0.076 | 56% | +0.008 | +0.145 |

- Old path: 4.4× higher mean cond_gap and 4.7× faster per step ≈ **~20× better wall-clock efficiency** at this warmstart.
- Interpretation: the adapter was trained 108K steps under the old gradient path. Switching to block-injection creates a temporary distribution shift — gradients arriving from a different direction than all prior optimization. The 56% positive rate (barely above chance) and near-zero first-half mean (+0.008) are consistent with the adapter adjusting to the new gradient signal, not with a fundamentally weaker learning signal. A definitive comparison would require training from scratch (or a much longer continued run).
- **Decision: continue production training with old path** (`use_block_injection=false`). TRAIN-6 block injection remains implemented and gated; re-evaluate if training is ever restarted from scratch, or after 50K+ more steps on the old path provide a stronger warmstart for the transition.

**Option C: correct-forward-Q injection** ✓ Implemented and smoke-validated (2026-05-14)

Motivated by the TRAIN-6 warmstart comparison result: block injection is 20× less wall-clock efficient due to Jacobians through all 25 frozen Flux blocks. Option C achieves a better gradient signal than the old end-sum path at much lower cost than full TRAIN-6.

**Approach:** Two-pass forward per step:
1. `_flux_forward_no_ip` — Flux forward without IP tokens, no grad (fast, already in graph).
2. `_flux_forward_with_ip_collect_q` — Flux forward with IP tokens injected, no grad. Collects Q vectors from IP-influenced hidden states at each block. These Q vectors are then used in the loss forward pass instead of the old end-sum approximation.

Cost vs TRAIN-6: the second no-grad forward adds ~1× Flux forward time (~1s at 512px) vs 5.97s backward through all blocks. Expected step time ~3–4s vs 14.2s for TRAIN-6.

Gated by `training.correct_forward_q: true` in `stage1_512px.yaml`. Incompatible with `use_block_injection: true` (flag check at startup).

**Smoke test results (2026-05-14, 100 steps from step 108,500, `correct_forward_q: true`, `use_block_injection: false`):**

Memory (stable throughout):
- `fwd` peak: **18.52 GB**
- `bwd+param` peak: **20.51 GB** — within 0.07 GB of old-path smoke. Option C adds negligible memory overhead.
- `ema` peak: **18.66 GB**

Step timing: **~1.1 s/step** uncontested (final 3 windows, GPU unshared). Early windows were slow (14–30 s/step) due to Metal graph recompilation at startup + concurrent GPU work on the same machine — not representative.

cond_gap (loss_null − loss_cond) per 10-step window:

| Window end | gap | % |
|---|---|---|
| 108,510 | -0.008 | -0.9% |
| 108,520 | +0.226 | +26.0% |
| 108,530 | -0.052 | -8.2% |
| 108,540 | +0.063 | +7.7% |
| 108,550 | -0.022 | -3.9% |
| 108,560 | +0.182 | +20.4% |
| 108,570 | **+0.459** | **+48.4%** |
| 108,580 | +0.321 | +29.5% |
| 108,590 | +0.256 | +31.4% |
| 108,600 | +0.042 | +7.1% |

8/10 windows positive; mean gap +0.149. Clean exit, no NaN, no OOM.

**cross_ref < self_ref warnings:** appeared in some windows, but driven by low sample counts (n=1 cross-ref in several windows). The final window (n=7 cross samples) correctly showed cross > self (+0.043). Not a real issue.

**Decision: enable `correct_forward_q: true` in production flywheel config.** Memory is safe, learning signal is positive, step time is acceptable. Quality comparison vs old path pending a longer run (the 100-step smoke has the same warmstart-noise caveat as the TRAIN-6 comparison).

**TRAIN-7: IP-Adapter production quality roadmap** (High priority, next major release)

Proof-of-concept validated (2026-05-11, `train/reports/ip_adapter_v1/`): the adapter architecture and training signal are sound. The model responds to the style reference with coherent, stable output (CLIP-I 0.53, no NaN, correct image structure). The gap to production quality is entirely a matter of scale and refinement — no architectural rethink is required. The following improvements are known to provide benefit, roughly in priority order:

1. **Larger resolution + more training steps** — the v1 run was a `--small` configuration. Higher resolution (1024px) exposes finer style features to the SigLIP encoder; more steps allow the PerceiverResampler and K/V projections to build a richer style space. Expected CLIP-I gain: +0.05–0.10. **Note (32 GB):** 1024px training on 4B Flux currently peaks at ~26 GB at 512px; 1024px roughly doubles the sequence length (256→1024 image tokens), which increases attention memory. Feasibility needs a short profiling run before committing to a full 1024px flywheel.

2. **Block-by-block injection (TRAIN-6)** ✓ Implemented — see TRAIN-6 profiling results above. Measured cost: 4.7× slower than old path (8.38s vs 1.78s/step clean; 19 vs 4 days for 200K steps). Memory: 21.54 GB bwd peak with `block_gradient_checkpointing: true`. Quality comparison vs old path in progress; decision pending results.

3. **Source data curation (PIPELINE-27)** ⛔ blocked on PIPELINE-25 — over time, bias shard selection toward high-signal style examples (diverse, distinctive styles; high self/cross-ref gap; low redundancy). Requires PIPELINE-25 (persistent raw pool); full JDB pool is ~202 tgzs × ~2-3 GB ≈ ~500 GB (well within the 16 TB cold volume). **Note:** the flywheel's `shard_selector.py` already does performance-attributed selection within the fixed precomputed pool — this item is about upstream control of which raw data gets downloaded and precomputed, not which precomputed shards to train on. That distinction matters: the flywheel is already doing the tractable part of curation.

4. **QUALITY-10 ablation harness** ✓ Done — running as part of flywheel trial 2 (iters 25, 30, 35, 40).

**References:** TRAIN-6 (block-by-block injection), PIPELINE-27 (data curation), PIPELINE-25 (raw pool prerequisite).
**Dependency summary:** Item 1 (resolution/scale) — unblocked, needs 1024px memory profiling run. Item 2 (TRAIN-6) — implemented, gated. Item 3 (PIPELINE-27) — unblocked on storage, blocked on PIPELINE-25 engineering. Item 4 (QUALITY-10) — done.

---

## Pipeline Improvements

**PIPE-ORCH-1: Orchestrator coverage gaps — paths not exercised by smoke run** (Low priority, code bugs fixed)

Smoke run 3 (2026-05-11) validated the happy path across all 14 steps × 2 chunks. Three code bugs found in audit were fixed (commit `cdd9fb0`, 2026-05-13):

- ~~`_check_hot_space()` dead code~~ — now called in `_stage_shards()` and `_stage_precomputed()` with pre-scanned transfer size before any copies begin. `staging_margin_gb: 50` is now enforced.
- ~~GPU_TOKEN race~~ — `_start_training()` returns early when window is gone but `EXIT_CODE` not yet written.
- ~~Duplicate dispatch on restart~~ — `_stager_dispatched_errors` pre-seeded from `dispatch_queue.jsonl` via `_load_open_dispatch_ids()`.

**Remaining: validation gaps only (no known code bugs)**
- LAION/COYO/WikiArt download paths and chunk 3+ sequencing — code generalises correctly; untested at scale.
- Real two-device stager (cold→hot copy path) — `_check_hot_space()` now wired; needs a real `/Volumes/16TBCold` → `/Volumes/2TBSSD` transfer to verify. Will be exercised naturally when PIPELINE-25 lands and the first cold-pool run is started.
- `stage.done` gate blocking training, `_poll_stager` retry after error, training crash one-retry + escalate — all coded correctly; never exercised end-to-end.
- GPU_TOKEN contention at production timing — documented; code fix applied; no observed failure.
- Download throttle stall false-positive — documented in DISPATCH.md Gap 6 as a known operator issue.
- `dispatch-resolve` UI-only clarification — documented in DISPATCH.md.

**PIPELINE-25: Persistent raw-data pool — decouple download from chunk staging** (unblocked — 16 TB cold volume at `/Volumes/16TBCold`)

First step of the cold storage migration. Currently `download_convert.py` downloads each JDB tgz directly into `staging/chunk{N}/raw/journeydb/` and deletes it immediately after conversion — no persistent pool, every re-run re-downloads everything.

**Storage:** ~202 tgzs × ~2-3 GB ≈ ~500 GB. Well within `/Volumes/16TBCold` capacity.

**Target layout (pool side, on cold volume):**
```
/Volumes/16TBCold/raw/journeydb/
  000.tgz   ← persistent pool; never auto-deleted
  001.tgz  …

/Volumes/16TBCold/raw/journeydb_anno/
  train_anno_realease_repath.jsonl.tgz  ← downloaded once, kept
```

**Hot-side staging (symlinks into cold pool):**
```
/Volumes/2TBSSD/staging/chunk{N}/raw/journeydb/
  000.tgz → /Volumes/16TBCold/raw/journeydb/000.tgz  ← symlink (same FS: instant)
```
When cold and hot are on different filesystems, the stager copies rather than symlinks; `_check_hot_space()` enforces headroom before any copy begins.

**Behaviour:**
- If pool tgz already exists, skip HuggingFace fetch entirely.
- After conversion: remove staging symlink/copy only, never the pool file.
- `pipeline_setup.py` populates staging symlinks for the selected scale + chunk.
- Purge logic: `full` reset removes staging but not pool; `--purge-pool` flag required to clear cold pool.

**`--pool-dir` override:** `download.pool_dir` config key or CLI flag allows pool to live anywhere (default: `/Volumes/16TBCold/raw/journeydb`).

**Implementation scope:**
- `downloader.py`: `_hf_download_file_guarded()` — check pool first; download to pool.
- `download_convert.py`: `run_jdb_download_convert()` — create staging symlinks before producer loop; remove symlink after conversion.
- `pipeline_setup.py`: "populate staging symlinks" step; report pool coverage vs. scale requirement.
- `pipeline_lib.py`: `RAW_POOL_DIR` constant pointing to cold pool dir.

**Prerequisite for:** PIPELINE-26 (versioned precompute), PIPELINE-27 (smart shard selection), PIPELINE-28 (data explorer).


**PIPELINE-26: Versioned precompute cache — cold storage migration** ⛔ blocked on PIPELINE-25

The precompute layer (Qwen3 embeddings, VAE latents, SigLIP features) is currently stored only on the hot SSD and has no versioning. If an encoder is updated, all downstream caches must be regenerated. This item migrates precompute to the cold volume under a versioned layout and adds a `cache_manager.py` tool to manage versions.

**Target layout:**
```
/Volumes/16TBCold/precompute/
  v1/  ← Qwen3-4B r1 + VAE flux-vae-v1 + SigLIP ViT-L/14
  v2/  …
  current/  ← symlink to active version
  manifests/
    v1_coverage.json   ← per-shard coverage map; drives precompute scheduling
```

**cache_manager.py** responsibilities:
- Create new version on encoder update; symlink `current/` to new version.
- Report coverage per shard (which embeddings exist, which are missing).
- Garbage-collect old versions beyond a configurable `keep_versions` limit.
- Support `--warm-start-precompute <old_version>` to copy a subset of the old cache into the new version for shards whose encoder did not change (avoids full recompute on partial encoder updates).

**Implementation scope:**
- `pipeline_lib.py`: `COLD_PRECOMPUTE_DIR`, `COLD_PRECOMPUTE_CURRENT` constants.
- `precompute_step.py`: write outputs to versioned cold path; maintain hot symlinks for active training.
- `cache_manager.py`: new script in `train/scripts/`.
- `pipeline_status.py` / `pipeline_doctor.py`: report precompute version + coverage.

**Prerequisite for:** PIPELINE-27, PIPELINE-28.

**PIPELINE-27: Smart precompute shard selection v2** (Low-Medium priority) ⛔ blocked on PIPELINE-25

This is the **meta flywheel** upstream layer: controlling which raw JDB tgzs are downloaded and precomputed, as opposed to which already-precomputed shards to train on (the latter is already done by `shard_selector.py`).

- Use eval metrics persisted in `shard_scores.db` (CLIP-I, self/cross-ref gap, style loss per shard) to bias the next chunk download toward high-signal style examples.
- **Hard prerequisite: PIPELINE-25.** Without a persistent raw pool there is no stable candidate set — each chunk's raw data is currently ephemeral.
- **Partial mitigation already in place:** `shard_selector.py` already does performance-attributed selection within the fixed precomputed pool (cond_gap scoring, recency penalty, diversity slots). PIPELINE-27 is the upstream complement.
- **Realistic impact:** currently training on ~320 of ~50,000 JDB shards. Random selection already provides wide style diversity; upstream curation becomes valuable as coverage grows.
- Output: `shard_scores.db` on cold volume + weighted download scheduling in `pipeline_setup.py`.

**PIPELINE-28: Data Intelligence Layer — data_explorer.py** ⛔ blocked on PIPELINE-25 + PIPELINE-26

As cold storage grows over months of campaigns, a CLI/TUI tool is needed to maintain visibility and support operational decisions. Without it, the cold volume becomes a black box.

**Capabilities:**

1. **Cold storage overview** — disk usage by category (raw, precompute, weights, metadata); precompute version summary; pool coverage vs. all-in scale.
2. **Shard browser** — top N shards by score (cond_gap, CLIP-I, style loss) with trend history from `shard_scores.db`; filter by score range, diversity cluster, or date added.
3. **Weight archive browser** — list flywheel campaigns with their summary metrics; show best weights per metric; support `pipeline_ctl warm-start <campaign>` command generation.
4. **Warm-start helper** — given a target config, suggest the closest historical checkpoint + precompute version to warm-start from; emit the exact `--warmstart` + `--pool-dir` + `--precompute-version` flags.
5. **Maintenance utilities** — validate precompute coverage vs. current pool; prune old precompute versions (respecting `keep_versions`); export a curated shard subset to a new cold sub-directory.
6. **Ablation history view** — query `ablation_history.db`; show Pareto-optimal configs across all campaigns; support `--warm-start-from` path generation for the ablation harness.

**Implementation:** `train/scripts/data_explorer.py`, standalone CLI (no server). Reads `shard_scores.db`, `ablation_history.db`, cold storage layout. No writes except `--prune` and `--export` subcommands (both require explicit confirmation).

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
