# FLUX.2 / iris.c — Improvement Backlog

Completed items are archived in [COMPLETED_BACKLOG.md](COMPLETED_BACKLOG.md).

---

## Training & Model Quality

**TRAIN-5: Gradient checkpointing + QLoRA foundation** (Medium priority)
- Prepare for future LoRA training and higher-rank adapters.
- Enable larger effective batch sizes on current hardware.

**TRAIN-6: Retrain IP-adapter with block-by-block injection** (Medium priority, next major release)
- Current training uses `_flux_forward_no_ip` + end-sum approximation: all IP contributions are summed and added to `h_final` after all 25 blocks. Q vectors are collected from a clean (no-IP) Flux forward, so earlier blocks cannot adapt their computation to the style signal. This limits quality; CLIP-I ~0.53 vs ~0.7–0.85 for canonical IP-Adapter.
- Replace with `_flux_forward_with_ip` as the actual training forward pass (block-by-block injection matching inference). Each block's Q is computed from IP-conditioned hidden states, matching the canonical IP-Adapter approach.
- **Warm-start**: current checkpoint (`best.safetensors`, step 95000) gives a good init for perceiver and ip_scale; `to_k_ip_stacked`/`to_v_ip_stacked` will re-learn at the correct injection points.
- **Memory cost**: gradient flows through IP injection at each of 25 blocks (no longer fully isolated from Flux), significantly increasing peak memory vs the current isolated approach. Requires profiling on 64 GB unified memory before committing.

**TRAIN-7: IP-Adapter production quality roadmap** (High priority, next major release)

Proof-of-concept validated (2026-05-11, `train/reports/ip_adapter_v1/`): the adapter architecture and training signal are sound. The model responds to the style reference with coherent, stable output (CLIP-I 0.53, no NaN, correct image structure). The gap to production quality is entirely a matter of scale and refinement — no architectural rethink is required. The following improvements are known to provide benefit, roughly in priority order:

1. **Larger resolution + more training steps** — the v1 run was a `--small` configuration. Higher resolution (1024px) exposes finer style features to the SigLIP encoder; more steps allow the PerceiverResampler and K/V projections to build a richer style space. Expected CLIP-I gain: +0.05–0.10.

2. **Block-by-block injection (TRAIN-6)** — the highest-leverage architectural fix. Currently all style influence arrives at `h_final` after the transformer has committed to its content structure; earlier blocks that govern texture and composition never see the style signal. Moving to block-by-block injection lets every layer adapt to the style. Expected CLIP-I gain: +0.15–0.30, bringing scores into the 0.7–0.85 range typical of production IP-Adapters.

3. **Source data curation (PIPELINE-27)** — over time, bias shard selection toward high-signal style examples (diverse, distinctive styles; high self/cross-ref gap; low redundancy). Degenerate or low-contrast shards dilute the training signal without contributing to style diversity. Synergises with PIPELINE-25 (persistent raw pool) which is a hard prerequisite for shard-level selection.

4. **QUALITY-10 ablation harness** — once the above are in place, systematic sweeps over `cross_ref_prob`, `style_loss_weight`, and freeze schedules will identify the best hyperparameter regime for the new scale.

**References:** TRAIN-6 (block-by-block injection), PIPELINE-27 (data curation), PIPELINE-25 (raw pool prerequisite), QUALITY-10 (ablation harness).

~~**QUALITY-10: Automated style feature ablation harness**~~ ✓ Done — `train/scripts/ablation_harness.py`

---

## Pipeline Improvements

**PIPE-ORCH-1: Orchestrator coverage gaps — paths not exercised by smoke run** (Medium priority)

Smoke run 3 (2026-05-11) validated the happy path across all 14 steps × 2 chunks. The following
surfaces have not been exercised and should be tested before relying on them in production.

**Not tested at all:**
- LAION/COYO/WikiArt download paths — smoke uses `jdb_only: true`
- More than 2 chunks — chunk sequencing beyond chunk 2 is untested
- Real two-device stager (copy, not symlink) — smoke cold/hot roots are on the same SSD;
  actual cold→hot transfers use `rsync`/copy and have never run
- `stage.done` gating on chunk N training — chunk 2 already had `stage.done` from a prior run
  in smoke, so the gate was never actually blocked waiting on a real staging transfer
- `_poll_stager` retry path for a genuine stager failure
- `dispatch-resolve` human-intervention flow under real pipeline pressure
- GPU_TOKEN contention under real timing (precompute + training each take hours in production;
  smoke steps were minutes — real interleaving is much tighter)

**Tested at smoke scale only (low confidence on production behaviour):**
- Download throttling during active training — worked at 2–3 GB; real downloads are multi-TB
- Staging margin / cleanup safety checks — real run has much tighter disk headroom
- Orchestrator restart recovery — triggered incidentally 3× during debugging; clean each time,
  but always during prep (not mid-training or mid-archive)

**Recommended validation steps:**
1. Run a `medium` scale smoke (e.g. 5–10% data, 3 chunks) to exercise the chunk 3+ path and
   the stage.done gate blocking chunk 2 training until staging completes.
2. On a two-device setup, run with real cold/hot separation to exercise the rsync stager path
   and verify staging margin checks don't false-positive on a nearly-full hot volume.
3. Simulate a stager failure (kill iris-stage mid-transfer) and verify `_poll_stager` retries
   cleanly and the dispatch queue surfaces the error.
4. Simulate a training crash (kill iris-train) and verify the one-retry + escalate path works
   and the checkpoint resume picks up from the correct step.

**PIPELINE-25: Persistent raw-data pool — decouple download from chunk staging**

Currently `download_convert.py` downloads each JDB tgz directly into `staging/chunk{N}/raw/journeydb/` and deletes it immediately after conversion. There is no persistent raw pool. Consequences: re-running any scale re-downloads all tgzs even if they were downloaded before; scale changes cause confusion about which tgzs belong to which chunk.

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


**PIPELINE-27: Smart precompute shard selection v2** (Low-Medium priority) ⛔ blocked on PIPELINE-25
- Build a performance-aware shard selector that uses eval metrics (CLIP-I, self/cross-ref gap, style loss) to dynamically bias the next chunk toward high-value shards.
- **Hard prerequisite: PIPELINE-25 (persistent raw-data pool).** Without a persistent pool there is no candidate set to select from — each chunk's raw data is ephemeral and selection is impossible.
- **Revised goal:** deliver higher style quality with fewer chunks by biasing shard selection toward underrepresented styles and away from shards where the current model already performs well.
- Note: `mine_hard_examples.py` already provides adaptive selection at the record level post-chunk; shard-level scoring is an upstream complement, not a replacement. Marginal gain is modest while training on ~320 of ~50,000 JDB shards since random selection already provides wide diversity.
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

## Flywheel Performance

~~**FLYWHEEL-BUG-1: Restart uses best-by-ref_gap checkpoint instead of latest**~~ ✓ Done

Fixed in `orchestrator.py`: on restart, prefer the chronologically latest `step_*.safetensors` from CKPT_DIR. Falls back to `get_best()` only if CKPT_DIR is empty, then to `base_checkpoint` from config.

**FLYWHEEL-TRIAL2: Pre-trial-2 warm-start checklist**

Before launching trial 2, complete the following:

1. **Set `base_checkpoint`** in `flywheel_sref_v1.yaml` to the best checkpoint from trial 1. Use `cond_gap` (not `ref_gap` — too noisy at 1000 steps) to select it:
   ```bash
   sqlite3 /Volumes/2TBSSD/flywheel_history.db \
     "SELECT iteration, checkpoint FROM checkpoint_log \
      WHERE flywheel_name='sref-v1' \
      ORDER BY cond_gap DESC LIMIT 1;"
   ```
   Set the returned path as `base_checkpoint:` in the config. This gives trial 2 a warm-started perceiver and K/V projections instead of random init, saving ~5–7 early iterations of convergence.

2. **Fix FLYWHEEL-ABL-1** (ablation config) if re-enabling ablation for trial 2. See entry below.

3. **Rename the flywheel** (`name: "sref-v2"`) and point to a fresh `flywheel_history.db` path, or clear the existing DB, so iteration numbering starts from 1.

4. **Stale-state cleanup** before launch:
   ```bash
   rm -f /Volumes/2TBSSD/logs/*.log /Volumes/2TBSSD/logs/*.jsonl
   rm -f /Volumes/2TBSSD/.heartbeat/*.json
   ```

**FLYWHEEL-METRIC-1: Composite score and best-checkpoint criterion use wrong primary metric** ✓ Done

Trial 1 showed `ref_gap` is consistently negative and noisy at 1000-step iteration budgets (avg −0.016 across all shards; only iter 7 positive). `cond_gap` is the reliable signal: positive, monotonically growing (0.182→0.385 over clean iters 7–10), and has 2.6× spread across shards (0.15–0.38) vs ref_gap's near-zero spread.

Fixed (same commit):
- `shard_selector.py` `_compute_raw_composite`: weights changed to `cond_gap=0.65 / ref_gap=0.20 / loss=0.15`; cond_gap normalization tightened from `[-3, +0.5]` to `[-0.5, +0.5]` to give meaningful spread at observed values.
- `flywheel_lib.py` `get_best`: `ORDER BY ref_gap DESC` → `ORDER BY cond_gap DESC`.
- `orchestrator.py` mark-best trigger: comparison switched from `ref_gap` to `cond_gap`.

**FLYWHEEL-ATTR-1: Attribution convergence too slow — min_attribution_obs=3 too conservative** ✓ Done

After 12 flywheel iterations, only 2 of 42 scored shards had `attr_confidence > 0`. The `min_attribution_obs=3` gate requires ≥3 inclusions AND ≥3 exclusions before attribution activates. With 20 shards/iteration from a 42-shard pool, high-frequency shards (e.g. shard 000002, selected 5/6 iters) never accumulate enough exclusion observations.

Fixed: `min_attribution_obs: 3 → 2` in `flywheel_sref_v1.yaml`. This activates attribution ~2× faster while still requiring evidence on both inclusion and exclusion sides.

Note: if the same shards keep dominating the performance slots (see shard 000002 above), `performance_weight: 0.60` or `recency_penalty: 0.30` may also need tuning in trial 2. A tighter recency window or higher exploration rate would force more exclusion observations for consistently top-ranked shards.

**FLYWHEEL-ABL-1: Fix ablation harness integration before enabling for trial 2**

The ablation burst was disabled for trial 1 (`ablation_every_n: 0`) after it launched a 12,000-step run (~12.8h) that would have blocked the flywheel for ~51h (4 runs × 12.8h). Three things need fixing before re-enabling:

1. **`steps_per_run` in `ablation_sref_v1.yaml`**: set proportionally to `steps_per_iteration`. Current `steps_per_run: 12000` has a comment "~14 min/run" which is wrong by 55× (actual: 12.8h at 0.26 steps/sec). A sensible default is `steps_per_run = steps_per_iteration` (same training budget as one flywheel iteration). For a 1000-step flywheel, set `steps_per_run: 1000`.
2. **`ablation_every_n`**: currently fires every 5 iterations. For short trials (1000 steps/iter) ablation overhead is proportional to the flywheel step budget, so every 5 is reasonable once step counts match.
3. **Ablation score metric**: `ablation_sref_v1.yaml` uses `clip_i_weight: 0.55` as objective, but trial 1 shows `ref_gap` is noisy at 1000 steps. Consider using `cond_gap` as the primary ablation objective, or average over the final N steps rather than the last log window.

**FLYWHEEL-PERF-1: Multiple adapter gradient steps per Flux forward**
- The frozen Flux forward pass dominates step time (~2.4s/step = 63%). The adapter backward is only ~1.4s/step.
- Since `qs` (Flux Q vectors) are `stop_gradient`'d and computed without any IP contribution, they can safely be reused across N adapter backward steps for the same (noise, timestep, target) sample. This is equivalent to N gradient descent steps on the same mini-batch — valid for style injection learning.
- Implementation: add `n_grad_steps_per_fwd: N` option to the training config and loop `compiled_step` N times before releasing `flux_state`. At N=3: effective step time drops from ~3.8s to ~2.2s (~1.7x throughput). Step counter, EMA, and heartbeat must all increment by 1 per inner loop iteration.
- Measured baseline: 3.8s/step at 512×512 (fwd=2.4s + eval=1.4s). At N=3: (2.4 + 3×1.4) / 3 = 1.8s/step.

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
