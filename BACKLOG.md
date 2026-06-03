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

## Pipeline / Storage Development Lessons

Lessons crystallised from the PIPELINE-26/27/28/29 implementation and two-pass code review (2026-05-14).

**6. `Path.suffix` truncates at the last dot — watch for multi-extension files.**
`Path("chunk1_final.ema.safetensors").suffix` returns `".safetensors"`, not `".ema.safetensors"`. A glob + suffix-based copy loop for `chunk{N}_final.*` would map both `chunk1_final.safetensors` and `chunk1_final.ema.safetensors` to the same destination name. Whichever glob returns first "wins" — the EMA checkpoint could silently replace the real weights in the cold archive. Fix: enumerate extensions explicitly rather than deriving them from the source filename. Use `Path.suffixes` or name-based filtering (e.g. `".ema." in f.name`) when multi-dot extensions are in scope.

**7. Match manifest keys exactly — don't assume snake_case consistency.**
`cache_manager.py` writes `"record_count"` to `manifest.json`; an early version of `data_explorer.py status` read `"records"` and silently returned 0 for all precompute record counts. The doctor code had already established the correct `m.get("record_count", m.get("records"))` fallback pattern — new code that reads the same manifests must use the same key and fallback. When in doubt: grep for the write site, not for the read site.

**8. Use config-derived paths everywhere, not just for writes.**
The stager's `archive_chunk()` correctly derives `cold_root / "metadata"` from the config. An early version of `shard_scorer.py` used the hardcoded `COLD_METADATA_DIR` constant for its input DB path but the config-derived `cold_root / "metadata"` for its output path. If `cold_root` differs from the default constant, the scorer reads from the wrong place. Rule: derive all storage paths from the same source (config > constant), and be consistent across both read and write paths in the same function.

**9. Two-pass review is worth it for storage-touching code.**
The first review pass (immediately after implementation) caught 4 bugs: missing `load_config` import, wrong fix-command chunk number, absolute symlinks, wrong DB source path. The second pass (after a day) caught 3 more: EMA file collision (`Path.suffix` truncation), wrong manifest key (`records` vs `record_count`), and path-source inconsistency in `shard_scorer.py`. Storage code modifies persistent state — bugs here can corrupt cold archives silently for many chunks before detection. Budget for at least two review passes before first production use.

---

## Platform Vision & Long-term Architecture

**Goal:** evolve iris.c from a fast inference engine into a fully autonomous, self-improving `--sref` optimization platform — running continuous flywheel campaigns (days/weeks/months) that automatically improve both training data and hyperparameters, culminating in open-weight release of a high-quality IP-Adapter.

### Dual Flywheel System

```
┌──────────────────────────────────────────────────────────────┐
│  Meta / Optimization Flywheel  (slower cadence)              │
│  Smart Shard Selection  +  Ablation Harness                  │
│  → Which data to train on  +  Which hyperparameters to use   │
│  ← shard_scores.db + ablation_history.db (persistent)        │
└────────────────────┬─────────────────────────────────────────┘
                     │ curated shards + best config
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Main Training Flywheel  (frequent)                          │
│  IP-Adapter training  →  eval metrics  →  shard scores       │
│  → cond_gap / CLIP-I / style loss feed back to meta          │
│  ← warm-started from best archived checkpoint                │
└────────────────────┬─────────────────────────────────────────┘
                     │ weights, embeddings, metrics
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Cold Storage — Long-term Knowledge Base                     │
│  Weights archive  +  Versioned precompute  +  Shard scores   │
│  +  Ablation history  +  Raw data pool                       │
│  Every campaign leaves a richer foundation for the next.     │
└──────────────────────────────────────────────────────────────┘
```

The meta flywheel decides what to train on and with what config. The main flywheel executes and feeds metrics back. Cold storage accumulates the knowledge: each campaign extends shard score history, adds scored configs to ablation history, and archives weights. Every new campaign starts smarter than the last.

### Warm-Start as a First-Class Principle

Starting fresh is the expensive fallback, not the default. Every layer of the system must support warm-starting from prior state:

- **Training:** new campaigns warm-start from the best archived checkpoint for the target config. `data_explorer.py --suggest-warmstart` emits the exact `--warmstart` + `--precompute-version` flags.
- **Ablation harness:** on startup, loads `ablation_history.db` and pre-seeds the Optuna TPE study with all prior scored runs before suggesting new candidates. A new campaign instantly inherits the full Pareto frontier from prior work — no redundant exploration.
- **Precompute:** `cache_manager.py --warm-start-precompute <old_version>` copies embeddings for shards whose encoder did not change, skipping full recompute on partial updates. 6-month-old embeddings remain valid if the encoder is unchanged.
- **Shard selection:** `shard_scores.db` accumulates cond_gap and CLIP-I contributions across all campaigns; scores improve with each run. The meta flywheel never starts from zero.

This compounds: the 10th campaign benefits from 9 campaigns of shard intelligence, hyperparameter Pareto history, and weight lineage — dramatically narrowing the search space and reducing time-to-quality.

### Cold Storage as Long-term Knowledge Base

Cold storage is not a backup or overflow — it is the primary accumulator of system intelligence:

- **`metadata/shard_scores.db`** — never truncated; score history grows with every campaign. The meta flywheel scores shards from the full history, not just the most recent run.
- **`metadata/ablation_history.db`** — every ablation run ever recorded, across all campaigns. The Optuna study is rebuilt from this on each new run; the Pareto frontier only improves.
- **`weights/flywheel-*/`** — full checkpoint lineage. Enables warm-starting any future experiment, bisecting quality regressions, and comparing approaches tried months apart.
- **`precompute/v*/`** — versioned encoder outputs that remain valid indefinitely for unchanged encoders.

**Rule:** cold storage is append-only except for explicit operator-triggered garbage collection. Pipeline operations never touch the raw pool, weight archives, or metadata databases.

### Storage Architecture

Three-tier design (target state — current system is hot + cold only):

- **Ultrahot tier** (internal NVMe SSD, ~2–8 TB) — lowest-latency path for the live inference server and web app. Holds only the active weights and embeddings needed to serve requests. Populated by the stager from hot; never written to by training.
- **Hot storage** (`/Volumes/2TBSSD`, 2 TB TB5 SSD) — fast working area for the active + next compute window only. Pipeline reads shards, precompute, and weights from here during training.
- **Cold storage** (`/Volumes/16TBCold`, 16 TB spinning disk) — source of truth and long-term knowledge base. Never auto-deleted by pipeline operations.

**JIT Data Stager** manages all directions:
- **Cold → Hot (staging):** before a compute window, stages raw data, precompute, and weights from cold to hot. Uses symlinks when on the same filesystem (near-instant); atomic copies across filesystems. `_check_hot_space()` enforces `staging_margin_gb` before any transfer.
- **Hot → Cold (archiving):** after a successful run, archives newly generated precompute embeddings, weight checkpoints, and per-campaign telemetry to cold. This is the write path — without it, cold never grows and warm-starts never improve.
- **Hot → Inference (promote):** after archiving, copies or symlinks the active checkpoint + its precompute version to the Ultrahot tier so the web app picks up new weights without restarting.

All directions are first-class operations. Staging populates the working set; archiving accumulates the knowledge; promotion makes results live. See **PIPELINE-30** for implementation.

See also **PIPELINE-32** and **PIPELINE-33** (Hot ↔ Ultrahot movement helper + policy) for the remaining tier-transition convenience layer that is still largely manual today.

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
│   ├── ablation_history.db # all ablation runs ever (feeds Optuna warm-start)
│   └── flywheel_logs/      # structured per-campaign JSON logs
├── reports/                # all HTML reports (flywheel, ablation, shard selection)
├── temp/                   # staging area for in-progress transfers
└── logs/                   # operational logs (pipeline, orchestrator)
```

This layout is the target state. Current hot-storage paths under `/Volumes/2TBSSD/` remain unchanged during the transition; the stager will progressively migrate source-of-truth data to cold as PIPELINE-25/26/29 land.

### Hardware scaling roadmap

Current: M1 Max, 32 GB unified memory, 2 TB hot + 16 TB cold.
Future: M5 Max Mac Studio (projected ~128–192 GB unified memory, dramatically higher compute). The dual-flywheel architecture, cold storage layout, and versioned precompute design are all intended to scale without structural changes — only config and scale parameters change. The accumulated knowledge base (shard scores, ablation history, weight archive) carries forward directly to any new hardware.

---

## Training & Model Quality

**TRAIN-7: IP-Adapter production quality roadmap** (High priority, next major release)

Proof-of-concept validated (2026-05-11, `train/reports/ip_adapter_v1/`): the adapter architecture and training signal are sound. The model responds to the style reference with coherent, stable output (CLIP-I 0.53, no NaN, correct image structure). The gap to production quality is entirely a matter of scale and refinement — no architectural rethink is required.

**Phased plan (full detail in memory file `train7_plan.md`):**

1. **Memory profiling run** (gate, < 2 hours) — run 60 steps at 768px and 1024px with
   `memory_profile: true` to measure actual per-fence peaks. The plans doc cited ~12 GB
   activation at 1024px, but that estimate predates the split-forward architecture
   (`train/train_ip_adapter.py:1476`) which materializes and frees all Flux intermediates
   before backward. Corrected estimate: retained `flux_state` at 1024px is only ~654 MB
   (Q vectors + h_final); estimated system peak ~21–22 GB vs 28 GB theoretical.
   See `train7_plan.md` §2 for exact probe config, run commands, and decision thresholds.

2. **Stage 2: 768px fine-tune** (20K steps, ~1 day) — `train/configs/stage2_768px.yaml`
   exists but is missing quality signals from Stage 1: `correct_forward_q`, `cross_ref_prob`,
   `patch_shuffle_prob`, `style_loss_weight`, `freeze_double_stream_scales`. Patch before
   launching. Target CLIP-I > 0.62. See `train7_plan.md` §3.

3. **Stage 3: 1024px fine-tune** (10K steps, conditional on profiling gate) — create
   `train/configs/stage3_1024px.yaml`. Warmstart from Stage 2. Target CLIP-I > 0.68.
   Requires adding `(1024, 1024)` to `BUCKETS` in `train/ip_adapter/dataset.py:62`.
   See `train7_plan.md` §4.

**Shard resolution — essential insight for 1024px:** Images are stored in shards at their
**original JPEG resolution** — `build_shards.py:321` only filters images smaller than 256px on
either dimension, no resizing. All resizing happens at training time in `dataset.py:444-445`
(Pillow LANCZOS to bucket+32px, then GPU random-crop). This means no shard rebuild is needed for
1024px training. However, any source image smaller than 1024px will be upscaled, which degrades
the VAE latent quality used for training (note: SigLIP is unaffected — it always resizes to
384×384 regardless of source size). **Before committing to a 1024px flywheel, run a shard sweep
to measure what fraction of images are < 512px on their shortest side.** See `train7_plan.md` §5.

**Training-time small-image filter (proposed, needs experimentation):** Rather than accepting all
upscaled images, add a `min_source_size` filter in `dataset.py` that skips images whose shortest
side is below a threshold relative to the training bucket. A 2× max-upscale heuristic (source ≥
50% of training resolution) is the natural starting point — e.g., 512px minimum for 1024px
training, 256px minimum for 512px training (current effective floor). The optimal threshold is an
open question: too aggressive a filter shrinks the effective dataset and risks domain bias; too
permissive lets in degraded signal. The right answer comes from an ablation across thresholds
(0, 384, 512, 768, native-only) measuring val CLIP-I alongside effective dataset size at each
threshold. See `train7_plan.md` §5.4 for the implementation sketch and ablation design.

**Dependency summary:** Profiling run (§1) and shard resolution sweep (§5) both unblocked and should run together; Stage 2 (§2) unblocked pending profiling; Stage 3 (§3) conditional on gate + sweep results. TRAIN-6, PIPELINE-27, QUALITY-10 — done, see COMPLETED_BACKLOG.md.

---

## Pipeline Improvements

**PIPE-ORCH-1: Orchestrator coverage gaps — paths not exercised by smoke run** (Low priority, code bugs fixed)

Smoke run 3 (2026-05-11) validated the happy path across all 14 steps × 2 chunks. Three code bugs found in audit were fixed (commit `cdd9fb0`, 2026-05-13):

- ~~`_check_hot_space()` dead code~~ — now called in `_stage_shards()` and `_stage_precomputed()` with pre-scanned transfer size before any copies begin. `staging_margin_gb: 50` is now enforced.
- ~~GPU_TOKEN race~~ — `_start_training()` returns early when window is gone but `EXIT_CODE` not yet written.
- ~~Duplicate dispatch on restart~~ — `_stager_dispatched_errors` pre-seeded from `dispatch_queue.jsonl` via `_load_open_dispatch_ids()`.

**Remaining: validation gaps only (no known code bugs)**
- LAION/COYO/WikiArt download paths and chunk 3+ sequencing — code generalises correctly; untested at scale.
- Real two-device stager (cold→hot copy path) — `_check_hot_space()` now wired; needs a real `/Volumes/16TBCold` → `/Volumes/2TBSSD` transfer to verify. Will be exercised on the next pipeline run (PIPELINE-25 done, cold pool active).
- `stage.done` gate blocking training, `_poll_stager` retry after error, training crash one-retry + escalate — all coded correctly; never exercised end-to-end.
- GPU_TOKEN contention at production timing — documented; code fix applied; no observed failure.
- Download throttle stall false-positive — documented in DISPATCH.md Gap 6 as a known operator issue.
- `dispatch-resolve` UI-only clarification — documented in DISPATCH.md.

**PIPELINE-25b: Stream-convert downloads — eliminate raw tgz disk writes** (Low priority, long-term)

Currently `download_convert.py` downloads each JDB tgz to disk, then reads it back for conversion. Since downloads are sequential (one tgz at a time) and tgzs are small enough to hold in memory (~2-3 GB each, well within the 32 GB system RAM), the raw bytes could be streamed directly through `_convert_tgz()` without touching disk at all.

**When applicable:** only when `raw_pool_root` is not configured (i.e. no persistent raw pool needed). If the raw pool is enabled, the tgz must land on disk anyway. This optimisation targets the no-pool path or cases where the caller explicitly opts out of raw storage.

**How it would work:**
- `hf_hub_download()` supports a streaming/file-object mode; alternatively, download to a `tempfile.SpooledTemporaryFile` in memory.
- Pass the in-memory buffer directly to `tarfile.open(fileobj=buf, mode="r:gz")` inside `_convert_tgz()`.
- WebDataset output tar is written to disk as today (it is the persistent artifact).
- Eliminates one full disk write + one full disk read per tgz; saves ~2-3 GB × N tgzs of I/O.

**Constraint:** HuggingFace's `hf_hub_download` API always writes to a local path; would need to switch to `huggingface_hub.file_download.http_get()` or `requests` + streaming response to avoid the intermediate file. Alternatively, download to a RAM-backed tmpfs (`/dev/shm` on Linux; macOS has no equivalent — would need to use `tempfile` with a memory-sized cap). Investigate feasibility before committing to this path.

**Interaction with converted pool:** if `converted_pool_root` is set, the Level 0 hit (skip download+convert entirely) already makes this optimisation irrelevant for warm runs. This item only matters for the first-time conversion of each tgz.

---

~~**DEDUP-1: Clean the converted pool at source + retroactive pool cleaning script** — Done. See COMPLETED_BACKLOG.md.~~

**DEDUP-3: clean_wds_pool self-dupe false positives on interrupted restart** (Medium — correctness)

When `clean_wds_pool.py` is killed mid-tar (after FAISS index is written to disk but before the `.deduped` sentinel is written), the FAISS index on disk contains vectors from the partially-processed tar. On restart, that tar has no sentinel so it is reprocessed — its images search against the index and find themselves, scoring similarity ≈ 1.0 above the 0.95 threshold, causing false-positive duplicate removal.

**Confirmed occurrence (2026-05-17):** 003.tar was partially indexed (23 vectors added, no sentinel written) before the process was killed due to swap pressure. The 4-49 batch was kicked off separately. 003.tar must be processed with a trimmed index to avoid false positives.

**Manual fix for 003.tar:**
```python
import faiss, numpy as np
index = faiss.read_index('/Volumes/16TBCold/metadata/dedup_index.faiss')
ids = open('/Volumes/16TBCold/metadata/dedup_index.ids').read().splitlines()
n_clean = len(ids) - 23  # remove 003.tar partial vectors (last 23 in insertion order)
vecs = np.zeros((n_clean, index.d), dtype=np.float32)
index.reconstruct_n(0, n_clean, vecs)
new_idx = faiss.IndexFlatIP(index.d)
new_idx.add(vecs)
faiss.write_index(new_idx, '/Volumes/16TBCold/metadata/dedup_index.faiss')
open('/Volumes/16TBCold/metadata/dedup_index.ids', 'w').write('\n'.join(ids[:n_clean]) + '\n')
```
Then run: `train/.venv/bin/python train/scripts/clean_wds_pool.py --tgz-range 3 3`

Note: trim must be done AFTER the 4-49 batch completes, as the 4-49 batch uses the current index as its duplicate reference. Trimming during that run would break its dedup consistency.

**Structural fix (future):** Before adding vectors for a new tar, write a `.processing` sentinel with the tar name and the current `index.ntotal`. On startup, if a `.processing` sentinel exists, truncate the index to the saved `ntotal` and remove the sentinel. This makes interrupted runs automatically safe to restart.

---

~~**DEDUP-2: Re-dedupe 001.tar after redownload** — Done (2026-05-17)~~

---

**DEDUP-4: Run clean_wds_pool on WikiArt and other datasets for cross-dataset dedup** (Medium — data quality)

The FAISS index at `COLD_METADATA_DIR/dedup_index.faiss` is persistent and global — `clip_dedup.py build-index` extends it on each run rather than rebuilding from scratch. Running `clean_wds_pool.py` with a different `--pool-dir` pointing at another dataset's converted tars will deduplicate against the full accumulated index, catching cross-dataset duplicates (e.g. WikiArt paintings that appear in LAION as scraped web images).

**Run order matters** — whichever dataset goes through first keeps its copy; later datasets lose their duplicates. Recommended order:

1. WikiArt — small, curated; run first so curated entries are preserved
2. JourneyDB — already running (tgz 1–100 current run)
3. LAION / COYO — largest, most overlap with everything; run last so dupes are removed from scraped data

**To run** (after WikiArt tars are converted and available on cold):
```bash
train/.venv/bin/python train/scripts/clean_wds_pool.py \
  --pool-dir /Volumes/16TBCold/converted/wikiart \
  # --index and --blocklist default to COLD_METADATA_DIR, same shared index
```

Repeat for each additional dataset. The `.deduped` sentinel per-tar makes runs idempotent.

---

~~**PIPELINE-DATA-1: Dedicated LAION/COYO download script + pipeline_setup integration** — Done (2026-05-17)~~

Implemented: `train/scripts/download_laion_coyo.py` (new), `downloader.py` (cold-pool symlink staging in `check_laion`/`check_coyo`), `pipeline_setup.py` (`_check_laion_coyo_pools` detection), `v2_pipeline.yaml` (`data_sources` block). Uses HF-hosted repos (no URL staleness). Configure `laion_hf_repo` in `v2_pipeline.yaml` before first run.

---

**TELEMETRY-1: Pipeline telemetry gaps — audit findings** (Multiple priority levels — see report)

Full audit report: [`plans/telemetry-audit.md`](plans/telemetry-audit.md)

**Summary of dead telemetry (logged but never consumed):**
- ~~`validator_chunk{N}.jsonl` events~~ — covered by T5 fix; metrics.json surfaces same signal
- ~~`ema_drift` in trainer heartbeat~~ — now displayed in `pipeline_status.py` (QW-5 done)
- ~~`siglip_coverage_pct` in trainer heartbeat~~ — now displayed in `pipeline_status.py` (QW-5 done)
- ~~`bucket_stats` in heartbeat/log~~ — now displayed as `buckets:` line in trainer status (T4 done 2026-05-17)
- ~~`logs/val_chunk{N}/metrics.json` (CLIP-I, adapter_delta)~~ — now shown per-chunk in pipeline_status.py (T5/QW-10 done 2026-05-17)
- ~~`selection_log` table in `shard_scores.db`~~ — now queryable via `data_explorer selection-history <campaign>` (T6 done 2026-05-17)

**Quick wins (low effort, implement opportunistically):**
- ~~**QW-1**: Add `grad_norm_final`, `ip_scale_final` indexed columns to `ablation_history.db`~~ — Done (2026-05-17)
- ~~**QW-3**: Add `steps_per_second` to trainer heartbeat~~ — Already done (was in heartbeat)
- ~~**QW-4**: Persist `_restart_counts` to `pipeline_state.json`~~ — Done (2026-05-17)
- ~~**QW-5**: Surface `ema_drift` and `siglip_coverage_pct` in `pipeline_status.py`~~ — Done (2026-05-17)
- ~~**QW-6**: Add `stopped_early` + `stop_step` to ablation DB~~ — Done (2026-05-17)
- ~~**QW-7**: Log `threshold_loss` and `skipped` count to `mine_hard_examples.jsonl` done event~~ — Done (2026-05-17)
- ~~**QW-8**: Dispatch WARNING when trainer heartbeat `mem_available_gb < 3.0`~~ — Done (2026-05-17)
- ~~**QW-10**: Make `pipeline_status.py` read `logs/val_chunk{N}/metrics.json`~~ — Done (2026-05-17)

**High-value deeper changes:**
- ~~**DA-6**: Write per-shard hard-example density into `shard_scores.db`~~ — Done (2026-05-17): `hard_example_count` column added; updated after each mine run
- ~~**DA-3**: Campaign-level summary table in persistent DB~~ — Done (2026-05-17): `campaign_summary` table in FlywheelDB; refreshed after each iteration; viewable via `data_explorer campaign-summary`
- ~~**DA-2**: Per-shard loss percentile distribution in mining output~~ — Done (2026-05-17): writes `shard_loss_percentiles.json` (p50/p75/p95/p99 per shard) after each mine run
- **DA-7**: Link validation metrics back to `ablation_history.db` via `post_train_validation` table — deferred (requires validator subprocess from ablation harness, ~1 day)

**PIPELINE-32: Small dedicated Hot ↔ Ultrahot tier movement helper script** (Medium — experiment velocity, unblocked)

With the three-tier storage model now active (Cold = 16 TB HDD as immutable source of truth, Hot = 2 TB external SSD as primary training compute, Ultrahot = internal NVMe for small/fast runs), operators need an easy, safe way to move a small active working set between Hot and Ultrahot.

Current situation:
- `data_stager.py` handles Cold ↔ Hot staging/archiving very well (including background operation and the orchestrator integration).
- Moving between the two SSD tiers (e.g. "I want to run a quick 8-shard ablation burst at maximum speed on the internal NVMe" or "I did some interesting small runs on Ultrahot, promote the new checkpoints + metrics back to Hot") is still manual, error-prone, and lacks the safety nets (size checks, manifests, provenance, heartbeats, resume) that the main stager provides.

Deliver a small, focused helper (suggested name: `ultrahot_stage.py` or a `data_stager.py ultrahot-*` subcommand) that supports:
- Stage a curated subset (explicit shard list, top-N from `shard_scores.db`, latest ablation run, specific campaign window, etc.) from Hot → Ultrahot.
- One-way or bidirectional sync of results (new checkpoints, updated `ablation_history.db`, metrics, small precompute deltas) back from Ultrahot → Hot.
- Proper handling of versioned precompute `current` symlinks, checkpoint lineage, and small manifests so the main pipeline/doctor can see what's resident on Ultrahot.
- Space estimation + safety abort if Ultrahot would go below a configured margin.
- Heartbeat + sentinel files compatible with the existing observability stack.
- Optional "ephemeral experiment" mode that cleans up after itself on Ultrahot when done (or on explicit `cleanup` command).

The tool should be usable both standalone (for quick experiments) and callable from `pipeline_ctl.py` / the ablation harness.

**PIPELINE-33: Define and codify lightweight Hot ↔ Ultrahot movement policy** (Low–Medium — operational clarity)

Accompanying the script in PIPELINE-32, document (and where helpful, lightly enforce) the intended policy for when work should live on Hot vs. Ultrahot:

- Size / duration heuristics (e.g. < ~40 shards and < 10k steps → prefer Ultrahot; larger or multi-day runs → Hot).
- How the ablation harness and `test_quality_features.py` / smoke runners declare their working-set size so the right tier is chosen automatically.
- Rules for promoting "interesting" results from Ultrahot back to Hot (and then to Cold archive).
- Visibility in `pipeline_status.py`, `pipeline_doctor.py --ai`, and `data_explorer` so an operator always knows what is currently on Ultrahot.
- Interaction with the main orchestrator (should it ever auto-stage tiny windows to Ultrahot, or is Ultrahot strictly a manual "fast lane" for the operator?).

Start with clear documentation + sensible CLI defaults in the new helper; wire deeper automation later once real usage patterns emerge.

**Effort estimate:** PIPELINE-32 ~2–3 days (script + integration + tests). PIPELINE-33 ~1 day (policy doc + small CLI/policy hooks). Both are high-leverage for daily experiment velocity on the current hardware.

---

## Ablation Harness Improvements

**ABL-1: Trial-level wallclock timeout** (High — safety, unblocked)

If the trainer hangs (Metal graph compilation stall, GPU lock deadlock, MPS crash), `_run_one` blocks forever in the `proc.stdout` line loop and the entire campaign freezes. There is no per-trial time budget and no recovery path. The fix is a background `TrialTimer` thread that sends SIGTERM after `trial_timeout_secs` and marks the result as `verdict=TIMEOUT`. Implemented in `ablation_harness.py` (new class `TrialTimer`, wired into `_run_one`). Config key: `trial_timeout_secs` (default 14400 = 4 h).

**Success criteria:** A deliberately hung trainer (inserted `time.sleep(9999)`) is killed within 60 s of the timeout, result written to DB as TIMEOUT, campaign proceeds to next trial. No regression on normal runs.

---

**ABL-2: Multi-signal early stopping** (High — quality + time savings)

`EarlyStopper` currently monitors only `cond_gap < min_cond_gap`. Two failure modes it misses:

1. **Loss explosion / NaN**: `loss_smooth > 5.0` should cause an immediate kill (no patience needed) — the run is unrecoverable. Currently only detected at scoring time, wasting potentially hours of GPU.
2. **Style signal absent**: `ref_gap < min_ref_gap` for `ref_gap_patience` consecutive snapshots means the adapter is not learning from the style reference at all. This is a distinct failure from low cond_gap (adapter may learn from null prompt but style is dead).

Extend `EarlyStopper.__init__` with `nan_loss_threshold`, `grad_norm_threshold`, `ref_gap_min`, `ref_gap_patience`. Add `trigger_reason` property for DB logging. Config YAML:

```yaml
early_stopping:
  enabled: true
  min_cond_gap: -0.3
  patience: 4
  min_snapshots: 5
  nan_loss_threshold: 5.0      # instant kill
  grad_norm_threshold: 50.0    # explosion guard (3 consecutive snapshots)
  ref_gap_min: -0.5            # style-dead guard
  ref_gap_patience: 6
```

**Success criteria:** Deliberately broken training (return constant loss=9.0) triggers immediate kill within 2 snapshots. Normal runs unaffected. Trigger reason recorded in DB.

---

**ABL-3: Pareto-front warm-start for NSGA-II** (Medium — quality)

`_warm_start_candidate` loads the single highest-scored experiment from a prior campaign and injects it as the first candidate. For NSGA-II this is sub-optimal: the multi-objective sampler needs multiple diverse seed points to bootstrap the Pareto front estimate — a single point gives it nothing to work with for N_initial trials. Add `_warm_start_pareto(top_k=3)` that loads the top-K Pareto-optimal experiments from the prior DB (excluding already-tried) and injects them as forced first candidates. Single-objective campaigns use the existing single-best path; NSGA-II campaigns use the Pareto path.

**Success criteria:** NSGA-II campaign warm-started from a prior run with 3 Pareto seeds produces a first-generation Pareto front within 5 trials vs 15+ without seeding.

---

**ABL-4: Pareto scatter plot in HTML report** (Medium — interpretability)

The ablation HTML report shows a sortable table with an `is_pareto` flag column. This makes it impossible to see the shape of the Pareto front or where individual trials cluster. Add a `<canvas>` scatter plot above the table: ref_gap on X, cond_gap on Y, one circle per scored trial, Pareto points outlined in gold, tooltip showing params and score on hover. Render inline with vanilla JS — no dependencies.

---

## Shard Scoring Improvements

**SHARD-1: Temporal momentum in shard score updates** (Medium — data quality)

`shard_selector.py score_update()` computes a cumulative running mean: older iterations count equally with recent ones. After 20+ flywheel iterations a shard's score is dominated by its earliest runs, which may have been under a very different hyperparam regime (before ablation tuning). Add exponential moving average (EMA) weighting in `score_update` with a configurable `temporal_decay` (default 0.85 — each new iteration counts as 85% new, 15% history). The existing `n_scored` counter is preserved for confidence estimation. Add `temporal_decay` as a top-level config option in `shard_selector.yaml` / flywheel YAML.

**Success criteria:** After 10 simulated score updates, the EMA-weighted mean responds to a regime change (scores flip from 0.3→0.7) in 3 iterations vs 10+ for the flat mean. Old campaigns unaffected (omitting `temporal_decay` keeps the existing equal-weight cumulative mean; temporal_decay=1.0 gives last-value, not flat mean).

---

**SHARD-2: Hyperparam-conditioned score table** (Low–Medium — long-term quality)

A shard that scores well with `style_loss_weight=0.12` may score poorly with `style_loss_weight=0.0`. The current IPS attribution averages across all hyperparam configurations, conflating effects. Add a `shard_scores_by_regime` table to `shard_scores.db` that stores a separate (score, n_obs) pair per shard per param-regime hash. The flywheel's `_select_shards` call can pass the current ablation best-params as the active regime, and `score_update` partitions attribution by regime. Useful once ≥3 ablation iterations have been run under a stable best-config.

**Dependencies:** Requires ABL-3 (stable best-params are needed as the regime key).

---

## Precompute / VAE Improvements

**PRECOMP-1: VAE tiling for high-resolution precomputation** (High — enables Stage 2/3, unblocked)

`precompute_all.py _preprocess_vae` encodes images at a single fixed resolution by resizing to `image_size` before encoding. Stage 2 (768px) and Stage 3 (1024px) training requires latents precomputed at that resolution. Encoding a 1024×1024 image through the Flux VAE in one pass requires ~6 GB of intermediate GPU memory — tight but feasible at 14 GB limit. However, for robustness and future-proofing (1280×1280+ or multi-crop), implement overlap-blend tiling: split into 512px tiles with 64px overlap, encode each tile independently, blend the overlap regions with a bilinear feather mask, reassemble the full latent.

**Implementation:** New function `_encode_vae_tiled(vae, img_tensor, tile_size=512, overlap=64)`. Used in the VAE encoding pass when `image_size > 512`. The existing single-pass path is preserved as the fast path for ≤512px.

**Success criteria:** A 1024×1024 image encoded via tiling produces a latent with ≤0.5% mean absolute difference vs single-pass encoding. Peak GPU memory stays below 8 GB during tiling (each tile is independent). No regression on 512px precompute.

---

**PRECOMP-2: Tiny proxy VAE encoder** (Low — long-term throughput)

The Flux VAE is the precompute bottleneck at ~200 ms/image on MPS. A small CNN trained on `{image → Flux VAE latent}` pairs could replicate it at ~20 ms/image, enabling 10× faster precompute for large datasets. Architecture: EfficientNet-B0-style encoder (5.3M params) with a final 1×1 conv projecting to 32 channels at stride 8 — output is the pre-patchification latent `[32, H//8, W//8]`. Train with MSE loss on precomputed VAE latent pairs. Expected fidelity: LPIPS < 0.04 vs real VAE. This is a training subproject, not a quick fix.

**Dependencies:** PRECOMP-1 (need high-res latents for training data), a corpus of precomputed VAE latents (~100K images sufficient for initial training).

**Effort:** 3–5 days (architecture, training loop, validation). Does not block any current pipeline work.

---

## Flywheel Management

**FLYWHEEL-1: Long-term campaign management and cross-campaign analysis** (unblocked — PIPELINE-29 done)

Individual campaigns are managed by the orchestrator. This item is the layer above: tracking how quality evolves across campaigns over weeks and months, detecting when a campaign strategy is played out, and deciding when to launch a new campaign vs. continue the current one.

**Campaign lifecycle states:**
- **Active** — training flywheel running, metrics improving.
- **Plateau** — campaign-level cond_gap trend flat for N flywheel iterations (distinct from step-level plateau in the ablation harness, which is per-run). Triggers a recommendation to either change strategy (new ablation config) or warm-start a new campaign.
- **Completed** — operator-marked as done; weights archived; summary written to cold.
- **Superseded** — a later campaign has exceeded this one on all metrics; annotated in weight archive.

**Key capabilities:**

1. **Campaign-level plateau detection** — track rolling mean cond_gap and CLIP-I over the last N flywheel iterations (e.g. N=5). If neither metric has improved by more than `min_delta` for N iterations, emit a WARNING to the doctor and recommend: (a) launch a new ablation run to find a better config, or (b) warm-start a new campaign from a different checkpoint.

2. **Cross-campaign comparison** — powered by `data_explorer compare`. Answers: is campaign B better than campaign A? Are we regressing on CLIP-I while improving cond_gap? The flywheel logs in `metadata/flywheel_logs/` store per-iteration metrics to make this tractable.

3. **Warm-start decision support** — when a plateau is detected, `data_explorer suggest-warmstart` queries the weight archive and ablation history to recommend the highest-leverage starting point for the next campaign. Considers: best historical CLIP-I, which ablation configs are Pareto-optimal, and what training steps have already been covered to avoid redundant work.

4. **Campaign summary generation** — at the end of each campaign (or on demand), generate a structured summary: total steps, peak CLIP-I, cond_gap trajectory, ablation iterations run, shards consumed, wall-clock time. Written to `metadata/flywheel_logs/campaign-{date}.json` and to `weights/flywheel-{date}/summary.json`.

**Implementation:**
- `flywheel.py`: add `_campaign_plateau_check()`, `_write_campaign_summary()`.
- `pipeline_doctor.py`: surface campaign-level plateau as a WARNING with suggested next action.
- `data_explorer.py`: `compare` and `suggest-warmstart` subcommands (see PIPELINE-28).
- `metadata/flywheel_logs/`: structured JSON per iteration, written by `flywheel.py`.

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

---

## Known Bugs — Code Review 2026-05-15

32 of 33 bugs fixed — see COMPLETED_BACKLOG.md. One remains open:

**BUG-M-004: Batch mode atomicity broken in iris_gpu_attention_mps_bf16** — latent (harmless at current call sites)
`iris_metal.m` (~lines 3149–3151). Between Phase 2 (CPU softmax) and Phase 3 (scores @ V), the function unconditionally commits and resets `g_tensor_cmd` regardless of `g_tensor_batch_mode`. In batch mode this splits the attention's Phase 1 and Phase 3 across separate command buffers.

**Investigation findings (2026-05-15):** The only call site is `iris_gpu_attention_bf16` (line 3293), which calls `iris_gpu_sync()` at line 3270 *before* entering, flushing any prior batch work. Phase 1 and Phase 3 being in separate command buffers is an inherent constraint of the CPU softmax — it cannot be avoided without a GPU softmax kernel. Ordering is fully maintained. Current risk: zero. Proper fix: implement GPU softmax to eliminate the CPU readback; medium effort, not needed until a future model requires true bf16 batch-mode attention across this path.

---

## Known Bugs — Pipeline Code Review 2026-05-15 (second pass)

All 14 bugs fixed in commit `76564f8` (2026-05-15). See COMPLETED_BACKLOG.md.

---

## Known Bugs — Inference/Training Cross-Reference Review 2026-05-15

All 9 bugs fixed: INFER-C-001 in commit `76564f8`; INFER-H-001 and INFER-M-001 in commit `ffecfcc`; INFER-M-002 and INFER-L-001–005 in commit `76564f8`. See COMPLETED_BACKLOG.md.

---

## C-Engine Static Review (Grok 4.3, 2026-06) — Triaged

External static review of the C inference engine (`grok_bug_report.md`, untracked).
Verified accurate on spot-check (C-01, H-01, H-04, L-01, M-08 confirmed against
source). Triaged below — severities re-calibrated from the original report. Scope
is the C binary only; none of this affects the Python training/flywheel pipeline.

### Do soon (cheap, low-risk)

- **GROK-1 (was H-01): Remove dead `rope_freqs`** (~30 min) — `iris_transformer_flux.c`.
  Field is malloc'd + `compute_rope_freqs()`'d + freed in 3 load paths (≈4651, 5149,
  5288) but **never read** in any forward (verified: only alloc/compute/free/NULL-guard
  references). Delete field + `compute_rope_freqs` helper if it has no other caller.
  Violates "leave no dead code".
- **GROK-2 (was M-08): Remove dead `iris_vae_load(FILE*)`** (~30 min) — `iris_vae.c:1057`,
  decl `iris.c:38`. The legacy `.bin` VAE loader has **zero call sites** (only decl +
  def). `#ifdef` out or delete; shrinks binary and removes a duplicate VAE load path.
- ~~**GROK-3 (was L-01): Dedup AGENT.md / CLAUDE.md**~~ — NOT A BUG (verified 2026-06).
  Already deduplicated: `AGENT.md` is the real file (git mode 100644) and `CLAUDE.md`
  is already a symlink to it (git mode 120000). Both the Grok report and a shallow
  `diff -q` were fooled — the files "match" precisely *because* one is a symlink to
  the other. No action; drift is already impossible.
- **GROK-4 (was H-04): Z-Image pad/pos-id cross-path golden test** (~2–3h) —
  `iris_transformer_zimage.c`. NOT a bug fix — a regression guard for the documented
  "CPU/GPU position-id mismatch under padded captions" pitfall. Add a unit test
  asserting byte-identical pos-ids for real tokens between the GPU-table and CPU-unified
  paths, exercised with `cap_len % 32 != 0`. Highest-value item here: cheap insurance
  on a pitfall that has already bitten the project.

### Roadmap decision (not a bug)

- **GROK-5 (was C-01/C-02): `--sref` / C IP-Adapter is declared but not implemented.**
  `train/export/iris_ip_adapter.h` provides the full public surface; no `.c` implements
  it; `main.c:1273-1275` cleanly rejects `--sref` ("not yet implemented, planned for
  v2.6"). This is a **deferred feature with a clean guard**, not a correctness defect
  (re-severed from the report's CRITICAL). Decision required: either (a) implement the
  C IP-Adapter (loader + Perceiver + get_kv + inject, wired into double/single block
  forwards, parity-checked vs `test_ip_adapter_inference.py`) and make it the P0, or
  (b) remove the public surface and stop advertising `--sref` until ready. Do not leave
  it half-advertised.

### Fix opportunistically (when already in that file — do not sweep)

- **GROK-6 (was H-02): Unify ad-hoc JSON extraction.** Multiple independent strstr/atoi
  config parsers (`iris.c`, `iris_transformer_flux.c`, `iris_safetensors.c`, `main.c`,
  tokenizer). Extract one minimal `iris_json.c` helper (no new deps) + adversarial tests.
  Real debt, but a dedicated refactor is risky for no functional gain.
- **GROK-7 (was H-03): Collapse duplicated block load/forward/free variants** (f32 / bf16
  / mmap / GPU / debug). Largest structural debt; touches every historical pitfall
  (timestep cache, RoPE indexing, sgemm B-cache). High-risk multi-week refactor — only
  undertake when a concrete feature (e.g. full IP-Adapter) forces touching these paths.
- **GROK-8 (was H-05): Per-ctx progress callbacks.** Global callback pointers in
  `iris_kernels.c` are not reentrant. Move to `iris_ctx` or document "not thread-safe,
  set from main thread only" at minimum.
- **GROK-9 (M-02/M-03/M-06): Error-reporting + cleanup consistency.** Standardise on
  `set_error` for user-visible failures (stderr for dev only); add `d[N-1]='\0'` after
  `strncpy(,,N-1)` even where `calloc` currently saves it; tighten OOM/bad-weight cleanup.
- **GROK-10 (M-01/M-04/L-07): Architectural-invariant asserts + magic-number cleanup.**
  Validate derived dims after config parse (`hidden == heads*128`, `axis_dim*4 ==
  head_dim`); centralise reference constants. Add `static_assert` where cheap.

### Build / nits (observation)

- **GROK-11 (M-07/M-10): Monolithic units + always-clean backend builds.** `iris_transformer_flux.c`
  (~5.3k LOC), `iris_metal.m` (~7k LOC); per-backend build dirs and only re-`xxd` shaders
  on change would speed dev. Cosmetic — defer.

Original report retained as `grok_bug_report.md` (untracked) for full detail.

---

## train/ Static Review (Grok 4.3, 2026-06) — Triaged

External static review of the Python training pipeline (`grok_train_bug_report.md`,
untracked). Verified accurate on spot-check (shell=True at doctor:3176, the
`/Users/fredrikhult/ultrahot` literal, path-asserting tests, 3k+ LOC files all
confirmed). Character note: found **zero CRITICAL and zero concrete correctness
defects** — entirely architectural/portability debt, most of it a known and
accepted single-host constraint. It also **missed the live precompute→train
fresh-write race** that failed flywheel iter 10, illustrating that static review
surfaces debt but not the operational bugs that actually bite. Severities below
re-calibrated to *urgency* (the report's HIGHs are mostly "hurts at multi-machine
/ handoff time", not "wrong results now").

### Done (cheap wins — actioned 2026-06)

- **GROK-T-1 (H-T01): ultrahot username literal** — DONE. `ULTRAHOT_ROOT` was
  `/Users/fredrikhult/ultrahot`; now `~/ultrahot` via `Path.home()` + a
  `PIPELINE_ULTRAHOT_ROOT` env override matching the `DATA_ROOT` convention. Same
  value on this machine, no behaviour change.
- **GROK-T-2 (H-T01): path-asserting storage tests** — DONE.
  `test_pipeline_storage.py::test_importable` asserted exact `/Volumes/16TBCold`
  literals (machine-coupled, tested a constant against its own literal). Rewritten
  to assert the real invariant — importable absolute `Path`s — leaving the
  derivation check to `test_derived_from_cold_root`.

### Open (real but deferred)

- **GROK-T-3 (H-T05): `shell=True` in doctor --fix** (doctor:3176). Kept (7 fix
  strings use pipes/globs/`&&`, so list-form would break them); documented inline
  why it's deliberate + the human-in-the-loop mitigation. Real remaining task:
  audit that no untrusted data flows into interpolated `issue.fix` paths
  (currently all from DATA_ROOT + numeric shard stems). Low risk on a single-user
  box.
- **GROK-T-4 (H-T01 remainder): centralise storage roots.** ~90 hardcoded
  `/Volumes`/`/Users` literals across leaf scripts + argparse defaults + docstrings.
  Proposal: one `pipeline_lib.get_storage(cfg)` returning a `StoragePaths` object;
  ban new literals. P0 *when* moving to a second machine / V3; not before.
- **GROK-T-5 (H-T03): mflux dependency + sys.path hacks.** Unvendored mflux +
  `sys.path.insert` in every entrypoint. Add `requirements-train.txt` pinning
  mflux/mlx/etc., make `train/` a real package (`python -m train.scripts.x`),
  drop the path shims. Reproducibility win; do at V3 packaging time.
- **GROK-T-6 (H-T04): LiveEncoderManager.** The 32 GB attach/detach + mflux
  private-weight manipulation is the most regression-prone code. Encapsulate in a
  `with live_encoders_for_batch():` context manager + a mem-profile assert + a
  hard error (not silent fallback) when precompute coverage is 100%. Worth doing
  before adding any new live-encoder feature.
- **GROK-T-7 (M-T03): config schema + load_and_validate.** 15+ yaml variants, no
  schema; typo'd keys fail silently or deep in the loop. Add a dataclass/validator
  + `config --validate`. Medium effort, real safety win.
- **GROK-T-8 (M-T05): state-machine / resume scenario tests.** Synthetic
  sentinels+heartbeats asserting `derive_chunk_state` + next-action across phantom,
  jetsam, last-chunk, manual-rm, hard-ex-mixing. Catches resume regressions before
  days of compute are wasted. (Note: would NOT have caught iter 10's fs race.)

### Fix opportunistically (when already in that file)

- **GROK-T-9 (H-T02): split god modules** (orchestrator 3480, doctor 3340,
  train_ip_adapter 3071, ablation 3011 LOC). Legitimate but high-risk multi-week
  refactor of the live state machine — only when V3/containerization forces it.
- **GROK-T-10 (M-T01/M-T02): sentinel TOCTOU + resume special-cases.** Optional
  sqlite step-state alongside sentinels; central pure-function "mixing policy" +
  "step-range math" that's unit-tested. Pay down when next touching transitions.
- **GROK-T-11 (M-T04): tar hardening.** Per-member size caps, optional sha256 at
  build, per-shard `.error` sentinels. Pedantic — current data is trusted JDB/LAION.
- **GROK-T-12 (M-T06): config-only script entrypoints.** Collapse duplicate argparse
  (paths that already live in yaml) so standalone and orchestrated paths can't drift.
- **GROK-T-13 (L-T03): log rotation/retention** — or at least a doctor check for
  total `logs/` size (fixed-name jsonl/.log can accumulate across runs).

Original report retained as `grok_train_bug_report.md` (untracked) for full detail.

---

## Testing-Suite Review (Grok 4.3, 2026-06) — Triaged

External review of the test suite's completeness/accuracy (`grok_testing_bug_report.md`,
untracked). Most numerically precise of the three Grok reports — verified exactly 209
Python test functions, and confirmed **zero** dedicated test coverage for orchestrator,
pipeline_doctor, ablation_harness, flywheel_lib, precompute_all, mine_hard_examples,
cache_manager, campaign_manager (all 8 grepped to 0 test files). Central finding is
correct and important: 209 tests but all concentrated in the ML core (loss/ema/model/
dataset/export) + storage primitives + the parity guard; the autonomous MLOps "brain"
is unit-test-free.

**One verified inaccuracy in the report:** it credits the storage tests (lines 26, 240)
as "improved post-audit to avoid machine-specific literals" — but `test_importable` was
still asserting exact `/Volumes/16TBCold` literals until commit `dff58d8` *this session*
(GROK-T-2). The report conflated the one abstract test with the suite as a whole.

**Relevance:** this is the report most validated by live failures — the flywheel failed
iters 1–10, iter 10 on the precompute→train shard-cache handoff. The report's P0
(synthetic handoff/state tests) is exactly the missing class.

### Done (targeted test for the live failure — actioned 2026-06)

- **GROK-TEST-1: shard-cache-filter contract test** — DONE. Extracted the
  precompute→train filter (was closures inside `train()`) into module-level
  `shard_internal_prefix` / `shard_has_cache` / `filter_shards_with_cache` in
  `train_ip_adapter.py`, and added `train/tests/test_shard_cache_filter.py` (12 tests).
  Covers the exact iter-10 failure modes: empty cache → 0 shards, naming mismatch
  (tar stem ≠ npz key prefix) → excluded, partial precompute (missing `_0049`) →
  excluded, qwen3/vae-only → excluded, ordering preserved. Pure (fake tar paths +
  empty npz touch-files; no mflux/Metal/data). Note: guards the *contract*, not the
  fs-visibility race that triggered iter 10 — that needs a different guard (GROK-TEST-2).

### Open (real but substantial — net-new test suites)

- **GROK-TEST-2 (P0): synthetic orchestrator state-machine harness.** PARTIALLY DONE
  (2026-06). `train/tests/test_orchestrator_state.py` added (14 tests): `derive_chunk_state`
  across all steps + error-precedence + last-done-wins + chunk independence, the
  CHUNK_STEPS/_STEP_TO_STATE contract, and ResourceManager non-GPU token semantics.
  Hermetic via `pipeline_lib.SENTINEL_DIR` monkeypatch (flywheel-safe — separate process).
  STILL OPEN: phantom-hard_ex detection, jetsam retry/backoff, chunk-transition (`_check_ready`
  gating), resume-from-N, last-chunk special cases, dispatch-queue seeding — these live in
  larger orchestrator methods that launch processes; testing them needs further extraction
  of pure cores. (None would catch iter 10's fs race — that's a separate guard.)
- **GROK-TEST-3 (P0): pipeline_doctor black-box tests.** PARTIALLY DONE (2026-06).
  `train/tests/test_pipeline_doctor.py` (12 tests): `_check_proxy_vae` (all 8 branches —
  silent/misconfigured/missing-ckpt/no-eval/failed-gates/healthy/unreadable) and
  `_check_error_sentinels` (none/critical/done-skip/multi-chunk). Hermetic via
  DATA_ROOT+SENTINEL_DIR monkeypatch + _issues reset. STILL OPEN: the phantom-completion,
  training-integrity, precompute-forensics, and stale-log detectors (more coupled to
  DB/log-mtime/heartbeat state — need richer synthetic fixtures).
- **GROK-TEST-4 (P1): model-quality regression automation.** Make `test_quality_features`
  emit machine-readable goldens (final cond/null gap, ip_scale stats, cross/self gap) on a
  fixed small set; gate in a `make test-quality`. No golden quality regression exists today.
- **GROK-TEST-5 (P2): perf/memory assertions.** Assert on telemetry the mini-loop already
  computes (step time, `mx.get_peak_memory`, grad-norm, ema-drift); micro-bench for the
  online-encoder overhead. Critical for the 32 GB tightrope; currently absent.
- **GROK-TEST-6 (P3): precompute_all + cache_manager units.** cache_manager PART DONE
  (2026-06): `train/tests/test_cache_manager.py` (13 tests) covers version_hash
  (determinism, key-order stability, config/git-sha sensitivity, format) and
  encoder_config_subset (vae flux_model dir-stripping, image_size version separation,
  qwen3/siglip/unknown subsets, end-to-end vae version stability). STILL OPEN: the
  precompute_all 1-pass iter + "already done" skip logic (needs synthetic tar/npz
  harness without real encoders).
- **GROK-TEST-7 (P4): property-based + flywheel/ablation DB roundtrip.** hypothesis for
  bucket selection / schedule / quant roundtrips / hard-ex t-sampling; minimal DB +
  warmstart roundtrip tests.

### Maintainability (nice-to-have)

- **GROK-TEST-8: pytest markers** (slow / requires_shards / requires_mps / quality) +
  a `make test-ci` (fast units + smoke + validate + C + run_test, data-req tests noted
  separately); an "untested modules" grep gate.

Original report retained as `grok_testing_bug_report.md` (untracked) for full detail.
