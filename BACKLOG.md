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

**PRODUCTION-READINESS PREREQUISITES (gating — from the warmup-run2/3 over-training finding, 2026-06).**
The IP-adapter over-trains: cond_gap (held-out conditioning quality) degrades while train_loss
falls. In the warmup flywheel this was severe (cond_gap +0.0273 → −0.0057 after ~2000 steps on
42 shards) because tiny per-iteration data → high epochs-per-sample. Production trains over
~200k-record chunks so the *severity* is far lower, but it shares the warm-start-and-keep-training
structure and is **currently unguarded**. Do NOT start a production run without these:

- **PROD-1: Create the held-out validation set + enable T-05.** The trainer's held-out
  validation (`_compute_val_loss`, T-05) is wired but DISABLED because no val set exists
  ("Validation held-out: not found — T-05 disabled"; the doctor's standing `val_set` warning).
  Run `pipeline_ctl create-val-set` + precompute it so T-05 has `_val_shards`. Prerequisite for
  ANY production run — without it, checkpoint selection has zero held-out signal.
- **PROD-2: Select checkpoints + early-stop on held-out COND_GAP, never train_loss.** The core
  lesson: train_loss (and even raw val_loss) fall while cond_gap degrades, so they are misleading
  stop signals. Extend T-05 to compute held-out cond_gap (loss_null − loss_cond) and select/early-
  stop at its peak. Bake the over-training signature (cond_gap down while train_loss down — already
  in the doctor's cond_gap-stall detector) into production monitoring. Also size the per-chunk step
  budget to chunk size (epochs-per-sample is the real overfit driver) — tune via the warmup/ablation.
- **FLYWHEEL-CKPT-1: per-iteration checkpoint archival (start_step=0 collision). DONE 2026-06-10.**
  With `--warmstart-weights` (resume_from_champion) or from-scratch mode, every iteration saved
  `step_0001000.safetensors`, so `ckpt_path = ckpts[-1]` resolved to the same clobbered file each
  iter → get_best's recorded path pointed at the latest (not best) weights, reintroducing
  compounding at iter-3+. **Fixed:** in start_step=0 modes the orchestrator now `os.replace`s the
  iteration's file to a unique `iter{N}_step_*.safetensors`, records that, and prunes archived
  files to keep only the champion (get_best) + current. resume_from_champion is now safe.

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

**PERF-IO-1: Routine ops = metadata lookups, not disk scans** (Medium — keeps the cold
HDD quiet during the flywheel). Principle: any non-troubleshooting path (status, doctor
summary, shard selection) should answer from metadata (SQLite DBs + per-version
`manifest.json` `record_count`/`shard_count` + sentinels/heartbeats), and full disk scans
should sit behind an explicit `--deep` flag for when you suspect metadata↔disk drift.

Audit (2026-06): the backbone is good — 5 DBs (`shard_index`, `shard_scores`,
`flywheel_history`, `experiments`, `monitoring`), manifests already carry the counts, and the
doctor's summary `records` is already manifest-backed. Remaining offenders that still walk the
cold dirs on routine calls:
- **DONE (2026-06):** `pipeline_doctor` listed the same cold precompute version dir ~2-3× per
  encoder per run (coverage count + `.tmp.npz` + `.npz.tmp.npz` hunts) — added a
  per-invocation memoized `_listdir_cached` so each cold dir is walked once.
- `pipeline_doctor._count_shards_with_precomp` (per-chunk precompute coverage) still walks the
  cold version dir — needs **per-chunk/shard coverage metadata** (extend the manifest, or
  derive from `shard_scores.db`) to become an O(1) lookup. Gotcha: must stay chunk-scoped.
- `pipeline_doctor._count_shards_for_chunk` → `shard_index.db` (count shard_ids in range).
  Gotcha: DB = *indexed* shards vs listdir = shards *present on disk*; validate equivalence
  before swapping.
- **`.tmp.npz` / `.npz.tmp.npz` crash-artifact hunts** → move behind `--deep` (they're
  troubleshooting, not routine — currently run on every invocation).
- `shard_selector.scan_shard_pool` globs the cold pool → `shard_index.db`.
- `pipeline_status._count_precomputed/_count_tars_recursive` rglob the **hot** staging dir
  (TTL-cached, lower priority — no manifest on in-flight staging; add a per-iter staging
  manifest if it becomes hot).

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

**ABL-FIDELITY: ablation-proxy fidelity + cross-run quality gate** (the ablation arm is a short, cold-start proxy for a warm-started long run — two fixes, both landed as opt-in/instrument, validation deferred to a GPU-free-window):

- ~~**Arm warm-start (opt-in)**~~ — DONE (default OFF). Arms cold-start (harness gets caches+shards, no checkpoint); the flywheel's own per-iter training warm-starts — a fidelity gap. Added `train_ip_adapter --warmstart-weights` (loads adapter weights but keeps start_step=0 — fresh schedule, unlike `--resume` which continues the step count), harness passthrough, and `orchestrator._ablation_warmstart_ckpt(fw_cfg, ckpt_dir)` gated on `ablation_warmstart_arms` (default false). Kept off because warm-start trades fidelity for *discrimination* (arms diverge less in 1000 steps from a shared checkpoint) — which is better is empirical, settle it with the gate below. Tests in test_orchestrator_state.py.
- ~~**Cross-run quality gate (instrument)**~~ — DONE (built, GPU eval deferred). `train/scripts/quality_gate.py`: `compare_quality(current, previous)` (pure verdict core — clip_i/clip_t/aesthetic higher-better, lpips/fid lower-better; REGRESSION/IMPROVED/NEUTRAL/NO_BASELINE) + `run_quality_gate` glue (golden-set eval → weight_registry register → compare-to-prior; eval/registry injectable) + CLI (`--fail-on-regression` gates a pipeline). 16 tests in test_quality_gate.py. This is the apples-to-apples *output*-quality comparison that validates whether an ablation-chosen config actually improved the long run vs the previous one — and the instrument to A/B warm-start vs cold-start arms.
- ~~**Auto-run wiring**~~ — DONE (default off). Flywheel campaign-end hook runs `quality_gate.py` on the champion checkpoint (`fw_db.get_best`) when `quality_gate: true` is set in the flywheel config; pure `orchestrator._quality_gate_target` gates it (4 tests). Skipped entirely when the flag is off (today). **OPEN (needs GPU / golden set):** configure the golden set + run the gate on the M5 Max; then the warm-start-arms A/B to set that default. No automatic golden regression *gate* in the chunk pipeline yet (GROK-TEST-4).

~~**ABL-1: Trial-level wallclock timeout** (High — safety)~~ — DONE (class `TrialTimer`
in `ablation_harness.py`, wired into `_run_one`; `trial_timeout_secs` config, default
14400). Unit-tested 2026-06 in `train/tests/test_ablation_safety.py` (fires SIGTERM after
timeout; cancel() before timeout is a no-op). The backlog entry was stale — the feature
shipped but had no test coverage.

---

~~**ABL-2: Multi-signal early stopping** (High — quality + time savings)~~ — DONE.
`EarlyStopper` monitors all four signals (cond_gap floor, loss/NaN instant-kill,
grad-norm explosion over 3 snapshots, ref_gap style-dead) with `trigger_reason` for
DB logging. Unit-tested 2026-06 in `train/tests/test_ablation_safety.py` (incl. the
backlog's own success criterion: constant loss=9.0 → kill on snapshot 1). Backlog
entry was stale — shipped without test coverage. Original design retained below.

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

~~**ABL-4: Pareto scatter plot in HTML report** (Medium — interpretability)~~ — DONE.
The scatter (`drawPareto` in `ablation_harness.py`: ref_gap X / cond_gap Y, Pareto
points starred + outlined, a gold "best-compromise" point, zero-lines, axis labels)
already existed — the entry was stale. The remaining spec gap, a **hover tooltip
showing the combo + score + metrics for any trial**, was added 2026-06 (vanilla JS,
no deps): `paretoTip` div + canvas-coord hit-testing in `drawPareto`. Locked in by
`train/tests/test_ablation_report.py` (renders synthetic results, asserts the
template fully formats and the tooltip/hover elements are present — the report is a
brittle str.format() template where one unbalanced brace silently breaks rendering).

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

~~**PRECOMP-2: Tiny proxy VAE encoder** (Low — long-term throughput)~~ — IMPLEMENTED
in v3.18.0 + v3.19.0 (`train/vae_distill/`). The design deviated from this sketch
(task-specific stride-8 encoder, not EfficientNet-B0; composite loss = channel-norm
MSE + decoded MSE + frequency-weighted MSE + distribution matching, not plain MSE)
for the reasons in `plans/precomp2-proxy-vae-design.md`. Ships small/default/medium
variants (3.4M/6.0M/9.3M), ProxyVAE with confidence gating + regression detection +
quality modes, evaluate_vae_proxy.py, benchmark_vae_proxy.py, compare_downstream_quality.py,
and precompute_all/orchestrator integration. **Still pending validation** (idle GPU +
a trained proxy): the 5–7× speedup number and the downstream A/B verdict — see the
flywheel-gated items. Migration: `plans/proxy-vae-v3.19-migration.md`.
  - **Validation runbook: `plans/proxy-vae-validation-runbook.md`** — train→Tier1→
    benchmark→Tier2 with pass gates, fallback-threshold tuning, subsampling alt arm,
    and sequencing against flywheel-idle / M5 bring-up.
  - **Two BLOCKING pre-trust items (must land before any proxy latent ships), from
    `grok_proxy_vae_analysis.md`:** (C-1) no golden test that C `iris_vae` encode/decode
    matches the teacher VAE the proxy is trained against — proxy can pass Tier-1/2 vs
    teacher yet degrade real C inference; add per-channel + decoded-LPIPS/PSNR golden +
    CPU/Metal decode parity. (C-2) brittle ad-hoc `vae/config.json` strstr/atoi parse can
    select the wrong z/scale/shift/BN branch; schema-validate or golden the resolved VAE
    config. These are doable NOW (no GPU) and are prerequisites regardless.

---

**PRECOMP-3: Precompute version key is over-sensitive — bound to whole-repo git SHA, not encoder identity** (High — directly causes redundant re-precompute; compounds with the v3.23.0 cache fix) — **CODE FIX LANDED; cold-data migration pending run-end.**

**Status (2026-06-03):** `cache_manager.version_hash` no longer mixes in the git SHA — the key is now `sha256(config_subset)` where `config_subset` carries a per-encoder `code_version` (`ENCODER_CODE_VERSION = {qwen3,vae,siglip → "1"}`, bump only on an encoder-output-semantics change). `encoder_config_subset` folds `code_version` in; `git_sha` is still accepted (call-site compat) and recorded in the manifest for provenance but does not affect the key. Added `cache_manager.py consolidate <encoder|all> [--apply]` (hardlink-based union of same-identity version dirs, dedup by filename, repoint `current`, drop redundant dirs) + `list`. `list_versions` now skips the `current` symlink (was a phantom version). Tests in `train/tests/test_cache_manager.py` (git-SHA-independence, code_version bump, consolidate dry-run/apply/skip-different-identity/ignore-empty-stub). **Remaining:** after the warmup-run1 flywheel finishes (iter 15) and the orchestrator is restarted to pick up the new key, run `cache_manager.py consolidate all --apply` on `/Volumes/16TBCold/precomputed` to fold the per-SHA dirs (`v_d9a32b`, `v_c56d1c`, + iters 12–15) into the single canonical `v_059443`-class version. Until that restart the running orchestrator keeps the old git-SHA key (it won't reload the edited module), so iters 12–15 still publish per-SHA dirs — expected, the consolidation captures them.

Original analysis (kept for context):

`version_hash(config_subset, git_sha)` in [cache_manager.py:60](train/scripts/cache_manager.py#L60) computes the precompute cache version as `sha256(json(config_subset) + git_sha[:8])`. The `config_subset` correctly captures encoder identity (e.g. qwen3: model + extracted layers + think_tags), but mixing in the **whole-repo git SHA** means *any* commit anywhere in the tree — orchestrator, doctor, docs, an unrelated C change — produces a brand-new version dir and forces a full re-precompute of the same shard pool, even though the encoder and its outputs are byte-identical.

**Observed impact (flywheel warmup-run1).** iters 10 and 11 published functionally identical Qwen3 config (`Qwen3-4B`, layers `[8,17,26]`, `think_tags:true`) under two different versions purely because we committed code between iterations:
- iter 10 → `v_d9a32b` (git_sha `31e647a4`, 194,127 records, 40 shards)
- iter 11 → `v_c56d1c` (git_sha `5d36078c`, 199,271 records, 40 shards)

The two iters share 12 anchor shards (`000000`–`000011`); those 12 were re-encoded from scratch in iter 11 instead of being reused from iter 10's cold version, despite zero change to the encoder. With the cache key fixed, iter 11 would have reused iter 10's overlapping shards (and the hot/cold `current` symlinks would not fragment across SHAs).

**Fix direction.** Key the precompute version on **encoder identity only**, not the global repo SHA:
- Replace `git_sha[:8]` in the version hash with a narrow fingerprint of the code that actually affects this encoder's output — e.g. a hash of the relevant encoder module(s) (`precompute_all.py` encoder fn + the model/config), or a hand-maintained `ENCODER_CODE_VERSION` bumped only when the encoding semantics change. The principle is already stated in this file's Warm-Start section ("6-month-old embeddings remain valid if the encoder is unchanged") — the implementation contradicts it.
- Keep `encoder_config_subset` as-is (it's correct).
- On change, add a one-shot migration/alias so existing `v_d9a32b`/`v_c56d1c` data is discoverable under the new key (or a `cache_manager.py --warm-start-precompute` pass to fold them forward) — don't strand the ~68 shards already on cold.

**Tests:** version_hash is stable across an unrelated git commit (same encoder config + different repo SHA → same version); version_hash *changes* when the encoder config subset changes; migration/alias resolves old version dirs. Pure-function, no GPU, no network — same pattern as the existing cache_manager tests.

---

**PRECOMP-4: Aspect-ratio bucketing, end-to-end** — full implementation plan: [plans/precomp4-aspect-bucketing.md](plans/precomp4-aspect-bucketing.md) (decision: per-image-bucket, NOT re-shard, to preserve the carry-forward cache + scores; shared `aspect_bucket` primitive landed in train/ip_adapter/bucketing.py). Original sketch: (re-shard by shape → multi-res precompute → aspect-aware loader → refine)** (High for final quality; **not** needed for the warmup bootstrap, which is correctly pinned to 512² square)

The data path squashes every image to a single **square** resolution and the training
loader does not honour aspect ratio, so the multi-resolution machinery that exists is
non-functional. Discovered when the quick training test hit 100% VAE cache-miss
(latent shape ≠ training bucket); root-caused to a chain of three issues:

1. **Precompute is single-resolution square.** `_preprocess_vae` does
   `PilImage.resize((image_size, image_size))` ([precompute_all.py:162](train/scripts/precompute_all.py#L162)) —
   a squash, not aspect-preserving. Every latent is `(32, S/8, S/8)` regardless of
   source aspect, so only one training bucket can ever match.
2. **The shard loader random-buckets.** `make_prefetch_loader(bucket=None)` picks
   `rng.choice(BUCKETS)` **per shard** ([dataset.py:445](train/ip_adapter/dataset.py#L445)),
   not the per-image aspect bucket. The aspect-aware `_select_bucket` exists but is
   unused by the shard loader. So even with per-bucket precompute, batches would
   mismatch. `_load_vae_latent` then rejects any shape mismatch as a cache miss
   ([dataset.py:235](train/ip_adapter/dataset.py#L235)); in cached mode (no live VAE)
   a miss is an unrecoverable skip → ~100% skip → trainer exits.
3. **Shards mix aspect ratios.** Tars hold arbitrary-shape images, so a per-shard
   single bucket squashes all of a shard's varied images into one shape.

**Current workaround (shipped):** `data.bucket: [512, 512]` pins training to the one
bucket that matches the square precompute (wired in train_ip_adapter.py; set in
stage1_512px.yaml). Training is then *consistent* (squashed image ↔ squashed latent)
and valid — the bootstrap IP-adapter learns aspect-tolerant style conditioning and is
a usable warm-start. Quality cost: geometric distortion + square output bias.

**Roadmap to distortion-free multi-aspect (do after the warmup proves the loop):**
- **Re-shard by aspect** — group images into aspect-homogeneous shards (recommended:
  makes the existing per-shard-single-bucket loader *correct*, gives clean same-shape
  caches and retrace-free homogeneous batches). Alternative: keep tars, do per-image
  bucket precompute (variable-shape latents per shard) + an aspect-grouping loader.
- **Multi-res precompute** — encode each shard at its bucket's native (non-square)
  resolution using the **teacher VAE** as ground truth, with the **distilled proxy
  (PRECOMP-2)** for speed (confidence-gated, teacher fallback; teacher authoritative
  for golden/validation). Large buckets use VAE tiling (PRECOMP-1). Retrain the proxy
  on the new pairs — the current proxy only saw square 512².
- **Fix the loader** — assign each image to `_select_bucket(w, h)` instead of
  `rng.choice`; build homogeneous batches; drop the 512² pin.
- **Keep SigLIP consistent** — SigLIP is architecturally fixed to 384² (so "full
  image" still resizes); today's 384² squash is *coherent* with the 512² VAE squash
  (conditioning aligns with target). Under the aspect regime, feed SigLIP the *same*
  aspect-bucketed image resized to 384² so conditioning and target stay aligned.
  Higher-detail style needs a larger/tiled vision encoder — separate, lower priority.
- **Refine the adapter** — warm-start from the 512² foundation weights; fine-tune on
  the multi-aspect cache to correct geometry/proportion handling.

Relates to PRECOMP-1 (tiling for large buckets) and PRECOMP-2 (proxy). The re-shard is
the big one-time data job; everything else composes on top.

**PRECOMP-5: Hot precompute cache (defer-rmtree LRU) — cut ~30% of cold↔hot transfers**
(Medium — pipeline throughput) — full plan: [plans/hot-precompute-cache.md](plans/hot-precompute-cache.md).
Each iter currently archives precompute to cold then `rmtree`s it from hot, so recurring
shards get **re-copied** cold→hot next time (hot/cold are different physical devices, so
staging is a real copy, not a symlink). **Measured** on warmup-run1 (13 iters,
flywheel_history.db `selected_shards`): **30.6% of each iter's shards were selected in the
immediately prior iter** (LRU-1 ≈ LRU-3 ≈ unbounded at 30.6/31.0/33.1% — almost all reuse is
consecutive; rising 30%→40% as a campaign matures). So a **1-iteration** cache captures ~92%
of the benefit. Fix: in the publish step, **decouple archive-to-cold (keep) from
rmtree-from-hot (defer)** — keep a content-addressed hot precompute cache, link recurring
shards into the per-iter dir, copy only cache-misses, evict LRU under a hard
`storage.hot_precompute_cache_gb` budget; add the hot cache as the first tier in
`effective_dir`. Footprint ≈ one iter's precompute (~170–210 GB — VAE 0.5 MB/rec × ~4.9k
rec/shard ≈ 2.5 GB/shard, ~4–5 GB/shard all encoders, × ~42 shards; from the cold VAE
manifest — the order already staged transiently today). Transfer-I/O win only (not encode/compute); complementary to PRECOMP-2
(proxy) and `--subsample-per-shard`. Touches live orchestrator publish/cleanup + DataStager
+ `effective_dir` — **land in a non-campaign window.**

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

## v3.21.0 Quality Loop — Remaining (GPU-gated + wiring)

v3.21.0 shipped the four quality-loop subsystems (experiment tracking, monitoring,
golden-set gate, preference store) with all decision/recording logic tested. These
are the deferred pieces — the GPU-bound execution and the orchestrator wiring that
needs a restart. See `plans/quality-loop-v3.21-migration.md` §7–8.

- **QL-1 (GPU): wire `run_golden_eval`'s 3-arm training + scoring** —
  `evaluate_golden_set.py` currently raises `NotImplementedError`. Reuse the
  `compare_downstream_quality.py` per-arm machinery (precompute golden shards per
  arm: real / proxy+fallback / proxy-forced → train short IP-Adapter → score
  CLIP-I/T + aesthetic + LPIPS + FID). Feed the arm metrics into the already-tested
  `regression_gate()` + `write_results()` + `maybe_disable_proxy()`. Run on an idle
  GPU. ~1–2 days. This is what actually exercises the trust gate end-to-end.
- **QL-2 (one-time, needs data): build the golden-set manifest** — a fixed,
  stratified ~3000-image set (≥30% natural LAION/COYO vs synthetic JourneyDB) via
  `campaign_manager.py`. Set `golden_set.manifest` in the config. Prereq for QL-1.
- **QL-3 (GPU): champion image-generation loop for the data flywheel** — sample
  candidate images from the Champion model, score them, and register the keepers via
  `PreferenceStore.record_synthetic()` so they re-enter scoring. The provenance store
  + `apply_preferences()` blend are done and tested; the generation/scoring loop is
  the GPU-bound piece. ~1–2 days.
- **QL-4 (wiring, needs orchestrator restart): auto-populate the new stores** —
  call `experiments.tracker.record_from_campaign()` from the orchestrator's
  post-campaign path, and the `monitoring.collector` recorders (proxy fallback rate,
  precompute speed, train loss, champion quality, system disk/mem) from the flywheel
  loop, so `--quality-report` / `--monitor` populate automatically. Pure code, but
  takes effect only after an orchestrator restart (don't interrupt a live campaign).
- ~~**QL-5: Slack alert sink**~~ — DONE (2026-06). `monitoring/sinks.py`:
  slack_payload (header + per-alert blocks + text fallback) + dispatch_slack
  (severity filter, env-var webhook resolution, mockable poster) + post_to_slack
  (stdlib urllib, no deps). Wired as `pipeline_doctor.py --monitor --notify`;
  webhook URL is a secret read from $SLACK_WEBHOOK_URL (config holds only the env
  var NAME). 16 tests (test_slack_sink.py); network POST mocked, never hit.

- **QL-6: Bidirectional Slack — trigger pipeline actions from Slack** (Medium —
  ops velocity; outbound QL-5 is the prereq, done). Let the operator drive the
  pipeline from Slack (e.g. `/iris status`, `/iris pause-flywheel`,
  `/iris resume`, `/iris quality-report`, `/iris promote-champion`, `/iris
  golden-eval`). This is a different shape from the outbound webhook — it needs an
  *inbound* path, so it carries real design + security weight:

  - **Transport.** Two options: (a) Slack Slash Commands / Events API → needs a
    public HTTPS endpoint Slack can POST to (ngrok/Cloudflare Tunnel/Tailscale
    Funnel from the Mac, or a tiny relay). (b) Socket Mode → a persistent outbound
    WebSocket (no inbound port; best for a home/NAT box). **Prefer Socket Mode** for
    a single Mac behind NAT — no public endpoint, no tunnel.
  - **Security (the hard part).** Verify Slack request signatures (`X-Slack-Signature`
    + timestamp, HMAC over the raw body with the signing secret) OR Socket Mode app
    token; allow-list the workspace + a specific channel + specific user IDs;
    map each command to an explicit, **whitelisted** action (never exec arbitrary
    strings); destructive actions (pause/promote/disable-proxy/launch) require a
    confirm step or a separate `--armed` token. Secrets via env (`SLACK_APP_TOKEN`,
    `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`), never in repo/config.
  - **Action surface.** Map to the existing control plane, don't reinvent it:
    `pipeline_ctl.py` (pause/resume/stop flywheel, restart-orchestrator,
    clear-error, force-next-chunk), `pipeline_doctor.py --ai/--monitor/--quality-report`,
    `experiments.tracker promote`. Read-only commands (status/report) are safe to
    enable first; gate state-changing ones behind the confirm/allow-list.
  - **Process model.** A small long-running `monitoring/slack_bot.py` (Socket Mode
    listener) in its own tmux window, supervised like the other pipeline processes;
    structured command log + audit trail of who triggered what; rate-limit.
  - **Dependency note.** Socket Mode needs the `slack_sdk` package in train/.venv
    (acceptable as a train-only dev dep; pin it). The Slash-Command/HTTP route can
    stay stdlib but requires the public endpoint.
  - **Build order.** (1) read-only `/iris status|report` over Socket Mode with
    signature/allow-list; (2) safe controls (pause/resume) behind confirm;
    (3) heavier actions (promote-champion, golden-eval launch) with `--armed`.
  - **Effort:** ~2–3 days incl. auth, allow-list, audit log, tests (command
    parsing + auth + action mapping are unit-testable with the Slack transport
    mocked, same pattern as test_slack_sink.py).

- **QL-7: `iris_slackd` — minimal hardened command daemon** — CODE+TESTS DONE
  (2026-06-05), deploy gated on Slack tokens. `train/scripts/slackd_core.py` (pure
  policy: command table, auth, parse, resolve, rate limit, audit — 39 tests incl. a
  2000-case fuzz proving no message text can synthesise an out-of-table argv) +
  `train/scripts/iris_slackd.py` (thin Socket-Mode shell + `Daemon.handle`, 15
  tests, `--self-test` runs the full pipeline with no SDK/network). All three build
  phases (read-only, armed gate, confirm-gated `stop`) implemented. Remaining to go
  live: `train/.venv/bin/pip install slack_sdk`, create the Slack app (Socket Mode +
  scopes per plan), export the 4 env vars, launch in an `iris-slackd` tmux window.
  See `plans/slack-command-daemon.md`. Original design notes below.

  A deliberately tiny long-running daemon whose only job is: listen to one Slack
  channel, and on a recognised command, invoke a **fixed, whitelisted pipeline
  script** — nothing else. Security and smallness are the features.

  - **Threat model first.** Assume the channel could be seen by more people than
    intended and that Slack messages are attacker-influenceable. Therefore: the
    daemon must be incapable of running anything not in its compiled-in allow-list,
    regardless of message content.
  - **Hard rules (non-negotiable):**
    - **Socket Mode only** — outbound WebSocket, no inbound port, no public
      endpoint, no tunnel. App token + bot token from env (`SLACK_APP_TOKEN`,
      `SLACK_BOT_TOKEN`); never in repo/config.
    - **Single channel allow-list** (`SLACK_CMD_CHANNEL` id) + **user allow-list**
      (`SLACK_CMD_USERS`, comma-sep ids). Messages from anywhere/anyone else are
      ignored and audit-logged.
    - **Fixed command table** mapping a short keyword → an explicit `argv list`
      (e.g. `status → [venv_py, pipeline_status.py]`, `doctor → [venv_py,
      pipeline_doctor.py, --ai]`, `quality → […, --quality-report]`,
      `pause → […, pipeline_ctl.py, pause-flywheel]`, `resume → […, resume-flywheel]`).
      **No user-supplied arguments** in v1 (or a strict per-command validator —
      e.g. chunk must match `^[1-9][0-9]?$`). **Never** `shell=True`, never string
      interpolation of message text into a command.
    - **Default read-only.** State-changing commands (pause/resume/clear-error)
      require `IRIS_SLACKD_ARMED=1` in the daemon's env; otherwise they are
      acknowledged-but-refused. Destructive ones (stop, force-next-chunk) always
      require an explicit `confirm <token>` reply.
    - **No GPU, no direct pipeline mutation** — the daemon only spawns the existing
      scripts (which already own locking/sentinels); it never touches state itself.
    - **Audit everything** to `logs/slackd.jsonl`: ts, user, channel, raw text,
      matched command (or "rejected: reason"), exit code. Rate-limit per user.
    - **Output back to the channel** = the script's stdout tail (truncated), via the
      QL-5 sink. Long output → a file snippet, not a wall of text.
  - **Process model.** `monitoring/slack_bot.py` (or `train/scripts/iris_slackd.py`),
    run in its own `iris-slackd` tmux window under caffeinate, supervised like the
    other pipeline processes; clean shutdown on SIGTERM; reconnect on socket drop.
  - **Why separate from QL-6.** QL-6 is the general design space (could grow
    interactive buttons, broad actions). QL-7 is the *minimum viable, maximally
    hardened* realisation: a closed command set that can only launch known scripts.
    Ship QL-7; let QL-6's richer surface be optional later.
  - **Dependency.** `slack_sdk` (Socket Mode) pinned in train/.venv — the one new
    train-only dep, flagged as a conscious choice.
  - **Tests (transport mocked, no network):** allow-list enforcement (wrong
    channel/user rejected + logged), command-table mapping (keyword → exact argv),
    unknown/again-malformed message rejected, armed-gate on state-changing commands,
    confirm-token flow, audit-log shape. Same mock pattern as test_slack_sink.py.
  - **Build order:** (1) listener + auth/allow-list + `status`/`doctor`/`quality`
    (read-only) + audit log; (2) `pause`/`resume` behind `IRIS_SLACKD_ARMED`;
    (3) `confirm`-gated heavier actions. **Effort:** ~1.5–2 days for (1)+(2).

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

- ~~**TB-001: Qwen3 Tokenizer Correctness** (P1)~~ — DONE. `debug/test_tokenizer.c`
  (58 tests) runs in `make test`.
- ~~**TB-010: Flash Attention vs Naive Attention Parity** (P2)~~ — DONE.
  `debug/test_kernels.c` flash-vs-naive parity runs in `make test`.
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

- ~~**GROK-1 (was H-01): Remove dead `rope_freqs`**~~ — DONE (2026-06). Removed the
  `rope_freqs` field, the 1D `compute_rope_freqs` helper, all 3 alloc/compute load
  sites + the free in `iris_transformer_flux.c` (verified never read in any forward).
  `iris_qwen3.c`'s own `compute_rope_freqs` is unrelated and untouched. Clean BLAS build,
  `make test-unit` green.
- ~~**GROK-2 (was M-08): Remove dead `iris_vae_load(FILE*)`**~~ — DONE (2026-06). Removed
  the legacy `.bin` VAE loader + its exclusive helpers (`read_uint32`, `read_floats`,
  `load_resblock`, `load_attnblock`) from `iris_vae.c` and the decl from `iris.c`
  (zero call sites). `free_resblock`/`free_attnblock` and the `_sf` safetensors loaders
  kept. Clean build.
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
- ~~**GROK-9 (M-02/M-03/M-06): Error-reporting + cleanup consistency.**~~ — CLOSED:
  WON'T DO (2026-06). Standardise on `set_error`, add `d[N-1]='\0'` after
  `strncpy(,,N-1)`, tighten OOM cleanup. Closed as a defensive sweep across the
  inference-path C files with no functional gain (the `strncpy` buffers are already
  `calloc`-zeroed, so the null-term is a no-op today). Same risk/reward as GROK-10.
  Reconsider only if a real defect surfaces or these files are being refactored anyway.
- ~~**GROK-10 (M-01/M-04/L-07): Architectural-invariant asserts + magic-number cleanup.**~~
  — CLOSED: WON'T DO (2026-06, user-skipped). Validate derived dims after config parse
  (`hidden == heads*head_dim`, `head_dim == 4*axis_dim`); add `static_assert`. Closed
  because it requires touching 4 config-parse sites across the critical 5.3k-LOC
  inference file for marginal benefit — the dims come from the model's own (correct)
  config; a speculative sweep the backlog itself flags "do not sweep". Reconsider only
  when already editing those load paths for a real feature.

### Build / nits (observation)

- **GROK-11 (M-07/M-10): Monolithic units + always-clean backend builds.** `iris_transformer_flux.c`
  (~5.3k LOC), `iris_metal.m` (~7k LOC); per-backend build dirs and only re-`xxd` shaders
  on change would speed dev. Cosmetic — defer.

Original report retained as `grok_bug_report.md` (untracked) for full detail.

---

## Proxy-VAE Design vs C Inference Review (Grok, 2026-06) — Triaged

External static review of the proxy-VAE design (`grok_proxy_vae_analysis.md`,
untracked) cross-referenced against the C `iris_vae`. File:line refs verified
accurate; severities re-calibrated. Net: the report inflated two "Criticals" that
block nothing today (the flywheel runs the real VAE; the proxy is pending
validation and not enabled), and most findings duplicate existing items. The one
genuine, non-duplicate insight is the **three-distribution framing**: teacher
(Python diffusers/mflux, makes precompute latents) → proxy (MLX student,
approximates teacher encoder) → **C `iris_vae` decoder** (inference). The proxy is
trained to match the teacher *encoder*, but generated latents are decoded by the C
decoder, so the binding invariant is proxy-distribution ↔ C-decoder. The design's
decoded-LPIPS loss already binds proxy outputs to decode correctly under the *real
decoder*; the residual gap is teacher-vs-C parity, which is **pre-existing and
independent of the proxy** (precompute already uses the Python teacher today).

Duplicates (no new action): config-parser brittleness = GROK-6; dead `.bin` VAE
loader = GROK-2; "pending validation" already stated under PRECOMP-2; GPU-resident
encode / scalar patch loops are perf-backlog.

**Full triage of all 7 grok reports: `plans/grok-review-triage.md`** (de-duplicated +
prioritized with disposition). Net: parser brittleness (G-5), VAE↔teacher parity (G-6),
generic-build correctness (G-9), and dead code (G-10) are RESOLVED; live priorities are
**G-1 (C-side IP-Adapter — the endgame: trained adapters can't yet run in the `iris`
binary)**, **G-2 (MLOps state-machine tests — in progress via pure-core extraction)**, and
GPU/idle-gated **G-3 (hardcoded paths) / G-4 (B-METAL-01 CPU softmax fallback)**.

- **GROK-VAE-1: C VAE inference-ground-truth guard** — DONE (2026-06). Added
  `debug/test_vae.c` (wired into `make test-unit`, CPU-only, flywheel-safe): builds
  small architecturally-real Flux/Z VAEs with seeded synthetic weights and exercises
  the real CPU encode/decode to assert shapes (16× compression), finiteness,
  bit-determinism, z_channels→latent_channels wiring (32→128, 16→64), and that the
  latent normalization **branch** is config-selected and uses the exact
  `(x-shift)*scaling` vs `(x-mean)/sqrt(var+eps)` form (the path the brittle
  `vae/config.json` parser feeds — guards C-2/GROK-6 at the integration level). Also
  added a "ground truth" contract note to the `iris_vae.c` header + AGENT.md/CLAUDE.md
  Flux-VAE section. This is the foundation a future teacher-golden comparison plugs
  into (a true cross-impl golden needs the GPU + Python teacher → flywheel-gated).
- **GROK-VAE-2 (Low): Z-Image proxy gap** — the proxy design is Flux-32ch only;
  Z-Image VAE (16ch + explicit scale/shift, no BN/quant) gets no precompute relief
  despite full C support. Note for if/when a Z-Image training campaign is run; not a
  defect. Would need a 16ch student preset + Z teacher path.

Original report retained as `grok_proxy_vae_analysis.md` (untracked) for full detail.

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

- ~~**GROK-T-3 (H-T05): `shell=True` in doctor --fix**~~ — AUDITED + HARDENED (2026-06).
  `run_fix_mode` is human-gated: each fix is printed verbatim and run only after a
  per-issue `y` confirm (not auto-exec); `shell=True` is deliberate (fix strings use
  pipes/globs/`&&`). Audited every interpolated value in every `fix=` string: all come
  from the pipeline's controlled namespace — path constants (DATA_ROOT/SENTINEL_DIR/
  PRECOMP_DIR/CKPT_DIR/LOG_DIR/…), the fixed qwen3/vae/siglip encoder set, numeric
  chunk/step (checkpoint ones `:07d`-forced to int), and compiled-in tmux window-name
  constants. **No external/untrusted data** (tar member names, file contents,
  heartbeat/log JSON, env, argv) reaches any fix string. The one operator-typed value
  — the campaign `name` in `_check_campaigns` — is now `shlex.quote()`d (defense-in-depth:
  robust to names with spaces/metachars; it was already human-gated, never
  attacker-reachable on a single-user box). 70 doctor tests green.
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
- **GROK-T-7 (M-T03): config schema + load_and_validate.** DONE (2026-06).
  `train/scripts/config_schema.py`: `validate_config(cfg)` flags unknown top-level
  sections (typos), non-mapping sections (ERROR), unknown `flywheel_health` knobs
  (doctor-consumed, silent-default class), and missing flywheel required keys; a
  standalone CLI and a `pipeline_ctl.py validate-config <path>` subcommand (exits
  non-zero on ERROR). Conservative — all 16 real configs pass clean. 11 tests in
  `test_config_schema.py`. Deferred: full per-key dataclass typing + wiring into
  `load_config` (kept non-invasive — validate is opt-in, not silent on every load).
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
  CHUNK_STEPS/_STEP_TO_STATE contract, and ResourceManager non-GPU token semantics, plus _resolve_proxy_vae_args
  (config→precompute flags + per-campaign overrides). Hermetic via
  `pipeline_lib.SENTINEL_DIR` monkeypatch (flywheel-safe — separate process).
  EXTENDED (2026-06): jetsam-vs-code-error retry/backoff now covered — extracted the
  pure `_retry_policy(reason, restarts)` decision out of the crash-handler method and
  tested it (jetsam → JETSAM_MAX_RETRIES with backoff, code error → 1 retry no delay,
  stop-at-limit), plus the already-pure crash functions `_parse_exit_code_from_msg`,
  `_diagnose_crash` (non-137 code_error / 137 jetsam-confirmed / 137 jetsam-assumed via
  monkeypatched system-log query), `_parse_last_mem_from_log`. 11 new tests.
  COMPLETED (2026-06): the remaining decision logic is now extracted + tested —
  `_ready_gate` (chunk-transition: wait_prev_train/wait_gpu/wait_stage/proceed_no_stage/
  proceed, incl. chunk-1-never-staged), `_should_attempt_stage` (predecessor-promoted
  staging gate), `_restart_plan` in pipeline_ctl (resume-from-N: reset exactly N..total,
  never a chunk before N), `_load_open_dispatch_ids` (dispatch seeding: open vs
  resolved-after-open), and the doctor phantom hard-ex last-chunk branch (INFO on last
  chunk vs CRITICAL mid-pipeline). All behaviour-preserving extractions; tests in
  test_orchestrator_state.py / test_pipeline_ctl.py / test_pipeline_doctor.py.
  Note: none of these would catch iter 10's fs race — that's the separate shard-cache
  guard (GROK-TEST-1 / GROK-VAE-1).
- **GROK-TEST-3 (P0): pipeline_doctor black-box tests.** PARTIALLY DONE (2026-06).
  `train/tests/test_pipeline_doctor.py` (12 tests): `_check_proxy_vae` (all 8 branches —
  silent/misconfigured/missing-ckpt/no-eval/failed-gates/healthy/unreadable) and
  `_check_error_sentinels` (none/critical/done-skip/multi-chunk), `_check_stale_logs`
  (log-older-than-sentinel WARNING / fresh-silent / no-sentinel-silent), and
  `_check_phantom_completions` (promoted.done-but-no-shards CRITICAL / shards-in-range-ok
  / not-promoted-silent — the headline phantom, same class as iter-10). Hermetic via
  DATA_ROOT + SENTINEL_DIR (both doctor and pipeline_lib bindings) + LOG_DIR/SHARDS_DIR/
  PRECOMP_DIR monkeypatch + _issues reset. EXTENDED (2026-06): training-integrity
  (NaN loss, non-zero exit, short-log-but-done, resume-past-end, clean-silent) and
  precompute-forensics (orphaned .tmp.npz, double-extension crash artifact, low
  coverage, clean-silent) detectors now covered. Also added tests for the newer
  flywheel observability checks — liveness reconciliation, failure-loop +
  fingerprinting, cache-coverage confirmation, trainer-anomaly (chunkless), campaign
  ETA, progress-stall, logs-disk. STILL OPEN: none of note for the doctor.
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
  qwen3/siglip/unknown subsets, end-to-end vae version stability). precompute_all
  building blocks DONE too (2026-06): `train/tests/test_precompute_all.py` (14 tests)
  — _scan_existing (the "already done" set), iter_shard (jpg+txt pairing, png,
  corrupt-tar-warns, whitespace strip), _save_npz_atomic (no torn writes) + the
  done→skip round-trip. STILL OPEN: full _process_shard_inner orchestration (needs
  model stubs).
- **GROK-TEST-7 (P4): property-based + flywheel/ablation DB roundtrip.** DB PART DONE
  (2026-06): `train/tests/test_flywheel_db.py` (10 tests) — insert/get roundtrip,
  selected_shards JSON, ordering, update (status/metrics/failed exit codes), get_best
  cond_gap selection (null-excluded, highest-wins), checkpoint_log + mark_best, campaign
  isolation. Uses an explicit tempdir db_path (never touches live flywheel_history.db).
  ABLATION DB DONE (2026-06): `train/tests/test_ablation_db.py` (15 tests) — AblationDB
  insert/get/update roundtrip, params_hash determinism + is_duplicate (per-run scoped),
  get_best (score-desc, unscored-excluded, limit), scored_only filter, run isolation,
  3-objective Pareto front (dominated excluded / non-dominated both kept), and
  post_train_validation roundtrip (weight_errors JSON, verdict defaults). Tempdir db_path,
  never touches live ablation_history.db. STILL OPEN: property-based (hypothesis) for
  bucket selection / schedule / quant / hard-ex t-sampling — deferred (would add the
  `hypothesis` dev dependency); ablation warmstart roundtrip.

Also new (2026-06): `train/tests/test_dataset_bucketing.py` (17 tests) — the
precompute↔train resolution contract (`_select_bucket` aspect math, `_load_vae_latent`
shape-rejection, and the headline "square-512 latent matches ONLY the (512,512) bucket"
that mandates the `data.bucket` pin). Guards the 4th warmup-run2 blocker; see PRECOMP-4.

### Maintainability (nice-to-have)

- **GROK-TEST-8: pytest markers** (slow / requires_shards / requires_mps / quality) +
  a `make test-ci` (fast units + smoke + validate + C + run_test, data-req tests noted
  separately); an "untested modules" grep gate.

Original report retained as `grok_testing_bug_report.md` (untracked) for full detail.
