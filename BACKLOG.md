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

Proof-of-concept validated (2026-05-11, `train/reports/ip_adapter_v1/`): the adapter architecture and training signal are sound. The model responds to the style reference with coherent, stable output (CLIP-I 0.53, no NaN, correct image structure). The gap to production quality is entirely a matter of scale and refinement — no architectural rethink is required. The following improvements are known to provide benefit, roughly in priority order:

1. **Larger resolution + more training steps** — the v1 run was a `--small` configuration. Higher resolution (1024px) exposes finer style features to the SigLIP encoder; more steps allow the PerceiverResampler and K/V projections to build a richer style space. Expected CLIP-I gain: +0.05–0.10. **Note (32 GB):** 1024px training on 4B Flux currently peaks at ~26 GB at 512px; 1024px roughly doubles the sequence length (256→1024 image tokens), which increases attention memory. Feasibility needs a short profiling run before committing to a full 1024px flywheel.

**Dependency summary:** Item 1 (resolution/scale) — unblocked, needs 1024px memory profiling run. Items 2–4 (TRAIN-6, PIPELINE-27, QUALITY-10) — done, see COMPLETED_BACKLOG.md.

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

**DEDUP-1: Clean the converted pool at source; one-off script for existing shards**

**Architectural motivation:** data quality should be enforced at the point of production, not at the point of consumption. The current blocklist-propagation approach (added 2026-05-15) requires every consumer (precompute, training, mining, any future step) to implement filtering correctly and silently fails if any consumer omits it.

**The right dedup point is the converted pool, not the training shards.** The converted pool (`cold_root/converted/journeydb/`) is the canonical source from which all training shards are built. If a pool WDS file is clean, every future shard build from it is automatically clean — across all chunks, campaigns, and scales — with no downstream filtering required. Since the vast majority of JDB data is still in the pool and has never been materialised into training shards, cleaning the pool addresses essentially all data at the right level.

**Two-track implementation:**

*Track 1 — converted pool dedup (pipeline change):*
After `download_convert.py` writes a new WDS file to the pool, run CLIP embedding on it directly, extend the cumulative FAISS index, find and remove any duplicate records (rewrite the pool WDS tar in place), and update the pool's record manifest. `build_shards.py` then always draws from clean source files; no shard-level filtering is ever needed.

*Track 2 — existing shard + precompute cleanup (one-off script):*
For the small number of training shards already built from pool data (current and past chunks): a `clean_existing_shards.py` script reads `duplicate_ids.txt`, rewrites each shard tar in place to remove duplicate records, and prunes the corresponding precomputed `.npz` files from each encoder cache dir + updates the manifest record count. Precompute cache entries are keyed by record ID so non-duplicate embeddings remain valid — no GPU recompute needed, only O(N_dups) file deletions and a manifest update.

**What becomes dead code once Track 1 is live:**
- `--blocklist` args in `precompute_all.py`, `train_ip_adapter.py`, `mine_hard_examples.py`
- `blocklist` param in `ip_adapter/dataset.py`
- Dedup state archive/restore in `data_stager._archive_dbs()` and `pipeline_setup._restore_dedup_state()`
- All `--blocklist` wiring in orchestrator launch commands for precompute, training, mining

**What stays:**
- Cumulative FAISS index (still needed for cross-chunk near-duplicate detection)
- `duplicate_ids.txt` passed to `build_shards.py --blocklist` for the next chunk (cross-chunk contamination prevention at build time)
- Cold archive of FAISS index + blocklist

**When to do:** after the first full production run confirms the blocklist workaround is stable. Not blocking anything today.

---

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

Found by full line-by-line agent review after the initial 32-bug pass. **0 of 14 fixed.**

### CRITICAL

**PIPE-C-001: GPU lock TOCTOU race — two GPU steps can run concurrently**
`train/scripts/orchestrator.py` — `ResourceManager.request()`
Calls `gpu_lock_holder() is None` then `acquire_gpu_lock(holder)` but ignores the return value. In the gap between check and call, another process can claim the lock. `self._holders[token]` is then set and `request()` returns `True` without the orchestrator actually holding the lock. Two GPU-intensive steps (precompute, training) run concurrently → OOM or model corruption.
Fix: `if not acquire_gpu_lock(holder): return False` after the existing `gpu_lock_holder()` check.

**PIPE-C-002: Post-training step error causes permanent pipeline deadlock**
`train/scripts/orchestrator.py` — chunk prep gate
`derive_chunk_state(N)` returns `ERROR` if ANY step for chunk N errors — including `mine`, `validate`, `archive` (post-training steps). The prep gate for chunk N+1 checks `derive_chunk_state(N) not in _training_or_later`; `ChunkState.ERROR` is not in `_training_or_later`, so chunk N+1 never starts prep. Once chunk N hits a permanent post-training error, the pipeline deadlocks silently — no doctor alert, no escalation.
Fix: replace `derive_chunk_state(chunk-1) not in _training_or_later` with `not is_done(chunk-1, "train")`.

**PIPE-C-003: Val set sentinel naming mismatch — unconditional re-download every invocation**
`train/scripts/pipeline_ctl.py` line 435 vs `train/scripts/download_convert.py`
`cmd_create_val_set` looks for `.converted/{idx:03d}.done` (with `.done` extension). `download_convert.py` writes `.converted/{idx:03d}` (no extension). The sentinel is never found → the val set triggers a fresh download and conversion on every `create-val-set` call, even when the pool already contains the converted tgz.
Fix: remove `.done` suffix in `pipeline_ctl.py` line 435.

**PIPE-C-004: Non-atomic CLIP embedding NPZ write — partial file silently skips dedup**
`train/scripts/clip_dedup.py` line 325
`np.savez(out_npz, ...)` writes directly to the output path. A crash mid-write leaves a partial `.npz`. On resume, the existence check passes → that shard is silently skipped in FAISS → its duplicates persist into training data.
Fix: write to `{stem}.tmp`, call `np.savez(tmp_stem, ...)`, then `os.replace(tmp_stem + ".npz", out_npz)`.

### HIGH

**PIPE-H-001: Precompute manifest writes layers [9,18,27] but cache_manager uses [8,17,26]**
`train/scripts/precompute_all.py` line 916 (manifest written to staging), `train/scripts/cache_manager.py` line 84 (`encoder_config_subset`)
The manifest records `"layers": [9, 18, 27]` (1-indexed); the orchestrator's `_promote_chunk` computes the version hash using `"layers": [8, 17, 26]` (0-indexed). The hashes diverge → `pipeline_doctor` reports qwen3 cache as mismatched/incomplete on every run after precompute completes. False CRITICAL alerts degrade operator trust.
Fix: change `"layers": [9, 18, 27]` to `"layers": [8, 17, 26]` in `precompute_all.py` line 916.

**PIPE-H-002: `--new-shards-first` optimization always treats all shards as new**
`train/scripts/precompute_all.py` — `_has_output` closure
Checks for `{stem}.npz` but precomputed files are named `{stem}_{id:04d}.npz`. The checked file never exists → all shards always classified as "new" → `--max-shards` never skips already-covered shards → resume-after-crash re-processes the same shards instead of making forward progress.
Fix: check for any file matching `{stem}_*.npz` pattern: `any(f.startswith(stem + "_") and f.endswith(".npz") for f in os.listdir(d))`.

**PIPE-H-003: Cache-miss `continue` in training loop doesn't increment step counter**
`train/train_ip_adapter.py` lines 1364–1378
`continue` statements for VAE and text encoder cache misses skip the entire `for _grad_i in range(_n_grad_steps)` body. `step += 1` is inside that body. When precompute coverage is poor, `step` stagnates and the training loop never reaches `step >= _end_step` — it runs indefinitely, consuming GPU without making progress.
Fix: add a skip counter; emit a warning after N consecutive skips; add a circuit breaker that exits with an error if skip rate exceeds 50% over 500 iterations.

**PIPE-H-004: Mining uses fixed t=500 — biases hard-example selection**
`train/scripts/mine_hard_examples.py` — `_eval_loss` and `_eval_loss_batch`
Both use `t_int = mx.array([500], ...)`. Flow-matching loss is highly noise-level-dependent. Using only t=500 (the midpoint) ranks samples by difficulty at one timestep only — samples hard at low/high t but easy at t=500 (or vice versa) are systematically mis-ranked. The fixed t was introduced to make the single-record fallback consistent with the batch path; both are now wrong together.
Fix: sample t from the logit-normal distribution matching training: `mx.clip((mx.sigmoid(mx.random.normal(shape=(B,))) * 1000).astype(mx.int32), 0, 999)`.

**PIPE-H-005: Temp training config `/tmp/iris_train_chunk{N}_config.yaml` never cleaned up**
`train/scripts/orchestrator.py` lines 1622–1628 (`_start_training`)
The temp YAML is created but never stored as an instance attribute. No `finally` block or `_post_step` cleanup call. Files accumulate across chunks and reboots (until manual `/tmp` cleanup). Stale configs survive orchestrator restart — if chunk numbering is reused, a stale config could be overwritten in a timing gap.
Fix: store the path as `self._active_train_tmp_cfg`; unlink in `_post_step` when `"train"` step completes or errors.

**PIPE-H-006: Non-atomic converted tar write in download_convert.py**
`train/scripts/download_convert.py` — `_convert_tgz`
The output `.tar` is written directly to its final path. A crash mid-write leaves a partial file with no sentinel. If a concurrent `build_shards.py` running from the shared pool reads the partial file, it gets a truncated-archive error. A stale pool sentinel + partial tar causes the tgz to be silently skipped at Level 0, producing a corrupt converted file.
Fix: write to `{idx:03d}.tar.tmp`, then `os.replace` to final path.

### MEDIUM

**PIPE-M-001: `filter_shards --start-idx` crashes on non-numeric shard stems**
`train/scripts/filter_shards.py` line 293
`int(os.path.splitext(os.path.basename(s))[0])` has no `try/except` around the `int()` conversion. Any non-numeric shard stem (WikiArt, LAION) causes a `ValueError` crash with no useful error message.
Fix: wrap in a helper `_shard_idx(s) -> Optional[int]`; skip shards where it returns `None`.

**PIPE-M-002: No existence check on cache dirs in mine_hard_examples**
`train/scripts/mine_hard_examples.py` lines 390–391
`os.listdir(args.qwen3_cache)` and `os.listdir(args.vae_cache)` called without checking the directories exist. Misconfiguration produces a generic `FileNotFoundError` with no indication of which argument is wrong.
Fix: explicit check + `sys.exit(1)` with `"--qwen3-cache directory not found: {path}"` message.

**PIPE-M-003: anchor_mix_ratio + hard_mix_ratio ≥ 1.0 produces nonsensical shard counts**
`train/ip_adapter/dataset.py` lines 362–368
`remaining_ratio = 1.0 - anchor_mix_ratio - hard_mix_ratio`. When sum ≥ 1.0, `remaining_ratio` ≤ 0; the `max(..., 0.01)` clamp makes `n_anchor` 20–100× the intended count. Training epoch is flooded with repeated anchor examples.
Fix: validate at startup: `if anchor_mix_ratio + hard_mix_ratio >= 1.0: raise ValueError(...)`.

**PIPE-M-004: Variable `max_seq` per batch triggers MLX graph retrace on long-caption batches**
`train/ip_adapter/dataset.py` line 467
`max_seq = max(max_actual, 512)`. If any caption exceeds 512 tokens, `max_seq > 512` for that batch. Rare long-caption batches produce a unique input shape → MLX retraces the graph (~5–15 seconds each) with no log indication. Warmed PSO graph cache does not cover these shapes.
Fix: enforce a hard cap at tokenization time — truncate at 512 tokens and always pad to exactly 512.

### LOW

**PIPE-L-001: Mining docstring says "random t" but uses fixed t=500**
`train/scripts/mine_hard_examples.py` — `_eval_loss` docstring
Says "Random t for unbiased ranking" but implementation uses fixed t=500. Actively misleading.
Fix: update docstring (or fix implementation per PIPE-H-004).

**PIPE-L-002: Control file write is non-atomic — pause signal can be lost on a poll cycle**
`train/scripts/orchestrator.py` line 2083
`open(CONTROL_FILE, "w")` + `json.dump()` is not atomic. `_check_control_signals` catching `json.JSONDecodeError` on empty/partial read silently drops the pause signal for that cycle.
Fix: write to `.tmp` + `os.replace`.

**PIPE-L-003: EMA drift metric samples first-5 parameters by dict insertion order**
`train/train_ip_adapter.py` lines 1564–1577
`list(_flat_online)[:5]` — first 5 tensors by class definition order, which may be bias vectors rather than weight matrices. Drift signal is noisy and unrepresentative.
Fix: sort by tensor size descending; sample top-5 largest.

---

## Known Bugs — Inference/Training Cross-Reference Review 2026-05-15

Found by agent reviewing all C inference files against Python training/precompute code. **0 of 9 fixed.**

### CRITICAL

**INFER-C-001: Chat template mismatch — all precomputed Qwen3 embeddings are invalid**
`train/scripts/precompute_all.py` lines 365–367
`_encode_qwen3()` calls `tokenizer.apply_chat_template(..., add_generation_prompt=True)` **without** `enable_thinking=False`. The Jinja template only appends `<think>\n\n</think>\n\n` when `enable_thinking` is explicitly `False`; without the kwarg, the block is skipped.

The C inference path (`iris_qwen3_tokenizer.c`, `qwen3_tokenize_chat`) always appends the think-tag tokens for Flux (when `skip_think_tags=0`). The live-encoding fallback in `train_ip_adapter.py` correctly passes `enable_thinking=False`.

Result: precomputed `.npz` embeddings are computed from a token sequence 7 tokens shorter than what inference and live-training-encoding produce. All pad alignments are shifted. IP-adapter weights trained from the precomputed cache are misaligned with inference embeddings whenever the cache is used (the default fast path). **The entire existing precomputed cache must be regenerated after this fix.**

Fix: add `enable_thinking=False` to `apply_chat_template` call in `precompute_all.py`; update the cache version hash in `cache_manager.py` to force regeneration.

### HIGH

**INFER-H-001: Strict aliasing UB in f32→f16 conversion in iris_metal.m**
`iris_metal.m` line 932
```c
uint32_t bits = *(uint32_t *)&f32;
```
Type-puns `float *` to `uint32_t *` — undefined behavior under C strict aliasing rules. All BF16/F16 weight conversions at load time pass through this path. Silently miscompiles under LTO or aggressive auto-vectorization.
Fix: `memcpy(&bits, &f32, sizeof(bits))`.

### MEDIUM

**INFER-M-001: Pad token embeddings differ between training (zeros) and inference (model output)**
`train/scripts/precompute_all.py` line 404–405; `train/ip_adapter/dataset.py`
Precompute saves only real tokens `h[j, :sl]`; dataloader zero-pads to 512. At inference, the full 512-token sequence runs through the model — pad positions produce non-zero hidden states (RMSNorm + attention + MLP still operate on them). Systematic discrepancy at every pad position beyond the real token count.
Fix option A: save the full padded-to-512 embeddings from precompute (larger storage, exact inference match). Fix option B: zero out pad positions in the C inference path after the forward pass.

**INFER-M-002: `iris_metal_sgemm_batch()` malloc not NULL-checked before use**
`iris_metal.m` lines 1640–1657
`cPtrs = malloc(batch_count * sizeof(...))` result is used immediately without a NULL check. Crash under memory pressure with no useful error message.
Fix: add NULL check; return/abort with a descriptive error on failure.

### LOW

**INFER-L-001: Dead `len` mutation statements in `parse_json_string` second pass**
`iris_qwen3_tokenizer.c` lines 227 and 232
After `malloc(len + 1)` has been called, the second pass updates `len` on lines 227 and 232 but `len` is never read again. No overflow — just dead statements that mislead future readers.
Fix: remove both dead `len` updates.

**INFER-L-002: Dead f32 `apply_rope_2d` Metal kernel — PSO compiled but never dispatched**
`iris_shaders.metal` lines 389–428; `iris_metal.m`
The `apply_rope_2d` f32 kernel uses the half-split RoPE convention. Active code uses `apply_rope_2d_bf16` (consecutive-pair, correct). `g_rope_2d_pipeline` is compiled and allocated at shader init but never dispatched. Wastes Metal shader compile time and PSO memory.
Fix: remove the kernel from `iris_shaders.metal` and associated init/dispatch code from `iris_metal.m`.

**INFER-L-003: Dead `iris_apply_rope` and `iris_compute_rope_freqs` in iris_kernels.c**
`iris_kernels.c` — `iris_apply_rope()` and `iris_compute_rope_freqs()`
Implement the half-split RoPE convention. No callers anywhere in the codebase. Dead code that creates confusion about which RoPE convention is active (consecutive-pair is correct).
Fix: remove both functions.

**INFER-L-004: Dead low-level Metal functions in iris_metal.m**
`iris_metal.m` / `iris_metal.h` — `iris_bf16_qk_rms_norm`, `iris_bf16_silu`, `iris_bf16_silu_mul`, `iris_metal_qk_rms_norm`
Declared in `iris_metal.h`, defined in `iris_metal.m`, but have no callers internally or externally. The transformer uses the tensor-API wrappers instead.
Fix: remove all four from `iris_metal.m` and their declarations from `iris_metal.h`.

**INFER-L-005: iris_qwen3.h doc comment still says "layers 9, 18, 27"**
`iris_qwen3.h` lines 100–101
Comment says "Extracts hidden states from layers 9, 18, 27" but the constants are `QWEN3_OUTPUT_LAYER_1=8`, `_2=17`, `_3=26` and the code uses 0-indexed loop counters. Misleading to any reader cross-referencing with the HuggingFace convention.
Fix: update comment to "Extracts hidden states at the output of loop iterations 8, 17, 26 (0-indexed, i.e. after `inner.layers[i]` runs)".
