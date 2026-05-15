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

- **Inference tier** (internal NVMe SSD, ~2–8 TB) — lowest-latency path for the live inference server and web app. Holds only the active weights and embeddings needed to serve requests. Populated by the stager from hot; never written to by training.
- **Hot storage** (`/Volumes/2TBSSD`, 2 TB TB5 SSD) — fast working area for the active + next compute window only. Pipeline reads shards, precompute, and weights from here during training.
- **Cold storage** (`/Volumes/16TBCold`, 16 TB spinning disk) — source of truth and long-term knowledge base. Never auto-deleted by pipeline operations.

**JIT Data Stager** manages all directions:
- **Cold → Hot (staging):** before a compute window, stages raw data, precompute, and weights from cold to hot. Uses symlinks when on the same filesystem (near-instant); atomic copies across filesystems. `_check_hot_space()` enforces `staging_margin_gb` before any transfer.
- **Hot → Cold (archiving):** after a successful run, archives newly generated precompute embeddings, weight checkpoints, and per-campaign telemetry to cold. This is the write path — without it, cold never grows and warm-starts never improve.
- **Hot → Inference (promote):** after archiving, copies or symlinks the active checkpoint + its precompute version to the inference tier so the web app picks up new weights without restarting.

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

2. **Block-by-block injection (TRAIN-6)** ✓ Implemented — see COMPLETED_BACKLOG.md. Cost: 4.7× slower than old path (8.38s vs 1.78s/step clean), 21.54 GB bwd peak. Decision: gated off; re-evaluate for from-scratch runs. Option C (correct_forward_q) is the active production path.

3. **Source data curation (PIPELINE-27)** ✓ Done — see COMPLETED_BACKLOG.md. Provenance sidecars + `shard_scorer.py` + scored download order in `pipeline_setup.py`. Requires `tgz_scores.json` to populate (needs one full pipeline run with provenance-enabled shards).

4. **QUALITY-10 ablation harness** ✓ Done — see COMPLETED_BACKLOG.md.

**References:** PIPELINE-27 (data curation, done), PIPELINE-25 (raw pool, done).
**Dependency summary:** Item 1 (resolution/scale) — unblocked, needs 1024px memory profiling run. Item 2 (TRAIN-6) — done, gated. Item 3 (PIPELINE-27) — done; scoring activates after first pipeline run with provenance-enabled shards. Item 4 (QUALITY-10) — done.

---

## Pipeline Improvements

**PIPE-SMOKE-1: Smoke/dev runs invisible to pipeline_doctor and pipeline_status** ✅ done (confirmed 2026-05-15)

Smoke and dev runs launch `train_ip_adapter.py` directly — no pipeline sentinels, no heartbeat files, no orchestrator involvement. `pipeline_doctor` and `pipeline_status` are completely blind to them. To check progress you must `tail -f /tmp/dev_run.log` or attach to the tmux window manually.

**Fix:** wire direct runs into the trainer heartbeat system.
- `train_ip_adapter.py`: when a `--run-name` flag is provided (or when config contains a `run_name` key), write a heartbeat to `/Volumes/2TBSSD/.heartbeat/trainer_{run_name}.json` at the normal `log_every` cadence. Same fields as the production trainer heartbeat (step, loss, cond_gap, mem peaks, eta_sec, etc.).
- `pipeline_status.py`: scan for `trainer_*.json` heartbeats in addition to `trainer.json`; display each active direct run as a separate row with its run name.
- `pipeline_doctor.py`: include direct-run heartbeats in the trainer health check; warn if a named run's heartbeat is stale (>5 min) but the tmux window still exists (likely hung).
- Standard run names: `smoke` (100 steps), `dev` (1,000 steps). The config file name is a reasonable default if `--run-name` is not given.

**Prerequisite for:** running dev/smoke tests with the same visibility as production runs.

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

**PIPELINE-25/26/27/28/29** ✅ done (2026-05-14) — see COMPLETED_BACKLOG.md.

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

**DATAMGMT-1: data_explorer diagnose — automated detection of stale, redundant, and misplaced data** ✅ done (2026-05-15)

Add a `data_explorer.py diagnose` subcommand that surfaces the same insights a human would spot by running `du -sh` and comparing directories — without requiring manual analysis.

**Checks to implement:**

1. **Redundant staging/** — if `hot_root/staging/chunk*` exists AND `cold_root/converted/` has the same tars, flag staging as redundant copies. Report estimated reclaimable space. (`staging/` is the V1 pipeline artifact; V2 stager stages directly to `shards/` and `precomputed/`.)

2. **Stale hot data** — dirs on hot that are not needed for the currently-staged chunks:
   - `hot_root/anchor_shards/` if already present on cold
   - `hot_root/hard_examples/` if already present on cold
   - `hot_root/precomputed/` versions older than cold's `current` symlink
   - `hot_root/checkpoints/` entries already archived to cold weights campaign

3. **Misplaced source-of-truth data** — data that exists only on hot with no cold copy (would be lost if hot fails):
   - `hot_root/shard_scores.db` without a corresponding `cold_root/metadata/shard_scores.db`
   - `hot_root/flywheel_history.db` without cold backup
   - Any `hot_root/weights/` checkpoint without a matching entry in `cold_root/weights/`

4. **Cross-tier duplicate detection** — directories present at the same path on both hot and cold where hot is not a symlink to cold (data duplicated unnecessarily):
   - `anchor_shards/`, `hard_examples/` — cold is source of truth; hot copy is redundant once cold has it

5. **Orphaned precompute versions** — version dirs in `cold_root/precomputed/{enc}/` that are not pointed to by `current` symlink and exceed `keep_versions` (already partially covered by `maintenance --prune` but should also appear in diagnose output)

6. **Training gap detection** — chunks in cold that have shards but no corresponding precompute coverage on cold (precompute was never run for that chunk range)

**Output format:**

```
  DIAGNOSE — data anomalies detected
  ────────────────────────────────────────────────────
  REDUNDANT  staging/chunk1-4 (282 GB) — same tars in cold/converted/journeydb
             → safe to delete: rm -rf /Volumes/2TBSSD/staging/
  REDUNDANT  anchor_shards/ on hot (21 GB) — cold copy exists
             → safe to delete: rm -rf /Volumes/2TBSSD/anchor_shards/
  STALE      checkpoints/stage1/chunk1_final.* — archived to flywheel-20260507
             → safe to delete: rm -rf /Volumes/2TBSSD/checkpoints/stage1/chunk1_final.*
  OK         shard_scores.db — cold backup present (47h ago)
  OK         weights/ — 2 campaigns archived to cold
```

**Implementation notes:**
- Read-only by default; never deletes. Print `rm -rf` commands as suggestions only.
- `--fix` flag: executes the suggested deletions with confirmation per item.
- `--json` flag for scripting (includes `reclaimable_gb` field).
- Runs after `du` sizes are collected (reuse parallel `_du_kb` infrastructure from `status`).
- Should complete in <30 seconds on hot SSD; cold scans use the same timeout-bound `_du_kb` pattern.

**PIPELINE-30: Inference tier — three-tier storage with internal NVMe as serving layer** (Medium priority, pre-web-app)

Add a third storage tier for the inference server and web app. The internal NVMe SSD becomes the lowest-latency path; hot (external SSD) remains the training working area; cold (HDD) remains the knowledge base.

**Motivation:** serving requests from the external SSD introduces USB-C bus latency and contention with pipeline I/O. The internal NVMe gives 2–3× lower latency for model weight reads and keeps inference bandwidth independent of training I/O.

**Config changes** — add to `storage:` block:
```yaml
storage:
  inference_root: /path/to/internal/nvme   # e.g. /Users/fredrikhult/inference
  # existing keys unchanged
  hot_root: /Volumes/2TBSSD
  cold_root: /Volumes/16TBCold
```

**`train/scripts/pipeline_lib.py`** — add `INFERENCE_ROOT` constant alongside `COLD_ROOT`.

**`train/scripts/data_stager.py`** — add `promote_to_inference(chunk)`:
- Copies/symlinks active checkpoint (`cold_root/weights/best/*.safetensors`) to `inference_root/weights/current.safetensors`.
- Copies/symlinks active precompute version dirs from hot to `inference_root/precomputed/{enc}/current` (symlinks if same filesystem, copies otherwise).
- Writes `inference_root/manifest.json`: `{"checkpoint": "...", "precompute_version": "v_xxx", "promoted_at": "..."}` — lets the web app detect weight updates without polling the file directly.
- Atomic: writes to a `inference_root/.staging/` dir, then renames into place so the web app never sees a partial state.
- Called by `archive_chunk()` after `_archive_dbs()`.

**`train/scripts/data_explorer.py status`** — add inference tier row showing:
- Inference root path and free GB.
- Active checkpoint name + promoted-at timestamp from `manifest.json`.
- Precompute version per encoder (from `manifest.json`).
- `stale` flag if `promoted_at` is >2 chunks old (new weights exist in cold but inference hasn't been updated).

**`train/scripts/pipeline_doctor.py`** — add WARNING if `cold_root/weights/best/` symlink target is newer than `inference_root/manifest.json`'s `promoted_at` (i.e., stager archived a better checkpoint but never promoted it to inference).

**Layout on inference tier:**
```
{inference_root}/
  weights/
    current.safetensors    ← active checkpoint (copy or symlink)
    current.json           ← sidecar with metrics
  precomputed/
    qwen3/current/         ← symlink or copy of hot version dir
    vae/current/
    siglip/current/
  manifest.json            ← atomic-updated on each promotion
```

**When to implement:** before the web app serves live inference. Not needed while inference runs from manually specified paths.

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

Discovered in a full four-layer code review (C core, Metal/GPU, pipeline scripts, trainer). Severity levels: **crash** > **wrong-result** > **leak** > **latent** > **minor**.

### C Core

**BUG-C-001: Transformer forward NULL not checked in samplers** — crash
`iris_sample.c`. All Euler samplers (`iris_sample_euler`, `_euler_with_refs`, `_euler_with_multi_refs`, `_euler_zimage`, `_euler_cfg`) use the transformer return value immediately without a NULL check. The transformer returns NULL on OOM or GPU fallback failure; the sampler then dereferences NULL in the velocity update loop. In the CFG path both `v_uncond` and `v_cond` are unchecked. Fix: check return value and abort the step cleanly.

**BUG-C-002: Stack buffer overflow in samplers via step_times array** — crash (latent)
`iris_sample.c` (~lines 305, 392, 522, 600, 682, 764, 847). Every sampler declares `double step_times[IRIS_MAX_STEPS]` (256 entries) and writes `step_times[step]` up to `num_steps-1` with no in-function guard. CLI validates `steps <= 256` before calling, but the samplers carry no guard themselves — unsafe as a library API if the caller skips validation.

**BUG-C-003: NULL dereference after unchecked malloc in load_double_block_weights** — crash
`iris_transformer_flux.c` (~lines 589–639, f32 path). Two `malloc` calls for `img_mlp_gate_weight` / `img_mlp_up_weight` (and text FFN equivalents) have no NULL check; `memcpy` is called immediately on the return value. Under memory pressure during model load, this crashes.

**BUG-C-004: Memory leak in ensure_work_buffers on partial allocation failure** — leak
`iris_transformer_flux.c` (~lines 1468–1529). When any of the ~20 sequential allocations fails, the function returns -1 but does not free the already-allocated buffers from the same call. They are orphaned — `work_seq_alloc = 0` signals failure so the struct never owns them, but they are not freed.

**BUG-C-005: Shadow variable in DEBUG build hides block_idx parameter** — wrong result (debug only)
`iris_transformer_flux.c` (~line 2222, `#ifdef DEBUG_DOUBLE_BLOCK`). A `static int block_idx = 0` inside the debug block shadows the function parameter of the same name. All debug prints show a sequential counter 0,1,2,... instead of the actual block being executed.

**BUG-C-006: Integer overflow in iris_image_create calloc size** — crash / heap corruption
`iris_image.c` (~line 29). `calloc(width * height * channels, sizeof(uint8_t))` — all three operands are `int`. At 32768×32768×4 the product overflows int32, wraps to zero or negative, allocates a tiny buffer, and subsequent pixel writes corrupt the heap. Fix: cast to `(size_t)width * (size_t)height * (size_t)channels`.

**BUG-C-007: Silent uninitialized output on malloc failure in iris_linear_nobias_bf16** — wrong result
`iris_kernels.c` (~lines 322–332). If `malloc` fails, the function returns without writing to `y`. The caller receives garbage from whatever was in the output buffer; the void return gives no indication of failure.

**BUG-C-008: NULL dereference in zi_final_forward on malloc failure** — crash
`iris_transformer_zimage.c` (~line 1593). `float *normed = malloc(seq * dim * sizeof(float))` has no NULL check. The immediately following loop dereferences `normed + s * dim`, crashing under memory pressure during Z-Image generation.

**BUG-C-009: Integer truncation of ftell() result in iris_img2img_debug_py** — minor
`iris.c` (~line 1815). `int noise_size = ftell(f) / sizeof(float)` — `ftell` returns `long`. Silently truncates for files larger than ~2 GB. Correct type is `long` or `size_t`.

---

### Metal / GPU

**BUG-M-001: NULL/nil check missing on MTLBuffer allocation at four sites** — crash
`iris_metal.m` (~lines 301–308, 739, 1074, 1182). `[g_device newBufferWithBytes:weights length:size options:]` can return nil on OOM. All four locations pass the result immediately to MPSMatrix or a compute pass without a nil check, crashing when Metal is under memory pressure.

**BUG-M-002: MTLBuffer leak in iris_gpu_tensor_alloc_persistent on struct malloc failure** — leak
`iris_metal.m` (~line 2539–2542). When `malloc(sizeof(struct iris_gpu_tensor))` fails, the function returns NULL but the already-allocated persistent MTLBuffer (`buf`) is never released. Contrast with `iris_gpu_tensor_alloc` which correctly calls `pool_release_buffer(buf)` before returning NULL.

**BUG-M-003: Pool buffer leak in iris_metal_sgemm_impl on bufferC allocation failure** — leak
`iris_metal.m` (~lines 760–765). When `cache_B=0` and batch mode is active, `bufferB` is obtained from the pool. If the subsequent `bufferC` allocation fails, the error path releases `bufferA` and `bufferC` but has no corresponding release for `bufferB`. The pool slot leaks permanently, shrinking the 64-slot activation buffer pool over many calls.

**BUG-M-004: Batch mode atomicity broken in iris_gpu_attention_mps_bf16** — wrong result
`iris_metal.m` (~lines 3124–3128). Between Phase 2 (CPU softmax) and Phase 3 (scores @ V), the function unconditionally commits and resets `g_tensor_cmd` regardless of `g_tensor_batch_mode`. In batch mode this splits GPU work from the same logical batch across two separate command buffers, violating ordering guarantees for any caller that assumed atomic batch submission.

**BUG-M-005: attention_fused_bf16 kernel: shared_q[128] has no kernel-level head_dim guard** — latent corruption
`iris_shaders.metal`. `threadgroup float shared_q[128]` is written for `d` up to `head_dim - 1` with no assertion that `head_dim <= 128`. All current models use head_dim=128 so it is safe today, but any future model with head_dim > 128 would produce silent threadgroup memory corruption.

**BUG-M-006: SDPA graph cache does not include scale in cache key** — latent wrong result
`iris_metal.m` `get_sdpa_graph_cache` and `get_sdpa_graph_cache_f32`. Cache keys on `{seq_q, seq_k, num_heads, head_dim}` only. The `scale` parameter is baked into the MPSGraph at construction time as a constant but is not stored in the key. Two calls with the same shape but different scale values silently reuse the first graph's scale. Safe today (scale always derived from head_dim, which is in the key), but fragile if scale is ever caller-specified.

**BUG-M-007: iris_gpu_linear unconditionally commits batch in the bias path** — wrong result
`iris_metal.m` (~lines 2786–2796). When a bias vector is provided, the function commits and waits on the live command buffer unconditionally — including when `g_tensor_batch_mode=1`. This prematurely splits any in-progress batch across two command buffers, breaking batch atomicity for all callers using `iris_gpu_linear` with bias in batch mode.

**BUG-M-008: causal_attention_fused shared_scores[512] guard is only in Obj-C caller, not in kernel** — latent corruption
`iris_shaders.metal` (~line 675 and ~line 1956). Both `causal_attention_fused` and `causal_attention_fused_bf16` allocate `threadgroup float shared_scores[512]` and write `shared_scores[key_idx]` for `key_idx` in `[0, seq)`. The only protection against `seq > 512` is a check in the Obj-C caller. The kernel itself has no guard; a new call site without that check would silently corrupt threadgroup memory.

**BUG-M-009: iris_metal_sgemm_batch allocates __strong id<MTLBuffer>* array with calloc** — ARC violation
`iris_metal.m` (~line 1618). `calloc` is used to allocate a `__strong id<MTLBuffer>*` array, bypassing ARC initialization. Works because nil is bitwise zero on current Apple platforms, but is an ARC specification violation. Future toolchain changes or non-zero nil representations could cause incorrect retain/release behavior.

---

### Pipeline Scripts

**BUG-P-001: _link_or_copy() creates absolute-path symlinks** — data loss on remount
`data_stager.py` (~line 586). `os.symlink(src.resolve(), dst)` creates absolute symlinks. If the cold or hot volume remounts at a different path (e.g. macOS assigns `/Volumes/2TBSSD 1`), all staged symlinks become dangling and precomputed data becomes unreachable. `update_best_symlinks()` in the same file correctly uses `os.path.relpath()` — the two code paths are inconsistent. Fix: `os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)`.

**BUG-P-002: _atomic_symlink() uses hardcoded temp name — race condition under concurrency** — wrong result (latent)
`data_stager.py` (~line 657) and `cache_manager.py` (~line 314). Both use a hardcoded temp name (`.current_stg_tmp` / `.current_tmp`) in the link's parent directory. Concurrent calls for the same encoder directory have the second process's `unlink()` delete the first process's temp symlink before `os.replace()` completes, leaving the symlink pointing to stale state. Currently serialized by the orchestrator; latent if parallelism is added. Fix: uuid-based or `tempfile.mkstemp()`-based temp names.

**BUG-P-003: pipeline_doctor.py generates syntactically wrong stager remediation commands** — wrong operator guidance
`pipeline_doctor.py` (~lines 766, 818). Fix commands suggest `data_stager.py --chunk N --phase archive` / `--phase stage`. The stager uses subcommand CLI: correct syntax is `data_stager.py archive --chunk N`. The suggested commands produce an argparse error; the doctor's fix advice actively misleads the operator.

**BUG-P-004: _recover_prep_window() resets hung-prep timer to zero after orchestrator restart** — monitoring gap
`orchestrator.py` (~line 1040). `_recover_prep_window()` rebuilds `_active_prep` without a `"started_at"` key. `_poll_prep_window()` falls back to `time.time()` on `.get("started_at", time.time())`, resetting the hung-prep elapsed timer to zero on every orchestrator restart. A prep step that ran for 5 hours before a restart needs another full `PREP_HUNG_HOURS` (6h) before the alert fires; a genuinely hung prep could run indefinitely through repeated restarts.

**BUG-P-005: --fix mode in data_explorer.py uses cmd.split() — breaks on paths with spaces** — latent wrong result
`data_explorer.py` (~line 2213). Repair commands are run via `subprocess.run(cmd.split(), ...)`. If any auto-generated fix command contains a path with a space, the arguments are mangled and the fix silently operates on the wrong target or does nothing. Current pipeline paths have no spaces; this is latent.

**BUG-P-006: Dead self-import in _render_status_html() leaves disk stats permanently None** — dead code
`data_explorer.py` (~lines 798–805). `from train.scripts.data_explorer import _disk_stats` inside a `try/except` always raises `ModuleNotFoundError`. `cold_used`, `cold_total`, `hot_used`, `hot_total` are initialized to None and never assigned. The function falls back to reading disk stats from `result["disk"]` so there is no visible crash, but the import block is dead code that should be removed.

---

### Trainer

**BUG-T-001: NameError crash when siglip cache is configured without qwen3/vae** — crash
`train/train_ip_adapter.py` (line 702). `_internal_prefix` is a nested function defined only inside `if qwen3_dir and vae_dir:` (lines 668–693) but referenced unconditionally at line 702 in the siglip cache path. If `qwen3_cache_dir` or `vae_cache_dir` is absent (e.g. live-inference mode with precomputed siglip only), the function is never defined and line 702 raises `NameError`, killing the process before training begins.

**BUG-T-002: grad_clip_pct always 0.0 in machine-readable heartbeat** — wrong metric
`train/train_ip_adapter.py` (lines 1688, 1794). `_grad_clip_steps` is reset to 0 at line 1688, then `grad_clip_pct` in `write_heartbeat` is computed from the already-zeroed counter at line 1794. The human-readable log print reads the counter before the reset (correct); the machine-readable heartbeat read by `pipeline_doctor` and `orchestrator` always shows 0% gradient clipping regardless of how many clipping events occurred.

**BUG-T-003: _compute_val_loss measures null-image loss, not IP-adapter conditioning quality** — wrong metric
`train/train_ip_adapter.py` (lines 1118–1125). `use_null_image = mx.array(True)` is hardcoded, so val_loss measures base flow-matching reconstruction with the adapter zeroed out — not image-conditioning quality. Checkpoint selection (`_purge_old_checkpoints`) keeps the "best 3" checkpoints ranked by val_loss; these are selected for the lowest null-image reconstruction loss, which has no relationship to IP-adapter performance. Fix: load siglip features from val shards and use `use_null_image = mx.array(False)`.

**BUG-T-004: Checkpoint written directly to final filename — partial file visible on crash** — minor
`train/train_ip_adapter.py` (lines 304–313). `_save_safetensors_streaming` writes directly to `step_NNNNNNN.safetensors`. If the process is killed mid-write (OOM, caffeinate failure), the corrupt partial file is picked up by `_purge_old_checkpoints` sorted listing and may displace a valid older checkpoint before the corruption is detected at load time. Fix: write to `.tmp` then `os.rename()` atomically.

**BUG-T-005: ip_out reshape assumes uniform hidden dim across all blocks** — latent wrong result
`train/train_ip_adapter.py` (lines 967, 982). `d_inner = h_final.shape[2]` is used to reshape `ip_out` from all blocks. This silently asserts `H_i * Hd_i == d_inner` for every double and single block. Safe for current Flux 4B and 9B (all blocks share the same `H*Hd`), but would crash with an opaque reshape error on any future variant with heterogeneous block dimensions.
