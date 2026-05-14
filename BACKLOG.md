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

Two-tier hot/cold split:

- **Cold storage** (`/Volumes/16TBCold`, 16 TB spinning disk) — source of truth and long-term knowledge base. Never auto-deleted by pipeline operations.
- **Hot storage** (`/Volumes/2TBSSD`, 2 TB TB5 SSD) — fast working area for the active + next compute window only.

**JIT Data Stager** manages both directions with equal importance:
- **Cold → Hot (staging):** before a compute window, stages raw data, precompute symlinks, and weights from cold to hot. Uses symlinks when on the same filesystem (near-instant); atomic copies across filesystems. `_check_hot_space()` enforces `staging_margin_gb` before any transfer.
- **Hot → Cold (archiving):** after a successful run, archives newly generated precompute embeddings, weight checkpoints, and per-campaign telemetry to cold. This is the write path — without it, cold never grows and warm-starts never improve.

Both directions are first-class operations. Staging populates the working set; archiving accumulates the knowledge. Neither is optional.

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

3. **Source data curation (PIPELINE-27)** ⛔ blocked on PIPELINE-25 — over time, bias shard selection toward high-signal style examples (diverse, distinctive styles; high self/cross-ref gap; low redundancy). Requires PIPELINE-25 (persistent raw pool); full JDB pool is ~202 tgzs × ~2-3 GB ≈ ~500 GB (well within the 16 TB cold volume). **Note:** the flywheel's `shard_selector.py` already does performance-attributed selection within the fixed precomputed pool — this item is about upstream control of which raw data gets downloaded and precomputed, not which precomputed shards to train on.

4. **QUALITY-10 ablation harness** ✓ Done — see COMPLETED_BACKLOG.md.

**References:** PIPELINE-27 (data curation), PIPELINE-25 (raw pool prerequisite).
**Dependency summary:** Item 1 (resolution/scale) — unblocked, needs 1024px memory profiling run. Item 2 (TRAIN-6) — done, gated. Item 3 (PIPELINE-27) — unblocked on storage, blocked on PIPELINE-25 engineering. Item 4 (QUALITY-10) — done.

---

## Pipeline Improvements

**PIPE-SMOKE-1: Smoke/dev runs invisible to pipeline_doctor and pipeline_status** (Low priority)

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


**PIPELINE-26: Versioned precompute cache — cold storage migration** ✅ done (2026-05-14)

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

**PIPELINE-27: Smart precompute shard selection v2** ✅ done (2026-05-14)

This is the **meta flywheel** upstream layer: controlling which raw JDB tgzs are downloaded and precomputed, as opposed to which already-precomputed shards to train on (the latter is already done by `shard_selector.py`).

- Use eval metrics persisted in `shard_scores.db` (CLIP-I, self/cross-ref gap, style loss per shard) to bias the next chunk download toward high-signal style examples.
- **Hard prerequisite: PIPELINE-25.** Without a persistent raw pool there is no stable candidate set — each chunk's raw data is currently ephemeral.
- **Partial mitigation already in place:** `shard_selector.py` already does performance-attributed selection within the fixed precomputed pool (cond_gap scoring, recency penalty, diversity slots). PIPELINE-27 is the upstream complement.
- **Realistic impact:** currently training on ~320 of ~50,000 JDB shards. Random selection already provides wide style diversity; upstream curation becomes valuable as coverage grows.
- Output: `shard_scores.db` on cold volume + weighted download scheduling in `pipeline_setup.py`.

**PIPELINE-28: Data Intelligence Layer — data_explorer.py** ✅ done (2026-05-14)

As cold storage grows over months of campaigns, observability into the knowledge base becomes critical. Without `data_explorer.py`, the cold volume is a black box: operator has no way to know what's been learned, which weights to warm-start from, or which shards are driving quality. This tool is essential for long-term usability of the platform.

**Subcommands:**

`data_explorer status` — full cold storage overview: disk usage by category, precompute version + coverage summary, raw pool completeness vs. all-in scale, metadata DB sizes and row counts. Entry point for any session.

`data_explorer shards [--top N] [--sort cond_gap|clip_i|style_loss] [--filter ...]` — browse `shard_scores.db`; show per-shard score history and trend across campaigns; highlight shards whose quality is improving vs. plateauing; filter by score range, diversity cluster, campaign, or date added.

`data_explorer weights [--campaign YYYYMMDD]` — browse `weights/flywheel-*/`; show per-campaign summary metrics (CLIP-I, cond_gap, training steps, config hash); annotate with "best ever" markers per metric; list available checkpoint steps within a campaign.

`data_explorer suggest-warmstart --config <yaml>` — key warm-start helper: given a target training config, query weight archive and ablation history to recommend the closest historical checkpoint + precompute version. Emits exact `--warmstart`, `--precompute-version`, and `--warm-start-from` flags ready to paste. Falls back to "train from scratch" with an explanation if nothing suitable exists.

`data_explorer ablation [--campaign ...] [--pareto]` — query `ablation_history.db` across all campaigns; show Pareto-optimal configs (cond_gap vs. ref_gap); compare Pareto frontiers between campaigns to visualise how the search space has improved; emit `--warm-start-from <path>` command for the ablation harness.

`data_explorer compare <campaign-A> <campaign-B>` — side-by-side campaign comparison: CLIP-I trend, cond_gap trend, shard overlap, config diff, step budget used. The primary tool for answering "is campaign N better than campaign N-1?"

`data_explorer maintenance` — read-only audit by default: validates precompute coverage vs. current pool, checks `best/` symlinks are valid, reports orphaned files. With `--prune` flag: GC old precompute versions (respects `keep_versions`). With `--export <subset>`: copies a curated shard subset to a new cold sub-directory. Both mutation subcommands require explicit `--confirm`.

**Implementation:** `train/scripts/data_explorer.py`, standalone CLI (no server). All reads are non-destructive. Mutation subcommands (`--prune`, `--export`) are gated behind `--confirm`. Output format: human-readable tables by default; `--json` flag for scripting.

**PIPELINE-29: Hot→Cold archiving — closing the knowledge accumulation loop** ✅ done (2026-05-14)

The staging direction (cold→hot) is partially wired. The archiving direction (hot→cold) is not. Without it, cold storage never grows and warm-starts can never improve — the knowledge accumulation loop is broken.

**What needs archiving and when:**

| Event | Archive target | Cold destination |
|---|---|---|
| Precompute step completes | Qwen3 / VAE / SigLIP embeddings | `precompute/v{N}/{shard_id}/` |
| Training milestone (e.g. 50K steps) | Checkpoint + EMA + optimizer state | `weights/flywheel-{date}/{step}/` |
| Training campaign ends | Final checkpoint + run summary JSON | `weights/flywheel-{date}/final/` |
| Eval step completes | Per-shard metrics (cond_gap, CLIP-I) | `metadata/shard_scores.db` (append) |
| Ablation run completes | Scored config + metrics | `metadata/ablation_history.db` (append) |

**Archiving semantics:**
- Archiving is always a copy (never a move) while the campaign is still active. The hot copy remains for fast access; the cold copy is the durable record.
- After archiving precompute, the hot cache may be pruned at the operator's discretion (freeing SSD space).
- Weights are never pruned automatically; `data_explorer.py maintenance --prune` handles GC with explicit confirmation.
- `weights/best/` symlinks are updated atomically after each archival: if the new checkpoint improves on the current best for any tracked metric, the symlink is updated.

**Implementation scope:**
- `data_stager.py`: add `archive_precompute()`, `archive_checkpoint()`, `update_best_symlinks()`.
- `orchestrator.py`: trigger archiving after `train.done` milestone and after `precompute.done`.
- `pipeline_lib.py`: `COLD_WEIGHTS_DIR`, `COLD_METADATA_DIR` constants.
- `pipeline_doctor.py`: warn if last archive is stale (>24h since last `train.done` without corresponding archive).

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
