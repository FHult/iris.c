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

**TRAIN-8: Fix held-out validation — BUG-T-003** (High priority — blocks honest checkpoint selection)

Validation is currently completely inert: the held-out directory does not exist so `_compute_val_loss()` always returns `None`, `val_loss.jsonl` is never written, and `_purge_old_checkpoints` falls back to recency-only checkpoint selection. This means we cannot tell whether training is genuinely improving on unseen data or just overfitting to the training shards, and the "best 3 checkpoints" protection in `_purge_old_checkpoints` is silently dead.

**Architecture: permanent cold-level split, not per-session**

The val set must live in cold storage permanently (`cold/validation/held_out/`). Re-picking different shards before each run would make val loss numbers incomparable across campaigns — you couldn't tell whether a lower val loss means the model improved or just that the new val set happened to be easier images. The exam questions must be the same every time.

The cold pool (converted tars) is not modified — no records are deleted. A small reserved tgz range is permanently routed to cold/validation rather than to training shards. Everything else is untouched.

There are three parts, all required.

**Part A: Reserve a val tgz range in config**

In `v2_pipeline.yaml`, shift all `tgz_ranges` to start from tgz `1` instead of `0`. Tgz `0` is permanently reserved for the validation set and excluded from all training scales:

```yaml
jdb:
  validation_tgzs: [0, 0]   # reserved — never included in training ranges
  tgz_ranges:
    dev:
      1: [1, 1]       # was [0, 0]
    small:
      1: [1, 5]       # was [0, 4]
    ...
```

`pipeline_doctor.py` should warn if any configured training tgz range overlaps `validation_tgzs`.

**Part B: `pipeline_ctl.py create-val-set` — one-time cold operation**

New subcommand that runs once, before the first training run, and is idempotent (no-op if sentinel exists):

1. Checks `cold/validation/held_out/.val_set_created` sentinel — exits immediately if present.
2. Confirms `cold/converted/journeydb/` has the val tgz already converted (if not, runs `download_convert.py` for tgz 0 with `--cold-only`).
3. Runs a mini `build_shards.py` on just the val tgz, writing 1–2 shards to `cold/validation/held_out/`.
4. Runs precompute (qwen3 + vae + siglip) on those shards, writing NPZ files to `cold/validation/precomputed/`.
5. Writes the sentinel.

Cold storage layout after this operation:
```
/Volumes/16TBCold/
  validation/
    held_out/
      shard-000000.tar
      shard-000001.tar
      .val_set_created      ← sentinel
    precomputed/
      qwen3/                ← val-set NPZ files (separate from training precompute)
      vae/
      siglip/
```

**Part C: per-session staging in `pipeline_setup.py`**

Before each training run, `pipeline_setup.py` (or the orchestrator pre-training check) stages the val set from cold to hot:

1. Symlinks/copies `cold/validation/held_out/*.tar` to `DATA_ROOT/validation/held_out/`.
2. Symlinks/copies val precomputed NPZ files from `cold/validation/precomputed/{enc}/` into the active cache dirs (`DATA_ROOT/precomputed/{enc}/current/`) alongside the training NPZ files. The val record stems are different from training stems, so there is no collision.

This is identical to the staging pattern for all other data — cold is source of truth, hot is the working copy.

**Part D: load real SigLIP features in `_compute_val_loss()`**

[train_ip_adapter.py:1123-1133](train/train_ip_adapter.py#L1123-L1133) — currently passes `_siglip_zero = mx.zeros((1, 729, siglip_dim))` to the forward pass. This measures base flow-matching reconstruction with a blank image embedding — not IP-adapter conditioning quality. The adapter's cross-attention learns nothing useful from zeros; the val loss produced has no relationship to style-following quality.

Fix: replace the `_siglip_zero` block with the same pattern the training loop uses:

```python
from ip_adapter.dataset import _load_siglip_embed
_siglip_np = _load_siglip_embed(_stem, dcfg.get("siglip_cache_dir"))
if _siglip_np is None:
    continue   # skip records without siglip precompute
_siglip = mx.array(_siglip_np[None], dtype=mx.bfloat16)
```

Then pass `_siglip` and `use_null_image=mx.array(False)` to `loss_fn_with_ip` / `loss_fn`. This is a ~5-line change. With real SigLIP features, val loss measures exactly what we care about: how well the adapter conditions on a real held-out style reference at the current weight state.

**Sizing — constant across all run scales**

The val set size is fixed and does not scale with training run size (smoke, small, large, all-in). The val set is a trend detector — you watch whether val loss goes up, down, or flat, not what its absolute value is. Consistency across campaigns is the whole point: if the val set changes between runs, numbers from different campaigns become incomparable and the early-warning signal is lost.

2 shards (~800 records at typical density) is the right target. The val loop currently caps at 16 records per call ([train_ip_adapter.py:1139](train/train_ip_adapter.py#L1139)), which is too low — 16 records gives a noisy estimate with high step-to-step variance. Increase the cap to 64 as part of this fix: 64 no-grad forwards at ~0.3s each adds ~20s every 1000 steps, which is acceptable overhead and produces a much more stable loss estimate. Beyond 64, diminishing returns.

The external quality metrics (CLIP-I, cond_gap from flywheel eval) are measured on a broader representative set and naturally reflect run scale. Val loss is not a substitute for those — it is the in-training signal that decides whether to keep training or stop early.

Cold storage cost: ~1 GB for shards + ~500 MB for precomputed embeddings. One-time, permanent.

**What correct behaviour looks like after all parts:**

- `pipeline_ctl.py create-val-set`: runs once, idempotent. Logs steps and writes sentinel.
- At training startup: `Validation held-out: 2 shards in .../validation/held_out` (not "not found").
- Every `val_every` steps (default 1000): `val_loss=X.XXXX (step N)` logged; appended to `val_loss.jsonl`.
- `_purge_old_checkpoints`: protects the 3 lowest-val-loss checkpoints in addition to the most recent N. A checkpoint that was peak quality 20,000 steps ago survives.
- Heartbeat `val_loss` field reflects true conditioning quality; `pipeline_doctor` and `pipeline_status` show it accurately.
- Val loss numbers are directly comparable across all future campaigns (same held-out records, same cold source).

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

---

**PIPELINE-31: Ultrahot-tier data prep — use internal NVMe for speed-critical small/experimental runs** (Medium priority)

The internal NVMe SSD (PIPELINE-30's Ultrahot tier) is 2–3× faster than the external hot SSD for random I/O. For disk-IO-bound steps — `download_convert`, `build_shards`, `filter_shards`, and `precompute` — running entirely on the internal NVMe can dramatically shorten the feedback loop for smoke and dev runs.

**Motivation:** today the full data prep pipeline (convert → build_shards → filter_shards → precompute) always uses hot (`/Volumes/2TBSSD`). For a smoke run (100 steps, 1 chunk, 1 tgz) this pipeline is the bottleneck — precompute of ~7,000 records at 145ms/record is ~17 min. On the faster internal NVMe that same I/O runs proportionally faster. Inference/precompute are already on the critical path before every training run; shaving this matters more than training-loop speed for iteration cadence.

**Disk space requirements per scale** (one chunk at a time, steady-state peak):

| Scale | Converted tars | Shards | Precomputed | Peak total | Fits 460 GB NVMe? |
|---|---|---|---|---|---|
| dev (1 tgz) | ~15 GB | ~5 GB | ~10 GB | **~30 GB** | yes (needs ~30 GB free) |
| smoke (1 tgz) | ~15 GB | ~5 GB | ~10 GB | **~30 GB** | yes |
| small (5 tgzs/chunk) | ~75 GB | ~70 GB | ~70 GB | **~215 GB** | borderline |
| medium (13 tgzs/chunk) | ~195 GB | ~180 GB | ~180 GB | **~555 GB** | no (exceeds 460 GB) |

**Current constraint:** internal NVMe has only ~52 GB free (88% used by inference weights and web app assets). To use this path even at smoke scale, ~30 GB must first be cleared. `data_explorer.py diagnose` (DATAMGMT-1) is the right tool to identify what can be moved. The implementation should be forward-looking for when more space is available (or when larger internal NVMe hardware is added).

**Practical target:** `ultrahot` tier is viable for dev/smoke runs and single-chunk small experiments. Medium and above should remain on hot. The system should warn when the selected scale exceeds the tier's available space rather than failing mid-run.

**Config changes** — add `data_prep_tier` key to `storage:` block:
```yaml
storage:
  data_prep_tier: hot            # hot (default) | ultrahot
  # All existing keys unchanged
  hot_root:  /Volumes/2TBSSD
  cold_root: /Volumes/16TBCold
  ultrahot_root: /Users/fredrikhult/ultrahot  # from PIPELINE-30
```

**CLI optionality** — all data prep scripts accept `--data-tier {hot,ultrahot}` flag that overrides the config value. No flag = read from config = default to `hot`. This keeps default behavior completely unchanged.

Scripts affected:
- `download_convert.py`: `--data-tier` routes output to `hot_root` or `ultrahot_root`
- `build_shards.py`: `--data-tier` controls where shard output lands
- `filter_shards.py` / `shard_scorer.py`: reads + writes from the active tier
- `precompute.py` / `cache_manager.py`: version dirs land on the active tier
- `data_stager.py stage`: when `data_prep_tier=ultrahot`, staging reads from cold and writes to Ultrahot root instead of hot root (the existing stage flow, different destination)

**Archiving** — after a successful Ultrahot-tier run, the standard archive flow copies results back to cold. PIPELINE-29's `archive_chunk()` already handles hot→cold; extend it with an `Ultrahot→cold` path that mirrors the same logic. Ultrahot-tier data is treated as ephemeral working area (same as hot); cold is still the source of truth.

**Space guard** — before any Ultrahot-tier prep begins, compute the expected peak disk usage from the pipeline config's tgz range and scale, compare against `shutil.disk_usage(ultrahot_root).free`, and abort with a clear message if space is insufficient (margin configurable, default 20 GB).

**Doctor integration** — when `data_prep_tier=ultrahot`, `pipeline_doctor.py` includes Ultrahot-tier free space in its disk summary and warns if free space is below `staging_margin_gb`.

**When to use Ultrahot tier:**
- Dev / smoke sanity runs (30 min wall-clock or less end-to-end)
- Ablation harness short runs (precompute already done, just need build_shards for new filter config)
- Single-chunk experiments with non-standard data slices
- Precompute re-runs after encoder update (CPU-bound, not I/O-bound — benefit smaller but still positive)

**When NOT to use Ultrahot tier:**
- Multi-chunk production runs at small scale or above
- Runs where Ultrahot tier is actively serving web app traffic (I/O contention)
- When ultrahot_root free space < 50 GB (system-level guard enforces this)

**Interaction with PIPELINE-30:** PIPELINE-30 defines the Ultrahot tier layout for serving weights. This item adds a *separate* working area on the same volume for data prep outputs. The two uses do not conflict — prep outputs live in `ultrahot_root/prep/{chunk}/` during a run and are deleted or archived to cold afterwards, leaving the `ultrahot_root/weights/` and `ultrahot_root/precomputed/` serving paths untouched.

**When to implement:** after PIPELINE-30 (Ultrahot tier established) and after internal NVMe has sufficient free space (use DATAMGMT-1 diagnose to identify reclaimable space first).

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
