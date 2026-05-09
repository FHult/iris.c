# FLUX.2 / iris.c — Improvement Backlog

Completed items are archived in [COMPLETED_BACKLOG.md](COMPLETED_BACKLOG.md).

---

## Web UI Features (open)

- [ ] **18. Batch prompt generation** — Submit a list of different prompts to generate in sequence.
- [ ] **20. Per-job timeout** — Prevent hung generations from blocking the queue forever.

---

## Training Quality Improvements

- ~~**TRAIN-4: Reset timing accumulator on resume to avoid misleading `data=` %**~~ ✅ DONE —
  On resume (`step == start_step > chunk_base_step`): reset `_t_eval_end = None` and
  `t0 = time.time()` so the first interval excludes idle time between crash and restart and
  startup/warmup overhead. Also added 300 s per-step cap on data-wait accumulation to discard
  sleep/wake stale timestamps (caffeinate failure), which was the root cause of the observed
  `data=288971.7s (99%)` at chunk 2 step 57,000.

---

## Pipeline Improvements

- ~~**PIPELINE-3: Never run pipeline jobs while training is active on 2TBSSD**~~ ✅ DONE —
  Replaced the hard block with `taskpolicy -d throttle` + `nice -n 10` wrappers on download,
  build, and filter. These steps now run concurrently with training; the kernel services their
  disk I/O only when the drive queue is otherwise idle so training reads always take priority.
  Precompute and clip_embed remain gated on `gpu_is_free()` (GPU contention is the hard limit).

- **PIPELINE-4: Investigate MLX CLIP for clip_embed step** — `clip_dedup.py` currently uses
  `open_clip` ViT-L-14 via PyTorch MPS (fp16) in `_load_clip()` / `_embed_batch()`. MLX is
  installed in the venv (`mlx==0.31.1`) but `mlx_clip` is not. MLX runs natively on Apple
  Silicon without PyTorch overhead and may offer meaningfully higher throughput for the embedding
  step. Measure actual img/s on a real chunk before implementing; only worthwhile if clip_embed
  is a bottleneck. Implementation: `pip install mlx-clip`, add MLX branch to `_load_clip()` /
  `_embed_batch()`.

- ~~**PIPELINE-23: Cap shard build count to match precompute `max_shards`**~~ ✅ DONE — Added
  `--max-shards` argument to `build_shards.py` (truncates record list after shuffle so we never
  build more shards than `precompute.max_shards` for the active scale). `_start_build()` in
  orchestrator reads `precompute.max_shards` (same config key used by precompute) and passes
  `--max-shards {n}` to the build script, logging the cap decision to the orchestrator JSONL.

- ~~**PIPELINE-24: `pipeline_setup.py` — clean-slate / selective-purge wizard**~~ ✅ DONE —
  Added `_interactive_reset_wizard()` (3 modes: resume / partial / full), `_find_checkpoints()`,
  `_archive_checkpoints()`, and `_purge_pipeline_state()`. Interactive mode prompts the user
  when existing state is found; `--reset {full,partial,resume}` skips the prompt for scripted
  use. `--ai --reset partial/full` emits `{"action":"purge","archived_to":…,"deleted_bytes":…}`
  and executes immediately. Checkpoint archiving is prompted before any destructive reset.
  Stale-state cleanup (logs/*.log, logs/*.jsonl, .heartbeat/*.json) is automated in both reset
  modes. `archive/` is never touched by any reset mode.

- **PIPELINE-25: Persistent raw-data pool — decouple download from chunk staging** — currently
  `download_convert.py` downloads each JDB tgz directly into `staging/chunk{N}/raw/journeydb/`
  and deletes it immediately after conversion. There is no persistent raw pool. Consequences:
  re-running any scale re-downloads all tgzs even if they were downloaded before; scale changes
  cause confusion about which tgzs belong to which chunk.

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
  - `pipeline_setup.py` populates `staging/chunk{N}/raw/journeydb/` with symlinks to the pool
    subset for the selected scale + chunk.
  - `PIPELINE-24` purge logic: `full` reset removes staging but not pool; only an explicit
    `--purge-pool` flag clears the pool.

  **`--pool-dir` override:** allow the raw pool to live on a different volume from `data_root`
  (e.g. spinning disk or NAS) via `download.pool_dir` config key or CLI flag.

  **Architectural note:** separating the immutable raw pool from ephemeral staging is the right
  foundation for two future directions:
  - **Cheap bulk storage**: raw tgzs are read-once, large, and not latency-sensitive. They can
    live on spinning disk or NAS while `staging/`, `shards/`, and `precomputed/` stay on fast
    NVMe. The `--pool-dir` flag makes this a one-line config change.
  - **Containerisation / multinode**: containers mount the raw pool as a read-only volume and
    write only to their own staging scratch. Multiple nodes can share one pool directory
    (NFS/object storage) and each populate independent staging from it without conflict.

  **Implementation scope:**
  - `downloader.py`: `_hf_download_file_guarded()` — check pool first; download to pool, not staging.
  - `download_convert.py`: `run_jdb_download_convert()` — create staging symlinks before producer loop; remove symlink (not pool file) after conversion.
  - `pipeline_setup.py`: add "populate staging symlinks" step; report pool coverage vs. scale requirement.
  - `pipeline_lib.py`: add `RAW_POOL_DIR = DATA_ROOT / "raw" / "journeydb"` constant.

- **PIPELINE-26: Standardise `--ai` mode across all pipeline scripts** — reduce AI agent token
  cost when polling pipeline state. Currently only `pipeline_doctor.py` and `pipeline_setup.py`
  have `--ai`. All scripts that produce human-readable output should emit compact JSON instead
  when `--ai` is passed.

  **Scripts that need `--ai` added:**

  | Script | Current output | `--ai` JSON shape |
  |--------|---------------|-------------------|
  | `pipeline_status.py` | Rich text: chunks, heartbeats, log tails | `{step, loss, eta_sec, chunks_done, active_chunk, issues[]}` |
  | `orchestrator.py` | Logs decisions to file; no stdout polling interface | `{state, active_chunk, last_poll_age_sec, pending_action}` (read-only snapshot; does not start the orchestrator) |
  | `pipeline_ctl.py` | Mix of prose + subprocess output | Each sub-command (`status`, `pause`, `resume`, `abort`, `retry`) emits `{ok, message}` |
  | `validator.py` | Prose pass/fail report | `{passed, issues[], clip_i_mean, weight_ok}` |
  | `validate_shards.py` | Per-shard pass/fail lines | `{total, passed, failed, corrupt_paths[]}` |
  | `validate_weights.py` | Prose weight sanity output | `{passed, issues[]}` |
  | `mine_hard_examples.py` | Progress + top-k stats | `{done, total, pct, top_k_loss_mean}` |
  | `precompute_all.py` | Per-shard progress | `{done, total, pct, eta_sec, errors[]}` |

  **Contract:** `--ai` flag emits valid JSON to stdout only. All progress/prose goes to stderr.
  Top-level `ok` (bool) or `passed` (bool) as the primary signal. On error:
  `{"ok": false, "error": "<message>"}`. No interactive prompts when `--ai` is set.

  **Priority order:** `pipeline_status.py` first, then `validator.py`, then the rest.

- ~~**PIPE-26: Start memory_pressure.log from orchestrator startup**~~ ✅ DONE — Memory watchdog
  daemon thread already starts in `Orchestrator.__init__` (unconditional, not on-demand),
  polling `vm_stat` every 30 s and writing to `LOG_DIR/memory_pressure.log`. Added
  `orchestrator_pid` and `memory_watchdog_log` fields to `run_metadata.json` so the log path
  is always recorded at startup (thread is daemon-owned; no separate PID to track).

---

## V3 — Versioned Precompute Cache

- **PRECOMP-1: Versioned, content-addressable precompute cache** — eliminate silent stale-data
  reuse and avoid redundant recomputation when switching between experiments or scales.

  **Current state:** `precompute_all.py` writes flat `.npz` files into fixed directories
  (`staging/chunk{N}/precomputed/{qwen3,vae,siglip}/`). Cache validity = file exists. No config
  hash, no git SHA, no manifest. Changing image size, SigLIP model, quantisation level, or
  prompt template silently reuses stale `.npz` files with no warning or invalidation.

  **Proposed directory layout:**
  ```
  data_root/precomputed/
    qwen3/
      v_a3f9c2/                    ← short hash of (config subset + git SHA)
        manifest.json
        000000_0000.npz
        …
      current -> v_a3f9c2/         ← symlink updated atomically after full precompute
    vae/
      v_b17d44/
        manifest.json
        …
      current -> v_b17d44/
    siglip/
      v_c9e012/
        manifest.json
        …
      current -> v_c9e012/
  ```

  **`manifest.json` format:**
  ```json
  {
    "version":      "v_a3f9c2",
    "created_at":   "2026-05-07T14:22:00Z",
    "git_sha":      "9f9463a",
    "config_hash":  "a3f9c2d8",
    "encoder":      "qwen3",
    "config": {
      "model":        "Qwen3-4B-Q4",
      "image_size":   512,
      "quant":        "int8",
      "siglip_model": "google/siglip-so400m-patch14-384",
      "layers":       [9, 18, 27]
    },
    "record_count": 412800,
    "shard_count":  80,
    "complete":     true
  }
  ```
  `complete: false` while precompute is running; set to `true` and `current` symlink updated
  atomically only on successful full completion.

  **Version hash derivation:**
  ```python
  import hashlib, json
  def _version_hash(config_subset: dict, git_sha: str) -> str:
      blob = json.dumps(config_subset, sort_keys=True) + git_sha[:8]
      return "v_" + hashlib.sha256(blob.encode()).hexdigest()[:6]
  ```

  **Relevant config fields per encoder:**

  | Encoder | Config fields that affect output |
  |---------|----------------------------------|
  | VAE     | `image_size`, `quant` (int8/fp16), flux model path |
  | Qwen3   | model path/variant, `layers` extraction indices, quant level, chat template |
  | SigLIP  | `siglip_model` name, `image_size`, quant level |

  **New file: `train/scripts/cache_manager.py`:**
  ```python
  class PrecomputeCache:
      def __init__(self, data_root: Path, encoder: str, config_subset: dict, git_sha: str): …
      def version(self) -> str: …
      def cache_dir(self) -> Path: …
      def is_complete(self) -> bool: …
      def record_exists(self, rec_id: str) -> bool: …
      def mark_complete(self, record_count: int, shard_count: int): …
      def all_records(self) -> set[str]: …
      @staticmethod
      def list_versions(data_root: Path, encoder: str) -> list[dict]: …
      @staticmethod
      def clear(data_root: Path, encoder: str, version: str | None = None): …
  ```

  **Changes to existing code:**

  *`precompute_all.py`*: instantiate `PrecomputeCache` per encoder at startup; write to
  `cache.cache_dir()`; call `cache.mark_complete()` after all shards finish; add `--clear-cache
  [encoder]` and `--list-cache` flags.

  *`ip_adapter/dataset.py`*: `make_prefetch_loader()` gains optional `cache_version: str | None
  = None`; resolves to `current` symlink target; falls back to old flat path for backwards compat.

  *`orchestrator.py` `_start_precompute()`*: pass version hash via `--cache-version`; record
  active version hashes in `run_metadata.json` after promotion.

  *`pipeline_setup.py`*: during existing-state detection, scan `precomputed/*/current` symlinks
  and report which versions are present and whether they match current config.

  **Invalidation behaviour:**
  - `cache_dir` exists + `manifest.complete == true` → skip (all records present).
  - `cache_dir` exists + `manifest.complete == false` → resume (partial run).
  - `cache_dir` does not exist → create and start fresh.
  - Old version directories are NOT auto-deleted; survive until `--clear-cache` or PIPELINE-24
    reset. `--clear-stale` removes all non-current versions.

  **Atomic `current` symlink update:**
  ```python
  tmp = cache_dir.parent / ".current_tmp"
  tmp.symlink_to(cache_dir.name)
  tmp.rename(cache_dir.parent / "current")  # atomic on POSIX
  ```

  **Backwards compatibility:** if `data_root/precomputed/qwen3/` contains `.npz` files directly
  (old flat layout), treat as unversioned legacy cache. Dataset loader uses it as fallback; a
  one-time `--migrate-cache` migration moves flat files into `v_legacy/`.

  **Implementation priority:** VAE first (most expensive, ~5 GB/chunk, no model variation).
  Qwen3 second (prompt template changes most often). SigLIP third.

---

## V3 — Style/Content Separation

These three changes work together to teach the adapter to extract style independently of content.
The core problem: training with reference=target lets the model use SigLIP content features as a
reconstruction shortcut rather than learning style as an independent signal.

- ~~**QUALITY-1: Cross-image reference permutation**~~ ✅ DONE — In 50% of training batches
  (`cross_ref_prob: 0.5` in config), `siglip_feats` is permuted along the batch dimension before
  `compiled_step`. Forces style/content separation; tracked via `loss_self_ref` / `loss_cross_ref`
  split in logs and heartbeat.

- ~~**QUALITY-2: Freeze double-stream IP scales to zero**~~ ✅ DONE — `freeze_double_stream_scales:
  true` in adapter config section zeros `adapter.scale[:nd]` after checkpoint load and zeroes
  the corresponding gradient slice in `compiled_step` so the optimizer never updates them.
  Blocks 0–4 (double-stream) stay silent; style is learnt via single-stream blocks only.

- ~~**QUALITY-3: Patch-shuffle augmentation on reference**~~ ✅ DONE — After `siglip_feats`
  resolved, with probability `patch_shuffle_prob: 0.5`, the 729-token SigLIP sequence is randomly
  permuted (`siglip_feats[:, perm, :]`). Destroys spatial layout while preserving per-patch
  texture statistics.

---

## V3 — Training Observability

- ~~**QUALITY-6: Cross-reference loss tracking**~~ ✅ DONE — `_self_ref_loss_sum/count` and
  `_cross_ref_loss_sum/count` accumulators added. At each log interval, prints
  `loss_ref: self=X.XXXX [n=N]  cross=X.XXXX [n=N]  gap=+X.XXXX` and warns if cross < self.
  Both fields included in the heartbeat JSON.

- ~~**QUALITY-8: Validate and tune `style_loss_weight`**~~ ✅ DONE — Set `style_loss_weight: 0.05`
  in `stage1_512px.yaml`. Monitor `style_loss` and `grad_norm` in the first 500 steps; reduce to
  0.01 if grad spikes appear. Style loss is already skipped on null-image steps (correct).

- ~~**QUALITY-9: Quality tracking script**~~ ✅ DONE — `train/scripts/quality_tracker.py`: reads
  `<ckpt_dir>/eval/step_*/eval_results.json`, `<ckpt_dir>/val_loss.jsonl`, and
  `DATA_ROOT/.heartbeat/trainer_chunk*.json`. Produces self-contained HTML with canvas charts
  (loss, CLIP-I/T, ip_scale). `--ai` emits compact JSON with `summary`, `top_action`,
  `data_points`. Pure stdlib; no numpy/matplotlib required.

---

## C Binary / CLI

- **B-001: --vary-from / --vary-strength CLI wiring** (~1 hour) — `main.c`, `iris.h`
- **B-002: Z-Image CFG infrastructure** (~1 day) — `iris_sample.c`, `iris.c`, `iris.h` — unblocks Z-Image-Omni-Base; do this before B-003
- **B-003: Negative prompt for distilled Flux** (~2 hours) — `iris.c`, `main.c` — prerequisite for Web UI Feature 1

---

## Web UI Features (advanced)

- **Prerequisite: extract fetchImageAsBase64()** — duplicated across 4 files; extract into shared util before touching any feature below
- **Feature 3: Enhanced Vary-from-History** (~2–3h) — fastest win, no C backend changes needed
- **Feature 2: Per-Slot Reference Strength + Style Reference Mode** (~3h UI / ~8h full C with backend)
- **Feature 4: Outpaint UI** (~5–7h)
- **Feature 1: Negative Prompt** (~3–4h server+UI + 4h C backend) — blocked on B-003

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

## Pipeline Scripts (Unimplemented)

- **pipeline_benchmark.sh** — Parse training log for steps/hour, ETA, timing breakdown. Quick win.
- **pipeline_validate.sh** — Generate N sample images from current checkpoint to spot-check quality
- **pipeline_export.sh** — Package adapter for deployment / int4 quantise for inference
- **pipeline_recaption.sh** — Re-caption short captions across dataset (parallelised, ~2 days GPU time)
