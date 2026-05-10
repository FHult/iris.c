# FLUX.2 / iris.c — Improvement Backlog

Completed items are archived in [COMPLETED_BACKLOG.md](COMPLETED_BACKLOG.md).

---

## Web UI Features (open)

- [ ] **18. Batch prompt generation** — Submit a list of different prompts to generate in sequence.
- [ ] **20. Per-job timeout** — Prevent hung generations from blocking the queue forever.

---

## Pipeline Improvements

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

### V4 Pipeline Speed & Precompute Optimizations (New)

**PRECOMP-2: Adopt distilled / tiny VAE encoder for precompute only** (HIGH priority)  
- Integrate a lightweight distilled Flux VAE encoder (TAEF1 by madebyollin, FLUX.2 Small Encoder, or custom distilled CNN) **only** for the precompute stage.  
- Keep the full accurate VAE decoder for final inference, validation, and export.  
- Tie into PRECOMP-1 for versioned caching (`vae_variant` in cache key).  
- Expected impact: 40–70% reduction in per-shard precompute time (biggest iteration speed win).  
- Add config flag: `vae_precompute_variant: full | tiny | taef1`

**PRECOMP-3: VAE tiling + intelligent batching** (HIGH priority)  
- Implement tiled mid-block attention for the 64×64 layer in the VAE encoder.  
- Dynamically adjust `--vae-batch` based on available memory (watchdog integration).  
- Expected additional 20–40% speedup on top of distilled encoder.

**PRECOMP-4: Multi-worker precompute with per-process memory caps** (Medium priority)  
- Support multiple parallel precompute workers with `mx.set_memory_limit` per process.  
- Smart shard distribution to avoid unified memory thrashing on 32–64 GB systems.

**PRECOMP-1: Versioned, content-addressable precompute cache** (HIGH priority)  
- Current flat cache is fragile when changing VAE variant, model, or config.  
- Implement hash-based cache keys (config + VAE variant + shard version).  
- Automatic invalidation + migration path from old cache.

**TRAIN-5: Gradient checkpointing + QLoRA foundation** (Medium priority)  
- Prepare for future LoRA training and higher-rank adapters.  
- Enable larger effective batch sizes on current hardware.

~~**PIPELINE-26: End-to-end pipeline profiler**~~ ✅ DONE — `pipeline_profile.py`: per-stage wall-clock from orchestrator JSONL launch events + sentinel mtimes; cross-chunk summary with bottleneck flag; VAE note when precompute is slowest. `pipeline_status.py`: timing footer in human output; `stage_mean_hours` + `bottleneck_stage` in `--ai` JSON.

**QUALITY-10: Automated style feature ablation harness** (Medium priority)  
- Extend `test_quality_features.py` to run matrix tests over combinations of:
  - `cross_ref_prob`, `patch_shuffle_prob`, `freeze_double_stream_scales`, `style_loss_weight`
- Generate comparative HTML reports with loss curves and final recommendation.

**PIPELINE-27: Smart precompute shard selection v2** (Medium-High priority)  
- Build a performance-aware shard selector that uses eval metrics (CLIP-I, self/cross-ref gap, style loss) to dynamically bias the next chunk toward high-value shards.  
- Strong synergy with:
  - Persistent raw-data pool (PIPELINE-25)
  - Versioned precompute cache (PRECOMP-1)
  - Distilled VAE (PRECOMP-2)
- Goal: Evolve from static stratification to a self-improving, curated training set that delivers higher style quality with fewer total samples and lower precompute cost.  
- Output: `shard_scores.json` + weighted sampling logic with configurable performance vs diversity trade-off.

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
