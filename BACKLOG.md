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

~~**PRECOMP-1: Versioned, content-addressable precompute cache**~~ ✅ DONE
- Implemented hash-based versioned dirs (`PRECOMP_DIR/{encoder}/v_{hash}/`) and atomic `current` symlink.
- `train/scripts/cache_manager.py` (new): `PrecomputeCache`, `encoder_config_subset()`, `get_git_sha()`, `version_hash()`, plus `list_versions()`, `clear()`, `migrate_legacy()` statics.
- `train/scripts/cache_inspect.py` (new): standalone diagnostic CLI — list, clear-stale, clear-version, migrate-legacy.
- `precompute_all.py`: writes incomplete/complete manifests in staging output dirs (best-effort); `--list-cache` and `--clear-stale` standalone ops.
- `orchestrator.py` `_promote_chunk()`: moves files to `PRECOMP_DIR/{encoder}/v_{hash}/` and updates `current` symlink atomically; training/mining paths now use `PrecomputeCache.effective_dir()` with fallback to flat layout.
- `train_ip_adapter.py`: resolves versioned cache dirs at startup via `PrecomputeCache.effective_dir()` when `--data-root` is given.
- `pipeline_setup.py`: reports versioned cache state (version, record count, staleness) in existing-state display and `--ai` JSON.
- Backwards compatible: existing flat `.npz` dirs detected as legacy; `--migrate-legacy` moves them to `v_legacy/` with a `current` symlink.

- **PRECOMP-1 (original spec, archived):** eliminate silent stale-data
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

**PRECOMP-2: Adopt distilled / tiny VAE encoder for precompute only** — ❌ NOT VIABLE  
- Investigated (2026-05-10): TAEF1 produces approximate latents in the same 32-channel space
  but with ~10% SSIM error. Using TAEF1 for precompute creates a train/inference distribution
  mismatch: the diffusion model is pretrained on full-VAE latents, and img2img at inference uses
  the full VAE encoder. Training on TAEF1 latents degrades quality without a clear recovery path.
  `diffusers` not installed; TAEF1 weights not available. The correct path is PRECOMP-3 (optimal
  batching of the full VAE encoder, which is already using Flash Attention internally).

~~**PRECOMP-3: VAE tiling + intelligent batching**~~ ✅ DONE  
- Profiled VAE encode on M1 Max at 512px: B=4 is optimal (145.7 ms/img); B=16 (prior runtime
  default) is 20% slower; B=32 (prior code default) is 74% slower; B=64 OOMs.
- Changed `--vae-batch` default from 32 → 4 in `precompute_all.py`.
- Added `precompute.vae_batch: 4` to `v2_pipeline.yaml`; orchestrator passes it as `--vae-batch`.
- Tiled mid-block attention: already implemented — mflux uses `mx.fast.scaled_dot_product_attention`
  (Flash Attention) and ships `VAETiler`; at 512px the full image is one tile, no-op needed.

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

**TRAIN-6: Retrain IP-adapter with block-by-block injection** (Medium priority, next major release)  
- Current training uses `_flux_forward_no_ip` + end-sum approximation: all IP contributions are summed and added to `h_final` after all 25 blocks. Q vectors are collected from a clean (no-IP) Flux forward, so earlier blocks cannot adapt their computation to the style signal. This limits quality; CLIP-I ~0.56 vs ~0.7–0.85 for canonical IP-Adapter.  
- Replace with `_flux_forward_with_ip` as the actual training forward pass (block-by-block injection matching inference). Each block's Q is computed from IP-conditioned hidden states, matching the canonical IP-Adapter approach.  
- **Warm-start**: current checkpoint (`best.safetensors`, step 95000) gives a good init for perceiver and ip_scale; `to_k_ip_stacked`/`to_v_ip_stacked` will re-learn at the correct injection points.  
- **Memory cost**: gradient flows through IP injection at each of 25 blocks (no longer fully isolated from Flux), significantly increasing peak memory vs the current isolated approach. Requires profiling on 64 GB unified memory before committing.

~~**PIPELINE-26: End-to-end pipeline profiler**~~ ✅ DONE — `pipeline_profile.py`: per-stage wall-clock from orchestrator JSONL launch events + sentinel mtimes; cross-chunk summary with bottleneck flag; VAE note when precompute is slowest. `pipeline_status.py`: timing footer in human output; `stage_mean_hours` + `bottleneck_stage` in `--ai` JSON.

**QUALITY-10: Automated style feature ablation harness** (Medium priority)  
- Extend `test_quality_features.py` to run matrix tests over combinations of:
  - `cross_ref_prob`, `patch_shuffle_prob`, `freeze_double_stream_scales`, `style_loss_weight`
- Generate comparative HTML reports with loss curves and final recommendation.

**PIPELINE-27: Smart precompute shard selection v2** (Low-Medium priority) ⛔ blocked on PIPELINE-25
- Build a performance-aware shard selector that uses eval metrics (CLIP-I, self/cross-ref gap, style loss) to dynamically bias the next chunk toward high-value shards.
- **Hard prerequisite: PIPELINE-25 (persistent raw-data pool).** Without a persistent pool there is no candidate set to select from — each chunk's raw data is ephemeral and selection is impossible.
- Synergy with:
  - Versioned precompute cache (PRECOMP-1) ✅ done — makes selective re-precomputing safe
  - Persistent raw-data pool (PIPELINE-25) ⛔ not done — hard blocker
- ~~Distilled VAE (PRECOMP-2)~~ — dependency removed. PRECOMP-2 was not viable; without cheap precompute the "score candidates before committing" approach costs as much as precomputing everything. Selectivity benefit only comes from quality, not cost reduction.
- **Revised goal:** deliver higher style quality with fewer chunks by biasing shard selection toward underrepresented styles and away from shards where the current model already performs well. "Lower precompute cost" is no longer part of the claim.
- Note: `mine_hard_examples.py` already provides adaptive selection at the record level post-chunk; shard-level scoring is an upstream complement, not a replacement. Marginal gain is modest while training on ~320 of ~50,000 JDB shards since random selection already provides wide diversity.
- Output: `shard_scores.json` + weighted sampling logic with configurable quality vs diversity trade-off.

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
