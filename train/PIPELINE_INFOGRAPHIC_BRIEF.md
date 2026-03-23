# IP-Adapter Training Pipeline — Infographic Brief

*Use this document with Claude.ai to generate a visual infographic of the end-to-end training pipeline.*

---

## What to Visualise

An end-to-end machine learning training pipeline that runs on a single Apple M1 Max Mac with a 2 TB external NVMe SSD. The goal is to train an IP-Adapter that gives a Flux image-generation model the ability to match the visual style of a reference photo.

The pipeline trains in **4 sequential chunks**, each adding a new batch of JourneyDB images. Chunk 1 builds the full infrastructure and base model. Chunks 2–4 fine-tune with progressively lower learning rates as more data is added.

---

## Full Dataset (All 4 Chunks)

| Source | Size (download) | Images | Format |
|--------|----------------|--------|--------|
| LAION-Aesthetics | ~150 GB | ~750K | Pre-downloaded WebDataset tars |
| COYO-700M subset | ~80 GB | ~250K | Pre-downloaded WebDataset tars |
| WikiArt | 27 GB | ~80K | HuggingFace → convert once |
| JourneyDB 000–049 | ~800 GB | ~1.05M | HuggingFace → convert (chunk 1) |
| JourneyDB 050–099 | ~800 GB | ~1.05M | HuggingFace → convert (chunk 2) |
| JourneyDB 100–149 | ~800 GB | ~1.05M | HuggingFace → convert (chunk 3) |
| JourneyDB 150–201 | ~832 GB | ~1.09M | HuggingFace → convert (chunk 4) |
| **TOTAL** | **~3.5 TB raw** | **~5.3M images** | |

Each JourneyDB chunk is downloaded (~800 GB), converted to WebDataset (~42 GB WDS), then the 800 GB raw data is deleted before the next step — the 2 TB SSD is never fully consumed at once.

---

## Pipeline Topology

### Phase 1 — Chunk 1: Build Base Dataset (~2–3 days)

```
INPUTS (pre-existing on disk)          AUTO-DOWNLOADED (chunk 1 only)
  LAION-Aesthetics  150 shards           WikiArt (HuggingFace, 27 GB)
  COYO-700M subset   50 shards           JourneyDB 000–049 (HuggingFace, ~800 GB)
         |                                      |
         |         ┌────────────────────────────┘
         |         ↓                       ↓
         |    [2a] WikiArt           [2b] JourneyDB        [3] CLIP Dedup
         |    convert to WDS         convert to WDS         on LAION only
         |    2 min · 80 shards      5 min · 210 shards     2 h · 124K IDs
         |    6 workers parallel     6 workers parallel
         |         |                       |                      |
         └─────────┴───────────────────────┘                      |
                           |                                       |
                           ↓                                       |
                    [4] Build Unified Shards  ←────────────────────┘
                    6 parallel workers · 2–4 h · I/O at ~3 GB/s
                    ~430 shards · 5,000 images each · ~2.1M total images
                           |
                           ↓  (background filter loop runs concurrently)
                    [5] Filter Shards
                    Parallel CPU · 30 min
                    Drop corrupt / too-small / bad-caption records
                    Expected keep rate: ~95%  (~2M usable images)
                           |
                           ↓
                    [7] Anchor Set              [8] Precompute Qwen3 + VAE
                    10K images · 5 min          Single unified pass · ~14 h
                    LAION + WikiArt only         ├─ Qwen3 text embeddings
                    (no JourneyDB style bias)    │   4-bit quantised · ~143 GB
                                                 └─ VAE image latents
                                                     int8 quantised · ~198 GB
                           |
                           ↓
                    [9] CHUNK 1 TRAINING
                    105,000 steps · LR = 1e-4 · ~44 h
                    → checkpoints/step_105000.safetensors
```

### Phase 2 — Chunks 2–4: Incremental Fine-tuning (per chunk, ~2 days each)

Each chunk follows the same pattern: download → convert → build new shards → precompute new shards → train.

```
  ┌─ CHUNK 2 ──────────────────────────────────────────────────────────────┐
  │  Download JDB 050–099 (~800 GB)                                        │
  │  → Convert to WDS (~210 shards · ~1.05M images) → delete raw 800 GB   │
  │  → Build + filter chunk 2 shards                                       │
  │  → Precompute Qwen3 + VAE for new shards (+~149 GB)                    │
  │  → Train 40,000 steps · LR = 3e-5 · ~17 h                             │
  │  → checkpoints/step_145000.safetensors                                 │
  └────────────────────────────────────────────────────────────────────────┘
         ↓
  ┌─ CHUNK 3 ──────────────────────────────────────────────────────────────┐
  │  Download JDB 100–149 (~800 GB) → Convert → Build + filter + precompute│
  │  → Train 40,000 steps · LR = 1e-5 · ~17 h                             │
  │  → checkpoints/step_185000.safetensors                                 │
  └────────────────────────────────────────────────────────────────────────┘
         ↓
  ┌─ CHUNK 4 ──────────────────────────────────────────────────────────────┐
  │  Download JDB 150–201 (~832 GB) → Convert → Build + filter + precompute│
  │  → Train 40,000 steps · LR = 1e-5 · ~17 h                             │
  │  → checkpoints/step_225000.safetensors  ← FINAL MODEL                 │
  └────────────────────────────────────────────────────────────────────────┘
```

### Cumulative Dataset Growth Across Chunks

```
After chunk 1:  ~2.0M images  (LAION + COYO + WikiArt + JDB 000–049)
After chunk 2:  ~3.1M images  (+1.05M JDB 050–099)
After chunk 3:  ~4.1M images  (+1.05M JDB 100–149)
After chunk 4:  ~5.2M images  (+1.09M JDB 150–201)
```

### Learning Rate Schedule (Cosine Decay per Chunk)

```
  1e-4 ─────╮
             ╲
  3e-5 ──────╮╲
              ╲ ╲
  1e-5 ────────╲──╲──────────────────────────
         Ch.1  Ch.2  Ch.3  Ch.4
        105K   40K   40K   40K  steps
```

---

## Hardware Utilisation by Step

| Step | CPU (P-cores) | GPU | I/O | Bottleneck |
|------|--------------|-----|-----|-----------|
| Download | 5% | 0% | ~500 MB/s (network) | Network bandwidth |
| Convert to WDS | 100% (6 cores) | 0% | Medium | CPU (JourneyDB: gzip decompress; WikiArt: parquet read + JPEG passthrough) |
| Build shards | 25% (I/O wait) | 0% | ~3 GB/s (saturated) | External SSD bandwidth |
| Filter shards | 100% (6 cores) | 0% | Medium | CPU (JPEG header) |
| Precompute Qwen3 | 30% | 60–80% | Low | GPU (text encode) |
| Precompute VAE | 20% | 80–95% | Medium | GPU (image encode) |
| Training | 15% | 95–100% | Low | GPU (transformer forward) |

---

## Data Volumes Across Full Training

```
                    CHUNK 1                    CHUNKS 2–4 (each)
                    ───────                    ─────────────────
Raw download:       ~1.1 TB                    ~800 GB (deleted after convert)
WDS sources:        ~490 GB (LAION+COYO+WA+JDB)  ~42 GB (JDB only)
Unified shards:     ~100 GB  (~430 tars)          ~42 GB  (~210 tars added)
Precomputed qwen3:  ~143 GB  (chunk 1 shards)     +~48 GB  per chunk
Precomputed vae:    ~198 GB  (chunk 1 shards)     +~66 GB  per chunk
─────────────────────────────────────────────────────────────────────
Model weights:        16 GB  (Flux Klein 4B + Qwen3 4B, constant)
Checkpoints:         ~20 GB  (periodic saves, all chunks)

Peak disk (chunk 1): ~750 GB  (raw JDB + WDS + shards + precomputed)
Peak disk (ch. 2–4): ~850 GB  (800 GB raw JDB + existing precomputed)
Final state (done):  ~600 GB  (shards + all precomputed + checkpoints)
```

---

## Training Summary

| Metric | Value |
|--------|-------|
| Total unique images | ~5.3M |
| Total JourneyDB images | ~4.24M (all 4 chunks) |
| Total training steps | 225,000 |
| Total GPU training time | ~95 h (~4 days) |
| Total wall-clock (incl. data prep) | ~3 weeks |
| Adapter parameters | 522M (BF16, ~1.0 GB) |
| Hardware | Single M1 Max, 64 GB unified memory |
| Framework | MLX (Apple Silicon Metal — no CUDA) |

---

## Key Technical Optimisations

1. **Parallel step launch** — WikiArt conversion, JourneyDB conversion, and CLIP deduplication all start simultaneously at the top of chunk 1. Both converters use `PERF_CORES − 2 = 6` workers.

1b. **JPEG passthrough in WikiArt conversion** — WikiArt parquet images are stored as original JPEG bytes. `convert_wikiart.py` uses `tj.decode_header()` for the size check and passes raw bytes through unchanged; only non-JPEG images (e.g. PNG) are re-encoded. Avoids generational quality loss and eliminates unnecessary decode/re-encode CPU work.

2. **Concurrent background filter** — `filter_shards.py` runs in a polling loop while `build_shards.py` writes, validating shards as they arrive rather than in a separate sequential pass.

3. **Unified precompute pass** — `precompute_all.py` reads each shard once and writes both Qwen3 embeddings and VAE latents, halving I/O vs sequential scripts.

4. **Pool initialiser pattern** — ML models (Qwen3, VAE, SigLIP) are loaded once per worker process at startup, not reloaded per shard. Saves 310× model reload overhead over the full precompute run.

5. **1-ahead prefetch** — While the GPU encodes batch N, CPU threads decode and preprocess batch N+1 in parallel, hiding JPEG decode latency behind GPU compute.

6. **4-bit / int8 quantised caches** — Precomputed embeddings are quantised before saving: Qwen3 and SigLIP use 4-bit nibble packing, VAE uses int8 per-channel absmax. ~8× smaller than float32 with negligible quality loss.

7. **Atomic shard writes** — All shard writes use `.tar.tmp` → `os.replace()` so a crash never leaves a partial/corrupted shard that would silently corrupt training data.

8. **Retry wrapper** — Precompute and training steps auto-retry up to 3× with 60 s delay so transient errors (Metal timeout, memory spike) recover without human intervention.

9. **Blocklist deduplication** — 124,340 near-duplicate IDs found by CLIP cosine similarity are blocked inline during shard building — no separate rebuild needed.

10. **Chunked training with anchor set** — A fixed 10K-image anchor set (LAION + WikiArt only, no JourneyDB) is mixed in at each chunk to prevent catastrophic forgetting as JourneyDB data is added incrementally.

---

## Suggested Infographic Layout

**Format**: Two-section vertical layout

### Section 1 — Chunk 1: Full Data Pipeline (top ~60% of poster)

Three swim lanes:

**Left lane — Data Sources**:
- Box: LAION 150 shards + COYO 50 shards (pre-existing, no download)
- Box: WikiArt 27 GB → download → convert → 80 WDS shards (parallel)
- Box: JourneyDB 000–049 ~800 GB → download → convert → 210 WDS shards (parallel)
- Arrow: all three feed into Build Unified Shards

**Centre lane — Processing spine**:
1. CLIP Dedup (parallel with conversions, 2h, 124K IDs blocked)
2. Build Unified Shards (6 workers, 3 GB/s I/O, ~430 shards)
3. Filter Shards (background loop, CPU-parallel)
4. Anchor Set (10K images, LAION+WikiArt only)
5. Precompute: Qwen3 + VAE (14h, single unified pass)
6. Chunk 1 Training (105K steps, LR=1e-4, ~44h)

**Right lane — Outputs**:
- `duplicate_ids.txt` (124K IDs)
- `shards/` ~430 tars, ~2M images
- `precomputed/qwen3/` 143 GB
- `precomputed/vae/` 198 GB
- `step_105000.safetensors`

### Section 2 — Chunks 2–4: Incremental Fine-tuning (bottom ~40%)

Horizontal timeline showing 3 repeating blocks (chunks 2, 3, 4), each with:
- Download bar (~800 GB, network-bound)
- Convert + shard + precompute bar (shorter)
- Training bar (40K steps, ~17h each)
- Checkpoint output

Show the cumulative dataset size growing under each chunk:
2.0M → 3.1M → 4.1M → 5.2M images

### Bottom band — Hardware utilisation bar
Colour-coded horizontal strip across the full width showing bottleneck type per step:
- Blue = I/O-bound (build shards, download)
- Orange = CPU-bound (filter, convert)
- Purple = GPU-bound (precompute, training)

### Key numbers to call out prominently (large text)
- **5.3M** total training images
- **4 JourneyDB chunks** (~1M images each)
- **225,000** total training steps
- **~3 weeks** total wall-clock
- **522M** adapter parameters
- **1 Mac** — no cloud, no CUDA
