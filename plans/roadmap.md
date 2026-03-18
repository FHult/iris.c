# iris.c Development Roadmap

**Last updated:** 2026-03-18

This document captures the proposed development trajectory across all active workstreams.
For the IP-Adapter training plan in full detail see [ip-adapter-training.md](ip-adapter-training.md).

---

## Current State (v2.5.4)

**What works today:**
- txt2img: Flux Klein 4B/9B (distilled + base), Z-Image-Turbo
- img2img / multi-reference in-context conditioning (Flux only, up to 4 refs)
- Hi-res fix (two-pass generation)
- LoRA weight loading
- CFG for Flux base models
- Embedding cache (4-bit quantized, 16-slot LRU)
- Interactive CLI (REPL) and web server mode
- Style presets, variation seeds
- MPS (Metal), BLAS, and generic CPU backends

**Known gaps vs Midjourney (prioritised):**
1. Style reference (`--sref`) — no style-isolated conditioning
2. Negative prompt — only wired for base models with CFG
3. Vary subtle/strong — controlled re-denoising from existing latent
4. Inpainting — mask + regenerate region
5. Outpainting — canvas extension
6. Z-Image base model (CFG) — Z-Image-Omni-Base not yet supported
7. Upscaling (subtle/creative passes)

---

## Two Parallel Tracks

The roadmap runs two tracks concurrently. Track A delivers user-visible features
immediately. Track B builds the training infrastructure for the full style reference
solution. Both start at the same time.

```
Track A: Shippable features in iris.c (C development)
Track B: IP-Adapter training pipeline (data + Python/MLX)
```

---

## Track A — Shippable Features

### A1: Path 1 Style Reference — Untwisting RoPE
**Target release:** v2.6
**Effort:** ~2–3 weeks active C development
**Dependencies:** none

Implements training-free style reference based on arXiv 2602.05013. Runs a
"style reference forward pass" through the Flux Klein single-stream blocks,
caches K/V tensors, then injects them into the generation pass with attenuated
high-frequency RoPE components. No new weights, no training required.

Quality is moderate — better than content img2img, not as clean as IP-Adapter.
Labelled as approximate style reference in CLI/docs; will be superseded by A4.

**New CLI flags:**
```
--sref PATH         Style reference image (one or more)
--sref-scale FLOAT  Style influence 0.0–1.0 (default 0.7)
```

**Files affected:**
- `iris_transformer_flux.c` — per-single-block K/V caching + RoPE attenuation
- `iris_sample.c` — style forward pass before denoising loop
- `iris.c` — new `iris_style_ref()` entry point
- `main.c` — `--sref`, `--sref-scale` argument parsing
- `iris.h` — API surface

**Implementation notes:**
- Cache K/V for all 20 single-stream blocks: `20 × seq × heads × head_dim × 2 × 4 bytes`
  → ~500MB at 512px (allocated once, reused across denoising steps)
- RoPE frequency attenuation: per-frequency scale vector applied to reference K before
  attention; damps high-frequency positional components, allows semantic/style to dominate
- Style mode = inject only single-stream blocks (5 double-stream blocks untouched)
- GPU (Metal BF16) path needs corresponding changes in `iris_metal.m`

---

### A2: Z-Image-Omni-Base CFG Support
**Target release:** v2.6 (same release as A1)
**Effort:** ~1 week active C development
**Dependencies:** none

Z-Image-Omni-Base uses the same S3-DiT architecture as Z-Image-Turbo but adds
CFG (guidance_scale 3–7, run transformer twice per step). The CFG denoising loop
already exists for Flux base models — adapt it for Z-Image.

Model weights: `Tongyi-MAI/Z-Image-Omni-Base` on HuggingFace.

**Files affected:**
- `iris_sample.c` — CFG denoising loop variant for Z-Image
- `iris_transformer_zimage.c` — verify forward pass accepts empty prompt path
- `iris.c` — route Z-Image-Omni-Base through CFG sampler
- `download_model.sh` — add Z-Image-Omni-Base download entry

**CLI usage (new model name):**
```bash
./iris -d zimage-omni-base -p "a fish" --cfg 5.0 -o /tmp/out.png
```

---

### A3: Negative Prompt (all models)
**Target release:** v2.6
**Effort:** ~2 days
**Dependencies:** none

Negative prompt already works for Flux base models (CFG path). Extend to:
- Z-Image-Omni-Base (CFG path being added in A2)
- Distilled models: wire as soft guidance via the existing `--cfg` override path

**New CLI flag:**
```
--negative "ugly, blurry, watermark"
```

---

### A4: IP-Adapter Full Style Reference
**Target release:** v2.7
**Effort:** 2–4 weeks active C development
**Dependencies:** Track B training complete; adapter weights available

Replaces/supersedes A1. Proper IP-Adapter conditioning via SigLIP vision encoder +
Perceiver Resampler + per-block decoupled cross-attention. Clean style isolation via
single-block injection mode.

**New files:**
- `iris_siglip.c` — SigLIP SO400M vision encoder (~1500 lines C)
- `iris_ipadapter.c` — Perceiver Resampler + weight loading (~500 lines C)

**Modified files:**
- `iris_transformer_flux.c` — K/V injection in `double_block_forward` and
  `single_block_forward`
- `iris.c` — `iris_ipadapter_forward()` entry point
- `main.c` — `--sref-mode style|content`, `--sref-model PATH`

---

### A5: Vary Subtle / Vary Strong
**Target release:** v2.7 or later
**Effort:** ~1 week
**Dependencies:** none

Re-denoise an existing generated image at a low noise level. "Subtle" = 15–25%
noise injection before re-denoising (tight variation). "Strong" = 50–65% noise.
Reuses the existing noise-injection img2img path with preset strength values.

**New CLI flags:**
```
--vary-from PATH       Base image to vary from
--vary-strength FLOAT  0.0–1.0 (presets: subtle=0.2, strong=0.6)
```

---

## Track B — IP-Adapter Training Pipeline

Full details in [ip-adapter-training.md](ip-adapter-training.md).
Summary of phases and timing:

### B1: Dataset Download
**Duration:** ~1 week (unattended, start immediately)
**Storage required:** ~400GB free

Run all three dataset downloads in parallel on day 1:
```bash
# Terminal 1: LAION-Aesthetics-v2 (1.2M images, ~150GB)
img2dataset --url_list laion_filtered.parquet ...

# Terminal 2: JourneyDB (500K images, ~80GB) — non-commercial research
huggingface-cli download JourneyDB/JourneyDB ...

# Terminal 3: COYO filtered (200K images, ~25GB)
img2dataset --url_list coyo_200k.parquet ...
```

WikiArt (100K, 1.7GB) downloads in minutes; can be done any time.

### B2: Data Pre-processing
**Duration:** ~2 days (mostly unattended, overlaps with A1/A2 dev)

1. Pre-filter LAION parquet (30 min, run before B1 starts)
2. CLIP-based deduplication via `clip-retrieval` + FAISS (~1.5 hours)
3. Moondream re-captioning for short captions, 2× parallel (~2 days, GPU-bound bottleneck)
4. Parallel shard merge into WebDataset at 5000 images/shard (turbojpeg + multiprocessing)

Output: ~1.55M unique images, 310 shards, style-quality captions.

### B2b: Pre-compute Frozen Encoders
**Duration:** ~1–2 days (unattended, overlaps with B3 active dev)
**Starts:** As soon as B2 sharding completes

Pre-compute and 4-bit/int8 quantise the three frozen encoder forward passes to eliminate
them from the training step hot path (saves ~14 hours of training wall-clock):

| Encoder | Storage | Wall-clock | Step saving |
|---|---|---|---|
| Qwen3 text embeds (4-bit) | ~143 GB | ~8–10 hours | ~200ms/step |
| VAE latents (int8) | ~198 GB | ~12–16 hours | ~180ms/step |
| SigLIP features (4-bit) | ~420 GB | ~4–6 hours | ~50ms/step |

**Recommended: pre-compute Qwen3 + VAE (341 GB total). Skip SigLIP unless >420 GB
remains after raw dataset storage.** SigLIP saves only ~1 hour for 420 GB.

### B3: MLX Training Code
**Duration:** 1–2 weeks active Python development
**Starts:** Week 3 (after data pipeline is complete)

- Fork mflux; subclass `Flux2Transformer` for IP injection
- Implement `IPAdapterKlein` in MLX (Perceiver Resampler + batched K/V)
- Wire `mx.compile()`, `mx.checkpoint()`, async prefetch, BF16 optimizer state
- Warmstart Perceiver Resampler from InstantX Flux.1-dev weights
- Validate on a small 10K-image smoke-test dataset before full run

**Language: Python + MLX (confirmed optimal — do not rewrite in C++ or Rust)**

After a full performance audit: after `mx.compile()` the Python overhead is ~5ms per step
against a ~1.9s Metal-bound step (<0.3%). The MLX C++ API exists and supports the same
autograd, but saves negligible time. Candle (Rust) lacks hand-tuned GEMM kernels and runs
slower. burn (Rust) uses wgpu-Metal abstraction with no MPSGraph path.
Custom Metal kernels where needed are implemented via `mx.fast.metal_kernel()` — callable
from Python with full autograd support. No rewrite needed.

### B4: Stage 1 Training (512px)
**Duration:** ~1.84 days (unattended) — scenario B (Qwen3 + VAE pre-computed)
**Starts:** Week 5

```
105,000 steps × ~1.52s/step ≈ 159,600 seconds ≈ 44.3 hours
(steps reduced ~15K by Perceiver warmstart vs raw 120K)
```

Without pre-compute (scenario C): 105,000 × ~1.9s ≈ **2.31 days**
With all three encoders pre-computed (scenario A): 105,000 × ~1.47s ≈ **1.79 days**

Monitor loss curve. Eval checkpoints at 2K-step intervals with 10 fixed
(prompt, style-ref) pairs. Expected first useful output at ~step 50K.

### B5: Stage 2 Training (768px)
**Duration:** ~1.05 days (unattended) — scenario B
**Starts:** After B4 quality is verified

```
20,000 steps × ~4.55s/step ≈ 91,000 seconds ≈ 25.3 hours
```

Without pre-compute (scenario C): 20,000 × ~5.0s ≈ **1.16 days**

### B6: Export and C Integration
Covered under A4 above. Runs in parallel with B4/B5.

---

## Parallel Execution Timeline

```
         Week 1     Week 2     Week 3     Week 4     Week 5     Week 6     Week 7-10
Track A  [A1 dev ──────────────][A2+A3 dev][──────────── A4 (iris_siglip + ipadapter) ──────────]
Track B  [B1 download ─────────][B2 prep ──][B2b pre-compute ─][B3 MLX ──][B4 ~1.8d][B5 ~1d][B6]

Releases:                        v2.6                                      v2.7
```

B2b (pre-compute encoders) overlaps with B3 active dev — runs unattended overnight
while training code is being written. B4+B5 total unattended training: **~2.89 days**
(down from ~3.47 days; saves ~14 hours, scenario B).

**v2.6** (end of week 3):
- Path 1 style reference (`--sref`, training-free)
- Z-Image-Omni-Base with CFG
- Negative prompt wired for all models

**v2.7** (week 9–10):
- Full IP-Adapter `--sref` (replaces Path 1)
- `iris_siglip.c` — SigLIP encoder in C
- `iris_ipadapter.c` — Perceiver Resampler + per-block injection

---

## Prerequisites / Decisions Before Starting

| Item | Decision needed | Owner |
|---|---|---|
| Storage — internal | Confirm whether internal SSD has 300GB+ free for active training shards | Before B1 |
| Storage — external | If using TB4 external SSD: use `caffeinate`, 2K-step checkpoints, quality cable | Before B4 |
| JourneyDB license | Non-commercial research only — confirm this use is acceptable | Before B1 |
| InstantX weights | Download 5.3GB Flux.1-dev IP-Adapter for Perceiver warmstart | Before B3 |
| v2.6 release messaging | Label Path 1 as "approximate style reference, superseded in v2.7" | Before A1 ships |
| IP-Adapter weight distribution | Decide whether trained adapter weights will be published | Before B4 |

### Storage layout recommendation

| Data | Location | Size |
|---|---|---|
| Active training shards | Internal SSD (preferred) or TB4 external | ~260 GB |
| Raw source datasets | TB4 external SSD | ~340 GB |
| Model weights | TB4 external SSD | ~20 GB |
| Checkpoints (active) | Either | ~12 GB |
| **Total external** | | **~370 GB** |
| **Total internal (if hosting shards)** | | **~275 GB** |

**Thunderbolt 4 I/O performance:** not a bottleneck. The training loop requires ~105 KB/s
sustained read at batch_size=2 — less than 0.01% of TB4's 2.5 GB/s bandwidth. The GPU
is the bottleneck at all times.

**Thunderbolt 4 reliability:** the only real concern. A connection dropout during a 2–3 day
unattended training run crashes the process. Always run training under `caffeinate -i -d`
and use 2K-step checkpoint intervals. Full mitigation details in
[ip-adapter-training.md — Phase 0](ip-adapter-training.md).

---

## Deferred / Later Consideration

Items not in the current roadmap but identified as valuable:

| Feature | Complexity | Notes |
|---|---|---|
| Inpainting (Vary Region) | High | Mask input + partial VAE encode; significant new code path |
| Outpainting (Zoom Out / Pan) | High | Canvas extension + latent padding; extends inpainting |
| Upscaling (subtle) | Medium | Super-resolution pass without hallucination; different from hi-res fix |
| Character/object reference (`--cref`) | High | Requires FaceID-style conditioning or fine-tuned IP-Adapter variant |
| Video generation | Very high | Entirely different architecture; out of scope |
| Personalization | Very high | Requires rating UI + fine-tuning loop; infrastructure problem |
