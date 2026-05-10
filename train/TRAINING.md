# IP-Adapter Training — Operational Notes

Empirical findings and strategic analysis for the precompute + training pipeline on M1 Max.

---

## Dataset vs Training Coverage

### Actual numbers (chunk 1, large scale default)

| Metric | Value |
|---|---|
| Chunk 1 shards (full pool) | 432 |
| Chunk 1 images | ~2.15M (avg 4,970/shard) |
| Stage 1 training samples | 200K steps × batch=2 = **400K** |
| Shards precomputed (`precompute.max_shards: 80`) | **~80** (of 432) |
| Image coverage | **~19%** |

Early planning documents claimed "78 visits per image over 120K steps" — that was
written for a small pilot (~3K images). At the 2M-image scale, most images are
never seen even at 200K steps.

---

## Quality Analysis: Diversity vs Repetition

### What the IP-Adapter is actually learning

The Perceiver Resampler and 25 K/V projection pairs learn a mapping from SigLIP
image features (729 tokens, 1152-dim) to style conditioning (128 IP tokens). This
is a smooth function over the SigLIP feature manifold — similar reference images
cluster together in feature space because SigLIP was trained on billions of diverse
images.

### Diversity dominates over repetition for this task

An IP-Adapter that has seen 1,000 distinct artistic styles generalizes to nearby
styles it never saw. An IP-Adapter that has seen 42 shards 3 times does not
generalize to styles absent from those 42 shards.

Reference implementations confirm this: InstantX's FLUX.1-dev IP-Adapter and all
SD-era adapters were trained on millions of diverse images with low repetition. The
diversity is what enables handling reference images never seen during training.

**Conclusion: more unique images seen once > fewer images seen many times.**

### Is 400K samples enough? Acceptable but not reference-class.

Reference IP-Adapters used 1–10M+ training samples. At 400K:
- The model learns reliable style transfer across common artistic styles
- Unusual or underrepresented styles may show weaker fidelity
- Good balance between quality and training time on M1 Max

Quality vs time tradeoff on M1 Max (1.5s/step, sizing formula below):

```
target_unique_samples × (1.5–2.5 passes) = total_training_samples
total_training_samples / batch_size = num_steps
```

| Steps | Wall clock | Shards needed | Unique images | Quality |
|---|---|---|---|---|
| 105K | 44h | 42 | 70K | Marginal |
| **200K** *(current default)* | **83h** | **80** | **133K** | **Acceptable** |
| 400K | 7 days | 160 | 267K | Good |
| 540K | 9.4 days | ALL (432) | 2.15M (0.5×) | Reference-class |

---

## Long-term Precompute Strategy

### Core principle

Do not precompute the full downloaded dataset. Only precompute the shards that
training will actually consume. For each chunk, randomly select the needed shards
from the full downloaded pool, stratified by source for diversity.

### Recommended approach

1. After each dataset download, randomly select shards proportionally by source
   (LAION/COYO/WikiArt/JourneyDB) to cover that chunk's training budget.
2. Precompute only those selected shards (Qwen3 + VAE).
3. Train on the selected shards.
4. Delete raw images and precomputed caches when done. Keep only anchor shards.

**Shard selection sizing:**

```
shards_needed = ceil((num_steps × batch_size) / avg_records_per_shard / 1.5_passes)
             ≈ ceil(num_steps × 2 / (4970 × 1.5))
```

### Revised chunk plan (recommended)

| Preset | Chunk 1 steps | Chunks 2-4 steps | Chunk 1 shards | Chunks 2-4 shards | Total wall clock |
|---|---|---|---|---|---|
| `small` | 50K | 15K | 21 | 7 | ~3 days |
| `medium` | 105K | 40K | 43 | 17 | ~7 days |
| **`large`** *(default)* | **200K** | **60K** | **80** | **25** | **~9 days** |
| `god-like` | 400K | 120K | 162 | 50 | ~18 days |
| `all-in` | 540K | 200K | ALL | ALL | ~26 days |

`all-in` precomputes the entire downloaded pool — no shard limit. Steps are sized
for ~0.5 pass through 432 shards at batch=2. Actual coverage depends on total
shards available when the pipeline runs.

Default: **`large`** — 200K chunk-1 steps, 80 stratified shards, ~2 weeks end-to-end.

Total precompute: **~2.5 days** across all chunks vs **~28 days** (full precompute
of all downloads across all chunks).

### Shard selection must be stratified

Do not use uniform random selection — it risks heavy representation of whichever
source has the most shards. Instead, select proportionally:

| Source | Shard count (chunk 1) | Select for 80-shard run |
|---|---|---|
| LAION-Aesthetics | ~150 | ~28 |
| JourneyDB | ~140 | ~26 |
| WikiArt | ~92 | ~17 |
| COYO | ~50 | ~9 |

For chunks 2-4, add new JDB shards proportionally while keeping a fixed selection
of ~60 shards from earlier chunks (continuity). The anchor replay set (10K images,
mixed at 20%) handles distribution shift.

### Starting training before precompute completes

Training can start as soon as ~80 shards are precomputed. The dataset loader
cycles through available shards in epochs. Do not add new shards mid-run; restart
from a checkpoint with the expanded shard list if needed.

---

## Precompute Performance (M1 Max)

### Throughput measurements

| Phase | Rate | Notes |
|---|---|---|
| Qwen3 encoding | ~0.044 s/record (avg) | Scales with caption length |
| VAE encoding | ~0.28 s/image | Dominates — 7.7× slower than Qwen3 |
| Total per shard | ~23 min (4,970 records) | Qwen3 ~4 min, VAE ~19 min |
| Full 432 shards | ~166h (~7 days) | Single-worker |

### VAE is the bottleneck

The Flux2 VAE encoder has a mid-block attention layer at 64×64 spatial resolution
(4,096 tokens) for 512×512 input images. This uses naive (non-Flash) attention —
the full 4096×4096 matrix is materialised per batch. Reducing `--vae-batch` does
not improve per-image throughput; the compute is the fundamental limit.

### Qwen3 batch timing by sequence length

First batch per shard: ~7s MPSGraph compilation overhead (one-time per worker).

| Avg seq len | Time/batch (8 records) |
|---|---|
| ~13 tokens | ~0.18s |
| ~35 tokens | ~0.36s |
| ~92 tokens | ~0.79s |

---

## VAE-Only Loading

The worker loads only Flux2 VAE weights, not the full 4B transformer + Qwen3
encoder (~15.5 GB saved). Implementation: `_load_flux_vae_only()` in
`train/scripts/precompute_all.py` using a duck-typed `_VaeOnly` weight definition.

---

## Wall Clock Summary

Large-scale default (200K chunk-1 steps, 80 shards), M1 Max throughput:

| Phase | Time |
|---|---|
| Chunk 1 precompute | ~31h (80 shards × 23 min) |
| Chunk 1 training | ~83h (200K steps × 1.5 s) |
| Chunks 2-4 precompute | ~3× 10h (25 shards each) |
| Chunks 2-4 training | ~3× 25h (60K steps each) |
| Per-chunk pipeline overhead (download, clip, build, validate) | ~3× 15h |
| **Total** | **~219h (~9 days training + precompute)** |

Full end-to-end wall clock including data infrastructure: **~2 weeks.**
Training steps can run overnight; the machine is not blocked during precompute.

---

## Style Separation Features (QUALITY-1/2/3/6/8)

These features are active by default in `stage1_512px.yaml` and require SigLIP features
(`training.siglip: true`). They are silently skipped when `siglip: false`.

| Feature | Config key | Default | Effect |
|---|---|---|---|
| Cross-ref permutation | `training.cross_ref_prob` | 0.5 | 50% of conditioned steps: permute batch-dim of SigLIP feats so reference image doesn't match caption. Forces style/content separation. |
| Patch shuffle | `training.patch_shuffle_prob` | 0.5 | 50% of conditioned steps: shuffle the 729-token SigLIP sequence. Destroys spatial layout, preserves texture statistics. |
| Freeze double-stream scales | `adapter.freeze_double_stream_scales` | true | Zeros adapter scale for double-stream blocks (0–4) at init and zeroes their gradients during training. Single-stream blocks (5–24) learn normally. |
| Style loss | `training.style_loss_weight` | 0.05 | Gram-matrix style loss weight. 0.0 disables. |
| Ref loss tracking | always on | — | Logs `loss_ref: self=X cross=X gap=+X` at each log interval. Heartbeat includes `loss_self_ref`/`loss_cross_ref`. |

### Interpreting cross-ref metrics

In the training log:
```
loss_ref: self=0.231 [n=52] cross=0.278 [n=48] gap=+0.047
```

- `self` = loss when reference image matches the caption (normal conditioning)
- `cross` = loss when reference image is shuffled across the batch (mismatched conditioning)
- `gap` = cross − self; positive means the model correctly finds mismatched references harder

**Healthy signal:** gap > 0.01 after ~2K steps. The adapter is learning to distinguish
reference style from unrelated images.

**Warning:** if `cross < self` after 1K steps, the adapter may be ignoring SigLIP
features entirely (losses equalize). Try lowering `cross_ref_prob` to 0.3.

### Pre-flight validation

Before running a full training chunk, validate that all QUALITY features are active
and producing healthy signal with the smoke-test script:

```bash
# Conservative (300 steps, cross_ref_prob=0.3, patch_shuffle_prob=0.3)
train/.venv/bin/python train/scripts/test_quality_features.py \
  --shards /Volumes/2TBSSD/shards \
  --siglip-cache /Volumes/2TBSSD/precomputed/siglip

# Production settings (500 steps, probabilities=0.5, matches stage1_512px.yaml)
train/.venv/bin/python train/scripts/test_quality_features.py \
  --shards /Volumes/2TBSSD/shards \
  --siglip-cache /Volumes/2TBSSD/precomputed/siglip \
  --aggressive

# AI/CI mode (compact JSON verdict to stdout)
train/.venv/bin/python train/scripts/test_quality_features.py \
  --shards /Volumes/2TBSSD/shards \
  --siglip-cache /Volumes/2TBSSD/precomputed/siglip \
  --aggressive --ai
```

The script writes an HTML report to `/tmp/quality_test/quality_test_report.html` by
default. Pass `--output-dir` and `--report` to override. Key verdicts:

- **PASS** — loss stable, cross > self, double-stream scales frozen at 0, single-stream learning
- **WARN** — minor issues (small gap, borderline stability) — check the HTML charts
- **FAIL** — training diverged, scales not frozen, or QUALITY features not firing

The `--no-freeze-double` flag disables QUALITY-2 for isolated testing. Individual
probabilities can be overridden with `--cross-ref-prob` and `--patch-shuffle-prob`.

---

## CLIP Embedding Backend (PIPELINE-4)

`clip_dedup.py embed` supports three backends via `--clip-backend`:

| Backend | Model | Speed (M1 Max, 200 img) | Notes |
|---|---|---|---|
| `open_clip` | ViT-L-14 | ~61 img/s | Default; requires `open-clip-torch` + `torch` |
| `mlx` | ViT-L-14 | ~30 img/s | Fallback; no PyTorch required; uses cached timm weights |
| `transformers` | ViT-B-32 | — | Last resort; lower quality (smaller model) |

**Default (`auto`):** prefers `open_clip → mlx → transformers` in that order.

The `mlx` backend loads weights directly from the safetensors file already cached by
open_clip (`~/.cache/huggingface/hub/models--timm--vit_large_patch14_clip_224.openai/`).
No separate download or conversion is needed.

**Parity:** MLX vs open_clip cosine similarity ≥ 0.9999 (validated on M1 Max).

**When to use `--clip-backend mlx`:**
- PyTorch / open_clip is not installed in the venv
- Reducing RAM footprint matters more than throughput

**Benchmark any shard:**
```bash
train/.venv/bin/python train/scripts/clip_dedup.py embed \
  --shards /Volumes/2TBSSD/shards \
  --embeddings /tmp/bench_out \
  --benchmark
```
