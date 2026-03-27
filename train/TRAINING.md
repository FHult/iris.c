# IP-Adapter Training — Operational Notes

Empirical findings and strategic analysis for the precompute + training pipeline on M1 Max.

---

## Dataset vs Training Coverage

### Actual numbers (chunk 1)

| Metric | Value |
|---|---|
| Chunk 1 shards | 432 |
| Chunk 1 images | ~2.15M (avg 4,970/shard) |
| Stage 1 training samples (current plan) | 105K steps × batch=2 = **210K** |
| Shards actually consumed | **~42** (of 432) |
| Image coverage | **~10%** |

The plan document claims "78 visits per image over 120K steps" — this is wrong at
the 2M-image scale. It was written for a small pilot (~3K images). At the actual
scale, most images are never seen.

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

### Is 210K samples enough? Probably not.

Reference IP-Adapters used 1–10M+ training samples. At 210K:
- The model likely learns basic style transfer
- It will underfit on unusual artistic styles underrepresented in the sample draw
- Style fidelity will be inconsistent across reference image types
- The current step count (105K) is too low for the dataset size

### Should we increase steps?

Yes, but paired with more shards — increasing steps alone on 42 shards overfits
those shards. The right formula:

```
target_unique_samples × (1.5–2.5 passes) = total_training_samples
total_training_samples / batch_size = num_steps
```

Quality vs time tradeoff on M1 Max (1.5s/step):

| Steps | Wall clock | Shards needed | Unique images | Quality |
|---|---|---|---|---|
| 105K (current plan) | 44h | 42 | 70K | Marginal |
| 200K | 83h | 80 | 133K | Acceptable |
| 400K | 7 days | 160 | 267K | Good |
| 540K | 9.4 days | ALL (432) | 2.15M (0.5×) | Reference-class |

**Recommended for chunk 1: increase to 200K steps with ~80 shards.** This doubles
wall-clock training time but meaningfully improves quality without being excessive.

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
| `large` | 200K | 60K | 81 | 25 | ~11 days |
| `god-like` | 400K | 120K | 162 | 50 | ~18 days |
| `all-in` | 540K | 200K | ALL | ALL | ~26 days |

`all-in` precomputes the entire downloaded pool — no shard limit. Steps are sized
for ~0.5 pass through 432 shards at batch=2. Actual coverage depends on total
shards available when the pipeline runs.

Recommended default: **`large`**.
Compare to current plan: 225K steps, 38h precompute, 470K images at 9% coverage.

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

Training can start as soon as ~80 shards are precomputed. Currently 13 shards are
done — wait for ~67 more (~26h at current rate) then begin chunk 1 training.

The dataset loader cycles through available shards in epochs. Do not add new shards
mid-run; restart from a checkpoint with the expanded shard list if needed.

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

| Phase | Current plan | Recommended plan |
|---|---|---|
| Chunk 1 precompute | ~17h (42 shards) | ~31h (80 shards) |
| Chunk 1 training | ~44h (105K steps) | ~83h (200K steps) |
| Chunks 2-4 precompute | ~3× 4–7h | ~3× 10h |
| Chunks 2-4 training | ~3× 25h (60K each) | ~3× 37h (90K each) |
| **Total** | **~170h (~7 days)** | **~275h (~11 days)** |

4 extra days of wall clock for meaningfully better model quality. The increased
step counts can run overnight in stages; the machine is not blocking anything else
during training.
