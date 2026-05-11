# IP-Adapter Training — Operational Notes

Empirical findings and strategic analysis for the precompute + training pipeline on M1 Max.

---

## Ablation Harness (QUALITY-10)

`train/scripts/ablation_harness.py` runs a matrix of short training experiments
with different QUALITY hyperparameter combinations and ranks them by style fidelity.

### When to use it

Run the harness before committing to a full production chunk when you want to answer:
- "Does `cross_ref_prob=0.5` outperform `0.3` at this scale?"
- "Does `patch_shuffle_prob=0.5` help or hurt with the current data mix?"
- "Is freezing double-stream scales worth the quality trade-off?"
- "What `style_loss_weight` gives the best ref_gap without instability?"

It does not require inference — ranking is derived entirely from training signal:
`ref_gap = mean(loss_cross_ref - loss_self_ref)` over the tail of each run.

### Quick start

```bash
# 4-combo exploration (small matrix, ~4× 8000 steps ≈ 3–5 hours):
python train/scripts/ablation_harness.py \
    --shards /Volumes/2TBSSD/shards \
    --output-dir train/reports/ablation_run1

# 12-combo medium matrix (freeze vs no-freeze sweep):
python train/scripts/ablation_harness.py --matrix medium --steps 5000 \
    --shards /Volumes/2TBSSD/shards \
    --output-dir train/reports/ablation_run2

# Resume an interrupted run:
python train/scripts/ablation_harness.py \
    --output-dir train/reports/ablation_run1 --resume

# Dry run — see the matrix without training:
python train/scripts/ablation_harness.py --matrix full --dry-run
```

### Matrix presets

| Preset | Variables swept | Combos | Recommended steps |
|--------|-----------------|--------|-------------------|
| `small` (default) | `cross_ref_prob=[0.3,0.5]` × `patch_shuffle_prob=[0.0,0.5]` | 4 | 8 000 |
| `medium` | adds `freeze_double=[T,F]`; 3-value `cross_ref` | 12 | 5 000–8 000 |
| `full` | all 4 variables at 3 values each | 54 | 5 000 |

Custom matrices can be defined in YAML (`--matrix-file`):

```yaml
ablation:
  variables:
    cross_ref_prob: [0.0, 0.3, 0.5]
    patch_shuffle_prob: [0.0, 0.5]
    freeze_double_stream_scales: [true, false]
    style_loss_weight: [0.0, 0.05, 0.1]
  steps_per_run: 8000
```

### Scoring formula

```
score = 100 × mean_ref_gap + 200 × mean_cond_gap − 3 × final_loss
```

`ref_gap` is the primary signal: a positive gap (cross-ref harder than self-ref)
means the adapter is style-aware. Without SigLIP precomputed embeddings `ref_gap`
will be absent — always run with `--siglip-cache` for meaningful results.

### Output

```
train/reports/ablation_run1/
  index.html          — ranked report with charts (open in browser)
  results.json        — slim results for all combos (no snapshots)
  runs/small/
    combo_001/
      config.yaml     — exact config used
      metrics.json    — full metric snapshots (used by index.html charts)
      training.log    — raw trainer output
```

### Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--matrix` | `small` | Preset name |
| `--matrix-file` | — | Custom YAML (overrides `--matrix`) |
| `--steps` | 8000 | Steps per combo |
| `--log-every` | auto | Default: `steps // 80`, min 50 |
| `--max-runs N` | — | Stop after N combos (for testing the harness) |
| `--resume` | off | Skip combos already in `results.json` |
| `--dry-run` | off | Print matrix without training |
| `--quiet` | off | Suppress per-step trainer output |
| `--keep-checkpoints` | off | Keep checkpoint files (disk-intensive) |
| `--ai` | off | Emit compact JSON summary to stdout |

### Signal quality vs run length

Reliable signal on ref_gap requires SigLIP cache and enough steps to pass warmup.
Rule of thumb for step budget:

| Goal | Steps | Notes |
|------|-------|-------|
| Screening (eliminate clearly bad configs) | 3 000 | Enough to see if ref_gap goes positive |
| Comparative ranking | 8 000 | Standard; catches mid-run instabilities |
| High-confidence ranking | 15 000 | Use for final config before a full chunk |

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

---

## Storage Tiers

### Overview

The pipeline supports a two-tier storage model for situations where fast NVMe
capacity is limited but a larger/cheaper drive (HDD or secondary SSD) is
available for long-term data.

| Tier | Role | Typical device |
|---|---|---|
| **Cold** (`cold_root`) | Source of truth. Raw shards, all versioned precompute caches, archived weights. Never auto-deleted. | HDD or secondary TB4/TB5 SSD |
| **Hot** (`hot_root`) | Active compute. Only the current and next chunk's data. | Primary NVMe SSD |

The stager (`train/scripts/data_stager.py`) moves data bidirectionally:

- **Staging (cold → hot):** Before precompute or training, symlinks or copies
  the required shards and versioned `.npz` caches from cold to hot storage.
- **Archiving (hot → cold):** After a chunk completes successfully, copies new
  precomputed data and adapter weight snapshots from hot back to cold.

### Single-SSD prototyping (default)

Set `cold_root == hot_root` (or omit the `storage:` block entirely). All
staging and archiving operations become no-ops — nothing is copied or
symlinked. This is the default in `v2_pipeline.yaml`.

### Two-device setup

Edit `train/configs/v2_pipeline.yaml`:

```yaml
storage:
  cold_root: "/Volumes/RawHDD"    # HDD — source of truth
  hot_root:  "/Volumes/FastTB5"   # fast NVMe — active compute
  staging_margin_gb: 250          # keep this much free headroom on hot
  cleanup_safety_gb: 100
  archive_after_chunk: true
  background_staging: true
  max_parallel_transfers: 3
```

**Same-device detection:** If `cold_root` and `hot_root` happen to share the
same physical filesystem (`os.stat().st_dev` match), symlinks are used
(instant, zero disk cost). On different physical devices, files are copied
atomically (temp-write → `os.replace`), so a crash never leaves partial data.

### Orchestrator lifecycle

The stager hooks into the orchestrator at two points automatically:

1. **Training start** (`notify_training_start`): When chunk N training begins,
   staging for chunk N+1 starts in background. By the time training finishes
   (~25–83 h), chunk N+1's shards and precomputed caches are already on hot.

2. **Training complete** (`archive_chunk_background`): When chunk N training
   succeeds, new precomputed data and weight snapshots are copied to cold in
   background while the orchestrator moves on to the next step.

Both run as low-priority daemon threads (`os.nice(10)`) and write heartbeats
to `.heartbeat/stager_chunk{N}.json` for visibility.

### Manual CLI usage

```bash
# Stage chunk 2's shards and precomputed data from cold to hot
train/.venv/bin/python train/scripts/data_stager.py stage --chunk 2

# Archive chunk 1's results from hot to cold
train/.venv/bin/python train/scripts/data_stager.py archive --chunk 1

# Show stager status (enabled, device type, free space, active threads)
train/.venv/bin/python train/scripts/data_stager.py status
```

### Hot storage budget (two-device setup)

At `large` scale with `max_shards: 80`:

| Data type | Approx size |
|---|---|
| 80 shards (`.tar`) | ~2 GB |
| Qwen3 precomputed (80 shards × ~5K records × ~30 KB) | ~12 GB |
| VAE precomputed (80 shards × ~5K records × ~8 KB) | ~3 GB |
| SigLIP precomputed (80 shards × ~5K records × ~2 KB) | ~0.8 GB |
| Checkpoint (`best.safetensors`) | ~2 GB |
| **Total active chunk** | **~20 GB** |

Two chunks (current + next) requires ~40 GB on hot, well within a 500 GB NVMe.
Cold storage accumulates all historical versions; at 4 chunks that is ~80 GB of
precomputed data plus ~8 GB of weight snapshots.
