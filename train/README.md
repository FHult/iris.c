# iris.c — IP-Adapter Training

Training infrastructure for the IP-Adapter companion model for Flux Klein 4B.
Enables `--sref` (style reference) conditioning at inference time.

Full specification in [plans/ip-adapter-training.md](../plans/ip-adapter-training.md).
Roadmap context in [plans/roadmap.md](../plans/roadmap.md).

---

## Directory structure

```
train/
  ip_adapter/               Reusable model code (importable Python package)
    __init__.py
    model.py                IPAdapterKlein + PerceiverResampler + inject() (MLX)
    loss.py                 Flow matching loss + fused Metal kernel
    ema.py                  Exponential moving average of adapter weights
    dataset.py              Two-level async prefetch: tarfile shard + turbojpeg sample
                            Multi-resolution bucketing, GPU augmentation (augment_mlx)
                            Pre-computed embed loading (Qwen3/VAE/SigLIP 4-bit/int8)
    utils.py                Performance core detection (sysctl hw.perflevel0.logicalcpu)
  scripts/
    prepare_laion.py        Pre-filter LAION-Aesthetics-v2 parquet
    download_datasets.sh    Kick off all dataset downloads (run when SSD arrives)
    build_shards.py         Merge raw downloads → unified WebDataset shards
                            multiprocessing.Pool(COMPUTE_WORKERS=6), turbojpeg, zstd-1
    filter_shards.py        Validate shards: drop corrupt/small/bad-caption records
                            multiprocessing.Pool(PERF_CORES=8), rewrites in place
    clip_dedup.py           CLIP-based deduplication via clip-retrieval + FAISS
                            --num_prepro_workers PERF_CORES; writes duplicate_ids.txt
    recaption.py            Moondream re-captioning (style-focused prompt, 2 parallel procs)
    precompute_qwen3.py     4-bit quantise Qwen3 text embeddings (~143 GB, saves 200ms/step)
    precompute_vae.py       int8 quantise VAE latents (~198 GB, saves 180ms/step)
    precompute_siglip.py    4-bit quantise SigLIP features (~420 GB, saves 50ms/step)
  configs/
    stage1_512px.yaml       Stage 1 training config (512px, ~105K steps)
    stage2_768px.yaml       Stage 2 training config (768px, ~20K steps)
    eval_prompts.txt        Fixed (prompt, style_ref) pairs for checkpoint eval
  tests/
    test_model.py           Unit tests for IPAdapterKlein (TODO)
    test_loss.py            Unit tests for flow matching loss (TODO)
  setup.sh                  Create Python venv + install all dependencies
  train_ip_adapter.py       Main training entry point
```

---

## Quick start

### 1. Install dependencies (once)

```bash
cd /path/to/iris.c
bash train/setup.sh
source train/.venv/bin/activate
```

### 2. Set up the data directory (once)

`train/data/` is the canonical data root. It can be a real directory (if local disk
has ≥ 450 GB free) or a symlink to an external SSD. The setup script detects
available space and guides the choice:

```bash
bash train/scripts/setup_data_dir.sh
```

To force a specific path (e.g. external SSD already mounted):

```bash
bash train/scripts/setup_data_dir.sh --external /Volumes/IrisData
```

After setup, all scripts and configs use `train/data/` regardless of where the
data physically lives. The symlink is transparent.

### 3. Today — pre-filter LAION parquet + download warmstart weights

```bash
source train/.venv/bin/activate

# Pre-filter LAION metadata (no images, runs on internal storage, ~1 hour)
python train/scripts/prepare_laion.py \
  --output train/data/raw/laion_filtered.parquet

# Download Perceiver Resampler warmstart weights (5.3 GB)
huggingface-cli download InstantX/FLUX.1-dev-IP-Adapter \
  --local-dir train/data/weights/flux_dev_ipadapter

# Download WikiArt (1.7 GB, tiny)
huggingface-cli download Artificio/WikiArt \
  --repo-type dataset --local-dir train/data/raw/wikiart
```

### 4. When external SSD arrives — download remaining datasets

```bash
# Print download commands for all 3 terminals
bash train/scripts/download_datasets.sh
```

### 5. After downloads complete — pre-process data (B2)

The convenience script runs the full pipeline in one invocation:

```bash
caffeinate -i -d bash train/scripts/run_preprocessing.sh
# Add --siglip to also precompute SigLIP features (~420 GB extra)
```

Or run steps individually:

```bash
source train/.venv/bin/activate

# Step A: Deduplicate LAION (~1.5h embed + 20min FAISS)
python train/scripts/clip_dedup.py all \
  --shards train/data/raw/laion \
  --embeddings train/data/embeddings \
  --output train/data/dedup_ids

# Step B: Merge all sources into unified shards (uses COMPUTE_WORKERS=6 P-cores)
python train/scripts/build_shards.py \
  --sources train/data/raw/laion \
            train/data/raw/journeydb \
            train/data/raw/coyo \
            train/data/raw/wikiart \
  --output train/data/shards \
  --blocklist train/data/dedup_ids/duplicate_ids.txt

# Step C: Filter pass (uses PERF_CORES=8 P-cores)
python train/scripts/filter_shards.py --shards train/data/shards

# Step D: Re-caption short captions (run two processes in separate terminals)
# Terminal 1: python train/scripts/recaption.py --shards train/data/shards --shard_start 0 --shard_end 474
# Terminal 2: python train/scripts/recaption.py --shards train/data/shards --shard_start 475 --shard_end 949

# Step E: Pre-compute frozen encoders
python train/scripts/precompute_qwen3.py \
  --shards train/data/shards --output train/data/precomputed/qwen3

python train/scripts/precompute_vae.py \
  --shards train/data/shards --output train/data/precomputed/vae

# SigLIP (~420 GB) — only if space permits after Qwen3+VAE
# python train/scripts/precompute_siglip.py \
#   --shards train/data/shards --output train/data/precomputed/siglip
```

### 6. Train (B4)

```bash
caffeinate -i -d \
python train/train_ip_adapter.py \
  --config train/configs/stage1_512px.yaml \
  2>&1 | tee logs/train_stage1.log &
```

---

## Architecture summary

See [plans/ip-adapter-training.md — Architecture Specification](../plans/ip-adapter-training.md).

| Component | Parameters | Trainable |
|---|---|---|
| Flux Klein 4B | 4B | Frozen |
| Qwen3 4B (Q4) | 2GB | Frozen |
| VAE encoder | 0.3GB | Frozen |
| SigLIP SO400M | 400M | Frozen |
| PerceiverResampler | ~50M | Yes |
| 25 × to_k_ip [3072,3072] | ~236M | Yes |
| 25 × to_v_ip [3072,3072] | ~236M | Yes |
| 25 × ip_scale | 25 | Yes |
| **Adapter total** | **~522M** | **Yes** |

---

## Memory budget (M1 Max 32GB)

| Component | Memory |
|---|---|
| Flux Klein 4B (BF16, frozen) | 8.0 GB |
| Qwen3 4B (Q4, frozen) | 2.0 GB |
| VAE encoder (frozen) | 0.3 GB |
| SigLIP SO400M (BF16, frozen) | 0.8 GB |
| IP-Adapter (BF16, trainable) | 1.0 GB |
| AdamW optimizer state (FP32) | 2.0 GB |
| Activations at 512px, batch=2 | 3.0 GB |
| OS + framework overhead | 2.0 GB |
| **Total** | **~19 GB** |

---

## Performance optimisations

See [plans/ip-adapter-training.md — Metal Optimisations](../plans/ip-adapter-training.md).

| Optimisation | Step time savings |
|---|---|
| MLX vs PyTorch MPS | −1.5s/step |
| Batched K/V (2 einsum vs 50 GEMM) | −0.3s |
| mx.compile() on loss step | −0.15s |
| BF16 optimizer state | −0.1s |
| Fused flow noise Metal kernel | −0.05s |
| Qwen3 + VAE pre-computed | −0.38s |
| **Total (scenario B)** | **~1.52s/step** |

---

## Training timeline

```
B4 Stage 1 (512px): 105,000 steps × ~1.52s = ~44h (~1.84 days)
B5 Stage 2 (768px):  20,000 steps × ~4.55s = ~25h (~1.05 days)
Total unattended training: ~2.89 days
```

After training, run `plans/ip-adapter-training.md Phase 6` to export and
integrate the weights into iris.c as `iris_ipadapter.c` + `iris_siglip.c`.
