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
    model.py                IPAdapterKlein + PerceiverResampler (MLX)
    loss.py                 Flow matching loss + fused Metal kernel
    ema.py                  Exponential moving average of adapter weights
    dataset.py              WebDataset loader with two-level async prefetch
  scripts/
    prepare_laion.py        Pre-filter LAION-Aesthetics-v2 parquet (run today)
    download_datasets.sh    Kick off all dataset downloads (run when SSD arrives)
    build_shards.py         Merge raw downloads → WebDataset shards (TODO)
    precompute_encoders.py  Pre-compute Qwen3/VAE/SigLIP features (TODO)
    clip_dedup.py           CLIP-based deduplication via FAISS (TODO)
    recaption.py            Moondream re-captioning for short captions (TODO)
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

### 2. Today — pre-filter LAION parquet + download warmstart weights

```bash
source train/.venv/bin/activate

# Pre-filter LAION metadata (no images, runs on internal storage, ~1 hour)
python train/scripts/prepare_laion.py --output laion_filtered.parquet

# Download Perceiver Resampler warmstart weights (5.3 GB, to internal storage)
huggingface-cli download InstantX/FLUX.1-dev-IP-Adapter \
  --local-dir ~/iris_data_staging/flux_dev_ipadapter

# Download WikiArt (1.7 GB, tiny — can go anywhere)
huggingface-cli download Artificio/WikiArt \
  --repo-type dataset --local-dir ~/iris_data_staging/wikiart
```

### 3. Tomorrow — when external SSD arrives

```bash
# Create directory structure and print download commands for all 3 terminals
bash train/scripts/download_datasets.sh \
  --ssd /Volumes/IrisData \
  --laion laion_filtered.parquet
```

### 4. After downloads complete — pre-process data (B2)

```bash
# TODO: run after scripts/clip_dedup.py, recaption.py, build_shards.py are complete
# python train/scripts/build_shards.py --ssd /Volumes/IrisData
```

### 5. Train (B4)

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
