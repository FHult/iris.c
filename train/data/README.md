# train/data — IP-Adapter Dataset Root

This directory is the single canonical root for all training data.
All scripts and configs reference paths under `train/data/` by default.

## Setup

Run the guided setup script once:

```bash
bash train/scripts/setup_data_dir.sh
```

It will:
1. Check your internal SSD for free space
2. Either create the directory structure locally (if ≥ 450 GB free), or
3. Prompt for the external volume path and create a symlink here

## Directory layout

```
train/data/
  raw/                  Downloaded source datasets (large, external SSD recommended)
    laion/              LAION-Aesthetics-v2 WebDataset shards (~150 GB)
    journeydb/          JourneyDB raw files (~80 GB)
    coyo/               COYO-700M filtered shards (~25 GB)
    wikiart/            WikiArt dataset (~2 GB)
  shards/               Merged + shuffled WebDataset shards (~260 GB)
  precomputed/          4-bit/int8 quantised encoder outputs
    qwen3/              Qwen3 text embeddings (~143 GB, saves 200ms/step)
    vae/                VAE latents (~198 GB, saves 180ms/step)
    siglip/             SigLIP features (~420 GB, optional)
  embeddings/           CLIP embeddings for deduplication
  dedup_ids/            Blocklist from CLIP dedup (small)
  checkpoints/          Training checkpoints (~2 GB per save × 5)
  weights/              Downloaded model weights (Flux Klein, Qwen3, SigLIP)
  logs/                 Training and download logs
```

## Symlink vs local

**Symlink to external SSD** (when internal disk < 450 GB free):
```bash
ln -sfn /Volumes/IrisData train/data
```

**Local directory** (when internal disk ≥ 450 GB free):
- Eliminates TB4 disconnection risk during multi-day training
- ~5 GB/s read vs ~2.5 GB/s on TB4 (I/O is not the bottleneck either way)
- Copying shards from external after download: `rsync -a --progress /Volumes/IrisData/ train/data/`

## Storage decision guide

| Component              | Size     | Priority | Notes                              |
|------------------------|----------|----------|------------------------------------|
| Active training shards | ~260 GB  | Internal | Eliminates TB4 dropout risk        |
| Precomputed Qwen3      | ~143 GB  | Internal | 6.7h Stage 1 savings               |
| Precomputed VAE        | ~198 GB  | Internal | 6.0h Stage 1 savings               |
| Raw source datasets    | ~257 GB  | External | Only needed at prep time           |
| Precomputed SigLIP     | ~420 GB  | External | Only if >420 GB free after above   |
| **Total (shards+precomputed)** | **~601 GB** | Internal | If space allows  |
