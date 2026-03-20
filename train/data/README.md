# train/data — IP-Adapter Dataset Root

This directory is always a real local directory — never a symlink in its entirety.
The large raw source datasets live under `train/data/raw/`, which *can* be symlinked
to an external SSD. Everything else stays local to eliminate TB4 dropout risk during
multi-day training runs.

## Setup

```bash
bash train/scripts/setup_data_dir.sh
```

## Directory layout and storage split

```
train/data/                     ← always local

  raw/                          ← symlink this to external SSD if < 260 GB local free
    laion/   (~150 GB)          ← external SSD only (raw img2dataset download)
    journeydb/ (~80 GB)         ← external SSD only
    coyo/    (~25 GB)           ← external SSD only
    wikiart/ (~2 GB)            ← small enough to download locally right now
    laion_filtered.parquet      ← small metadata filter output, keep local
    coyo_filtered.parquet       ← small metadata filter output, keep local

  shards/    (~260 GB)          ← LOCAL — read every step during training
  precomputed/                  ← LOCAL if possible (saves hours of training time)
    qwen3/   (~143 GB)          ← saves 200ms/step = 6.7h over Stage 1
    vae/     (~198 GB)          ← saves 180ms/step = 6.0h over Stage 1
    siglip/  (~420 GB)          ← external SSD if not enough local space

  embeddings/ (~4 GB)           ← CLIP embeddings for dedup (kept after dedup for ref)
  dedup_ids/  (tiny)            ← blocklist output from clip_dedup.py
  weights/    (~6 GB)           ← warmstart weights (InstantX IP-Adapter)
  checkpoints/ (~10 GB active)  ← last 5 checkpoints × ~2 GB each
  logs/                         ← training and download logs
```

## Storage decision guide

| Subdirectory         | Size     | Location    | Why                                      |
|----------------------|----------|-------------|------------------------------------------|
| raw/laion            | ~150 GB  | External    | Only needed during Phase 1–2 prep        |
| raw/journeydb        | ~80 GB   | External    | Only needed during Phase 1–2 prep        |
| raw/coyo             | ~25 GB   | External    | Only needed during Phase 1–2 prep        |
| raw/wikiart          | ~2 GB    | **Local**   | Small — download now while waiting for SSD |
| shards/              | ~260 GB  | **Local**   | Read every step; TB4 dropout = training crash |
| precomputed/qwen3    | ~143 GB  | **Local**   | Dequantised every step; TB4 risk         |
| precomputed/vae      | ~198 GB  | **Local**   | Dequantised every step; TB4 risk         |
| precomputed/siglip   | ~420 GB  | Either      | 50ms/step saving; external OK if tight   |
| weights/             | ~6 GB    | **Local**   | Loaded once at training start            |
| checkpoints/         | ~10 GB   | **Local**   | Written every 2000 steps                 |

**Minimum local space for safe training: ~630 GB** (shards + qwen3 + vae + weights + checkpoints)
**With siglip: ~1050 GB**

If local space is between 260–630 GB, prioritise: shards → qwen3 → vae (in that order).

## Symlinking only raw/

After running `setup_data_dir.sh`, if raw downloads don't fit locally:

```bash
# Remove empty raw/ stub and replace with symlink to external SSD
rmdir train/data/raw
ln -s /Volumes/IrisData/raw train/data/raw

# Everything else in train/data/ stays local
```

Or let `setup_data_dir.sh --external /Volumes/IrisData` do this automatically.
