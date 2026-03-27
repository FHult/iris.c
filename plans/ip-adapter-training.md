# IP-Adapter Training Plan for Flux Klein 4B
## Style Reference (`--sref`) Capability

**Goal:** Train an IP-Adapter companion model for Flux Klein 4B that enables style
reference conditioning at inference time, equivalent to Midjourney's `--sref` feature.

This plan is Track B of the overall roadmap. See [roadmap.md](roadmap.md) for how
this fits alongside the shippable-now Track A work (Path 1 style ref, Z-Image-Omni-Base).

**Hardware:** M1 Max 32GB unified memory
**Training framework:** MLX (not PyTorch — see Phase 0)
**Estimated active dev time:** 4–6 weeks
**Estimated elapsed time:** 6–8 weeks (most phases run unattended)

---

## Background

### Why IP-Adapter (not T-offset or LoRA)

Three approaches were evaluated:

| Approach | Style quality | New weights | Training | Notes |
|---|---|---|---|---|
| T-offset naive (extra tokens at T=50) | Poor — content bleed | None | None | Model not trained for this; T-offset has no semantic meaning for style |
| Training-free "Untwisting RoPE" (arXiv 2602.05013) | Moderate | None | None | RoPE frequency attenuation in single-stream blocks; viable first step |
| IP-Adapter (this plan) | High | Yes (~1GB) | Required | Decoupled cross-attention; style isolated at block-type level |
| LoRA | N/A | Yes | Required | LoRA can't add new conditioning inputs; only modifies existing weights |

LoRA is not suitable for dynamic per-image style references — it bakes a specific style into
the model weights. IP-Adapter learns to accept any reference image at inference time.

### Style isolation mechanism

IP-Adapter injects image conditioning into all transformer blocks. Style-only mode
at inference is achieved by injecting **only into single-stream blocks** (blocks 5–24 in
Klein 4B), which mechanistic analysis of Flux confirms are the appearance/style blocks.
Content+style mode uses all 25 blocks.

---

## Architecture Specification

### Vision encoder (frozen, not trained)

**Model:** `google/siglip-so400m-patch14-384` (400M parameters)

- Input: 384×384 reference image
- Output: 729 patch tokens at 1152-dim
- Not trained; loaded and frozen

### Image projection / Perceiver Resampler (trainable, ~50M params)

- 128 learnable query vectors at 3072-dim
- Cross-attention over the 729 SigLIP patch tokens
- LayerNorm output
- Output: 128 image conditioning tokens at 3072-dim

```
SigLIP output [B, 729, 1152]
       ↓  cross-attention with 128 learned queries
Image tokens [B, 128, 3072]
```

### Per-block decoupled cross-attention (trainable, ~472M params)

Added to all **25 Klein 4B blocks** (5 double-stream + 20 single-stream):

- `to_k_ip`: `Linear(3072, 3072, bias=False)` — one per block
- `to_v_ip`: `Linear(3072, 3072, bias=False)` — one per block
- `scale`: learnable scalar per block, initialized to 1.0

At inference:
- Image Q attends to `k_ip(image_tokens)` and `v_ip(image_tokens)`
- Result scaled by `scale[block_idx]` and added to standard attention output

**Style-only inference mode:** inject only into the 20 single-stream blocks (blocks 5–24),
set scale=0 for the 5 double-stream blocks.

### Total trainable parameters

| Component | Parameters |
|---|---|
| Perceiver Resampler | ~50M |
| 25 × to_k_ip (3072×3072) | ~236M |
| 25 × to_v_ip (3072×3072) | ~236M |
| 25 × scale scalars | ~25 |
| **Total** | **~522M** |

**Adapter weights at BF16:** ~1.0 GB safetensors file

---

## Memory Budget (Training, M1 Max 32GB)

| Component | Memory | Notes |
|---|---|---|
| Flux Klein 4B (BF16, frozen) | 8.0 GB | No grad |
| Qwen3 4B (Q4 quantized, frozen) | 2.0 GB | 4-bit quant; frozen so quality cost is acceptable |
| VAE encoder (frozen) | 0.3 GB | Encode path only |
| SigLIP SO400M (BF16, frozen) | 0.8 GB | No grad |
| IP-Adapter adapter (BF16, trainable) | 1.0 GB | Grad enabled |
| AdamW optimizer state (FP32) | 2.0 GB | 2 moments × 522M params |
| Activations at 512px, batch_size=1 | 3.0 GB | |
| OS + framework overhead | 2.0 GB | |
| **Total** | **~19 GB** | ~13 GB headroom in 32 GB |

Training at 1024px would push activations to ~12GB, making the total ~28GB — tight.
**512px is the target training resolution.**

With `mx.checkpoint()` on transformer blocks (see Phase 3), activations drop by ~600MB at
512px, enabling batch_size=2 and making 768px training feasible in Stage 2.

---

## Phase 0: Infrastructure Setup
**Duration: 2–3 days**

### Storage and hardware

**Required free space:** ~400GB for dataset + ~15GB for checkpoints + headroom → plan for
~450GB total.

**Thunderbolt 4 external SSD is fine for dataset storage but carries a reliability risk
for long unattended training runs.** I/O throughput is not a concern — the training loop
requires less than 0.01% of TB4's 2.5 GB/s read bandwidth (~105 KB/s sustained at
batch_size=2 and 1.9s/step). The bottleneck is always GPU, not storage.

The real risk is connection dropout over a 2–3 day unattended run. A TB4 disconnection
crashes the training process; worst-case loss is progress since the last checkpoint.

**Recommended storage layout:**

| Data | Location | Reason |
|---|---|---|
| Active training shards (~260GB) | Internal SSD (if 300GB+ free) | Eliminates dropout risk |
| Raw source datasets (LAION, JourneyDB, COYO) | External TB4 SSD | Only needed at prep time |
| Model weights (Flux Klein, Qwen3, SigLIP) | External TB4 SSD | Loaded once at startup |
| Checkpoints (~12GB active) | Either | Small enough for either location |

If internal SSD does not have 300GB free, keep shards on the external drive and apply
the reliability mitigations below.

**Mitigations when training from external SSD:**

```bash
# Prevent macOS sleep from disconnecting the TB4 drive — critical for unattended runs
caffeinate -i -d python train_ip_adapter.py \
  --data_path /Volumes/ExternalSSD/train_shards \
  --output_dir checkpoints/stage1 \
  > logs/train_stage1.log 2>&1 &

# Reduce checkpoint interval from 5000 to 2000 steps
# Worst-case lost work drops from ~2.6 hours to ~1 hour at negligible storage cost
# (2000-step checkpoints: ~2GB every ~1 hour vs ~2GB every ~2.6 hours)
```

Additional precautions:
- Use a certified Thunderbolt 4 cable, not a generic USB-C cable
- Do not move the laptop or cable during a training run
- Verify the drive stays mounted with a periodic health-check script:

```bash
# health_check.sh — run in background during training
while true; do
    if [ ! -d "/Volumes/ExternalSSD/train_shards" ]; then
        echo "$(date): TB4 drive unmounted — training may have crashed" | \
          tee -a logs/health.log
    fi
    sleep 300  # check every 5 minutes
done &
```

**Copying shards to internal SSD before training (if space permits):**

```bash
# ~260GB at internal SSD write speeds (~5 GB/s) — completes in ~52 seconds
rsync -a --progress /Volumes/ExternalSSD/train_shards/ ~/train_shards/
```

### Framework choice: MLX, not PyTorch

The training loop uses **MLX** throughout. PyTorch MPS and MLX cannot share Metal buffers
without copying — mixing them in the same training loop wastes memory and adds overhead.
MLX is the right choice because:

- mflux already implements the full Flux Klein 4B forward pass in MLX
- MLX's lazy evaluation batches Metal command submissions automatically; PyTorch MPS
  dispatches eagerly with per-op kernel launch overhead
- `nn.freeze()` + `nn.value_and_grad()` gives clean frozen/trainable separation
  with no explicit `no_grad` blocks required
- `mx.checkpoint()` (from mlx-lm) provides gradient checkpointing for transformer blocks
- `mx.fast.metal_kernel()` with `.vjp` supports custom Metal kernels with full autograd

Estimated step time: **~2.3s** (vs ~4.0s with PyTorch MPS — see Metal Optimisations section).

### Python environment

```bash
pyenv install 3.11.9
pyenv local 3.11.9

python -m venv ip_train_env
source ip_train_env/bin/activate

pip install mlx mlx-lm           # MLX + training utilities (gradient checkpointing)
pip install mflux                # Flux Klein MLX forward pass
pip install safetensors webdataset huggingface_hub
pip install Pillow tqdm wandb
```

### Verify MLX and Metal

```python
import mlx.core as mx
print(mx.default_device())   # must print Device(gpu, 0)

# Verify BF16 matmul
x = mx.random.normal((64, 3072), dtype=mx.bfloat16)
y = mx.random.normal((3072, 3072), dtype=mx.bfloat16)
z = x @ y
mx.eval(z)
print("MLX BF16 matmul OK")
```

### Training codebase

Base: `https://github.com/filipstrand/mflux` (MLX Flux Klein inference)

Work required:
- Subclass `Flux2Transformer` to capture per-block Q tensors and accept IP injection
- Implement IP-Adapter architecture in MLX (Perceiver Resampler + per-block K/V projections)
- Add `mx.checkpoint()` to transformer block `__call__` methods
- Implement MLX training loop with `nn.value_and_grad` and AdamW
- Replace Qwen3 with Q4-quantized Qwen3 via `mlx_lm.utils.quantize_model`

---

## Phase 1: Dataset Acquisition
**Duration: 7–14 days (bandwidth-limited, runs unattended)**
**Target: 2,000,000 images, ~257 GB storage**

### License summary

| Source | License | Commercial use |
|---|---|---|
| LAION-Aesthetics | CC-BY metadata; images vary by source | Not cleared |
| JourneyDB | Non-commercial research only | No |
| COYO-700M | Apache 2.0 metadata; images vary | Not cleared |
| WikiArt | Non-commercial research only | No |
| Museum open-access (MET, Rijksmuseum) | Public domain works | Yes |

This dataset composition is suitable for research and personal use. For commercial
deployment of the trained adapter, replace LAION/JourneyDB with commercially-licensed
sources (Unsplash API, Pexels API, museum open-access collections).

### 1.1 LAION-Aesthetics-v2 — 1,200,000 images

```bash
pip install img2dataset

# Download metadata (~40 GB)
huggingface-cli download laion/laion2B-en-aesthetic \
  --repo-type dataset --local-dir laion_meta

# Download images: 512px, aesthetic score >= 5.5
img2dataset \
  --url_list laion_meta \
  --input_format parquet \
  --url_col URL \
  --caption_col TEXT \
  --output_format webdataset \
  --output_folder data/laion_512 \
  --image_size 512 \
  --resize_only_if_bigger True \
  --resize_mode keep_ratio \
  --processes_count 16 \
  --thread_count 64 \
  --min_value aesthetic 5.5 \
  --save_additional_columns '["aesthetic"]'
```

Expected yield: ~70% URL success rate. Stop once 1.2M images are collected.
Download time at 1 Gbps: ~18–24 hours, ~150 GB.

### 1.2 JourneyDB — 500,000 images (non-commercial research)

```bash
# Accept terms at https://journeydb.github.io then:
huggingface-cli download JourneyDB/JourneyDB \
  --repo-type dataset --local-dir data/journeydb_raw

# Convert 500K random sample to WebDataset shards at 512px
python scripts/shard_journeydb.py \
  --input data/journeydb_raw \
  --output data/journeydb_512 \
  --num_samples 500000 \
  --image_size 512
```

Full dataset is 683 GB; the 500K subset is ~80 GB.

### 1.3 WikiArt — 100,000 images (style diversity)

```bash
huggingface-cli download Artificio/WikiArt \
  --repo-type dataset --local-dir data/wikiart
# 1.71 GB, ~2 minutes
```

Prepend style label to captions: `"An impressionist painting showing {original_caption}"`.
This improves text-image alignment during training for the art subset.

### 1.4 COYO filtered — 200,000 images (photographic diversity)

```bash
huggingface-cli download dwb2023/filtered-coyo-700M-beta \
  --repo-type dataset --local-dir coyo_meta

# Sample 200K by highest aesthetic score
python scripts/sample_coyo.py \
  --meta coyo_meta --n 200000 --output coyo_200k.parquet

img2dataset \
  --url_list coyo_200k.parquet \
  --input_format parquet \
  --url_col url --caption_col text \
  --output_format webdataset \
  --output_folder data/coyo_512 \
  --image_size 512
```

### Dataset summary

| Source | Images | Storage | Purpose |
|---|---|---|---|
| LAION-Aesthetics ≥5.5 | 1,200,000 | ~150 GB | Broad diversity |
| JourneyDB | 500,000 | ~80 GB | Stylistic richness |
| COYO filtered | 200,000 | ~25 GB | Photo realism |
| WikiArt | 100,000 | ~2 GB | Art style diversity |
| **Total** | **2,000,000** | **~257 GB** | |

---

## Phase 2: Pre-processing Pipeline
**Duration: 2–3 days (CPU-bound, runs unattended)**

> **Always run pre-processing under `caffeinate`.**
> The full pipeline takes 2–3 days. A display sleep or system sleep mid-run will
> stall or corrupt in-progress tar writes and shard merges.
>
> ```bash
> caffeinate -i -d bash scripts/run_preprocessing.sh
> ```
>
> `-i` prevents idle sleep; `-d` prevents display sleep. Wrap the entire pipeline
> (phases 2.1–2.6) in a single `caffeinate` invocation, not per-command.
> This is the same mitigation used for the training run in Phase 4.

### 2.0 CPU core allocation — detect and target performance cores

M1 Max has 8 performance cores and 2 efficiency cores. Data preparation work should
explicitly target the performance cores. Never blindly use `os.cpu_count()` (returns 10
including efficiency cores) or a hardcoded number.

Detect performance core count at the start of each script:

```python
import subprocess, os

def get_performance_core_count():
    """Return the number of performance cores on Apple Silicon.
    Falls back to os.cpu_count() on non-Apple hardware."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            text=True
        ).strip()
        return int(out)
    except Exception:
        return os.cpu_count()

PERF_CORES = get_performance_core_count()  # 8 on M1 Max
```

Use this value to set worker counts throughout the pipeline. Leave 1–2 performance
cores free for the OS and Metal command encoding when GPU is also active:

```python
CPU_WORKERS     = PERF_CORES          # I/O-bound tasks (download, shard read)
COMPUTE_WORKERS = max(1, PERF_CORES - 2)  # CPU-compute tasks (decode, encode, norm)
```

**Per-phase allocation guide:**

| Phase | Task | Worker count | Rationale |
|---|---|---|---|
| 2.2 img2dataset download | network + resize | `PERF_CORES` (8) | I/O-bound, all cores useful |
| 2.3 CLIP embedding | GPU-backed, CPU pre-proc | `PERF_CORES` (8) | CPU feeds GPU batches |
| 2.4 Moondream re-caption | 2 parallel procs × GPU | `PERF_CORES // 2` per proc | Shared GPU; split CPU evenly |
| 2.5 Shard merge/shuffle | compression (zstd) | `COMPUTE_WORKERS` (6) | Leave headroom for I/O |
| 2.6 Filter pass | decode + stat check | `PERF_CORES` (8) | Pure CPU, saturate all |
| Training prefetch | async loader threads | 2 | GPU is bottleneck; more adds noise |

### 2.1 Pre-filter LAION parquet before downloading

Filter to aesthetic ≥ 5.5 and minimum dimensions before img2dataset runs.
Reduces img2dataset's URL resolution work by ~75%:

```python
import pandas as pd
df = pd.read_parquet("laion_meta/", columns=["URL","TEXT","aesthetic","width","height","punsafe"])
df = df[(df["aesthetic"] >= 5.5) & (df["punsafe"] < 0.05) &
        (df["width"] >= 256) & (df["height"] >= 256)]
df.to_parquet("laion_filtered.parquet", index=False)
# Pass laion_filtered.parquet to img2dataset instead of the full meta directory
```

### 2.2 Run all dataset downloads in parallel

LAION and COYO are network-bound; JourneyDB is a single HuggingFace pull.
All three can run simultaneously — cuts total download from ~2 weeks to ~1 week.

`img2dataset` uses `--processes_count` for CPU workers and `--thread_count` for I/O
threads per worker. Set both explicitly using the detected performance core count:

```bash
# M1 Max: PERF_CORES=8, so processes_count=8, thread_count=16 (2 I/O threads per worker)
PERF_CORES=$(sysctl -n hw.perflevel0.logicalcpu)
IO_THREADS=$((PERF_CORES * 2))

img2dataset \
  --url_list laion_filtered.parquet \
  --input_format parquet \
  --url_col URL --caption_col TEXT \
  --output_format webdataset \
  --output_folder data/laion_512 \
  --image_size 512 \
  --processes_count $PERF_CORES \
  --thread_count $IO_THREADS \
  --resize_mode keep_ratio_largest &

huggingface-cli download JourneyDB/JourneyDB \
  --repo-type dataset --local-dir data/journeydb &

img2dataset \
  --url_list coyo_200k.parquet \
  --input_format parquet \
  --url_col url --caption_col text \
  --output_format webdataset \
  --output_folder data/coyo_512 \
  --image_size 512 \
  --processes_count $PERF_CORES \
  --thread_count $IO_THREADS \
  --resize_mode keep_ratio_largest &

wait
```

On M1 Max (`PERF_CORES=8`): `--processes_count 8 --thread_count 16`. This saturates all
8 performance cores with parallel resize workers, each running 2 I/O threads for
pipelined URL fetch + decode. Efficiency cores are left for the OS scheduler.

### 2.3 CLIP-based deduplication

LAION has ~15–20% near-duplicate images. Remove them before training to maximise
dataset diversity. Uses `clip-retrieval` + FAISS (~1.5 hours for embedding, ~20 minutes
for FAISS search on 2M images):

```bash
pip install clip-retrieval autofaiss

# Embed all 2M images with CLIP ViT-L/14 (~1.5 hours on M1 Max)
# num_prepro_workers feeds CPU decode/preprocess pipeline ahead of GPU batching
PERF_CORES=$(sysctl -n hw.perflevel0.logicalcpu)
clip-retrieval inference \
  --input_dataset data/train_shards \
  --output_folder data/embeddings \
  --clip_model ViT-L/14 \
  --batch_size 256 \
  --num_prepro_workers $PERF_CORES

# Build FAISS index and find duplicates (cosine similarity > 0.95)
clip-retrieval index \
  --embeddings_folder data/embeddings \
  --index_folder data/faiss_index \
  --max_index_memory_usage "12GB"

clip-retrieval deduplication \
  --embeddings_folder data/embeddings \
  --output_folder data/dedup_ids \
  --threshold 0.95
```

Expected removal: 200–400K duplicate images. Output is a blocklist of IDs to skip during
training (no need to rebuild shards). Effective unique dataset: ~1.6M images.

### 2.4 Re-caption short captions with Moondream (style-focused)

~20% of LAION images have captions under 10 words. Re-caption these with a style-focused
prompt to generate the style vocabulary the adapter needs to learn:

```bash
pip install mlx-vlm

# Run two parallel processes to cover 400K images in ~2 days
python scripts/recaption.py \
  --input data/train_shards --shard_start 0 --shard_end 474 \
  --model mlx-community/moondream2-4bit \
  --min_caption_words 10 \
  --prompt "Describe this image's visual style, colors, lighting, and artistic technique." &

python scripts/recaption.py \
  --input data/train_shards --shard_start 475 --shard_end 949 \
  --model mlx-community/moondream2-4bit \
  --min_caption_words 10 \
  --prompt "Describe this image's visual style, colors, lighting, and artistic technique." &
```

The style-focused prompt generates captions like "painterly impressionist style with warm
amber tones, soft bokeh, and textured brushwork" — directly useful training signal for a
style adapter. Generic "describe this image" prompts describe objects, not style.

### 2.5 Merge and shuffle into unified WebDataset shards

```bash
# 380 tar shards × 5000 images each (larger shards = fewer shard transitions during training)
# Shuffle across sources to prevent source clustering in batches
# workers = PERF_CORES - 2: leave headroom for I/O and OS scheduler during zstd compression
COMPUTE_WORKERS=$(( $(sysctl -n hw.perflevel0.logicalcpu) - 2 ))
python scripts/merge_and_shuffle.py \
  --sources data/laion_512 data/journeydb_512 data/coyo_512 data/wikiart \
  --output data/train_shards \
  --shard_size 5000 \
  --shuffle \
  --workers $COMPUTE_WORKERS \
  --compression zstd \
  --compression_level 1 \
  --blocklist data/dedup_ids/duplicate_ids.txt
```

**Compression level:** use `zstd -1` (fastest), not the zstd default (`-3`). At level 1,
zstd compresses ~40% faster with negligible size increase (<2%). During training the loader
reads shards sequentially — decompression speed at level 1 is identical to level 3 (both
are CPU-trivial compared to JPEG decode). The shard-write phase is the only time
compression level matters for elapsed time.

Larger shards (5000 vs 2000) reduce filesystem open/close operations by 60% during
training. The blocklist excludes deduplicated images without rebuilding sources.

**Parallel shard-writing implementation (`scripts/merge_and_shuffle.py`):**

The bottleneck in shard creation is JPEG decode + resize + tar-write. This is embarrassingly
parallel across shards — each worker owns a disjoint output shard range and writes
independently with no shared state:

```python
import os, subprocess, multiprocessing, math, random, tarfile, io
from PIL import Image

def get_perf_cores():
    try:
        return int(subprocess.check_output(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
    except Exception:
        return os.cpu_count()

def write_shard_range(args):
    """Worker function: writes a contiguous range of output shards."""
    shard_ids, records, output_dir, shard_size, blocklist = args
    blocklist_set = set(blocklist)
    idx = 0
    for shard_id in shard_ids:
        shard_path = os.path.join(output_dir, f"{shard_id:06d}.tar")
        with tarfile.open(shard_path, "w") as tar:
            written = 0
            while written < shard_size and idx < len(records):
                rec = records[idx]; idx += 1
                if rec["id"] in blocklist_set:
                    continue
                # Decode, validate, resize, re-encode
                try:
                    img = Image.open(io.BytesIO(rec["jpg"]))
                    if img.width < 256 or img.height < 256:
                        continue
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=95)
                    _tar_add(tar, f"{shard_id:06d}_{written:04d}.jpg", buf.getvalue())
                    _tar_add(tar, f"{shard_id:06d}_{written:04d}.txt",
                             rec["txt"].encode())
                    written += 1
                except Exception:
                    continue  # skip corrupt records

def main(sources, output_dir, shard_size, workers, blocklist_path):
    # Load all record metadata (paths only; actual decode happens in workers)
    records = load_all_records(sources)  # list of {id, jpg_path, txt}
    random.shuffle(records)

    n_shards = math.ceil(len(records) / shard_size)
    # Split shard index ranges across workers
    shard_ranges = [list(range(i, n_shards, workers)) for i in range(workers)]
    blocklist = load_blocklist(blocklist_path)

    # Each worker gets its own non-overlapping shard IDs — no locking needed
    work_items = [
        (shard_ranges[w], records, output_dir, shard_size, blocklist)
        for w in range(workers)
    ]
    with multiprocessing.Pool(processes=workers) as pool:
        pool.map(write_shard_range, work_items)
```

**Install turbojpeg before running shard creation or the training prefetch:**

```bash
brew install libjpeg-turbo
pip install PyTurboJPEG
```

```python
from turbojpeg import TurboJPEG
_jpeg = TurboJPEG()  # one instance per process (not thread-safe across processes)

def decode_jpeg(raw_bytes):
    return _jpeg.decode(raw_bytes)   # 2–4× faster than Image.open(BytesIO(...))

def encode_jpeg(img_array, quality=85):
    return _jpeg.encode(img_array, quality=quality)
```

Use `decode_jpeg` / `encode_jpeg` throughout shard writing and the training prefetch thread.
The quality=85 setting also applies here (see section 2.5 compression note).

Key design decisions:
- **No shared state between workers** — each worker writes to its own shard files. No locks, no queues, no coordination overhead.
- **Interleaved shard ownership** (`range(i, n_shards, workers)`) — worker 0 owns shards 0, 8, 16 …; worker 1 owns 1, 9, 17 … This keeps the global shuffle intact across worker boundaries rather than giving each worker a contiguous block of records.
- **`multiprocessing.Pool` not `threading.Pool`** — Python's GIL blocks true CPU parallelism for CPU-bound work (JPEG decode, resize). `multiprocessing` bypasses the GIL; each worker runs in its own process on its own performance core.
- With `COMPUTE_WORKERS=6` on M1 Max: ~6× throughput on JPEG decode + resize vs single-threaded.

### 2.6 Filter pass

Drop records with:
- Corrupted or unreadable images
- `width < 256` or `height < 256`
- Empty captions, captions < 5 words, captions that are filenames or URLs

The filter pass should also use `multiprocessing.Pool(processes=PERF_CORES)` — validation
(open image, check dims, check caption) is pure CPU and parallelises identically to shard
writing. Process shards in parallel; each worker validates and rewrites its assigned shards:

```python
PERF_CORES = get_perf_cores()  # 8 on M1 Max

def filter_shard(shard_path):
    """Validate all records in one shard; return path + keep-count."""
    kept = []
    with tarfile.open(shard_path) as tar:
        for record in iter_records(tar):
            if is_valid(record):   # dim check + caption check + decode check
                kept.append(record)
    rewrite_shard(shard_path, kept)
    return len(kept)

with multiprocessing.Pool(processes=PERF_CORES) as pool:
    counts = pool.map(filter_shard, glob.glob("data/train_shards/*.tar"))
print(f"Kept {sum(counts)} records across {len(counts)} shards")
```

Expected loss: ~3–5% → ~1.55M usable unique images, 310 shards.

### 2.7 Pre-compute frozen forward passes

Three encoders run during every training step with frozen weights: SigLIP, VAE, and Qwen3.
At 2.15M images across ~430 shards, 105K steps × batch=2 covers only ~10% of the dataset
(~1 visit per image on the selected shards). Pre-computing and caching encoder outputs
eliminates repeated frozen forward passes and is especially valuable since VAE dominates
at ~0.28 s/image. Cache all three if storage permits. See `train/TRAINING.md` for full
coverage analysis and per-chunk shard sizing guidance.

**Combined savings if all three are pre-computed:**
```
SigLIP  ~50ms/step  × 120K = 1.7h  (420 GB at 4-bit)
Qwen3  ~200ms/step  × 120K = 6.7h  (143 GB at 4-bit)
VAE    ~180ms/step  × 120K = 6.0h  (198 GB at int8)
─────────────────────────────────────────────────────
Total  ~430ms/step saved   = 14.4h  (761 GB total)
```

Step time drops from ~1.9s to ~1.47s — a further 22% reduction beyond what is already
in the plan. Decide per-encoder based on available storage.

**Storage decision guide:**

| Encoder | 4-bit / int8 storage | Step saving | Pre-compute? |
|---|---|---|---|
| Qwen3 text embeds | ~143 GB (4-bit) | ~200ms | Yes if 143GB free |
| VAE latents | ~198 GB (int8) | ~180ms | Yes if 198GB free |
| SigLIP features | ~420 GB (4-bit) | ~50ms | Only if >420GB free after above |

Qwen3 has the best saving-per-GB ratio (~1.4ms/GB) and should be pre-computed first.

---

**Qwen3 text embeddings — pre-compute at 4-bit (~143 GB):**

```python
# scripts/precompute_qwen3.py
# Run once after Phase 2 sharding; takes ~8 hours on M1 Max (1.9M images × 200ms)
import mlx.core as mx
import numpy as np, os, glob

def quantize_4bit_seq(arr):
    """Per-token absmax 4-bit quantisation. arr: [seq, 7680]"""
    scale = np.abs(arr).max(axis=-1, keepdims=True) / 7.0
    q = np.clip(np.round(arr / (scale + 1e-8)), -8, 7).astype(np.int8)
    q_packed = ((q[:, 0::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)).astype(np.uint8)
    return q_packed, scale.astype(np.float16)

# Process shard by shard — avoids holding all embeddings in memory
for shard in sorted(glob.glob("data/train_shards/*.tar")):
    for rec in iter_shard(shard):
        emb = text_encoder(rec["caption"])        # [seq, 7680] float16
        q_packed, scale = quantize_4bit_seq(emb.astype(np.float32))
        np.savez(f"data/qwen3_q4/{rec['id']}.npz", q=q_packed, scale=scale)
```

At training time, dequantise in the prefetch thread (CPU-trivial):

```python
def load_text_embed(rec_id):
    d = np.load(f"data/qwen3_q4/{rec_id}.npz")
    lo = (d["q"] & 0x0F).astype(np.int8)
    hi = ((d["q"] >> 4) & 0x0F).astype(np.int8)
    q  = np.empty((d["q"].shape[0], d["q"].shape[1] * 2), dtype=np.int8)
    q[:, 0::2] = lo; q[:, 1::2] = hi
    return (q.astype(np.float32) * d["scale"]).astype(np.float16)
```

---

**VAE latents — pre-compute at int8 (~198 GB):**

```python
# scripts/precompute_vae.py
# ~6 hours on M1 Max (1.55M images × 180ms encode)
import mlx.core as mx
import numpy as np, os

def quantize_int8(arr):
    """Per-channel absmax int8 quantisation. arr: [32, 64, 64]"""
    scale = np.abs(arr).max(axis=(1, 2), keepdims=True) / 127.0
    q = np.clip(np.round(arr / (scale + 1e-8)), -128, 127).astype(np.int8)
    return q, scale.astype(np.float16)

for shard in sorted(glob.glob("data/train_shards/*.tar")):
    for rec in iter_shard(shard):
        img = preprocess_vae(rec["jpg"])           # resize, normalise
        latent = vae.encode(img[None])[0]          # [32, H/8, W/8]
        q, scale = quantize_int8(latent.numpy())
        np.savez(f"data/vae_int8/{rec['id']}.npz", q=q, scale=scale)
```

Dequantise in prefetch thread:

```python
def load_vae_latent(rec_id):
    d = np.load(f"data/vae_int8/{rec_id}.npz")
    return (d["q"].astype(np.float32) * d["scale"]).astype(np.float16)
```

---

**SigLIP embeddings — pre-compute if storage permits (420 GB at 4-bit):**

At full BF16, SigLIP features are 1.9M × 729 × 1152 × 2 bytes = 3.2 TB — not practical.
But 4-bit quantised SigLIP features drop to ~420 GB. Saves ~50ms/step = 1.7h Stage 1.
Only pre-compute if 420 GB remains after Qwen3 + VAE pre-compute storage is allocated.

**SigLIP embeddings — pre-compute if storage permits:**

At full BF16, SigLIP features are 1.9M × 729 × 1152 × 2 bytes = 3.2 TB — not practical.
But 4-bit quantised SigLIP features drop to ~400 GB, which fits on a TB4 external SSD
with space to spare. If 400 GB is available, pre-computing eliminates the ~50ms/step
frozen GPU forward pass during training:

```
50ms/step × 120,000 steps = 6,000,000ms ≈ 1.7 hours saved during Stage 1 alone
```

Pre-compute with 4-bit quantisation using MLX:

```python
import mlx.core as mx
import numpy as np
from mlx_vlm import load  # or load SigLIP directly

PERF_CORES = get_perf_cores()

def quantize_4bit(arr):
    """Per-token absmax 4-bit quantisation into uint8 pairs + scale."""
    # arr: [729, 1152] float32
    scale = np.abs(arr).max(axis=-1, keepdims=True) / 7.0  # 4-bit signed range -8..7
    q = np.clip(np.round(arr / (scale + 1e-8)), -8, 7).astype(np.int8)
    # Pack pairs of 4-bit values into uint8
    q_packed = ((q[:, 0::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)).astype(np.uint8)
    return q_packed, scale.astype(np.float16)

def embed_shard(shard_path, siglip_model, output_dir):
    records = load_shard(shard_path)
    for rec in records:
        img = preprocess_siglip(rec["jpg"])          # resize to 384×384, normalise
        feats = siglip_model(img[None])               # [1, 729, 1152]
        q_packed, scale = quantize_4bit(feats[0].numpy())
        np.savez_compressed(
            os.path.join(output_dir, rec["id"] + ".npz"),
            q=q_packed, scale=scale
        )

# Parallelise across shards — SigLIP GPU-bound per batch, CPU-bound across shards
# Use 2 processes only: GPU is shared between them
with multiprocessing.Pool(processes=2) as pool:
    pool.starmap(embed_shard, [(s, model, "data/siglip_q4") for s in shards])
```

Storage estimate at 4-bit: 1.9M × (729 × 576 bytes packed + 729 × 2 bytes scale) ≈ **~420 GB**

At training time, dequantise in the prefetch thread using the same `dequantize_4bit`
function as Qwen3 above (identical bit-packing format).

---

## Phase 3: Training Code
**Duration: 7–14 days active development**

### 3.1 IP-Adapter architecture in MLX

```python
# ip_adapter_klein.py
import mlx.core as mx
import mlx.nn as nn

class IPAdapterKlein(nn.Module):
    def __init__(self, num_tokens=128, hidden_size=3072, siglip_dim=1152,
                 num_blocks=25):
        # Perceiver Resampler: 128 learned queries cross-attending to 729 SigLIP tokens
        self.query_tokens = mx.random.normal((num_tokens, hidden_size)) * 0.02
        self.cross_attn   = nn.MultiHeadAttention(hidden_size, num_heads=24,
                                                   key_input_dims=siglip_dim)
        self.norm         = nn.LayerNorm(hidden_size)

        # Per-block projections — stacked for batched GEMM (see section 3.2)
        # Shape: [num_blocks, hidden_size, hidden_size]
        self.to_k_ip_stacked = mx.random.normal(
            (num_blocks, hidden_size, hidden_size)) * 0.02
        self.to_v_ip_stacked = mx.random.normal(
            (num_blocks, hidden_size, hidden_size)) * 0.02
        self.scale = mx.ones((num_blocks,))

    def get_image_embeds(self, siglip_features):
        # siglip_features: [B, 729, 1152]
        q = mx.broadcast_to(self.query_tokens[None],
                             (siglip_features.shape[0],) + self.query_tokens.shape)
        out = self.cross_attn(q, siglip_features, siglip_features)
        return self.norm(out)  # [B, 128, 3072]

    def get_kv_all(self, ip_embeds):
        # ip_embeds: [B, 128, 3072]
        # Batched GEMM: [B, 128, 3072] x [25, 3072, 3072] → [B, 25, 128, 3072]
        # 25 block projections in 2 Metal dispatches instead of 50
        k = mx.einsum('btd,nde->bnte', ip_embeds, self.to_k_ip_stacked)
        v = mx.einsum('btd,nde->bnte', ip_embeds, self.to_v_ip_stacked)
        return k, v  # [B, 25, 128, 3072] each
```

### 3.2 Metal optimisations in the forward pass

**Batched K/V projection** — replaces 50 separate GEMMs (25 blocks × K + V) with
2 batched einsum calls dispatched as single Metal kernels:

```python
k_ip_all, v_ip_all = ip_adapter.get_kv_all(ip_embeds)
# Shape: [B, 25, 128, 3072] — all blocks pre-computed before the transformer loop
```

**Gradient checkpointing on transformer blocks** — prevents storing 25 blocks of
activations simultaneously (saves ~600MB at 512px):

```python
from mlx_lm.tuner.trainer import grad_checkpoint
from mflux.models.flux2 import Flux2TransformerBlock, Flux2SingleTransformerBlock

# Patch both block types — applies to all instances
grad_checkpoint(Flux2TransformerBlock)
grad_checkpoint(Flux2SingleTransformerBlock)
```

**Fused noise scheduler** — one Metal kernel for both `add_noise` and `get_velocity`:

```python
_fused_noise_kernel = mx.fast.metal_kernel(
    name="fused_flow_noise",
    input_names=["latent", "noise", "alpha", "sigma"],
    output_names=["noisy", "target"],
    source="""
        uint i = thread_position_in_grid.x;
        float l = latent[i], n = noise[i];
        float a = alpha[0],  s = sigma[0];
        noisy[i]  = a * l + s * n;
        target[i] = a * n - s * l;
    """,
)

def fused_flow_noise(latent, noise, alpha_t, sigma_t):
    return _fused_noise_kernel(
        inputs=[latent, noise,
                mx.array([alpha_t], dtype=mx.float32),
                mx.array([sigma_t], dtype=mx.float32)],
        grid=(latent.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[latent.shape, latent.shape],
        output_dtypes=[latent.dtype, latent.dtype],
    )
```

### 3.3 Warmstart the Perceiver Resampler

Even in a full Path 2 training run, borrowing the Perceiver Resampler weights from the
Flux.1-dev IP-Adapter gives a meaningful initialization. Both models share hidden_size=3072
so the projection dimensions match. Only the per-block K/V projections learn from scratch.
Estimated effect: ~10–15K fewer training steps needed for convergence:

```python
from safetensors import safe_open

pretrained_path = "FLUX.1-dev-IP-Adapter.safetensors"  # InstantX, 5.3GB
with safe_open(pretrained_path, framework="numpy") as f:
    resampler_keys = [k for k in f.keys() if k.startswith("image_proj.")]
    resampler_weights = {k.replace("image_proj.", ""): mx.array(f.get_tensor(k))
                         for k in resampler_keys}

ip_adapter.load_resampler_weights(resampler_weights)
# Only to_k_ip_stacked and to_v_ip_stacked remain randomly initialised
```

### 3.4 Async prefetch buffer

```python
import queue, threading

def make_prefetch_loader(shard_paths, batch_size=2, sample_buffer=6):
    """
    Two-level prefetch pipeline:
      Level 1 (shard thread)  — pre-decompresses the next tar into memory while
                                current shard is being consumed. Eliminates the
                                0.5–2s pause at each of the 310 shard boundaries.
      Level 2 (sample thread) — decodes JPEG + dequantises pre-computed embeds +
                                builds batches. Uses turbojpeg for 2–4× faster decode.
    GPU (main thread)         — training step; never sees shard boundary stalls.
    """
    import queue, threading, tarfile, io
    from turbojpeg import TurboJPEG
    _jpeg = TurboJPEG()

    shard_q  = queue.Queue(maxsize=2)    # level 1: pre-decompressed shard contents
    sample_q = queue.Queue(maxsize=sample_buffer)  # level 2: decoded batches

    def shard_loader():
        for path in shard_paths:
            with tarfile.open(path) as tar:
                contents = {m.name: tar.extractfile(m).read()
                            for m in tar.getmembers() if m.isfile()}
            shard_q.put(contents)
        shard_q.put(None)

    def sample_decoder():
        while (shard := shard_q.get()) is not None:
            records = list(iter_shard_contents(shard))   # parse .jpg/.txt pairs
            for i in range(0, len(records) - batch_size + 1, batch_size):
                batch_recs = records[i:i+batch_size]
                imgs   = [_jpeg.decode(r["jpg"]) for r in batch_recs]
                # Load pre-computed embeds if available, else return None for runtime encode
                texts  = [load_text_embed(r["id"])  for r in batch_recs]
                vaes   = [load_vae_latent(r["id"])  for r in batch_recs]
                sample_q.put((imgs, texts, vaes))
        sample_q.put(None)

    threading.Thread(target=shard_loader,   daemon=True).start()
    threading.Thread(target=sample_decoder, daemon=True).start()

    while (item := sample_q.get()) is not None:
        yield item
```

### 3.5 Block injection with pre-computed K/V

Subclass `Flux2TransformerBlock.__call__` to accept and inject IP K/V:

```python
def patched_double_block(self, hidden_states, encoder_hidden_states,
                         temb_mod, temb_mod_txt, rotary_emb,
                         k_ip=None, v_ip=None, ip_scale=1.0):
    # ... standard block forward ...
    if k_ip is not None:
        # img_q already computed inside standard block; pass via closure
        ip_attn = mx.fast.scaled_dot_product_attention(
            img_q, k_ip, v_ip, scale=(head_dim ** -0.5)
        )
        hidden_states = hidden_states + ip_scale * ip_attn.reshape(*hidden_states.shape)
    return encoder_hidden_states, hidden_states
```

The transformer loop passes `k_ip=k_ip_all[:, block_idx]`, `v_ip=v_ip_all[:, block_idx]`
and `ip_scale=ip_adapter.scale[block_idx]` at each block iteration.

### 3.6 Training step in MLX

```python
import mlx.core as mx
import mlx.nn as nn
import random

def make_train_step(transformer, siglip, vae, text_encoder, ip_adapter, optimizer):
    # Freeze all base models — only ip_adapter has trainable_parameters()
    transformer.freeze()
    siglip.freeze()
    vae.freeze()
    text_encoder.freeze()

    def loss_fn(ip_adapter_params, batch):
        images, captions = batch
        ip_adapter.update(ip_adapter_params)

        # Frozen inference — no gradient storage for base models
        siglip_feats = siglip(images)           # [1, 729, 1152]
        text_embeds  = text_encoder(captions)   # [1, seq, 7680]
        latents      = vae.encode(images)       # [1, 32, H/8, W/8]

        # Null conditioning dropout
        if random.random() < 0.30:
            ip_embeds = mx.zeros((1, 128, 3072), dtype=mx.bfloat16)
        else:
            ip_embeds = ip_adapter.get_image_embeds(siglip_feats)

        # Pre-compute all 25 block K/V projections (2 Metal dispatches total)
        k_ip_all, v_ip_all = ip_adapter.get_kv_all(ip_embeds)

        t      = mx.random.randint(0, 1000, shape=(1,))
        noise  = mx.random.normal(latents.shape, dtype=latents.dtype)
        alpha_t, sigma_t = get_schedule_values(t)
        noisy, target = fused_flow_noise(latents, noise, alpha_t, sigma_t)

        # Forward through checkpointed Klein 4B with IP injection
        pred = transformer(noisy, text_embeds, t,
                           k_ip_all=k_ip_all, v_ip_all=v_ip_all,
                           ip_scale=ip_adapter.scale)

        return mx.mean((pred - target) ** 2)

    loss_and_grad = nn.value_and_grad(ip_adapter, loss_fn)

    def train_step(batch):
        loss, grads = loss_and_grad(ip_adapter.trainable_parameters(), batch)
        grads, _ = mx.clip_by_global_norm(grads, max_norm=1.0)
        optimizer.update(ip_adapter, grads)
        # async_eval: start next step's graph construction on CPU while GPU finishes
        # this step. Overlaps ~5–10ms Python overhead with GPU execution.
        mx.async_eval(ip_adapter.parameters(), optimizer.state, loss)
        return loss

    return train_step
```

### 3.7 mx.compile() on the loss+backward function

Compiles element-wise-heavy ops (loss, noise scheduler, dropout masking) into fused Metal
kernels. Saves ~150ms/step. Three rules to make it work:

```python
# 1. Include mx.random.state so noise sampling isn't baked as a constant
state = [ip_adapter.state, optimizer.state, mx.random.state]
compiled_loss_and_grad = mx.compile(loss_and_grad_fn, inputs=state, outputs=state)

# 2. Keep mx.eval() OUTSIDE compiled region — it triggers the Metal dispatch
def train_step(batch):
    loss, grads = compiled_loss_and_grad(ip_adapter.trainable_parameters(), batch)
    grads, _ = mx.clip_by_global_norm(grads, max_norm=1.0)
    optimizer.update(ip_adapter, grads)
    mx.eval(ip_adapter.parameters(), optimizer.state, loss)
    return loss.item()

# 3. Python-level conditionals must be outside compiled region.
# Pass null-dropout decision as mx.array, not a Python bool:
use_null_image = mx.array(random.random() < 0.30)
use_null_text  = mx.array(random.random() < 0.10)
# Inside compiled fn: mx.where(use_null_image, zero_embeds, ip_embeds)
```

### 3.8 BF16 optimizer state and batch_size=2

Keep ip_adapter weights in bfloat16 throughout — MLX AdamW stores optimizer state
in the same dtype as the parameters. This halves optimizer state from ~2GB to ~1GB,
and combined with gradient checkpointing frees enough memory for batch_size=2:

```
batch_size=2 memory: ~17GB base + ~3GB extra activations = ~20GB ✓ in 32GB
```

Batch_size=2 improves gradient signal and partially compensates for gradient noise at
small batch sizes.

### 3.9 Multi-resolution bucketing

Train across multiple aspect ratios simultaneously instead of fixed 512×512. This makes
`--sref` work well for portrait, landscape, and square reference images:

```python
BUCKETS = [
    (512, 512), (512, 768), (768, 512),
    (640, 640), (512, 896), (896, 512),
]
# Each training step: sample a bucket, load images matching that aspect ratio.
# WebDataset shard contains per-image dimension metadata for bucket assignment.
```

Do NOT use 256px progressive pretraining — Flux's patchification is designed for ≥512px
and 256px produces degenerate outputs.

### 3.10 GPU image augmentation

Run random flip and random crop in MLX after loading (eliminates CPU→GPU copy):

```python
def augment_mlx(img):
    # Random horizontal flip
    if mx.random.uniform() > 0.5:
        img = mx.flip(img, axis=2)
    # Random crop: pad source to 544px, crop to 512px
    h = int(mx.random.randint(0, 32).item())
    w = int(mx.random.randint(0, 32).item())
    return img[:, h:h+512, w:w+512, :]
```

### 3.11 Optimizer and EMA

```python
import mlx.optimizers as optim

# Cosine schedule with linear warmup
lr_schedule = optim.join_schedules([
    optim.linear_schedule(1e-6, 1e-4, steps=2000),
    optim.cosine_decay(1e-4, decay_steps=118000, eta_min=1e-6),
], [2000])

optimizer = optim.AdamW(
    learning_rate=lr_schedule,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# EMA: manual exponential moving average of adapter parameters
ema_params = ip_adapter.parameters()  # copy at init

def update_ema(ema, model, decay=0.9999):
    return mx.tree_map(lambda e, m: decay * e + (1 - decay) * m,
                       ema, model.parameters())

# Update EMA every 10 steps, not every step.
# At decay=0.9999 the EMA moves <0.1% per step — every-10-step update is
# indistinguishable in output quality and saves ~7ms × 120K / 10 ≈ 23 minutes.
EMA_UPDATE_INTERVAL = 10
```

### 3.12 Checkpoint strategy

Save every 2000 steps (TB4 reliability mitigation). Keep last 5 checkpoints + best
(lowest eval loss). Each checkpoint: ~1 GB adapter + ~1 GB EMA = ~2 GB per save.

**Save on a background thread** — writing 2 GB to disk takes several seconds and would
stall the training loop. Copy weights to CPU-side numpy first, then write asynchronously:

```python
import threading
from safetensors.numpy import save_file

def save_checkpoint_async(ip_adapter, ema_params, step, path):
    # Copy to numpy on the main thread (fast — unified memory, no DMA transfer)
    # mx.eval() first to ensure all pending async_eval ops are complete
    mx.eval(ip_adapter.parameters())
    state = {k: np.array(v) for k, v in ip_adapter.parameters().items()}
    ema   = {f"ema.{k}": np.array(v) for k, v in ema_params.items()}
    payload = {**state, **ema}
    ckpt_path = f"{path}/step_{step:07d}.safetensors"

    # Write to disk in background — GPU continues training
    def _write():
        save_file(payload, ckpt_path)
    threading.Thread(target=_write, daemon=True).start()

def purge_old_checkpoints(path, keep=5):
    checkpoints = sorted(glob.glob(f"{path}/step_*.safetensors"))
    for ckpt in checkpoints[:-keep]:
        os.remove(ckpt)
```

The `mx.eval()` call before copying ensures `async_eval` from recent training steps has
completed before reading weights. The disk write itself happens entirely off the hot path.

---

## Phase 4: Stage 1 Training — 512px
**Duration: ~5.6 days continuous**

```
Resolution:              512 × 512
Steps:                   120,000
Batch size:              1 (gradient accumulation × 4 optional — see note)
LR:                      1e-4, cosine decay to 1e-6
LR warmup:               2000 steps (linear)
Optimizer:               MLX AdamW (β1=0.9, β2=0.999, wd=0.01)
Gradient clipping:       max_norm=1.0 via mx.clip_by_global_norm
EMA decay:               0.9999
Null image dropout:      30%
Null text dropout:       10%
Checkpoint interval:     every 5000 steps
Eval samples:            10 fixed (prompt, reference) pairs at each checkpoint
Gradient checkpointing:  enabled on all 25 transformer blocks
```

Run unattended:

```bash
nohup python train_ip_adapter.py \
  --data_path data/train_shards \
  --output_dir checkpoints/stage1 \
  --resolution 512 \
  --max_steps 120000 \
  > logs/train_stage1.log 2>&1 &
```

**Timeline (scenario B — Qwen3 + VAE pre-computed, recommended):**
105,000 steps × ~1.52s/step ≈ 159,600 seconds ≈ **~1.84 days**

**Timeline (scenario A — all three encoders pre-computed):**
105,000 steps × ~1.47s/step ≈ 154,350 seconds ≈ **~1.79 days**

**Timeline (scenario C — no pre-compute):**
105,000 steps × ~1.9s/step ≈ 199,500 seconds ≈ **~2.31 days**

Steps reduced by ~15K from Perceiver warmstart. Scenario B is recommended: 99% of
savings for 45% of the pre-compute storage cost (341 GB vs 761 GB).

Step time breakdown — scenario B (Qwen3 + VAE pre-computed, SigLIP at runtime):

| Component | Time | Notes |
|---|---|---|
| SigLIP forward (frozen, no grad) | ~50ms | Runtime (skip if scenario A) |
| Qwen3 dequant + load from cache | ~0ms | Hidden in two-level prefetch |
| VAE latent load + dequant | ~0ms | Hidden in two-level prefetch |
| Batched K/V projection (2 dispatches) | ~50ms | |
| Flux Klein 4B forward + IP inject (checkpointed, × 2) | ~1.0s | Metal-bound |
| Backward through adapter only | ~0.4s | |
| AdamW (BF16 state) + EMA (amortised ÷10) | ~64ms | EMA every 10 steps |
| async_eval sync | ~40ms | Overlapped with next step CPU work |
| **Total** | **~1.60s** | |
| (two-level prefetch hides all I/O + dequant) | **~−0.08s net** | |
| **Effective step time** | **~1.52s** | |

Scenario A removes the 50ms SigLIP row → **~1.47s/step**.

### Quality milestones

| Checkpoint | Expected quality |
|---|---|
| Step 5K | Loss decreasing; generated images show weak, blurry style influence |
| Step 20K | Recognizable style transfer begins; composition still bleeds |
| Step 50K | Usable style transfer in single-block injection mode |
| Step 80K | Refined; content/style better separated |
| Step 120K | Stage 1 complete; evaluate before starting Stage 2 |

---

## Phase 5: Stage 2 Training — 768px (optional)
**Duration: ~1.8 days**

Higher resolution sharpens texture and fine detail transfer.
Start from Stage 1 best checkpoint. Gradient checkpointing makes 768px feasible in 32GB.

```
Resolution:   768 × 768
Steps:        20,000 additional
LR:           1e-5 (10× lower than Stage 1)
Batch size:   1
```

**Timeline (scenario B — Qwen3 + VAE pre-computed):**
20,000 × ~4.55s/step ≈ 91,000 seconds ≈ **~1.05 days**

**Timeline (scenario C — no pre-compute):**
20,000 × ~5.0s/step ≈ 100,000 seconds ≈ **~1.16 days**

At 768px, the Flux forward pass dominates (~3.5s/step) and the frozen encoder savings
(~430ms) are proportionally smaller than at 512px. Gradient checkpointing caps the
memory delta; batch_size remains 1.

---

## Phase 5b: Incremental Training on Additional Data Chunks

**Context:** The full JourneyDB dataset (~3 TB) and other large sources exceed the 2 TB SSD.
Training proceeds in chunks: download a subset, train, delete raw data, download next subset, resume.

### Approach

Each chunk follows this cycle:

```
1. Download raw chunk N to SSD          (~150–300 GB)
2. build_shards + filter_shards         (merge into shards/)
3. precompute_qwen3 + precompute_vae    (extend precomputed/)
4. Resume training from last checkpoint (~20,000–40,000 steps per chunk)
5. Delete raw chunk N from SSD          (free ~150–300 GB)
6. Repeat with chunk N+1
```

Resume training with lower LR (prevent large weight updates on already-learned content):

```bash
python train/train_ip_adapter.py \
  --config train/configs/stage1_512px.yaml \
  --resume_from checkpoints/stage1/step_XXXXX.safetensors \
  --max_steps 40000 \
  --lr 3e-5 \
  --output_dir checkpoints/stage1_chunk2
```

LR schedule per chunk:
| Chunk | LR | Rationale |
|---|---|---|
| 1 (initial) | 1e-4 | Full learning from warmstart |
| 2 | 3e-5 | Consolidate without overwriting |
| 3+ | 1e-5 | Fine adjustment only |

### Distribution Shift Mitigation

**Risk:** Training exclusively on JourneyDB (synthetic Midjourney style) in later chunks
causes the model to drift — it forgets the aesthetic diversity from LAION/WikiArt/COYO.

**Mitigations:**

1. **Anchor replay set** — keep a fixed 10,000-sample subset from chunk 1 (diverse LAION +
   WikiArt) on local disk (~1.5 GB). Mix 20% anchor samples into every subsequent chunk's
   shard set. `build_shards.py --anchor train/data/anchor_shards/` handles this automatically
   (see below).

2. **Mixed chunk composition** — never train a chunk that is >70% from a single source.
   If chunk 2 is JourneyDB 050-099, mix in COYO or WikiArt shards to keep source diversity.

3. **EMA weights** — use EMA checkpoint (not raw weights) as the base for each new chunk.
   EMA smooths out within-chunk drift before it compounds across chunks.

4. **Loss monitoring** — if anchor-set validation loss increases >15% relative to the
   previous chunk's final loss, reduce LR further or increase anchor mixing ratio.

### Anchor Set Creation

Run once after Phase 2, before starting Phase 4:

```bash
python train/scripts/create_anchor_set.py \
  --shards train/data/shards \
  --output train/data/anchor_shards \
  --n 10000 \
  --sources laion wikiart coyo    # exclude journeydb from anchor
```

The anchor set lives on local disk permanently (~1.5 GB) and is never deleted between chunks.

### Storage Budget Per Chunk

| Item | Size | Disk | Delete after chunk? |
|---|---|---|---|
| Raw images (one chunk) | ~150–300 GB | SSD | Yes |
| Chunk shards | ~80–150 GB | SSD | Yes (after precompute) |
| Precomputed embeds | ~50–100 GB | Local | Yes (after training) |
| Anchor shards | ~1.5 GB | Local | Never |
| Checkpoints (EMA) | ~500 MB | Local | Keep last 2 only |

Peak SSD usage per cycle: ~600 GB (well within 1.8 TB).

### Recommended JourneyDB Chunking

| Chunk | Files | Approx images | LR |
|---|---|---|---|
| 1 (current) | 000–049 | ~1M | 1e-4 |
| 2 | 050–099 | ~1M | 3e-5 |
| 3 | 100–149 | ~1M | 1e-5 |
| 4 | 150–201 | ~1M | 1e-5 |

---

## Phase 6: Export and Integrate into iris.c
**Duration: 2–4 weeks active development**

### 6.1 Export adapter weights

```python
from safetensors.torch import save_file

# Use EMA weights for best quality
with ema.average_parameters():
    state = {k: v.cpu().bfloat16() for k, v in ip_adapter.state_dict().items()}
save_file(state, "klein_ip_adapter_bf16.safetensors")
```

### 6.2 New file: `iris_siglip.c`

SigLIP SO400M is a standard Vision Transformer (simpler than Qwen3 — no GQA, no RoPE,
standard absolute position embedding):

- 27×27 = 729 patch embeddings from 384×384 input
- 27 transformer layers
- 16 attention heads, head_dim=72 (1152/16)
- Implementation complexity: ~1500 lines C, similar scale to `iris_qwen3.c`
- Output: `[729, 1152]` patch token tensor

### 6.3 Modifications to `iris_transformer_flux.c`

Add IP-Adapter injection to both `double_block_forward` and `single_block_forward`.
Each block receives an optional `ip_embeds [128, 3072]` parameter:

1. Compute `k_ip = ip_to_k[block_idx] @ ip_embeds`
2. Compute `v_ip = ip_to_v[block_idx] @ ip_embeds`
3. Compute attention: `img_q` attends to `k_ip`, `v_ip`
4. Scale and add to standard attention output

Approximately 40 lines of new code per block type, plus the cross-attention kernel.

### 6.4 New function: `iris_ipadapter_forward()`

```c
// Returns ip_embeds [128 * hidden] to thread through the denoising loop.
// Called once per generation before the sampling loop.
float *iris_ipadapter_forward(iris_ipadapter_t *ipa,
                               const iris_image *ref_image);
```

Internal steps:
1. Resize reference image to 384×384
2. Run SigLIP encoder → `[729, 1152]`
3. Run Perceiver Resampler → `[128, 3072]`
4. Return `ip_embeds` to caller

### 6.5 CLI additions to `main.c`

```
--sref PATH           Path to style reference image
--sref-scale FLOAT    Style influence 0.0–1.0 (default 0.7)
--sref-mode MODE      'style'   = inject single-stream blocks only (default)
                      'content' = inject all 25 blocks
```

---

## Optimisations Summary

### Training loop

| Optimisation | Mechanism | Step saving |
|---|---|---|
| MLX vs PyTorch MPS | Lazy eval, Metal command batching | ~1.5s |
| Batched K/V projection | `mx.einsum` — 50 GEMMs → 2 dispatches | ~0.3s |
| Batched SDPA | 25 → 1 Metal SDPA call | ~0.2s |
| Gradient checkpointing | `mx.checkpoint` on all 25 blocks | −0.3s (recompute cost) |
| Fused noise scheduler | Custom `mx.fast.metal_kernel` | ~0.05s |
| `mx.compile()` | Fuses element-wise ops | ~0.15s |
| Async prefetch | Overlap I/O with GPU | ~0.04s (hidden) |
| BF16 optimizer state → batch_size=2 | Halved optim memory | ~0.1s net |
| Warmstart Perceiver Resampler | Better init → fewer steps | ~15K steps |
| GPU augmentation | MLX flip/crop ops | ~0.03s |
| Pre-computed SigLIP (4-bit, ~420 GB) | Eliminate frozen GPU fwd pass | ~0.05s if storage permits |
| Pre-computed Qwen3 (4-bit, ~143 GB) | Eliminate frozen text encode | ~0.20s if storage permits |
| Pre-computed VAE latents (int8, ~198 GB) | Eliminate frozen VAE encode | ~0.18s if storage permits |
| Two-level prefetch (shard + sample) | Eliminate shard-boundary stalls | ~0.5–2s per boundary hidden |
| turbojpeg JPEG decode | 2–4× faster decode in prefetch + shard write | reduces prefetch thread time |
| `mx.async_eval` | Pipeline next step CPU work with GPU | ~5–10ms/step |
| Background checkpoint save | Non-blocking 2 GB disk write | ~few seconds per save |
| EMA update every 10 steps | Reduce EMA compute by 10× | ~23 min total |

**Result: ~1.9s/step vs ~4.0s/step baseline (52% faster); ~15K fewer steps needed**
**With all three pre-computed encoders: ~1.47s/step (further 22% reduction)**

### Data preparation

| Optimisation | Time saving |
|---|---|
| Parallel downloads (all 3 simultaneous) | ~1 week vs ~2 weeks |
| Pre-filter LAION parquet | ~30% faster img2dataset |
| CLIP deduplication | +1.5 hours; removes 200–400K wasted training samples |
| Re-captioning short captions | +2 days; measurably better style alignment |
| Larger shard size (5K vs 2K) | ~60% fewer shard transitions during training |
| Performance-core targeting (`hw.perflevel0.logicalcpu`) | ~20–25% faster vs `os.cpu_count()` on M1 Max |
| Parallel shard writing (`multiprocessing.Pool`, no shared state) | ~6× vs single-threaded JPEG decode + tar-write |
| Parallel filter pass (one process per shard, all perf cores) | ~8× vs single-threaded validation pass |
| zstd level 1 vs default level 3 | ~40% faster shard-write; negligible size increase |
| turbojpeg in shard write + filter pass | 2–4× faster JPEG decode at every image touch |
| `caffeinate -i -d` wrapping full pipeline | prevents sleep corruption over 2–3 day run |
| Pre-compute Qwen3 4-bit (~143 GB) | −200ms/step → ~6.7h saved in Stage 1 |
| Pre-compute VAE latents int8 (~198 GB) | −180ms/step → ~6.0h saved in Stage 1 |
| Pre-compute SigLIP 4-bit (~420 GB) | −50ms/step → ~1.7h saved in Stage 1 |

Custom Metal kernels with backward pass support (`mx.fast.metal_kernel` + `.vjp`) are
available if profiling identifies further bottlenecks. The gradient accumulation kernel
(atomic in-place grad add) is the next candidate if effective batch_size > 2 is wanted.

### Language choice: Python + MLX vs C++ or Rust

**Conclusion: Python + MLX + `mx.compile()` is the confirmed optimal choice. Do not rewrite
in C++ or Rust.**

Evaluated after a full performance audit:

| Language/framework | Metal execution | Python overhead | Practical step time |
|---|---|---|---|
| Python + MLX + `mx.compile()` | ~1.85s | ~5ms (tree-flatten) | **~1.9s** |
| MLX C++ API (same Metal) | ~1.85s | ~0ms | ~1.85s |
| Candle (Rust, Metal backend) | ~2.0s (no hand-tuned GEMM) | ~0ms | ~2.0s |
| burn (Rust, wgpu-Metal) | ~2.5s+ (wgpu abstraction) | ~0ms | ~2.5s+ |
| PyTorch MPS | ~3.8s | ~5ms | ~4.0s |

**Why C++ saves nothing:**
The MLX C++ API (`mlx/transforms.h`) fully supports autograd — `vjp()`, `value_and_grad()`,
`grad()`, `checkpoint()`, `compile()` — with identical Metal dispatch paths as Python.
After `mx.compile()`, Python overhead is ~1–5ms per step from MLX's internal array
tree-flattening. Against a ~1.9s Metal-bound step, that is <0.3% of total time.
Rewriting Phase 3 in C++ to eliminate this 5ms would be engineering waste.

**Why Candle/burn are slower:**
- Candle has a Metal backend (custom `.metal` kernels) but lacks the hand-tuned GEMM
  kernels that MLX provides; transformer GEMMs on M1 Max are noticeably slower.
- burn abstracts Metal through wgpu, preventing direct MPSGraph use. No native flash
  attention. Not competitive for large matrix workloads.

**Correct use of custom Metal in this plan:**
The `mx.fast.metal_kernel()` API (Python-callable) handles any remaining bottlenecks
by compiling custom Metal with full autograd support. This covers the fused noise
scheduler kernel and any future gradient accumulation needs. No separate C++ layer needed.

---

## Total Timeline Summary

| Phase | Duration | Type | vs original |
|---|---|---|---|
| 0. Infrastructure setup | 2–3 days | Active dev | — |
| 1. Dataset download | 7–14 days | Unattended | — |
| 2. Pre-processing (shard + filter) | 1.5–2 days | Unattended | −0.5–1 day (parallelism + turbojpeg) |
| 2b. Pre-compute encoders | 1–2 days | Unattended | new; overlaps Phase 3 |
| 3. Training code | 7–14 days | Active dev | — |
| 4. Stage 1 training (512px) | **~1.84 days** | Unattended | −0.47 days (−11 hours) |
| 5. Stage 2 training (768px) | **~1.05 days** | Unattended | −0.11 days (−3 hours) |
| 6. C inference integration | 14–28 days | Active dev | — |
| **Total elapsed** | **~6–8 weeks** | | unchanged |
| **Total active dev time** | **~4–6 weeks** | | unchanged |
| **Total unattended training** | **~2.89 days** | | vs ~3.47 days (−14 hours) |

Timings above use scenario B (Qwen3 + VAE pre-computed, ~341 GB). Scenario A (all three
encoders) saves a further ~3 hours at the cost of 420 GB additional storage for SigLIP.

**Why total elapsed is unchanged:** the critical path is active dev (Phases 0, 3, 6).
All unattended phases run in parallel with active dev — Phase 2b pre-compute runs
overnight during Phase 3 development. The 14 hours saved in training are real but do
not compress the calendar unless development finishes ahead of schedule.

**Where the savings matter most:** a training restart from a mid-run checkpoint (e.g.
divergence at step 80K requiring rollback to step 50K) costs ~1.1 days at 1.52s/step
vs ~1.5 days at 1.9s/step. Faster step time buys retry headroom within the same window.

Unattended critical path: download (1 week) → pre-process (2 days) → pre-compute (1–2 days, overlaps) → train (2.89 days).

---

## Key Risks and Mitigations

**Training divergence at batch_size=1**
The Perceiver Resampler can collapse (all 128 query tokens converge to the same
representation) if LR is too high early in training.
*Mitigation:* 2000-step linear warmup + gradient clipping at max_norm=1.0.
If loss spikes after ~10K steps, reduce LR to 3e-5 and restart from the last checkpoint.

**Content bleed in style-only mode**
The adapter learns to encode both style and content (since reference = target during
training). Style isolation is achieved at inference by single-block-only injection,
but some bleed may remain.
*Mitigation:* Add a training augmentation pass where a different image of the same
style category is used as the reference (requires grouping the WikiArt and JourneyDB
subsets by style label).

**MPS BF16 NaN**
PyTorch MPS occasionally produces NaN in BF16 attention operations on certain
hardware/OS combinations.
*Mitigation:* If NaN appears in the loss after step 0, add `--fp32_attention` flag
to force the transformer attention computation to FP32. Slight memory increase
but stable.

**Storage during training**
Each checkpoint pair (adapter + EMA) is ~2 GB. At 2K-step intervals over 120K steps:
60 checkpoints = 120 GB if not pruned.
*Mitigation:* Auto-purge, keep only the last 5 checkpoints + best (by eval loss).
Active storage for checkpoints: ~12 GB.

**Thunderbolt 4 connection dropout (if training shards are on external SSD)**
A TB4 disconnection during a 2–3 day unattended run crashes the training process.
*Mitigation:* Use `caffeinate -i -d` to prevent sleep; use 2K-step checkpoint interval
to limit worst-case loss to ~1 hour; prefer copying active shards to internal SSD
if 300GB+ free. See Phase 0 for full details.

**Qwen3 Q4 quality degradation**
Q4 quantization of Qwen3 introduces noise in text embeddings, which may slightly
reduce prompt adherence quality in training samples.
*Mitigation:* Qwen3 is frozen; its role during training is to provide text conditioning
context, not to be optimized. The adapter learns to condition on image features
regardless of minor text embedding noise. If quality is visibly impacted, use Q8
quantization instead (doubles Qwen3 memory to ~4 GB, still fits comfortably).

---

## Relationship to Existing Codebase

This plan adds the following to iris.c:

| New component | Depends on | Complexity |
|---|---|---|
| `iris_siglip.c` | `iris_safetensors.c`, `iris_metal.m` | ~1500 lines C |
| `iris_ipadapter.c` | `iris_siglip.c`, `iris_safetensors.c` | ~500 lines C |
| Modifications to `iris_transformer_flux.c` | Existing block forward functions | ~200 lines C |
| `iris_shaders.metal` additions | Existing attention kernels | ~100 lines Metal |
| `main.c` CLI additions | Existing arg parsing | ~50 lines C |

The base Flux Klein 4B model weights are **not modified**. The adapter loads alongside
the existing model as a separate safetensors file, analogous to how LoRA weights
are currently loaded via `iris_lora.c`.

---

## Reference Links

- Training framework: https://github.com/ml-explore/mlx
- mflux (Flux Klein MLX forward pass): https://github.com/filipstrand/mflux
- mlx-lm grad_checkpoint pattern: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py
- MLX custom Metal kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- PyTorch reference training codebase: https://github.com/clf28/x-flux-ip-adapter
- XLabs Flux IP-Adapter (v2 trained at 512→1024): https://huggingface.co/XLabs-AI/flux-ip-adapter-v2
- InstantX Flux IP-Adapter (Flux.1-dev, 5.3GB BF16): https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter
- SigLIP vision encoder: https://huggingface.co/google/siglip-so400m-patch14-384
- Untwisting RoPE (training-free alternative): https://arxiv.org/abs/2602.05013
- JourneyDB dataset: https://huggingface.co/datasets/JourneyDB/JourneyDB
- LAION-Aesthetics-v2: https://huggingface.co/datasets/laion/laion2B-en-aesthetic
- img2dataset: https://github.com/rom1504/img2dataset
