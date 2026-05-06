# iris.c — Pure C Image Synthesis Engine

**iris.c** is a self-contained C implementation of two image synthesis model families, with zero Python or CUDA dependencies at inference time:

- **Flux.2 Klein** (4B and 9B) — Black Forest Labs' rectified-flow diffusion transformer
- **Z-Image-Turbo** (6B) — Tongyi-MAI's S3-DiT distilled model

Three integrated components:

| Component | Description |
|-----------|-------------|
| `iris` binary | CLI inference: text-to-image, img2img, interactive REPL |
| `web/` | Flask web UI with generation history, queuing, LoRA support |
| `train/` | MLX-based IP-Adapter training pipeline (Apple Silicon) |

---

## Quick Start — Inference

```bash
# Build (Apple Silicon recommended)
make mps        # Apple Silicon GPU — fastest
make blas       # Intel Mac or Linux with OpenBLAS
make generic    # Pure C, no dependencies — slowest

# Download model (~16 GB)
./download_model.sh 4b

# Generate an image
./iris -d flux-klein-4b -p "a cat sitting on a windowsill" -o cat.png
```

No Python, no CUDA, no virtual environment required.

---

## Supported Models

| Model | Flag | Steps | Speed | License |
|-------|------|-------|-------|---------|
| Flux.2 Klein 4B distilled | `flux-klein-4b` | 4 | ~7.6s @ 512² on M3 Max | MIT |
| Flux.2 Klein 4B base | `flux-klein-4b-base` | 50 | ~25× slower | MIT |
| Flux.2 Klein 9B distilled | `flux-klein-9b` | 4 | higher quality | Non-commercial |
| Flux.2 Klein 9B base | `flux-klein-9b-base` | 50 | highest quality | Non-commercial |
| Z-Image-Turbo 6B | `zimage-turbo` | 9 | 8 NFE | Apache 2.0 |

Architecture is autodetected from each model's config files. No hardcoded dimensions.

---

## Building

```bash
make            # show available backends
make mps        # macOS Apple Silicon (Metal GPU) — recommended
make blas       # BLAS/Accelerate acceleration
make generic    # pure C fallback

make test       # run test suite
make test-quick # quick 64×64 sanity check
make clean
```

**Linux OpenBLAS:**
```bash
sudo apt install libopenblas-dev   # Ubuntu/Debian
sudo dnf install openblas-devel    # Fedora
make blas
```

---

## Inference CLI

### Text-to-Image

```bash
./iris -d flux-klein-4b -p "A fluffy orange cat sitting on a windowsill" -o cat.png
./iris -d zimage-turbo  -p "a fish" -o fish.png
```

### Image-to-Image

Flux uses in-context conditioning: the reference image is passed as extra tokens rather than added as noise. The model attends to both the reference and the prompt simultaneously.

```bash
./iris -d flux-klein-4b -p "oil painting, impressionist style" -i photo.png -o painting.png
```

### Multi-Reference

Combine elements from multiple images. Each reference gets a distinct T-offset in the RoPE position encoding so the model can attend to them independently:

```bash
./iris -d flux-klein-4b -i car.png -i beach.png -p "a sports car on the beach" -o result.png
```

### Interactive REPL

```bash
./iris -d flux-klein-4b
```

```
iris> a red sports car
Done -> /tmp/.../image-0001.png  (ref $0)
iris> 512x512 $0 oil painting version
Done -> /tmp/.../image-0002.png  (ref $1)
iris> $0 $1 combine them
Done -> /tmp/.../image-0003.png
```

Syntax: `[WxH] [$ref ...] prompt` — size and references are optional inline prefixes.

Commands: `!help` `!save` `!load` `!seed` `!size` `!steps` `!guidance` `!linear` `!power` `!explore` `!show` `!quit`

### Terminal Image Display

```bash
./iris -d flux-klein-4b -p "a robot" -o robot.png --show        # show final image
./iris -d flux-klein-4b -p "a robot" -o robot.png --show-steps  # watch denoising
```

Supported protocols: Kitty, Ghostty, iTerm2, WezTerm, Konsole. Auto-detected from `$TERM` / `$TERM_PROGRAM`.

### Key Options

```
-d PATH     Model directory
-p TEXT     Prompt
-o PATH     Output (.png or .ppm)
-i PATH     Reference image (repeatable for multi-ref)
-W / -H N   Width / height in pixels (multiples of 16, max 1792)
-s N        Steps (default: auto — 4 distilled, 50 base, 9 Z-Image)
-S N        Seed (-1 for random; seed always printed to stderr)
-g N        CFG guidance (default: auto — 1.0 distilled, 4.0 base, 0.0 Z-Image)
--linear    Linear timestep schedule (base model experimentation)
--power     Power-curve schedule; --power-alpha N sets exponent (default 2.0)
--no-mmap   Load all weights upfront (faster on BLAS, more RAM)
--show      Display in terminal
-v          Verbose timing output
```

### Memory

Memory-mapped weights (default) reduce peak from ~16 GB to ~4–5 GB for the 4B model, making inference possible on 8 GB systems. MPS mode uses zero-copy pointers into the mapped region — mmap is fastest on Apple Silicon. BLAS users with 32+ GB may prefer `--no-mmap` to avoid per-step bf16→f32 conversion.

| Model | Peak with mmap | Peak without mmap |
|-------|---------------|-------------------|
| 4B | ~4–5 GB | ~16 GB |
| 9B | ~8–10 GB | ~32 GB |

### PNG Metadata

Generated PNGs embed `flux:seed` and `flux:model` in the metadata, so the seed is always recoverable:

```bash
exiftool image.png | grep flux
python3 -c "from PIL import Image; print(Image.open('image.png').info)"
```

---

## Performance

Benchmarks on M3 Max (128 GB), Flux 4B distilled, 4 steps, including model load, no warmup:

| Resolution | iris (MPS) | PyTorch (MPS) |
|------------|-----------|---------------|
| 256×256 | 5.2 s | 11 s |
| 512×512 | 7.6 s | 13 s |
| 1024×1024 | 19 s | 25 s |

Community benchmarks (512×512, Flux 4B distilled):

| Hardware | Backend | Time |
|----------|---------|------|
| M3 Ultra | MPS | 4.5 s |
| M3 Max | MPS | 7.6 s |
| M4 MacBook Pro | MPS | 19 s |
| M1 Max MacBook Pro | MPS | 39.9 s |
| AMD Ryzen 7800X3D | BLAS | 47.8 s |
| Intel i5-1135G7 | BLAS | 218 s |

---

## Web UI

A Flask-based web interface wrapping the `iris` binary in persistent server mode:

```bash
pip install flask
python web/server.py --model-dir flux-klein-4b
# → http://localhost:5000
```

Features:
- Generation queue with SSE progress streaming
- Full parameter controls (steps, guidance, schedule, size, seed)
- Style presets and prompt templates
- Image-to-image and multi-reference generation
- LoRA loading and scale control
- Generation history with search and filtering
- Lightbox with metadata overlay, download, delete
- Side-by-side comparison view
- Variation grid
- Dark/light theme
- Model switching and download from UI

```
web/server.py          Flask server + IrisServer process manager
web/static/            Frontend (vanilla JS, no framework)
```

---

## IP-Adapter Training Pipeline

Train a style-reference IP-Adapter on top of Flux Klein 4B using MLX on Apple Silicon. The adapter adds `--sref`-style visual conditioning: given a reference image, the model generates images matching its style.

### Prerequisites

```bash
# Python 3.11+, Apple Silicon Mac (M1 or later)
cd train && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Model weights (`flux-klein-model`) must be present at the repo root (symlink or download).

### First Run — Setup Wizard

```bash
train/.venv/bin/python train/scripts/pipeline_setup.py
```

Interactive wizard that:
1. Presents all run scales with time/disk estimates
2. Reviews quality feature settings
3. Validates prerequisites (disk, tmux, model, venv)
4. Creates required directories
5. Generates a pipeline config
6. Outputs exact start/status/stop commands

**For automation** (Claude Code / DISPATCH):
```bash
train/.venv/bin/python train/scripts/pipeline_setup.py --ai --scale small
```
Returns a JSON blob with `ready`, `checks`, `dirs`, `existing_state`, and `commands`.

### Run Scales

| Scale | Chunks | Steps | Disk | Time | Purpose |
|-------|--------|-------|------|------|---------|
| `dev` | 1 | 200 | ~15 GB | ~2 h | iris.c binary integration testing — no quality features |
| `smoke` | 1 | 100 | ~40 GB | ~6 h | Full-pipeline validation — all quality features |
| `small` | 4 | 95 K total | ~250 GB | ~3–4 days | First quality checkpoint |
| `medium` | 4 | 225 K total | ~600 GB | ~10–14 days | Production quality |
| `large` | 4 | 380 K total | ~1.2 TB | ~3–5 weeks | High quality |
| `all-in` | 4 | 1.14 M total | ~2.2 TB | ~2–3 months | Maximum quality |

### Starting the Pipeline

After running the setup wizard:

```bash
PIPELINE_DATA_ROOT=/Volumes/2TBSSD \
  train/.venv/bin/python train/scripts/pipeline_ctl.py restart-orchestrator \
  --config train/configs/v2_pipeline_active.yaml
```

### Pipeline Steps (per chunk)

```
download → convert → filter_shards → clip_embed → clip_index → clip_dups
→ build_shards → precompute → promoted → validate_shards
→ train → mine → validate
```

Each step writes a sentinel file under `{DATA_ROOT}/pipeline/chunk{N}/{step}.done`. The orchestrator is fully resumable — kill and restart at any time.

### Quality Features

| Feature | Config key | What it does |
|---------|-----------|--------------|
| SigLIP conditioning | `training.siglip: true` | Precomputes 384×384 SigLIP features for visual conditioning during training and mining. ~120 GB per run. |
| Hard-example mining | `training.mine: true` | Evaluates per-sample loss after each chunk, extracts the hardest records for replay in later chunks. |
| EMA checkpoint | `training.mine_use_ema: true` | Uses EMA weights for mining loss evaluation — more stable than raw checkpoint. |
| CLIP dedup | `skip_dedup: false` | Removes near-duplicate images (cosine > 0.95) before training. |
| Anchor shard mixing | auto-populated after chunk 1 | 20% of each batch uses chunk 1 shards to prevent forgetting. |

`smoke` through `all-in` scales run with all quality features enabled. `dev` disables all of them for fastest turnaround.

### Monitoring and Control

```bash
# Check status + doctor summary
PIPELINE_DATA_ROOT=/Volumes/2TBSSD \
  train/.venv/bin/python train/scripts/pipeline_ctl.py status

# Machine-readable JSON for AI/DISPATCH
train/.venv/bin/python train/scripts/pipeline_doctor.py --ai

# Pause / resume / abort
train/.venv/bin/python train/scripts/pipeline_ctl.py pause
train/.venv/bin/python train/scripts/pipeline_ctl.py resume
train/.venv/bin/python train/scripts/pipeline_ctl.py abort

# Restart from a specific chunk (restores archived checkpoint)
train/.venv/bin/python train/scripts/pipeline_ctl.py restart-from-chunk 2
```

### Pipeline Script Reference

| Script | Role |
|--------|------|
| `pipeline_setup.py` | First-run wizard: validates environment, creates dirs, generates config |
| `orchestrator.py` | State machine driving all pipeline steps end-to-end |
| `pipeline_ctl.py` | Operator interface: pause / resume / abort / retry / status |
| `pipeline_doctor.py` | Deep diagnostic: cross-checks sentinels vs. actual artifacts and logs |
| `pipeline_status.py` | Live progress view: step, loss, ETA, log tails |
| `pipeline_lib.py` | Shared primitives: state I/O, sentinels, heartbeats, tmux helpers |
| `precompute_all.py` | Single-pass Qwen3 + VAE + SigLIP precompute with restart-safe progress |
| `mine_hard_examples.py` | Loss-ranked hard example extraction |
| `clip_dedup.py` | CLIP embedding + FAISS deduplication |
| `validate_shards.py` | Shard integrity scan before training |
| `validator.py` | Post-chunk validation: weight check + CLIP-I scoring |
| `download_convert.py` | JourneyDB tgz download + image extraction |
| `build_shards.py` | WebDataset shard assembly |

Operational reference: `train/DISPATCH.md`

---

## C Library API

Link against `libflux.a` and `#include "iris.h"` to use iris.c as a library:

```c
iris_ctx *ctx = iris_load_dir("flux-klein-4b");

iris_params params = IRIS_PARAMS_DEFAULT;
params.width  = 512;
params.height = 512;
params.seed   = 42;

iris_image *img = iris_generate(ctx, "a fluffy cat", &params);
iris_image_save(img, "cat.png");

iris_image_free(img);
iris_free(ctx);
```

Key functions: `iris_load_dir`, `iris_free`, `iris_generate`, `iris_img2img`, `iris_image_load`, `iris_image_save`, `iris_image_free`, `iris_set_seed`, `iris_get_error`, `iris_is_distilled`, `iris_is_zimage`.

---

## Source Layout

```
iris.c / iris.h                  Main library — model loading, generation routing
iris_transformer_flux.c          Flux MMDiT double/single block forward pass
iris_transformer_zimage.c        Z-Image S3-DiT forward pass (noise/context refiners + main layers)
iris_sample.c                    Euler ODE denoising loop, timestep schedules
iris_qwen3.c / iris_qwen3_tokenizer.c  Qwen3 text encoder + BPE tokenizer
iris_vae.c                       VAE encoder / decoder
iris_kernels.c                   CPU kernels: softmax, RMSNorm, GELU, ROPE
iris_metal.m / iris_metal.h      Metal GPU runtime: command buffers, weight cache, buffer pools
iris_shaders.metal               All custom Metal compute kernels
iris_safetensors.c               Safetensors weight loader
iris_image.c / png.c / jpeg.c    Image I/O: PNG, JPEG, PPM
iris_lora.c                      LoRA weight loading and application
embcache.c                       4-bit quantized embedding cache
iris_cli.c                       Interactive REPL
main.c                           CLI entry point
web/                             Flask web UI
train/                           IP-Adapter training pipeline (MLX, Apple Silicon)
  train_ip_adapter.py            Training loop
  ip_adapter/                    Adapter model, loss, EMA, dataset loader
  scripts/                       Pipeline orchestration scripts
  configs/                       Pipeline and training configs
```

---

## Model Download

```bash
# 4B distilled (~16 GB)
./download_model.sh 4b
python download_model.py 4b          # alternative using huggingface_hub

# 4B base (~16 GB)
./download_model.sh 4b-base

# 9B distilled/base (~30 GB, gated — requires HuggingFace token)
# Accept license: https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
./download_model.sh 9b --token YOUR_HF_TOKEN
./download_model.sh 9b-base --token YOUR_HF_TOKEN
```

---

## Testing

```bash
make test        # full test suite (3 test cases against reference vectors)
make test-quick  # 64×64 sanity check only
```

Tests compare generated images against `test_vectors/` using per-pixel difference tolerance to allow minor floating-point variation across platforms and backends.

---

## For AI Assistants

**Primary reference files:**

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Full architecture, implementation details, known pitfalls, development rules |
| `train/DISPATCH.md` | Pipeline operational reference — read before any pipeline work |
| `BACKLOG.md` | Open and completed work items across all components |
| `BUGS.md` | Known issues and observed anomalies |
| `plans/` | Architecture and design documents |

**Starting a session on this repo:**

1. Read `CLAUDE.md` — covers all models, file roles, critical implementation details, and known bugs
2. For pipeline work: `train/.venv/bin/python train/scripts/pipeline_doctor.py --ai` — returns the current pipeline state as compact JSON
3. For inference work: check `BUGS.md` and run `make test`

**Key architectural invariants** (from CLAUDE.md — do not violate):
- Concatenation order for Flux attention is `[TEXT, IMAGE]`
- AdaLN: `out = (1 + scale) * norm(x) + shift`
- RoPE rotation: `out0 = cos·x0 − sin·x1`, `out1 = cos·x1 + sin·x0`
- Z-Image sequence order is `[IMAGE | CAPTION]` (opposite of Flux)
- Dynamic tensors (VAE K/V) must use `iris_metal_sgemm()`, not `_cached()`
- Never hardcode model dimensions — read from config at runtime

**Pipeline sentinel files** are the authoritative state source. Never infer step state from logs or heartbeats alone — always read sentinels or call `pipeline_doctor.py --ai`.

---

## License

MIT. The 9B model weights have a non-commercial use restriction from Black Forest Labs.
