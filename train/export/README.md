# IP-Adapter Export

Converts a trained IP-Adapter checkpoint into a bundle loadable by iris.c.

## Bundle format

```
adapter_weights.safetensors   — quantized weight tensors (mmap-ready)
adapter_meta.json             — dimensions, quant mode, provenance
```

### Tensor layout

| Export key | Shape | Description |
|---|---|---|
| `perceiver.query_tokens` | `[Q, D]` | Learned cross-attn query embeddings |
| `perceiver.query_proj` | `[D, D]` | Query projection |
| `perceiver.key_proj` | `[D, S]` | Key projection (SigLIP dim) |
| `perceiver.value_proj` | `[D, S]` | Value projection (SigLIP dim) |
| `perceiver.out_proj` | `[D, D]` | Output projection |
| `perceiver.norm_weight` | `[D]` | LayerNorm weight (F32 always) |
| `perceiver.norm_bias` | `[D]` | LayerNorm bias (F32 always) |
| `ip_k_stacked` | `[N, D, D]` | Stacked K projections for all blocks |
| `ip_v_stacked` | `[N, D, D]` | Stacked V projections for all blocks |
| `ip_scale` | `[N]` | Per-block blend weight (F32 always) |

`Q` = num_image_tokens, `D` = hidden_dim, `S` = siglip_dim, `N` = num_blocks.

For INT8 exports each quantised tensor has a companion `<name>.scale` tensor
containing per-row F32 dequant factors. Dequant: `x[i,:] = q_i8[i,:] * scale[i]`.

### adapter_meta.json fields

```json
{
  "version": 1,
  "adapter_type": "ip_adapter_klein",
  "model_target": "flux-klein-4b",
  "iris_version": "v2.7",
  "num_blocks": 25,
  "num_double_blocks": 5,
  "num_single_blocks": 20,
  "hidden_dim": 3072,
  "num_image_tokens": 16,
  "siglip_dim": 1152,
  "style_only": false,
  "quant": "bfloat16",
  "training_step": 50000,
  "source_checkpoint": "step_050000.safetensors",
  "export_time": "2026-05-07T12:00:00Z",
  "tensors": { "<name>": { "shape": [...], "dtype": "BF16" }, ... }
}
```

---

## CLI

```bash
# BF16 (default, recommended — same precision as training)
python train/export/export_adapter.py \
    --checkpoint /Volumes/2TBSSD/checkpoints/stage1/step_050000.safetensors \
    --output     /tmp/iris_ip_adapter/

# Use EMA weights (better inference quality)
python train/export/export_adapter.py --checkpoint ... --use-ema --output ...

# INT8 (~2× smaller, requires C-side INT8 dequant kernel)
python train/export/export_adapter.py --checkpoint ... --quant int8 --output ...

# Style-only mode (zero ip_scale for double-stream blocks 0–4)
python train/export/export_adapter.py --checkpoint ... --style-only --output ...

# Validate bundle immediately after writing
python train/export/export_adapter.py --checkpoint ... --output ... --validate
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--checkpoint PATH` | required | Input `.safetensors` checkpoint |
| `--output DIR` | required | Output directory (created if absent) |
| `--quant MODE` | `bfloat16` | Quantisation: `bfloat16`, `float16`, `int8` |
| `--use-ema` | off | Load EMA weights instead of live weights |
| `--style-only` | off | Zero ip_scale for double-stream blocks (0–4) |
| `--step N` | auto | Override training step in metadata |
| `--validate` | off | Re-open bundle after writing and verify |

---

## Quantisation comparison

| Mode | Size (4B adapter) | Precision loss | C dequant required |
|---|---|---|---|
| `bfloat16` | ~45 MB | Negligible (same as training) | No (direct mmap) |
| `float16` | ~45 MB | Negligible | No (cast at load) |
| `int8` | ~23 MB | Small (~0.5% metric) | Yes (row * scale[row]) |

BF16 is recommended unless you have strict memory constraints.

---

## C integration

See [iris_ip_adapter.h](iris_ip_adapter.h) for the full API. Minimal usage:

```c
#include "iris_ip_adapter.h"

// Load bundle (once at startup)
iris_ip_adapter_t *ip = iris_ip_adapter_load("/path/to/bundle");

// Per image: compress SigLIP features to num_image_tokens embeddings
float *ip_embeds = malloc(ip->num_image_tokens * ip->hidden_dim * sizeof(float));
iris_ip_adapter_perceive(ip, siglip_feats, n_siglip, ip_embeds);

// Pre-allocate K/V buffers (reused across blocks)
float *k_ip = malloc(ip->num_image_tokens * ip->hidden_dim * sizeof(float));
float *v_ip = malloc(ip->num_image_tokens * ip->hidden_dim * sizeof(float));

// Per transformer block: inject IP attention into img_hidden
for (int b = 0; b < ip->num_blocks; b++) {
    iris_ip_adapter_get_kv(ip, b, ip_embeds, k_ip, v_ip);
    // ... Flux block forward pass updates img_q and img_hidden ...
    iris_ip_adapter_inject(ip, b, img_q, img_seq, k_ip, v_ip, img_hidden);
}

free(k_ip); free(v_ip); free(ip_embeds);
iris_ip_adapter_free(ip);
```

### INT8 dequant in C

```c
// Dequantise block b row r of ip_k_stacked:
int8_t  *q_row   = (int8_t *)ip->ip_k_stacked + (b * D + r) * D;
float    s       = ip->ip_k_scale[b * D + r];
float    w[D];
for (int c = 0; c < D; c++) w[c] = q_row[c] * s;
```

---

## Running the tests

```bash
cd train
.venv/bin/python -m pytest tests/test_export.py -v
```

The smoke test creates a synthetic checkpoint, runs the full export pipeline
(bfloat16 and int8 modes), and validates the output bundle — no real checkpoint needed.
