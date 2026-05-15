# Plan: LoRA Support for flux2.c

**Status: Implemented** — `iris_lora.c` is complete and listed in the active codebase.

## Overview

Add support for loading and applying LoRA (Low-Rank Adaptation) weights to the
Flux transformer at inference time. This enables community-trained style,
subject, and concept adaptations without full model finetuning.

LoRA modifies linear projections as: `output = W @ x + scale * (lora_B @ (lora_A @ x))`
where `lora_A` is [rank, in_dim] and `lora_B` is [out_dim, rank], with rank typically 4-128.

## LoRA File Formats

There are three major formats in the Flux ecosystem. All use safetensors.

### 1. XLabs Format (most popular on Civitai)

Inspected from `XLabs-AI/flux-RealismLora` (22MB, rank=16, bf16):

```
double_blocks.{i}.processor.qkv_lora1.down.weight   [rank, hidden]      (img QKV down)
double_blocks.{i}.processor.qkv_lora1.up.weight      [hidden*3, rank]    (img QKV up, fused)
double_blocks.{i}.processor.qkv_lora2.down.weight   [rank, hidden]      (txt QKV down)
double_blocks.{i}.processor.qkv_lora2.up.weight      [hidden*3, rank]    (txt QKV up, fused)
double_blocks.{i}.processor.proj_lora1.down.weight   [rank, hidden]      (img output proj down)
double_blocks.{i}.processor.proj_lora1.up.weight     [hidden, rank]      (img output proj up)
double_blocks.{i}.processor.proj_lora2.down.weight   [rank, hidden]      (txt output proj down)
double_blocks.{i}.processor.proj_lora2.up.weight     [hidden, rank]      (txt output proj up)
```

Key characteristics:
- `lora1` = image stream, `lora2` = text stream
- QKV is **fused**: up.weight is [hidden*3, rank], needs splitting into Q, K, V
- `down` = lora_A (low rank projection), `up` = lora_B (back to full dim)
- Only targets double block attention (QKV + output proj), not MLP or single blocks
- Block indices go to 18 (Flux.1 Dev has 19 double blocks)

### 2. Kohya/sd-scripts Format

```
lora_unet_double_blocks_{i}_img_attn_qkv.lora_down.weight    [rank, hidden]
lora_unet_double_blocks_{i}_img_attn_qkv.lora_up.weight      [hidden*3, rank]
lora_unet_double_blocks_{i}_txt_attn_qkv.lora_down.weight
lora_unet_double_blocks_{i}_txt_attn_qkv.lora_up.weight
lora_unet_double_blocks_{i}_img_attn_proj.lora_down.weight
lora_unet_double_blocks_{i}_img_attn_proj.lora_up.weight
lora_unet_double_blocks_{i}_txt_attn_proj.lora_down.weight
lora_unet_double_blocks_{i}_txt_attn_proj.lora_up.weight
lora_unet_single_blocks_{i}_linear1.lora_down.weight          [rank, hidden]
lora_unet_single_blocks_{i}_linear1.lora_up.weight            [hidden*3+mlp*2, rank]
lora_unet_single_blocks_{i}_linear2.lora_down.weight
lora_unet_single_blocks_{i}_linear2.lora_up.weight
```

Key characteristics:
- Uses `lora_down`/`lora_up` naming
- Explicit `img`/`txt` prefixes
- Can target single blocks too (linear1 = fused QKV+MLP, linear2 = fused proj+MLP)
- `alpha` metadata may be present for per-layer scaling

### 3. Diffusers/PEFT Format

Inspected from `ByteDance/Hyper-SD` Flux LoRA (1.4GB, rank=64, f32, 1008 keys, 54 unique patterns):

**Double blocks (transformer_blocks):**
```
transformer.transformer_blocks.{i}.attn.to_q.lora_A.weight          [rank, hidden]
transformer.transformer_blocks.{i}.attn.to_q.lora_B.weight          [hidden, rank]
transformer.transformer_blocks.{i}.attn.to_k.lora_A/B.weight
transformer.transformer_blocks.{i}.attn.to_v.lora_A/B.weight
transformer.transformer_blocks.{i}.attn.to_out.0.lora_A/B.weight
transformer.transformer_blocks.{i}.attn.add_q_proj.lora_A/B.weight  (text Q)
transformer.transformer_blocks.{i}.attn.add_k_proj.lora_A/B.weight  (text K)
transformer.transformer_blocks.{i}.attn.add_v_proj.lora_A/B.weight  (text V)
transformer.transformer_blocks.{i}.attn.to_add_out.lora_A/B.weight  (text output)
transformer.transformer_blocks.{i}.ff.net.0.proj.lora_A/B.weight    (img FFN gate+up)
transformer.transformer_blocks.{i}.ff.net.2.lora_A/B.weight         (img FFN down)
transformer.transformer_blocks.{i}.ff_context.net.0.proj.lora_A/B.weight  (txt FFN gate+up)
transformer.transformer_blocks.{i}.ff_context.net.2.lora_A/B.weight      (txt FFN down)
transformer.transformer_blocks.{i}.norm1.linear.lora_A/B.weight     (img AdaLN mod)
transformer.transformer_blocks.{i}.norm1_context.linear.lora_A/B.weight  (txt AdaLN mod)
```

**Single blocks (single_transformer_blocks):**
```
transformer.single_transformer_blocks.{i}.attn.to_q.lora_A/B.weight
transformer.single_transformer_blocks.{i}.attn.to_k.lora_A/B.weight
transformer.single_transformer_blocks.{i}.attn.to_v.lora_A/B.weight
transformer.single_transformer_blocks.{i}.proj_mlp.lora_A/B.weight
transformer.single_transformer_blocks.{i}.proj_out.lora_A/B.weight
transformer.single_transformer_blocks.{i}.norm.linear.lora_A/B.weight
```

**Global projections:**
```
transformer.x_embedder.lora_A/B.weight                              [rank, 128] / [hidden, rank]
transformer.context_embedder.lora_A/B.weight                        [rank, text_dim] / [hidden, rank]
transformer.proj_out.lora_A/B.weight
transformer.norm_out.linear.lora_A/B.weight
transformer.time_text_embed.timestep_embedder.linear_1/2.lora_A/B.weight
transformer.time_text_embed.guidance_embedder.linear_1/2.lora_A/B.weight
transformer.time_text_embed.text_embedder.linear_1/2.lora_A/B.weight
```

Key characteristics:
- Uses `lora_A`/`lora_B` naming
- Q, K, V are **already split** (not fused)
- Full `transformer.` prefix
- Targets **everything**: attention, FFN, norms, embedders, time embed (54 patterns)
- Typically f32, much larger files than XLabs (1.4GB vs 22MB at same rank)

## Architecture Mapping

### Base Model Weight Names → LoRA Targets

| Base model weight | XLabs key | Kohya key |
|---|---|---|
| `transformer_blocks.{i}.attn.to_q.weight` | `double_blocks.{i}.processor.qkv_lora1` (split Q) | `double_blocks_{i}_img_attn_qkv` (split Q) |
| `transformer_blocks.{i}.attn.to_k.weight` | same (split K) | same (split K) |
| `transformer_blocks.{i}.attn.to_v.weight` | same (split V) | same (split V) |
| `transformer_blocks.{i}.attn.to_out.0.weight` | `double_blocks.{i}.processor.proj_lora1` | `double_blocks_{i}_img_attn_proj` |
| `transformer_blocks.{i}.attn.add_q_proj.weight` | `double_blocks.{i}.processor.qkv_lora2` (split Q) | `double_blocks_{i}_txt_attn_qkv` (split Q) |
| `transformer_blocks.{i}.attn.add_k_proj.weight` | same (split K) | same (split K) |
| `transformer_blocks.{i}.attn.add_v_proj.weight` | same (split V) | same (split V) |
| `transformer_blocks.{i}.attn.to_add_out.weight` | `double_blocks.{i}.processor.proj_lora2` | `double_blocks_{i}_txt_attn_proj` |

### Flux.2 Klein vs Flux.1 Dev Block Counts

| Model | Double Blocks | Single Blocks |
|---|---|---|
| Flux.1 Dev | 19 | 38 |
| Flux.2 Klein 4B | 5 | 20 |
| Flux.2 Klein 9B | 8 | 24 |

Most community LoRAs are trained for Flux.1 Dev (19 double blocks). They will
**partially** work with Klein models — only blocks 0..4 (or 0..7) will match.
LoRA keys for non-existent blocks are silently skipped. This is standard
behavior and often produces useful results since early blocks carry the most
style/concept information.

## Implementation Plan

### Phase 1: Core Data Structures

**New file: `flux_lora.c` + `flux_lora.h`**

```c
/* A single LoRA adapter for one linear projection */
typedef struct {
    float *lora_A;      /* [rank, in_dim] - down projection */
    float *lora_B;      /* [out_dim, rank] - up projection */
    int rank;
    int in_dim;
    int out_dim;
} lora_adapter_t;

/* LoRA state for the entire transformer */
typedef struct {
    float scale;                    /* Global LoRA strength (0.0 = disabled, 1.0 = full) */

    /* Double block adapters (per block, per projection) */
    int num_double_blocks;
    lora_adapter_t *double_img_q;   /* [num_double_blocks] */
    lora_adapter_t *double_img_k;
    lora_adapter_t *double_img_v;
    lora_adapter_t *double_img_proj;
    lora_adapter_t *double_txt_q;
    lora_adapter_t *double_txt_k;
    lora_adapter_t *double_txt_v;
    lora_adapter_t *double_txt_proj;

    /* Single block adapters (if present) */
    int num_single_blocks;
    lora_adapter_t *single_q;       /* [num_single_blocks] */
    lora_adapter_t *single_k;
    lora_adapter_t *single_v;
    lora_adapter_t *single_proj;

    /* Scratch buffer for intermediate computation */
    float *scratch;                 /* [max_seq_len * max_rank] */
    int scratch_size;
} lora_state_t;
```

### Phase 2: Loading (format detection + parsing)

**In `flux_lora.c`:**

```c
lora_state_t *lora_load(const char *path, int num_double_blocks,
                        int num_single_blocks, int hidden_size,
                        int mlp_hidden, float scale);
void lora_free(lora_state_t *lora);
```

1. Open safetensors file using existing `safetensors_open()`
2. Auto-detect format by inspecting first key:
   - Contains `processor.` → XLabs format
   - Contains `lora_unet_` → Kohya format
   - Contains `transformer.` → Diffusers format
3. Parse keys according to detected format
4. For fused QKV weights: split up_weight into Q, K, V portions (`out_dim/3` each)
5. Convert bf16 → f32 if needed (LoRA weights are small, f32 is fine)
6. Skip keys for block indices >= model's block count (with warning)
7. Allocate scratch buffer: `max_seq_len * max_rank * sizeof(float)`

### Phase 3: Application (the core math)

**In `flux_lora.c`:**

```c
/* Apply LoRA correction to a linear projection output.
 * Called AFTER the base linear projection.
 * out += scale * lora_B @ (lora_A @ x)
 *
 * x:   [seq_len, in_dim]   - input to the linear layer
 * out: [seq_len, out_dim]  - output (modified in-place)
 */
void lora_apply(const lora_adapter_t *adapter, float scale,
                const float *x, float *out, int seq_len, float *scratch);
```

Implementation:
1. `scratch = lora_A @ x^T` → scratch is [rank, seq_len]
2. `out += scale * lora_B @ scratch` → adds [out_dim, seq_len] to output
3. Uses BLAS `cblas_sgemm` when available, fallback to naive loops

**For GPU path:** Apply LoRA on CPU after GPU linear, or implement a small
Metal kernel. Since LoRA rank is tiny (4-128), the CPU overhead is minimal
compared to the base matmul. Start with CPU-only LoRA application.

### Phase 4: Integration into Transformer Forward Pass

**In `flux_transformer.c`:**

Add `lora_state_t *lora` field to `flux_transformer_t`.

Modify the CPU forward path (`double_block_forward`, `single_block_forward`):

```c
/* Example: after img Q projection in double_block_forward */
LINEAR_BF16_OR_F32(img_q, img_norm, blk->img_q_weight, blk->img_q_weight_bf16,
                   img_seq, hidden, hidden);
if (trans->lora && trans->lora->double_img_q[block_idx].lora_A) {
    lora_apply(&trans->lora->double_img_q[block_idx], trans->lora->scale,
               img_norm, img_q, img_seq, trans->lora->scratch);
}
```

This pattern repeats for each of the 8 projections per double block and
4 projections per single block.

For the **GPU paths** (`double_block_forward_bf16`, `single_block_forward_gpu_chained`):
- Download GPU tensor to CPU after the linear projection
- Apply LoRA correction on CPU
- Upload back to GPU
- This is acceptable because LoRA rank is small and the download/upload
  is tiny compared to the base matmul

**Alternative GPU approach** (optimization, Phase 5):
- Add a simple Metal kernel that computes `out += scale * B @ (A @ x)`
- Keep LoRA weights as small GPU buffers
- Avoids CPU roundtrip entirely

### Phase 5: CLI and Server Integration

**In `main.c` and `flux.h`:**

```c
/* New fields in flux_params or new API function */
flux_ctx *ctx = flux_init("model-dir");
flux_load_lora(ctx, "path/to/lora.safetensors", 0.8);  /* scale=0.8 */
flux_unload_lora(ctx);
```

**CLI flags:**
```
--lora <path>        Load LoRA adapter
--lora-scale <float> LoRA strength (default 1.0, range 0.0-2.0)
```

**Server mode:**
- Accept `lora` and `lora_scale` in JSON request (per-generation)
- Or load a default LoRA on startup via CLI flag
- Hot-swap: calling `flux_load_lora` again replaces the current LoRA

**Web UI:**
- Add LoRA file upload in controls panel
- Scale slider (0.0 - 2.0)
- Display loaded LoRA name

## Compatibility Notes

### Flux.1 Dev LoRAs on Flux.2 Klein

- Flux.1 Dev has 19 double + 38 single blocks; Klein 4B has 5 + 20
- Block-indexed LoRA adapters for blocks >= model count are skipped
- hidden_size differs: Dev=3072, Klein 4B=3072 (same!), Klein 9B=4096
- **4B Klein can use Flux.1 Dev LoRAs** (same hidden size, fewer blocks → partial but useful)
- **9B Klein cannot** (different hidden size → dimension mismatch on all weights)
- We should validate dimensions on load and skip mismatched adapters with a warning

### Memory Impact

For a rank-16 LoRA on 5 double blocks (4B Klein):
- Per adapter: rank * (in_dim + out_dim) * 4 bytes
- QKV (fused): 16 * (3072 + 3072*3) * 4 = 768KB per stream per block
- Proj: 16 * (3072 + 3072) * 4 = 384KB per stream per block
- Total per double block: (768 + 384) * 2 streams = 2.25MB
- 5 blocks: ~11MB total
- Plus scratch: 512 * 128 * 4 = 256KB

**Total LoRA memory: ~12MB** — negligible compared to the 8-18GB base model.

## File Changes Summary

| File | Changes |
|---|---|
| **flux_lora.c** (new) | LoRA loading, format detection, application |
| **flux_lora.h** (new) | Public API: `lora_load`, `lora_apply`, `lora_free` |
| **flux_transformer.c** | Add `lora_state_t *lora` to struct, apply after each linear projection |
| **flux.c** | Add `flux_load_lora()` / `flux_unload_lora()` public API |
| **flux.h** | Declare LoRA API functions |
| **main.c** | Parse `--lora` / `--lora-scale` flags, wire up server mode |
| **Makefile** | Add `flux_lora.c` to SRCS |

## Implementation Order

1. `flux_lora.h` — data structures and API declarations
2. `flux_lora.c` — loading (XLabs format first, easiest), application function
3. `flux_transformer.c` — integrate into CPU forward path
4. `flux.c` + `flux.h` — public API
5. `main.c` + Makefile — CLI flags, build
6. Test with XLabs Realism LoRA on 4B Klein
7. Add Kohya format support
8. Add Diffusers format support
9. GPU path optimization (if CPU roundtrip is measurably slow)
10. Web UI integration

## Testing Strategy

1. Download `XLabs-AI/flux-RealismLora` (22MB)
2. Generate same prompt with and without LoRA
3. Verify output changes meaningfully (realism boost visible)
4. Test with scale=0.0 → output should match no-LoRA exactly
5. Test with LoRA for wrong model size → should warn and skip
6. Test with LoRA that has more blocks than model → should use available blocks
7. Benchmark: measure overhead of LoRA application vs base generation

---

## Phase 2: GPU-Accelerated LoRA (after Phase 1 is proven with real LoRAs)

**Goal:** Re-enable the Metal bf16 fast path while LoRA is active, eliminating
the current full CPU fallback. LoRA rank is small (4-128) so the extra GPU
kernel overhead is negligible compared to the base matmul.

### Approach

Add a small Metal compute kernel that computes `out += scale * B @ (A @ x)`
on-GPU, keeping all tensors in GPU memory throughout the forward pass.

**Steps:**

1. **Store LoRA weights on GPU** — after `lora_load()`, copy `lora_A` and `lora_B`
   float arrays into Metal buffers (one pair per adapter). Store these in
   `lora_adapter_t` alongside the existing CPU pointers.

2. **New Metal kernel in `flux_shaders.metal`:**
   ```metal
   // Compute: out += scale * lora_B @ (lora_A @ x)
   // lora_A: [rank, in_dim], lora_B: [out_dim, rank], x: [seq_len, in_dim]
   // Uses two sequential matmuls with a small [rank, seq_len] intermediate
   kernel void lora_apply(...)
   ```
   Since rank is tiny (≤128), the intermediate fits in threadgroup memory.

3. **Hook into the bf16 GPU forward path** — after each existing GPU linear
   projection call (in `double_block_forward_bf16` etc.), if the matching
   `lora_adapter_t` has GPU buffers, dispatch the Metal kernel.

4. **Remove the `&& !tf->lora` bypass** — once GPU LoRA works, the condition
   becomes unnecessary and the bf16 path runs unconditionally again.

5. **Keep CPU `lora_apply()` as fallback** — for `generic` and `blas` builds
   without Metal, behavior is unchanged.

### Prerequisites
- Phase 1 LoRA proven to produce correct visual results with real community LoRAs
- Benchmark shows the CPU fallback is meaningfully slower (worth the complexity)

### Scope
- Only needed for `mps` (Metal) build target
- `blas` and `generic` builds continue using CPU LoRA (already fast enough there)
- Web UI LoRA controls (file picker, scale slider, loaded LoRA name display)
