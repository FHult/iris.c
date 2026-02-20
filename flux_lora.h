/*
 * flux_lora.h - LoRA (Low-Rank Adaptation) support for FLUX transformer
 *
 * Applies LoRA adapters at inference time without modifying model weights.
 * Formula: out += scale * (lora_B @ (lora_A @ x))
 * where lora_A is [rank, in_dim] and lora_B is [out_dim, rank].
 *
 * Supported formats (auto-detected from tensor key names):
 *   XLabs  - "double_blocks.N.processor.qkv_lora1.down.weight"
 *   Kohya  - "lora_unet_double_blocks_N_img_attn_qkv.lora_down.weight"
 *   Diffusers - "transformer.transformer_blocks.N.attn.to_q.lora_A.weight"
 */

#ifndef FLUX_LORA_H
#define FLUX_LORA_H

#include <stddef.h>

/* A single LoRA adapter for one linear projection */
typedef struct {
    float *lora_A;  /* [rank, in_dim] - down projection */
    float *lora_B;  /* [out_dim, rank] - up projection */
    int rank;
    int in_dim;
    int out_dim;
} lora_adapter_t;

/* LoRA state for the entire transformer */
typedef struct {
    float scale;            /* LoRA strength (0.0=disabled, 1.0=full) */

    int num_double_blocks;
    lora_adapter_t *double_img_q;    /* [num_double_blocks] */
    lora_adapter_t *double_img_k;
    lora_adapter_t *double_img_v;
    lora_adapter_t *double_img_proj;
    lora_adapter_t *double_txt_q;
    lora_adapter_t *double_txt_k;
    lora_adapter_t *double_txt_v;
    lora_adapter_t *double_txt_proj;

    int num_single_blocks;
    /* Single block fused adapters (Kohya linear1/linear2 style) */
    lora_adapter_t *single_linear1; /* [num_single_blocks] fused QKV+MLP input projection */
    lora_adapter_t *single_linear2; /* [num_single_blocks] fused attention+MLP output projection */

    /* Scratch buffer for lora_A @ x intermediate: [rank * max_seq_len] */
    float *scratch;
    int scratch_len;  /* max seq_len this scratch supports */
    int max_rank;     /* largest rank across all loaded adapters */
} lora_state_t;

/*
 * Load LoRA from a safetensors file.
 * num_double_blocks, num_single_blocks, hidden_size: from transformer config.
 * scale: LoRA strength (typically 0.5-1.0).
 * Returns NULL on failure.
 */
lora_state_t *lora_load(const char *path, int num_double_blocks,
                         int num_single_blocks, int hidden_size,
                         float scale);

/*
 * Apply a LoRA adapter: out += scale * lora_B @ (lora_A @ x)
 * x:   [seq_len, in_dim]   input to the linear layer
 * out: [seq_len, out_dim]  output (modified in-place)
 * scratch: caller-provided buffer of at least [rank * seq_len] floats
 */
void lora_apply(const lora_adapter_t *adapter, float scale,
                const float *x, float *out, int seq_len, float *scratch);

/*
 * Free all LoRA resources.
 */
void lora_free(lora_state_t *lora);

#endif /* FLUX_LORA_H */
