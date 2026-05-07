/*
 * iris_ip_adapter.h — IP-Adapter loader for iris.c
 *
 * Loads the bundle produced by:
 *   python train/export/export_adapter.py --checkpoint ... --output DIR
 *
 * Bundle layout (DIR/):
 *   adapter_weights.safetensors   — mmap-ready weight tensors
 *   adapter_meta.json             — dimensions, quant mode, provenance
 *
 * Quick start:
 *   iris_ip_adapter_t *ip = iris_ip_adapter_load("/path/to/bundle");
 *   // once per image:
 *   iris_ip_adapter_perceive(ip, siglip_feats, n_siglip, ip_embeds);
 *   // once per transformer block:
 *   iris_ip_adapter_get_kv(ip, block_idx, ip_embeds, k_ip, v_ip);
 *   // inside the transformer block, after native self-attention:
 *   iris_ip_adapter_inject(ip, block_idx, img_q, img_seq, k_ip, v_ip, img_hidden);
 *   iris_ip_adapter_free(ip);
 *
 * Quantisation modes:
 *   bfloat16  Default. Weights stored as BF16 in safetensors; loaded as F32.
 *   float16   Weights stored as F16; loaded as F32.
 *   int8      Large linear weights stored as INT8 + F32 per-row scale.
 *             Dequantise: row i = q_i8[i,:] * scale[i]
 *             Scale tensors have suffix ".scale" in safetensors.
 *
 * Thread safety:
 *   iris_ip_adapter_load / iris_ip_adapter_free are NOT thread-safe.
 *   After loading, const operations (perceive / get_kv / inject) are re-entrant
 *   provided each caller supplies its own output buffers.
 *
 * Dependency:
 *   iris_safetensors.h — for safetensors_open / safetensors_get_f32 / etc.
 *   iris_kernels.h     — for rms_norm, gemm helpers
 */

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Configuration struct
 * ---------------------------------------------------------------------- */

typedef struct {
    /* Dimensions (from adapter_meta.json) */
    int num_blocks;           /* total IP blocks = num_double + num_single      */
    int num_double_blocks;    /* Flux double-stream blocks this adapter covers  */
    int num_single_blocks;    /* Flux single-stream blocks this adapter covers  */
    int hidden_dim;           /* Flux transformer hidden dim (3072 for 4B)      */
    int num_image_tokens;     /* PerceiverResampler output tokens (Q)           */
    int siglip_dim;           /* SigLIP feature dim (1152 for SigLIP-400M)      */
    int style_only;           /* 1 = double-stream ip_scale zeroed at export    */

    /* Quantisation mode: "bfloat16" | "float16" | "int8" */
    char quant[16];

    /* -----------------------------------------------------------------
     * Perceiver weights (all float32 after load, regardless of quant)
     * ---------------------------------------------------------------- */

    /* query_tokens  [num_image_tokens, hidden_dim] — learned query embeddings */
    float *query_tokens;

    /* Perceiver cross-attention projections */
    float *query_proj;        /* [hidden_dim, hidden_dim]  */
    float *key_proj;          /* [hidden_dim, siglip_dim]  */
    float *value_proj;        /* [hidden_dim, siglip_dim]  */
    float *out_proj;          /* [hidden_dim, hidden_dim]  */

    /* LayerNorm on query side */
    float *norm_weight;       /* [hidden_dim]  float32 always */
    float *norm_bias;         /* [hidden_dim]  float32 always */

    /* -----------------------------------------------------------------
     * IP-Adapter projections (stacked across all num_blocks blocks)
     * ---------------------------------------------------------------- */

    /* ip_k_stacked  [num_blocks, hidden_dim, hidden_dim]
     * ip_v_stacked  [num_blocks, hidden_dim, hidden_dim]
     *
     * Block i: ptr = ip_k_stacked + i * hidden_dim * hidden_dim
     */
    float *ip_k_stacked;
    float *ip_v_stacked;

    /* ip_scale  [num_blocks]  float32 — per-block blend weight */
    float *ip_scale;

    /* -----------------------------------------------------------------
     * INT8 dequant scale tensors (NULL unless quant == "int8")
     *
     * Each scale tensor has shape matching the "rows" dimension of the
     * corresponding weight (per-row symmetric quant):
     *   query_proj  [hidden_dim]
     *   key_proj    [hidden_dim]
     *   value_proj  [hidden_dim]
     *   out_proj    [hidden_dim]
     *   ip_k_stacked [num_blocks * hidden_dim]   (row = block*hidden + row_in_block)
     *   ip_v_stacked [num_blocks * hidden_dim]
     *
     * Dequant example (query_proj row i):
     *   float *w = (float *)ip->query_proj + i * ip->hidden_dim;  // cast INT8→F32 * scale[i]
     * ---------------------------------------------------------------- */
    float *query_proj_scale;
    float *key_proj_scale;
    float *value_proj_scale;
    float *out_proj_scale;
    float *ip_k_scale;        /* [num_blocks * hidden_dim] */
    float *ip_v_scale;        /* [num_blocks * hidden_dim] */

    /* Opaque handle to the open safetensors file (keeps mmap alive) */
    void  *_sf_handle;

    /* Backing heap allocation for F32-converted weights (if not mmap direct) */
    void  *_heap;
} iris_ip_adapter_t;


/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

/*
 * iris_ip_adapter_load — open bundle at bundle_dir, populate struct.
 *
 * Returns NULL on error (prints diagnostic to stderr).
 * The returned pointer must be freed with iris_ip_adapter_free().
 */
iris_ip_adapter_t *iris_ip_adapter_load(const char *bundle_dir);

/*
 * iris_ip_adapter_free — release all resources.
 */
void iris_ip_adapter_free(iris_ip_adapter_t *a);


/* -------------------------------------------------------------------------
 * Inference
 * ---------------------------------------------------------------------- */

/*
 * iris_ip_adapter_perceive — PerceiverResampler forward pass.
 *
 * Compresses SigLIP features [n_siglip, siglip_dim] to a fixed set of
 * image tokens [num_image_tokens, hidden_dim] via cross-attention with
 * learned query embeddings.
 *
 * Inputs:
 *   siglip_feats  float32 [n_siglip * siglip_dim]   — SigLIP encoder output
 *   n_siglip      number of SigLIP tokens            — typically 256 or 729
 *
 * Output:
 *   ip_embeds     caller-allocated float32 [num_image_tokens * hidden_dim]
 */
void iris_ip_adapter_perceive(
    const iris_ip_adapter_t *a,
    const float *siglip_feats,   int n_siglip,
          float *ip_embeds        /* [num_image_tokens * hidden_dim] */
);

/*
 * iris_ip_adapter_get_kv — project ip_embeds to K/V for transformer block block_idx.
 *
 * Inputs:
 *   block_idx   0-based index into the stacked weight tensors
 *   ip_embeds   float32 [num_image_tokens * hidden_dim]
 *
 * Outputs (caller-allocated):
 *   k_ip  float32 [num_image_tokens * hidden_dim]
 *   v_ip  float32 [num_image_tokens * hidden_dim]
 */
void iris_ip_adapter_get_kv(
    const iris_ip_adapter_t *a,
    int block_idx,
    const float *ip_embeds,
          float *k_ip,            /* [num_image_tokens * hidden_dim] */
          float *v_ip             /* [num_image_tokens * hidden_dim] */
);

/*
 * iris_ip_adapter_inject — compute IP cross-attention and accumulate into img_hidden.
 *
 * Computes:
 *   attn = scaled_dot_product_attention(img_q, k_ip, v_ip)   // [img_seq, hidden_dim]
 *   img_hidden += ip_scale[block_idx] * attn
 *
 * Inputs:
 *   block_idx       0-based block index
 *   img_q           float32 [img_seq * hidden_dim]  — query from Flux block
 *   img_seq         number of image patch tokens
 *   k_ip, v_ip      float32 [num_image_tokens * hidden_dim]  — from get_kv
 *
 * In/out:
 *   img_hidden      float32 [img_seq * hidden_dim]  — modified in-place
 *
 * When ip_scale[block_idx] == 0 (style_only mode, double-stream blocks)
 * this function is a no-op and returns immediately.
 */
void iris_ip_adapter_inject(
    const iris_ip_adapter_t *a,
    int block_idx,
    const float *img_q,          int img_seq,
    const float *k_ip,
    const float *v_ip,
          float *img_hidden       /* [img_seq * hidden_dim]  modified in-place */
);


/* -------------------------------------------------------------------------
 * Usage example (pseudo-code, within Flux double block forward pass)
 * ---------------------------------------------------------------------- */

/*
 * // --- Setup (once at model load) ---
 * iris_ip_adapter_t *ip = iris_ip_adapter_load("/path/to/bundle");
 *
 * // --- Per image (once before denoising loop) ---
 * float *siglip = encode_siglip(reference_image);          // [n_siglip, siglip_dim]
 * float *ip_embeds = malloc(ip->num_image_tokens * ip->hidden_dim * sizeof(float));
 * iris_ip_adapter_perceive(ip, siglip, n_siglip, ip_embeds);
 *
 * float *k_ip = malloc(ip->num_image_tokens * ip->hidden_dim * sizeof(float));
 * float *v_ip = malloc(ip->num_image_tokens * ip->hidden_dim * sizeof(float));
 *
 * // --- Inside Flux denoising loop, per block ---
 * for (int block = 0; block < ip->num_blocks; block++) {
 *     iris_ip_adapter_get_kv(ip, block, ip_embeds, k_ip, v_ip);
 *
 *     // ... Flux block forward pass produces img_hidden ...
 *
 *     iris_ip_adapter_inject(ip, block, img_q, img_seq, k_ip, v_ip, img_hidden);
 * }
 *
 * // --- Cleanup ---
 * free(k_ip); free(v_ip); free(ip_embeds);
 * iris_ip_adapter_free(ip);
 */

#ifdef __cplusplus
}
#endif
