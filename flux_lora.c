/*
 * flux_lora.c - LoRA support for the FLUX transformer
 *
 * Supports three safetensors formats (auto-detected):
 *   XLabs     - "double_blocks.N.processor.qkv_lora1.down.weight"
 *   Kohya     - "lora_unet_double_blocks_N_img_attn_qkv.lora_down.weight"
 *   Diffusers - "transformer.transformer_blocks.N.attn.to_q.lora_A.weight"
 *
 * All three formats support double-stream blocks (img+txt Q/K/V + output proj).
 * Kohya additionally supports single-stream blocks (linear1 fused QKV+MLP, linear2 fused proj).
 */

#include "flux_lora.h"
#include "flux_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* ========================================================================
 * Math: apply LoRA correction
 * ======================================================================== */

/*
 * Apply LoRA: out += scale * lora_B @ (lora_A @ x^T)
 *
 * x:       [seq_len, in_dim]   input (the same input that was fed to the linear layer)
 * out:     [seq_len, out_dim]  output (modified in-place)
 * scratch: [seq_len, rank]     caller-provided workspace
 */
void lora_apply(const lora_adapter_t *a, float scale,
                const float *x, float *out, int seq_len, float *scratch) {
    int rank = a->rank;
    int in_dim = a->in_dim;
    int out_dim = a->out_dim;

#ifdef USE_BLAS
    /* Step 1: scratch = x @ lora_A^T  →  [seq_len, rank] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, rank, in_dim,
                1.0f,
                x, in_dim,
                a->lora_A, in_dim,
                0.0f, scratch, rank);

    /* Step 2: out += scale * scratch @ lora_B^T  →  [seq_len, out_dim] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, rank,
                scale,
                scratch, rank,
                a->lora_B, rank,
                1.0f, out, out_dim);
#else
    /* Naive fallback (slow but correct) */
    /* Step 1: scratch[s, r] = sum_d x[s,d] * lora_A[r,d] */
    for (int s = 0; s < seq_len; s++) {
        for (int r = 0; r < rank; r++) {
            float sum = 0.0f;
            for (int d = 0; d < in_dim; d++) {
                sum += x[s * in_dim + d] * a->lora_A[r * in_dim + d];
            }
            scratch[s * rank + r] = sum;
        }
    }
    /* Step 2: out[s, o] += scale * sum_r scratch[s,r] * lora_B[o,r] */
    for (int s = 0; s < seq_len; s++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            for (int r = 0; r < rank; r++) {
                sum += scratch[s * rank + r] * a->lora_B[o * rank + r];
            }
            out[s * out_dim + o] += scale * sum;
        }
    }
#endif
}

/* ========================================================================
 * Format detection
 * ======================================================================== */

typedef enum {
    FORMAT_UNKNOWN = 0,
    FORMAT_XLABS,
    FORMAT_KOHYA,
    FORMAT_DIFFUSERS,
} lora_format_t;

static lora_format_t detect_format(const safetensors_file_t *sf) {
    for (int i = 0; i < sf->num_tensors; i++) {
        const char *name = sf->tensors[i].name;
        if (strstr(name, "processor."))      return FORMAT_XLABS;
        if (strstr(name, "lora_unet_"))      return FORMAT_KOHYA;
        if (strstr(name, "transformer.transformer_blocks") ||
            strstr(name, "transformer.single_transformer_blocks") ||
            strstr(name, "transformer.x_embedder"))
            return FORMAT_DIFFUSERS;
    }
    return FORMAT_UNKNOWN;
}

/* ========================================================================
 * Adapter helpers
 * ======================================================================== */

static lora_adapter_t *alloc_adapters(int n) {
    lora_adapter_t *arr = calloc(n, sizeof(lora_adapter_t));
    return arr;
}

static void free_adapter(lora_adapter_t *a) {
    if (!a) return;
    free(a->lora_A);
    free(a->lora_B);
    a->lora_A = NULL;
    a->lora_B = NULL;
}

/*
 * Load one simple (non-fused) adapter from a safetensors file.
 * down_key: key for lora_A ([rank, in_dim])
 * up_key:   key for lora_B ([out_dim, rank])
 * expected_in/out: expected dimensions (-1 to skip check)
 */
static int load_adapter(lora_adapter_t *a, const safetensors_file_t *sf,
                        const char *down_key, const char *up_key, int *max_rank,
                        int expected_in, int expected_out) {
    const safetensor_t *down = safetensors_find(sf, down_key);
    const safetensor_t *up   = safetensors_find(sf, up_key);
    if (!down || !up) return 0;
    if (down->ndim < 2 || up->ndim < 2) return 0;

    int rank    = (int)down->shape[0];
    int in_dim  = (int)down->shape[1];
    int out_dim = (int)up->shape[0];

    if (expected_in  >= 0 && in_dim  != expected_in) {
        fprintf(stderr, "LoRA: skipping %s: in_dim=%d (expected %d)\n",
                down_key, in_dim, expected_in);
        return 0;
    }
    if (expected_out >= 0 && out_dim != expected_out) {
        fprintf(stderr, "LoRA: skipping %s: out_dim=%d (expected %d)\n",
                up_key, out_dim, expected_out);
        return 0;
    }

    a->rank    = rank;
    a->in_dim  = in_dim;
    a->out_dim = out_dim;
    a->lora_A  = safetensors_get_f32(sf, down);  /* [rank, in_dim] */
    a->lora_B  = safetensors_get_f32(sf, up);    /* [out_dim, rank] */

    if (!a->lora_A || !a->lora_B) {
        free(a->lora_A); free(a->lora_B);
        a->lora_A = a->lora_B = NULL;
        return 0;
    }

    if (rank > *max_rank) *max_rank = rank;
    return 1;
}

/*
 * Load fused QKV adapter and split into separate Q, K, V adapters.
 * up_weight is [hidden*3, rank]; split into 3 × [hidden, rank].
 * lora_A is shared (duplicated) across Q, K, V.
 */
static int load_adapter_split_qkv(lora_adapter_t *aq, lora_adapter_t *ak, lora_adapter_t *av,
                                   const safetensors_file_t *sf,
                                   const char *down_key, const char *up_key,
                                   int hidden, int *max_rank) {
    const safetensor_t *down = safetensors_find(sf, down_key);
    const safetensor_t *up   = safetensors_find(sf, up_key);
    if (!down || !up) return 0;
    if (down->ndim < 2 || up->ndim < 2) return 0;

    int rank   = (int)down->shape[0];
    int in_dim = (int)down->shape[1];
    int out_dim_fused = (int)up->shape[0];  /* should be hidden*3 */

    if (out_dim_fused != hidden * 3) {
        fprintf(stderr, "LoRA: unexpected fused QKV out_dim %d (expected %d)\n",
                out_dim_fused, hidden * 3);
        return 0;
    }

    float *down_f32 = safetensors_get_f32(sf, down);  /* [rank, in_dim] */
    float *up_f32   = safetensors_get_f32(sf, up);    /* [hidden*3, rank] */
    if (!down_f32 || !up_f32) {
        free(down_f32); free(up_f32);
        return 0;
    }

    size_t a_bytes = (size_t)rank * in_dim * sizeof(float);
    size_t b_bytes = (size_t)hidden * rank * sizeof(float);

    /* Q */
    aq->rank = rank; aq->in_dim = in_dim; aq->out_dim = hidden;
    aq->lora_A = malloc(a_bytes);
    aq->lora_B = malloc(b_bytes);

    /* K */
    ak->rank = rank; ak->in_dim = in_dim; ak->out_dim = hidden;
    ak->lora_A = malloc(a_bytes);
    ak->lora_B = malloc(b_bytes);

    /* V */
    av->rank = rank; av->in_dim = in_dim; av->out_dim = hidden;
    av->lora_A = malloc(a_bytes);
    av->lora_B = malloc(b_bytes);

    if (!aq->lora_A || !aq->lora_B || !ak->lora_A || !ak->lora_B ||
        !av->lora_A || !av->lora_B) {
        free(aq->lora_A); free(aq->lora_B);
        free(ak->lora_A); free(ak->lora_B);
        free(av->lora_A); free(av->lora_B);
        free(down_f32); free(up_f32);
        return 0;
    }

    /* Copy shared lora_A to all three */
    memcpy(aq->lora_A, down_f32, a_bytes);
    memcpy(ak->lora_A, down_f32, a_bytes);
    memcpy(av->lora_A, down_f32, a_bytes);

    /* Split lora_B: first hidden rows → Q, next → K, last → V */
    memcpy(aq->lora_B, up_f32,                         b_bytes);
    memcpy(ak->lora_B, up_f32 + hidden * rank,         b_bytes);
    memcpy(av->lora_B, up_f32 + hidden * rank * 2,     b_bytes);

    free(down_f32);
    free(up_f32);

    if (rank > *max_rank) *max_rank = rank;
    return 1;
}

/* ========================================================================
 * XLabs format loading
 * ======================================================================== */

static void load_xlabs(lora_state_t *lora, const safetensors_file_t *sf, int hidden) {
    int loaded = 0;
    char dk[256], uk[256];

    for (int i = 0; i < lora->num_double_blocks; i++) {
        /* Image stream: fused QKV (lora1) */
        snprintf(dk, sizeof(dk), "double_blocks.%d.processor.qkv_lora1.down.weight", i);
        snprintf(uk, sizeof(uk), "double_blocks.%d.processor.qkv_lora1.up.weight", i);
        if (load_adapter_split_qkv(&lora->double_img_q[i], &lora->double_img_k[i],
                                    &lora->double_img_v[i], sf, dk, uk, hidden, &lora->max_rank))
            loaded++;

        /* Image stream: output projection (lora1) */
        snprintf(dk, sizeof(dk), "double_blocks.%d.processor.proj_lora1.down.weight", i);
        snprintf(uk, sizeof(uk), "double_blocks.%d.processor.proj_lora1.up.weight", i);
        if (load_adapter(&lora->double_img_proj[i], sf, dk, uk, &lora->max_rank, hidden, hidden))
            loaded++;

        /* Text stream: fused QKV (lora2) */
        snprintf(dk, sizeof(dk), "double_blocks.%d.processor.qkv_lora2.down.weight", i);
        snprintf(uk, sizeof(uk), "double_blocks.%d.processor.qkv_lora2.up.weight", i);
        if (load_adapter_split_qkv(&lora->double_txt_q[i], &lora->double_txt_k[i],
                                    &lora->double_txt_v[i], sf, dk, uk, hidden, &lora->max_rank))
            loaded++;

        /* Text stream: output projection (lora2) */
        snprintf(dk, sizeof(dk), "double_blocks.%d.processor.proj_lora2.down.weight", i);
        snprintf(uk, sizeof(uk), "double_blocks.%d.processor.proj_lora2.up.weight", i);
        if (load_adapter(&lora->double_txt_proj[i], sf, dk, uk, &lora->max_rank, hidden, hidden))
            loaded++;
    }

    fprintf(stderr, "LoRA: loaded %d XLabs adapters across %d double blocks\n",
            loaded, lora->num_double_blocks);
}

/* ========================================================================
 * Kohya format loading
 * ======================================================================== */

static void load_kohya(lora_state_t *lora, const safetensors_file_t *sf, int hidden) {
    int loaded = 0;
    char dk[256], uk[256];

    for (int i = 0; i < lora->num_double_blocks; i++) {
        /* Image stream: fused QKV */
        snprintf(dk, sizeof(dk), "lora_unet_double_blocks_%d_img_attn_qkv.lora_down.weight", i);
        snprintf(uk, sizeof(uk), "lora_unet_double_blocks_%d_img_attn_qkv.lora_up.weight", i);
        if (load_adapter_split_qkv(&lora->double_img_q[i], &lora->double_img_k[i],
                                    &lora->double_img_v[i], sf, dk, uk, hidden, &lora->max_rank))
            loaded++;

        /* Image stream: output projection */
        snprintf(dk, sizeof(dk), "lora_unet_double_blocks_%d_img_attn_proj.lora_down.weight", i);
        snprintf(uk, sizeof(uk), "lora_unet_double_blocks_%d_img_attn_proj.lora_up.weight", i);
        if (load_adapter(&lora->double_img_proj[i], sf, dk, uk, &lora->max_rank, hidden, hidden))
            loaded++;

        /* Text stream: fused QKV */
        snprintf(dk, sizeof(dk), "lora_unet_double_blocks_%d_txt_attn_qkv.lora_down.weight", i);
        snprintf(uk, sizeof(uk), "lora_unet_double_blocks_%d_txt_attn_qkv.lora_up.weight", i);
        if (load_adapter_split_qkv(&lora->double_txt_q[i], &lora->double_txt_k[i],
                                    &lora->double_txt_v[i], sf, dk, uk, hidden, &lora->max_rank))
            loaded++;

        /* Text stream: output projection */
        snprintf(dk, sizeof(dk), "lora_unet_double_blocks_%d_txt_attn_proj.lora_down.weight", i);
        snprintf(uk, sizeof(uk), "lora_unet_double_blocks_%d_txt_attn_proj.lora_up.weight", i);
        if (load_adapter(&lora->double_txt_proj[i], sf, dk, uk, &lora->max_rank, hidden, hidden))
            loaded++;
    }

    /* Single blocks: Kohya uses fused linear1 (QKV+MLP) and linear2 (proj+mlp_down).
     * Dimensions vary by model, so no fixed expected size (-1 = no check). */
    for (int i = 0; i < lora->num_single_blocks; i++) {
        snprintf(dk, sizeof(dk), "lora_unet_single_blocks_%d_linear1.lora_down.weight", i);
        snprintf(uk, sizeof(uk), "lora_unet_single_blocks_%d_linear1.lora_up.weight", i);
        if (load_adapter(&lora->single_linear1[i], sf, dk, uk, &lora->max_rank, -1, -1))
            loaded++;

        snprintf(dk, sizeof(dk), "lora_unet_single_blocks_%d_linear2.lora_down.weight", i);
        snprintf(uk, sizeof(uk), "lora_unet_single_blocks_%d_linear2.lora_up.weight", i);
        if (load_adapter(&lora->single_linear2[i], sf, dk, uk, &lora->max_rank, -1, -1))
            loaded++;
    }

    fprintf(stderr, "LoRA: loaded %d Kohya adapters across %d double + %d single blocks\n",
            loaded, lora->num_double_blocks, lora->num_single_blocks);
}

/* ========================================================================
 * Diffusers format loading
 * ======================================================================== */

static void load_diffusers(lora_state_t *lora, const safetensors_file_t *sf, int hidden)
{
    int loaded = 0;
    char dk[256], uk[256];

    for (int i = 0; i < lora->num_double_blocks; i++) {
        /* Image stream: separate Q, K, V */
        snprintf(dk, sizeof(dk), "transformer.transformer_blocks.%d.attn.to_q.lora_A.weight", i);
        snprintf(uk, sizeof(uk), "transformer.transformer_blocks.%d.attn.to_q.lora_B.weight", i);
        if (load_adapter(&lora->double_img_q[i], sf, dk, uk, &lora->max_rank, hidden, hidden)) loaded++;

        snprintf(dk, sizeof(dk), "transformer.transformer_blocks.%d.attn.to_k.lora_A.weight", i);
        snprintf(uk, sizeof(uk), "transformer.transformer_blocks.%d.attn.to_k.lora_B.weight", i);
        if (load_adapter(&lora->double_img_k[i], sf, dk, uk, &lora->max_rank, hidden, hidden)) loaded++;

        snprintf(dk, sizeof(dk), "transformer.transformer_blocks.%d.attn.to_v.lora_A.weight", i);
        snprintf(uk, sizeof(uk), "transformer.transformer_blocks.%d.attn.to_v.lora_B.weight", i);
        if (load_adapter(&lora->double_img_v[i], sf, dk, uk, &lora->max_rank, hidden, hidden)) loaded++;

        snprintf(dk, sizeof(dk), "transformer.transformer_blocks.%d.attn.to_out.0.lora_A.weight", i);
        snprintf(uk, sizeof(uk), "transformer.transformer_blocks.%d.attn.to_out.0.lora_B.weight", i);
        if (load_adapter(&lora->double_img_proj[i], sf, dk, uk, &lora->max_rank, hidden, hidden)) loaded++;

        /* Text stream: add_q_proj, add_k_proj, add_v_proj, to_add_out */
        snprintf(dk, sizeof(dk), "transformer.transformer_blocks.%d.attn.add_q_proj.lora_A.weight", i);
        snprintf(uk, sizeof(uk), "transformer.transformer_blocks.%d.attn.add_q_proj.lora_B.weight", i);
        if (load_adapter(&lora->double_txt_q[i], sf, dk, uk, &lora->max_rank, hidden, hidden)) loaded++;

        snprintf(dk, sizeof(dk), "transformer.transformer_blocks.%d.attn.add_k_proj.lora_A.weight", i);
        snprintf(uk, sizeof(uk), "transformer.transformer_blocks.%d.attn.add_k_proj.lora_B.weight", i);
        if (load_adapter(&lora->double_txt_k[i], sf, dk, uk, &lora->max_rank, hidden, hidden)) loaded++;

        snprintf(dk, sizeof(dk), "transformer.transformer_blocks.%d.attn.add_v_proj.lora_A.weight", i);
        snprintf(uk, sizeof(uk), "transformer.transformer_blocks.%d.attn.add_v_proj.lora_B.weight", i);
        if (load_adapter(&lora->double_txt_v[i], sf, dk, uk, &lora->max_rank, hidden, hidden)) loaded++;

        snprintf(dk, sizeof(dk), "transformer.transformer_blocks.%d.attn.to_add_out.lora_A.weight", i);
        snprintf(uk, sizeof(uk), "transformer.transformer_blocks.%d.attn.to_add_out.lora_B.weight", i);
        if (load_adapter(&lora->double_txt_proj[i], sf, dk, uk, &lora->max_rank, hidden, hidden)) loaded++;
    }

    /* Single blocks: proj_out → single_linear2 (dims vary, no fixed size check) */
    for (int i = 0; i < lora->num_single_blocks; i++) {
        /* For diffusers single blocks, Q/K/V are separate but use the same input (norm).
         * We load them as separate adapters and store in single_linear1_q/k/v.
         * Since our struct doesn't have separate single Q/K/V, we skip for now and
         * note in the warning below. The user can use Kohya format for full single block support. */

        /* proj_out → single_linear2 */
        snprintf(dk, sizeof(dk), "transformer.single_transformer_blocks.%d.proj_out.lora_A.weight", i);
        snprintf(uk, sizeof(uk), "transformer.single_transformer_blocks.%d.proj_out.lora_B.weight", i);
        if (load_adapter(&lora->single_linear2[i], sf, dk, uk, &lora->max_rank, -1, -1)) loaded++;
    }

    if (lora->num_single_blocks > 0) {
        fprintf(stderr, "LoRA: note: diffusers single-block Q/K/V LoRAs not supported "
                "(only proj_out loaded); use Kohya format for full single block support\n");
    }

    fprintf(stderr, "LoRA: loaded %d diffusers adapters across %d double + %d single blocks\n",
            loaded, lora->num_double_blocks, lora->num_single_blocks);
}

/* ========================================================================
 * Public API
 * ======================================================================== */

lora_state_t *lora_load(const char *path, int num_double_blocks,
                         int num_single_blocks, int hidden_size,
                         float scale) {
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) {
        fprintf(stderr, "LoRA: failed to open %s\n", path);
        return NULL;
    }

    lora_format_t fmt = detect_format(sf);
    if (fmt == FORMAT_UNKNOWN) {
        fprintf(stderr, "LoRA: unknown format in %s\n", path);
        safetensors_close(sf);
        return NULL;
    }

    const char *fmt_names[] = {"unknown", "XLabs", "Kohya", "Diffusers"};
    fprintf(stderr, "LoRA: loading %s format from %s\n", fmt_names[fmt], path);
    fprintf(stderr, "LoRA: model has %d double blocks, %d single blocks, hidden=%d\n",
            num_double_blocks, num_single_blocks, hidden_size);

    lora_state_t *lora = calloc(1, sizeof(lora_state_t));
    if (!lora) {
        safetensors_close(sf);
        return NULL;
    }

    lora->scale             = scale;
    lora->num_double_blocks = num_double_blocks;
    lora->num_single_blocks = num_single_blocks;

    /* Allocate adapter arrays */
    lora->double_img_q    = alloc_adapters(num_double_blocks);
    lora->double_img_k    = alloc_adapters(num_double_blocks);
    lora->double_img_v    = alloc_adapters(num_double_blocks);
    lora->double_img_proj = alloc_adapters(num_double_blocks);
    lora->double_txt_q    = alloc_adapters(num_double_blocks);
    lora->double_txt_k    = alloc_adapters(num_double_blocks);
    lora->double_txt_v    = alloc_adapters(num_double_blocks);
    lora->double_txt_proj = alloc_adapters(num_double_blocks);
    lora->single_linear1  = alloc_adapters(num_single_blocks);
    lora->single_linear2  = alloc_adapters(num_single_blocks);

    if (!lora->double_img_q || !lora->double_img_k || !lora->double_img_v ||
        !lora->double_img_proj || !lora->double_txt_q || !lora->double_txt_k ||
        !lora->double_txt_v || !lora->double_txt_proj ||
        !lora->single_linear1 || !lora->single_linear2) {
        lora_free(lora);
        safetensors_close(sf);
        return NULL;
    }

    switch (fmt) {
        case FORMAT_XLABS:     load_xlabs(lora, sf, hidden_size);     break;
        case FORMAT_KOHYA:     load_kohya(lora, sf, hidden_size);     break;
        case FORMAT_DIFFUSERS: load_diffusers(lora, sf, hidden_size); break;
        default: break;
    }

    safetensors_close(sf);

    if (lora->max_rank == 0) {
        fprintf(stderr, "LoRA: no adapters loaded (wrong format or model mismatch?)\n");
        lora_free(lora);
        return NULL;
    }

    /* Allocate scratch buffer: large enough for max_rank × any sequence length.
     * Max sequence: 1792×1792 image at 16x compression = 112×112 = 12544 tokens + 512 text = 13056 */
    lora->scratch_len = 14000;
    lora->scratch = malloc((size_t)lora->max_rank * lora->scratch_len * sizeof(float));
    if (!lora->scratch) {
        lora_free(lora);
        return NULL;
    }

    fprintf(stderr, "LoRA: ready (max_rank=%d, scale=%.2f)\n", lora->max_rank, scale);
    return lora;
}

void lora_free(lora_state_t *lora) {
    if (!lora) return;

    for (int i = 0; i < lora->num_double_blocks; i++) {
        free_adapter(&lora->double_img_q[i]);
        free_adapter(&lora->double_img_k[i]);
        free_adapter(&lora->double_img_v[i]);
        free_adapter(&lora->double_img_proj[i]);
        free_adapter(&lora->double_txt_q[i]);
        free_adapter(&lora->double_txt_k[i]);
        free_adapter(&lora->double_txt_v[i]);
        free_adapter(&lora->double_txt_proj[i]);
    }
    for (int i = 0; i < lora->num_single_blocks; i++) {
        free_adapter(&lora->single_linear1[i]);
        free_adapter(&lora->single_linear2[i]);
    }

    free(lora->double_img_q);
    free(lora->double_img_k);
    free(lora->double_img_v);
    free(lora->double_img_proj);
    free(lora->double_txt_q);
    free(lora->double_txt_k);
    free(lora->double_txt_v);
    free(lora->double_txt_proj);
    free(lora->single_linear1);
    free(lora->single_linear2);
    free(lora->scratch);
    free(lora);
}
