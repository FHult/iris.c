/*
 * iris_ip_adapter.c — C inference for trained IP-Adapter bundles (G-1, Phase 1).
 *
 * Implements iris_ip_adapter.h: load a bundle exported by
 * export_adapter.py and run the three stages the Flux denoiser needs —
 *   perceive : SigLIP feats -> ip_embeds   (PerceiverResampler: MHA + LayerNorm)
 *   get_kv   : ip_embeds     -> k_ip,v_ip   (per-block projection)
 *   inject   : img_q,k,v     -> contribution (scale * SDPA), added to img_hidden
 *
 * Math mirrors train/ip_adapter/model.py exactly (validated by debug/test_ip_adapter.c
 * against committed goldens):
 *   - nn.Linear is x @ W^T (bias=False here), W stored [out, in].
 *   - nn.MultiHeadAttention: proj -> reshape to (heads, seq, head_dim) ->
 *     SDPA(scale = 1/sqrt(head_dim)) -> flatten -> out_proj.
 *   - nn.LayerNorm: (x-mean)/sqrt(var+eps)*w + b, eps 1e-5, biased var.
 *   - get_kv: einsum btd,de->te  ==  ip_embeds @ ip_{k,v}_stacked[block].
 *   - inject: scale[block] * SDPA(img_q, k_ip, v_ip), no out_proj.
 *
 * Weights are dequantised to f32 at load: f16/bf16/f32 via safetensors_get_f32, int8
 * via per-row scale (load_tensor_f32). CPU path; uses iris_kernels GEMM/attention.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "iris_safetensors.h"
#include "iris_kernels.h"
#include "iris_config_parse.h"
#include "iris_ip_adapter.h"

#define LN_EPS 1e-5f

/* [seq, heads*head_dim] -> [heads, seq, head_dim] (head-major) */
static void to_heads(float *dst, const float *src, int seq, int heads, int head_dim) {
    int hidden = heads * head_dim;
    for (int h = 0; h < heads; h++)
        for (int s = 0; s < seq; s++)
            memcpy(dst + (h * seq + s) * head_dim,
                   src + s * hidden + h * head_dim,
                   head_dim * sizeof(float));
}

/* [heads, seq, head_dim] -> [seq, heads*head_dim] */
static void from_heads(float *dst, const float *src, int seq, int heads, int head_dim) {
    int hidden = heads * head_dim;
    for (int h = 0; h < heads; h++)
        for (int s = 0; s < seq; s++)
            memcpy(dst + s * hidden + h * head_dim,
                   src + (h * seq + s) * head_dim,
                   head_dim * sizeof(float));
}

/* Multi-head SDPA over flat [seq,hidden] tensors: out[seq_q,hidden] = MHA(q,k,v).
 * scale = 1/sqrt(head_dim). Reuses iris_attention (expects [heads,seq,head_dim]). */
static int mha_sdpa(float *out, const float *q, const float *k, const float *v,
                    int seq_q, int seq_k, int heads, int head_dim) {
    int hidden = heads * head_dim;
    float *qh = malloc((size_t)heads * seq_q * head_dim * sizeof(float));
    float *kh = malloc((size_t)heads * seq_k * head_dim * sizeof(float));
    float *vh = malloc((size_t)heads * seq_k * head_dim * sizeof(float));
    float *ah = malloc((size_t)heads * seq_q * head_dim * sizeof(float));
    if (!qh || !kh || !vh || !ah) { free(qh); free(kh); free(vh); free(ah); return -1; }
    to_heads(qh, q, seq_q, heads, head_dim);
    to_heads(kh, k, seq_k, heads, head_dim);
    to_heads(vh, v, seq_k, heads, head_dim);
    iris_attention(ah, qh, kh, vh, 1, heads, seq_q, seq_k, head_dim,
                   1.0f / sqrtf((float)head_dim));
    from_heads(out, ah, seq_q, heads, head_dim);
    (void)hidden;
    free(qh); free(kh); free(vh); free(ah);
    return 0;
}

/* Load one tensor as f32. Passes f32/f16/bf16 through safetensors_get_f32; for an
 * int8 tensor, dequantises with the companion "<name>.scale" (per-row symmetric:
 * x[r,:] = q[r,:] * scale[r], rows = numel/last_dim) — the export's int8 format. */
static float *load_tensor_f32(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) return NULL;
    if (t->dtype != DTYPE_I8) return safetensors_get_f32(sf, t);

    char sname[300];
    snprintf(sname, sizeof(sname), "%s.scale", name);
    const safetensor_t *st = safetensors_find(sf, sname);
    if (!st) return NULL;
    float *scale = safetensors_get_f32(sf, st);
    const int8_t *q = (const int8_t *)safetensors_data(sf, t);
    if (!scale || !q) { free(scale); return NULL; }

    int64_t n = safetensor_numel(t);
    int64_t cols = (t->ndim > 0) ? t->shape[t->ndim - 1] : n;
    int64_t rows = (cols > 0) ? n / cols : 0;
    float *out = malloc((size_t)n * sizeof(float));
    if (out) {
        for (int64_t r = 0; r < rows; r++)
            for (int64_t c = 0; c < cols; c++)
                out[r * cols + c] = (float)q[r * cols + c] * scale[r];
    }
    free(scale);
    return out;
}

iris_ip_adapter_t *iris_ip_adapter_load(const char *bundle_dir) {
    if (!bundle_dir) return NULL;
    char path[1024];

    iris_ip_adapter_t *a = calloc(1, sizeof(iris_ip_adapter_t));
    if (!a) return NULL;

    /* meta.json -> dims */
    snprintf(path, sizeof(path), "%s/adapter_meta.json", bundle_dir);
    FILE *mf = fopen(path, "r");
    if (!mf) { free(a); return NULL; }
    char meta[4096];
    size_t mn = fread(meta, 1, sizeof(meta) - 1, mf);
    meta[mn] = '\0';
    fclose(mf);
    a->num_blocks        = cfg_int(meta, "num_blocks", 0);
    a->num_double_blocks = cfg_int(meta, "num_double_blocks", 0);
    a->num_single_blocks = cfg_int(meta, "num_single_blocks", 0);
    a->hidden_dim        = cfg_int(meta, "hidden_dim", 0);
    a->num_image_tokens  = cfg_int(meta, "num_image_tokens", 0);
    a->siglip_dim        = cfg_int(meta, "siglip_dim", 0);
    a->style_only        = cfg_bool(meta, "style_only", 0);
    if (cfg_find_value(meta, "quant")) {
        const char *q = cfg_find_value(meta, "quant");
        while (*q == ' ' || *q == '"') q++;
        strncpy(a->quant, q, sizeof(a->quant) - 1);
        char *e = strchr(a->quant, '"'); if (e) *e = '\0';
    }
    if (a->hidden_dim <= 0 || a->num_blocks <= 0) { free(a); return NULL; }

    /* weights (dequantised to f32) */
    snprintf(path, sizeof(path), "%s/adapter_weights.safetensors", bundle_dir);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { free(a); return NULL; }
    a->_sf_handle = sf;

    #define LOAD(field, name) a->field = load_tensor_f32(sf, name)
    LOAD(query_tokens, "perceiver.query_tokens");
    LOAD(query_proj,   "perceiver.query_proj");
    LOAD(key_proj,     "perceiver.key_proj");
    LOAD(value_proj,   "perceiver.value_proj");
    LOAD(out_proj,     "perceiver.out_proj");
    LOAD(norm_weight,  "perceiver.norm_weight");
    LOAD(norm_bias,    "perceiver.norm_bias");
    LOAD(ip_k_stacked, "ip_k_stacked");
    LOAD(ip_v_stacked, "ip_v_stacked");
    LOAD(ip_scale,     "ip_scale");
    #undef LOAD

    if (!a->query_tokens || !a->query_proj || !a->key_proj || !a->value_proj ||
        !a->out_proj || !a->norm_weight || !a->norm_bias ||
        !a->ip_k_stacked || !a->ip_v_stacked || !a->ip_scale) {
        iris_ip_adapter_free(a);
        return NULL;
    }
    /* f32 copies are owned by us now; the file can be closed. */
    safetensors_close(sf);
    a->_sf_handle = NULL;
    return a;
}

void iris_ip_adapter_free(iris_ip_adapter_t *a) {
    if (!a) return;
    free(a->query_tokens); free(a->query_proj); free(a->key_proj);
    free(a->value_proj); free(a->out_proj); free(a->norm_weight);
    free(a->norm_bias); free(a->ip_k_stacked); free(a->ip_v_stacked);
    free(a->ip_scale);
    if (a->_sf_handle) safetensors_close((safetensors_file_t *)a->_sf_handle);
    free(a);
}

void iris_ip_adapter_perceive(const iris_ip_adapter_t *a,
                              const float *siglip, int n_siglip,
                              float *ip_embeds) {
    int H = a->hidden_dim, S = a->siglip_dim;
    int T = a->num_image_tokens, hd = 128, heads = H / hd;

    float *Q = malloc((size_t)T * H * sizeof(float));
    float *K = malloc((size_t)n_siglip * H * sizeof(float));
    float *V = malloc((size_t)n_siglip * H * sizeof(float));
    float *attn = malloc((size_t)T * H * sizeof(float));
    if (!Q || !K || !V || !attn) { free(Q); free(K); free(V); free(attn); return; }

    /* projections: x @ W^T  (Linear, bias=False) */
    iris_matmul_t(Q, a->query_tokens, a->query_proj, T, H, H);
    iris_matmul_t(K, siglip, a->key_proj, n_siglip, S, H);
    iris_matmul_t(V, siglip, a->value_proj, n_siglip, S, H);

    mha_sdpa(attn, Q, K, V, T, n_siglip, heads, hd);

    /* out_proj then LayerNorm into ip_embeds */
    iris_matmul_t(ip_embeds, attn, a->out_proj, T, H, H);
    for (int t = 0; t < T; t++) {
        float *row = ip_embeds + (size_t)t * H;
        float mean = 0.0f;
        for (int i = 0; i < H; i++) mean += row[i];
        mean /= H;
        float var = 0.0f;
        for (int i = 0; i < H; i++) { float d = row[i] - mean; var += d * d; }
        var /= H;
        float inv = 1.0f / sqrtf(var + LN_EPS);
        for (int i = 0; i < H; i++)
            row[i] = (row[i] - mean) * inv * a->norm_weight[i] + a->norm_bias[i];
    }
    free(Q); free(K); free(V); free(attn);
}

void iris_ip_adapter_get_kv(const iris_ip_adapter_t *a, int block_idx,
                            const float *ip_embeds, float *k_ip, float *v_ip) {
    int H = a->hidden_dim, T = a->num_image_tokens;
    size_t off = (size_t)block_idx * H * H;
    /* einsum btd,de->te : ip_embeds @ stacked[block] (no transpose) */
    iris_matmul(k_ip, ip_embeds, a->ip_k_stacked + off, T, H, H);
    iris_matmul(v_ip, ip_embeds, a->ip_v_stacked + off, T, H, H);
}

void iris_ip_adapter_inject(const iris_ip_adapter_t *a, int block_idx,
                            const float *img_q, int img_seq,
                            const float *k_ip, const float *v_ip,
                            float *img_hidden) {
    float s = a->ip_scale[block_idx];
    if (s == 0.0f) return;                 /* style-only: double-stream no-op */
    int H = a->hidden_dim, T = a->num_image_tokens, hd = 128, heads = H / hd;
    float *attn = malloc((size_t)img_seq * H * sizeof(float));
    if (!attn) return;
    mha_sdpa(attn, img_q, k_ip, v_ip, img_seq, T, heads, hd);
    for (size_t i = 0; i < (size_t)img_seq * H; i++)
        img_hidden[i] += s * attn[i];
    free(attn);
}
