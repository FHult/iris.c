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

    int64_t n = safetensor_numel(t);
    int64_t cols = (t->ndim > 0) ? t->shape[t->ndim - 1] : n;
    int64_t rows = (cols > 0) ? n / cols : 0;
    /* The companion scale must have exactly one entry per row — a malformed
     * bundle would otherwise read past the scale buffer. */
    if (safetensor_numel(st) != rows) {
        fprintf(stderr, "ip_adapter: %s.scale has %lld entries, expected %lld rows\n",
                name, (long long)safetensor_numel(st), (long long)rows);
        return NULL;
    }
    float *scale = safetensors_get_f32(sf, st);
    const int8_t *q = (const int8_t *)safetensors_data(sf, t);
    if (!scale || !q) { free(scale); return NULL; }
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
    a->ip_sched_mult = 1.0f;   /* schedule off by default → constant injection */

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
    /* PerceiverResampler head count is INDEPENDENT of the Flux block head count.
     * The perceiver was trained with perceiver_heads (e.g. 16) and head_dim =
     * hidden_dim/perceiver_heads (e.g. 192), NOT the Flux invariant head_dim 128.
     * Fall back to hidden_dim/128 only for legacy bundles that predate this key
     * (where the trained perceiver happened to use that grouping). */
    a->perceiver_heads   = cfg_int(meta, "perceiver_heads", 0);
    a->style_only        = cfg_bool(meta, "style_only", 0);
    /* Conditioning mode (SREF-LEAK-2): default "siglip" when the key is absent. */
    strncpy(a->cond_mode, "siglip", sizeof(a->cond_mode) - 1);
    if (cfg_find_value(meta, "cond_mode")) {
        const char *c = cfg_find_value(meta, "cond_mode");
        while (*c == ' ' || *c == '"') c++;
        strncpy(a->cond_mode, c, sizeof(a->cond_mode) - 1);
        char *e = strchr(a->cond_mode, '"'); if (e) *e = '\0';
    }
    a->csd_dim = cfg_int(meta, "csd_dim", 768);
    if (cfg_find_value(meta, "quant")) {
        const char *q = cfg_find_value(meta, "quant");
        while (*q == ' ' || *q == '"') q++;
        strncpy(a->quant, q, sizeof(a->quant) - 1);
        char *e = strchr(a->quant, '"'); if (e) *e = '\0';
    }
    if (a->hidden_dim <= 0 || a->num_blocks <= 0) { free(a); return NULL; }
    if (a->perceiver_heads <= 0 || a->hidden_dim % a->perceiver_heads != 0)
        a->perceiver_heads = a->hidden_dim / 128;   /* legacy fallback */

    /* weights (dequantised to f32) */
    snprintf(path, sizeof(path), "%s/adapter_weights.safetensors", bundle_dir);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { free(a); return NULL; }
    a->_sf_handle = sf;

    #define LOAD(field, name) a->field = load_tensor_f32(sf, name)
    int is_csd    = (strcmp(a->cond_mode, "csd") == 0);
    int is_hybrid = (strcmp(a->cond_mode, "hybrid") == 0);
    LOAD(query_tokens, "perceiver.query_tokens");   /* csd: CSD queries; siglip/hybrid: perceiver */
    LOAD(norm_weight,  "perceiver.norm_weight");
    LOAD(norm_bias,    "perceiver.norm_bias");
    if (is_csd) {
        LOAD(film_weight, "perceiver.film_weight");
        LOAD(film_bias,   "perceiver.film_bias");
    } else {
        /* siglip OR hybrid: the SigLIP PerceiverResampler */
        LOAD(query_proj,   "perceiver.query_proj");
        LOAD(key_proj,     "perceiver.key_proj");
        LOAD(value_proj,   "perceiver.value_proj");
        LOAD(out_proj,     "perceiver.out_proj");
        LOAD(in_gamma,     "perceiver.in_gamma");   /* optional (legacy bundles lack it) */
        LOAD(in_beta,      "perceiver.in_beta");    /* optional */
    }
    if (is_hybrid) {
        /* hybrid: the CSD module's OWN weights (separate from the perceiver above) */
        LOAD(csd_query_tokens, "csd.query_tokens");
        LOAD(csd_film_weight,  "csd.film_weight");
        LOAD(csd_film_bias,    "csd.film_bias");
        LOAD(csd_norm_weight,  "csd.norm_weight");
        LOAD(csd_norm_bias,    "csd.norm_bias");
        LOAD(group_gate,       "group_gate");   /* optional leak-reduction gate (SREF-COMBINE-1) */
    }
    LOAD(ip_k_stacked, "ip_k_stacked");
    LOAD(ip_v_stacked, "ip_v_stacked");
    LOAD(ip_scale,     "ip_scale");
    #undef LOAD

    int ok = a->query_tokens && a->norm_weight && a->norm_bias &&
             a->ip_k_stacked && a->ip_v_stacked && a->ip_scale;
    if (is_csd)
        ok = ok && a->film_weight && a->film_bias;
    else
        ok = ok && a->query_proj && a->key_proj && a->value_proj && a->out_proj;
    if (is_hybrid)
        ok = ok && a->csd_query_tokens && a->csd_film_weight && a->csd_film_bias &&
             a->csd_norm_weight && a->csd_norm_bias;
    if (!ok) {
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
    free(a->ip_scale); free(a->in_gamma); free(a->in_beta);
    free(a->film_weight); free(a->film_bias);
    free(a->csd_query_tokens); free(a->csd_film_weight); free(a->csd_film_bias);
    free(a->csd_norm_weight); free(a->csd_norm_bias); free(a->group_gate);
    if (a->_sf_handle) safetensors_close((safetensors_file_t *)a->_sf_handle);
    free(a);
}

/* CSDImageProj FiLM path → out[T*H]. Generic over the weight set so cond_mode=="csd"
 * (whole adapter) and cond_mode=="hybrid" (the CSD half) share one implementation.
 * film = film_weight @ csd + film_bias → (scale, shift); token = qtok*(1+scale)+shift;
 * then per-token LayerNorm (biased var, LN_EPS). Mirrors CSDImageProj. */
static int perceive_csd_film(int H, int T, int C, const float *csd,
                             const float *qtok, const float *fw, const float *fb,
                             const float *nw, const float *nb, float *out) {
    float *film = malloc((size_t)2 * H * sizeof(float));
    if (!film) return -1;
    iris_matmul_t(film, csd, fw, 1, C, 2 * H);            /* [1,2H] = csd[1,C] @ fw[2H,C]^T */
    for (int i = 0; i < 2 * H; i++) film[i] += fb[i];
    const float *scale = film;       /* [H] */
    const float *shift = film + H;   /* [H] */
    for (int t = 0; t < T; t++) {
        float *row = out + (size_t)t * H;
        const float *q = qtok + (size_t)t * H;
        for (int d = 0; d < H; d++) row[d] = q[d] * (1.0f + scale[d]) + shift[d];
        float mean = 0.0f, var = 0.0f;
        for (int d = 0; d < H; d++) mean += row[d];
        mean /= H;
        for (int d = 0; d < H; d++) { float e = row[d] - mean; var += e * e; }
        var /= H;
        float inv = 1.0f / sqrtf(var + LN_EPS);
        for (int d = 0; d < H; d++)
            row[d] = (row[d] - mean) * inv * nw[d] + nb[d];
    }
    free(film);
    return 0;
}

/* SigLIP PerceiverResampler cross-attention → out[T*H] from n_siglip rows. Uses the
 * SigLIP perceiver weights (query_tokens/query_proj/key_proj/value_proj/out_proj/norm_*
 * + optional in_gamma/in_beta). T = output token count (num_image_tokens for siglip mode,
 * num_image_tokens/2 for the hybrid SigLIP half). */
static int perceive_siglip_mha(const iris_ip_adapter_t *a, const float *siglip,
                               int n_siglip, int T, float *out) {
    int H = a->hidden_dim, S = a->siglip_dim;
    /* PerceiverResampler MHA: head count is perceiver_heads, head_dim = H/heads.
     * This is NOT the Flux block head_dim (128) — using the wrong head_dim mis-groups
     * Q/K/V and collapses ip_embeds toward a pooled vector (IP-ADAPTER-INFER-1). */
    int heads = a->perceiver_heads, hd = H / heads;

    float *Q = malloc((size_t)T * H * sizeof(float));
    float *K = malloc((size_t)n_siglip * H * sizeof(float));
    float *V = malloc((size_t)n_siglip * H * sizeof(float));
    float *attn = malloc((size_t)T * H * sizeof(float));
    if (!Q || !K || !V || !attn) { free(Q); free(K); free(V); free(attn); return -1; }

    /* Input normalization (IP-ADAPTER-INFER-1 fix): standardize each SigLIP DIMENSION
     * across the token axis, then learned affine. Skipped for legacy bundles (NULL). */
    const float *sig_in = siglip;
    float *sig_n = NULL;
    if (a->in_gamma && a->in_beta) {
        sig_n = malloc((size_t)n_siglip * S * sizeof(float));
        if (!sig_n) { free(Q); free(K); free(V); free(attn); return -1; }
        for (int d = 0; d < S; d++) {
            float mean = 0.0f;
            for (int t = 0; t < n_siglip; t++) mean += siglip[(size_t)t * S + d];
            mean /= n_siglip;
            float var = 0.0f;
            for (int t = 0; t < n_siglip; t++) {
                float e = siglip[(size_t)t * S + d] - mean; var += e * e;
            }
            var /= n_siglip;
            float inv = 1.0f / sqrtf(var + LN_EPS);
            float g = a->in_gamma[d], b = a->in_beta[d];
            for (int t = 0; t < n_siglip; t++)
                sig_n[(size_t)t * S + d] = (siglip[(size_t)t * S + d] - mean) * inv * g + b;
        }
        sig_in = sig_n;
    }

    /* projections: x @ W^T  (Linear, bias=False) */
    iris_matmul_t(Q, a->query_tokens, a->query_proj, T, H, H);
    iris_matmul_t(K, sig_in, a->key_proj, n_siglip, S, H);
    iris_matmul_t(V, sig_in, a->value_proj, n_siglip, S, H);

    if (mha_sdpa(attn, Q, K, V, T, n_siglip, heads, hd) != 0) {
        free(Q); free(K); free(V); free(attn); free(sig_n);
        return -1;
    }

    /* out_proj then per-token LayerNorm into out */
    iris_matmul_t(out, attn, a->out_proj, T, H, H);
    for (int t = 0; t < T; t++) {
        float *row = out + (size_t)t * H;
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
    free(Q); free(K); free(V); free(attn); free(sig_n);
    return 0;
}

int iris_ip_adapter_perceive(const iris_ip_adapter_t *a,
                             const float *siglip, int n_siglip,
                             float *ip_embeds) {
    int H = a->hidden_dim;

    /* CSD conditioning (SREF-LEAK-2): `siglip` is the single CSD style vector [csd_dim]
     * (n_siglip ignored). FiLM the learned query tokens (reuses query_tokens/norm_*). */
    if (strcmp(a->cond_mode, "csd") == 0)
        return perceive_csd_film(H, a->num_image_tokens, a->csd_dim, siglip,
                                 a->query_tokens, a->film_weight, a->film_bias,
                                 a->norm_weight, a->norm_bias, ip_embeds);

    /* Hybrid conditioning (SREF-COMBINE-1): `siglip` is the packed feature [n_siglip, S]
     * where the last row is the CSD vector zero-padded to S. The SigLIP perceiver consumes
     * the first (n_siglip-1) rows → first half of ip_embeds; the CSD FiLM module consumes
     * the last row's first csd_dim values → second half. Matches the Python concat order
     * [sig_tok, csd_tok] in IPAdapterKlein.get_image_embeds. */
    if (strcmp(a->cond_mode, "hybrid") == 0) {
        int half = a->num_image_tokens / 2;
        int n_sig = n_siglip - 1;                                   /* SigLIP rows (729) */
        const float *csd = siglip + (size_t)n_sig * a->siglip_dim;  /* last row, first csd_dim */
        if (perceive_siglip_mha(a, siglip, n_sig, half, ip_embeds) != 0)
            return -1;
        if (perceive_csd_film(H, half, a->csd_dim, csd,
                              a->csd_query_tokens, a->csd_film_weight, a->csd_film_bias,
                              a->csd_norm_weight, a->csd_norm_bias,
                              ip_embeds + (size_t)half * H) != 0)
            return -1;
        return 0;
    }

    /* SigLIP-only PerceiverResampler. */
    return perceive_siglip_mha(a, siglip, n_siglip, a->num_image_tokens, ip_embeds);
}

void iris_ip_adapter_get_kv(const iris_ip_adapter_t *a, int block_idx,
                            const float *ip_embeds, float *k_ip, float *v_ip) {
    int H = a->hidden_dim, T = a->num_image_tokens;
    size_t off = (size_t)block_idx * H * H;
    /* einsum btd,de->te : ip_embeds @ stacked[block] (no transpose) */
    iris_matmul(k_ip, ip_embeds, a->ip_k_stacked + off, T, H, H);
    iris_matmul(v_ip, ip_embeds, a->ip_v_stacked + off, T, H, H);
    /* SREF-COMBINE-1 hybrid leak gate: scale each group's V by its per-block weight.
     * SigLIP = first T/2 tokens, CSD = the rest. K unscaled. Mirrors get_kv_all. */
    if (a->group_gate) {
        int half = T / 2;
        float gs = a->group_gate[(size_t)block_idx * 2 + 0];
        float gc = a->group_gate[(size_t)block_idx * 2 + 1];
        for (int t = 0; t < half; t++)
            for (int d = 0; d < H; d++) v_ip[(size_t)t * H + d] *= gs;
        for (int t = half; t < T; t++)
            for (int d = 0; d < H; d++) v_ip[(size_t)t * H + d] *= gc;
    }
}

void iris_ip_adapter_inject(const iris_ip_adapter_t *a, int block_idx,
                            const float *img_q, int img_seq,
                            const float *k_ip, const float *v_ip,
                            float *img_hidden) {
    float s = a->ip_scale[block_idx] * a->ip_sched_mult;
    if (s == 0.0f) return;                 /* style-only no-op, or schedule gated this step off */
    int H = a->hidden_dim, T = a->num_image_tokens, hd = 128, heads = H / hd;
    float *attn = malloc((size_t)img_seq * H * sizeof(float));
    if (!attn) return;
    mha_sdpa(attn, img_q, k_ip, v_ip, img_seq, T, heads, hd);
    for (size_t i = 0; i < (size_t)img_seq * H; i++)
        img_hidden[i] += s * attn[i];
    free(attn);
}

int iris_ip_adapter_set_schedule(iris_ip_adapter_t *a, const char *spec) {
    if (!a) return -1;
    /* Reset to "none" first so a bad spec leaves a defined (inert) state. */
    a->ip_sched_kind = 0;
    a->ip_sched_param = 0.0f;
    a->ip_sched_mult = 1.0f;
    if (!spec || !*spec || strcmp(spec, "none") == 0) return 0;
    float f = 0.0f;
    if (sscanf(spec, "late:%f", &f) == 1)       a->ip_sched_kind = 1;
    else if (sscanf(spec, "early:%f", &f) == 1) a->ip_sched_kind = 2;
    else {
        fprintf(stderr, "ip-schedule: bad spec '%s' (want none|late:F|early:F)\n", spec);
        return -1;
    }
    if (!isfinite(f) || f < 0.0f || f > 1.0f) {
        fprintf(stderr, "ip-schedule: fraction %.3f out of [0,1]\n", (double)f);
        a->ip_sched_kind = 0;
        return -1;
    }
    a->ip_sched_param = f;
    return 0;
}

void iris_ip_adapter_set_step(iris_ip_adapter_t *a, int step, int num_steps) {
    if (!a) return;
    if (a->ip_sched_kind == 0 || num_steps <= 0) { a->ip_sched_mult = 1.0f; return; }
    /* Step fraction in [0,1]: 0 at the first (noisiest) step, 1 at the last (cleanest). */
    float frac = (num_steps > 1) ? (float)step / (float)(num_steps - 1) : 1.0f;
    switch (a->ip_sched_kind) {
        case 1: a->ip_sched_mult = (frac >= a->ip_sched_param) ? 1.0f : 0.0f; break; /* late  */
        case 2: a->ip_sched_mult = (frac <  a->ip_sched_param) ? 1.0f : 0.0f; break; /* early */
        default: a->ip_sched_mult = 1.0f;
    }
}
