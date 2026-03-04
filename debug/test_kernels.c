/*
 * test_kernels.c - Unit tests for CPU kernel functions
 *
 * Tests flux_kernels.c functions in isolation (no model, no GPU):
 *   - flux_softmax_cpu: numerical stability, probabilities sum to 1
 *   - flux_rms_norm: per-row normalization with scale weights
 *   - flux_matmul / flux_matmul_t: small known-output matrix multiply
 *   - flux_silu / flux_silu_mul: activation correctness
 *   - flux_apply_rope / flux_compute_rope_freqs: rotation identity & orthogonality
 *   - flux_patchify / flux_unpatchify: roundtrip invertibility
 *   - flux_axpy / flux_add: element-wise operations
 *   - flux_rng_seed: seeded reproducibility
 *
 * Build: gcc -O2 -I. -o /tmp/flux_test_kernels debug/test_kernels.c flux_kernels.c -lm
 * Run:   /tmp/flux_test_kernels
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Pull in the kernel declarations without Metal or BLAS */
#include "iris_kernels.h"

static int failures = 0;
static int passes = 0;

static void check_f(const char *name, float got, float expected, float tol) {
    float diff = fabsf(got - expected);
    if (diff > tol) {
        fprintf(stderr, "FAIL %s: got %.8f, expected %.8f (diff %.2e, tol %.2e)\n",
                name, got, expected, diff, tol);
        failures++;
    } else {
        printf("PASS %s\n", name);
        passes++;
    }
}

static void check_true(const char *name, int cond) {
    if (!cond) {
        fprintf(stderr, "FAIL %s\n", name);
        failures++;
    } else {
        printf("PASS %s\n", name);
        passes++;
    }
}

/* =========================================================================
 * softmax
 * ========================================================================= */

static void test_softmax_sums_to_one(void) {
    /* Single row: probabilities must sum to 1.0 */
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    flux_softmax_cpu(x, 1, 4);

    float sum = x[0] + x[1] + x[2] + x[3];
    check_f("softmax sum=1", sum, 1.0f, 1e-5f);
    check_true("softmax all positive", x[0] > 0 && x[1] > 0 && x[2] > 0 && x[3] > 0);
    /* Max-input element must have highest probability */
    check_true("softmax argmax preserved", x[3] > x[2] && x[2] > x[1] && x[1] > x[0]);
}

static void test_softmax_uniform(void) {
    /* Equal logits -> uniform distribution */
    float x[] = {5.0f, 5.0f, 5.0f, 5.0f};
    flux_softmax_cpu(x, 1, 4);
    check_f("softmax uniform[0]", x[0], 0.25f, 1e-5f);
    check_f("softmax uniform[1]", x[1], 0.25f, 1e-5f);
    check_f("softmax uniform[2]", x[2], 0.25f, 1e-5f);
    check_f("softmax uniform[3]", x[3], 0.25f, 1e-5f);
}

static void test_softmax_large_values(void) {
    /* Numerically stable with large values (would overflow without max-shift) */
    float x[] = {1000.0f, 1001.0f, 1002.0f};
    flux_softmax_cpu(x, 1, 3);
    float sum = x[0] + x[1] + x[2];
    check_f("softmax large values sum=1", sum, 1.0f, 1e-5f);
    check_true("softmax large values finite", isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2]));
}

static void test_softmax_multirow(void) {
    /* Each row must independently sum to 1 */
    float x[] = {1.0f, 2.0f,   /* row 0 */
                 3.0f, 1.0f};   /* row 1 */
    flux_softmax_cpu(x, 2, 2);
    check_f("softmax multirow[0] sum", x[0] + x[1], 1.0f, 1e-5f);
    check_f("softmax multirow[1] sum", x[2] + x[3], 1.0f, 1e-5f);
    /* Row 1: higher first element -> higher prob */
    check_true("softmax multirow[1] argmax", x[2] > x[3]);
}

static void test_softmax_known_values(void) {
    /* softmax([0, 1]) = [1/(1+e), e/(1+e)] */
    float x[] = {0.0f, 1.0f};
    flux_softmax_cpu(x, 1, 2);
    float e = expf(1.0f);
    check_f("softmax known[0]", x[0], 1.0f / (1.0f + e), 1e-5f);
    check_f("softmax known[1]", x[1], e / (1.0f + e), 1e-5f);
}

/* =========================================================================
 * RMSNorm
 * ========================================================================= */

static void test_rmsnorm_identity_weight(void) {
    /* weight=all-ones: output[i] = x[i] / rms(x) */
    float x[] = {3.0f, 4.0f};          /* rms = sqrt((9+16)/2) = sqrt(12.5) */
    float weight[] = {1.0f, 1.0f};
    float out[2];
    float eps = 1e-6f;
    flux_rms_norm(out, x, weight, 1, 2, eps);

    float rms = sqrtf((3.0f*3.0f + 4.0f*4.0f) / 2.0f + eps);
    check_f("rmsnorm identity[0]", out[0], 3.0f / rms, 1e-5f);
    check_f("rmsnorm identity[1]", out[1], 4.0f / rms, 1e-5f);
}

static void test_rmsnorm_scaling(void) {
    /* weight doubles each element */
    float x[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float weight[] = {2.0f, 2.0f, 2.0f, 2.0f};
    float out[4];
    flux_rms_norm(out, x, weight, 1, 4, 1e-6f);
    /* rms(all-ones vector len 4) = 1, so out = weight * x/rms = 2 */
    check_f("rmsnorm scale[0]", out[0], 2.0f, 1e-4f);
    check_f("rmsnorm scale[1]", out[1], 2.0f, 1e-4f);
}

static void test_rmsnorm_zero_vector(void) {
    /* Near-zero input: eps prevents divide-by-zero, output should be finite */
    float x[] = {0.0f, 0.0f};
    float weight[] = {1.0f, 1.0f};
    float out[2];
    flux_rms_norm(out, x, weight, 1, 2, 1e-5f);
    check_true("rmsnorm zero finite[0]", isfinite(out[0]));
    check_true("rmsnorm zero finite[1]", isfinite(out[1]));
}

static void test_rmsnorm_multirow(void) {
    /* Each row is normalized independently */
    float x[] = {3.0f, 4.0f,    /* row 0: rms = sqrt(12.5) */
                 1.0f, 0.0f};   /* row 1: rms ~ 1/sqrt(2) */
    float weight[] = {1.0f, 1.0f};
    float out[4];
    flux_rms_norm(out, x, weight, 2, 2, 1e-6f);

    float rms0 = sqrtf((9.0f + 16.0f) / 2.0f + 1e-6f);
    float rms1 = sqrtf((1.0f + 0.0f) / 2.0f + 1e-6f);
    check_f("rmsnorm multirow[0][0]", out[0], 3.0f / rms0, 1e-4f);
    check_f("rmsnorm multirow[0][1]", out[1], 4.0f / rms0, 1e-4f);
    check_f("rmsnorm multirow[1][0]", out[2], 1.0f / rms1, 1e-4f);
    check_f("rmsnorm multirow[1][1]", out[3], 0.0f / rms1, 1e-4f);
}

/* =========================================================================
 * matmul
 * ========================================================================= */

static void test_matmul_identity(void) {
    /* C = A @ I -> C = A */
    float A[] = {1.0f, 2.0f,
                 3.0f, 4.0f};
    float I[] = {1.0f, 0.0f,
                 0.0f, 1.0f};
    float C[4];
    flux_matmul(C, A, I, 2, 2, 2);
    check_f("matmul identity[0,0]", C[0], 1.0f, 1e-5f);
    check_f("matmul identity[0,1]", C[1], 2.0f, 1e-5f);
    check_f("matmul identity[1,0]", C[2], 3.0f, 1e-5f);
    check_f("matmul identity[1,1]", C[3], 4.0f, 1e-5f);
}

static void test_matmul_known(void) {
    /* [1,2] @ [[5],[6]] = [1*5+2*6] = [17] */
    float A[] = {1.0f, 2.0f};
    float B[] = {5.0f, 6.0f};
    float C[1];
    flux_matmul(C, A, B, 1, 2, 1);
    check_f("matmul known[0]", C[0], 17.0f, 1e-4f);
}

static void test_matmul_2x3_3x2(void) {
    /* [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] */
    float A[] = {1.0f, 2.0f, 3.0f,
                 4.0f, 5.0f, 6.0f};
    float B[] = {7.0f,  8.0f,
                 9.0f,  10.0f,
                 11.0f, 12.0f};
    float C[4];
    flux_matmul(C, A, B, 2, 3, 2);
    /* Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64] */
    /* Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154] */
    check_f("matmul 2x3x2[0,0]", C[0], 58.0f,  1e-3f);
    check_f("matmul 2x3x2[0,1]", C[1], 64.0f,  1e-3f);
    check_f("matmul 2x3x2[1,0]", C[2], 139.0f, 1e-3f);
    check_f("matmul 2x3x2[1,1]", C[3], 154.0f, 1e-3f);
}

static void test_matmul_t(void) {
    /* C = A @ B^T: A[2,3], B[2,3] -> C[2,2] = A @ B^T */
    float A[] = {1.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f};
    float B[] = {1.0f, 0.0f, 0.0f,  /* B^T col 0 = B row 0 */
                 0.0f, 1.0f, 0.0f};
    float C[4];
    flux_matmul_t(C, A, B, 2, 3, 2);
    /* A @ B^T = I when A==B and both are orthonormal rows */
    check_f("matmul_t identity[0,0]", C[0], 1.0f, 1e-5f);
    check_f("matmul_t identity[0,1]", C[1], 0.0f, 1e-5f);
    check_f("matmul_t identity[1,0]", C[2], 0.0f, 1e-5f);
    check_f("matmul_t identity[1,1]", C[3], 1.0f, 1e-5f);
}

/* =========================================================================
 * SiLU / SiLU-mul
 * ========================================================================= */

static void test_silu_known(void) {
    /* SiLU(0) = 0 * sigmoid(0) = 0.5 * 0 = 0 */
    /* SiLU(1) = 1 / (1 + e^-1) = 0.7310586 */
    /* SiLU(-1) = -1 / (1 + e) = -0.2689414 */
    float x[] = {0.0f, 1.0f, -1.0f};
    flux_silu(x, 3);
    check_f("silu(0)", x[0], 0.0f,       1e-4f);
    check_f("silu(1)", x[1], 0.7310586f, 1e-4f);
    check_f("silu(-1)", x[2], -0.2689414f, 1e-4f);
}

static void test_silu_mul(void) {
    /* silu_mul computes silu(gate) * up in place */
    float gate[] = {1.0f, 0.0f};
    float up[]   = {2.0f, 3.0f};
    flux_silu_mul(gate, up, 2);
    check_f("silu_mul[0]", gate[0], 0.7310586f * 2.0f, 1e-4f);
    check_f("silu_mul[1]", gate[1], 0.0f,              1e-4f);
}

/* =========================================================================
 * axpy / add
 * ========================================================================= */

static void test_axpy(void) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    flux_axpy(a, 2.0f, b, 3);
    /* a[i] += 2 * b[i] */
    check_f("axpy[0]", a[0], 9.0f,  1e-5f);
    check_f("axpy[1]", a[1], 12.0f, 1e-5f);
    check_f("axpy[2]", a[2], 15.0f, 1e-5f);
}

static void test_add(void) {
    float a[] = {1.0f, 2.0f};
    float b[] = {3.0f, 4.0f};
    float out[2];
    flux_add(out, a, b, 2);
    check_f("add[0]", out[0], 4.0f, 1e-5f);
    check_f("add[1]", out[1], 6.0f, 1e-5f);
}

/* =========================================================================
 * RoPE
 * ========================================================================= */

static void test_rope_zero_position(void) {
    /* pos=0 -> cos=1, sin=0 -> rotation is identity */
    int seq = 1, heads = 1, head_dim = 4;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float freqs[4];  /* [seq=1, head_dim/2=2, 2] */
    int pos[] = {0};
    flux_compute_rope_freqs(freqs, pos, seq, head_dim, 10000.0f);
    /* cos(0)=1, sin(0)=0 for all dimensions */
    check_f("rope freqs cos[0]", freqs[0], 1.0f, 1e-5f);
    check_f("rope freqs sin[0]", freqs[1], 0.0f, 1e-5f);

    /* Apply RoPE: with identity rotation, x unchanged */
    float x_orig[4];
    memcpy(x_orig, x, sizeof(x));
    flux_apply_rope(x, freqs, 1, seq, heads, head_dim);
    check_f("rope identity[0]", x[0], x_orig[0], 1e-4f);
    check_f("rope identity[1]", x[1], x_orig[1], 1e-4f);
    check_f("rope identity[2]", x[2], x_orig[2], 1e-4f);
    check_f("rope identity[3]", x[3], x_orig[3], 1e-4f);
}

static void test_rope_preserves_norm(void) {
    /* RoPE is a rotation, so it preserves vector norm */
    int seq = 1, heads = 1, head_dim = 4;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float freqs[4];
    int pos[] = {7};
    flux_compute_rope_freqs(freqs, pos, seq, head_dim, 10000.0f);

    float norm_before = 0.0f;
    for (int i = 0; i < head_dim; i++) norm_before += x[i] * x[i];

    flux_apply_rope(x, freqs, 1, seq, heads, head_dim);

    float norm_after = 0.0f;
    for (int i = 0; i < head_dim; i++) norm_after += x[i] * x[i];

    check_f("rope preserves norm", norm_after, norm_before, 1e-4f);
}

static void test_rope_freqs_range(void) {
    /* cos/sin values must be in [-1, 1] */
    int seq = 4, head_dim = 8;
    float freqs[4 * 4 * 2];
    int pos[] = {0, 1, 10, 100};
    flux_compute_rope_freqs(freqs, pos, seq, head_dim, 10000.0f);
    int ok = 1;
    for (int i = 0; i < seq * (head_dim / 2) * 2; i++) {
        if (freqs[i] < -1.0f - 1e-5f || freqs[i] > 1.0f + 1e-5f) ok = 0;
    }
    check_true("rope freqs in [-1,1]", ok);
}

/* =========================================================================
 * patchify / unpatchify roundtrip
 * ========================================================================= */

static void test_patchify_roundtrip(void) {
    /* patchify then unpatchify must recover the original tensor */
    int batch = 1, channels = 2, H = 4, W = 4, p = 2;
    int n = batch * channels * H * W;
    float *in = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) in[i] = (float)i;

    int out_ch = channels * p * p;
    int outH = H / p, outW = W / p;
    float *patched = malloc(batch * out_ch * outH * outW * sizeof(float));
    float *recovered = malloc(n * sizeof(float));

    flux_patchify(patched, in, batch, channels, H, W, p);
    flux_unpatchify(recovered, patched, batch, channels, outH, outW, p);

    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (fabsf(recovered[i] - in[i]) > 1e-5f) { ok = 0; break; }
    }
    check_true("patchify/unpatchify roundtrip", ok);

    free(in); free(patched); free(recovered);
}

static void test_patchify_patch_size_1(void) {
    /* patch_size=1 is a no-op */
    int batch = 1, channels = 3, H = 2, W = 2;
    int n = batch * channels * H * W;
    float in[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    float out[12];
    flux_patchify(out, in, batch, channels, H, W, 1);
    int ok = 1;
    for (int i = 0; i < n; i++) {
        if (fabsf(out[i] - in[i]) > 1e-5f) { ok = 0; break; }
    }
    check_true("patchify patch_size=1 is noop", ok);
}

/* =========================================================================
 * RNG reproducibility
 * ========================================================================= */

static void test_rng_seed_reproducible(void) {
    /* Same seed -> same sequence */
    flux_rng_seed(42);
    float a[8];
    flux_randn(a, 8);

    flux_rng_seed(42);
    float b[8];
    flux_randn(b, 8);

    int ok = 1;
    for (int i = 0; i < 8; i++) {
        if (a[i] != b[i]) { ok = 0; break; }
    }
    check_true("rng same seed reproducible", ok);
}

static void test_rng_different_seeds(void) {
    /* Different seeds -> different sequences */
    flux_rng_seed(1);
    float a[4];
    flux_randn(a, 4);

    flux_rng_seed(2);
    float b[4];
    flux_randn(b, 4);

    int differs = 0;
    for (int i = 0; i < 4; i++) {
        if (a[i] != b[i]) { differs = 1; break; }
    }
    check_true("rng different seeds differ", differs);
}

static void test_rng_uniform_range(void) {
    /* Uniform samples should be in [0, 1) */
    flux_rng_seed(99);
    float buf[1000];
    flux_rand(buf, 1000);
    int ok = 1;
    for (int i = 0; i < 1000; i++) {
        if (buf[i] < 0.0f || buf[i] >= 1.0f) { ok = 0; break; }
    }
    check_true("rng uniform in [0,1)", ok);
}

/* =========================================================================
 * Flash Attention vs Naive Attention parity (TB-010)
 *
 * flux_attention  layout: Q/K/V [batch, heads, seq, head_dim]
 * flux_flash_attention layout: Q/K/V [seq, heads*head_dim]
 *
 * For batch=1, heads=1 these layouts are identical, so we can feed the
 * same buffer to both and compare outputs directly.
 * For heads>1 we generate data in flash layout [seq, heads, head_dim]
 * and transpose to [heads, seq, head_dim] for the naive call.
 * ========================================================================= */

/* Transpose [heads, seq, dim] <-> [seq, heads, dim] in-place using a tmp buf */
static void transpose_hs(float *dst, const float *src, int heads, int seq, int dim) {
    for (int h = 0; h < heads; h++)
        for (int s = 0; s < seq; s++)
            for (int d = 0; d < dim; d++)
                dst[h * seq * dim + s * dim + d] = src[s * heads * dim + h * dim + d];
}

static float max_diff_f(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static float mean_diff_f(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += fabsf(a[i] - b[i]);
    return s / n;
}

static void test_flash_parity_single_head(void) {
    /* batch=1, heads=1: layouts identical, feed same buffer to both */
    int seq_q = 8, seq_k = 12, head_dim = 16;
    float scale = 1.0f / sqrtf((float)head_dim);
    int n_q = seq_q * head_dim, n_k = seq_k * head_dim;

    float *Q = malloc(n_q * sizeof(float));
    float *K = malloc(n_k * sizeof(float));
    float *V = malloc(n_k * sizeof(float));
    float *out_naive = malloc(n_q * sizeof(float));
    float *out_flash = malloc(n_q * sizeof(float));

    flux_rng_seed(12345);
    flux_randn(Q, n_q);
    flux_randn(K, n_k);
    flux_randn(V, n_k);

    /* Naive: batch=1, heads=1, layout same as flash for heads=1 */
    flux_attention(out_naive, Q, K, V, 1, 1, seq_q, seq_k, head_dim, scale);
    /* Flash: seq-major layout */
    flux_flash_attention(out_flash, Q, K, V, seq_q, seq_k, 1, head_dim, scale);

    float md = mean_diff_f(out_naive, out_flash, n_q);
    float mx = max_diff_f(out_naive, out_flash, n_q);
    char name[64];
    snprintf(name, sizeof(name), "flash parity h=1 mean_diff<1e-3 (%.2e)", md);
    check_true(name, md < 1e-3f);
    snprintf(name, sizeof(name), "flash parity h=1 max_diff<5e-3 (%.2e)", mx);
    check_true(name, mx < 5e-3f);

    free(Q); free(K); free(V); free(out_naive); free(out_flash);
}

static void test_flash_parity_multi_head(void) {
    /* heads=4, seq_q=16, seq_k=16, head_dim=8 */
    int heads = 4, seq_q = 16, seq_k = 16, head_dim = 8;
    float scale = 1.0f / sqrtf((float)head_dim);
    int hidden = heads * head_dim;

    /* Allocate in flash layout: [seq, heads*head_dim] */
    float *Q_flash = malloc(seq_q * hidden * sizeof(float));
    float *K_flash = malloc(seq_k * hidden * sizeof(float));
    float *V_flash = malloc(seq_k * hidden * sizeof(float));
    /* Naive layout: [heads, seq, head_dim] (batch=1) */
    float *Q_naive = malloc(heads * seq_q * head_dim * sizeof(float));
    float *K_naive = malloc(heads * seq_k * head_dim * sizeof(float));
    float *V_naive = malloc(heads * seq_k * head_dim * sizeof(float));
    float *out_naive = malloc(heads * seq_q * head_dim * sizeof(float));
    float *out_flash = malloc(seq_q * hidden * sizeof(float));
    /* Transposed flash output for comparison */
    float *out_flash_t = malloc(heads * seq_q * head_dim * sizeof(float));

    flux_rng_seed(99999);
    flux_randn(Q_flash, seq_q * hidden);
    flux_randn(K_flash, seq_k * hidden);
    flux_randn(V_flash, seq_k * hidden);

    /* Build naive-layout copies */
    transpose_hs(Q_naive, Q_flash, heads, seq_q, head_dim);
    transpose_hs(K_naive, K_flash, heads, seq_k, head_dim);
    transpose_hs(V_naive, V_flash, heads, seq_k, head_dim);

    flux_attention(out_naive, Q_naive, K_naive, V_naive,
                   1, heads, seq_q, seq_k, head_dim, scale);
    flux_flash_attention(out_flash, Q_flash, K_flash, V_flash,
                         seq_q, seq_k, heads, head_dim, scale);

    /* Transpose flash output back to [heads, seq, head_dim] for comparison */
    transpose_hs(out_flash_t, out_flash, heads, seq_q, head_dim);

    float md = mean_diff_f(out_naive, out_flash_t, heads * seq_q * head_dim);
    float mx = max_diff_f(out_naive, out_flash_t, heads * seq_q * head_dim);
    char name[64];
    snprintf(name, sizeof(name), "flash parity h=4 mean_diff<1e-3 (%.2e)", md);
    check_true(name, md < 1e-3f);
    snprintf(name, sizeof(name), "flash parity h=4 max_diff<5e-3 (%.2e)", mx);
    check_true(name, mx < 5e-3f);

    free(Q_flash); free(K_flash); free(V_flash);
    free(Q_naive); free(K_naive); free(V_naive);
    free(out_naive); free(out_flash); free(out_flash_t);
}

static void test_flash_parity_large_seq(void) {
    /* seq > 64 triggers the tiled path in flux_flash_attention */
    int seq_q = 96, seq_k = 128, head_dim = 8;
    float scale = 1.0f / sqrtf((float)head_dim);

    float *Q = malloc(seq_q * head_dim * sizeof(float));
    float *K = malloc(seq_k * head_dim * sizeof(float));
    float *V = malloc(seq_k * head_dim * sizeof(float));
    float *out_naive = malloc(seq_q * head_dim * sizeof(float));
    float *out_flash = malloc(seq_q * head_dim * sizeof(float));

    flux_rng_seed(77777);
    flux_randn(Q, seq_q * head_dim);
    flux_randn(K, seq_k * head_dim);
    flux_randn(V, seq_k * head_dim);

    flux_attention(out_naive, Q, K, V, 1, 1, seq_q, seq_k, head_dim, scale);
    flux_flash_attention(out_flash, Q, K, V, seq_q, seq_k, 1, head_dim, scale);

    float md = mean_diff_f(out_naive, out_flash, seq_q * head_dim);
    float mx = max_diff_f(out_naive, out_flash, seq_q * head_dim);
    char name[64];
    snprintf(name, sizeof(name), "flash parity tiled mean_diff<1e-3 (%.2e)", md);
    check_true(name, md < 1e-3f);
    snprintf(name, sizeof(name), "flash parity tiled max_diff<5e-3 (%.2e)", mx);
    check_true(name, mx < 5e-3f);

    free(Q); free(K); free(V); free(out_naive); free(out_flash);
}

static void test_flash_identity_v(void) {
    /* If V is the identity matrix (head_dim == seq_k), output == softmax(QK^T/s) */
    int seq_q = 4, seq_k = 4, head_dim = 4;
    float scale = 1.0f / sqrtf((float)head_dim);

    float Q[] = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};
    float K[] = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};
    /* V = I: output[i] = softmax(QK^T[i]) */
    float V[] = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};
    float out_naive[16], out_flash[16];

    flux_attention(out_naive, Q, K, V, 1, 1, seq_q, seq_k, head_dim, scale);
    flux_flash_attention(out_flash, Q, K, V, seq_q, seq_k, 1, head_dim, scale);

    float md = mean_diff_f(out_naive, out_flash, 16);
    check_true("flash identity-V mean_diff<1e-5", md < 1e-5f);
    /* With K=Q=I and V=I, diagonal queries should dominate */
    check_true("flash identity-V q0 peaks at d0", out_flash[0] > out_flash[1]);
    check_true("flash identity-V q1 peaks at d1", out_flash[5] > out_flash[4]);
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(void) {
    printf("=== Kernel unit tests ===\n\n");

    printf("-- softmax --\n");
    test_softmax_sums_to_one();
    test_softmax_uniform();
    test_softmax_large_values();
    test_softmax_multirow();
    test_softmax_known_values();

    printf("\n-- rmsnorm --\n");
    test_rmsnorm_identity_weight();
    test_rmsnorm_scaling();
    test_rmsnorm_zero_vector();
    test_rmsnorm_multirow();

    printf("\n-- matmul --\n");
    test_matmul_identity();
    test_matmul_known();
    test_matmul_2x3_3x2();
    test_matmul_t();

    printf("\n-- silu --\n");
    test_silu_known();
    test_silu_mul();

    printf("\n-- axpy / add --\n");
    test_axpy();
    test_add();

    printf("\n-- rope --\n");
    test_rope_zero_position();
    test_rope_preserves_norm();
    test_rope_freqs_range();

    printf("\n-- patchify --\n");
    test_patchify_roundtrip();
    test_patchify_patch_size_1();

    printf("\n-- rng --\n");
    test_rng_seed_reproducible();
    test_rng_different_seeds();
    test_rng_uniform_range();

    printf("\n-- flash attention parity (TB-010) --\n");
    test_flash_parity_single_head();
    test_flash_parity_multi_head();
    test_flash_parity_large_seq();
    test_flash_identity_v();

    printf("\n");
    if (failures > 0) {
        fprintf(stderr, "%d/%d FAILED\n", failures, passes + failures);
        return 1;
    }
    printf("All %d kernel tests PASSED\n", passes);
    return 0;
}
