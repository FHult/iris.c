/*
 * test_vae.c - White-box parity/golden tests for the C VAE (GROK-VAE-1).
 *
 * The C iris_vae is the *inference ground truth*: it encodes img2img/ref
 * conditioning and decodes every generated latent. The proxy/teacher VAEs used
 * at precompute time are only trusted to the extent that this C implementation
 * stays distributionally and structurally compatible with them (see
 * plans/precomp2-proxy-vae-design.md Tier-2, and the contract note in
 * iris_vae.c's header). Until now nothing guarded the C encode/decode integration
 * itself (only iris_patchify roundtrip in test_kernels.c).
 *
 * This builds small but architecturally-real VAEs (full channel widths
 * 128/256/512, ch_mult [1,2,4,4], 32 groups) with deterministic synthetic
 * weights and exercises the real CPU encode/decode code paths to assert:
 *   - encode/decode produce the architecture-implied shapes (16x compression);
 *   - all outputs are finite (no NaN/Inf from the conv/groupnorm/attn stack);
 *   - encode and decode are bit-for-bit deterministic (no uninit-memory bugs);
 *   - z_channels -> latent_channels wiring (Flux 32->128, Z-Image 16->64);
 *   - the latent normalization BRANCH is config-selected and uses the exact
 *     `(x - shift) * scaling` form vs the batch-norm `(x - mean)/sqrt(var+eps)`
 *     form -- the path the brittle vae/config.json parser feeds (GROK-6 / C-2).
 *
 * It loads no model, touches no GPU, and reads no data volumes -- safe to run
 * alongside a live flywheel. White-box: includes iris_vae.c directly.
 *
 * Build: cc -O2 -I. -o /tmp/iris_test_vae debug/test_vae.c \
 *            iris_kernels.c iris_safetensors.c -lm
 * Run:   /tmp/iris_test_vae
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "iris.h"
#include "iris_vae_config.h"

/* The decoder allocates its output image via iris_image_create(); provide a
 * minimal local definition so we don't have to drag in iris_image.c (and its
 * PNG/JPEG link deps). The prototypes come from iris.h. */
iris_image *iris_image_create(int width, int height, int channels) {
    iris_image *img = (iris_image *)malloc(sizeof(iris_image));
    if (!img) return NULL;
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = (uint8_t *)calloc((size_t)width * height * channels, 1);
    if (!img->data) { free(img); return NULL; }
    return img;
}
void iris_image_free(iris_image *img) {
    if (img) { free(img->data); free(img); }
}

/* Pull in the real VAE under test (CPU-only: no USE_METAL defined here). */
#include "iris_vae.c"

/* ------------------------------------------------------------------------- */

static int failures = 0;
static int passes = 0;

static void check_true(const char *name, int cond) {
    if (!cond) { fprintf(stderr, "FAIL %s\n", name); failures++; }
    else { printf("PASS %s\n", name); passes++; }
}

static void check_f(const char *name, float got, float expected, float tol) {
    float diff = fabsf(got - expected);
    if (diff > tol) {
        fprintf(stderr, "FAIL %s: got %.8f expected %.8f (diff %.2e tol %.2e)\n",
                name, got, expected, diff, tol);
        failures++;
    } else { printf("PASS %s\n", name); passes++; }
}

/* ------------------------------------------------------------------------- */
/* Deterministic synthetic-weight builder                                    */
/* ------------------------------------------------------------------------- */

static uint64_t g_rng;
static void rng_seed(uint64_t s) { g_rng = s ? s : 0x9E3779B97F4A7C15ULL; }
static float rng_f(void) {
    /* LCG (Numerical Recipes constants); maps to [-1, 1]. */
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t x = (uint32_t)(g_rng >> 32);
    return ((float)x / 4294967295.0f) * 2.0f - 1.0f;
}
static float *alloc_rand(size_t n, float scale) {
    float *p = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) p[i] = rng_f() * scale;
    return p;
}
static float *alloc_fill(size_t n, float v) {
    float *p = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) p[i] = v;
    return p;
}

#define CONV_W 0.05f   /* small weights keep the deep stack numerically tame */

static void build_resblock(vae_resblock_t *b, int in_ch, int out_ch) {
    b->in_channels = in_ch;
    b->out_channels = out_ch;
    b->norm1_weight = alloc_fill(in_ch, 1.0f);
    b->norm1_bias   = alloc_fill(in_ch, 0.0f);
    b->conv1_weight = alloc_rand((size_t)out_ch * in_ch * 9, CONV_W);
    b->conv1_bias   = alloc_fill(out_ch, 0.0f);
    b->norm2_weight = alloc_fill(out_ch, 1.0f);
    b->norm2_bias   = alloc_fill(out_ch, 0.0f);
    b->conv2_weight = alloc_rand((size_t)out_ch * out_ch * 9, CONV_W);
    b->conv2_bias   = alloc_fill(out_ch, 0.0f);
    if (in_ch != out_ch) {
        b->skip_weight = alloc_rand((size_t)out_ch * in_ch, CONV_W);
        b->skip_bias   = alloc_fill(out_ch, 0.0f);
    } else {
        b->skip_weight = NULL;
        b->skip_bias   = NULL;
    }
}

static void build_attn(vae_attnblock_t *a, int ch) {
    a->channels = ch;
    a->norm_weight = alloc_fill(ch, 1.0f);
    a->norm_bias   = alloc_fill(ch, 0.0f);
    a->q_weight = alloc_rand((size_t)ch * ch, CONV_W); a->q_bias = alloc_fill(ch, 0.0f);
    a->k_weight = alloc_rand((size_t)ch * ch, CONV_W); a->k_bias = alloc_fill(ch, 0.0f);
    a->v_weight = alloc_rand((size_t)ch * ch, CONV_W); a->v_bias = alloc_fill(ch, 0.0f);
    a->out_weight = alloc_rand((size_t)ch * ch, CONV_W); a->out_bias = alloc_fill(ch, 0.0f);
}

/* Mirrors iris_vae_load_safetensors_ex() allocation/wiring, but fills weights
 * with seeded synthetic values. `eps` is taken as a parameter (instead of being
 * derived from scaling) so two VAEs can share an identical upstream stack while
 * differing only in the final normalization branch. `with_quant` allocates the
 * Flux-only quant/post_quant 1x1 convs. Identity batch-norm stats are allocated
 * whenever scaling==0 (so the BN branch reduces to x/sqrt(1+eps)). */
static iris_vae_t *build_vae(uint64_t seed, int z_channels, float scaling,
                             float shift, float eps, int with_quant, int max_dim) {
    rng_seed(seed);
    iris_vae_t *vae = (iris_vae_t *)calloc(1, sizeof(iris_vae_t));
    int ch_mult[4] = {1, 2, 4, 4};

    vae->z_channels = z_channels;
    vae->latent_channels = z_channels * 4;
    vae->base_channels = 128;
    vae->num_res_blocks = 2;
    vae->num_groups = 32;
    vae->max_h = max_dim;
    vae->max_w = max_dim;
    vae->scaling_factor = scaling;
    vae->shift_factor = shift;
    vae->eps = eps;
    for (int i = 0; i < 4; i++) vae->ch_mult[i] = ch_mult[i];

    int mid_ch = vae->base_channels * ch_mult[3];   /* 512 */
    int z_ch2 = z_channels * 2;

    /* Encoder ----------------------------------------------------------- */
    vae->enc_conv_in_weight = alloc_rand((size_t)128 * 3 * 9, CONV_W);
    vae->enc_conv_in_bias   = alloc_fill(128, 0.0f);

    vae->enc_down_blocks = (vae_resblock_t *)calloc(4 * vae->num_res_blocks,
                                                    sizeof(vae_resblock_t));
    int bi = 0;
    for (int level = 0; level < 4; level++) {
        int ch = vae->base_channels * ch_mult[level];
        int prev = (level == 0) ? vae->base_channels
                                : vae->base_channels * ch_mult[level - 1];
        for (int r = 0; r < vae->num_res_blocks; r++) {
            int in_ch = (r == 0 && level > 0) ? prev : ch;
            build_resblock(&vae->enc_down_blocks[bi++], in_ch, ch);
        }
    }
    vae->enc_downsample = (vae_downsample_t *)calloc(3, sizeof(vae_downsample_t));
    for (int i = 0; i < 3; i++) {
        int ch = vae->base_channels * ch_mult[i];
        vae->enc_downsample[i].channels = ch;
        vae->enc_downsample[i].conv_weight = alloc_rand((size_t)ch * ch * 9, CONV_W);
        vae->enc_downsample[i].conv_bias = alloc_fill(ch, 0.0f);
    }
    build_resblock(&vae->enc_mid_block1, mid_ch, mid_ch);
    build_attn(&vae->enc_mid_attn, mid_ch);
    build_resblock(&vae->enc_mid_block2, mid_ch, mid_ch);
    vae->enc_norm_out_weight = alloc_fill(mid_ch, 1.0f);
    vae->enc_norm_out_bias   = alloc_fill(mid_ch, 0.0f);
    vae->enc_conv_out_weight = alloc_rand((size_t)z_ch2 * mid_ch * 9, CONV_W);
    vae->enc_conv_out_bias   = alloc_fill(z_ch2, 0.0f);

    if (with_quant) {
        vae->quant_conv_weight = alloc_rand((size_t)z_ch2 * z_ch2, CONV_W);
        vae->quant_conv_bias   = alloc_fill(z_ch2, 0.0f);
    }

    /* Decoder ----------------------------------------------------------- */
    vae->dec_conv_in_weight = alloc_rand((size_t)mid_ch * z_channels * 9, CONV_W);
    vae->dec_conv_in_bias   = alloc_fill(mid_ch, 0.0f);
    build_resblock(&vae->dec_mid_block1, mid_ch, mid_ch);
    build_attn(&vae->dec_mid_attn, mid_ch);
    build_resblock(&vae->dec_mid_block2, mid_ch, mid_ch);

    vae->dec_up_blocks = (vae_resblock_t *)calloc(4 * (vae->num_res_blocks + 1),
                                                  sizeof(vae_resblock_t));
    bi = 0;
    for (int level = 3; level >= 0; level--) {
        int ch = vae->base_channels * ch_mult[level];
        int prev = (level == 3) ? mid_ch
                                : vae->base_channels * ch_mult[level + 1];
        for (int r = 0; r < vae->num_res_blocks + 1; r++) {
            int in_ch = (r == 0) ? prev : ch;
            build_resblock(&vae->dec_up_blocks[bi++], in_ch, ch);
        }
    }
    vae->dec_upsample = (vae_upsample_t *)calloc(3, sizeof(vae_upsample_t));
    for (int i = 0; i < 3; i++) {
        int ch = vae->base_channels * ch_mult[3 - i];
        vae->dec_upsample[i].channels = ch;
        vae->dec_upsample[i].conv_weight = alloc_rand((size_t)ch * ch * 9, CONV_W);
        vae->dec_upsample[i].conv_bias = alloc_fill(ch, 0.0f);
    }
    vae->dec_norm_out_weight = alloc_fill(128, 1.0f);
    vae->dec_norm_out_bias   = alloc_fill(128, 0.0f);
    vae->dec_conv_out_weight = alloc_rand((size_t)3 * 128 * 9, CONV_W);
    vae->dec_conv_out_bias   = alloc_fill(3, 0.0f);

    if (with_quant) {
        vae->post_quant_conv_weight = alloc_rand((size_t)z_channels * z_channels, CONV_W);
        vae->post_quant_conv_bias   = alloc_fill(z_channels, 0.0f);
    }

    /* Identity batch-norm stats for the Flux normalization branch. */
    if (scaling == 0.0f) {
        int lc = vae->latent_channels;
        vae->bn_mean = alloc_fill(lc, 0.0f);
        vae->bn_var  = alloc_fill(lc, 1.0f);
    }

    /* Work buffers sized for this (tiny) image; matches loader formula. */
    size_t max_spatial = (size_t)vae->max_h * vae->max_w;
    vae->work_size = 4 * (size_t)vae->base_channels * max_spatial * sizeof(float);
    vae->work1 = malloc(vae->work_size);
    vae->work2 = malloc(vae->work_size);
    vae->work3 = malloc(vae->work_size);
    return vae;
}

/* Deterministic synthetic input image tensor [3, H, W] in [-1, 1]. */
static float *make_input(uint64_t seed, int H, int W) {
    rng_seed(seed);
    float *t = (float *)malloc((size_t)3 * H * W * sizeof(float));
    for (size_t i = 0; i < (size_t)3 * H * W; i++) t[i] = rng_f();
    return t;
}

static int all_finite(const float *p, size_t n) {
    for (size_t i = 0; i < n; i++) if (!isfinite(p[i])) return 0;
    return 1;
}

/* ------------------------------------------------------------------------- */
/* Tests                                                                     */
/* ------------------------------------------------------------------------- */

#define IMG 32   /* 32x32 image -> /8 = 4x4 -> patchify 2x2 -> 2x2 latent */

static void test_flux_encode_shape_finite(void) {
    iris_vae_t *vae = build_vae(1, 32, 0.0f, 0.0f, 1e-4f, 1, IMG);
    int oh = -1, ow = -1;
    float *img = make_input(7, IMG, IMG);
    float *lat = iris_vae_encode(vae, img, 1, IMG, IMG, &oh, &ow);

    check_true("flux encode non-null", lat != NULL);
    check_true("flux latent_channels==128", vae->latent_channels == 128);
    check_true("flux encode out_h==H/16", oh == IMG / 16);
    check_true("flux encode out_w==W/16", ow == IMG / 16);
    if (lat) {
        size_t n = (size_t)vae->latent_channels * oh * ow;
        check_true("flux latent finite", all_finite(lat, n));
        free(lat);
    }
    free(img);
    iris_vae_free(vae);
}

static void test_flux_decode_shape_finite(void) {
    iris_vae_t *vae = build_vae(1, 32, 0.0f, 0.0f, 1e-4f, 1, IMG);
    int oh = -1, ow = -1;
    float *img = make_input(7, IMG, IMG);
    float *lat = iris_vae_encode(vae, img, 1, IMG, IMG, &oh, &ow);
    iris_image *out = lat ? iris_vae_decode(vae, lat, 1, oh, ow) : NULL;

    check_true("flux decode non-null", out != NULL);
    if (out) {
        check_true("flux decode width==IMG", out->width == IMG);
        check_true("flux decode height==IMG", out->height == IMG);
        check_true("flux decode channels==3", out->channels == 3);
        /* uint8 output is finite by construction; assert it actually ran
         * (not left all-zero by an early return). */
        int nonzero = 0;
        for (size_t i = 0; i < (size_t)IMG * IMG * 3; i++)
            if (out->data[i] != 0) { nonzero = 1; break; }
        check_true("flux decode produced output", nonzero);
        iris_image_free(out);
    }
    free(lat);
    free(img);
    iris_vae_free(vae);
}

static void test_encode_deterministic(void) {
    iris_vae_t *vae = build_vae(1, 32, 0.0f, 0.0f, 1e-4f, 1, IMG);
    int oh = 0, ow = 0;
    float *img = make_input(7, IMG, IMG);
    float *a = iris_vae_encode(vae, img, 1, IMG, IMG, &oh, &ow);
    float *b = iris_vae_encode(vae, img, 1, IMG, IMG, &oh, &ow);
    int same = (a && b) &&
               memcmp(a, b, (size_t)vae->latent_channels * oh * ow * sizeof(float)) == 0;
    check_true("encode is bit-deterministic", same);
    free(a); free(b); free(img);
    iris_vae_free(vae);
}

static void test_decode_deterministic(void) {
    iris_vae_t *vae = build_vae(1, 32, 0.0f, 0.0f, 1e-4f, 1, IMG);
    int oh = 0, ow = 0;
    float *img = make_input(7, IMG, IMG);
    float *lat = iris_vae_encode(vae, img, 1, IMG, IMG, &oh, &ow);
    iris_image *a = iris_vae_decode(vae, lat, 1, oh, ow);
    iris_image *b = iris_vae_decode(vae, lat, 1, oh, ow);
    int same = (a && b) &&
               memcmp(a->data, b->data, (size_t)IMG * IMG * 3) == 0;
    check_true("decode is bit-deterministic", same);
    iris_image_free(a); iris_image_free(b);
    free(lat); free(img);
    iris_vae_free(vae);
}

static void test_zimage_latent_channels(void) {
    /* Z-Image: z_channels=16 -> latent_channels=64; explicit scale/shift, no
     * quant/BN. Guards the z_channels -> latent_channels wiring driven by the
     * vae/config.json parser. */
    iris_vae_t *vae = build_vae(2, 16, 0.3611f, 0.1159f, 1e-6f, 0, IMG);
    int oh = -1, ow = -1;
    float *img = make_input(7, IMG, IMG);
    float *lat = iris_vae_encode(vae, img, 1, IMG, IMG, &oh, &ow);

    check_true("zimage latent_channels==64", vae->latent_channels == 64);
    check_true("zimage encode non-null", lat != NULL);
    check_true("zimage out_h==H/16", oh == IMG / 16);
    if (lat) {
        check_true("zimage latent finite",
                   all_finite(lat, (size_t)vae->latent_channels * oh * ow));
        free(lat);
    }
    free(img);
    iris_vae_free(vae);
}

static void test_normalization_branch(void) {
    /* Two VAEs identical upstream (same seed, weights, eps, quant) differing
     * ONLY in the final normalization branch:
     *   A: scaling=0  -> Flux batch-norm, identity stats: lat = pre/sqrt(1+eps)
     *   B: scaling=S  -> Z-Image scaling:                 lat = (pre - F)*S
     * Hence B[i] must equal (A[i]*sqrt(1+eps) - F)*S. This isolates and pins the
     * exact normalization formula/constants and the scaling!=0 branch select
     * (the config-driven path; cf. GROK-6 / C-2). */
    const float eps = 1e-4f, S = 0.3611f, F = 0.1159f;
    iris_vae_t *a = build_vae(42, 32, 0.0f, 0.0f, eps, 1, IMG);
    iris_vae_t *b = build_vae(42, 32, S,    F,    eps, 1, IMG);

    int oh = 0, ow = 0, oh2 = 0, ow2 = 0;
    float *img = make_input(7, IMG, IMG);
    float *la = iris_vae_encode(a, img, 1, IMG, IMG, &oh, &ow);
    float *lb = iris_vae_encode(b, img, 1, IMG, IMG, &oh2, &ow2);

    int ok = (la && lb && oh == oh2 && ow == ow2);
    check_true("norm-branch shapes match", ok);
    if (ok) {
        size_t n = (size_t)a->latent_channels * oh * ow;
        float root = sqrtf(1.0f + eps);
        float max_abs = 0.0f, max_rel = 0.0f;
        for (size_t i = 0; i < n; i++) {
            float pre = la[i] * root;                 /* recover pre-norm value */
            float expected = (pre - F) * S;           /* Z-Image scaling form   */
            float d = fabsf(lb[i] - expected);
            if (d > max_abs) max_abs = d;
            float denom = fabsf(expected) + 1e-6f;
            if (d / denom > max_rel) max_rel = d / denom;
        }
        check_f("norm-branch max abs error", max_abs, 0.0f, 1e-3f);
        check_f("norm-branch max rel error", max_rel, 0.0f, 1e-2f);
    }
    free(la); free(lb); free(img);
    iris_vae_free(a);
    iris_vae_free(b);
}

/* ------------------------------------------------------------------------- */
/* vae/config.json parse (grok C-2): the strstr/atoi parse that selects        */
/* z_channels (latent shape) + scaling/shift (normalization branch). A         */
/* mis-parse of a pretty-printed or variant config silently poisons every      */
/* precomputed latent, so pin the resolved values directly.                    */
/* ------------------------------------------------------------------------- */
static void test_vae_config_parse(void) {
    int z; float sc, sh;
    #define RESET() do { z = IRIS_VAE_Z_CHANNELS; sc = 0.0f; sh = 0.0f; } while (0)

    /* Z-Image config: overrides all three. */
    RESET();
    iris_parse_vae_config(
        "{\"latent_channels\": 16, \"scaling_factor\": 0.3611, \"shift_factor\": 0.1159}",
        &z, &sc, &sh);
    check_true("config zimage z_channels=16", z == 16);
    check_f("config zimage scaling", sc, 0.3611f, 1e-5f);
    check_f("config zimage shift",   sh, 0.1159f, 1e-5f);

    /* Flux config: none of the keys present -> defaults preserved (32, BN path). */
    RESET();
    iris_parse_vae_config("{\"in_channels\": 3, \"out_channels\": 3}", &z, &sc, &sh);
    check_true("config flux default z_channels=32", z == 32);
    check_f("config flux default scaling (BN)", sc, 0.0f, 1e-9f);
    check_f("config flux default shift (BN)",   sh, 0.0f, 1e-9f);

    /* Pretty-printed, multi-line, extra whitespace around colons. */
    RESET();
    iris_parse_vae_config(
        "{\n  \"latent_channels\" : 16 ,\n  \"scaling_factor\" : 0.5\n}", &z, &sc, &sh);
    check_true("config pretty z_channels=16", z == 16);
    check_f("config pretty scaling", sc, 0.5f, 1e-6f);

    /* Substring guard: keys ending in '..._channels' (e.g. block_out_channels)    */
    /* must NOT hijack the leading-quote-anchored "latent_channels" search.        */
    RESET();
    iris_parse_vae_config(
        "{\"sample_size\": 1024, \"block_out_channels\": [128, 256]}", &z, &sc, &sh);
    check_true("config no false latent_channels match -> default 32", z == 32);

    /* Empty / missing file content -> defaults preserved (override-only contract).*/
    RESET();
    iris_parse_vae_config("", &z, &sc, &sh);
    check_true("config empty -> default z_channels=32", z == 32);

    /* Scientific-notation scaling parses via atof. */
    RESET();
    iris_parse_vae_config("{\"scaling_factor\": 1.5e-1}", &z, &sc, &sh);
    check_f("config sci-notation scaling", sc, 0.15f, 1e-6f);

    /* latent_channels <= 0 is rejected (keeps default), guarding a malformed 0.   */
    RESET();
    iris_parse_vae_config("{\"latent_channels\": 0}", &z, &sc, &sh);
    check_true("config latent_channels=0 rejected -> default 32", z == 32);

    #undef RESET
}

int main(void) {
    printf("=== VAE white-box tests (CPU, synthetic weights) ===\n");
    test_flux_encode_shape_finite();
    test_flux_decode_shape_finite();
    test_encode_deterministic();
    test_decode_deterministic();
    test_zimage_latent_channels();
    test_normalization_branch();
    test_vae_config_parse();

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}
