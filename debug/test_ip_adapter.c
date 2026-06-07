/*
 * debug/test_ip_adapter.c — parity test for the C IP-Adapter port (G-1, Phase 1).
 *
 * Loads the committed synthetic bundle (debug/fixtures/ip_adapter/bundle) and asserts
 * the three stages reproduce the Python reference goldens (computed from the same f16
 * bundle weights by debug/gen_ip_adapter_fixture.py): perceive / get_kv / inject.
 *
 * Standalone CPU test (no GPU, no venv, no real model). Build:
 *   cc -O2 -I. -DUSE_BLAS -DACCELERATE_NEW_LAPACK -o /tmp/t \
 *      debug/test_ip_adapter.c iris_ip_adapter.c iris_kernels.c iris_safetensors.c \
 *      -framework Accelerate -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "iris_ip_adapter.h"

/* fixture dims (debug/fixtures/ip_adapter/shapes.json) */
#define HID 256
#define TOK 8
#define SIG_DIM 64
#define SIG_SEQ 16
#define IMG_SEQ 12
#define BLK 0
#define FIX "debug/fixtures/ip_adapter"

static int failures = 0, passes = 0;

static float *load_bin(const char *path, long n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    float *p = malloc(n * sizeof(float));
    long got = fread(p, sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "short read %s: %ld/%ld\n", path, got, n); free(p); return NULL; }
    return p;
}

static void compare(const char *name, const float *got, const float *gold, long n) {
    double dot = 0, na = 0, nb = 0, maxabs = 0;
    for (long i = 0; i < n; i++) {
        double a = got[i], b = gold[i], d = fabs(a - b);
        if (d > maxabs) maxabs = d;
        dot += a * b; na += a * a; nb += b * b;
    }
    double corr = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
    int ok = (corr > 0.999 && maxabs < 0.05);
    printf("%-10s corr=%.6f max_abs=%.5f  %s\n", name, corr, maxabs, ok ? "PASS" : "FAIL");
    if (ok) passes++; else failures++;
}

/* Goldens carry a quant suffix ("" for f16, "_int8" for int8); inputs are shared. */
static void run_bundle(const char *bundle, const char *suffix, const char *label,
                       const float *siglip, const float *img_q) {
    char p[256];
    printf("--- %s (%s) ---\n", label, bundle);
    iris_ip_adapter_t *a = iris_ip_adapter_load(bundle);
    if (!a) { fprintf(stderr, "FAIL load %s\n", bundle); failures++; return; }
    int meta_ok = (a->hidden_dim == HID && a->num_image_tokens == TOK &&
                   a->siglip_dim == SIG_DIM && a->num_blocks == 5);
    printf("meta dims  quant=%-8s %s\n", a->quant, meta_ok ? "PASS" : "FAIL");
    meta_ok ? passes++ : failures++;

    #define GLD(name) (snprintf(p, sizeof(p), FIX "/%s%s.bin", name, suffix), p)
    float *g_embeds = load_bin(GLD("gold_ip_embeds"), (long)TOK * HID);
    float *g_k      = load_bin(GLD("gold_k_ip_b0"),   (long)TOK * HID);
    float *g_v      = load_bin(GLD("gold_v_ip_b0"),   (long)TOK * HID);
    float *g_inj    = load_bin(GLD("gold_inject_b0"), (long)IMG_SEQ * HID);
    #undef GLD
    if (!g_embeds || !g_k || !g_v || !g_inj) { failures++; return; }

    float *embeds = malloc((long)TOK * HID * sizeof(float));
    iris_ip_adapter_perceive(a, siglip, SIG_SEQ, embeds);
    compare("perceive", embeds, g_embeds, (long)TOK * HID);

    float *k = malloc((long)TOK * HID * sizeof(float));
    float *v = malloc((long)TOK * HID * sizeof(float));
    iris_ip_adapter_get_kv(a, BLK, g_embeds, k, v);   /* golden ip_embeds isolates the stage */
    compare("get_kv k", k, g_k, (long)TOK * HID);
    compare("get_kv v", v, g_v, (long)TOK * HID);

    float *hidden = calloc((long)IMG_SEQ * HID, sizeof(float));
    iris_ip_adapter_inject(a, BLK, img_q, IMG_SEQ, g_k, g_v, hidden);
    compare("inject", hidden, g_inj, (long)IMG_SEQ * HID);

    free(embeds); free(k); free(v); free(hidden);
    free(g_embeds); free(g_k); free(g_v); free(g_inj);
    iris_ip_adapter_free(a);
}

int main(void) {
    float *siglip = load_bin(FIX "/in_siglip.bin", (long)SIG_SEQ * SIG_DIM);
    float *img_q  = load_bin(FIX "/in_img_q.bin",  (long)IMG_SEQ * HID);
    if (!siglip || !img_q) return 1;

    run_bundle(FIX "/bundle",      "",      "float16", siglip, img_q);
    run_bundle(FIX "/bundle_int8", "_int8", "int8",    siglip, img_q);

    free(siglip); free(img_q);
    printf("\n%d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}
