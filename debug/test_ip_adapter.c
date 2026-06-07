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
#include "train/export/iris_ip_adapter.h"

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

int main(void) {
    iris_ip_adapter_t *a = iris_ip_adapter_load(FIX "/bundle");
    if (!a) { fprintf(stderr, "FAIL load\n"); return 1; }
    printf("loaded: hidden=%d blocks=%d(%d+%d) tokens=%d siglip=%d quant=%s\n",
           a->hidden_dim, a->num_blocks, a->num_double_blocks, a->num_single_blocks,
           a->num_image_tokens, a->siglip_dim, a->quant);
    int meta_ok = (a->hidden_dim == HID && a->num_image_tokens == TOK &&
                   a->siglip_dim == SIG_DIM && a->num_blocks == 5);
    printf("meta dims    %s\n", meta_ok ? "PASS" : "FAIL");
    meta_ok ? passes++ : failures++;

    float *siglip   = load_bin(FIX "/in_siglip.bin",      (long)SIG_SEQ * SIG_DIM);
    float *img_q    = load_bin(FIX "/in_img_q.bin",       (long)IMG_SEQ * HID);
    float *g_embeds = load_bin(FIX "/gold_ip_embeds.bin", (long)TOK * HID);
    float *g_k      = load_bin(FIX "/gold_k_ip_b0.bin",   (long)TOK * HID);
    float *g_v      = load_bin(FIX "/gold_v_ip_b0.bin",   (long)TOK * HID);
    float *g_inj    = load_bin(FIX "/gold_inject_b0.bin", (long)IMG_SEQ * HID);
    if (!siglip || !img_q || !g_embeds || !g_k || !g_v || !g_inj) return 1;

    /* perceive: siglip -> ip_embeds */
    float *embeds = malloc((long)TOK * HID * sizeof(float));
    iris_ip_adapter_perceive(a, siglip, SIG_SEQ, embeds);
    compare("perceive", embeds, g_embeds, (long)TOK * HID);

    /* get_kv: use the golden ip_embeds as input (isolates the stage) */
    float *k = malloc((long)TOK * HID * sizeof(float));
    float *v = malloc((long)TOK * HID * sizeof(float));
    iris_ip_adapter_get_kv(a, BLK, g_embeds, k, v);
    compare("get_kv k", k, g_k, (long)TOK * HID);
    compare("get_kv v", v, g_v, (long)TOK * HID);

    /* inject: use the golden k/v + input img_q; img_hidden starts at 0 so the
     * result is exactly the contribution the golden recorded. */
    float *hidden = calloc((long)IMG_SEQ * HID, sizeof(float));
    iris_ip_adapter_inject(a, BLK, img_q, IMG_SEQ, g_k, g_v, hidden);
    compare("inject", hidden, g_inj, (long)IMG_SEQ * HID);

    iris_ip_adapter_free(a);
    printf("\n%d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}
