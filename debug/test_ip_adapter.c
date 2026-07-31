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
/* PerceiverResampler head count (debug/gen_ip_adapter_fixture.py PERCEIVER_HEADS).
 * Deliberately != HID/128 (=2): the perceiver head_dim is HID/PHEADS, NOT the Flux
 * block's 128. A bundle where these coincide masks IP-ADAPTER-INFER-1. */
#define PHEADS 4
#define CSD_DIM 32          /* SREF-LEAK-2: synthetic CSD style dim (cond_mode="csd") */
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

static void compare_tol(const char *name, const float *got, const float *gold, long n,
                        double max_tol) {
    double dot = 0, na = 0, nb = 0, maxabs = 0;
    for (long i = 0; i < n; i++) {
        double a = got[i], b = gold[i], d = fabs(a - b);
        if (d > maxabs) maxabs = d;
        dot += a * b; na += a * a; nb += b * b;
    }
    double corr = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
    int ok = (corr > 0.999 && maxabs < max_tol);
    printf("%-10s corr=%.6f max_abs=%.5f  %s\n", name, corr, maxabs, ok ? "PASS" : "FAIL");
    if (ok) passes++; else failures++;
}

static void compare(const char *name, const float *got, const float *gold, long n) {
    compare_tol(name, got, gold, n, 0.05);
}

/* Goldens carry a quant suffix ("" for f16, "_int8" for int8); inputs are shared. */
static void run_bundle(const char *bundle, const char *suffix, const char *label,
                       const float *siglip, const float *img_q) {
    char p[256];
    printf("--- %s (%s) ---\n", label, bundle);
    iris_ip_adapter_t *a = iris_ip_adapter_load(bundle);
    if (!a) { fprintf(stderr, "FAIL load %s\n", bundle); failures++; return; }
    int meta_ok = (a->hidden_dim == HID && a->num_image_tokens == TOK &&
                   a->siglip_dim == SIG_DIM && a->num_blocks == 5 &&
                   a->perceiver_heads == PHEADS);
    printf("meta dims  quant=%-8s pheads=%d %s\n", a->quant, a->perceiver_heads,
           meta_ok ? "PASS" : "FAIL");
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
    /* Tight tolerance: the perceiver head grouping (PHEADS, head_dim=HID/PHEADS)
     * must match exactly. The buggy HID/128 grouping diverges here (and only here);
     * a loose tolerance would let it pass — that is IP-ADAPTER-INFER-1. */
    compare_tol("perceive", embeds, g_embeds, (long)TOK * HID, 1e-3);

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

/* CSD-mode parity (SREF-LEAK-2): cond_mode="csd" — FiLM over a single [CSD_DIM] vector
 * instead of cross-attention. Same get_kv/inject stages (ip_embeds is [TOK,HID] in both
 * modes). The tight 1e-3 perceive tolerance guards the train↔infer FiLM math/shape mirror. */
static void run_csd_bundle(const float *img_q) {
    const char *bundle = FIX "/bundle_csd";
    char p[256];
    printf("--- csd (%s) ---\n", bundle);
    iris_ip_adapter_t *a = iris_ip_adapter_load(bundle);
    if (!a) { fprintf(stderr, "FAIL load %s\n", bundle); failures++; return; }
    int meta_ok = (a->hidden_dim == HID && a->num_image_tokens == TOK &&
                   a->num_blocks == 5 && strcmp(a->cond_mode, "csd") == 0 &&
                   a->csd_dim == CSD_DIM);
    printf("meta dims  cond_mode=%-6s csd_dim=%d %s\n", a->cond_mode, a->csd_dim,
           meta_ok ? "PASS" : "FAIL");
    meta_ok ? passes++ : failures++;

    float *csd = load_bin(FIX "/in_csd.bin", (long)CSD_DIM);
    #define GLD(name) (snprintf(p, sizeof(p), FIX "/%s_csd.bin", name), p)
    float *g_embeds = load_bin(GLD("gold_ip_embeds"), (long)TOK * HID);
    float *g_k      = load_bin(GLD("gold_k_ip_b0"),   (long)TOK * HID);
    float *g_v      = load_bin(GLD("gold_v_ip_b0"),   (long)TOK * HID);
    float *g_inj    = load_bin(GLD("gold_inject_b0"), (long)IMG_SEQ * HID);
    #undef GLD
    if (!csd || !g_embeds || !g_k || !g_v || !g_inj) { failures++; return; }

    float *embeds = malloc((long)TOK * HID * sizeof(float));
    iris_ip_adapter_perceive(a, csd, 1, embeds);   /* n_siglip ignored in csd mode */
    compare_tol("perceive", embeds, g_embeds, (long)TOK * HID, 1e-3);

    float *k = malloc((long)TOK * HID * sizeof(float));
    float *v = malloc((long)TOK * HID * sizeof(float));
    iris_ip_adapter_get_kv(a, BLK, g_embeds, k, v);
    compare("get_kv k", k, g_k, (long)TOK * HID);
    compare("get_kv v", v, g_v, (long)TOK * HID);

    float *hidden = calloc((long)IMG_SEQ * HID, sizeof(float));
    iris_ip_adapter_inject(a, BLK, img_q, IMG_SEQ, g_k, g_v, hidden);
    compare("inject", hidden, g_inj, (long)IMG_SEQ * HID);

    free(csd); free(embeds); free(k); free(v); free(hidden);
    free(g_embeds); free(g_k); free(g_v); free(g_inj);
    iris_ip_adapter_free(a);
}

/* Hybrid-mode parity (SREF-COMBINE-1): cond_mode="hybrid" — the SigLIP PerceiverResampler
 * produces the first num_image_tokens/2 tokens and the CSD FiLM the next, concatenated. The
 * input is the packed [SIG_SEQ+1, SIG_DIM] feature (last row = CSD padded). Guards BOTH halves
 * AND the concat order against the Python golden (1e-3). */
static void run_hybrid_bundle(const float *img_q) {
    const char *bundle = FIX "/bundle_hybrid";
    char p[256];
    const long HTOK = 2L * TOK;
    printf("--- hybrid (%s) ---\n", bundle);
    iris_ip_adapter_t *a = iris_ip_adapter_load(bundle);
    if (!a) { fprintf(stderr, "FAIL load %s\n", bundle); failures++; return; }
    int meta_ok = (a->hidden_dim == HID && a->num_image_tokens == HTOK &&
                   a->num_blocks == 5 && strcmp(a->cond_mode, "hybrid") == 0 &&
                   a->csd_dim == CSD_DIM && a->siglip_dim == SIG_DIM &&
                   a->perceiver_heads == PHEADS);
    printf("meta dims  cond_mode=%-6s csd_dim=%d toks=%d %s\n", a->cond_mode, a->csd_dim,
           a->num_image_tokens, meta_ok ? "PASS" : "FAIL");
    meta_ok ? passes++ : failures++;

    float *feat = load_bin(FIX "/in_hybrid.bin", (long)(SIG_SEQ + 1) * SIG_DIM);
    #define GLD(name) (snprintf(p, sizeof(p), FIX "/%s_hybrid.bin", name), p)
    float *g_embeds = load_bin(GLD("gold_ip_embeds"), HTOK * HID);
    float *g_k      = load_bin(GLD("gold_k_ip_b0"),   HTOK * HID);
    float *g_v      = load_bin(GLD("gold_v_ip_b0"),   HTOK * HID);
    float *g_inj    = load_bin(GLD("gold_inject_b0"), (long)IMG_SEQ * HID);
    #undef GLD
    if (!feat || !g_embeds || !g_k || !g_v || !g_inj) { failures++; return; }

    float *embeds = malloc(HTOK * HID * sizeof(float));
    iris_ip_adapter_perceive(a, feat, SIG_SEQ + 1, embeds);   /* packed: last row = CSD */
    compare_tol("perceive", embeds, g_embeds, HTOK * HID, 1e-3);

    float *k = malloc(HTOK * HID * sizeof(float));
    float *v = malloc(HTOK * HID * sizeof(float));
    iris_ip_adapter_get_kv(a, BLK, g_embeds, k, v);
    compare("get_kv k", k, g_k, HTOK * HID);
    compare("get_kv v", v, g_v, HTOK * HID);

    float *hidden = calloc((long)IMG_SEQ * HID, sizeof(float));
    iris_ip_adapter_inject(a, BLK, img_q, IMG_SEQ, g_k, g_v, hidden);
    compare("inject", hidden, g_inj, (long)IMG_SEQ * HID);

    free(feat); free(embeds); free(k); free(v); free(hidden);
    free(g_embeds); free(g_k); free(g_v); free(g_inj);
    iris_ip_adapter_free(a);
}

/* Per-block IP-injection propagation parity (M3, review 2026-07-30). The stages above test each
 * IP op in ISOLATION at block 0; they do NOT test that a block's injection PROPAGATES. C inference
 * (iris_transformer_flux.c) injects k_ip/v_ip into the post-block hidden PER BLOCK, so block i+1
 * derives its image-Q from a state that already carries block i's injection — matching the Python
 * `use_block_injection=True` path (_flux_forward_with_ip). This reproduces that CORRECT per-block-
 * injected forward and guards: propagation across ≥3 blocks, per-block scale application, and the
 * flat per-block index mapping (double 0..nd-1, single nd+j). derive_q = per-head RMSNorm
 * (head_dim=128, the Flux invariant) of the CURRENT hidden — stands in for the block's post-QK-norm
 * PRE-RoPE image-Q and makes Q depend on accumulated injections. Mirrors gen_ip_adapter_fixture.py
 * _derive_q / _forward_block_injected bit-for-bit. Tight 1e-3 tolerance. */
#define BP_EPS 1e-6f
static void run_block_prop(const float *siglip) {
    const char *bundle = FIX "/bundle_blockprop";
    printf("--- block-injection propagation (M3) (%s) ---\n", bundle);
    iris_ip_adapter_t *a = iris_ip_adapter_load(bundle);
    if (!a) { fprintf(stderr, "FAIL load %s\n", bundle); failures++; return; }
    int H = a->hidden_dim, T = a->num_image_tokens, NB = a->num_blocks;
    int hd = 128, heads = H / hd, S = IMG_SEQ;
    int meta_ok = (H == HID && T == TOK && NB == 5 && a->perceiver_heads == PHEADS);
    printf("meta dims  blocks=%d pheads=%d %s\n", NB, a->perceiver_heads, meta_ok ? "PASS" : "FAIL");
    meta_ok ? passes++ : failures++;

    float *gamma = load_bin(FIX "/in_blockprop_gamma.bin", (long)hd);
    float *h0    = load_bin(FIX "/in_blockprop_h0.bin", (long)S * H);
    float *gold  = load_bin(FIX "/gold_blockprop.bin", (long)S * H);
    if (!gamma || !h0 || !gold) { failures++; return; }

    float *ip = malloc((size_t)T * H * sizeof(float));
    iris_ip_adapter_perceive(a, siglip, SIG_SEQ, ip);

    float *h = malloc((size_t)S * H * sizeof(float));
    memcpy(h, h0, (size_t)S * H * sizeof(float));
    float *q = malloc((size_t)S * H * sizeof(float));
    float *k = malloc((size_t)T * H * sizeof(float));
    float *v = malloc((size_t)T * H * sizeof(float));

    for (int i = 0; i < NB; i++) {
        /* derive_q: per-head RMSNorm(head_dim) of the CURRENT hidden (carries injections 0..i-1) */
        for (int s = 0; s < S; s++)
            for (int hh = 0; hh < heads; hh++) {
                const float *xin = h + (size_t)s * H + hh * hd;
                float *xout = q + (size_t)s * H + hh * hd;
                float ss = 0.0f;
                for (int d = 0; d < hd; d++) ss += xin[d] * xin[d];
                float inv = 1.0f / sqrtf(ss / (float)hd + BP_EPS);
                for (int d = 0; d < hd; d++) xout[d] = xin[d] * inv * gamma[d];
            }
        iris_ip_adapter_get_kv(a, i, ip, k, v);
        iris_ip_adapter_inject(a, i, q, S, k, v, h);   /* h += scale[i] * SDPA(q, k_ip, v_ip) */
    }
    /* Golden is the per-block-injected forward; a regression to the end-sum approximation
     * (all Q from h0, contributions summed once) differs by ~20% here — well outside 1e-3. */
    compare_tol("blockprop", h, gold, (long)S * H, 1e-3);

    free(gamma); free(h0); free(gold); free(ip); free(h); free(q); free(k); free(v);
    iris_ip_adapter_free(a);
}

static void check(const char *name, int ok) {
    printf("%-24s %s\n", name, ok ? "PASS" : "FAIL");
    if (ok) passes++; else failures++;
}

/* DP-7 injection schedule: set_schedule parsing + set_step per-step multiplier.
 * Pure logic (no weights), so a zeroed struct suffices. */
static void test_schedule(void) {
    printf("--- injection schedule (DP-7) ---\n");
    iris_ip_adapter_t a;
    memset(&a, 0, sizeof(a));

    /* none → constant: mult 1.0 at every step */
    check("set none ok", iris_ip_adapter_set_schedule(&a, "none") == 0);
    iris_ip_adapter_set_step(&a, 0, 4); check("none @0 =1", a.ip_sched_mult == 1.0f);
    iris_ip_adapter_set_step(&a, 3, 4); check("none @3 =1", a.ip_sched_mult == 1.0f);

    /* late:0.5 over 4 steps: frac=step/3 → 0,.33,.67,1; inject when frac>=0.5 */
    check("set late ok", iris_ip_adapter_set_schedule(&a, "late:0.5") == 0);
    iris_ip_adapter_set_step(&a, 0, 4); check("late @0 off", a.ip_sched_mult == 0.0f);
    iris_ip_adapter_set_step(&a, 1, 4); check("late @1 off", a.ip_sched_mult == 0.0f);
    iris_ip_adapter_set_step(&a, 2, 4); check("late @2 on",  a.ip_sched_mult == 1.0f);
    iris_ip_adapter_set_step(&a, 3, 4); check("late @3 on",  a.ip_sched_mult == 1.0f);

    /* early:0.5 is the complement (inject when frac<0.5) */
    check("set early ok", iris_ip_adapter_set_schedule(&a, "early:0.5") == 0);
    iris_ip_adapter_set_step(&a, 0, 4); check("early @0 on",  a.ip_sched_mult == 1.0f);
    iris_ip_adapter_set_step(&a, 2, 4); check("early @2 off", a.ip_sched_mult == 0.0f);

    /* late:0.0 = always on (inertness edge — matches schedule-off behaviour) */
    iris_ip_adapter_set_schedule(&a, "late:0.0");
    iris_ip_adapter_set_step(&a, 0, 4); check("late:0.0 @0 =1", a.ip_sched_mult == 1.0f);

    /* bad specs rejected and leave the inert state (kind 0, mult 1.0) */
    check("reject bogus", iris_ip_adapter_set_schedule(&a, "bogus:1") == -1);
    check("reject late:2", iris_ip_adapter_set_schedule(&a, "late:2") == -1);
    check("reject late:nan", iris_ip_adapter_set_schedule(&a, "late:nan") == -1);
    iris_ip_adapter_set_step(&a, 0, 4); check("bad→inert =1", a.ip_sched_mult == 1.0f);

    /* num_steps==1 must not divide by zero */
    iris_ip_adapter_set_schedule(&a, "late:0.5");
    iris_ip_adapter_set_step(&a, 0, 1); check("1-step safe", a.ip_sched_mult == 1.0f);
}

int main(void) {
    test_schedule();

    float *siglip = load_bin(FIX "/in_siglip.bin", (long)SIG_SEQ * SIG_DIM);
    float *img_q  = load_bin(FIX "/in_img_q.bin",  (long)IMG_SEQ * HID);
    if (!siglip || !img_q) return 1;

    run_bundle(FIX "/bundle",      "",      "float16", siglip, img_q);
    run_bundle(FIX "/bundle_int8", "_int8", "int8",    siglip, img_q);
    run_csd_bundle(img_q);
    run_hybrid_bundle(img_q);
    run_block_prop(siglip);   /* M3: per-block injection propagation (uses its own h0/gamma) */

    free(siglip); free(img_q);
    printf("\n%d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}
