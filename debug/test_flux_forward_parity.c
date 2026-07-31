/*
 * debug/test_flux_forward_parity.c — SCAFFOLD (model-gated, SKIPS by default).
 *
 * Boundary: Flux base transformer forward (velocity) — C `iris_transformer_forward`
 * vs the mflux/flux2 Python reference. This is a train<->infer / reference-parity
 * boundary (Training->Inference Correctness Protocol #1) that currently has NO guard.
 *
 * STATUS ON THIS MACHINE: SKIPS. There are no Flux transformer weights present
 * (checked 2026-07-31: /Volumes/2TBSSD/models holds only csd_vit_l_style.safetensors;
 * no flux-klein-* dir, no flux_env/flux2 reference). This scaffold therefore cannot
 * be executed here. It is NOT a passing test and must never be reported as one.
 *
 * HOW IT WORKS (golden-fixture parity, matches the protocol):
 * It does NOT run the model itself (that needs the full weighted graph + GPU). It
 * compares two float32 dumps of the SAME forward call:
 *   FLUX_PARITY_C_BIN   — velocity dumped by the C path (see "to produce" below)
 *   FLUX_PARITY_REF_BIN — velocity dumped by the mflux/flux2 reference
 * on the SAME latent/text/timestep inputs (a fixed seed fixture). Parity gate:
 * cosine corr > 0.999 AND max_abs <= 1e-2 (full-network f32-vs-bf16 accumulates
 * more than a single-op 1e-3; tighten once a real dump exists — see TODO).
 *
 * TO PRODUCE THE FIXTURES (follow-up, needs weights on a hot SSD):
 *   1. C side: add a one-shot dump of the step-0 velocity in the zimage/flux Euler
 *      loop (or a dedicated harness that loads the transformer and calls
 *      iris_transformer_forward once on the fixture inputs), fwrite f32 -> C_BIN.
 *   2. Ref side: run flux2/ (or mflux) forward on the identical inputs, dump f32.
 *   3. Point the two env vars at the dumps and run this via `make test-parity`.
 *
 * Build (compiles standalone; verified):
 *   cc -O2 -I. -o /tmp/test_flux_forward_parity debug/test_flux_forward_parity.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static long file_size_floats(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long b = ftell(f);
    fclose(f);
    return (b < 0) ? -1 : b / (long)sizeof(float);
}

static float *read_bin(const char *path, long n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    float *p = (float *)malloc((size_t)n * sizeof(float));
    if (!p) { fclose(f); return NULL; }
    long got = (long)fread(p, sizeof(float), (size_t)n, f);
    fclose(f);
    if (got != n) { free(p); return NULL; }
    return p;
}

int main(void) {
    const char *c_bin   = getenv("FLUX_PARITY_C_BIN");
    const char *ref_bin = getenv("FLUX_PARITY_REF_BIN");

    if (!c_bin || !ref_bin) {
        printf("SKIP flux-forward-parity: fixtures absent "
               "(set FLUX_PARITY_C_BIN and FLUX_PARITY_REF_BIN to run).\n");
        printf("  Reason: no Flux transformer weights / mflux reference on this "
               "machine; see file header for how to generate the two f32 dumps.\n");
        return 0;  /* SKIP, not pass */
    }

    long nc = file_size_floats(c_bin), nr = file_size_floats(ref_bin);
    if (nc < 0 || nr < 0) {
        printf("SKIP flux-forward-parity: a fixture path does not exist "
               "(C_BIN=%s REF_BIN=%s).\n", c_bin, ref_bin);
        return 0;  /* SKIP */
    }
    if (nc != nr) {
        fprintf(stderr, "FAIL flux-forward-parity: length mismatch %ld vs %ld "
                        "(fixtures describe different forwards).\n", nc, nr);
        return 1;
    }

    float *a = read_bin(c_bin, nc), *b = read_bin(ref_bin, nr);
    if (!a || !b) { fprintf(stderr, "FAIL flux-forward-parity: read error.\n"); return 1; }

    double dot = 0, na = 0, nb = 0, max_abs = 0;
    for (long i = 0; i < nc; i++) {
        double x = a[i], y = b[i], d = fabs(x - y);
        dot += x * y; na += x * x; nb += y * y;
        if (d > max_abs) max_abs = d;
    }
    double corr = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
    printf("flux-forward-parity: n=%ld corr=%.6f max_abs=%.6f\n", nc, corr, max_abs);

    int pass = (corr > 0.999) && (max_abs <= 1e-2);
    printf("%s flux-forward-parity (corr>0.999 && max_abs<=1e-2)\n",
           pass ? "PASS" : "FAIL");
    free(a); free(b);
    return pass ? 0 : 1;
}
