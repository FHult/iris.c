/*
 * debug/test_zimage_transformer_parity.c — SCAFFOLD, BLOCKED (SKIPS).
 *
 * Boundary: Z-Image S3-DiT transformer + its 3-axis RoPE (T=32,H=48,W=48;
 * theta=256) and the [IMAGE | CAPTION] sequence order / padded position-id
 * construction (CPU vs MPS). These are reference boundaries with NO golden guard.
 *
 * WHY THIS IS BLOCKED (not merely fixture-absent):
 * There is NO Z-Image-Turbo model on this machine (checked 2026-07-31: a full
 * scan of /Volumes/2TBSSD finds no Z-Image dir; the VAE fails to load without it).
 * A golden fixture for the transformer/RoPE cannot be generated OR validated here.
 * This is distinct from the Flux/Qwen3 scaffolds (weights merely absent) — for
 * Z-Image there is nothing to run against at all. This file therefore SKIPS and
 * is explicitly NOT a passing test. Documented in the report and BACKLOG
 * ZIMAGE-SCHED-1 vicinity: "the whole Z-Image path is unguarded by make test".
 *
 * WHAT IT WILL CHECK ONCE A MODEL EXISTS (golden-fixture parity):
 * Compare two f32 dumps of the SAME transformer call on fixture inputs:
 *   ZI_PARITY_C_BIN   — output of zi_transformer_forward (C)
 *   ZI_PARITY_REF_BIN — output of diffusers transformer_z_image (Python ref)
 * plus a dedicated RoPE cos/sin dump so the 3-axis split + caption-padding
 * position ids are checked independently of the block math. Gate: corr > 0.999
 * AND max_abs <= 1e-2 (full network). A separate, tighter RoPE-only fixture
 * (corr > 0.9999, max_abs <= 1e-4) should guard the position-id construction that
 * bit CPU/GPU before (Z-Image Known Pitfall #4).
 *
 * Build (compiles standalone; verified):
 *   cc -O2 -I. -o /tmp/test_zimage_transformer_parity debug/test_zimage_transformer_parity.c -lm
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
    const char *c_bin   = getenv("ZI_PARITY_C_BIN");
    const char *ref_bin = getenv("ZI_PARITY_REF_BIN");

    if (!c_bin || !ref_bin) {
        printf("SKIP zimage-transformer-parity: BLOCKED — no Z-Image-Turbo model "
               "on this machine (VAE load fails; no model dir found).\n");
        printf("  A golden fixture cannot be generated or validated here. This is "
               "not a fixture-absent skip; there is nothing to run against.\n");
        return 0;  /* SKIP (blocked), not pass */
    }

    /* Reached only if a future engineer supplies dumps on a machine WITH the model. */
    long nc = file_size_floats(c_bin), nr = file_size_floats(ref_bin);
    if (nc < 0 || nr < 0) {
        printf("SKIP zimage-transformer-parity: a fixture path does not exist.\n");
        return 0;  /* SKIP */
    }
    if (nc != nr) {
        fprintf(stderr, "FAIL zimage-transformer-parity: length mismatch %ld vs %ld.\n", nc, nr);
        return 1;
    }
    float *a = read_bin(c_bin, nc), *b = read_bin(ref_bin, nr);
    if (!a || !b) { fprintf(stderr, "FAIL zimage-transformer-parity: read error.\n"); return 1; }

    double dot = 0, na = 0, nb = 0, max_abs = 0;
    for (long i = 0; i < nc; i++) {
        double x = a[i], y = b[i], d = fabs(x - y);
        dot += x * y; na += x * x; nb += y * y;
        if (d > max_abs) max_abs = d;
    }
    double corr = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
    printf("zimage-transformer-parity: n=%ld corr=%.6f max_abs=%.6f\n", nc, corr, max_abs);
    int pass = (corr > 0.999) && (max_abs <= 1e-2);
    printf("%s zimage-transformer-parity (corr>0.999 && max_abs<=1e-2)\n", pass ? "PASS" : "FAIL");
    free(a); free(b);
    return pass ? 0 : 1;
}
