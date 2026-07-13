/*
 * test_csdmod.c - parity guard for the C CSDModulation vs the Python training module.
 * Loads the exported csd_mod weights + a golden fixture (debug/gen_csdmod_fixture.py) and
 * checks C csd_mod_forward reproduces the Python CSDModulation output to a tight tolerance
 * (corr > 0.999, max_abs <= 1e-3). Weights/fixtures live on 2TBSSD -> run on demand
 * (like debug/vae_parity.c), not in the hermetic make test.
 *
 *   gen: train/.venv/bin/python debug/gen_csdmod_fixture.py --weights <dir>/csd_mod.safetensors --out <dir>
 *   run: ./test_csdmod <dir>
 */
#include "iris_csdmod.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float *read_f32(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    float *p = malloc((size_t)n * sizeof(float));
    size_t r = fread(p, sizeof(float), (size_t)n, f);
    fclose(f);
    if ((int)r != n) { fprintf(stderr, "short read %s (%zu/%d)\n", path, r, n); free(p); return NULL; }
    return p;
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "/Volumes/2TBSSD/sref_eval/joint_v1_c_export";
    char wpath[512], ipath[512], gpath[512];
    snprintf(wpath, sizeof(wpath), "%s/csd_mod.safetensors", dir);
    snprintf(ipath, sizeof(ipath), "%s/csdmod_input.f32", dir);
    snprintf(gpath, sizeof(gpath), "%s/csdmod_golden.f32", dir);

    csd_mod_t m;
    if (csd_mod_load(&m, wpath)) { fprintf(stderr, "FAIL: load %s\n", wpath); return 1; }
    printf("loaded csd_mod: csd=%d mlp=%d hidden=%d\n", m.csd_dim, m.mlp_dim, m.hidden_dim);

    float *csd    = read_f32(ipath, m.csd_dim);
    float *golden = read_f32(gpath, m.hidden_dim);
    if (!csd || !golden) { csd_mod_free(&m); return 1; }

    float *delta = malloc((size_t)m.hidden_dim * sizeof(float));
    float *work  = malloc((size_t)m.mlp_dim * sizeof(float));
    csd_mod_forward(&m, csd, delta, work);

    double ma = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    int n = m.hidden_dim;
    for (int i = 0; i < n; i++) {
        double a = delta[i], b = golden[i], d = fabs(a - b);
        if (d > ma) ma = d;
        sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
    }
    double cov = sab / n - (sa / n) * (sb / n);
    double va  = saa / n - (sa / n) * (sa / n);
    double vb  = sbb / n - (sb / n) * (sb / n);
    double corr = (va > 0 && vb > 0) ? cov / sqrt(va * vb) : 0.0;
    printf("corr=%.6f  max_abs=%.3e  (want corr>0.999, max_abs<=1e-3)\n", corr, ma);

    int pass = (corr > 0.999) && (ma <= 1e-3);
    printf("%s\n", pass ? "PASS" : "FAIL");

    free(csd); free(golden); free(delta); free(work);
    csd_mod_free(&m);
    return pass ? 0 : 1;
}
