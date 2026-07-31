/*
 * debug/test_zimage_scheduler.c — HERMETIC golden guard for the Z-Image
 * FlowMatch-Euler sigma schedule (`iris_zimage_schedule` in iris_sample.c).
 *
 * WHY THIS EXISTS
 * ---------------
 * The whole Z-Image denoising path was previously unguarded by `make test`
 * (BACKLOG ZIMAGE-SCHED-1). The sigma construction is pure math — num_steps,
 * a static (resolution-blind) shift=3.0, a shifted linspace, and a terminal 0 —
 * so it needs NO model, NO weights, NO GPU, and can be pinned exactly here.
 *
 * This test links the REAL production function out of iris_sample.c (via
 * -Wl,-dead_strip, which drops the transformer/VAE code paths the scheduler
 * never calls). It is therefore a genuine regression guard on shipped behavior,
 * not a copy of the math.
 *
 * WHAT IS PINNED
 * --------------
 *  (a) M1  — default 8 NFE => 9 sigma values with sigma[8]==0. Guards the
 *            already-merged 9->8 default-steps fix.
 *  (b) EXACT sigma snapshot at 512^2 and 1024^2 under the current static
 *            shift=3.0.
 *  (c) IRIS_ZIMAGE_SHIFT env override changes the array; default (unset / bad
 *            value) stays 3.0.
 *
 * IMPORTANT — THESE VALUES ENCODE CURRENT, NOT NECESSARILY CORRECT, BEHAVIOR.
 * ------------------------------------------------------------------------
 * The snapshot pins the *current, unresolved-H2* static-shift behavior. It is
 * NOT a claim that these are the official-correct sigmas. Per BACKLOG
 * ZIMAGE-SCHED-1, the C scheduler uses a resolution-BLIND static shift=3.0,
 * whereas the only reference on this machine (mflux) uses a resolution-
 * DEPENDENT shift (~1.88 @512^2, ~3.16 @1024^2). Consequently, under the
 * current code the 512^2 and 1024^2 arrays are BYTE-IDENTICAL (image_seq_len is
 * ignored) — the test asserts that identity deliberately, so that if/when the
 * schedule is switched to a resolution-dependent shift (closing ZIMAGE-SCHED-1)
 * this guard fires and must be updated together with a new authoritative golden.
 *
 * Build (matches Makefile test-unit line; -dead_strip keeps this hermetic):
 *   cc -O2 -I. -Wl,-dead_strip -o /tmp/iris_test_zimage_scheduler \
 *      debug/test_zimage_scheduler.c iris_sample.c -lm
 * Also verified to compile+pass under full production flags
 * (-O3 -march=native -ffast-math -flto -DUSE_BLAS -DACCELERATE_NEW_LAPACK
 *  -framework Accelerate); the -O2 vs production delta is ~1e-7 (noise),
 * well inside the 1e-5 snapshot tolerance.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Real production function (iris_sample.c). */
extern float *iris_zimage_schedule(int num_steps, int image_seq_len);

/* Representative image_seq_len values (patch_size=2 over the 16-ch latent):
 *   512^2  -> latent 64x64  -> /2 -> 32x32 = 1024 patches
 *   1024^2 -> latent 128x128 -> /2 -> 64x64 = 4096 patches
 * (image_seq_len is currently ignored by the static shift; see header.) */
#define SEQ_512  1024
#define SEQ_1024 4096

#define TOL 1e-5f

static int failures = 0, passes = 0;
static void ok(const char *name, int cond) {
    if (cond) { printf("PASS %s\n", name); passes++; }
    else { fprintf(stderr, "FAIL %s\n", name); failures++; }
}
static int close_f(float a, float b) { return fabsf(a - b) <= TOL; }

/* Golden sigmas under the CURRENT static shift=3.0 (resolution-blind). */
static const float GOLD_SHIFT3[9] = {
    1.000000000f, 0.947542489f, 0.882787704f, 0.800837338f, 0.693793058f,
    0.548045635f, 0.337972105f, 0.008928538f, 0.000000000f
};
/* Golden sigmas under IRIS_ZIMAGE_SHIFT=3.16 (env-override sanity). */
static const float GOLD_SHIFT316[9] = {
    1.000000000f, 0.950074732f, 0.888080359f, 0.809038460f, 0.704796433f,
    0.561017215f, 0.349943727f, 0.009896723f, 0.000000000f
};

static int array_matches(const float *got, const float *want, int n) {
    for (int i = 0; i < n; i++) if (!close_f(got[i], want[i])) return 0;
    return 1;
}

int main(void) {
    /* Keep env clean: default (unset) must mean shift=3.0. */
    unsetenv("IRIS_ZIMAGE_SHIFT");

    /* ---- (a) M1: default 8 NFE => 9 sigma values, terminal 0. ---- */
    const int NFE = 8;                       /* zimage default_steps (iris.c) */
    float *s = iris_zimage_schedule(NFE, SEQ_1024);
    ok("schedule non-NULL", s != NULL);
    /* Length is num_steps+1 = 9; sanity the boundaries the loop guarantees. */
    ok("M1 sigma[0]==1 (sigma_max)", close_f(s[0], 1.0f));
    ok("M1 sigma[8]==0 (terminal, 9th value)", s[8] == 0.0f);
    /* Strictly decreasing across the 8 active steps (well-formed schedule). */
    int mono = 1;
    for (int i = 0; i < NFE; i++) if (!(s[i] > s[i + 1])) mono = 0;
    ok("M1 strictly decreasing sigma[0..8]", mono);
    free(s);

    /* ---- (b) EXACT snapshot @512^2 and @1024^2 under static shift=3.0. ---- */
    float *s512  = iris_zimage_schedule(NFE, SEQ_512);
    float *s1024 = iris_zimage_schedule(NFE, SEQ_1024);
    ok("snapshot 512^2 matches golden (shift=3.0)",  array_matches(s512,  GOLD_SHIFT3, 9));
    ok("snapshot 1024^2 matches golden (shift=3.0)", array_matches(s1024, GOLD_SHIFT3, 9));
    /* Current behavior is resolution-BLIND: the two arrays are identical.
     * This deliberately encodes the unresolved-H2 static shift (see header). */
    ok("512^2 == 1024^2 (resolution-blind static shift, ZIMAGE-SCHED-1)",
       array_matches(s512, s1024, 9));
    /* Pin the documented penultimate near-zero sigma (BACKLOG: 0.0089 @1024^2). */
    ok("penultimate sigma ~= 0.008929 (documented near-zero final step)",
       close_f(s1024[7], 0.008928538f));
    free(s512);
    free(s1024);

    /* ---- (c) IRIS_ZIMAGE_SHIFT override. ---- */
    setenv("IRIS_ZIMAGE_SHIFT", "3.16", 1);
    float *sov = iris_zimage_schedule(NFE, SEQ_1024);
    ok("env override shift=3.16 changes array", !array_matches(sov, GOLD_SHIFT3, 9));
    ok("env override shift=3.16 matches its golden", array_matches(sov, GOLD_SHIFT316, 9));
    free(sov);

    /* Non-positive / garbage override is ignored -> stays at 3.0 default. */
    setenv("IRIS_ZIMAGE_SHIFT", "0", 1);
    float *szero = iris_zimage_schedule(NFE, SEQ_1024);
    ok("override '0' ignored -> default 3.0", array_matches(szero, GOLD_SHIFT3, 9));
    free(szero);

    setenv("IRIS_ZIMAGE_SHIFT", "notanumber", 1);
    float *sbad = iris_zimage_schedule(NFE, SEQ_1024);
    ok("override 'notanumber' ignored -> default 3.0", array_matches(sbad, GOLD_SHIFT3, 9));
    free(sbad);

    /* Unset again -> default 3.0. */
    unsetenv("IRIS_ZIMAGE_SHIFT");
    float *sdef = iris_zimage_schedule(NFE, SEQ_1024);
    ok("unset -> default 3.0", array_matches(sdef, GOLD_SHIFT3, 9));
    free(sdef);

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}
