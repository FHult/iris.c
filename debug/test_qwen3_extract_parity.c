/*
 * debug/test_qwen3_extract_parity.c — SCAFFOLD (model-gated, SKIPS by default).
 *
 * Boundary: Flux text-encoder layer EXTRACTION. Flux concatenates Qwen3 hidden
 * states from layers 8, 17, 26 (0-indexed) into [seq, 7680] (4B) / [seq, 12288]
 * (9B); Z-Image instead takes hidden_states[-2] -> [seq, 2560]. A wrong layer
 * index or concat order silently degrades every prompt. This reference boundary
 * (Training->Inference Correctness Protocol #1) currently has NO guard.
 *
 * STATUS ON THIS MACHINE: SKIPS. No Qwen3 encoder weights and no flux_env
 * reference are present (checked 2026-07-31). Cannot execute here; NOT a passing
 * test.
 *
 * HOW IT WORKS (golden-fixture parity):
 * Compares two f32 dumps of the encoder output for the SAME prompt/tokenization:
 *   QWEN3_PARITY_C_BIN   — [seq, out_dim] from the C qwen3_encoder (Flux extraction:
 *                          concat layers 8/17/26; or Z-Image: hidden_states[-2]).
 *   QWEN3_PARITY_REF_BIN — same extraction from the transformers/flux_env reference.
 * Gate: corr > 0.999 AND max_abs <= 1e-3 (single encoder, tighter than full Flux
 * forward). The concat-order check is implicit: a swapped 8<->26 block or a wrong
 * layer index collapses corr well below the gate.
 *
 * TO PRODUCE THE FIXTURES (follow-up, needs weights):
 *   1. C side: dump qwen3_encoder output for a fixed prompt (add a --dump path or a
 *      tiny harness calling the encoder), fwrite f32 [seq*out_dim] -> C_BIN.
 *   2. Ref side: transformers Qwen3, extract the SAME layer indices, same chat
 *      template (see CLAUDE.md "Flux Text Encoder"), dump f32 -> REF_BIN.
 *   3. Set the two env vars and run via `make test-parity`.
 *
 * Build (compiles standalone; verified):
 *   cc -O2 -I. -o /tmp/test_qwen3_extract_parity debug/test_qwen3_extract_parity.c -lm
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
    const char *c_bin   = getenv("QWEN3_PARITY_C_BIN");
    const char *ref_bin = getenv("QWEN3_PARITY_REF_BIN");

    if (!c_bin || !ref_bin) {
        printf("SKIP qwen3-extract-parity: fixtures absent "
               "(set QWEN3_PARITY_C_BIN and QWEN3_PARITY_REF_BIN to run).\n");
        printf("  Reason: no Qwen3 encoder weights / transformers reference on this "
               "machine; see file header for how to generate the two f32 dumps.\n");
        return 0;  /* SKIP, not pass */
    }

    long nc = file_size_floats(c_bin), nr = file_size_floats(ref_bin);
    if (nc < 0 || nr < 0) {
        printf("SKIP qwen3-extract-parity: a fixture path does not exist "
               "(C_BIN=%s REF_BIN=%s).\n", c_bin, ref_bin);
        return 0;  /* SKIP */
    }
    if (nc != nr) {
        fprintf(stderr, "FAIL qwen3-extract-parity: length mismatch %ld vs %ld "
                        "(likely a wrong extracted dim / layer set).\n", nc, nr);
        return 1;
    }

    float *a = read_bin(c_bin, nc), *b = read_bin(ref_bin, nr);
    if (!a || !b) { fprintf(stderr, "FAIL qwen3-extract-parity: read error.\n"); return 1; }

    double dot = 0, na = 0, nb = 0, max_abs = 0;
    for (long i = 0; i < nc; i++) {
        double x = a[i], y = b[i], d = fabs(x - y);
        dot += x * y; na += x * x; nb += y * y;
        if (d > max_abs) max_abs = d;
    }
    double corr = (na > 0 && nb > 0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
    printf("qwen3-extract-parity: n=%ld corr=%.6f max_abs=%.6f\n", nc, corr, max_abs);

    int pass = (corr > 0.999) && (max_abs <= 1e-3);
    printf("%s qwen3-extract-parity (corr>0.999 && max_abs<=1e-3)\n",
           pass ? "PASS" : "FAIL");
    free(a); free(b);
    return pass ? 0 : 1;
}
