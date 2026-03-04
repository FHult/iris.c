/*
 * test_embcache.c - Unit tests for the embedding cache (embcache.c)
 *
 * Tests the 4-bit quantization and multi-slot cache logic without any model:
 *   - emb_quantize_4bit / emb_dequantize_4bit: roundtrip accuracy
 *   - Uniform / alternating / large value inputs
 *   - emb_cache_store / emb_cache_lookup: hit/miss semantics
 *   - emb_cache_has: presence check
 *   - emb_cache_clear / emb_cache_free: cleanup
 *   - emb_cache_stats: entry count and memory accounting
 *   - Multi-slot: different prompts land in cache independently
 *   - Overwrite: same slot prompt evicts old entry
 *   - NULL and edge-case safety
 *
 * Build: gcc -O2 -I. -o /tmp/iris_test_embcache debug/test_embcache.c embcache.c -lm
 * Run:   /tmp/iris_test_embcache
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "embcache.h"

static int failures = 0;
static int passes = 0;

static void check_true(const char *name, int cond) {
    if (!cond) {
        fprintf(stderr, "FAIL %s\n", name);
        failures++;
    } else {
        printf("PASS %s\n", name);
        passes++;
    }
}

static void check_f(const char *name, float got, float expected, float tol) {
    float diff = fabsf(got - expected);
    if (diff > tol) {
        fprintf(stderr, "FAIL %s: got %.6f, expected %.6f (diff %.2e)\n",
                name, got, expected, diff);
        failures++;
    } else {
        printf("PASS %s\n", name);
        passes++;
    }
}

/* =========================================================================
 * 4-bit quantization roundtrip
 * ========================================================================= */

static void test_quant_roundtrip_uniform(void) {
    /* All same value -> quantization is lossless (all map to same bin) */
    int n = 64;
    float *data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) data[i] = 3.14f;

    emb_quantized_t *q = emb_quantize_4bit(data, n);
    check_true("quant uniform not null", q != NULL);

    float *out = emb_dequantize_4bit(q);
    check_true("dequant uniform not null", out != NULL);

    /* With uniform input the range collapses to ~eps, so all values ≈ input */
    for (int i = 0; i < n; i++) {
        if (fabsf(out[i] - data[i]) > 0.5f) {
            fprintf(stderr, "FAIL quant uniform roundtrip[%d]: got %.4f, expected %.4f\n",
                    i, out[i], data[i]);
            failures++;
            free(out); emb_quantized_free(q); free(data);
            return;
        }
    }
    printf("PASS quant uniform roundtrip\n");
    passes++;

    free(out);
    emb_quantized_free(q);
    free(data);
}

static void test_quant_roundtrip_linear(void) {
    /* Linear ramp: max quantization error ≤ 1/15 * range per block */
    int n = 128;
    float *data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) data[i] = (float)i;

    emb_quantized_t *q = emb_quantize_4bit(data, n);
    float *out = emb_dequantize_4bit(q);

    /* Block size=32, range per block=31. Max error = 31/15 ≈ 2.07 */
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(out[i] - data[i]);
        if (e > max_err) max_err = e;
    }
    check_true("quant linear max_err < 2.5", max_err < 2.5f);

    free(out);
    emb_quantized_free(q);
    free(data);
}

static void test_quant_roundtrip_signed(void) {
    /* Negative values: [-10, 10] range, 4-bit -> max error ≤ 20/15 ≈ 1.33 */
    int n = 32;
    float data[32];
    for (int i = 0; i < n; i++) data[i] = -10.0f + (20.0f / (n - 1)) * i;

    emb_quantized_t *q = emb_quantize_4bit(data, n);
    float *out = emb_dequantize_4bit(q);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(out[i] - data[i]);
        if (e > max_err) max_err = e;
    }
    check_true("quant signed max_err < 1.5", max_err < 1.5f);

    /* Extremes should be recovered exactly (they're the block min/max) */
    check_f("quant signed min", out[0],    data[0],    1e-4f);
    check_f("quant signed max", out[n-1],  data[n-1],  1e-4f);

    free(out);
    emb_quantized_free(q);
}

static void test_quant_metadata(void) {
    int n = 65;  /* Crosses a block boundary */
    float *data = malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) data[i] = (float)i * 0.5f;

    emb_quantized_t *q = emb_quantize_4bit(data, n);
    check_true("quant metadata num_elements", q->num_elements == n);
    /* ceil(65/32) = 3 blocks */
    check_true("quant metadata num_blocks", q->num_blocks == 3);

    emb_quantized_free(q);
    free(data);
}

static void test_quant_free_null(void) {
    emb_quantized_free(NULL);
    printf("PASS quant free NULL (no crash)\n");
    passes++;
}

static void test_quant_null_input(void) {
    emb_quantized_t *q = emb_quantize_4bit(NULL, 10);
    check_true("quant null input returns null", q == NULL);

    q = emb_quantize_4bit((float[]){1.0f}, 0);
    check_true("quant zero elements returns null", q == NULL);
}

/* =========================================================================
 * Cache store / lookup / has
 * ========================================================================= */

static void test_cache_miss_on_empty(void) {
    emb_cache_free();
    emb_cache_init();

    float *result = emb_cache_lookup("hello");
    check_true("cache miss on empty", result == NULL);
    check_true("cache has miss on empty", !emb_cache_has("hello"));
}

static void test_cache_store_and_hit(void) {
    emb_cache_free();
    emb_cache_init();

    float emb[] = {1.0f, 2.0f, 3.0f, 4.0f};
    emb_cache_store("test prompt", emb, 4);

    check_true("cache has after store", emb_cache_has("test prompt"));

    int n = 0;
    float *out = emb_cache_lookup_ex("test prompt", &n);
    check_true("cache lookup not null", out != NULL);
    check_true("cache lookup num_elements", n == 4);

    /* Values are 4-bit quantized; the range [1,4] gives max error ≤ 3/15 ≈ 0.2 */
    if (out) {
        check_f("cache hit emb[0]", out[0], emb[0], 0.3f);
        check_f("cache hit emb[3]", out[3], emb[3], 0.3f);
        free(out);
    }
}

static void test_cache_miss_wrong_prompt(void) {
    emb_cache_free();
    emb_cache_init();

    float emb[] = {1.0f, 2.0f};
    emb_cache_store("prompt A", emb, 2);

    float *out = emb_cache_lookup("prompt B");
    check_true("cache miss wrong prompt", out == NULL);
    if (out) free(out);
}

static void test_cache_clear(void) {
    emb_cache_free();
    emb_cache_init();

    float emb[] = {5.0f, 6.0f};
    emb_cache_store("p1", emb, 2);
    check_true("cache has before clear", emb_cache_has("p1"));

    emb_cache_clear();
    check_true("cache miss after clear", !emb_cache_has("p1"));
}

static void test_cache_overwrite(void) {
    emb_cache_free();
    emb_cache_init();

    /* Store a value, then overwrite with a very different one in the same slot.
     * To guarantee same slot we use the same prompt key. */
    float emb1[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float emb2[] = {10.0f, 10.0f, 10.0f, 10.0f};

    emb_cache_store("same key", emb1, 4);
    emb_cache_store("same key", emb2, 4);

    float *out = emb_cache_lookup("same key");
    check_true("cache overwrite not null", out != NULL);
    if (out) {
        /* After overwrite, should reflect emb2 (~10), not emb1 (~0) */
        check_true("cache overwrite value updated", out[0] > 5.0f);
        free(out);
    }
}

static void test_cache_stats(void) {
    emb_cache_free();
    emb_cache_init();

    int entries; size_t mem;
    emb_cache_stats(&entries, &mem);
    check_true("cache stats empty entries=0", entries == 0);
    check_true("cache stats empty mem=0", mem == 0);

    float emb[64];
    for (int i = 0; i < 64; i++) emb[i] = (float)i;
    emb_cache_store("stats test", emb, 64);

    emb_cache_stats(&entries, &mem);
    check_true("cache stats 1 entry", entries == 1);
    check_true("cache stats mem > 0", mem > 0);
}

static void test_cache_null_safety(void) {
    emb_cache_free();
    emb_cache_init();

    /* These should not crash */
    emb_cache_store(NULL, NULL, 0);
    float *r = emb_cache_lookup(NULL);
    check_true("cache lookup null prompt", r == NULL);
    check_true("cache has null prompt", !emb_cache_has(NULL));
}

static void test_cache_multiple_prompts(void) {
    /* Store two different prompts and verify they coexist (different slots) */
    emb_cache_free();
    emb_cache_init();

    float emb_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float emb_b[] = {9.0f, 8.0f, 7.0f, 6.0f};

    /* Use prompts that hash to different slots (best-effort; cache has 4 slots) */
    emb_cache_store("alpha", emb_a, 4);
    emb_cache_store("beta",  emb_b, 4);

    /* Both should be retrievable if they're in different slots */
    int has_a = emb_cache_has("alpha");
    int has_b = emb_cache_has("beta");

    /* At minimum one of them must be present (the cache may evict if same slot) */
    check_true("cache multi: at least one prompt cached", has_a || has_b);

    /* Whichever is present must return the right values */
    if (has_a) {
        float *out = emb_cache_lookup("alpha");
        if (out) {
            check_true("cache multi alpha value", out[0] < 5.0f);  /* ~1 */
            free(out);
        }
    }
    if (has_b) {
        float *out = emb_cache_lookup("beta");
        if (out) {
            check_true("cache multi beta value", out[0] > 5.0f);   /* ~9 */
            free(out);
        }
    }
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(void) {
    printf("=== Embcache unit tests ===\n\n");

    printf("-- 4-bit quantization --\n");
    test_quant_roundtrip_uniform();
    test_quant_roundtrip_linear();
    test_quant_roundtrip_signed();
    test_quant_metadata();
    test_quant_free_null();
    test_quant_null_input();

    printf("\n-- cache semantics --\n");
    test_cache_miss_on_empty();
    test_cache_store_and_hit();
    test_cache_miss_wrong_prompt();
    test_cache_clear();
    test_cache_overwrite();
    test_cache_stats();
    test_cache_null_safety();
    test_cache_multiple_prompts();

    printf("\n");
    if (failures > 0) {
        fprintf(stderr, "%d/%d FAILED\n", failures, passes + failures);
        return 1;
    }
    printf("All %d embcache tests PASSED\n", passes);
    return 0;
}
