/*
 * test_lora.c - Unit tests for LoRA math (lora_apply)
 *
 * Tests the lora_apply function in isolation:
 *   1. Identity check: scale=0 produces no change
 *   2. Correctness: known weights produce expected output
 *   3. Rank-1 case: easy to verify by hand
 *
 * Build: gcc -O2 -I.. -o /tmp/test_lora debug/test_lora.c ../iris_lora.c ../iris_safetensors.c -lm
 * Run:   /tmp/test_lora
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../iris_lora.h"

static int failures = 0;

static void check(const char *name, float got, float expected, float tol) {
    float diff = fabsf(got - expected);
    if (diff > tol) {
        fprintf(stderr, "FAIL %s: got %.6f, expected %.6f (diff %.2e)\n",
                name, got, expected, diff);
        failures++;
    } else {
        printf("PASS %s\n", name);
    }
}

/* Test 1: scale=0 → output unchanged */
static void test_scale_zero(void) {
    lora_adapter_t a;
    int seq = 3, in_dim = 4, out_dim = 2, rank = 2;
    a.rank    = rank;
    a.in_dim  = in_dim;
    a.out_dim = out_dim;

    /* Arbitrary lora_A and lora_B */
    float lora_A[] = {1, 0, 0, 0,
                      0, 1, 0, 0};
    float lora_B[] = {1, 0,
                      0, 1};
    a.lora_A = lora_A;
    a.lora_B = lora_B;

    /* x: [seq, in_dim] */
    float x[] = {1,2,3,4,  5,6,7,8,  9,10,11,12};

    /* out: [seq, out_dim] */
    float out[] = {10,20,  30,40,  50,60};
    float out_orig[6];
    memcpy(out_orig, out, sizeof(out));

    float scratch[6 * 2];
    lora_apply(&a, 0.0f, x, out, seq, scratch);

    for (int i = 0; i < 6; i++) {
        char name[64];
        snprintf(name, sizeof(name), "scale=0 out[%d]", i);
        check(name, out[i], out_orig[i], 1e-5f);
    }
}

/* Test 2: rank-1, in_dim=2, out_dim=2, verify by hand
 * lora_A = [1, 2]           (1×2)
 * lora_B = [[3], [4]]       (2×1)
 * x[0] = [1, 0]
 * lora_A @ x[0] = [1*1 + 2*0] = [1]  (scratch: [1])
 * scale * lora_B @ scratch = 0.5 * [[3],[4]] @ [1] = [[1.5],[2]]
 * out[0] += [1.5, 2.0]
 */
static void test_rank1_correctness(void) {
    lora_adapter_t a;
    a.rank = 1; a.in_dim = 2; a.out_dim = 2;
    float lora_A[] = {1.0f, 2.0f};       /* [rank=1, in=2] */
    float lora_B[] = {3.0f, 4.0f};       /* [out=2, rank=1] */
    a.lora_A = lora_A;
    a.lora_B = lora_B;

    float x[] = {1.0f, 0.0f};  /* [seq=1, in=2] */
    float out[] = {10.0f, 20.0f};  /* [seq=1, out=2] */
    float scratch[2];

    lora_apply(&a, 0.5f, x, out, 1, scratch);

    /* Expected: out[0] = 10 + 0.5 * 3 * 1 = 11.5
     *           out[1] = 20 + 0.5 * 4 * 1 = 22.0 */
    check("rank1 out[0]", out[0], 11.5f, 1e-5f);
    check("rank1 out[1]", out[1], 22.0f, 1e-5f);
}

/* Test 3: multi-token, verify linearity */
static void test_multi_token(void) {
    lora_adapter_t a;
    int rank = 2, in_dim = 3, out_dim = 2, seq = 2;
    a.rank = rank; a.in_dim = in_dim; a.out_dim = out_dim;

    /* lora_A: [2, 3] */
    float lora_A[] = {1, 0, 0,
                      0, 1, 0};
    /* lora_B: [2, 2] */
    float lora_B[] = {2, 0,
                      0, 3};
    a.lora_A = lora_A;
    a.lora_B = lora_B;

    /* x: [2, 3] */
    float x[] = {1, 2, 3,
                 4, 5, 6};
    float out[] = {0, 0,   0, 0};
    float scratch[4 * 2];

    lora_apply(&a, 1.0f, x, out, seq, scratch);

    /* Token 0: scratch = lora_A @ [1,2,3]^T = [1, 2]
     * out += lora_B @ [1,2] = [2*1+0*2, 0*1+3*2] = [2, 6]
     * Token 1: scratch = lora_A @ [4,5,6]^T = [4, 5]
     * out += lora_B @ [4,5] = [2*4, 3*5] = [8, 15]
     */
    check("multi_token out[0]", out[0], 2.0f, 1e-5f);
    check("multi_token out[1]", out[1], 6.0f, 1e-5f);
    check("multi_token out[2]", out[2], 8.0f, 1e-5f);
    check("multi_token out[3]", out[3], 15.0f, 1e-5f);
}

/* Test 4: lora_free handles NULL gracefully */
static void test_free_null(void) {
    lora_free(NULL);
    printf("PASS free_null (no crash)\n");
}

int main(void) {
    printf("=== LoRA unit tests ===\n");
    test_scale_zero();
    test_rank1_correctness();
    test_multi_token();
    test_free_null();

    if (failures > 0) {
        fprintf(stderr, "\n%d FAILED\n", failures);
        return 1;
    }
    printf("\nAll LoRA tests PASSED\n");
    return 0;
}
