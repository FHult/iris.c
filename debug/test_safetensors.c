/*
 * debug/test_safetensors.c — unit tests for the safetensors header parser.
 *
 * The parser had no dedicated C test, yet it underpins every weight load. Pins:
 *   - "__metadata__":null  is skipped (mlx writes this — it previously broke parse);
 *   - "__metadata__":{...} object form is also skipped (diffusers);
 *   - F32 / F16 / BF16 / I8 dtypes parse, with correct shape / numel / data;
 *   - safetensors_get_f32 dequantises F16 and BF16 to the expected values;
 *   - find() returns NULL for an absent tensor.
 *
 * Builds tiny in-memory safetensors files (8-byte LE header len + JSON + data) in
 * /tmp — no model, no GPU. Build:
 *   cc -O2 -I. -o /tmp/t debug/test_safetensors.c iris_safetensors.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "iris_safetensors.h"

static int failures = 0, passes = 0;
static void ok(const char *name, int cond) {
    if (cond) { printf("PASS %s\n", name); passes++; }
    else { fprintf(stderr, "FAIL %s\n", name); failures++; }
}

/* Write [uint64 LE header_len][json][data] to path. */
static int write_st(const char *path, const char *json, const void *data, size_t data_len) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint64_t hlen = strlen(json);
    fwrite(&hlen, sizeof(uint64_t), 1, f);     /* little-endian on this target */
    fwrite(json, 1, hlen, f);
    if (data_len) fwrite(data, 1, data_len, f);
    fclose(f);
    return 0;
}

/* IEEE-754 half / bfloat16 encodings of a few exact values. */
static uint16_t f16_one  = 0x3C00, f16_half = 0x3800;   /* 1.0, 0.5 */
static uint16_t bf16_two = 0x4000, bf16_qtr = 0x3E80;   /* 2.0, 0.25 */

int main(void) {
    const char *path = "/tmp/iris_test_st.safetensors";

    /* Case 1: __metadata__:null + F32 / F16 / BF16 / I8 tensors.
       data layout (contiguous from data start):
         a   F32 [2,2]  -> [0,16)
         h   F16 [2]    -> [16,20)
         b   BF16 [2]   -> [20,24)
         q   I8  [2,3]  -> [24,30)
         q.scale F32[2] -> [30,38)                                   */
    {
        unsigned char data[38];
        float a[4] = {1.0f, -2.0f, 3.5f, 0.0f};
        memcpy(data + 0, a, 16);
        uint16_t h[2] = {f16_one, f16_half};   memcpy(data + 16, h, 4);
        uint16_t b[2] = {bf16_two, bf16_qtr};  memcpy(data + 20, b, 4);
        int8_t q[6] = {1, -2, 127, -127, 0, 5}; memcpy(data + 24, q, 6);
        float sc[2] = {0.5f, 2.0f};            memcpy(data + 30, sc, 8);

        const char *json =
            "{\"__metadata__\":null,"
            "\"a\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
            "\"h\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[16,20]},"
            "\"b\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[20,24]},"
            "\"q\":{\"dtype\":\"I8\",\"shape\":[2,3],\"data_offsets\":[24,30]},"
            "\"q.scale\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[30,38]}}";
        write_st(path, json, data, sizeof(data));

        safetensors_file_t *sf = safetensors_open(path);
        ok("open with __metadata__:null", sf != NULL);
        if (sf) {
            const safetensor_t *ta = safetensors_find(sf, "a");
            ok("find a (F32)", ta && ta->dtype == DTYPE_F32 && safetensor_numel(ta) == 4);
            float *fa = ta ? safetensors_get_f32(sf, ta) : NULL;
            ok("F32 values", fa && fa[0]==1.0f && fa[1]==-2.0f && fa[2]==3.5f && fa[3]==0.0f);
            free(fa);

            const safetensor_t *th = safetensors_find(sf, "h");
            float *fh = th ? safetensors_get_f32(sf, th) : NULL;
            ok("F16 dtype + dequant", th && th->dtype==DTYPE_F16 &&
               fh && fabsf(fh[0]-1.0f)<1e-6 && fabsf(fh[1]-0.5f)<1e-6);
            free(fh);

            const safetensor_t *tb = safetensors_find(sf, "b");
            float *fb = tb ? safetensors_get_f32(sf, tb) : NULL;
            ok("BF16 dtype + dequant", tb && tb->dtype==DTYPE_BF16 &&
               fb && fabsf(fb[0]-2.0f)<1e-6 && fabsf(fb[1]-0.25f)<1e-6);
            free(fb);

            const safetensor_t *tq = safetensors_find(sf, "q");
            const int8_t *dq = tq ? (const int8_t *)safetensors_data(sf, tq) : NULL;
            ok("I8 dtype + shape + data", tq && tq->dtype==DTYPE_I8 &&
               safetensor_numel(tq)==6 && dq && dq[0]==1 && dq[2]==127 && dq[3]==-127);

            ok("scale present", safetensors_find(sf, "q.scale") != NULL);
            ok("absent tensor -> NULL", safetensors_find(sf, "nope") == NULL);
            safetensors_close(sf);
        }
    }

    /* Case 2: __metadata__ object form is also skipped. */
    {
        unsigned char data[4];
        float x = 7.0f; memcpy(data, &x, 4);
        const char *json =
            "{\"__metadata__\":{\"format\":\"pt\",\"k\":\"v\"},"
            "\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
        write_st(path, json, data, sizeof(data));
        safetensors_file_t *sf = safetensors_open(path);
        ok("open with __metadata__ object", sf != NULL);
        if (sf) {
            const safetensor_t *t = safetensors_find(sf, "x");
            float *f = t ? safetensors_get_f32(sf, t) : NULL;
            ok("object-meta value read", f && f[0]==7.0f);
            free(f);
            safetensors_close(sf);
        }
    }

    remove(path);
    printf("\n%d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}
