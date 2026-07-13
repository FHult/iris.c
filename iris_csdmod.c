/*
 * iris_csdmod.c - CSDModulation MLP for the SREF joint-backbone style adapter.
 * See iris_csdmod.h. The math is identical to train/ip_adapter/model.py:CSDModulation:
 *   h = fc1(csd); h = h * sigmoid(h) (SiLU); delta = fc2(h).
 */
#include "iris_csdmod.h"
#include "iris_safetensors.h"
#include "iris_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int csd_mod_load(csd_mod_t *m, const char *path) {
    memset(m, 0, sizeof(*m));
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) {
        fprintf(stderr, "csd_mod: cannot open %s\n", path);
        return 1;
    }
    const safetensor_t *f1 = safetensors_find(sf, "fc1.weight"); /* [mlp, csd]    */
    const safetensor_t *f2 = safetensors_find(sf, "fc2.weight"); /* [hidden, mlp] */
    const safetensor_t *b1 = safetensors_find(sf, "fc1.bias");
    const safetensor_t *b2 = safetensors_find(sf, "fc2.bias");
    if (!f1 || !f2 || !b1 || !b2 || f1->ndim != 2 || f2->ndim != 2) {
        fprintf(stderr, "csd_mod: missing/malformed fc1/fc2 tensors in %s\n", path);
        safetensors_close(sf);
        return 1;
    }
    m->mlp_dim    = (int)f1->shape[0];
    m->csd_dim    = (int)f1->shape[1];
    m->hidden_dim = (int)f2->shape[0];
    if ((int)f2->shape[1] != m->mlp_dim) {
        fprintf(stderr, "csd_mod: fc1/fc2 dim mismatch (%d vs %lld)\n",
                m->mlp_dim, (long long)f2->shape[1]);
        safetensors_close(sf);
        return 1;
    }
    /* safetensors_get_f32 mallocs a copy (see iris_safetensors.c) -> safe after close. */
    m->fc1_w = safetensors_get_f32(sf, f1);
    m->fc1_b = safetensors_get_f32(sf, b1);
    m->fc2_w = safetensors_get_f32(sf, f2);
    m->fc2_b = safetensors_get_f32(sf, b2);
    safetensors_close(sf);
    if (!m->fc1_w || !m->fc1_b || !m->fc2_w || !m->fc2_b) {
        csd_mod_free(m);
        return 1;
    }
    return 0;
}

void csd_mod_forward(const csd_mod_t *m, const float *csd, float *delta, float *work) {
    /* work = fc1(csd) + b1   -> [mlp_dim] */
    iris_linear(work, csd, m->fc1_w, m->fc1_b, 1, m->csd_dim, m->mlp_dim);
    /* SiLU in place: work = work * sigmoid(work) */
    iris_silu(work, m->mlp_dim);
    /* delta = fc2(work) + b2 -> [hidden_dim] */
    iris_linear(delta, work, m->fc2_w, m->fc2_b, 1, m->mlp_dim, m->hidden_dim);
}

void csd_mod_free(csd_mod_t *m) {
    if (!m) return;
    free(m->fc1_w);
    free(m->fc1_b);
    free(m->fc2_w);
    free(m->fc2_b);
    memset(m, 0, sizeof(*m));
}
