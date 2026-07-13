/*
 * iris_csdmod.h - CSDModulation: FiLM a content-invariant CSD style vector into
 * the Flux DiT timestep embedding (temb) for the SREF joint-backbone style adapter.
 *
 *   delta = fc2( silu( fc1(csd) ) )        (2-layer MLP, SiLU activation)
 *   temb += style_scale * delta            (applied once; delta is const across steps)
 *
 * Mirrors train/ip_adapter/model.py:CSDModulation exactly. csd is the 768-d L2-normalised
 * CSD style embedding of the reference image (computed upstream in Python and passed in).
 */
#ifndef IRIS_CSDMOD_H
#define IRIS_CSDMOD_H

typedef struct {
    int csd_dim;      /* 768  (fc1 in)  */
    int mlp_dim;      /* 1024 (fc1 out / fc2 in) */
    int hidden_dim;   /* 3072 (fc2 out; == transformer hidden / temb dim) */
    float *fc1_w;     /* [mlp_dim, csd_dim]     (row-major, W^T layout for iris_linear) */
    float *fc1_b;     /* [mlp_dim]              */
    float *fc2_w;     /* [hidden_dim, mlp_dim]  */
    float *fc2_b;     /* [hidden_dim]           */
} csd_mod_t;

/* Load fc1/fc2 weight+bias from a safetensors file whose keys are
 * "fc1.weight","fc1.bias","fc2.weight","fc2.bias". Dims are inferred from the shapes.
 * Returns 0 on success, non-zero on failure (m is zeroed on failure). */
int csd_mod_load(csd_mod_t *m, const char *path);

/* delta[hidden_dim] = fc2(silu(fc1(csd))). `work` must hold >= mlp_dim floats.
 * The caller multiplies by style_scale and adds delta to temb. */
void csd_mod_forward(const csd_mod_t *m, const float *csd, float *delta, float *work);

void csd_mod_free(csd_mod_t *m);

#endif /* IRIS_CSDMOD_H */
