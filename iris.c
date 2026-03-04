/*
 * FLUX Main Implementation
 *
 * Main entry point for the FLUX.2 klein 4B inference engine.
 * Ties together all components: tokenizer, text encoder, VAE, transformer, sampling.
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
#include "iris_qwen3.h"
#include "iris_lora.h"
#include "embcache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#ifdef USE_METAL
#include "iris_metal.h"
#endif

/* ========================================================================
 * Forward Declarations for Internal Types
 * ======================================================================== */

typedef struct flux_tokenizer flux_tokenizer;
typedef struct flux_vae flux_vae_t;
typedef struct flux_transformer flux_transformer_t;

/* Internal function declarations */
extern flux_tokenizer *flux_tokenizer_load(const char *path);
extern void flux_tokenizer_free(flux_tokenizer *tok);
extern int *flux_tokenize(flux_tokenizer *tok, const char *text,
                          int *num_tokens, int max_len);

extern flux_vae_t *flux_vae_load(FILE *f);
extern flux_vae_t *flux_vae_load_safetensors(safetensors_file_t *sf);
extern flux_vae_t *flux_vae_load_safetensors_ex(safetensors_file_t *sf,
                                                  int z_channels,
                                                  float scaling_factor,
                                                  float shift_factor);
extern void flux_vae_free(flux_vae_t *vae);
extern float *flux_vae_encode(flux_vae_t *vae, const float *img,
                              int batch, int H, int W, int *out_h, int *out_w);
extern flux_image *flux_vae_decode(flux_vae_t *vae, const float *latent,
                                   int batch, int latent_h, int latent_w);
extern float *flux_image_to_tensor(const flux_image *img);

extern flux_transformer_t *flux_transformer_load(FILE *f);
extern flux_transformer_t *flux_transformer_load_safetensors(const char *model_dir);
extern flux_transformer_t *flux_transformer_load_safetensors_mmap(const char *model_dir);
extern void flux_transformer_free(flux_transformer_t *tf);
extern int flux_transformer_hidden_size(flux_transformer_t *tf);
extern int flux_transformer_num_double_layers(flux_transformer_t *tf);
extern int flux_transformer_num_single_layers(flux_transformer_t *tf);
extern void flux_transformer_set_lora(flux_transformer_t *tf, lora_state_t *lora);
extern float *flux_transformer_forward(flux_transformer_t *tf,
                                        const float *img_latent, int img_h, int img_w,
                                        const float *txt_emb, int txt_seq,
                                        float timestep);

extern float *flux_sample_euler(void *transformer, void *text_encoder,
                                float *z, int batch, int channels, int h, int w,
                                const float *text_emb, int text_seq,
                                const float *schedule, int num_steps,
                                void (*progress_callback)(int step, int total));
extern float *flux_sample_euler_with_refs(void *transformer, void *text_encoder,
                                          float *z, int batch, int channels, int h, int w,
                                          const float *ref_latent, int ref_h, int ref_w,
                                          int t_offset,
                                          const float *text_emb, int text_seq,
                                          const float *schedule, int num_steps,
                                          void (*progress_callback)(int step, int total));

/* Multi-reference support */
typedef struct {
    const float *latent;
    int h, w;
    int t_offset;
} flux_ref_t;

extern float *flux_sample_euler_with_multi_refs(void *transformer, void *text_encoder,
                                                float *z, int batch, int channels, int h, int w,
                                                const flux_ref_t *refs, int num_refs,
                                                const float *text_emb, int text_seq,
                                                const float *schedule, int num_steps,
                                                void (*progress_callback)(int step, int total));

/* CFG sampling (for base model) */
extern float *flux_sample_euler_cfg(void *transformer, void *text_encoder,
                                     float *z, int batch, int channels, int h, int w,
                                     const float *text_emb_cond, int text_seq_cond,
                                     const float *text_emb_uncond, int text_seq_uncond,
                                     float guidance_scale,
                                     const float *schedule, int num_steps,
                                     void (*progress_callback)(int step, int total));
extern float *flux_sample_euler_cfg_with_refs(void *transformer, void *text_encoder,
                                               float *z, int batch, int channels, int h, int w,
                                               const float *ref_latent, int ref_h, int ref_w,
                                               int t_offset,
                                               const float *text_emb_cond, int text_seq_cond,
                                               const float *text_emb_uncond, int text_seq_uncond,
                                               float guidance_scale,
                                               const float *schedule, int num_steps,
                                               void (*progress_callback)(int step, int total));
extern float *flux_sample_euler_cfg_with_multi_refs(void *transformer, void *text_encoder,
                                                     float *z, int batch, int channels, int h, int w,
                                                     const flux_ref_t *refs, int num_refs,
                                                     const float *text_emb_cond, int text_seq_cond,
                                                     const float *text_emb_uncond, int text_seq_uncond,
                                                     float guidance_scale,
                                                     const float *schedule, int num_steps,
                                                     void (*progress_callback)(int step, int total));

extern float *flux_linear_schedule(int num_steps);
extern float *flux_power_schedule(int num_steps, float alpha);
extern float *flux_official_schedule(int num_steps, int image_seq_len);
extern float *flux_zimage_schedule(int num_steps, int image_seq_len);
extern float *flux_init_noise(int batch, int channels, int h, int w, int64_t seed);

/* Z-Image transformer and sampling */
typedef struct zi_transformer zi_transformer_t;
extern zi_transformer_t *zi_transformer_load_safetensors(const char *model_dir,
                                                           int dim, int n_heads,
                                                           int n_layers, int n_refiner,
                                                           int cap_feat_dim, int in_channels,
                                                           int patch_size, float rope_theta,
                                                           const int *axes_dims);
extern void zi_transformer_free(zi_transformer_t *tf);
extern float *flux_sample_euler_zimage(void *transformer,
                                        float *z, int batch, int channels, int h, int w,
                                        int patch_size,
                                        const float *cap_feats, int cap_seq,
                                        const float *schedule, int num_steps,
                                        void (*progress_callback)(int step, int total));

/* Return schedule based on params: linear, power, or official shifted sigmoid. */
static float *flux_selected_schedule(const flux_params *p, int image_seq_len) {
    if (p->power_schedule)
        return flux_power_schedule(p->num_steps, p->power_alpha);
    if (p->linear_schedule)
        return flux_linear_schedule(p->num_steps);
    return flux_official_schedule(p->num_steps, image_seq_len);
}

/* Return Z-Image schedule based on params.
 * Default is Z-Image FlowMatch schedule; linear/power are explicit overrides. */
static float *flux_selected_zimage_schedule(const flux_params *p, int image_seq_len) {
    if (p->power_schedule)
        return flux_power_schedule(p->num_steps, p->power_alpha);
    if (p->linear_schedule)
        return flux_linear_schedule(p->num_steps);
    return flux_zimage_schedule(p->num_steps, image_seq_len);
}

/* ========================================================================
 * Text Encoder (Qwen3)
 * ======================================================================== */

/* Qwen3 text encoder is implemented in flux_qwen3.c */

/* ========================================================================
 * Main Context Structure
 * ======================================================================== */

struct flux_ctx {
    /* Components */
    flux_tokenizer *tokenizer;
    qwen3_encoder_t *qwen3_encoder;
    flux_vae_t *vae;
    flux_transformer_t *transformer;
    zi_transformer_t *zi_transformer;

    /* Configuration */
    int max_width;
    int max_height;
    int default_steps;
    float default_guidance;
    int is_distilled;  /* 1 = distilled (4-step), 0 = base (50-step CFG) */
    int text_dim;      /* Text embedding dimension (7680 for 4B, varies for 9B) */
    int is_non_commercial; /* 1 if model has non-commercial license (9B) */
    int num_heads;     /* Transformer attention heads (24 for 4B, 32 for 9B) */
    int is_zimage;     /* 1 = Z-Image S3-DiT, 0 = Flux MMDiT */

    /* Z-Image specific config (from transformer/config.json) */
    int zi_dim;            /* Hidden dim (3840) */
    int zi_n_layers;       /* Main transformer layers (30) */
    int zi_n_refiner;      /* Noise/context refiner layers (2) */
    int zi_cap_feat_dim;   /* Caption feature dim (2560) */
    int zi_in_channels;    /* VAE latent channels (16) */
    int zi_patch_size;     /* Spatial patch size (2) */
    float zi_rope_theta;   /* RoPE theta (256.0) */
    int zi_axes_dims[3];   /* RoPE axis dims [32, 48, 48] */
    int zi_latent_channels;/* Patchified latent channels (64 = 16*2*2) */

    /* VAE config (read from vae/config.json) */
    int vae_z_channels;    /* Latent channels before patchify (32 Flux, 16 Z-Image) */
    float vae_scaling;     /* Scaling factor (0.3611 for Z-Image, 0 = use batch norm) */
    float vae_shift;       /* Shift factor (0.1159 for Z-Image, 0 = use batch norm) */

    /* Model info */
    char model_name[64];
    char model_version[32];
    char model_dir[512];  /* For reloading text encoder if released */

    /* Memory mode */
    int use_mmap;  /* Use mmap for text encoder (lower memory, slower) */
    int keep_models_loaded;  /* Server mode: keep encoder+transformer resident */
};

/* Global error message */
static char g_error_msg[256] = {0};

const char *flux_get_error(void) {
    return g_error_msg;
}

void flux_set_step_image_callback(flux_ctx *ctx, flux_step_image_cb_t callback) {
    flux_step_image_callback = callback;
    flux_step_image_vae = callback ? ctx->vae : NULL;
}

static void set_error(const char *msg) {
    strncpy(g_error_msg, msg, sizeof(g_error_msg) - 1);
    g_error_msg[sizeof(g_error_msg) - 1] = '\0';
}

/* ========================================================================
 * Model Loading from HuggingFace-style directory with safetensors files
 * ======================================================================== */

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

/* Main model loading entry point. Parses model_index.json to auto-detect
 * the model type (Flux vs Z-Image, distilled vs base), then reads
 * transformer/config.json and vae/config.json for architecture parameters
 * (hidden dim, heads, layers, etc.). Only the VAE (~300MB) is loaded
 * eagerly; the text encoder and transformer are deferred to generation
 * time so they can be swapped in/out on memory-constrained systems. */
flux_ctx *flux_load_dir(const char *model_dir) {
    char path[1024];

    flux_ctx *ctx = calloc(1, sizeof(flux_ctx));
    if (!ctx) {
        set_error("Out of memory");
        return NULL;
    }

    /* Set defaults - max 1792x1792 (requires ~18GB VAE work buffers) */
    ctx->max_width = FLUX_VAE_MAX_DIM;
    ctx->max_height = FLUX_VAE_MAX_DIM;
    strncpy(ctx->model_version, "1.0", sizeof(ctx->model_version) - 1);
    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);

    /* Autodetect model type from model_index.json.
     * Distilled model has "is_distilled": true, base model does not.
     * Z-Image has "_class_name": "ZImagePipeline". */
    ctx->is_distilled = 1;  /* Default to distilled */
    ctx->is_zimage = 0;
    snprintf(path, sizeof(path), "%s/model_index.json", model_dir);
    if (file_exists(path)) {
        FILE *f = fopen(path, "r");
        if (f) {
            char buf[4096];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            buf[n] = '\0';
            fclose(f);
            /* Check for Z-Image pipeline */
            if (strstr(buf, "ZImagePipeline") || strstr(buf, "Z-Image")) {
                ctx->is_zimage = 1;
            }
            /* If "is_distilled" is present and true, it's distilled.
             * If absent, it's the base model. */
            if (!strstr(buf, "\"is_distilled\": true") &&
                !strstr(buf, "\"is_distilled\":true")) {
                ctx->is_distilled = 0;
            }
        }
    }

    /* Read transformer/config.json to determine model size and architecture. */
    int num_heads = 24;  /* default 4B */
    ctx->text_dim = 7680;  /* default 4B: 3 * 2560 */
    snprintf(path, sizeof(path), "%s/transformer/config.json", model_dir);
    if (file_exists(path)) {
        FILE *f = fopen(path, "r");
        if (f) {
            char buf[8192];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            buf[n] = '\0';
            fclose(f);
            char *p;
            if ((p = strstr(buf, "\"num_attention_heads\""))) {
                if ((p = strchr(p, ':'))) num_heads = atoi(p + 1);
            }
            int joint_dim = 0;
            if ((p = strstr(buf, "\"joint_attention_dim\""))) {
                if ((p = strchr(p, ':'))) joint_dim = atoi(p + 1);
            }
            if (joint_dim > 0) ctx->text_dim = joint_dim;

            /* Z-Image autodetection: look for Z-Image-specific fields */
            if ((p = strstr(buf, "\"cap_feat_dim\""))) {
                ctx->is_zimage = 1;
            }

            /* Parse Z-Image config if detected */
            if (ctx->is_zimage) {
                /* dim (hidden size) */
                ctx->zi_dim = 3840;
                if ((p = strstr(buf, "\"dim\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_dim = atoi(colon + 1);
                }
                num_heads = ctx->zi_dim / 128;  /* head_dim = 128 */

                /* n_layers */
                ctx->zi_n_layers = 30;
                if ((p = strstr(buf, "\"n_layers\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_n_layers = atoi(colon + 1);
                }

                /* n_refiner_layers */
                ctx->zi_n_refiner = 2;
                if ((p = strstr(buf, "\"n_refiner_layers\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_n_refiner = atoi(colon + 1);
                }

                /* cap_feat_dim */
                ctx->zi_cap_feat_dim = 2560;
                if ((p = strstr(buf, "\"cap_feat_dim\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_cap_feat_dim = atoi(colon + 1);
                }

                /* in_channels */
                ctx->zi_in_channels = 16;
                if ((p = strstr(buf, "\"in_channels\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_in_channels = atoi(colon + 1);
                }

                /* patch_size */
                ctx->zi_patch_size = 2;
                if ((p = strstr(buf, "\"patch_size\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_patch_size = atoi(colon + 1);
                }

                /* rope_theta */
                ctx->zi_rope_theta = 256.0f;
                if ((p = strstr(buf, "\"rope_theta\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_rope_theta = atof(colon + 1);
                }

                /* axes_dims - parse JSON array [32, 48, 48] */
                ctx->zi_axes_dims[0] = 32;
                ctx->zi_axes_dims[1] = 48;
                ctx->zi_axes_dims[2] = 48;
                if ((p = strstr(buf, "\"axes_dims\""))) {
                    char *bracket = strchr(p, '[');
                    if (bracket) {
                        ctx->zi_axes_dims[0] = atoi(bracket + 1);
                        char *comma1 = strchr(bracket, ',');
                        if (comma1) {
                            ctx->zi_axes_dims[1] = atoi(comma1 + 1);
                            char *comma2 = strchr(comma1 + 1, ',');
                            if (comma2) {
                                ctx->zi_axes_dims[2] = atoi(comma2 + 1);
                            }
                        }
                    }
                }

                /* Derived values */
                ctx->zi_latent_channels = ctx->zi_in_channels *
                                          ctx->zi_patch_size * ctx->zi_patch_size;
                ctx->text_dim = ctx->zi_cap_feat_dim;  /* 2560 for Z-Image */
            }
        }
    }

    /* Read vae/config.json for Z-Image scaling/shift factors */
    ctx->vae_z_channels = FLUX_VAE_Z_CHANNELS;  /* default: 32 */
    ctx->vae_scaling = 0.0f;
    ctx->vae_shift = 0.0f;
    snprintf(path, sizeof(path), "%s/vae/config.json", model_dir);
    if (file_exists(path)) {
        FILE *f = fopen(path, "r");
        if (f) {
            char buf[4096];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            buf[n] = '\0';
            fclose(f);
            char *p;
            if ((p = strstr(buf, "\"latent_channels\""))) {
                char *colon = strchr(p, ':');
                if (colon) {
                    int lc = atoi(colon + 1);
                    if (lc > 0) ctx->vae_z_channels = lc;
                }
            }
            if ((p = strstr(buf, "\"scaling_factor\""))) {
                char *colon = strchr(p, ':');
                if (colon) ctx->vae_scaling = atof(colon + 1);
            }
            if ((p = strstr(buf, "\"shift_factor\""))) {
                char *colon = strchr(p, ':');
                if (colon) ctx->vae_shift = atof(colon + 1);
            }
        }
    }

    /* Determine model variant name based on architecture. */
    if (ctx->is_zimage) {
        int hidden_size = ctx->zi_dim;
        const char *size_label = "6B";  /* Z-Image-Turbo is 6B */
        if (hidden_size != 3840) {
            size_label = (hidden_size > 3840) ? "large" : "small";
        }
        ctx->is_non_commercial = 0;  /* Z-Image is Apache 2.0 */
        ctx->num_heads = num_heads;
        ctx->default_steps = 9;       /* 8 NFE = 9 scheduler steps */
        ctx->default_guidance = 0.0f; /* No CFG for Z-Image-Turbo */
        ctx->is_distilled = 1;        /* Treat as distilled (no CFG) */
        snprintf(ctx->model_name, sizeof(ctx->model_name),
                 "Z-Image-Turbo-%s", size_label);
    } else {
        int hidden_size = num_heads * 128;  /* head_dim is always 128 */
        const char *size_label = (hidden_size > 3072) ? "9B" : "4B";
        ctx->is_non_commercial = (hidden_size > 3072) ? 1 : 0;
        ctx->num_heads = num_heads;

        if (ctx->is_distilled) {
            ctx->default_steps = 4;
            ctx->default_guidance = 1.0f;
            snprintf(ctx->model_name, sizeof(ctx->model_name),
                     "FLUX.2-klein-%s", size_label);
        } else {
            ctx->default_steps = 50;
            ctx->default_guidance = 4.0f;
            snprintf(ctx->model_name, sizeof(ctx->model_name),
                     "FLUX.2-klein-base-%s", size_label);
        }
    }

    /* Load VAE only at startup (~300MB).
     * Transformer and text encoder are loaded on-demand during generation
     * to support systems with limited RAM (e.g., 16GB). */
    snprintf(path, sizeof(path), "%s/vae/diffusion_pytorch_model.safetensors", model_dir);
    if (file_exists(path)) {
        safetensors_file_t *sf = safetensors_open(path);
        if (sf) {
            ctx->vae = flux_vae_load_safetensors_ex(sf,
                ctx->vae_z_channels, ctx->vae_scaling, ctx->vae_shift);
            safetensors_close(sf);
        }
    }

    /* Verify VAE is loaded */
    if (!ctx->vae) {
        set_error("Failed to load VAE - cannot generate images");
        flux_free(ctx);
        return NULL;
    }

    /* Verify transformer dir exists (will be loaded on-demand) */
    snprintf(path, sizeof(path), "%s/transformer/config.json", model_dir);
    if (!file_exists(path)) {
        /* Fallback: check for single safetensors file */
        snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors", model_dir);
        if (!file_exists(path)) {
            set_error("Transformer model not found (missing config.json and safetensors)");
            flux_free(ctx);
            return NULL;
        }
    }
    /* Text encoder and transformer are loaded on-demand to reduce peak memory. */

    /* Initialize RNG */
    flux_rng_seed((uint64_t)time(NULL));

    return ctx;
}

void flux_free(flux_ctx *ctx) {
    if (!ctx) return;

    flux_tokenizer_free(ctx->tokenizer);
    qwen3_encoder_free(ctx->qwen3_encoder);
    flux_vae_free(ctx->vae);
    flux_transformer_free(ctx->transformer);
    zi_transformer_free(ctx->zi_transformer);

    free(ctx);
}

void flux_set_mmap(flux_ctx *ctx, int enable) {
    if (ctx) ctx->use_mmap = enable;
}

void flux_set_keep_models_loaded(flux_ctx *ctx, int enable) {
    if (ctx) ctx->keep_models_loaded = enable;
}

int flux_is_distilled(flux_ctx *ctx) {
    return ctx ? ctx->is_distilled : 1;
}

int flux_is_zimage(flux_ctx *ctx) {
    return ctx ? ctx->is_zimage : 0;
}

static int flux_load_transformer_if_needed(flux_ctx *ctx);

void flux_set_base_mode(flux_ctx *ctx) {
    if (!ctx) return;
    ctx->is_distilled = 0;
    ctx->default_steps = 50;
    ctx->default_guidance = 4.0f;
    const char *size_label = ctx->is_non_commercial ? "9B" : "4B";
    snprintf(ctx->model_name, sizeof(ctx->model_name),
             "FLUX.2-klein-base-%s", size_label);
}

int flux_load_lora(flux_ctx *ctx, const char *path, float scale) {
    if (!ctx) return -1;
    if (!path || !*path) return -1;

    /* Load transformer on-demand (it's lazy-loaded, may not be present yet) */
    if (!flux_load_transformer_if_needed(ctx)) return -1;

    int hidden = flux_transformer_hidden_size(ctx->transformer);
    int num_double = flux_transformer_num_double_layers(ctx->transformer);
    int num_single = flux_transformer_num_single_layers(ctx->transformer);

    lora_state_t *lora = lora_load(path, num_double, num_single, hidden, scale);
    if (!lora) return -1;

    flux_transformer_set_lora(ctx->transformer, lora);
    return 0;
}

void flux_unload_lora(flux_ctx *ctx) {
    if (!ctx || !ctx->transformer) return;
    flux_transformer_set_lora(ctx->transformer, NULL);
}

/* Free the Qwen3 text encoder (~4-8GB) to make room for the transformer.
 * The encoder and transformer can't coexist in memory on most machines,
 * so this is called after text encoding and before denoising. On Metal,
 * also resets all GPU state (weight caches, pools) to avoid stale data
 * when the transformer loads into the same memory regions. */
void flux_release_text_encoder(flux_ctx *ctx) {
    if (!ctx || !ctx->qwen3_encoder) return;

    /* In server mode, keep encoder loaded to avoid reload + GPU cache rebuild */
    if (ctx->keep_models_loaded) return;

    qwen3_encoder_free(ctx->qwen3_encoder);
    ctx->qwen3_encoder = NULL;

#ifdef USE_METAL
    /* Reset all GPU state to ensure clean slate for transformer.
     * This clears weight caches, activation pools, and pending commands. */
    flux_metal_reset();
#endif
}

/* Lazy-load the Flux transformer from safetensors files. Deferred to
 * generation time because the text encoder must be freed first -- both
 * are too large to fit in memory simultaneously. Once loaded, the
 * transformer persists across generations (no reload per image). */
static int flux_load_transformer_if_needed(flux_ctx *ctx) {
    if (ctx->transformer) return 1;  /* Already loaded */

    if (flux_phase_callback) flux_phase_callback("Loading FLUX.2 transformer", 0);
    if (ctx->use_mmap) {
        ctx->transformer = flux_transformer_load_safetensors_mmap(ctx->model_dir);
    } else {
        ctx->transformer = flux_transformer_load_safetensors(ctx->model_dir);
    }
    if (flux_phase_callback) flux_phase_callback("Loading FLUX.2 transformer", 1);

    if (!ctx->transformer) {
        set_error("Failed to load transformer");
        return 0;
    }
    return 1;
}

/* Load Z-Image transformer on-demand if not already loaded */
static int flux_load_zimage_transformer_if_needed(flux_ctx *ctx) {
    if (ctx->zi_transformer) return 1;  /* Already loaded */

    if (flux_phase_callback) flux_phase_callback("Loading Z-Image transformer", 0);
    ctx->zi_transformer = zi_transformer_load_safetensors(
        ctx->model_dir,
        ctx->zi_dim, ctx->zi_dim / 128, ctx->zi_n_layers, ctx->zi_n_refiner,
        ctx->zi_cap_feat_dim, ctx->zi_in_channels, ctx->zi_patch_size,
        ctx->zi_rope_theta, ctx->zi_axes_dims);
    if (flux_phase_callback) flux_phase_callback("Loading Z-Image transformer", 1);

    if (!ctx->zi_transformer) {
        set_error("Failed to load Z-Image transformer");
        return 0;
    }
    return 1;
}

/* Get transformer for debugging */
void *flux_get_transformer(flux_ctx *ctx) {
    if (!ctx) return NULL;
    if (ctx->is_zimage) return ctx->zi_transformer;
    return ctx->transformer;
}

/* ========================================================================
 * Text Encoding
 * ======================================================================== */

/* Run the prompt through Qwen3 to produce text embeddings. For Flux models,
 * hidden states from layers 8, 17, 26 are concatenated to form [512, text_dim].
 * For Z-Image, takes hidden_states[-2] and reports the real (unpadded) token
 * count via out_seq_len. The returned embedding buffer is still max-seq padded;
 * Z-Image consumes only the first out_seq_len tokens. */
float *flux_encode_text(flux_ctx *ctx, const char *prompt, int *out_seq_len) {
    if (!ctx || !prompt) {
        *out_seq_len = 0;
        return NULL;
    }

    /* Check embedding cache first */
    float *cached = emb_cache_lookup(prompt);
    if (cached) {
        if (flux_phase_callback) flux_phase_callback("encoding text (cached)", 0);
        if (flux_phase_callback) flux_phase_callback("encoding text (cached)", 1);
        *out_seq_len = QWEN3_MAX_SEQ_LEN;
        return cached;
    }

    /* Load encoder if not already loaded */
    if (!ctx->qwen3_encoder && ctx->model_dir[0]) {
        if (flux_phase_callback) flux_phase_callback("Loading Qwen3 encoder", 0);
        ctx->qwen3_encoder = qwen3_encoder_load(ctx->model_dir, ctx->use_mmap);
        if (flux_phase_callback) flux_phase_callback("Loading Qwen3 encoder", 1);
        if (!ctx->qwen3_encoder) {
            fprintf(stderr, "Warning: Failed to load Qwen3 text encoder\n");
        }
    }

    if (!ctx->qwen3_encoder) {
        if (ctx->is_zimage) {
            /* Z-Image requires a real (unpadded) token sequence length. */
            *out_seq_len = 0;
            set_error("Qwen3 text encoder unavailable for Z-Image");
            return NULL;
        }
        /* Flux fallback: return zero padded embeddings. */
        *out_seq_len = QWEN3_MAX_SEQ_LEN;
        return (float *)calloc(QWEN3_MAX_SEQ_LEN * ctx->text_dim, sizeof(float));
    }

    /* Set extraction mode: Z-Image uses single layer, Flux uses 3-layer concat */
    qwen3_set_extraction_mode(ctx->qwen3_encoder, ctx->is_zimage ? 1 : 0);

    /* Encode text using Qwen3 */
    if (flux_phase_callback) flux_phase_callback("encoding text", 0);

    int num_real_tokens = 0;
    float *embeddings = qwen3_encode_text_ex(ctx->qwen3_encoder, prompt,
                                               &num_real_tokens);
    if (flux_phase_callback) flux_phase_callback("encoding text", 1);

    /* Store in cache for future lookups */
    if (embeddings) {
        emb_cache_store(prompt, embeddings,
                        QWEN3_MAX_SEQ_LEN * ctx->text_dim);
    }

    if (ctx->is_zimage) {
        /* Z-Image: use only real tokens from the padded embedding buffer. */
        *out_seq_len = num_real_tokens;
    } else {
        /* Flux: return full padded sequence (512) */
        *out_seq_len = QWEN3_MAX_SEQ_LEN;
    }
    return embeddings;
}

/* ========================================================================
 * Z-Image Generation
 * ======================================================================== */

/* Z-Image txt2img pipeline with pre-computed embeddings. Unlike Flux which
 * works in post-patchification space [128, H/16, W/16], Z-Image initializes
 * noise at pre-patchification dimensions [16, H/8, W/8] and the transformer
 * operates there. After denoising (8 NFE from 9 scheduler steps, where the
 * last step is a no-op because sigma_min=0), the output is patchified to
 * [64, H/16, W/16] for VAE decode. */
static flux_image *flux_generate_zimage_with_embeddings(flux_ctx *ctx,
                                                          const float *text_emb,
                                                          int text_seq,
                                                          const flux_params *p_in) {
    if (!ctx || !text_emb || text_seq <= 0) {
        set_error("Invalid context or embeddings");
        return NULL;
    }

    flux_params p;
    if (p_in) {
        p = *p_in;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Validate dimensions */
    if (p.width <= 0) p.width = 1024;   /* Z-Image default: 1024x1024 */
    if (p.height <= 0) p.height = 1024;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;

    /* Ensure dimensions are divisible by 16 */
    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        set_error("Image dimensions exceed maximum (1792x1792)");
        return NULL;
    }

    /* Release text encoder to free memory before loading transformer */
    flux_release_text_encoder(ctx);

    /* Load Z-Image transformer on-demand (persistent across generations). */
    if (!flux_load_zimage_transformer_if_needed(ctx)) {
        return NULL;
    }

    /* Z-Image latent dimensions:
     * The transformer works at pre-patchification: [in_ch, H/8, W/8]
     * where in_ch=16, and patchification happens inside the transformer.
     * VAE decode expects post-patchification: [latent_ch, H/16, W/16]
     * where latent_ch = in_ch * ps * ps = 64. */
    int ps = ctx->zi_patch_size;  /* 2 */
    int pre_h = p.height / 8;    /* H/8: pre-patchification spatial */
    int pre_w = p.width / 8;
    int in_ch = ctx->zi_in_channels;  /* 16 */
    int post_h = pre_h / ps;     /* H/16: post-patchification spatial */
    int post_w = pre_w / ps;
    int image_seq_len = post_h * post_w;

    /* Initialize noise at pre-patchification dimensions: [in_ch, H/8, W/8] */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, in_ch, pre_h, pre_w, seed);

    /* Get Z-Image schedule (default FlowMatch; linear/power if explicitly requested). */
    float *schedule = flux_selected_zimage_schedule(&p, image_seq_len);

    /* Sample using Z-Image Euler method.
     * The transformer takes [in_ch, pre_h, pre_w] and returns same shape. */
    float *denoised = flux_sample_euler_zimage(
        ctx->zi_transformer, z, 1, in_ch, pre_h, pre_w,
        ps,
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL
    );

    free(z);
    free(schedule);

    if (!denoised) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Patchify transformer output for VAE decode:
     * [1, in_ch, H/8, W/8] -> [1, latent_ch, H/16, W/16]
     * where latent_ch = in_ch * ps * ps = 64 */
    int latent_ch = in_ch * ps * ps;
    float *latent = (float *)malloc(latent_ch * post_h * post_w * sizeof(float));
    flux_patchify(latent, denoised, 1, in_ch, pre_h, pre_w, ps);
    free(denoised);

    /* Decode latent to image */
    flux_image *img = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        img = flux_vae_decode(ctx->vae, latent, 1, post_h, post_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    }

    free(latent);
    return img;
}

static flux_image *flux_generate_zimage(flux_ctx *ctx, const char *prompt,
                                          const flux_params *p_in) {
    /* Encode text (Z-Image mode: extraction mode 1, single layer) */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        set_error("Failed to encode prompt");
        return NULL;
    }

    flux_image *img = flux_generate_zimage_with_embeddings(ctx, text_emb, text_seq, p_in);
    free(text_emb);
    return img;
}

/* ========================================================================
 * Image Generation
 * ======================================================================== */

/* Main text-to-image entry point. Routes to Z-Image or Flux pipeline based
 * on model type. For Flux: encodes text via Qwen3, frees the encoder, loads
 * the transformer, initializes Gaussian noise in latent space, then runs
 * Euler ODE denoising (4 steps distilled / 50 steps base with CFG) followed
 * by VAE decode. For base models, an empty-prompt encoding is also produced
 * for Classifier-Free Guidance (two sequential transformer passes per step). */
flux_image *flux_generate(flux_ctx *ctx, const char *prompt,
                          const flux_params *params) {
    if (!ctx || !prompt) {
        set_error("Invalid context or prompt");
        return NULL;
    }

    /* Route to Z-Image pipeline if appropriate */
    if (ctx->is_zimage) {
        return flux_generate_zimage(ctx, prompt, params);
    }

    /* Use defaults if params is NULL */
    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Validate dimensions */
    if (p.width <= 0) p.width = FLUX_DEFAULT_WIDTH;
    if (p.height <= 0) p.height = FLUX_DEFAULT_HEIGHT;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;
    float guidance = (p.guidance > 0) ? p.guidance : ctx->default_guidance;

    /* Ensure dimensions are divisible by 16 */
    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        set_error("Image dimensions exceed maximum (1792x1792)");
        return NULL;
    }

    /* Encode text (and unconditioned text for CFG in base model) */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        set_error("Failed to encode prompt");
        return NULL;
    }

    float *text_emb_uncond = NULL;
    int text_seq_uncond = 0;
    if (!ctx->is_distilled) {
        const char *neg = p.negative_prompt ? p.negative_prompt : "";
        text_emb_uncond = flux_encode_text(ctx, neg, &text_seq_uncond);
        if (!text_emb_uncond) {
            free(text_emb);
            set_error("Failed to encode negative prompt for CFG");
            return NULL;
        }
    }

    /* Release text encoder to free ~8GB before loading transformer */
    flux_release_text_encoder(ctx);

    /* Load transformer now (after text encoder is freed to reduce peak memory) */
    if (!flux_load_transformer_if_needed(ctx)) {
        free(text_emb);
        free(text_emb_uncond);
        return NULL;
    }

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Initialize noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Get schedule */
    float *schedule = flux_selected_schedule(&p, image_seq_len);

    /* Sample */
    float *latent;
    if (ctx->is_distilled) {
        latent = flux_sample_euler(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
            text_emb, text_seq,
            schedule, p.num_steps,
            NULL
        );
    } else {
        latent = flux_sample_euler_cfg(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
            text_emb, text_seq,
            text_emb_uncond, text_seq_uncond,
            guidance,
            schedule, p.num_steps,
            NULL
        );
    }

    free(z);
    free(schedule);
    free(text_emb);
    free(text_emb_uncond);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode latent to image */
    flux_image *img = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    }

    free(latent);

    return img;
}

/* ========================================================================
 * Generation with Pre-computed Embeddings
 * ======================================================================== */

/* Generate from pre-computed text embeddings, skipping the Qwen3 encoding
 * step. Useful for the embedding cache (repeat prompts without re-encoding).
 * Only supports distilled models and Z-Image -- base model CFG requires two
 * separate embeddings (conditioned + empty prompt) which this API doesn't
 * provide, so it would produce incorrect results. */
flux_image *flux_generate_with_embeddings(flux_ctx *ctx,
                                           const float *text_emb, int text_seq,
                                           const flux_params *params) {
    if (!ctx || !text_emb) {
        set_error("Invalid context or embeddings");
        return NULL;
    }

    if (ctx->is_zimage) {
        return flux_generate_zimage_with_embeddings(ctx, text_emb, text_seq, params);
    }

    /* This API only supports the distilled (non-CFG) sampler since the
     * caller provides a single embedding.  Warn if used with a base model
     * because results will be incorrect without CFG. */
    if (!ctx->is_distilled) {
        fprintf(stderr, "Warning: flux_generate_with_embeddings() does not "
                        "support CFG. Use flux_generate() for base models.\n");
    }

    /* Load transformer if not already loaded */
    if (!flux_load_transformer_if_needed(ctx)) {
        return NULL;
    }

    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Validate dimensions */
    if (p.width <= 0) p.width = FLUX_DEFAULT_WIDTH;
    if (p.height <= 0) p.height = FLUX_DEFAULT_HEIGHT;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        set_error("Image dimensions exceed maximum (1792x1792)");
        return NULL;
    }

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Initialize noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Get schedule */
    float *schedule = flux_selected_schedule(&p, image_seq_len);

    /* Sample - note: pre-computed embeddings only support distilled path.
     * CFG requires two embeddings which the caller doesn't provide. */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL  /* progress_callback */
    );

    free(z);
    free(schedule);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode latent to image */
    flux_image *img = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    } else {
        set_error("No VAE loaded");
        free(latent);
        return NULL;
    }

    free(latent);
    return img;
}

/* Generate with external embeddings and external noise */
flux_image *flux_generate_with_embeddings_and_noise(flux_ctx *ctx,
                                                     const float *text_emb, int text_seq,
                                                     const float *noise, int noise_size,
                                                     const flux_params *params) {
    if (!ctx || !text_emb || !noise) {
        set_error("Invalid context, embeddings, or noise");
        return NULL;
    }

    if (ctx->is_zimage) {
        set_error("Z-Image does not support external embedding/noise generation API. "
                  "Use flux_generate() with a prompt.");
        return NULL;
    }

    /* Load transformer if not already loaded */
    if (!flux_load_transformer_if_needed(ctx)) {
        return NULL;
    }

    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Validate dimensions */
    if (p.width <= 0) p.width = FLUX_DEFAULT_WIDTH;
    if (p.height <= 0) p.height = FLUX_DEFAULT_HEIGHT;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        set_error("Image dimensions exceed maximum (1792x1792)");
        return NULL;
    }

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;
    int expected_noise_size = FLUX_LATENT_CHANNELS * latent_h * latent_w;

    if (noise_size != expected_noise_size) {
        char err[256];
        snprintf(err, sizeof(err), "Noise size mismatch: got %d, expected %d",
                 noise_size, expected_noise_size);
        set_error(err);
        return NULL;
    }

    /* Copy external noise */
    float *z = (float *)malloc(expected_noise_size * sizeof(float));
    memcpy(z, noise, expected_noise_size * sizeof(float));

    /* Get schedule */
    float *schedule = flux_selected_schedule(&p, image_seq_len);

    /* Sample */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL  /* progress_callback */
    );

    free(z);
    free(schedule);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode latent to image */
    flux_image *img = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    } else {
        set_error("No VAE loaded");
        free(latent);
        return NULL;
    }

    free(latent);
    return img;
}

/* ========================================================================
 * Attention Memory Budget
 * ======================================================================== */

/* 4 GB — MPSTemporaryNDArray hard limit. */
#define ATTENTION_MAX_BYTES ((size_t)4ULL << 30)

/* Compute worst-case attention matrix size in bytes.
 * All image dimensions are in pixels (multiples of 16).
 * ref_dims is [h0, w0, h1, w1, ...] in pixels. */
static size_t attention_bytes(int num_heads,
                              int out_h, int out_w,
                              const int *ref_dims, int num_refs,
                              int txt_seq)
{
    size_t total_seq = (size_t)(out_h / 16) * (out_w / 16);
    for (int i = 0; i < num_refs; i++)
        total_seq += (size_t)(ref_dims[i*2] / 16) * (ref_dims[i*2+1] / 16);
    total_seq += txt_seq;
    return (size_t)num_heads * total_seq * total_seq * sizeof(float);
}

/* Shrink reference pixel dimensions so attention fits under 4 GB.
 * ref_dims: [h0, w0, h1, w1, ...] in pixels, modified in-place.
 * Returns 1 if any reference was shrunk, 0 if already fits. */
static int fit_refs_for_attention(int num_heads,
                                  int out_h, int out_w,
                                  int *ref_dims, int num_refs,
                                  int txt_seq)
{
    if (attention_bytes(num_heads, out_h, out_w,
                        ref_dims, num_refs, txt_seq) <= ATTENTION_MAX_BYTES)
        return 0;

    int shrunk = 0;
    for (;;) {
        /* Find reference with the most latent tokens. */
        int best = -1;
        size_t best_tok = 0;
        for (int i = 0; i < num_refs; i++) {
            size_t tok = (size_t)(ref_dims[i*2] / 16) *
                         (ref_dims[i*2+1] / 16);
            if (tok > best_tok) { best_tok = tok; best = i; }
        }
        if (best < 0 || best_tok <= 1) break;  /* can't shrink further */

        /* Scale both dimensions by 0.9, round down to multiple of 16. */
        int h = (int)(ref_dims[best*2] * 0.9f) / 16 * 16;
        int w = (int)(ref_dims[best*2+1] * 0.9f) / 16 * 16;
        if (h < 16) h = 16;
        if (w < 16) w = 16;

        /* No progress — already at minimum. */
        if (h == ref_dims[best*2] && w == ref_dims[best*2+1]) break;

        ref_dims[best*2]   = h;
        ref_dims[best*2+1] = w;
        shrunk = 1;

        if (attention_bytes(num_heads, out_h, out_w,
                            ref_dims, num_refs, txt_seq) <= ATTENTION_MAX_BYTES)
            break;
    }
    return shrunk;
}

/* ========================================================================
 * Image-to-Image Generation
 * ======================================================================== */

/* Image-to-image generation via in-context conditioning. The reference image
 * is VAE-encoded into latent tokens with a RoPE T offset (T=10), while the
 * target starts from pure noise (T=0). Both are concatenated and fed to the
 * transformer, which attends to reference tokens via joint attention -- this
 * is fundamentally different from traditional img2img that adds noise to the
 * encoded image. References are dynamically resized if the resulting attention
 * matrix would exceed the 4GB MPS memory limit. */
flux_image *flux_img2img(flux_ctx *ctx, const char *prompt,
                         const flux_image *input, const flux_params *params) {
    if (!ctx || !prompt || !input) {
        set_error("Invalid parameters");
        return NULL;
    }
    if (ctx->is_zimage) {
        set_error("img2img is not supported for Z-Image");
        return NULL;
    }

    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Use input image dimensions if not specified */
    if (p.width <= 0) p.width = input->width;
    if (p.height <= 0) p.height = input->height;

    /* Clamp to VAE max dimensions, preserving aspect ratio */
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        float scale = (float)FLUX_VAE_MAX_DIM /
                      (p.width > p.height ? p.width : p.height);
        p.width = (int)(p.width * scale);
        p.height = (int)(p.height * scale);
    }

    /* Ensure divisible by 16 */
    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;

    /* Normalise strength: 0 means "not set", treat as 1.0 (in-context conditioning). */
    float strength = p.img2img_strength;
    if (strength <= 0.0f || strength > 1.0f) strength = 1.0f;

    /* For noise-injection (strength < 1.0), the reference must be encoded at
     * exactly the target output size so the latents can be blended.
     * For in-context conditioning (strength == 1.0), we may shrink the reference
     * to stay within the GPU attention memory budget. */
    int ref_w = p.width, ref_h = p.height;
    if (strength >= 1.0f) {
        int ref_dims[2] = { p.height, p.width };
        if (fit_refs_for_attention(ctx->num_heads, p.height, p.width,
                                    ref_dims, 1, FLUX_MAX_SEQ_LEN)) {
            fprintf(stderr, "Note: reference image resized from %dx%d to %dx%d "
                    "(GPU attention memory limit)\n",
                    p.width, p.height, ref_dims[1], ref_dims[0]);
            ref_h = ref_dims[0];
            ref_w = ref_dims[1];
        }
    }

    /* Resize input if needed */
    flux_image *resized = NULL;
    const flux_image *img_to_use = input;
    if (input->width != ref_w || input->height != ref_h) {
        resized = flux_image_resize(input, ref_w, ref_h);
        if (!resized) {
            set_error("Failed to resize input image");
            return NULL;
        }
        img_to_use = resized;
    }

    /* Resolve steps and guidance */
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;
    float guidance = (p.guidance > 0) ? p.guidance : ctx->default_guidance;

    /* Encode text */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        if (resized) flux_image_free(resized);
        set_error("Failed to encode prompt");
        return NULL;
    }

    float *text_emb_uncond = NULL;
    int text_seq_uncond = 0;
    if (!ctx->is_distilled) {
        const char *neg = p.negative_prompt ? p.negative_prompt : "";
        text_emb_uncond = flux_encode_text(ctx, neg, &text_seq_uncond);
        if (!text_emb_uncond) {
            free(text_emb);
            if (resized) flux_image_free(resized);
            set_error("Failed to encode negative prompt for CFG");
            return NULL;
        }
    }

    /* Release text encoder to free ~8GB before loading transformer */
    flux_release_text_encoder(ctx);

    /* Load transformer now (after text encoder is freed to reduce peak memory) */
    if (!flux_load_transformer_if_needed(ctx)) {
        free(text_emb);
        free(text_emb_uncond);
        if (resized) flux_image_free(resized);
        return NULL;
    }

    /* Encode image to latent */
    if (flux_phase_callback) flux_phase_callback("encoding reference image", 0);
    float *img_tensor = flux_image_to_tensor(img_to_use);
    if (resized) flux_image_free(resized);

    int latent_h, latent_w;
    float *img_latent = NULL;

    if (ctx->vae) {
        img_latent = flux_vae_encode(ctx->vae, img_tensor, 1,
                                     ref_h, ref_w, &latent_h, &latent_w);
    } else {
        /* Placeholder if no VAE */
        latent_h = ref_h / 16;
        latent_w = ref_w / 16;
        img_latent = (float *)calloc(FLUX_LATENT_CHANNELS * latent_h * latent_w, sizeof(float));
    }

    free(img_tensor);
    if (flux_phase_callback) flux_phase_callback("encoding reference image", 1);

    if (!img_latent) {
        free(text_emb);
        free(text_emb_uncond);
        set_error("Failed to encode image");
        return NULL;
    }

    int num_steps = p.num_steps;
    int out_lat_h = p.height / 16;
    int out_lat_w = p.width / 16;
    int image_seq_len = out_lat_h * out_lat_w;

    float *schedule = flux_selected_schedule(&p, image_seq_len);
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;

    float *latent;

    if (strength < 1.0f) {
        /*
         * Noise-injection img2img (strength < 1.0):
         * - Blend the reference latent with Gaussian noise at timestep t_start
         *   where t_start corresponds to (1 - strength) along the schedule
         * - Denoise only the remaining fraction of the schedule
         * - Lower strength  → start closer to the reference → subtler variation
         * - Higher strength → start further from the reference → freer variation
         */
        int start_step = (int)((1.0f - strength) * num_steps);
        if (start_step < 0) start_step = 0;
        if (start_step >= num_steps) start_step = num_steps - 1;
        float t_start = schedule[start_step];

        float *eps = flux_init_noise(1, FLUX_LATENT_CHANNELS, out_lat_h, out_lat_w, seed);
        int latent_size = FLUX_LATENT_CHANNELS * out_lat_h * out_lat_w;

        /* z = t_start * noise + (1 - t_start) * ref_latent */
        float *z_blend = (float *)malloc(latent_size * sizeof(float));
        for (int i = 0; i < latent_size; i++)
            z_blend[i] = t_start * eps[i] + (1.0f - t_start) * img_latent[i];

        free(eps);
        free(img_latent);

        int remaining = num_steps - start_step;
        if (ctx->is_distilled) {
            latent = flux_sample_euler(ctx->transformer, ctx->qwen3_encoder,
                                       z_blend, 1, FLUX_LATENT_CHANNELS, out_lat_h, out_lat_w,
                                       text_emb, text_seq,
                                       &schedule[start_step], remaining, NULL);
        } else {
            latent = flux_sample_euler_cfg(ctx->transformer, ctx->qwen3_encoder,
                                           z_blend, 1, FLUX_LATENT_CHANNELS, out_lat_h, out_lat_w,
                                           text_emb, text_seq,
                                           text_emb_uncond, text_seq_uncond,
                                           guidance,
                                           &schedule[start_step], remaining, NULL);
        }
        free(z_blend);
    } else {
        /*
         * In-context conditioning img2img (strength == 1.0, default):
         * - Reference is VAE-encoded and concatenated as extra tokens in the transformer
         * - Target latent starts from pure noise and denoises over all steps
         * - The model attends to the reference via joint attention (loose structural guidance)
         */
        float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, out_lat_h, out_lat_w, seed);
        int t_offset = 10;

        if (ctx->is_distilled) {
            latent = flux_sample_euler_with_refs(
                ctx->transformer, ctx->qwen3_encoder,
                z, 1, FLUX_LATENT_CHANNELS, out_lat_h, out_lat_w,
                img_latent, latent_h, latent_w,
                t_offset, text_emb, text_seq,
                schedule, num_steps, NULL);
        } else {
            latent = flux_sample_euler_cfg_with_refs(
                ctx->transformer, ctx->qwen3_encoder,
                z, 1, FLUX_LATENT_CHANNELS, out_lat_h, out_lat_w,
                img_latent, latent_h, latent_w,
                t_offset, text_emb, text_seq,
                text_emb_uncond, text_seq_uncond,
                guidance, schedule, num_steps, NULL);
        }
        free(z);
        free(img_latent);
    }

    free(schedule);
    free(text_emb);
    free(text_emb_uncond);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode */
    flux_image *result = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        result = flux_vae_decode(ctx->vae, latent, 1, out_lat_h, out_lat_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    }

    free(latent);
    return result;
}

/* ========================================================================
 * Multi-Reference Generation
 * ======================================================================== */

/* Multi-reference image generation dispatcher. Zero refs routes to txt2img,
 * one ref to the optimized single-reference img2img path. For multiple refs,
 * each reference is VAE-encoded with a distinct RoPE T offset (10, 20, 30...)
 * so the transformer can distinguish them spatially. All reference latents
 * participate in joint attention alongside the noised target tokens. */
flux_image *flux_multiref(flux_ctx *ctx, const char *prompt,
                          const flux_image **refs, int num_refs,
                          const flux_params *params) {
    if (!ctx || !prompt) {
        set_error("Invalid parameters");
        return NULL;
    }
    if (ctx->is_zimage) {
        set_error("multi-reference img2img is not supported for Z-Image");
        return NULL;
    }

    /* No references - text-to-image */
    if (!refs || num_refs == 0) {
        return flux_generate(ctx, prompt, params);
    }

    /* Single reference - use optimized path */
    if (num_refs == 1) {
        return flux_img2img(ctx, prompt, refs[0], params);
    }

    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Use first reference dimensions if not specified */
    if (p.width <= 0) p.width = refs[0]->width;
    if (p.height <= 0) p.height = refs[0]->height;

    /* Clamp to VAE max dimensions */
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        float scale = (float)FLUX_VAE_MAX_DIM /
                      (p.width > p.height ? p.width : p.height);
        p.width = (int)(p.width * scale);
        p.height = (int)(p.height * scale);
    }

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;

    /* Resolve steps and guidance */
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;
    float guidance = (p.guidance > 0) ? p.guidance : ctx->default_guidance;

    /* Encode text */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        set_error("Failed to encode prompt");
        return NULL;
    }

    float *text_emb_uncond = NULL;
    int text_seq_uncond = 0;
    if (!ctx->is_distilled) {
        const char *neg = p.negative_prompt ? p.negative_prompt : "";
        text_emb_uncond = flux_encode_text(ctx, neg, &text_seq_uncond);
        if (!text_emb_uncond) {
            free(text_emb);
            set_error("Failed to encode negative prompt for CFG");
            return NULL;
        }
    }

    flux_release_text_encoder(ctx);

    if (!flux_load_transformer_if_needed(ctx)) {
        free(text_emb);
        free(text_emb_uncond);
        return NULL;
    }

    /* Build reference pixel dimensions, clamped and rounded to 16. */
    int *ref_pixel_dims = (int *)malloc(num_refs * 2 * sizeof(int));
    for (int i = 0; i < num_refs; i++) {
        int rh = (refs[i]->height / 16) * 16;
        int rw = (refs[i]->width / 16) * 16;
        if (rh > FLUX_VAE_MAX_DIM) rh = FLUX_VAE_MAX_DIM;
        if (rw > FLUX_VAE_MAX_DIM) rw = FLUX_VAE_MAX_DIM;
        if (rh < 16) rh = 16;
        if (rw < 16) rw = 16;
        ref_pixel_dims[i*2]   = rh;
        ref_pixel_dims[i*2+1] = rw;
    }

    /* Shrink references if attention would exceed 4 GB. */
    if (fit_refs_for_attention(ctx->num_heads, p.height, p.width,
                                ref_pixel_dims, num_refs, FLUX_MAX_SEQ_LEN)) {
        fprintf(stderr,
                "Note: reference images resized to fit GPU attention "
                "memory limit\n");
    }

    /* Encode all reference images */
    flux_ref_t *ref_latents = (flux_ref_t *)malloc(num_refs * sizeof(flux_ref_t));
    float **ref_data = (float **)malloc(num_refs * sizeof(float *));
    flux_image **resized_imgs = (flux_image **)calloc(num_refs, sizeof(flux_image *));

    for (int i = 0; i < num_refs; i++) {
        const flux_image *ref = refs[i];
        const flux_image *img_to_use = ref;

        int ref_h = ref_pixel_dims[i*2];
        int ref_w = ref_pixel_dims[i*2+1];

        /* Resize only if dimensions differ from original */
        if (ref->width != ref_w || ref->height != ref_h) {
            resized_imgs[i] = flux_image_resize(ref, ref_w, ref_h);
            if (!resized_imgs[i]) {
                for (int j = 0; j < i; j++) {
                    free(ref_data[j]);
                    if (resized_imgs[j]) flux_image_free(resized_imgs[j]);
                }
                free(ref_latents);
                free(ref_data);
                free(resized_imgs);
                free(ref_pixel_dims);
                free(text_emb);
                free(text_emb_uncond);
                set_error("Failed to resize reference image");
                return NULL;
            }
            img_to_use = resized_imgs[i];
        }

        /* Encode to latent at reference's own size */
        float *tensor = flux_image_to_tensor(img_to_use);
        int lat_h, lat_w;
        ref_data[i] = flux_vae_encode(ctx->vae, tensor, 1,
                                       img_to_use->height, img_to_use->width,
                                       &lat_h, &lat_w);
        free(tensor);

        if (!ref_data[i]) {
            for (int j = 0; j < i; j++) {
                free(ref_data[j]);
                if (resized_imgs[j]) flux_image_free(resized_imgs[j]);
            }
            if (resized_imgs[i]) flux_image_free(resized_imgs[i]);
            free(ref_latents);
            free(ref_data);
            free(resized_imgs);
            free(ref_pixel_dims);
            free(text_emb);
            free(text_emb_uncond);
            set_error("Failed to encode reference image");
            return NULL;
        }

        ref_latents[i].latent = ref_data[i];
        ref_latents[i].h = lat_h;
        ref_latents[i].w = lat_w;
        ref_latents[i].t_offset = 10 * (i + 1);  /* 10, 20, 30, ... */
    }

    /* Free resized images (latents are now encoded) */
    for (int i = 0; i < num_refs; i++) {
        if (resized_imgs[i]) flux_image_free(resized_imgs[i]);
    }
    free(resized_imgs);
    free(ref_pixel_dims);

    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    float *schedule = flux_selected_schedule(&p, image_seq_len);
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Sample with multi-reference conditioning */
    float *latent;
    if (ctx->is_distilled) {
        latent = flux_sample_euler_with_multi_refs(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
            ref_latents, num_refs,
            text_emb, text_seq,
            schedule, p.num_steps,
            NULL
        );
    } else {
        latent = flux_sample_euler_cfg_with_multi_refs(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
            ref_latents, num_refs,
            text_emb, text_seq,
            text_emb_uncond, text_seq_uncond,
            guidance,
            schedule, p.num_steps,
            NULL
        );
    }

    /* Cleanup */
    free(z);
    for (int i = 0; i < num_refs; i++) {
        free(ref_data[i]);
    }
    free(ref_data);
    free(ref_latents);
    free(schedule);
    free(text_emb);
    free(text_emb_uncond);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode */
    flux_image *result = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        result = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    }

    free(latent);
    return result;
}

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

void flux_set_seed(int64_t seed) {
    flux_rng_seed((uint64_t)seed);
}

const char *flux_model_info(flux_ctx *ctx) {
    static char info[256];
    const char *type;
    if (!ctx) {
        return "No model loaded";
    }
    if (ctx->is_zimage) type = "zimage";
    else type = ctx->is_distilled ? "distilled" : "base";
    snprintf(info, sizeof(info), "%s v%s (%s, %d steps, guidance %.1f)",
             ctx->model_name, ctx->model_version,
             type,
             ctx->default_steps, ctx->default_guidance);
    return info;
}

int flux_text_dim(flux_ctx *ctx) {
    return ctx ? ctx->text_dim : 7680;
}

int flux_is_non_commercial(flux_ctx *ctx) {
    return ctx ? ctx->is_non_commercial : 0;
}

/* ========================================================================
 * Low-level API
 * ======================================================================== */

/* Public API: VAE-encode an RGB image to latent space. Converts the image
 * to a float tensor, runs the VAE encoder, and returns the latent buffer
 * with dimensions in out_h/out_w (each 1/16 of the pixel dimensions). */
float *flux_encode_image(flux_ctx *ctx, const flux_image *img,
                         int *out_h, int *out_w) {
    if (!ctx || !img || !ctx->vae) {
        *out_h = *out_w = 0;
        return NULL;
    }

    float *tensor = flux_image_to_tensor(img);
    if (!tensor) return NULL;

    float *latent = flux_vae_encode(ctx->vae, tensor, 1,
                                    img->height, img->width, out_h, out_w);
    free(tensor);
    return latent;
}

/* Public API: VAE-decode a latent tensor back to an RGB image.
 * Latent dimensions are 1/16 of the output pixel dimensions. */
flux_image *flux_decode_latent(flux_ctx *ctx, const float *latent,
                               int latent_h, int latent_w) {
    if (!ctx || !latent || !ctx->vae) return NULL;
    if (flux_phase_callback) flux_phase_callback("decoding image", 0);
    flux_image *img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
    if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    return img;
}

float *flux_denoise_step(flux_ctx *ctx, const float *z, float t,
                         const float *text_emb, int text_len,
                         int latent_h, int latent_w) {
    if (!ctx || !z || !text_emb) return NULL;

    /* Load transformer if not already loaded */
    if (!flux_load_transformer_if_needed(ctx)) {
        return NULL;
    }

    return flux_transformer_forward(ctx->transformer,
                                    z, latent_h, latent_w,
                                    text_emb, text_len, t);
}

/* Debug function: img2img with external inputs from Python */
flux_image *flux_img2img_debug_py(flux_ctx *ctx, const flux_params *params) {
    if (!ctx) {
        set_error("Invalid context");
        return NULL;
    }

    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Load Python's noise */
    FILE *f_noise = fopen("/tmp/py_noise.bin", "rb");
    if (!f_noise) {
        set_error("Cannot open /tmp/py_noise.bin");
        return NULL;
    }
    fseek(f_noise, 0, SEEK_END);
    int noise_size = ftell(f_noise) / sizeof(float);
    fseek(f_noise, 0, SEEK_SET);
    float *noise = (float *)malloc(noise_size * sizeof(float));
    fread(noise, sizeof(float), noise_size, f_noise);
    fclose(f_noise);
    fprintf(stderr, "[DEBUG] Loaded noise: %d floats\n", noise_size);

    /* Load Python's ref_latent */
    FILE *f_ref = fopen("/tmp/py_ref_latent.bin", "rb");
    if (!f_ref) {
        free(noise);
        set_error("Cannot open /tmp/py_ref_latent.bin");
        return NULL;
    }
    fseek(f_ref, 0, SEEK_END);
    int ref_size = ftell(f_ref) / sizeof(float);
    fseek(f_ref, 0, SEEK_SET);
    float *ref_latent = (float *)malloc(ref_size * sizeof(float));
    fread(ref_latent, sizeof(float), ref_size, f_ref);
    fclose(f_ref);
    fprintf(stderr, "[DEBUG] Loaded ref_latent: %d floats\n", ref_size);

    /* Load Python's text_emb */
    FILE *f_txt = fopen("/tmp/py_text_emb.bin", "rb");
    if (!f_txt) {
        free(noise);
        free(ref_latent);
        set_error("Cannot open /tmp/py_text_emb.bin");
        return NULL;
    }
    fseek(f_txt, 0, SEEK_END);
    int txt_size = ftell(f_txt) / sizeof(float);
    fseek(f_txt, 0, SEEK_SET);
    float *text_emb = (float *)malloc(txt_size * sizeof(float));
    fread(text_emb, sizeof(float), txt_size, f_txt);
    fclose(f_txt);
    int text_seq = 512;
    fprintf(stderr, "[DEBUG] Loaded text_emb: %d floats (%d x %d)\n",
            txt_size, text_seq, txt_size / text_seq);

    /* Load transformer */
    if (!flux_load_transformer_if_needed(ctx)) {
        free(noise);
        free(ref_latent);
        free(text_emb);
        return NULL;
    }

    /* Dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Get schedule */
    float *schedule = flux_selected_schedule(&p, image_seq_len);

    /* Sample with refs */
    float *latent = flux_sample_euler_with_refs(
        ctx->transformer, NULL,
        noise, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        ref_latent, latent_h, latent_w,
        10,  /* t_offset */
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL  /* progress_callback */
    );

    free(noise);
    free(ref_latent);
    free(schedule);
    free(text_emb);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode */
    flux_image *result = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        result = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    }

    free(latent);
    return result;
}
