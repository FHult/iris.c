/*
 * FLUX CLI Application
 *
 * Command-line interface for FLUX.2 klein 4B image generation.
 *
 * Usage:
 *   flux -d model/ -p "prompt" -o output.png [options]
 *
 * Options:
 *   -d, --dir PATH        Path to model directory (safetensors)
 *   -p, --prompt TEXT     Text prompt for generation
 *   -o, --output PATH     Output image path
 *   -W, --width N         Output width (default: 256)
 *   -H, --height N        Output height (default: 256)
 *   -s, --steps N         Number of sampling steps (default: 4)
 *   -S, --seed N          Random seed (-1 for random)
 *   -i, --input PATH      Input image for img2img
 *   -q, --quiet           No output, just generate
 *   -v, --verbose         Extra detailed output
 *   --server              Server mode: keep model loaded, read JSON from stdin
 *   -h, --help            Show help
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_cli.h"
#include "embcache.h"
#include "iris_qwen3.h"
#include "terminals.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#ifdef USE_METAL
#include "iris_metal.h"
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
/* OpenBLAS introspection functions */
extern int openblas_get_num_threads(void);
extern int openblas_get_num_procs(void);
extern char *openblas_get_corename(void);
extern char *openblas_get_config(void);
extern void openblas_set_num_threads(int num_threads);
#endif
#endif

/* ========================================================================
 * Verbosity Levels
 * ======================================================================== */

typedef enum {
    OUTPUT_QUIET = 0,    /* No output */
    OUTPUT_NORMAL = 1,   /* Progress and essential info */
    OUTPUT_VERBOSE = 2   /* Detailed debugging info */
} output_level_t;

static output_level_t output_level = OUTPUT_NORMAL;

/* ========================================================================
 * CLI Progress Callbacks
 * ======================================================================== */

static int cli_current_step = 0;
static int cli_legend_printed = 0;
static term_graphics_proto cli_graphics_proto = TERM_PROTO_NONE;

/* Called at the start of each sampling step */
static void cli_step_callback(int step, int total) {
    if (output_level == OUTPUT_QUIET) return;

    /* Print legend before first step */
    if (!cli_legend_printed) {
        fprintf(stderr, "Denoising (d=double block, s=single blocks, F=final):\n");
        cli_legend_printed = 1;
    }

    /* Print newline to end previous step's progress (if any) */
    if (cli_current_step > 0) {
        fprintf(stderr, "\n");
    }
    cli_current_step = step;
    fprintf(stderr, "  Step %d/%d ", step, total);
    fflush(stderr);
}

/* Called for each substep within transformer forward */
static void cli_substep_callback(iris_substep_type_t type, int index, int total) {
    if (output_level == OUTPUT_QUIET) return;
    (void)total;

    switch (type) {
        case IRIS_SUBSTEP_DOUBLE_BLOCK:
            fputc('d', stderr);
            break;
        case IRIS_SUBSTEP_SINGLE_BLOCK:
            /* Print 's' every 5 single blocks to avoid too much output */
            if ((index + 1) % 5 == 0) {
                fputc('s', stderr);
            }
            break;
        case IRIS_SUBSTEP_FINAL_LAYER:
            fputc('F', stderr);
            break;
    }
    fflush(stderr);
}

/* Track phase timing (wall-clock) */
static struct timeval cli_phase_start_tv;
static const char *cli_current_phase = NULL;

/* Called at phase boundaries (encoding text, decoding image, etc.) */
static void cli_phase_callback(const char *phase, int done) {
    if (output_level == OUTPUT_QUIET) return;

    if (!done) {
        /* If we were showing step progress, end that line first */
        if (cli_current_step > 0) {
            fprintf(stderr, "\n");
            cli_current_step = 0;
        }

        /* Phase starting */
        cli_current_phase = phase;
        gettimeofday(&cli_phase_start_tv, NULL);

        /* Capitalize first letter for display */
        char display[64];
        strncpy(display, phase, sizeof(display) - 1);
        display[sizeof(display) - 1] = '\0';
        if (display[0] >= 'a' && display[0] <= 'z') {
            display[0] -= 32;
        }

        fprintf(stderr, "%s...", display);
        fflush(stderr);
    } else {
        /* Phase finished */
        struct timeval now;
        gettimeofday(&now, NULL);
        double elapsed = (now.tv_sec - cli_phase_start_tv.tv_sec) +
                         (now.tv_usec - cli_phase_start_tv.tv_usec) / 1000000.0;
        fprintf(stderr, " done (%.1fs)\n", elapsed);
        cli_current_phase = NULL;
    }
}

/* Set up CLI progress callbacks */
/* Step image callback - display intermediate images in terminal */
static void cli_step_image_callback(int step, int total, const iris_image *img) {
    (void)total;
    fprintf(stderr, "\n[Step %d]\n", step);
    terminal_display_image(img, cli_graphics_proto);
}

static void cli_setup_progress(void) {
    cli_current_step = 0;
    cli_legend_printed = 0;
    cli_current_phase = NULL;
    iris_step_callback = cli_step_callback;
    iris_substep_callback = cli_substep_callback;
    iris_phase_callback = cli_phase_callback;
}

/* Clean up after generation (print final newline) */
static void cli_finish_progress(void) {
    if (cli_current_step > 0) {
        fprintf(stderr, "\n");
        cli_current_step = 0;
    }
    iris_step_callback = NULL;
    iris_substep_callback = NULL;
    iris_phase_callback = NULL;
}

/* ========================================================================
 * Timing Helper (wall-clock time)
 * ======================================================================== */

static struct timeval timer_start_tv;

static void timer_begin(void) {
    gettimeofday(&timer_start_tv, NULL);
}

static double timer_end(void) {
    struct timeval now;
    gettimeofday(&now, NULL);
    return (now.tv_sec - timer_start_tv.tv_sec) +
           (now.tv_usec - timer_start_tv.tv_usec) / 1000000.0;
}

/* ========================================================================
 * Output Helpers
 * ======================================================================== */

/* Print if not quiet */
#define LOG_NORMAL(...) do { if (output_level >= OUTPUT_NORMAL) fprintf(stderr, __VA_ARGS__); } while(0)

/* Print only in verbose mode */
#define LOG_VERBOSE(...) do { if (output_level >= OUTPUT_VERBOSE) fprintf(stderr, __VA_ARGS__); } while(0)

/* ========================================================================
 * Default Values
 * ======================================================================== */

#define DEFAULT_WIDTH 256
#define DEFAULT_HEIGHT 256
#define DEFAULT_STEPS 4
#define MAX_INPUT_IMAGES 16

/* ========================================================================
 * Simple JSON Helpers (for server mode)
 * ======================================================================== */

/*
 * Skip a JSON value starting at *p, advancing *p past it.
 * Handles strings, objects, arrays, and scalars (numbers/bool/null).
 * Used by json_find_key to skip values while scanning for a target key.
 */
static void skip_json_value(const char **p) {
    if (**p == '"') {
        (*p)++;
        while (**p && **p != '"') {
            if (**p == '\\' && *(*p + 1)) (*p) += 2;
            else (*p)++;
        }
        if (**p == '"') (*p)++;
    } else if (**p == '{' || **p == '[') {
        int depth = 1;
        (*p)++;
        while (**p && depth > 0) {
            if (**p == '"') {
                (*p)++;
                while (**p && **p != '"') {
                    if (**p == '\\' && *(*p + 1)) (*p) += 2;
                    else (*p)++;
                }
                if (**p == '"') (*p)++;
            } else if (**p == '{' || **p == '[') {
                depth++; (*p)++;
            } else if (**p == '}' || **p == ']') {
                depth--; (*p)++;
            } else {
                (*p)++;
            }
        }
    } else {
        /* Number, bool, null: advance to next delimiter */
        while (**p && **p != ',' && **p != '}' && **p != ']' && **p != '\n') (*p)++;
    }
}

/*
 * Find `key` as a top-level key in a JSON object and return a pointer to
 * the start of its value (past the colon and whitespace).
 *
 * Only matches true top-level keys; skips over string values and nested
 * objects/arrays structurally, preventing false matches caused by a key
 * name appearing inside a string value.
 *
 * Returns NULL if the key is not found.
 */
static const char *json_find_key(const char *json, const char *key) {
    if (!json || !key) return NULL;
    size_t klen = strlen(key);

    const char *p = json;
    while (*p && *p != '{') p++;
    if (!*p) return NULL;
    p++;

    while (*p) {
        /* Skip whitespace and commas between members */
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
        if (!*p || *p == '}') break;

        /* Expect a quoted key */
        if (*p != '"') return NULL;
        p++;
        const char *ks = p;
        while (*p && *p != '"') {
            if (*p == '\\' && *(p + 1)) p += 2;
            else p++;
        }
        int match = ((size_t)(p - ks) == klen && memcmp(ks, key, klen) == 0);
        if (*p == '"') p++;

        /* Skip colon */
        while (*p == ' ' || *p == '\t') p++;
        if (*p != ':') return NULL;
        p++;
        while (*p == ' ' || *p == '\t') p++;

        if (match) return p;

        /* Not our key: skip the value and continue */
        skip_json_value(&p);
    }
    return NULL;
}

/* Extract string value from JSON (returns malloc'd string, caller must free).
 * Only matches top-level keys — immune to injection via value content. */
static char *json_get_string(const char *json, const char *key) {
    const char *p = json_find_key(json, key);
    if (!p || *p != '"') return NULL;
    p++;

    const char *end = p;
    while (*end && *end != '"') {
        if (*end == '\\' && *(end + 1)) end += 2;
        else end++;
    }

    size_t len = end - p;
    char *result = malloc(len + 1);
    if (!result) return NULL;
    memcpy(result, p, len);
    result[len] = '\0';
    return result;
}

/* Extract integer value from JSON */
static int json_get_int(const char *json, const char *key, int default_val) {
    const char *p = json_find_key(json, key);
    if (!p) return default_val;
    if (strncmp(p, "null", 4) == 0) return default_val;
    return atoi(p);
}

/* Extract int64 value from JSON */
static int64_t json_get_int64(const char *json, const char *key, int64_t default_val) {
    const char *p = json_find_key(json, key);
    if (!p) return default_val;
    if (strncmp(p, "null", 4) == 0) return default_val;
    return atoll(p);
}

/* Extract boolean value from JSON */
static int json_get_bool(const char *json, const char *key, int default_val) {
    const char *p = json_find_key(json, key);
    if (!p) return default_val;
    if (strncmp(p, "null", 4) == 0) return default_val;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return atoi(p) != 0;
}

/* Extract float value from JSON */
static float json_get_float(const char *json, const char *key, float default_val) {
    const char *p = json_find_key(json, key);
    if (!p) return default_val;
    if (strncmp(p, "null", 4) == 0) return default_val;
    return (float)atof(p);
}

/* Extract array of strings from JSON (returns array of malloc'd strings, caller must free)
 * Returns number of strings found, or 0 if key not found or not an array.
 * The paths array must be pre-allocated with max_paths capacity. */
static int json_get_string_array(const char *json, const char *key, char **paths, int max_paths) {
    const char *p = json_find_key(json, key);
    if (!p || *p != '[') return 0;
    p++;

    int count = 0;
    while (*p && *p != ']' && count < max_paths) {
        while (*p && (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n')) p++;
        if (*p == ']' || !*p) break;

        if (*p != '"') {
            while (*p && *p != ',' && *p != ']') p++;
            continue;
        }
        p++;

        const char *end = p;
        while (*end && *end != '"') {
            if (*end == '\\' && *(end + 1)) end += 2;
            else end++;
        }

        size_t len = end - p;
        if (len > 0) {
            char *str = malloc(len + 1);
            if (str) {
                memcpy(str, p, len);
                str[len] = '\0';
                paths[count++] = str;
            }
        }

        p = end;
        if (*p == '"') p++;
    }

    return count;
}

/*
 * Write a JSON-safe escaped string value (with surrounding quotes) to stdout.
 * Escapes backslash, double-quote, and control characters per RFC 8259.
 * Use this instead of printf("%s") whenever a string comes from user input
 * or could contain arbitrary characters.
 */
static void json_print_escaped(const char *s) {
    putchar('"');
    if (s) {
        while (*s) {
            unsigned char c = (unsigned char)*s++;
            if      (c == '"')  { putchar('\\'); putchar('"'); }
            else if (c == '\\') { putchar('\\'); putchar('\\'); }
            else if (c == '\n') { putchar('\\'); putchar('n'); }
            else if (c == '\r') { putchar('\\'); putchar('r'); }
            else if (c == '\t') { putchar('\\'); putchar('t'); }
            else if (c < 0x20)  { printf("\\u%04x", c); }
            else                { putchar(c); }
        }
    }
    putchar('"');
}

/*
 * Return 1 if path contains no ".." components (directory traversal prevention).
 * Rejects empty paths and any path component equal to "..".
 */
static int path_is_safe(const char *path) {
    if (!path || !path[0]) return 0;
    const char *p = path;
    while (*p) {
        while (*p == '/') p++;
        if (!*p) break;
        const char *end = p;
        while (*end && *end != '/') end++;
        if ((size_t)(end - p) == 2 && p[0] == '.' && p[1] == '.')
            return 0;
        p = end;
    }
    return 1;
}

/*
 * Generate a cryptographically random seed.
 * Uses arc4random_buf on Apple platforms, /dev/urandom elsewhere.
 * Falls back to time(NULL) only if /dev/urandom is unavailable.
 */
static int64_t random_seed(void) {
#ifdef __APPLE__
    uint64_t val;
    arc4random_buf(&val, sizeof(val));
    return (int64_t)(val & (uint64_t)INT64_MAX);
#else
    int64_t val = 0;
    FILE *f = fopen("/dev/urandom", "rb");
    if (f) {
        (void)fread(&val, sizeof(val), 1, f);
        fclose(f);
        val &= INT64_MAX;
    } else {
        val = (int64_t)time(NULL);
    }
    return val;
#endif
}

/* ========================================================================
 * Server Mode — threaded request queue with cancel support (SC-1, SC-2)
 * ======================================================================== */

/* ---- Job descriptor ---- */
#define SERVER_QUEUE_MAX 8

typedef struct {
    char    job_id[32];
    char   *prompt;
    char   *output_path;
    char   *input_path;
    char   *ref_paths[4];
    int     num_refs;
    int     width, height, steps;
    int64_t seed;
    int     show_steps;
    float   guidance;
    float   img2img_strength;
    char   *negative_prompt;
    int     linear_schedule;
    int     power_schedule;
    float   power_alpha;
    char   *lora_path;
    float   lora_scale;
} server_job_t;

/* ---- Queue state (protected by queue_mutex) ---- */
static server_job_t *job_queue[SERVER_QUEUE_MAX];
static int  queue_head = 0, queue_tail = 0, queue_count = 0;
static int  server_shutdown = 0;
static char current_job_id[32] = {0}; /* job running in worker, "" if idle */
static pthread_mutex_t queue_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  queue_cond  = PTHREAD_COND_INITIALIZER;

/* stdout mutex — prevents interleaving between main and worker thread */
static pthread_mutex_t stdout_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Job ID counter (main thread only) */
static int next_job_id = 0;

/* ---- Per-worker timing/state (worker thread only) ---- */
static struct timeval worker_gen_start;
static struct timeval worker_phase_start;
static struct timeval worker_step_start;
static char worker_step_base[512] = {0};
static char worker_job_id[32]     = {0};

static void server_free_job(server_job_t *job) {
    if (!job) return;
    free(job->prompt);
    free(job->output_path);
    free(job->input_path);
    free(job->negative_prompt);
    free(job->lora_path);
    for (int i = 0; i < job->num_refs; i++) free(job->ref_paths[i]);
    free(job);
}

/* ---- Callbacks (called from worker thread only) ---- */

static void server_step_callback(int step, int total) {
    struct timeval now;
    gettimeofday(&now, NULL);
    double step_time = (now.tv_sec - worker_step_start.tv_sec) +
                       (now.tv_usec - worker_step_start.tv_usec) / 1000000.0;
    double elapsed   = (now.tv_sec - worker_gen_start.tv_sec) +
                       (now.tv_usec - worker_gen_start.tv_usec) / 1000000.0;
    pthread_mutex_lock(&stdout_mutex);
    printf("{\"event\":\"progress\",\"job_id\":\"%s\",\"step\":%d,\"total\":%d,"
           "\"step_time\":%.2f,\"elapsed\":%.2f}\n",
           worker_job_id, step, total, step_time, elapsed);
    fflush(stdout);
    pthread_mutex_unlock(&stdout_mutex);
    worker_step_start = now;
}

static void server_phase_callback(const char *phase, int done) {
    struct timeval now;
    gettimeofday(&now, NULL);
    if (!done) {
        worker_phase_start = now;
        worker_step_start  = now;
        double elapsed = (now.tv_sec - worker_gen_start.tv_sec) +
                         (now.tv_usec - worker_gen_start.tv_usec) / 1000000.0;
        pthread_mutex_lock(&stdout_mutex);
        printf("{\"event\":\"phase\",\"job_id\":\"%s\",\"phase\":", worker_job_id);
        json_print_escaped(phase);
        printf(",\"elapsed\":%.2f}\n", elapsed);
        fflush(stdout);
        pthread_mutex_unlock(&stdout_mutex);
    } else {
        double phase_time = (now.tv_sec - worker_phase_start.tv_sec) +
                            (now.tv_usec - worker_phase_start.tv_usec) / 1000000.0;
        double elapsed    = (now.tv_sec - worker_gen_start.tv_sec) +
                            (now.tv_usec - worker_gen_start.tv_usec) / 1000000.0;
        pthread_mutex_lock(&stdout_mutex);
        printf("{\"event\":\"phase_done\",\"job_id\":\"%s\",\"phase\":", worker_job_id);
        json_print_escaped(phase);
        printf(",\"phase_time\":%.2f,\"elapsed\":%.2f}\n", phase_time, elapsed);
        fflush(stdout);
        pthread_mutex_unlock(&stdout_mutex);
    }
}

static void server_step_image_callback(int step, int total, const iris_image *img) {
    if (!worker_step_base[0]) return;
    char step_path[600];
    snprintf(step_path, sizeof(step_path), "%s_step_%d.png", worker_step_base, step);
    if (iris_image_save(img, step_path) == 0) {
        struct timeval now;
        gettimeofday(&now, NULL);
        double elapsed = (now.tv_sec - worker_gen_start.tv_sec) +
                         (now.tv_usec - worker_gen_start.tv_usec) / 1000000.0;
        pthread_mutex_lock(&stdout_mutex);
        printf("{\"event\":\"step_image\",\"job_id\":\"%s\",\"step\":%d,\"total\":%d,\"path\":",
               worker_job_id, step, total);
        json_print_escaped(step_path);
        printf(",\"elapsed\":%.2f}\n", elapsed);
        fflush(stdout);
        pthread_mutex_unlock(&stdout_mutex);
    }
}

/* ---- Worker: execute one job ---- */
static void server_run_job(iris_ctx *ctx, server_job_t *job) {
    strncpy(worker_job_id, job->job_id, sizeof(worker_job_id) - 1);
    worker_job_id[sizeof(worker_job_id) - 1] = '\0';

    /* Apply LoRA */
    if (job->lora_path) {
        if (iris_load_lora(ctx, job->lora_path, job->lora_scale) != 0)
            fprintf(stderr, "Warning: Failed to load LoRA from %s\n", job->lora_path);
    } else {
        iris_unload_lora(ctx);
    }

    /* Set seed */
    int64_t actual_seed = (job->seed >= 0) ? job->seed : random_seed();
    iris_set_seed(actual_seed);

    /* Initialize timing */
    gettimeofday(&worker_gen_start, NULL);
    worker_phase_start = worker_gen_start;
    worker_step_start  = worker_gen_start;

    /* Step image base path */
    strncpy(worker_step_base, job->output_path, sizeof(worker_step_base) - 1);
    worker_step_base[sizeof(worker_step_base) - 1] = '\0';
    size_t base_len = strlen(worker_step_base);
    if (base_len > 4 && strcmp(worker_step_base + base_len - 4, ".png") == 0)
        worker_step_base[base_len - 4] = '\0';

    if (job->show_steps)
        iris_set_step_image_callback(ctx, server_step_image_callback);
    else
        iris_set_step_image_callback(ctx, NULL);

    pthread_mutex_lock(&stdout_mutex);
    printf("{\"event\":\"status\",\"job_id\":\"%s\",\"seed\":%lld}\n",
           job->job_id, (long long)actual_seed);
    fflush(stdout);
    pthread_mutex_unlock(&stdout_mutex);

    iris_params params = {
        .width            = job->width,
        .height           = job->height,
        .num_steps        = job->steps,
        .seed             = actual_seed,
        .guidance         = job->guidance,
        .img2img_strength = job->img2img_strength,
        .negative_prompt  = (job->negative_prompt && job->negative_prompt[0])
                                ? job->negative_prompt : NULL,
        .linear_schedule  = job->linear_schedule,
        .power_schedule   = job->power_schedule,
        .power_alpha      = job->power_alpha,
    };

    iris_image *output = NULL;
    if (job->num_refs > 0) {
        iris_image *refs[4] = {NULL, NULL, NULL, NULL};
        int loaded_refs = 0, load_error = 0;
        for (int i = 0; i < job->num_refs && !load_error; i++) {
            refs[i] = iris_image_load(job->ref_paths[i]);
            if (!refs[i]) {
                pthread_mutex_lock(&stdout_mutex);
                printf("{\"event\":\"error\",\"job_id\":\"%s\","
                       "\"message\":\"Failed to load reference image %d\"}\n",
                       job->job_id, i + 1);
                fflush(stdout);
                pthread_mutex_unlock(&stdout_mutex);
                load_error = 1;
            } else {
                loaded_refs++;
            }
        }
        if (!load_error)
            output = iris_multiref(ctx, job->prompt,
                                   (const iris_image **)refs, loaded_refs, &params);
        for (int i = 0; i < loaded_refs; i++) iris_image_free(refs[i]);
        if (load_error) return;
    } else if (job->input_path && job->input_path[0]) {
        iris_image *input = iris_image_load(job->input_path);
        if (!input) {
            pthread_mutex_lock(&stdout_mutex);
            printf("{\"event\":\"error\",\"job_id\":\"%s\","
                   "\"message\":\"Failed to load input image\"}\n", job->job_id);
            fflush(stdout);
            pthread_mutex_unlock(&stdout_mutex);
            return;
        }
        output = iris_img2img(ctx, job->prompt, input, &params);
        iris_image_free(input);
    } else if (iris_is_distilled(ctx)) {
        /* Distilled txt2img: use embedding cache for repeated prompts */
        int text_dim = iris_text_dim(ctx);
        int seq_len  = QWEN3_MAX_SEQ_LEN;
        int emb_elements = 0;
        float *embeddings = emb_cache_lookup_ex(job->prompt, &emb_elements);
        if (embeddings) {
            if (text_dim <= 0 || emb_elements <= 0 || (emb_elements % text_dim) != 0) {
                free(embeddings);
                embeddings = NULL;
            } else {
                seq_len = emb_elements / text_dim;
            }
        }
        if (embeddings) {
            output = iris_generate_with_embeddings(ctx, embeddings, seq_len, &params);
            free(embeddings);
        } else {
            embeddings = iris_encode_text(ctx, job->prompt, &seq_len);
            if (embeddings) {
                emb_cache_store(job->prompt, embeddings, seq_len * text_dim);
                iris_release_text_encoder(ctx);
                output = iris_generate_with_embeddings(ctx, embeddings, seq_len, &params);
                free(embeddings);
            }
        }
    } else {
        /* Base model txt2img: CFG requires two encodings, skip cache */
        output = iris_generate(ctx, job->prompt, &params);
    }

    /* Check for cancellation (iris_cancel_requested set while generation ran) */
    if (iris_cancel_requested) {
        iris_image_free(output);
        pthread_mutex_lock(&stdout_mutex);
        printf("{\"event\":\"cancelled\",\"job_id\":\"%s\"}\n", job->job_id);
        fflush(stdout);
        pthread_mutex_unlock(&stdout_mutex);
        return;
    }

    if (!output) {
        const char *err = iris_get_error();
        char errbuf[512];
        snprintf(errbuf, sizeof(errbuf), "Generation failed: %s", err ? err : "unknown error");
        pthread_mutex_lock(&stdout_mutex);
        printf("{\"event\":\"error\",\"job_id\":\"%s\",\"message\":", job->job_id);
        json_print_escaped(errbuf);
        printf("}\n");
        fflush(stdout);
        pthread_mutex_unlock(&stdout_mutex);
        return;
    }

    if (iris_image_save_with_seed(output, job->output_path, actual_seed) != 0) {
        pthread_mutex_lock(&stdout_mutex);
        printf("{\"event\":\"error\",\"job_id\":\"%s\","
               "\"message\":\"Failed to save image\"}\n", job->job_id);
        fflush(stdout);
        pthread_mutex_unlock(&stdout_mutex);
        iris_image_free(output);
        return;
    }
    iris_image_free(output);

    struct timeval complete_tv;
    gettimeofday(&complete_tv, NULL);
    double total_time = (complete_tv.tv_sec - worker_gen_start.tv_sec) +
                        (complete_tv.tv_usec - worker_gen_start.tv_usec) / 1000000.0;

    pthread_mutex_lock(&stdout_mutex);
    printf("{\"event\":\"complete\",\"job_id\":\"%s\",\"output\":", job->job_id);
    json_print_escaped(job->output_path);
    printf(",\"seed\":%lld,\"total_time\":%.2f}\n", (long long)actual_seed, total_time);
    fflush(stdout);
    pthread_mutex_unlock(&stdout_mutex);
}

/* ---- Worker thread ---- */
static void *server_worker(void *arg) {
    iris_ctx *ctx = (iris_ctx *)arg;
    for (;;) {
        pthread_mutex_lock(&queue_mutex);
        while (queue_count == 0 && !server_shutdown)
            pthread_cond_wait(&queue_cond, &queue_mutex);
        if (server_shutdown && queue_count == 0) {
            pthread_mutex_unlock(&queue_mutex);
            break;
        }
        server_job_t *job = job_queue[queue_head];
        job_queue[queue_head] = NULL;
        queue_head = (queue_head + 1) % SERVER_QUEUE_MAX;
        queue_count--;
        strncpy(current_job_id, job->job_id, sizeof(current_job_id) - 1);
        current_job_id[sizeof(current_job_id) - 1] = '\0';
        iris_cancel_requested = 0;
        pthread_mutex_unlock(&queue_mutex);

        server_run_job(ctx, job);
        server_free_job(job);

        pthread_mutex_lock(&queue_mutex);
        current_job_id[0] = '\0';
        pthread_mutex_unlock(&queue_mutex);
    }
    return NULL;
}

/* ---- Main server loop ---- */
static int run_server_mode(iris_ctx *ctx) {
    char line[65536];

    emb_cache_init();
    iris_step_callback    = server_step_callback;
    iris_phase_callback   = server_phase_callback;
    iris_substep_callback = NULL;

    pthread_mutex_lock(&stdout_mutex);
    printf("{\"event\":\"ready\",\"model\":"); json_print_escaped(iris_model_info(ctx));
    printf(",\"is_distilled\":%s,\"is_zimage\":%s}\n",
           iris_is_distilled(ctx) ? "true" : "false",
           iris_is_zimage(ctx) ? "true" : "false");
    fflush(stdout);
    pthread_mutex_unlock(&stdout_mutex);

    fprintf(stderr, "Server mode: ready for requests\n");

    pthread_t worker;
    pthread_create(&worker, NULL, server_worker, ctx);

    while (fgets(line, sizeof(line), stdin) != NULL) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        /* Handle cancel events */
        char *event_type = json_get_string(line, "event");
        if (event_type && strcmp(event_type, "cancel") == 0) {
            char *cid = json_get_string(line, "job_id");
            if (cid) {
                int cancelled = 0;
                pthread_mutex_lock(&queue_mutex);
                /* Search queue for this job */
                for (int i = 0; i < queue_count && !cancelled; i++) {
                    int idx = (queue_head + i) % SERVER_QUEUE_MAX;
                    if (job_queue[idx] && strcmp(job_queue[idx]->job_id, cid) == 0) {
                        server_free_job(job_queue[idx]);
                        /* Close gap in circular buffer */
                        for (int j = i; j < queue_count - 1; j++) {
                            int a = (queue_head + j)     % SERVER_QUEUE_MAX;
                            int b = (queue_head + j + 1) % SERVER_QUEUE_MAX;
                            job_queue[a] = job_queue[b];
                        }
                        job_queue[(queue_head + queue_count - 1) % SERVER_QUEUE_MAX] = NULL;
                        queue_count--;
                        cancelled = 1;
                    }
                }
                /* If not queued, check if it is the running job */
                if (!cancelled && current_job_id[0] && strcmp(current_job_id, cid) == 0) {
                    iris_cancel_requested = 1;
                    cancelled = 1;
                }
                pthread_mutex_unlock(&queue_mutex);

                pthread_mutex_lock(&stdout_mutex);
                if (cancelled)
                    printf("{\"event\":\"cancelled\",\"job_id\":\"%s\"}\n", cid);
                else
                    printf("{\"event\":\"error\",\"message\":\"Unknown job_id\"}\n");
                fflush(stdout);
                pthread_mutex_unlock(&stdout_mutex);
                free(cid);
            }
            free(event_type);
            continue;
        }
        free(event_type);

        /* Parse generate request */
        char *prompt         = json_get_string(line, "prompt");
        char *output_path    = json_get_string(line, "output");
        char *input_path     = json_get_string(line, "input_image");
        char *ref_paths[4]   = {NULL, NULL, NULL, NULL};
        int   num_refs       = json_get_string_array(line, "reference_images", ref_paths, 4);
        int   width          = json_get_int(line, "width",  DEFAULT_WIDTH);
        int   height         = json_get_int(line, "height", DEFAULT_HEIGHT);
        int   steps          = json_get_int(line, "steps",  DEFAULT_STEPS);
        int64_t seed         = json_get_int64(line, "seed", -1);
        int   show_steps     = json_get_bool(line, "show_steps", 1);
        float guidance       = json_get_float(line, "guidance", 0.0f);
        float img2img_str    = json_get_float(line, "img2img_strength", 1.0f);
        char *neg_prompt     = json_get_string(line, "negative_prompt");
        char *schedule       = json_get_string(line, "schedule");
        char *req_lora       = json_get_string(line, "lora");
        float req_lora_scale = json_get_float(line, "lora_scale", 1.0f);

/* Helper: emit an error JSON line (no job_id yet) and free parsed strings */
#define EARLY_ERROR(msg) do { \
    pthread_mutex_lock(&stdout_mutex); \
    printf("{\"event\":\"error\",\"message\":\"%s\"}\n", (msg)); \
    fflush(stdout); \
    pthread_mutex_unlock(&stdout_mutex); \
    free(prompt); free(output_path); free(input_path); \
    free(neg_prompt); free(schedule); free(req_lora); \
    for (int _i = 0; _i < num_refs; _i++) free(ref_paths[_i]); \
    continue; \
} while(0)

        if (!prompt || !output_path)
            EARLY_ERROR("Missing prompt or output");
        if (width < 64 || width > 1792 || width % 16 != 0)
            EARLY_ERROR("Width must be 64-1792 and divisible by 16");
        if (height < 64 || height > 1792 || height % 16 != 0)
            EARLY_ERROR("Height must be 64-1792 and divisible by 16");
        if (steps < 1 || steps > IRIS_MAX_STEPS) {
            pthread_mutex_lock(&stdout_mutex);
            printf("{\"event\":\"error\",\"message\":\"Steps must be 1-%d\"}\n", IRIS_MAX_STEPS);
            fflush(stdout);
            pthread_mutex_unlock(&stdout_mutex);
            free(prompt); free(output_path); free(input_path);
            free(neg_prompt); free(schedule); free(req_lora);
            for (int i = 0; i < num_refs; i++) free(ref_paths[i]);
            continue;
        }
        {
            int unsafe = !path_is_safe(output_path) ||
                         (input_path && !path_is_safe(input_path)) ||
                         (req_lora    && !path_is_safe(req_lora));
            for (int i = 0; i < num_refs && !unsafe; i++)
                if (!path_is_safe(ref_paths[i])) unsafe = 1;
            if (unsafe) EARLY_ERROR("Unsafe file path");
        }

#undef EARLY_ERROR

        /* Check queue capacity */
        pthread_mutex_lock(&queue_mutex);
        if (queue_count >= SERVER_QUEUE_MAX) {
            pthread_mutex_unlock(&queue_mutex);
            pthread_mutex_lock(&stdout_mutex);
            printf("{\"event\":\"error\",\"message\":\"Server queue full, try again later\"}\n");
            fflush(stdout);
            pthread_mutex_unlock(&stdout_mutex);
            free(prompt); free(output_path); free(input_path);
            free(neg_prompt); free(schedule); free(req_lora);
            for (int i = 0; i < num_refs; i++) free(ref_paths[i]);
            continue;
        }

        /* Build job — takes ownership of all malloc'd strings */
        server_job_t *job = (server_job_t *)calloc(1, sizeof(server_job_t));
        if (!job) {
            pthread_mutex_unlock(&queue_mutex);
            pthread_mutex_lock(&stdout_mutex);
            printf("{\"event\":\"error\",\"message\":\"Out of memory\"}\n");
            fflush(stdout);
            pthread_mutex_unlock(&stdout_mutex);
            free(prompt); free(output_path); free(input_path);
            free(neg_prompt); free(schedule); free(req_lora);
            for (int i = 0; i < num_refs; i++) free(ref_paths[i]);
            continue;
        }

        snprintf(job->job_id, sizeof(job->job_id), "job_%06d", ++next_job_id);
        job->prompt          = prompt;
        job->output_path     = output_path;
        job->input_path      = input_path;
        job->num_refs        = num_refs;
        for (int i = 0; i < num_refs; i++) job->ref_paths[i] = ref_paths[i];
        job->width           = width;
        job->height          = height;
        job->steps           = steps;
        job->seed            = seed;
        job->show_steps      = show_steps;
        job->guidance        = guidance;
        job->img2img_strength = img2img_str;
        job->negative_prompt = neg_prompt;
        job->lora_path       = req_lora;
        job->lora_scale      = req_lora_scale;
        job->power_alpha     = 2.0f; /* default; overridden below if power schedule */

        if (schedule) {
            if (strcmp(schedule, "linear") == 0)
                job->linear_schedule = 1;
            else if (strcmp(schedule, "power") == 0) {
                job->power_schedule = 1;
                job->power_alpha    = json_get_float(line, "power_alpha", 2.0f);
            }
            free(schedule);
        }

        /* Enqueue */
        job_queue[queue_tail] = job;
        queue_tail = (queue_tail + 1) % SERVER_QUEUE_MAX;
        int position = queue_count + 1; /* 1 = next behind current job */
        queue_count++;
        pthread_cond_signal(&queue_cond);
        pthread_mutex_unlock(&queue_mutex);

        pthread_mutex_lock(&stdout_mutex);
        printf("{\"event\":\"queued\",\"job_id\":\"%s\",\"position\":%d}\n",
               job->job_id, position);
        fflush(stdout);
        pthread_mutex_unlock(&stdout_mutex);
    }

    /* EOF: drain queue and shut down worker */
    pthread_mutex_lock(&queue_mutex);
    server_shutdown = 1;
    pthread_cond_signal(&queue_cond);
    pthread_mutex_unlock(&queue_mutex);
    pthread_join(worker, NULL);

    emb_cache_free();
    return 0;
}

/* ========================================================================
 * Usage and Help
 * ======================================================================== */

static void print_usage(const char *prog) {
    fprintf(stderr, "FLUX.2 klein - Pure C Image Generation\n\n");
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d, --dir PATH        Path to model directory\n");
    fprintf(stderr, "  -p, --prompt TEXT     Text prompt for generation\n");
    fprintf(stderr, "  -o, --output PATH     Output image path (.png, .ppm)\n\n");
    fprintf(stderr, "Generation options:\n");
    fprintf(stderr, "  -W, --width N         Output width (default: %d)\n", DEFAULT_WIDTH);
    fprintf(stderr, "  -H, --height N        Output height (default: %d)\n", DEFAULT_HEIGHT);
    fprintf(stderr, "  -s, --steps N         Sampling steps (default: auto, 4 distilled / 50 base / 9 zimage)\n");
    fprintf(stderr, "  -g, --guidance N      CFG guidance scale (default: auto, 1.0 distilled / 4.0 base / 0.0 zimage)\n");
    fprintf(stderr, "  -S, --seed N          Random seed (-1 for random)\n");
    fprintf(stderr, "      --linear          Use linear timestep schedule (default: shifted sigmoid)\n");
    fprintf(stderr, "      --power           Use power curve timestep schedule (default alpha: 2.0)\n");
    fprintf(stderr, "      --power-alpha N   Set power schedule exponent (default: 2.0)\n\n");
    fprintf(stderr, "Model options:\n");
    fprintf(stderr, "      --base            Force base model mode (undistilled, CFG enabled)\n\n");
    fprintf(stderr, "Reference images (img2img / multi-reference):\n");
    fprintf(stderr, "  -i, --input PATH      Reference image (can specify up to %d)\n", MAX_INPUT_IMAGES);
    fprintf(stderr, "                        Multiple -i flags combine images via in-context conditioning\n\n");
    fprintf(stderr, "Output options:\n");
    fprintf(stderr, "  -q, --quiet           Silent mode, no output\n");
    fprintf(stderr, "  -v, --verbose         Detailed output\n");
    fprintf(stderr, "      --show            Display image in terminal (auto-detects Kitty/Ghostty/iTerm2/WezTerm/Konsole)\n");
    fprintf(stderr, "      --show-steps      Display each denoising step (slower)\n");
    fprintf(stderr, "      --zoom N          Terminal image zoom factor (default: 2 for Retina)\n\n");
    fprintf(stderr, "LoRA options:\n");
    fprintf(stderr, "      --lora PATH       Load LoRA adapter (XLabs, Kohya, or Diffusers .safetensors)\n");
    fprintf(stderr, "      --lora-scale N    LoRA strength (default: 1.0, range: 0.0-2.0)\n");
    fprintf(stderr, "      --img2img-strength N  Noise injection strength for img2img (0.0-1.0, default: 1.0=in-context)\n");
    fprintf(stderr, "  -N, --negative TEXT   Negative prompt for CFG (base models only; ignored for distilled)\n\n");
    fprintf(stderr, "Other options:\n");
    fprintf(stderr, "  -e, --embeddings PATH Load pre-computed text embeddings\n");
    fprintf(stderr, "  -m, --mmap            Use memory-mapped weights (default, fastest on MPS)\n");
    fprintf(stderr, "      --no-mmap         Disable mmap, load all weights upfront\n");
    fprintf(stderr, "      --server          Server mode: keep model loaded, read JSON from stdin\n");
    fprintf(stderr, "      --no-license-info Suppress non-commercial license warning\n");
    fprintf(stderr, "      --blas-threads N  Set number of BLAS threads (OpenBLAS only)\n");
    fprintf(stderr, "  -h, --help            Show this help\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s -d model/ -p \"a cat on a rainbow\" -o cat.png\n", prog);
    fprintf(stderr, "  %s -d model/ -p \"oil painting\" -i photo.png -o art.png\n", prog);
    fprintf(stderr, "  %s -d model/ -p \"combine them\" -i car.png -i beach.png -o result.png\n", prog);
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(int argc, char *argv[]) {
#ifdef USE_METAL
    iris_metal_init();
#endif

    /* Command line options */
    static struct option long_options[] = {
        {"dir",        required_argument, 0, 'd'},
        {"prompt",     required_argument, 0, 'p'},
        {"output",     required_argument, 0, 'o'},
        {"width",      required_argument, 0, 'W'},
        {"height",     required_argument, 0, 'H'},
        {"steps",      required_argument, 0, 's'},
        {"guidance",   required_argument, 0, 'g'},
        {"seed",       required_argument, 0, 'S'},
        {"input",      required_argument, 0, 'i'},
        {"embeddings", required_argument, 0, 'e'},
        {"noise",      required_argument, 0, 'n'},
        {"quiet",      no_argument,       0, 'q'},
        {"verbose",    no_argument,       0, 'v'},
        {"help",       no_argument,       0, 'h'},
        {"version",    no_argument,       0, 'V'},
        {"mmap",       no_argument,       0, 'm'},
        {"no-mmap",    no_argument,       0, 'M'},
        {"show",       no_argument,       0, 'k'},
        {"show-steps", no_argument,       0, 'K'},
        {"zoom",       required_argument, 0, 'z'},
        {"base",       no_argument,       0, 'B'},
        {"linear",     no_argument,       0, 'L'},
        {"power",      no_argument,       0, 256},
        {"power-alpha",required_argument, 0, 257},
        {"debug-py",   no_argument,       0, 'D'},
        {"server",     no_argument,       0, 'R'},
        {"no-license-info", no_argument, 0, 258},
        {"blas-threads",required_argument, 0, 259},
        {"lora",             required_argument, 0, 260},
        {"lora-scale",       required_argument, 0, 261},
        {"img2img-strength", required_argument, 0, 262},
        {"negative",         required_argument, 0, 'N'},
        {0, 0, 0, 0}
    };

    /* Parse arguments */
    char *model_dir = NULL;
    char *prompt = NULL;
    char *output_path = NULL;
    char *input_paths[MAX_INPUT_IMAGES] = {NULL};
    int num_inputs = 0;
    char *embeddings_path = NULL;
    char *noise_path = NULL;

    iris_params params = {
        .width = DEFAULT_WIDTH,
        .height = DEFAULT_HEIGHT,
        .num_steps = 0,   /* 0 = auto from model type */
        .seed = -1,
        .guidance = 0.0f, /* 0 = auto from model type */
        .power_alpha = 2.0f
    };

    int width_set = 0, height_set = 0, steps_set = 0;
    int use_mmap = 1;  /* mmap is default (fastest on MPS) */
    int show_image = 0;
    int show_steps = 0;
    int debug_py = 0;
    int server_mode = 0;
    int force_base = 0;
    int no_license_info = 0;
    int blas_threads = 0; (void)blas_threads;
    char *lora_path = NULL;
    float lora_scale = 1.0f;
    term_graphics_proto graphics_proto = detect_terminal_graphics();

    int opt;
    while ((opt = getopt_long(argc, argv, "d:p:o:W:H:s:g:S:i:t:e:n:N:qvhVmMD",
                              long_options, NULL)) != -1) {
        switch (opt) {
            case 'd': model_dir = optarg; break;
            case 'p': prompt = optarg; break;
            case 'o': output_path = optarg; break;
            case 'W': params.width = atoi(optarg); width_set = 1; break;
            case 'H': params.height = atoi(optarg); height_set = 1; break;
            case 's': params.num_steps = atoi(optarg); steps_set = 1; break;
            case 'g': params.guidance = atof(optarg); break;
            case 'S': params.seed = atoll(optarg); break;
            case 'i':
                if (num_inputs < MAX_INPUT_IMAGES) {
                    input_paths[num_inputs++] = optarg;
                } else {
                    fprintf(stderr, "Warning: Maximum %d input images supported\n", MAX_INPUT_IMAGES);
                }
                break;
            case 'e': embeddings_path = optarg; break;
            case 'n': noise_path = optarg; break;
            case 'q': output_level = OUTPUT_QUIET; break;
            case 'v': output_level = OUTPUT_VERBOSE; iris_verbose = 1; break;
            case 'h': print_usage(argv[0]); return 0;
            case 'V':
                fprintf(stderr, "FLUX.2 klein v2.3.0\n");
                return 0;
            case 'm': use_mmap = 1; break;
            case 'M': use_mmap = 0; break;
            case 'k': show_image = 1; break;
            case 'K': show_steps = 1; break;
            case 'z': terminal_set_zoom(atoi(optarg)); break;
            case 'B': force_base = 1; break;
            case 'L': params.linear_schedule = 1; break;
            case 256: params.power_schedule = 1; break;
            case 257: params.power_alpha = atof(optarg); params.power_schedule = 1; break;
            case 258: no_license_info = 1; break;
            case 'D': debug_py = 1; break;
            case 'R': server_mode = 1; break;
            case 259: blas_threads = atoi(optarg); break;
            case 260: lora_path = optarg; break;
            case 261: lora_scale = (float)atof(optarg); break;
            case 262: params.img2img_strength = (float)atof(optarg); break;
            case 'N': params.negative_prompt = optarg; break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    /* BLAS: apply thread setting regardless of quiet mode */
#if defined(USE_BLAS) && !defined(USE_METAL) && !defined(__APPLE__)
    if (blas_threads > 0) openblas_set_num_threads(blas_threads);
#endif

    /* Backend banner (suppressed by --quiet) */
    if (output_level != OUTPUT_QUIET) {
#ifdef USE_METAL
        if (iris_metal_available()) {
            long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
            char cpu_brand[128] = "Apple Silicon";
            size_t len = sizeof(cpu_brand);
            sysctlbyname("machdep.cpu.brand_string", cpu_brand, &len, NULL, 0);
            fprintf(stderr, "MPS: Metal GPU | %s | %ld cores\n", cpu_brand, ncpu);
        }
#elif defined(USE_BLAS)
#ifdef __APPLE__
        {
            char cpu_brand[128] = "Apple Silicon";
            size_t len = sizeof(cpu_brand);
            sysctlbyname("machdep.cpu.brand_string", cpu_brand, &len, NULL, 0);
            long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
            fprintf(stderr, "BLAS: Accelerate | %s | %ld cores\n", cpu_brand, ncpu);
            if (blas_threads > 0)
                fprintf(stderr, "Warning: --blas-threads ignored (Accelerate manages threading automatically)\n");
        }
#else
        fprintf(stderr, "BLAS: OpenBLAS | %s | %d threads / %d procs\n",
                openblas_get_corename(),
                openblas_get_num_threads(),
                openblas_get_num_procs());
        fprintf(stderr, "      %s\n", openblas_get_config());
#endif
#else
        fprintf(stderr, "Generic: Pure C backend (no acceleration)\n");
#endif
    }

    /* Validate required arguments */
    if (!model_dir) {
        fprintf(stderr, "Error: Model directory (-d) is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Interactive mode: -d provided but no -p, -e, -o, --debug-py, or --server */
    int interactive_mode = (!prompt && !embeddings_path && !output_path && !debug_py && !server_mode);

    if (!interactive_mode && !server_mode) {
        if (!prompt && !embeddings_path && !debug_py) {
            fprintf(stderr, "Error: Prompt (-p) or embeddings file (-e) is required\n\n");
            print_usage(argv[0]);
            return 1;
        }
        if (!output_path) {
            fprintf(stderr, "Error: Output path (-o) is required\n\n");
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Validate parameters */
    if (params.width < 64 || params.width > 4096) {
        fprintf(stderr, "Error: Width must be between 64 and 4096\n");
        return 1;
    }
    if (params.height < 64 || params.height > 4096) {
        fprintf(stderr, "Error: Height must be between 64 and 4096\n");
        return 1;
    }
    if (steps_set && (params.num_steps < 1 || params.num_steps > IRIS_MAX_STEPS)) {
        fprintf(stderr, "Error: Steps must be between 1 and %d\n", IRIS_MAX_STEPS);
        return 1;
    }

    /* Set seed (not for server mode - each request has its own seed) */
    int64_t actual_seed = -1;
    if (!server_mode) {
        if (params.seed >= 0) {
            actual_seed = params.seed;
        } else {
            actual_seed = (int64_t)time(NULL);
        }
        iris_set_seed(actual_seed);
        LOG_NORMAL("Seed: %lld\n", (long long)actual_seed);

    }

    /* Verbose header */
    LOG_VERBOSE("FLUX.2 klein Image Generator\n");
    LOG_VERBOSE("================================\n");
    LOG_VERBOSE("Model: %s\n", model_dir);
    if (prompt) LOG_VERBOSE("Prompt: %s\n", prompt);
    LOG_VERBOSE("Output: %s\n", output_path);
    LOG_VERBOSE("Size: %dx%d\n", params.width, params.height);
    LOG_VERBOSE("Steps: %d\n", params.num_steps);
    if (num_inputs > 0) {
        LOG_VERBOSE("Input images: %d\n", num_inputs);
        for (int i = 0; i < num_inputs; i++) {
            LOG_VERBOSE("  [%d] %s\n", i + 1, input_paths[i]);
        }
    }

    /* Load model (VAE only at startup, other components loaded on-demand) */
    LOG_NORMAL("Loading VAE...");
    if (output_level >= OUTPUT_NORMAL) fflush(stderr);
    timer_begin();

    iris_ctx *ctx = iris_load_dir(model_dir);
    if (!ctx) {
        fprintf(stderr, "\nError: Failed to load model: %s\n", iris_get_error());
        return 1;
    }

    /* Enable mmap mode if requested (reduces memory, slower inference) */
    if (use_mmap) {
        iris_set_mmap(ctx, 1);
        LOG_VERBOSE("  Using mmap mode for text encoder (lower memory)\n");
    }

    /* Override model type if --base was specified */
    if (force_base) {
        iris_set_base_mode(ctx);
    }

    /* Load LoRA adapter if requested */
    if (lora_path) {
        LOG_NORMAL("Loading LoRA: %s (scale=%.2f)...", lora_path, lora_scale);
        if (output_level >= OUTPUT_NORMAL) fflush(stderr);
        if (iris_load_lora(ctx, lora_path, lora_scale) != 0) {
            fprintf(stderr, "\nWarning: Failed to load LoRA from %s\n", lora_path);
        } else {
            LOG_NORMAL(" done\n");
        }
    }

    /* Resolve auto-parameters now that we know the model type */
    if (!steps_set || params.num_steps <= 0) {
        if (iris_is_zimage(ctx))
            params.num_steps = 9;  /* Z-Image-Turbo: 9 scheduler steps (8 NFE) */
        else
            params.num_steps = iris_is_distilled(ctx) ? 4 : 50;
    }
    if (params.guidance <= 0) {
        if (iris_is_zimage(ctx))
            params.guidance = 0.0f;
        else
            params.guidance = iris_is_distilled(ctx) ? 1.0f : 4.0f;
    }

    double load_time = timer_end();
    LOG_NORMAL(" done (%.1fs)\n", load_time);
    LOG_NORMAL("Model: %s\n", iris_model_info(ctx));

    /* Non-commercial license warning for 9B model */
    if (iris_is_non_commercial(ctx) && !no_license_info
        && output_level != OUTPUT_QUIET) {
        fprintf(stderr,
            "\nNOTE: This model is released under a NON COMMERCIAL LICENSE.\n"
            "The output can only be used under the terms of the\n"
            "FLUX non-commercial license:\n"
            "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md\n"
            "(use --no-license-info to suppress this message)\n\n");
    }

    /* Interactive mode: start REPL */
    if (interactive_mode) {
        int rc = iris_cli_run(ctx, model_dir);
        iris_free(ctx);
        return rc;
    }

    /* Enter server mode if requested */
    if (server_mode) {
        iris_set_keep_models_loaded(ctx, 1);
        int result = run_server_mode(ctx);
        iris_free(ctx);
#ifdef USE_METAL
        iris_metal_cleanup();
#endif
        return result;
    }

    /* Set up progress callbacks (for normal and verbose modes) */
    if (output_level >= OUTPUT_NORMAL) {
        cli_setup_progress();
    }

    /* Set up step image callback if requested */
    if (show_steps) {
        if (graphics_proto == TERM_PROTO_NONE) {
            fprintf(stderr, "Warning: --show-steps requires a supported terminal (Kitty, Ghostty, iTerm2, WezTerm, or Konsole)\n");
        } else {
            cli_graphics_proto = graphics_proto;
            iris_set_step_image_callback(ctx, cli_step_image_callback);
        }
    }

    /* Generate image */
    iris_image *output = NULL;
    struct timeval total_start_tv;
    gettimeofday(&total_start_tv, NULL);

    if (debug_py) {
        /* ============== Debug mode: use Python inputs ============== */
        LOG_NORMAL("Debug mode: loading Python inputs from /tmp/py_*.bin\n");
        output = iris_img2img_debug_py(ctx, &params);
    } else if (num_inputs > 0) {
        /* ============== Image-to-image mode (single or multi-reference) ============== */
        LOG_NORMAL("Loading %d input image%s...", num_inputs, num_inputs > 1 ? "s" : "");
        if (output_level >= OUTPUT_NORMAL) fflush(stderr);
        timer_begin();

        iris_image *inputs[MAX_INPUT_IMAGES];
        for (int i = 0; i < num_inputs; i++) {
            inputs[i] = iris_image_load(input_paths[i]);
            if (!inputs[i]) {
                fprintf(stderr, "\nError: Failed to load input image: %s\n", input_paths[i]);
                for (int j = 0; j < i; j++) iris_image_free(inputs[j]);
                iris_free(ctx);
                return 1;
            }
        }

        LOG_NORMAL(" done (%.1fs)\n", timer_end());
        for (int i = 0; i < num_inputs; i++) {
            LOG_VERBOSE("  Input[%d]: %dx%d, %d channels\n",
                        i + 1, inputs[i]->width, inputs[i]->height, inputs[i]->channels);
        }

        /* Use first input image dimensions if not explicitly set */
        if (!width_set) params.width = inputs[0]->width;
        if (!height_set) params.height = inputs[0]->height;

        /* Generate with multi-reference */
        output = iris_multiref(ctx, prompt, (const iris_image **)inputs, num_inputs, &params);

        for (int i = 0; i < num_inputs; i++) {
            iris_image_free(inputs[i]);
        }

    } else if (embeddings_path) {
        /* ============== External embeddings mode ============== */
        LOG_NORMAL("Loading embeddings...");
        if (output_level >= OUTPUT_NORMAL) fflush(stderr);
        timer_begin();

        FILE *emb_file = fopen(embeddings_path, "rb");
        if (!emb_file) {
            fprintf(stderr, "\nError: Failed to open embeddings file: %s\n", embeddings_path);
            iris_free(ctx);
            return 1;
        }

        fseek(emb_file, 0, SEEK_END);
        long file_size = ftell(emb_file);
        fseek(emb_file, 0, SEEK_SET);

        int text_dim = iris_text_dim(ctx);
        int text_seq = file_size / (text_dim * sizeof(float));

        float *text_emb = (float *)malloc(file_size);
        if (fread(text_emb, 1, file_size, emb_file) != (size_t)file_size) {
            fprintf(stderr, "\nError: Failed to read embeddings file\n");
            free(text_emb);
            fclose(emb_file);
            iris_free(ctx);
            return 1;
        }
        fclose(emb_file);

        LOG_NORMAL(" done (%.1fs)\n", timer_end());
        LOG_VERBOSE("  Embeddings: %d tokens x %d dims (%.2f MB)\n",
                    text_seq, text_dim, file_size / (1024.0 * 1024.0));

        /* Load noise if provided */
        float *noise = NULL;
        int noise_size = 0;
        if (noise_path) {
            LOG_VERBOSE("Loading noise from %s...\n", noise_path);

            FILE *noise_file = fopen(noise_path, "rb");
            if (!noise_file) {
                fprintf(stderr, "Error: Failed to open noise file: %s\n", noise_path);
                free(text_emb);
                iris_free(ctx);
                return 1;
            }

            fseek(noise_file, 0, SEEK_END);
            long noise_file_size = ftell(noise_file);
            fseek(noise_file, 0, SEEK_SET);

            noise_size = noise_file_size / sizeof(float);
            noise = (float *)malloc(noise_file_size);
            if (fread(noise, 1, noise_file_size, noise_file) != (size_t)noise_file_size) {
                fprintf(stderr, "Error: Failed to read noise file\n");
                free(noise);
                free(text_emb);
                fclose(noise_file);
                iris_free(ctx);
                return 1;
            }
            fclose(noise_file);
            LOG_VERBOSE("  Noise: %d floats\n", noise_size);
        }

        /* Generate */
        if (noise) {
            output = iris_generate_with_embeddings_and_noise(ctx, text_emb, text_seq,
                                                              noise, noise_size, &params);
            free(noise);
        } else {
            output = iris_generate_with_embeddings(ctx, text_emb, text_seq, &params);
        }
        free(text_emb);

    } else {
        /* ============== Text-to-image mode ============== */
        /* Note: iris_generate handles text encoding internally.
         * We can't easily time it separately without modifying the library.
         * The progress callbacks will show denoising progress. */
        output = iris_generate(ctx, prompt, &params);
    }

    /* Finish progress display */
    cli_finish_progress();

    /* Clear step image callback if it was set */
    if (show_steps) {
        iris_set_step_image_callback(ctx, NULL);
    }

    if (!output) {
        fprintf(stderr, "Error: Generation failed: %s\n", iris_get_error());
        iris_free(ctx);
        return 1;
    }

    struct timeval total_end_tv;
    gettimeofday(&total_end_tv, NULL);
    double total_time = (total_end_tv.tv_sec - total_start_tv.tv_sec) +
                        (total_end_tv.tv_usec - total_start_tv.tv_usec) / 1000000.0;
    LOG_VERBOSE("Generated in %.1fs total\n", total_time);
    LOG_VERBOSE("  Output: %dx%d, %d channels\n",
                output->width, output->height, output->channels);

    /* Save output */
    LOG_NORMAL("Saving...");
    if (output_level >= OUTPUT_NORMAL) fflush(stderr);
    timer_begin();

    if (iris_image_save_with_seed(output, output_path, actual_seed) != 0) {
        fprintf(stderr, "\nError: Failed to save image: %s\n", output_path);
        iris_image_free(output);
        iris_free(ctx);
        return 1;
    }

    LOG_NORMAL(" %s %dx%d (%.1fs)\n", output_path, output->width, output->height, timer_end());

    /* Display image in terminal if requested */
    if (show_image) {
        terminal_display_png(output_path, graphics_proto);
    }

    /* Print total time (always, unless quiet) */
    struct timeval final_tv;
    gettimeofday(&final_tv, NULL);
    double total_time_final = (final_tv.tv_sec - total_start_tv.tv_sec) +
                              (final_tv.tv_usec - total_start_tv.tv_usec) / 1000000.0;
    LOG_NORMAL("Total generation time: %.1f seconds\n", load_time + total_time_final);

    /* Cleanup */
    iris_image_free(output);
    iris_free(ctx);

#ifdef USE_METAL
    iris_metal_cleanup();
#endif

    return 0;
}
