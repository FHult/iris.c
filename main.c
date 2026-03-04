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
#include "terminals.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

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

/* Extract string value from JSON (returns malloc'd string, caller must free) */
static char *json_get_string(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;

    p += strlen(pattern);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    if (*p != '"') return NULL;
    p++;

    const char *end = p;
    while (*end && *end != '"') {
        if (*end == '\\' && *(end+1)) end += 2;
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
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;

    p += strlen(pattern);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;

    /* Handle null */
    if (strncmp(p, "null", 4) == 0) return default_val;

    return atoi(p);
}

/* Extract int64 value from JSON */
static int64_t json_get_int64(const char *json, const char *key, int64_t default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;

    p += strlen(pattern);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;

    /* Handle null */
    if (strncmp(p, "null", 4) == 0) return default_val;

    return atoll(p);
}

/* Extract boolean value from JSON */
static int json_get_bool(const char *json, const char *key, int default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;

    p += strlen(pattern);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;

    /* Handle null */
    if (strncmp(p, "null", 4) == 0) return default_val;
    /* Handle true/false */
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    /* Handle numeric 0/1 */
    return atoi(p) != 0;
}

/* Extract float value from JSON */
static float json_get_float(const char *json, const char *key, float default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;

    p += strlen(pattern);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;

    /* Handle null */
    if (strncmp(p, "null", 4) == 0) return default_val;

    return (float)atof(p);
}

/* Extract array of strings from JSON (returns array of malloc'd strings, caller must free)
 * Returns number of strings found, or 0 if key not found or not an array.
 * The paths array must be pre-allocated with max_paths capacity. */
static int json_get_string_array(const char *json, const char *key, char **paths, int max_paths) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0;

    p += strlen(pattern);
    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    if (*p != '[') return 0;
    p++; /* skip '[' */

    int count = 0;
    while (*p && *p != ']' && count < max_paths) {
        /* Skip whitespace and commas */
        while (*p && (*p == ' ' || *p == ',' || *p == '\t' || *p == '\n')) p++;
        if (*p == ']' || !*p) break;

        if (*p != '"') {
            /* Skip non-string values */
            while (*p && *p != ',' && *p != ']') p++;
            continue;
        }
        p++; /* skip opening quote */

        const char *end = p;
        while (*end && *end != '"') {
            if (*end == '\\' && *(end+1)) end += 2;
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
        if (*p == '"') p++; /* skip closing quote */
    }

    return count;
}

/* ========================================================================
 * Server Mode
 * ======================================================================== */

/* Track phase timing for server mode */
static struct timeval server_phase_start_tv;
static struct timeval server_step_start_tv;
static struct timeval server_generation_start_tv;

/* Current output path base for step images (set before each generation) */
static char server_step_image_base[512] = {0};

/* Server mode progress callback - output JSON status updates */
static void server_step_callback(int step, int total) {
    struct timeval now;
    gettimeofday(&now, NULL);

    /* Calculate step time (time since last step or phase start) */
    double step_time = (now.tv_sec - server_step_start_tv.tv_sec) +
                       (now.tv_usec - server_step_start_tv.tv_usec) / 1000000.0;

    /* Calculate elapsed time since generation started */
    double elapsed = (now.tv_sec - server_generation_start_tv.tv_sec) +
                     (now.tv_usec - server_generation_start_tv.tv_usec) / 1000000.0;

    printf("{\"event\":\"progress\",\"step\":%d,\"total\":%d,\"step_time\":%.2f,\"elapsed\":%.2f}\n",
           step, total, step_time, elapsed);
    fflush(stdout);

    /* Update step start time for next step */
    server_step_start_tv = now;
}

static void server_phase_callback(const char *phase, int done) {
    struct timeval now;
    gettimeofday(&now, NULL);

    if (!done) {
        /* Phase starting */
        server_phase_start_tv = now;
        server_step_start_tv = now;

        double elapsed = (now.tv_sec - server_generation_start_tv.tv_sec) +
                         (now.tv_usec - server_generation_start_tv.tv_usec) / 1000000.0;

        printf("{\"event\":\"phase\",\"phase\":\"%s\",\"elapsed\":%.2f}\n", phase, elapsed);
        fflush(stdout);
    } else {
        /* Phase finished */
        double phase_time = (now.tv_sec - server_phase_start_tv.tv_sec) +
                            (now.tv_usec - server_phase_start_tv.tv_usec) / 1000000.0;
        double elapsed = (now.tv_sec - server_generation_start_tv.tv_sec) +
                         (now.tv_usec - server_generation_start_tv.tv_usec) / 1000000.0;

        printf("{\"event\":\"phase_done\",\"phase\":\"%s\",\"phase_time\":%.2f,\"elapsed\":%.2f}\n",
               phase, phase_time, elapsed);
        fflush(stdout);
    }
}

/* Server mode step image callback - save intermediate image and emit JSON */
static void server_step_image_callback(int step, int total, const iris_image *img) {
    if (!server_step_image_base[0]) return;

    /* Build step image path: /path/to/output_step_N.png */
    char step_path[600];
    snprintf(step_path, sizeof(step_path), "%s_step_%d.png", server_step_image_base, step);

    /* Save the intermediate image */
    if (iris_image_save(img, step_path) == 0) {
        struct timeval now;
        gettimeofday(&now, NULL);
        double elapsed = (now.tv_sec - server_generation_start_tv.tv_sec) +
                         (now.tv_usec - server_generation_start_tv.tv_usec) / 1000000.0;

        printf("{\"event\":\"step_image\",\"step\":%d,\"total\":%d,\"path\":\"%s\",\"elapsed\":%.2f}\n",
               step, total, step_path, elapsed);
        fflush(stdout);
    }
}

/* Run server mode - keeps model loaded and processes JSON requests from stdin */
static int run_server_mode(iris_ctx *ctx) {
    char line[65536];

    /* Initialize embedding cache for repeat-prompt acceleration */
    emb_cache_init();

    /* Set up server-mode callbacks */
    iris_step_callback = server_step_callback;
    iris_phase_callback = server_phase_callback;
    iris_substep_callback = NULL;

    fprintf(stderr, "Server mode: ready for requests\n");
    printf("{\"event\":\"ready\",\"model\":\"%s\",\"is_distilled\":%s,\"is_zimage\":%s}\n",
           iris_model_info(ctx),
           iris_is_distilled(ctx) ? "true" : "false",
           iris_is_zimage(ctx) ? "true" : "false");
    fflush(stdout);

    while (fgets(line, sizeof(line), stdin) != NULL) {
        /* Skip empty lines */
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;

        /* Parse JSON request */
        char *prompt = json_get_string(line, "prompt");
        char *output_path = json_get_string(line, "output");
        char *input_path = json_get_string(line, "input_image");
        char *ref_paths[4] = {NULL, NULL, NULL, NULL};
        int num_refs = json_get_string_array(line, "reference_images", ref_paths, 4);
        int width = json_get_int(line, "width", DEFAULT_WIDTH);
        int height = json_get_int(line, "height", DEFAULT_HEIGHT);
        int steps = json_get_int(line, "steps", DEFAULT_STEPS);
        int64_t seed = json_get_int64(line, "seed", -1);
        int show_steps = json_get_bool(line, "show_steps", 1);
        float guidance = json_get_float(line, "guidance", 0.0f);
        float img2img_strength = json_get_float(line, "img2img_strength", 1.0f);
        char *negative_prompt = json_get_string(line, "negative_prompt");
        char *schedule = json_get_string(line, "schedule");
        char *req_lora_path = json_get_string(line, "lora");
        float req_lora_scale = json_get_float(line, "lora_scale", 1.0f);

        /* Validate request */
        if (!prompt || !output_path) {
            printf("{\"event\":\"error\",\"message\":\"Missing prompt or output\"}\n");
            fflush(stdout);
            free(prompt); free(output_path); free(input_path);
            free(negative_prompt); free(schedule); free(req_lora_path);
            for (int i = 0; i < num_refs; i++) free(ref_paths[i]);
            continue;
        }

        /* Validate parameters */
        if (width < 64 || width > 1792 || width % 16 != 0) {
            printf("{\"event\":\"error\",\"message\":\"Width must be 64-1792 and divisible by 16\"}\n");
            fflush(stdout);
            free(prompt); free(output_path); free(input_path);
            free(negative_prompt); free(schedule); free(req_lora_path);
            for (int i = 0; i < num_refs; i++) free(ref_paths[i]);
            continue;
        }
        if (height < 64 || height > 1792 || height % 16 != 0) {
            printf("{\"event\":\"error\",\"message\":\"Height must be 64-1792 and divisible by 16\"}\n");
            fflush(stdout);
            free(prompt); free(output_path); free(input_path);
            free(negative_prompt); free(schedule); free(req_lora_path);
            for (int i = 0; i < num_refs; i++) free(ref_paths[i]);
            continue;
        }

        /* Apply LoRA if requested (per-request hot-swap) */
        if (req_lora_path) {
            if (iris_load_lora(ctx, req_lora_path, req_lora_scale) != 0) {
                fprintf(stderr, "Warning: Failed to load LoRA from %s\n", req_lora_path);
            }
            free(req_lora_path);
        } else {
            iris_unload_lora(ctx);
        }

        /* Set seed */
        int64_t actual_seed = (seed >= 0) ? seed : (int64_t)time(NULL);
        iris_set_seed(actual_seed);

        /* Initialize generation timing */
        gettimeofday(&server_generation_start_tv, NULL);
        server_phase_start_tv = server_generation_start_tv;
        server_step_start_tv = server_generation_start_tv;

        /* Set up step image base path (output_path without .png extension) */
        strncpy(server_step_image_base, output_path, sizeof(server_step_image_base) - 1);
        server_step_image_base[sizeof(server_step_image_base) - 1] = '\0';
        size_t base_len = strlen(server_step_image_base);
        if (base_len > 4 && strcmp(server_step_image_base + base_len - 4, ".png") == 0) {
            server_step_image_base[base_len - 4] = '\0';
        }

        /* Enable/disable step image callback based on show_steps parameter */
        if (show_steps) {
            iris_set_step_image_callback(ctx, server_step_image_callback);
        } else {
            iris_set_step_image_callback(ctx, NULL);
        }

        /* Report seed */
        printf("{\"event\":\"status\",\"seed\":%lld}\n", (long long)actual_seed);
        fflush(stdout);

        /* Set up params */
        iris_params params = {
            .width = width,
            .height = height,
            .num_steps = steps,
            .seed = actual_seed,
            .guidance = guidance,
            .img2img_strength = img2img_strength,
            .negative_prompt = (negative_prompt && negative_prompt[0]) ? negative_prompt : NULL,
        };

        /* Apply schedule if specified */
        if (schedule) {
            if (strcmp(schedule, "linear") == 0) {
                params.linear_schedule = 1;
            } else if (strcmp(schedule, "power") == 0) {
                params.power_schedule = 1;
                params.power_alpha = json_get_float(line, "power_alpha", 2.0f);
            }
            free(schedule);
        }

        /* Generate */
        iris_image *output = NULL;
        if (num_refs > 0) {
            /* Multi-reference mode */
            iris_image *refs[4] = {NULL, NULL, NULL, NULL};
            int loaded_refs = 0;
            int load_error = 0;

            for (int i = 0; i < num_refs && !load_error; i++) {
                refs[i] = iris_image_load(ref_paths[i]);
                if (!refs[i]) {
                    printf("{\"event\":\"error\",\"message\":\"Failed to load reference image %d\"}\n", i + 1);
                    fflush(stdout);
                    load_error = 1;
                } else {
                    loaded_refs++;
                }
            }

            if (!load_error) {
                output = iris_multiref(ctx, prompt, (const iris_image **)refs, loaded_refs, &params);
            }

            /* Free loaded reference images */
            for (int i = 0; i < loaded_refs; i++) {
                iris_image_free(refs[i]);
            }

            if (load_error) {
                free(prompt); free(output_path); free(input_path);
                free(negative_prompt);
                for (int i = 0; i < num_refs; i++) free(ref_paths[i]);
                continue;
            }
        } else if (input_path && strlen(input_path) > 0) {
            /* Img2img mode (backwards compatibility) */
            iris_image *input = iris_image_load(input_path);
            if (!input) {
                printf("{\"event\":\"error\",\"message\":\"Failed to load input image\"}\n");
                fflush(stdout);
                free(prompt); free(output_path); free(input_path);
                free(negative_prompt);
                for (int i = 0; i < num_refs; i++) free(ref_paths[i]);
                continue;
            }
            output = iris_img2img(ctx, prompt, input, &params);
            iris_image_free(input);
        } else {
            /* Text-to-image mode */
            output = iris_generate(ctx, prompt, &params);
        }

        if (!output) {
            printf("{\"event\":\"error\",\"message\":\"Generation failed: %s\"}\n", iris_get_error());
            fflush(stdout);
            free(prompt); free(output_path); free(input_path);
            free(negative_prompt);
            for (int i = 0; i < num_refs; i++) free(ref_paths[i]);
            continue;
        }

        /* Save output */
        if (iris_image_save_with_seed(output, output_path, actual_seed) != 0) {
            printf("{\"event\":\"error\",\"message\":\"Failed to save image\"}\n");
            fflush(stdout);
            iris_image_free(output);
            free(prompt); free(output_path); free(input_path);
            free(negative_prompt);
            for (int i = 0; i < num_refs; i++) free(ref_paths[i]);
            continue;
        }

        iris_image_free(output);

        /* Calculate total generation time */
        struct timeval complete_tv;
        gettimeofday(&complete_tv, NULL);
        double total_time = (complete_tv.tv_sec - server_generation_start_tv.tv_sec) +
                            (complete_tv.tv_usec - server_generation_start_tv.tv_usec) / 1000000.0;

        /* Report success */
        printf("{\"event\":\"complete\",\"output\":\"%s\",\"seed\":%lld,\"total_time\":%.2f}\n",
               output_path, (long long)actual_seed, total_time);
        fflush(stdout);

        free(prompt); free(output_path); free(input_path);
        free(negative_prompt);
        for (int i = 0; i < num_refs; i++) free(ref_paths[i]);
    }

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
                fprintf(stderr, "FLUX.2 klein v1.0.0\n");
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
