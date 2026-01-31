/*
 * Fast bf16 to f16 safetensors converter using Metal GPU
 *
 * Converts bf16 weights to f16 for faster MPS loading on Apple Silicon.
 * Uses GPU parallelism for fast conversion.
 *
 * Usage: convert_bf16_to_f16 <input.safetensors> <output.safetensors>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#ifdef USE_METAL
#include "flux_metal.h"
#endif

/* ========================================================================
 * Safetensors Parsing (minimal, just what we need)
 * ======================================================================== */

typedef struct {
    char name[256];
    char dtype[16];
    int64_t shape[8];
    int ndim;
    size_t data_offset;
    size_t data_size;
} tensor_info_t;

typedef struct {
    void *data;
    size_t file_size;
    size_t header_size;
    char *header_json;
    tensor_info_t *tensors;
    int num_tensors;
} safetensors_t;

static void skip_whitespace(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

static int parse_string(const char **p, char *out, size_t max_len) {
    skip_whitespace(p);
    if (**p != '"') return -1;
    (*p)++;
    size_t i = 0;
    while (**p && **p != '"' && i < max_len - 1) {
        if (**p == '\\') { (*p)++; }
        out[i++] = *(*p)++;
    }
    out[i] = '\0';
    if (**p == '"') (*p)++;
    return 0;
}

static int64_t parse_int(const char **p) {
    skip_whitespace(p);
    int64_t val = 0;
    int neg = (**p == '-');
    if (neg) (*p)++;
    while (**p >= '0' && **p <= '9') val = val * 10 + (*(*p)++ - '0');
    return neg ? -val : val;
}

static safetensors_t *safetensors_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return NULL; }

    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;

    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { perror("mmap"); return NULL; }

    uint64_t header_size;
    memcpy(&header_size, data, 8);

    safetensors_t *sf = calloc(1, sizeof(safetensors_t));
    sf->data = data;
    sf->file_size = file_size;
    sf->header_size = header_size;
    sf->header_json = malloc(header_size + 1);
    memcpy(sf->header_json, (char *)data + 8, header_size);
    sf->header_json[header_size] = '\0';

    /* Count tensors (rough estimate) */
    int max_tensors = 1024;
    sf->tensors = calloc(max_tensors, sizeof(tensor_info_t));

    /* Parse header */
    const char *p = sf->header_json;
    skip_whitespace(&p);
    if (*p != '{') { free(sf); return NULL; }
    p++;

    while (*p && *p != '}' && sf->num_tensors < max_tensors) {
        skip_whitespace(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        char name[256];
        if (parse_string(&p, name, sizeof(name)) != 0) break;

        skip_whitespace(&p);
        if (*p != ':') break;
        p++;

        if (strcmp(name, "__metadata__") == 0) {
            /* Skip metadata object */
            skip_whitespace(&p);
            if (*p == '{') {
                int depth = 1; p++;
                while (*p && depth > 0) {
                    if (*p == '{') depth++;
                    else if (*p == '}') depth--;
                    p++;
                }
            }
            continue;
        }

        tensor_info_t *t = &sf->tensors[sf->num_tensors];
        strncpy(t->name, name, sizeof(t->name) - 1);

        /* Parse tensor info */
        skip_whitespace(&p);
        if (*p != '{') break;
        p++;

        while (*p && *p != '}') {
            skip_whitespace(&p);
            if (*p == ',') { p++; continue; }

            char key[64];
            if (parse_string(&p, key, sizeof(key)) != 0) break;
            skip_whitespace(&p);
            if (*p != ':') break;
            p++;
            skip_whitespace(&p);

            if (strcmp(key, "dtype") == 0) {
                parse_string(&p, t->dtype, sizeof(t->dtype));
            } else if (strcmp(key, "shape") == 0) {
                if (*p != '[') break;
                p++;
                t->ndim = 0;
                while (*p && *p != ']' && t->ndim < 8) {
                    skip_whitespace(&p);
                    if (*p == ',') { p++; continue; }
                    t->shape[t->ndim++] = parse_int(&p);
                }
                if (*p == ']') p++;
            } else if (strcmp(key, "data_offsets") == 0) {
                if (*p != '[') break;
                p++;
                size_t start = (size_t)parse_int(&p);
                skip_whitespace(&p);
                if (*p == ',') p++;
                size_t end = (size_t)parse_int(&p);
                t->data_offset = start;
                t->data_size = end - start;
                skip_whitespace(&p);
                if (*p == ']') p++;
            }
        }
        if (*p == '}') p++;
        sf->num_tensors++;
    }

    return sf;
}

static void safetensors_close(safetensors_t *sf) {
    if (!sf) return;
    munmap(sf->data, sf->file_size);
    free(sf->header_json);
    free(sf->tensors);
    free(sf);
}

static const void *safetensors_tensor_data(safetensors_t *sf, tensor_info_t *t) {
    return (const char *)sf->data + 8 + sf->header_size + t->data_offset;
}

/* ========================================================================
 * BF16 to F16 Conversion
 * ======================================================================== */

/* CPU scalar conversion (fallback) */
static inline uint16_t bf16_to_f16_scalar(uint16_t bf16) {
    uint32_t sign = (bf16 >> 15) & 0x1;
    int32_t exp = (bf16 >> 7) & 0xFF;
    uint32_t mant = bf16 & 0x7F;

    if (exp == 0) return sign << 15;
    if (exp == 0xFF) return (sign << 15) | 0x7C00 | (mant ? 0x200 : 0);

    int32_t new_exp = exp - 127 + 15;
    if (new_exp <= 0) return sign << 15;
    if (new_exp >= 31) return (sign << 15) | 0x7C00;

    return (sign << 15) | (new_exp << 10) | (mant << 3);
}

/* CPU conversion with basic parallelism hint */
static void convert_bf16_to_f16_cpu(const uint16_t *in, uint16_t *out, size_t n) {
    #pragma omp parallel for if(n > 100000)
    for (size_t i = 0; i < n; i++) {
        out[i] = bf16_to_f16_scalar(in[i]);
    }
}

#ifdef USE_METAL
/* GPU conversion using Metal */
static void convert_bf16_to_f16_gpu(const uint16_t *in, uint16_t *out, size_t n) {
    /* Upload to GPU, convert, download */
    flux_metal_bf16_to_f16_bulk(in, out, (int)n);
}
#endif

/* ========================================================================
 * Safetensors Writing
 * ======================================================================== */

static int write_safetensors(const char *path, safetensors_t *sf,
                             uint16_t **converted_data, int *is_converted) {
    /* Build new header JSON */
    size_t json_capacity = sf->header_size * 2;
    char *json = malloc(json_capacity);
    size_t json_len = 0;

    json[json_len++] = '{';

    size_t current_offset = 0;

    for (int i = 0; i < sf->num_tensors; i++) {
        tensor_info_t *t = &sf->tensors[i];

        if (i > 0) json[json_len++] = ',';

        /* Name */
        json_len += snprintf(json + json_len, json_capacity - json_len,
                             "\"%s\":{", t->name);

        /* Dtype - change BF16 to F16 */
        const char *dtype = is_converted[i] ? "F16" : t->dtype;
        json_len += snprintf(json + json_len, json_capacity - json_len,
                             "\"dtype\":\"%s\",", dtype);

        /* Shape */
        json_len += snprintf(json + json_len, json_capacity - json_len,
                             "\"shape\":[");
        for (int d = 0; d < t->ndim; d++) {
            if (d > 0) json[json_len++] = ',';
            json_len += snprintf(json + json_len, json_capacity - json_len,
                                 "%lld", (long long)t->shape[d]);
        }
        json[json_len++] = ']';
        json[json_len++] = ',';

        /* Data offsets */
        size_t data_size = t->data_size;
        json_len += snprintf(json + json_len, json_capacity - json_len,
                             "\"data_offsets\":[%zu,%zu]}",
                             current_offset, current_offset + data_size);

        current_offset += data_size;
    }

    json[json_len++] = '}';
    json[json_len] = '\0';

    /* Write file */
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen"); free(json); return -1; }

    /* Header size (8 bytes) */
    uint64_t header_size = json_len;
    fwrite(&header_size, 8, 1, f);

    /* Header JSON */
    fwrite(json, 1, json_len, f);

    /* Tensor data */
    for (int i = 0; i < sf->num_tensors; i++) {
        tensor_info_t *t = &sf->tensors[i];
        if (is_converted[i]) {
            fwrite(converted_data[i], 1, t->data_size, f);
        } else {
            fwrite(safetensors_tensor_data(sf, t), 1, t->data_size, f);
        }
    }

    fclose(f);
    free(json);
    return 0;
}

/* ========================================================================
 * Main
 * ======================================================================== */

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <input.safetensors> <output.safetensors>\n", prog);
    fprintf(stderr, "\nConverts bf16 tensors to f16 for faster MPS loading.\n");
}

int main(int argc, char **argv) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];

    printf("Opening: %s\n", input_path);
    safetensors_t *sf = safetensors_open(input_path);
    if (!sf) {
        fprintf(stderr, "Failed to open input file\n");
        return 1;
    }

    printf("Found %d tensors\n", sf->num_tensors);

#ifdef USE_METAL
    printf("Initializing Metal GPU...\n");
    if (flux_metal_init() != 0) {
        fprintf(stderr, "Warning: Metal init failed, using CPU\n");
    }
    flux_metal_init_shaders();
#endif

    /* Allocate conversion buffers */
    uint16_t **converted_data = calloc(sf->num_tensors, sizeof(uint16_t *));
    int *is_converted = calloc(sf->num_tensors, sizeof(int));

    int converted_count = 0;
    size_t total_converted_bytes = 0;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < sf->num_tensors; i++) {
        tensor_info_t *t = &sf->tensors[i];

        if (strcmp(t->dtype, "BF16") != 0) {
            continue;  /* Keep non-bf16 as-is */
        }

        size_t num_elements = t->data_size / 2;
        const uint16_t *src = safetensors_tensor_data(sf, t);

        converted_data[i] = malloc(t->data_size);
        is_converted[i] = 1;

        printf("  [%d/%d] Converting %s (%zu elements)...",
               i + 1, sf->num_tensors, t->name, num_elements);
        fflush(stdout);

#ifdef USE_METAL
        convert_bf16_to_f16_gpu(src, converted_data[i], num_elements);
#else
        convert_bf16_to_f16_cpu(src, converted_data[i], num_elements);
#endif

        printf(" done\n");

        converted_count++;
        total_converted_bytes += t->data_size;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\nConverted %d tensors (%.1f MB) in %.2f seconds\n",
           converted_count, total_converted_bytes / (1024.0 * 1024.0), elapsed);

    if (converted_count > 0) {
        double throughput = (total_converted_bytes / (1024.0 * 1024.0)) / elapsed;
        printf("Throughput: %.1f MB/s\n", throughput);
    }

    printf("\nWriting: %s\n", output_path);
    if (write_safetensors(output_path, sf, converted_data, is_converted) != 0) {
        fprintf(stderr, "Failed to write output file\n");
        return 1;
    }

    /* Cleanup */
    for (int i = 0; i < sf->num_tensors; i++) {
        free(converted_data[i]);
    }
    free(converted_data);
    free(is_converted);
    safetensors_close(sf);

#ifdef USE_METAL
    flux_metal_cleanup();
#endif

    printf("Done!\n");
    return 0;
}
