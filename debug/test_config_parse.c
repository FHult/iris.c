/*
 * debug/test_config_parse.c — golden tests for iris_config_parse.h.
 *
 * iris.c autodetects model dimensions (heads, layers, channels, axes_dims, …) by
 * reading config.json / model_index.json with these helpers. A misparse silently
 * loads the wrong architecture, so pin: anchored key match (no substring
 * false-positives like "dim" inside "cap_feat_dim"), compact + pretty-printed
 * JSON, missing-key defaults, bool/array parsing.
 *
 * Build: cc -O2 -I. -o /tmp/iris_test_config debug/test_config_parse.c
 * Run:   /tmp/iris_test_config
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "iris_config_parse.h"

static int failures = 0, passes = 0;
static void ok(const char *name, int cond) {
    if (cond) { printf("PASS %s\n", name); passes++; }
    else { fprintf(stderr, "FAIL %s\n", name); failures++; }
}

int main(void) {
    /* Flux transformer-style config */
    const char *flux =
        "{\n  \"num_attention_heads\": 32,\n  \"joint_attention_dim\": 12288,\n"
        "  \"is_distilled\": true\n}";
    ok("flux heads", cfg_int(flux, "num_attention_heads", -1) == 32);
    ok("flux joint_dim", cfg_int(flux, "joint_attention_dim", -1) == 12288);
    ok("flux is_distilled true", cfg_bool(flux, "is_distilled", 0) == 1);

    /* Compact (no spaces) */
    const char *compact = "{\"num_attention_heads\":24,\"is_distilled\":false}";
    ok("compact heads", cfg_int(compact, "num_attention_heads", -1) == 24);
    ok("compact is_distilled false", cfg_bool(compact, "is_distilled", 1) == 0);

    /* Missing key -> default */
    ok("missing int default", cfg_int(flux, "n_layers", 30) == 30);
    ok("missing bool default", cfg_bool(flux, "absent", 1) == 1);
    ok("absent is_distilled -> base default 0", cfg_bool("{\"x\":1}", "is_distilled", 0) == 0);

    /* Anchored match: "dim" must NOT match inside "cap_feat_dim" / "hidden_dim". */
    const char *zi =
        "{\n  \"cap_feat_dim\": 2560,\n  \"hidden_dim\": 9999,\n  \"dim\": 3840,\n"
        "  \"n_layers\": 30,\n  \"n_refiner_layers\": 2,\n  \"in_channels\": 16,\n"
        "  \"patch_size\": 2,\n  \"rope_theta\": 256.0,\n  \"axes_dims\": [32, 48, 48]\n}";
    ok("anchored dim != cap_feat_dim", cfg_int(zi, "dim", -1) == 3840);
    ok("cap_feat_dim", cfg_int(zi, "cap_feat_dim", -1) == 2560);
    ok("n_layers != n_refiner_layers", cfg_int(zi, "n_layers", -1) == 30);
    ok("n_refiner_layers", cfg_int(zi, "n_refiner_layers", -1) == 2);
    ok("in_channels", cfg_int(zi, "in_channels", -1) == 16);
    ok("patch_size", cfg_int(zi, "patch_size", -1) == 2);
    ok("rope_theta", fabsf(cfg_float(zi, "rope_theta", -1.0f) - 256.0f) < 1e-4f);

    /* axes_dims array */
    int axes[3] = {0, 0, 0};
    int cnt = cfg_int_array(zi, "axes_dims", axes, 3);
    ok("axes count", cnt == 3);
    ok("axes values", axes[0] == 32 && axes[1] == 48 && axes[2] == 48);

    /* array with fewer elements: only present entries overwritten */
    int two[3] = {7, 7, 7};
    cfg_int_array("{\"a\": [1, 2]}", "a", two, 3);
    ok("short array keeps default tail", two[0] == 1 && two[1] == 2 && two[2] == 7);

    /* cfg_contains (class-name tags) */
    ok("contains tag", cfg_contains("{\"_class_name\": \"ZImagePipeline\"}", "ZImagePipeline"));
    ok("not contains", !cfg_contains(flux, "ZImagePipeline"));

    /* scientific notation float */
    ok("sci float", fabsf(cfg_float("{\"s\": 3.6e-1}", "s", 0.0f) - 0.36f) < 1e-5f);

    printf("\n%d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}
