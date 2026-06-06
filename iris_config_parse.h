#ifndef IRIS_CONFIG_PARSE_H
#define IRIS_CONFIG_PARSE_H

#include <string.h>
#include <stdlib.h>

/*
 * Minimal, robust readers for the small `config.json` / `model_index.json` files
 * iris loads at startup to autodetect model dimensions. These centralize and make
 * testable the ad-hoc `strstr(buf,"\"key\"") + strchr(':') + atoi` pattern that was
 * duplicated ~15 times across iris.c — a misparse silently selects wrong model
 * dimensions (heads, layers, channels, …). Not a full JSON parser; just enough for
 * flat numeric/bool/short-array config fields.
 *
 * Key matching is anchored on the quoted key ("dim" will not match inside
 * "cap_feat_dim") and the value is read from the first ':' after the key, so
 * pretty-printed and compact JSON both work. Absent key -> caller's default.
 */

/* Return a pointer just past the ':' that follows "key", or NULL if absent. */
static inline const char *cfg_find_value(const char *json, const char *key) {
    char q[128];
    size_t kl = strlen(key);
    if (kl + 3 >= sizeof(q)) return NULL;     /* "key"\0 must fit */
    q[0] = '"';
    memcpy(q + 1, key, kl);
    q[1 + kl] = '"';
    q[2 + kl] = '\0';
    const char *p = strstr(json, q);
    if (!p) return NULL;
    p = strchr(p + kl + 2, ':');              /* skip the quoted key, find its ':' */
    return p ? p + 1 : NULL;
}

static inline int cfg_int(const char *json, const char *key, int dflt) {
    const char *v = cfg_find_value(json, key);
    return v ? atoi(v) : dflt;
}

static inline float cfg_float(const char *json, const char *key, float dflt) {
    const char *v = cfg_find_value(json, key);
    return v ? (float)atof(v) : dflt;
}

/* "key": true / false -> 1 / 0; absent or unrecognized -> dflt. Whitespace after
 * the colon is tolerated. */
static inline int cfg_bool(const char *json, const char *key, int dflt) {
    const char *v = cfg_find_value(json, key);
    if (!v) return dflt;
    while (*v == ' ' || *v == '\t' || *v == '\n' || *v == '\r') v++;
    if (strncmp(v, "true", 4) == 0)  return 1;
    if (strncmp(v, "false", 5) == 0) return 0;
    return dflt;
}

/* Plain substring presence (for class-name / pipeline tags, not key:value). */
static inline int cfg_contains(const char *json, const char *needle) {
    return strstr(json, needle) != NULL;
}

/* Parse "key": [a, b, c, ...] into out[0..n). Returns the count parsed; entries
 * not present in the array keep their prior value (caller pre-sets defaults). */
static inline int cfg_int_array(const char *json, const char *key, int *out, int n) {
    char q[128];
    size_t kl = strlen(key);
    if (kl + 3 >= sizeof(q)) return 0;
    q[0] = '"';
    memcpy(q + 1, key, kl);
    q[1 + kl] = '"';
    q[2 + kl] = '\0';
    const char *p = strstr(json, q);
    if (!p) return 0;
    p = strchr(p + kl + 2, '[');
    if (!p) return 0;
    p++;
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
        if (*p == ']' || *p == '\0') break;
        out[i] = atoi(p);
        cnt++;
        const char *comma = strchr(p, ',');
        const char *close = strchr(p, ']');
        if (!comma || (close && close < comma)) break;
        p = comma + 1;
    }
    return cnt;
}

#endif /* IRIS_CONFIG_PARSE_H */
