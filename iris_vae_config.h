#ifndef IRIS_VAE_CONFIG_H
#define IRIS_VAE_CONFIG_H

#include <string.h>
#include <stdlib.h>

/*
 * iris_parse_vae_config — extract latent_channels / scaling_factor / shift_factor
 * from a NUL-terminated vae/config.json buffer.
 *
 * Override-only: the caller pre-sets defaults (z_channels = IRIS_VAE_Z_CHANNELS,
 * scaling = 0, shift = 0); a field is applied only if its key is present (and, for
 * latent_channels, parses to a positive int). This is byte-for-byte the historical
 * inline parse from iris.c, extracted so the parser is unit-testable.
 *
 * Why this matters (grok C-2): the resolved {z_channels, scaling, shift} selects the
 * VAE normalization branch (explicit (x-shift)*scale vs batch-norm) and the latent
 * channel count fed to the transformer. A brittle parse that mis-reads a pretty-printed
 * or variant config silently poisons every precomputed latent. Keep this dependency-free
 * (string.h/stdlib.h only) so both iris.c and debug/test_vae.c can compile it.
 */
static inline void iris_parse_vae_config(const char *json,
        int *z_channels, float *scaling, float *shift) {
    const char *p;
    if ((p = strstr(json, "\"latent_channels\""))) {
        const char *colon = strchr(p, ':');
        if (colon) {
            int lc = atoi(colon + 1);
            if (lc > 0) *z_channels = lc;
        }
    }
    if ((p = strstr(json, "\"scaling_factor\""))) {
        const char *colon = strchr(p, ':');
        if (colon) *scaling = (float)atof(colon + 1);
    }
    if ((p = strstr(json, "\"shift_factor\""))) {
        const char *colon = strchr(p, ':');
        if (colon) *shift = (float)atof(colon + 1);
    }
}

#endif /* IRIS_VAE_CONFIG_H */
