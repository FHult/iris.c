# iris.c — Code Backlog

Items that are ready to implement (no blocking dependencies) but deferred for a future
session. Each item has been analysed; effort estimates are based on existing code patterns.

---

## B-001: `--vary-from` / `--vary-strength` CLI wiring (A5)

**Effort:** ~1 hour
**Unblocks:** A5 (Vary Subtle / Vary Strong) — full feature complete after this change

`img2img_strength` already exists in `iris_params`. `iris_img2img()` already exists.
Only missing piece is CLI argument wiring onto the existing infrastructure.

**`iris.h`** — add to `iris_params`:
```c
const char *vary_from;    /* --vary-from: base image path */
float       vary_strength; /* --vary-strength: 0.0–1.0 (subtle=0.2, strong=0.6) */
```

**`main.c`** — add to long_options (next available int: 265, 266):
```c
{"vary-from",     required_argument, 0, 265},
{"vary-strength", required_argument, 0, 266},
```

Switch cases:
```c
case 265: vary_from_path = optarg; break;
case 266: params.img2img_strength = (float)atof(optarg); break;
```

After parsing, if `vary_from_path` is set, load it as `input_paths[0]` and route through
the existing `iris_img2img()` call. The denoising mechanism is identical to img2img —
only the CLI labels and preset strengths differ.

Usage text to add (under Reference images section):
```
Vary (img2img re-denoise):
      --vary-from PATH      Base image to vary from (Flux only)
      --vary-strength N     Re-denoise strength 0.0-1.0
                            Presets: subtle ≈ 0.2, strong ≈ 0.6
```

**Files:** `main.c`, `iris.h`

---

## B-002: Z-Image CFG infrastructure (foundation for A2 + A3)

**Effort:** ~1 day
**Unblocks:** A2 (Z-Image-Omni-Base) + A3 (negative prompt for Z-Image)

`iris_sample_euler_zimage` has no unconditional pass. Extend signature with two optional
params (NULL/0 = no CFG, preserves all existing call sites unchanged):

**`iris_sample.c`** — extend `iris_sample_euler_zimage`:
```c
float *iris_sample_euler_zimage(void *transformer,
                                float *z, int batch, int channels, int h, int w,
                                int patch_size,
                                const float *cap_feats,       int cap_seq,
                                const float *cap_feats_uncond, int cap_seq_uncond,
                                float guidance_scale,
                                const float *schedule, int num_steps,
                                void (*progress_callback)(int step, int total));
```

Inside the denoising step loop, after the conditioned transformer call:
```c
if (cap_feats_uncond && guidance_scale > 1.0f) {
    float *v_uncond = zimage_transformer_forward(tf, z_curr,
                          pre_h, pre_w, timestep,
                          cap_feats_uncond, cap_seq_uncond);
    if (!v_uncond) { free(z_curr); return NULL; }
    for (int i = 0; i < latent_size; i++)
        v_cond[i] = v_uncond[i] + guidance_scale * (v_cond[i] - v_uncond[i]);
    free(v_uncond);
}
```

**`iris.c`** — in `iris_generate_zimage_with_embeddings()`:
- Encode `p.negative_prompt` (or empty string) into `text_emb_uncond` when
  `guidance_scale > 1.0f`
- Pass through to `iris_sample_euler_zimage`

**`iris.h`** — update `iris_sample_euler_zimage` declaration to match new signature.

All existing callers in `iris.c` pass `NULL, 0, 0.0f` — no behaviour change.

**Files:** `iris_sample.c`, `iris.c`, `iris.h`

---

## B-003: Negative prompt for distilled Flux (completing A3)

**Effort:** ~2 hours
**Unblocks:** A3 complete for all model types

Currently `iris.c` routes distilled models to `iris_sample_euler` (no CFG) regardless
of `--negative`. Fix: if `negative_prompt` is non-empty AND `guidance > 0`, route
through the CFG sampler even on distilled models.

There are three generation dispatch sites in `iris.c` that have the same pattern.
Each currently reads:

```c
if (ctx->is_distilled) {
    latent = iris_sample_euler(...);
} else {
    /* encode uncond */
    latent = iris_sample_euler_cfg(..., guidance, ...);
}
```

Change each to:

```c
int use_cfg = !ctx->is_distilled ||
              (p.negative_prompt && p.negative_prompt[0] && p.guidance > 0);
if (use_cfg) {
    if (!text_emb_uncond) {
        const char *neg = p.negative_prompt ? p.negative_prompt : "";
        text_emb_uncond = iris_encode_text(ctx, neg, &text_seq_uncond);
    }
    latent = iris_sample_euler_cfg(..., guidance, ...);
} else {
    latent = iris_sample_euler(...);
}
```

The existing uncond encoding block (already present for base models) only needs the
`use_cfg` condition relaxed — the encode path is already correct.

Update usage text: remove "base models only; ignored for distilled" caveat from `--negative`.

**Files:** `iris.c`, `main.c` (usage string only)

---

## Notes

- Items are listed in recommended implementation order (B-001 fastest, B-002 most
  foundational, B-003 completes A3).
- B-002 must be done before Z-Image-Omni-Base (A2) can be wired in.
- B-003 is independent of B-002 (Flux only).
- None of these require new model weights or external dependencies.
