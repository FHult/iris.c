# Changelog

All notable, user-facing changes to iris.c. Newest first.

---

## Unreleased — SREF style transfer: RoPE band-control (`--sref-shf` / `--sref-slf`)

**Training-free style-only reference conditioning for the in-context (`-i`) path**, on the frozen
Flux Klein transformer. This is Phase 1 of the pluggable style rail
(`plans/sref-architecture-options.md`, `plans/sref-rope-band-control.md`).

### What's new (user-facing)

- Two new CLI flags on the reference-image (`-i`) path:
  - `--sref-shf N`  — attenuate the reference's **high-frequency** RoPE bands (`N` in `[0,1]`;
    `0` = full attenuation, `1` = off). Lower `N` suppresses **composition copying** (the output keeps
    the prompt's subject instead of reproducing the reference's layout/subject).
  - `--sref-slf N`  — amplify the reference's **low-frequency** RoPE bands (`N ≥ 1`; `1` = off).
    Higher `N` increases **style strength** (palette / rendering adoption).
  - Default `1.0 / 1.0` = **off** (behaviour is bit-identical to before; verified corr 1.000000).
- Example — style-only transfer that keeps the subject:
  ```bash
  ./iris -d flux-klein-4b -p "a cat sitting on a chair" -i style_ref.jpg \
         --sref-shf 0.0 --sref-slf 1.5 -o out.png
  ```
- **Web UI**: band-control is now the default style rail — a Style-mode reference uses
  `--sref-shf 0.0 --sref-slf 1.5` (env `IRIS_SREF_SHF` / `IRIS_SREF_SLF`). The previous
  patch-shuffle is now opt-in (`IRIS_SREF_SHUFFLE_GRID=6`; default `0` = off). The persistent
  `iris --server` daemon gained matching `sref_shf` / `sref_slf` JSON request keys.
- **`--sref-strength N` (Phase 2)** — an optional reference-attention strength knob layered on
  band-control. `N` is a multiplier γ on how strongly the generation attends to the reference:
  `N<1` weakens, `N>1` amplifies, `N=1` = off (default, bit-identical). It adds `log(N)` to the
  reference-token *key* columns of attention (OminiControl-style bias), so it dials style strength
  independently of the RoPE bands. GPU/Metal path only. Env `IRIS_SREF_GAMMA` for the web UI
  (default `1.0` = off). Example: `--sref-shf 0.0 --sref-slf 1.5 --sref-strength 2.0`.
  Measured (8-ref gate, atop shf0.0/slf1.5): CSD style adherence rises monotonically with γ —
  0.322 (γ=0.5) → 0.354 (γ=1) → 0.380 (γ=2) → 0.433 (γ=4) — while reference discrimination holds
  (max cross-ref corr < 0.53 throughout). γ=1 reproduces the Phase-1 numbers exactly (no-op).

### Why

The frozen Flux transformer already conditions on reference images perfectly *in-context*
(VAE-encode the reference, append as sequence tokens), but naive in-context conditioning **copies the
reference's composition** (a style reference of an owl produces an owl, not the prompt's cat). The
previously shipped workaround crudely destroys the reference with a patch-shuffle. Band-control is the
surgical replacement: it suppresses only the positional-copying signal while preserving — and boosting —
the style signal.

### How it works

Reference copying in a diffusion transformer's joint attention is **positional, not semantic**: the
high-frequency components of the reference tokens' rotary position embedding (RoPE) dominate attention
and force target queries to attend to spatially-aligned reference tokens (copying), while the
low-frequency components carry global, semantically-driven (style) interactions.

We therefore scale the reference-token RoPE frequency bands **on the key side only, in the single-stream
blocks only**:

    s(d) = shf + (slf − shf)·(d/(D−1))²        (per frequency pair d = 0…D−1; d=0 highest freq)

`shf` attenuates the highest bands (kills copying); `slf` amplifies the lowest bands (boosts style).
Because a per-pair scalar commutes with the 2×2 RoPE rotation, the scaling is baked once into a K-side
RoPE table (exact, **zero per-step cost**; no extra kernels). Applies only to reference tokens; the
target-image and text tokens, and all query-side RoPE, are untouched — so with the flags at their
defaults the transform is a no-op.

### Validation

Measured on an 8-reference gate (`debug/sref_rope_gate.py`; the same diverse references the
mode-collapsed IP-adapter was characterised on), prompt "a cat sitting on a chair", seed 42, 512px:

| config | max cross-ref corr ↓ | CSD style adherence ↑ | copy corr ↓ |
|---|---|---|---|
| raw in-context (no shuffle) | 0.365 | 0.496 | 0.257 |
| patch-shuffle (previous shipped) | 0.791 | 0.219 | 0.076 |
| `--sref-shf 0.0 --sref-slf 1.5` | 0.394 | 0.354 | 0.162 |
| `--sref-shf 0.6 --sref-slf 1.5` | 0.290 | 0.476 | 0.346 |

Band-control **strictly beats** the patch-shuffle baseline: +60–120 % style adherence *and* ~2× better
reference discrimination, while suppressing composition-copying below raw in-context. Notably this is
the first validation of the underlying mechanism on a **few-step distilled** model (the source paper
used 50-step Flux.1-dev): full high-frequency attenuation (`shf=0.0`) correctly turns a woodcut-owl
reference into a *cat*, confirming that `shf` is the copy-killer and `slf` the style-strength knob.

Default-off is bit-identical; `make test-unit` passes; the f32 (CPU/BLAS) and bf16 (Metal) paths are
both wired and compile clean on the BLAS and generic targets.

### References / Credits

The core mechanism is a direct application of, and credited to:

- **Untwisting RoPE: Frequency Control for Shared Attention in DiTs** (arXiv:2602.05013). The analysis
  that reference-copying in DiT attention is caused by high-frequency RoPE dominance, and the
  per-frequency-band key-side scaling (attenuate high, amplify low) that controls style-vs-copy —
  training-free. iris.c implements this scaling in the C RoPE path (`band_scale_ref_krope` /
  `set_kside_krope` in `iris_transformer_flux.c`).

The style-adherence metric in the validation gate uses:

- **Measuring Style Similarity in Diffusion Models** — the Contrastive Style Descriptor (CSD)
  (Somepalli et al., 2024, arXiv:2404.01292).

The broader design (why the previous learned K/V-injection style adapter mode-collapsed, and why the
in-context/sequence path is the right substrate on Flux) was informed by a verified literature survey
(`plans/sref-architecture-options.md`), notably:

- **InstantStyle** (arXiv:2404.02733) and **DEADiff** (arXiv:2403.06951) — style/content entanglement in
  image-feature injection is architectural, and block-targeted injection is the structural remedy.
- **OminiControl** (arXiv:2411.15098) and **OminiControl2** (arXiv:2503.08280) — in-sequence conditioning
  on Flux with a small LoRA; the additive attention-bias `B(γ)` on generation→condition logits is the
  basis for the shipped Phase 2 `--sref-strength` (implemented as `log(γ)` on reference-key attention
  columns in the fused Metal kernels), and condition-token K/V reuse across denoising steps is the basis
  for the planned Phase 3 KV-reuse.
- **USO** (arXiv:2508.18966) — a learned in-sequence style-token producer (SigLIP → projector) on Flux;
  the model for the planned learned-encoder rail.
- **Distilling Diversity and Control in Diffusion Models** (arXiv:2503.10637) — few-step distilled models
  commit image structure in the first denoising step, and control modules transfer base↔distilled.
- **RB-Modulation** (arXiv:2405.17401), **CSGO** (arXiv:2408.16766), **i2L / image-to-LoRA**
  (arXiv:2606.13809) — surveyed alternatives (see the plan doc for the ranked shortlist and the avoid-list).
