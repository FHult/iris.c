# Changelog

All notable, user-facing changes to iris.c. Newest first.

---

## v5.3.0 — Learned generic style adapter (any reference → its style)

### What's new (user-facing) — web UI

- **New "Learned Style" reference mode.** Upload *any* reference image and its style transfers onto your
  prompt's subject — not just the handful of trained Style Library looks, and including painterly and
  graphic styles that training-free band-control can't do. It sits alongside the Style Library in the
  "Use as:" selector. The reference is CSD-encoded and a trained adapter applies its look; your prompt
  still drives the content. Works best on the base model (`--model-dir flux-klein-4b-base`).

### Under the hood — the adapter, reimplemented in C

- The learned joint-backbone style adapter — the first to break the reference-collapse that killed three
  prior attempts (it discriminates references at the pixel level: woodcut → ink, impressionism → shimmer)
  — now runs in the shipped iris/Metal engine:
  - `iris_csdmod.c` — the CSDModulation MLP FiLM'd into the timestep embedding, parity-guarded bit-exact
    vs the Python trainer (`debug/test_csdmod.c`: corr 1.0, max_abs 3e-8 under production flags).
  - **Style-CFG** — the style is injected only in the *conditional* CFG forward, so classifier-free
    guidance amplifies it instead of cancelling it. Default strength 0.4.
  - LoRA loads via the existing `--lora`; new `--sref-csdmod` / `--sref-csd` / `--sref-scale` flags plus
    the resident-daemon path. `train/lora/export_joint_to_c.py` and `dump_csd.py` produce the artifacts.
- Non-sref generation is unchanged — the adapter is fully gated off by default (`make test` green).

---

## v5.2.0 — Style-transfer UX + SREF research foundations

### What's new (user-facing) — web UI

- **References default to Style mode.** Uploading a reference now conditions on its *style* by default
  (previously composition), and a single global **Composition / Style** toggle replaces the confusing
  per-slot dropdowns. Directly fixes the "too many options / no style transfer from the defaults" report.
- **Expectation-setting hint** under the toggle: training-free band-control style transfer works best
  with **bold, graphic** references (line-art, flat illustration, high-contrast); painterly and
  photographic references transfer only weakly (see `SREF-STYLE-CEILING`).

No C/inference changes — the `iris` binary is unchanged from v5.1.0.

### Under the hood — SREF research (no shipped generation change)

Groundwork toward *strong* style transfer for the painterly / semi-real references band-control cannot do:

- **A real style-transfer metric** — `debug/sref_scorecard.py` + a frozen eval set that separates style
  adherence, composition leak, and prompt adherence **per reference type**. The gate every prior effort
  lacked (a single mean had hidden the painterly failure).
- **Learned in-sequence style encoder — investigated and ruled out on this stack.** A USO-style
  SigLIP→style-token projector (with and without a jointly-trained DiT LoRA) does not bind style through
  the frozen Flux backbone — three decisive negatives, root cause and probes documented in BACKLOG
  `SREF-LEARNED-STAGE1`.
- **Retrieval-hybrid instant-LoRA — the new direction, Phase 0 validated.** A per-style LoRA *does*
  impart painterly style (the first mechanism on this stack to do so); the remaining work is
  data/cluster curation. Scoped in `plans/sref-retrieval-hybrid-project.md`.

---

## v5.1.0 — Saved style codes on the band-control rail

### Saved style codes on the band-control rail (web UI)

Saved **style codes** (the Midjourney `--sref <code>` model — reuse a look without re-uploading the
image) now work on the default **band-control** rail, with **no trained adapter required**. Saving a
code stores the reference image (`web/output/sref/<sha>_ref.png`); generating with a `style_code`
resolves that image and replays it through the in-context style path (band-control), so a saved look
transfers its style while the prompt drives the subject. The opt-in trained IP-Adapter
(`IRIS_SREF_ADAPTER=1`) still routes codes through its precomputed features when enabled.
`POST /sref/codes` no longer requires `IRIS_IP_BUNDLE`, and the saved-styles gallery is always
available (`sref_enabled: true`).

---

## v5.0.0 — SREF style transfer: RoPE band-control (`--sref-shf` / `--sref-slf`)

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
  columns in the fused Metal kernels). Condition-token K/V reuse across denoising steps (OminiControl2)
  was the basis for a proposed Phase 3, but a viability probe found the prerequisite asymmetric mask both
  degrades style quality (−28% CSD adherence) and forces the slow custom kernel (~40× on this stack) on
  4-step distilled Flux, so Phase 3 was abandoned (see BACKLOG SREF-ROPE-PHASE3).
- **USO** (arXiv:2508.18966) — a learned in-sequence style-token producer (SigLIP → projector) on Flux;
  the model for the planned learned-encoder rail.
- **Distilling Diversity and Control in Diffusion Models** (arXiv:2503.10637) — few-step distilled models
  commit image structure in the first denoising step, and control modules transfer base↔distilled.
- **RB-Modulation** (arXiv:2405.17401), **CSGO** (arXiv:2408.16766), **i2L / image-to-LoRA**
  (arXiv:2606.13809) — surveyed alternatives (see the plan doc for the ranked shortlist and the avoid-list).
