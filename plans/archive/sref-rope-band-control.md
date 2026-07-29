# SREF Precision In-Context: RoPE Band-Control + Strength Bias + Reference-KV Reuse

Implementation plan for shortlist item #1 of `plans/sref-architecture-options.md` (see BACKLOG
SREF-ARCH-RESEARCH). Self-contained: written to be executed by a fresh session.

## Goal

Upgrade the shipped `--sref` style rail from "destroy reference content crudely (patch-shuffle)"
to "suppress positional copying surgically" — zero training, C-only:

1. **Phase 1 — RoPE frequency-band control**: per-band scaling of *reference-token keys* so
   attention to the reference is semantic (style) rather than spatially aligned (copying).
2. **Phase 2 — Strength bias**: principled `--sref-strength` via an additive attention bias on
   generation→reference logits.
3. **Phase 3 — Reference-KV reuse**: compute reference-token K/V once (step 1), reuse for steps
   2–4 (perf only; independent of 1–2).

Each phase ships (validated + committed) independently. Phase 1 is the value; do not start
Phase 3 before Phase 1 is validated.

## Evidence base (verified claims; details in plans/sref-architecture-options.md)

- "Untwisting RoPE: Frequency Control for Shared Attention in DiTs" (arXiv 2602.05013):
  reference copying in DiT attention sharing is POSITIONAL — high-frequency RoPE components
  dominate attention and force spatially-aligned copying; low-frequency components carry
  global/semantic (style) interaction. Fix: per-chunk scale
  `s_d = s_hf + (s_lf − s_hf) · d̃^β` (β=2, d̃∈[0,1] from highest→lowest frequency),
  `s_hf ∈ (0,1)` attenuates the highest bands, `s_lf > 1` amplifies the lowest; applied ONLY to
  reference-image KEYS, ONLY in the SINGLE-STREAM blocks; both scales linearly increased over
  timesteps. Training-free. Their user study: 2.40 vs StyleAligned 0.74 vs IP-Adapter 0.44.
- OminiControl (2411.15098): conditioning strength via attention bias `B(γ)` on the logits of
  generation-token queries → condition-token keys (γ=0 removes influence, γ>1 amplifies).
- OminiControl2 (2503.08280): condition-token K/V computed once and reused across denoising
  steps, enabled by masking condition→noisy attention (84.7% of their conditioning-overhead
  reduction).
- Our own record: in-context conditioning discriminates references perfectly (the substrate is
  proven); patch-shuffle is the current content-destruction hack (web/server.py
  `content_destroy_png`, grid 6).

**Open paper question (resolve during Phase 1, do not block on it):** the exact semantics of
"modulating" — magnitude-scaling the rotated key components (what this plan specifies: scale the
K contribution of band d by s_d) vs scaling the rotation *frequency*. The verification pass
supports multiplicative scaling of reference keys. Fetch arXiv 2602.05013 §method if reachable
and check; if frequency-scaling turns out to be the paper's variant, it is a 5-line change to
the same code site (scale `angle` instead of `cos/sin` output) — implement magnitude first,
A/B if in doubt. Magnitude-scaling is self-consistently justified: it directly reduces the
high-frequency bands' contribution to the QK dot product.

## Current code map (verified anchors)

- Reference tokens: `iris.c` `iris_img2img` (~line 1271) — reference image VAE-encoded to latent
  tokens, appended to the image stream, RoPE T offset per reference (T=10,20,30 …;
  `t_offset` plumbing at iris.c:76-107). Multi-reference supported. In-context = strength 1.0.
- RoPE tables: `iris_transformer_flux.c` `compute_rope_2d` (~line 869) and
  `compute_rope_2d_with_t_offset` (~line 929). Layout per token: 128 floats = 4 axes × 32 dims;
  axis order T(0-31) H(32-63) W(64-95) L(96-127); within an axis, 16 frequency PAIRS stored as
  `[d*2], [d*2+1]` sharing one cos/sin; `base_freqs[d] = 1/theta^(2d/32)` so **pair d=0 is the
  HIGHEST frequency, d=15 the lowest** (d̃ = d/15). For reference tokens: T axis = rotation by
  t_offset, H/W = spatial position, L = identity (cos=1, sin=0).
- Cached tables: transformer struct holds `cached_ref_rope_cos/sin` and
  `cached_combined_rope_cos/sin` (~lines 305-319) — rebuilt only when shapes change.
- RoPE application: CPU `apply_rope_2d`-family in iris_transformer_flux.c; GPU kernels
  `apply_rope_unified`, `apply_rope_2d_bf16`, `rope_bf16` in iris_shaders.metal.
  **Known pitfall (CLAUDE.md): the unified GPU kernel uses consecutive-pair indexing (d, d+1),
  not axis-half indexing — keep any new table layout identical to the existing one.**
- Attention: double blocks concat `[TEXT, IMAGE(+REF)]` K/V (`iris_metal_attention_fused` /
  `iris_gpu_attention_fused_bf16` call sites ~lines 1692-1960); single blocks full self-attn
  (~2635, 2884, 3254). MPSGraph SDPA is the primary route with custom-kernel fallback
  (iris_metal.m `g_sdpa_graph_cache`).
- Shipped sref path: web/server.py — style upload → `content_destroy_png` (patch-shuffle,
  `SREF_STYLE_SHUFFLE_GRID=6`) → in-context img2img at full strength. Locate the CLI's
  reference/style flags in main.c for the new options.

## Phase 1 — RoPE band-control on reference keys

### Math

For reference tokens only, for the K projection only, scale each frequency pair d of the H and
W axes by:

    s(d) = s_hf + (s_lf − s_hf) · (d/15)^β        β = 2 (fixed)

Defaults to sweep: s_hf ∈ {0.0, 0.2, 0.4, 0.6}, s_lf ∈ {1.0, 1.25, 1.5, 2.0}.
- T axis: leave at 1.0 by default (relative T rotation is a constant per-reference phase, not
  the spatial-copying force). Add an option to include it in the sweep later.
- L axis: identity rotation for image tokens — scaling it would scale raw key content, NOT
  position. Never scale L.
- Apply in **single-stream blocks only** by default (paper + our champion's
  freeze_double_stream_scales both say appearance/style lives in single-stream). Config flag to
  extend to double blocks for the sweep.
- Timestep schedule: linear ramp of both scales across the 4 steps from a start fraction to the
  full value, e.g. `s(t) = lerp(s_start, s_end, step/(n_steps−1))`. First implementation: no
  schedule (constant); add the ramp as a follow-up sweep axis. (Paper: both scales increase
  over time — i.e. weakest high-freq suppression late.)

### Implementation (the exact-and-free trick)

Rotation is per-pair 2×2 orthogonal; a scalar per pair commutes with it. Therefore
"apply RoPE, then scale pair d of K by s(d)" == "apply a K-side RoPE table whose cos/sin entries
for pair d are pre-multiplied by s(d)". So:

1. Build a **second reference RoPE table for keys**: `cached_ref_rope_cos_k/sin_k` =
   the existing `compute_rope_2d_with_t_offset` output with `cos,sin` of H/W pairs multiplied
   by s(d). (Target-image and text tokens: unchanged tables. Reference Q: unchanged table.)
2. Thread a K-specific table through RoPE application for the reference rows in single-stream
   blocks. Today Q and K share one table per token; the change is: where RoPE is applied to K
   for reference-token rows in single blocks, use the K-table. CPU path and both GPU kernel
   paths (f32 + bf16) must change identically — respect the consecutive-pair GPU indexing.
3. If threading a second table through the fused/unified kernels is invasive, the fallback is a
   post-RoPE elementwise scale on reference K rows (a tiny extra kernel / loop). Prefer the
   baked-table version: zero per-step cost, no new kernel.
4. Config plumbing: two floats on the generation params (`sref_rope_shf`, `sref_rope_slf`,
   default 1.0/1.0 = OFF → bitwise-identical behavior when unused). CLI flags
   `--sref-shf`, `--sref-slf`; env `IRIS_SREF_SHF/SLF` for web/server.py. Applied only to
   tokens that are references (t_offset > 0).
5. Rebuild the K-table when scales change (they're per-generation constants; per-step only if
   the schedule lands).

### Gate script (build first, before the C change)

`debug/sref_rope_gate.py` — the in-context analogue of debug/sref_ref_discrimination.py:
- Inputs: ≥4 diverse reference IMAGES (reuse the style refs under /Volumes/2TBSSD/sref_eval/
  e.g. the WikiArt paintings + churchill line-art + flat sticker used for the adapter gate),
  fixed prompt ("a cat sitting on a chair"), fixed seed, size 512.
- For each ref: run `./iris` with the ref as style reference (current shipped invocation) →
  outputs. Metrics:
  (a) DISCRIMINATION: pairwise pixel corr across refs — must stay LOW (in-context already
      discriminates; the band-scaling must not destroy this). max cross-ref corr < 0.90.
  (b) STYLE ADHERENCE: CSD cosine (train/scripts/csd_features.py) between each output and its
      OWN reference — higher is better; report mean.
  (c) CONTENT LEAKAGE: CSD/SigLIP similarity between output and reference should come from
      style, so also report prompt adherence proxy (CLIP text sim if available; else eyeball
      grid) and pixel corr output-vs-reference (high = copying; should DROP vs the
      no-band-scaling baseline).
- Emit one JSON row per config; save the image grid per config for eyeballing.

### Sweep + A/B protocol

Baselines: (A) shipped rail as-is (patch-shuffle ON, bands OFF). (B) shuffle OFF, bands OFF —
expected to show copying (this is the failure the shuffle was hiding; it anchors the metric).
Then grid: shuffle OFF × {s_hf} × {s_lf}; then best-cell × shuffle ON (they may compose).
Deliverable: table in BACKLOG + chosen defaults wired into web/server.py (keep
patch-shuffle available behind its env; if bands beat shuffle cleanly, make shuffle opt-in).
Wrap the sweep in `caffeinate` (AGENT.md rule).

## Phase 2 — Strength bias γ

Additive bias `log(γ)` on attention logits for generation-token (and text-token) queries against
reference-token keys, all blocks where references participate. γ=1 → no-op; γ=0 → −inf ≈ mask.
- CPU path: add bias on the logits slice before softmax (the column range of reference tokens is
  known from the sequence layout: [TEXT, IMAGE, REF...]).
- GPU: custom fused kernel gains an optional (bias, col_start, col_end) argument. MPSGraph SDPA
  path: route sref generations through the explicit-mask SDPA variant if one exists (BL-008
  added a causal-mask SDPA for Qwen3 — same shape of change: additive mask tensor), else fall
  back to the custom kernel when γ≠1. Do NOT regress the non-sref fast path: γ==1 must take the
  existing route (guard at call site).
- CLI `--sref-strength` maps to γ (replace/alias the current latent-blend semantics for the
  style path; keep old behavior for plain img2img). Validate: γ sweep {0.5, 1, 2, 4} shifts CSD
  style adherence monotonically while discrimination stays intact.

## Phase 3 — Reference-KV reuse (perf; only after Phase 1 ships)

Reference latents are clean and constant across steps; their hidden trajectory varies only via
(a) timestep modulation (AdaLN) and (b) attention FROM reference TO noisy tokens.
- Add an asymmetric mask: reference-token queries attend only to [TEXT, REF] columns (not the
  noisy image). Validate this alone first — it changes output; run the gate + golden-image
  diff. (OminiControl2 evidence says quality holds; our 4-step distilled case must be checked.)
- Then: at step 1, record per-block reference K/V (post-RoPE, post-QK-norm) for all 25 blocks;
  steps 2–4 skip the reference rows entirely (no Q/K/V projection, no MLP, no attention rows for
  them) and splice the cached K/V columns into each block's attention. Accept the frozen-at-
  step-1 modulation approximation — that is the OminiControl2 design. Cache in bf16
  (25 blocks × 2 × ref_seq × 3072; ~315 MB at 1024 ref tokens — pool it, free after generation).
- Expected win: reference share of sequence × ~3/4 of transformer time (measure with the
  existing zImage/flux timing counters; report s/img before/after at 512px 1-ref and 2-ref).
- Keep behind a flag (`--sref-kv-reuse`, default ON only after A/B shows no visible quality
  delta + gate unchanged).

## Validation & project rules (mandatory)

- `make test` after every phase; `make mps` (NOT bare make) to rebuild the shipped binary; run
  a manual flux sanity gen (CLAUDE.md commands). Verify CPU/BLAS backend still matches GPU
  (project rule: optimizing one backend must not regress others — run one gen per backend and
  compare corr > 0.99 for identical seeds where the backend is deterministic).
- Default-off invariant: with all new flags at defaults (1.0/1.0/γ=1/no-reuse), outputs must be
  bit-identical (or corr ~1.0) to pre-change — add a golden-image check to the PR.
- Log every sweep result + the chosen defaults to BACKLOG (numbers + date). Findings that kill
  or confirm the paper's claims on 4-step distilled → BACKLOG (this is the first test of that
  question; the paper only validated 50-step Flux.1-dev).
- caffeinate any sweep/batch run. Use BSD commands (AGENT.md).
- Commit per phase; never commit unrelated files.

## Acceptance criteria

- Phase 1: on the 8-ref gate, some (s_hf, s_lf) cell achieves — discrimination max cross-ref
  corr < 0.90 AND output-vs-own-reference pixel corr lower than the shuffle-off baseline
  (copying suppressed) AND CSD style adherence ≥ patch-shuffle baseline (style preserved).
  If NO cell beats patch-shuffle on style adherence without losing discrimination, that is a
  valid negative result — log it to BACKLOG (the paper's mechanism may not survive 4-step
  distillation) and stop before Phases 2–3 rather than tuning forever.
- Phase 2: γ monotonically trades style strength vs prompt adherence with defaults no-op.
- Phase 3: ≥20% wall-clock reduction at 512px 1-ref with gate + visual parity.
