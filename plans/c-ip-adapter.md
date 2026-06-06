# Plan: C-side IP-Adapter inference (G-1)

**Status:** design / not started. The endgame for the whole training flywheel — it's
the bridge from "we trained an adapter" to "it runs in the `iris` binary." Until this
lands, every trained adapter is only evaluable through the Python harness
(`test_ip_adapter_inference.py`), never in the shipped product.

**Trigger:** start once warmup-run2 (first campaign on the corrected VAE-Q1 convention)
yields a champion checkpoint worth shipping. Phase 1 can begin earlier (it validates the
port against precomputed features, GPU-light).

## Recommendation

Do **Phase 0→2 against a champion checkpoint via the precomputed-feature path first** —
that ships in-C adapter evaluation without the SigLIP port; **Phase 3 follows for
interactive use**. **Trigger is a quality champion from warmup-run2.**

---

## Current state (what's done, what's missing)

- **Done:** `train/export/iris_ip_adapter.h` declares the complete C API + struct;
  `train/export/export_adapter.py` produces the bundle; `train/ip_adapter/model.py` is the
  MLX reference; `main.c` already parses `--sref`/`--sref-scale` (then rejects at
  `main.c:1273` "not yet implemented").
- **Missing:** `iris_ip_adapter.c` (no implementation); the inject hooks inside the Flux
  transformer blocks; a SigLIP encoder in C; CLI wiring to actually run an adapter.

## Architecture recap (the math to port, from model.py)

Per generated image, once before the denoising loop:
1. **Perceive** (`PerceiverResampler`): learned `query_tokens [128, 3072]` cross-attend to
   SigLIP features `[729, 1152]` → `ip_embeds [128, 3072]`.
   `cross_attn = MultiHeadAttention(dims=3072, heads=24, key_input_dims=1152)` + a LayerNorm
   on the query side (`norm_weight/bias`). (model.py:25-60)

Per transformer block (N = num_double + num_single; 4B = 5 + 20 = 25):
2. **get_kv**: `k_ip = ip_embeds @ ip_k_stacked[block]`, `v_ip = ip_embeds @ ip_v_stacked[block]`
   where `ip_{k,v}_stacked` are `[N, 3072, 3072]`. Output `[128, 3072]` each. (model.py:125)
3. **inject**: the block's image query (POST QK-RMSNorm — see contract below) does
   scaled-dot-product attention against `k_ip/v_ip`, and the result is blended in:
   `attn = SDPA(img_q, k_ip, v_ip)` (scale `head_dim**-0.5`, head_dim=128);
   `img_hidden += ip_scale[block] * attn`. No-op when `ip_scale[block] == 0`
   (style-only zeroes the double-stream blocks). (model.py:178-214, header:186-228)

**Critical contract (header:186-200):** `img_q` is passed in ALREADY QK-RMSNorm'd — the
same post-norm Q the native self-attention uses. Do NOT re-normalize inside inject. Getting
this wrong is the most likely silent-divergence bug.

## Bundle format (export_adapter.py → the C loader reads this)

- `adapter_weights.safetensors`:
  - `perceiver.query_tokens` `[128, 3072]`
  - `perceiver.{query,key,value,out}_proj` (key/value are `[3072, 1152]`, query/out `[3072,3072]`)
  - `perceiver.norm_{weight,bias}` `[3072]` (always f32)
  - `ip_k_stacked`, `ip_v_stacked` `[N, 3072, 3072]`
  - `ip_scale` `[N]` (f32)
- `adapter_meta.json`: `num_blocks, num_double_blocks, num_single_blocks, hidden_dim,
  head_dim, num_heads, num_image_tokens, siglip_dim, quant`.
- **Quant**: `bfloat16 | float16 | int8`. int8 = per-row symmetric, scale stored as
  `<name>.scale`; dequant `x[i,:] = q[i,:] * scale[i]`. (export_adapter.py:50, 290)

`iris_safetensors.c` already handles bf16→f32; int8 needs a small dequant on load (or a
cached-bias-style INT8 GEMM path later — start by dequanting to f32 at load).

## The SigLIP crux + the intermediate that unblocks shipping

The adapter conditions on SigLIP-400M features `[729, 1152]`. The C binary has **no SigLIP
encoder** — porting that ViT is a sub-project of its own (Phase 3). To ship/evaluate trained
adapters without it first, support a **precomputed-feature path**: the pipeline already
precomputes SigLIP features per image, so `--ip-features feats.npy` (or a small raw `.bin`)
feeds `[729,1152]` directly to `perceive`. This is also exactly the Phase-1 parity input.

---

## Phased implementation

### Phase 0 — scaffolding + parity fixtures (GPU-light, can start now)
- Add `iris_ip_adapter.c` + wire into Makefile (all three targets); empty stubs matching
  the header so it links.
- Export a tiny **golden bundle** from a real checkpoint + a Python-side dump of the
  reference tensors at each stage for a fixed SigLIP input: `ip_embeds` (post-perceive),
  `k_ip/v_ip` for a couple of blocks, and `inject` output for a fixed `img_q`. Commit as
  small fixtures (a few hundred KB) under `debug/`.
- `debug/test_ip_adapter.c` (CPU-only, into `make test-unit`): load bundle, run each stage,
  assert against the golden dumps (corr > 0.999 / small max-abs, like the VAE-1 harness).

### Phase 1 — `iris_ip_adapter.c` (CPU), validated against fixtures
- `iris_ip_adapter_load`: parse `adapter_meta.json` with `iris_config_parse.h` helpers
  (already hardened); open the safetensors; dequant int8→f32 or bf16→f32 into the struct.
- `iris_ip_adapter_perceive`: MultiHeadAttention(query_tokens, siglip, siglip) + the query
  LayerNorm. Reuse `iris_matmul`/`iris_linear` + the existing attention/softmax kernels
  (now alias-safe). Output `ip_embeds [128,3072]`.
- `iris_ip_adapter_get_kv`: two GEMMs against `ip_{k,v}_stacked[block]`.
- `iris_ip_adapter_inject`: SDPA(img_q, k_ip, v_ip) → `img_hidden += ip_scale*attn`; early
  return when `ip_scale==0`. Use the existing `flash_attention`/SDPA path (head_dim=128).
- **Gate:** Phase 0 fixtures pass at every stage. This proves the port with zero transformer
  changes and (mostly) no GPU.

### Phase 2 — transformer inject hooks (CPU → bf16 → MPS)
- Hook sites (iris_transformer_flux.c): `double_block_forward` (CPU 2207, bf16 1946),
  `single_block_forward` (CPU 3449, bf16 2967, gpu 2454/2702). Inject **after** the native
  attention + QK-norm, passing the post-norm image Q (the header contract).
- Thread an `iris_ip_adapter_t *` (nullable) through the transformer forward + a per-image
  `ip_embeds` (computed once). Block index maps: double blocks 0..num_double-1, single
  blocks num_double..N-1.
- Order: CPU/generic path first (cheapest to validate vs Python end-to-end on precomputed
  features), then the bf16 distilled path (production), then the MPS f32 path. Keep a single
  inject implementation; the per-path work is marshalling Q/hidden in/out of GPU tensors.
- **Gate:** `iris --ip-features … ` end-to-end output matches the Python
  `_flux_forward_with_ip` reference within the run_test pixel tolerance; `ip_scale=0`
  reproduces the no-adapter image bit-for-bit.

### Phase 3 — SigLIP encoder in C/Metal (interactive `--ip ref.png`)
- Port SigLIP-400M vision tower (patchify → ViT blocks → `[729,1152]`). Largest piece;
  mirror the Qwen3/VAE loaders (safetensors + bf16 cache + MPS GEMM). Validate features vs
  the Python/precompute encoder (the pipeline's SigLIP) to ~1e-2.
- Wire `--ip ref.png` → encode_siglip → perceive → denoise.

### Phase 4 — `--sref` training-free variant (separate, A1/v2.6)
- The "approximate, training-free" style ref (RoPE-attenuation) is independent of the
  trained adapter and SigLIP. Lower priority; document separately. Keep `--ip` (trained)
  and `--sref` (training-free) as distinct flags.

---

## CLI / wiring
- `--ip BUNDLE_DIR` (trained adapter), `--ip-scale F` (maps to `sref_strength` /
  `effective_scale`), `--ip-style-only` (zero double-stream, model.py:142), and the
  intermediate `--ip-features PATH` (precomputed `[729,1152]`, skips Phase 3).
- Remove the `main.c:1273` hard-error once Phase 2 is real; keep it for `--sref` until Phase 4.
- Load the adapter once (like the VAE), compute `ip_embeds` once per image, reuse `k_ip/v_ip`
  buffers across steps.

## Parity-test strategy (the spine)
Mirror the VAE-Q1/VAE-1 approach: Python reference dumps at each boundary + C asserts.
- Unit (Phase 0/1, `make test-unit`): perceive / get_kv / inject vs golden fixtures.
- Integration (Phase 2): full generate with precomputed features vs Python; and
  `ip_scale=0` == baseline.
- E2E (Phase 3): C SigLIP features vs pipeline SigLIP; full `--ip ref.png` vs Python.

## Risks / open questions
- **QK-norm contract** (header:186): must reuse the post-norm Q; the bf16 path computes Q in
  a fused kernel — extracting the post-norm Q there is the fiddliest part.
- **MultiHeadAttention parity**: MLX `nn.MultiHeadAttention` packs/scales a specific way;
  match its projection order + scaling exactly (golden fixture catches drift).
- **int8 path**: start by dequanting to f32 at load (simplest correct); a true INT8 GEMM is
  a later perf option, not needed for correctness.
- **Memory**: `ip_{k,v}_stacked` are `[25, 3072, 3072]` ≈ 470M params each in f32 (~1.9 GB ×2).
  Keep bf16 in the cache and convert per-block, or compute K/V per block from a smaller
  representation — watch the 32 GB budget. (bf16 storage ≈ 0.94 GB each.)
- **head_dim source**: derive from the block's Q (`head_dim=128`) per model.py:200, not a
  hardcode, so 9B (head_dim still 128, more heads) works unchanged.

## Effort / sequencing
- Phase 0+1: ~2-3 days (the adapter is small; the work is careful parity).
- Phase 2: ~2-3 days (3 forward variants × marshalling + validation).
- Phase 3 (SigLIP-in-C): ~1 week (a full ViT port + MPS).
- Phase 4: ~2-3 days.
Do **Phase 0→2 against a champion checkpoint using precomputed features** first — that ships
adapter evaluation in C without the SigLIP port. Phase 3 follows for interactive use.
