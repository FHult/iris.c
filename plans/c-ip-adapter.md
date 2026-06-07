# Plan: C-side IP-Adapter inference (G-1)

**Status:** **Phase 0 + Phase 1 DONE (2026-06-07)** — `iris_ip_adapter.c` (load /
perceive / get_kv / inject) is **bit-exact** with the Python reference (corr 1.000000,
max_abs 0.00000) on the committed synthetic fixtures, in `make test-unit`. Generator:
`debug/gen_ip_adapter_fixture.py`; test: `debug/test_ip_adapter.c`. (Also fixed an
iris_safetensors bug en route: mlx's `"__metadata__":null` broke the header parse.)
**Next: Phase 2** — wire `inject` into the Flux transformer blocks (CPU → bf16 → MPS).
The endgame for the whole training flywheel — the bridge from "we trained an adapter"
to "it runs in the `iris` binary." Until Phase 2/3 land, adapters run in C only via this
unit-tested module, not yet end-to-end in generation.

**Trigger:** start once warmup-run2 (first campaign on the corrected VAE-Q1 convention)
yields a champion checkpoint worth shipping. Phase 1 can begin earlier (it validates the
port against precomputed features, GPU-light).

## Recommendation

Do **Phase 0→2 against a champion checkpoint via the precomputed-feature path first** —
that ships in-C adapter evaluation without the SigLIP port; **Phase 3 follows for
interactive use**. **Trigger is a quality champion from warmup-run2.**

### What counts as a "champion checkpoint"

"Champion" has two meanings; only the cheap one is automatic.

**1. Training-internal champion (automatic).** `FlywheelDB.get_best()` ranks iterations by
`cond_gap DESC` and records it as `best_checkpoint` (+ `best.safetensors`). The metrics:
- `cond_gap = loss_null − loss_cond` (train_ip_adapter.py:2077): how much lower the diffusion
  loss is *with* image conditioning vs without — i.e. the adapter is actually using the
  reference. Higher = stronger; `< 1%` triggers a "may not be learning" warning. Primary
  signal (stable at the 1000-step iteration budget).
- `ref_gap = loss_cross_ref − loss_self_ref` (2112): style/content separation. Noisy at 1000
  steps, so secondary.
- **Limit:** these are training-loss *proxies* — they show the adapter learned to exploit
  conditioning, NOT that generated images look good or transfer. A high-cond_gap checkpoint
  can still produce over-cooked or off images. `get_best` gives a *candidate*, not a verdict.

**2. True output-quality champion (the real test — GPU-gated).** The apples-to-apples signal
is the golden-set eval (`quality_gate.py` + `evaluate_golden_set`): generate from a fixed
prompt+reference set and score clip_i / clip_t / aesthetic (higher better) + lpips / fid
(lower better) vs a baseline / prior campaign. **This requires running the adapter through
inference — i.e. G-1 itself** (or, until then, the Python `test_ip_adapter_inference.py`
harness on idle GPU). The loop closes: the definitive champion signal needs the inference
path this plan builds.

**Checklist to declare a champion** — needs (1) AND (2):
1. *Automatic (free):* `cond_gap` trending up and comfortably > 1% (not flatlining), `ref_gap`
   climbing toward/above 0, grad stable → flywheel records `best_checkpoint`.
2. *Definitive (idle GPU):* golden-set eval on that checkpoint shows clip_i/aesthetic up vs
   the base model (no adapter), no lpips/fid regression, ideally beating the prior campaign.
   `test_ip_adapter_inference.py` also saves side-by-side panels for eyeballing.

**warmup-run2 caveat — two different bars.** warmup-run2 is an *exploration bootstrap* (first
campaign on the corrected VAE-Q1 convention, 1000 steps/iter to validate the loop), not the
final shippable model. So:
- **For validating G-1 (Phases 0–2): a low bar suffices.** Any checkpoint where cond_gap
  shows the adapter is genuinely conditioning is enough to prove the C inference port is
  correct (parity vs Python on the same checkpoint is what matters, not the checkpoint's
  absolute quality).
- **For shipping to users: the high bar (golden-set win) applies**, and the production
  champion will likely come from a later, longer, tuned campaign — not warmup-run2.

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

#### Phase 2 detailed design (execution-ready)

**Exact hook point (CPU double block, `double_block_forward` ~2207).** The image Q is
projected into `tf->work2` (~2257), QK-RMSNorm'd at `apply_qk_norm(img_q, img_k, …)`
(~2278), then RoPE'd at `apply_rope_2d(img_q, …)` (~2284), then joint attention →
out-proj → gate → residual into `img_hidden`. The header contract says inject takes the
**post-QK-norm** Q and adds `ip_scale·SDPA(Q, k_ip, v_ip)` into `img_hidden` *after* the
native attention+residual. So inject is called **inside** the block (where `img_q` is
live) near the end, using the saved `img_q` and the post-residual `img_hidden`.

**Open parity question to resolve first (the fiddly bit):** does training's
`_flux_forward_with_ip` reuse the Q **before or after RoPE**? Native attention uses
post-RoPE Q; the IP cross-attention may use the post-QK-norm pre-RoPE Q (k_ip/v_ip are
not RoPE'd — they're SigLIP-derived, position-free). model.py `inject` takes `img_q` with
no RoPE applied to k/v, so almost certainly **post-QK-norm, pre-RoPE**. Confirm by dumping
both from the Python path and matching — extend the Phase-0 fixture with a block-level
golden (img_q candidates + the IP contribution) before wiring. Getting this wrong is the
#1 silent-divergence risk.

**State threading.** Add to the transformer forward (and block-forward signatures, or via
`tf->`): `iris_ip_adapter_t *ip` (nullable → no-op when absent), `const float *ip_embeds`
(`[num_image_tokens, hidden]`, computed once per image before the denoise loop via
`perceive`), and reusable per-block `k_ip`/`v_ip` scratch (`[num_image_tokens, hidden]`).
Per block: `get_kv(block_idx, ip_embeds, k_ip, v_ip)` then `inject(block_idx, img_q,
img_seq, k_ip, v_ip, img_hidden)`. Block index: double blocks `0..num_double-1`, single
blocks `num_double..N-1`. Skip entirely when `ip == NULL` or `ip_scale[block]==0`.

**Per-variant work (single inject impl; the variants only marshal Q/hidden):**
- CPU `double_block_forward` (2207) + `single_block_forward` (3449): `img_q` and
  `img_hidden` are already f32 in `tf->work*` — pass directly. Easiest; do first.
- bf16 `double_block_forward_bf16` (1946) + `single_block_forward_bf16` (2967): Q is in a
  fused bf16 kernel. Read back the post-QK-norm img_q slice to f32 (or compute inject in
  the existing f32 readback), run inject, accumulate into the f32 hidden before re-upload.
- MPS-resident single (`single_block_forward_gpu` 2454 / `_chained` 2702): same, reusing
  the path's existing GPU↔CPU marshalling; no new GPU kernels needed (inject is small —
  CPU SDPA on `[img_seq, hidden]` × `[128, hidden]`).

**CLI / setup (main.c).** Add `--ip BUNDLE_DIR`, `--ip-features PATH` (precomputed SigLIP
`[729,1152]` → skips Phase 3), `--ip-scale F` (→ `sref_strength`/`effective_scale`),
`--ip-style-only`. Load the adapter once (like the VAE); on `--ip-features`, `perceive`
once into `ip_embeds`; thread `ip`/`ip_embeds` into the denoise loop. Replace the
`main.c:1273` reject for `--ip` (keep it for `--sref`).

**Test gates (in order).**
1. *Block-level CPU golden* (extend `gen_ip_adapter_fixture.py` + `test_ip_adapter.c`):
   feed a synthetic post-norm img_q + a baseline img_hidden through the inject hook and
   assert `img_hidden += contribution` matches the Python `_flux_forward_with_ip` IP term.
   Resolves the pre/post-RoPE question hermetically.
2. *End-to-end* (`debug/`): `iris --ip-features feats.bin` vs a Python reference generate
   on the same checkpoint+features — within `run_test` pixel tolerance.
3. *No-op invariant*: `ip_scale=0` (or no `--ip`) reproduces the baseline image
   **bit-for-bit** (guards that the hook is truly inert when disabled).

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
