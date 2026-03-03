# flux2.c — Persistent Memory

## Project Basics
- Pure C + Metal inference engine for Flux (4B/9B) and Z-Image-Turbo (6B)
- Build targets: MPS (Apple Silicon), BLAS, generic
- Zero external ML-framework dependency policy (see CLAUDE.md)
- Always run `make test` after changes; commit only when tests pass

## Key Files
- `flux_shaders.metal` — all custom GPU compute kernels
- `flux_metal.m` — MPS/MPSGraph wiring, buffer pools, weight cache
- `flux_metal.h` — GPU API surface
- `flux_transformer.c` — Flux double/single block forward pass
- `flux_zimage_transformer.c` — Z-Image S3-DiT forward pass
- `flux_kernels.c` — CPU kernels; also routes to GPU when large enough
- `flux_sample.c` — Euler ODE denoising loop
- `flux_qwen3.c` — Qwen3 text encoder
- `flux_vae.c` — VAE encoder/decoder
- `embcache.c` — 4-bit quantized embedding cache

## Architecture Patterns
- Weight buffers cached by CPU pointer (512-entry LRU) — static only
- Dynamic tensors (VAE K/V) must use `flux_metal_sgemm()` NOT `_cached()` variant
- Activation buffers pooled (64 slots); deferred release queue prevents race
- Batch mode: `flux_metal_begin_batch()` / `flux_metal_end_batch()` for multi-op
- BF16 weights: MPS can't do f32×bf16, so bf16→f16 conversion at load time
- `flux_metal_warmup_bf16()` should be called after model load to pre-warm cache

## Known Bugs / Pitfalls
- MPS B-cache misuse → VAE decode corruption (fixed: split `_sgemm` / `_sgemm_cached`)
- seq_k hard limit ~1024 in `attention_fused` (threadgroup memory cap) — BL-001
- Linear graph cache limited to 32 entries — may miss on 9B+CFG — BL-007

## Performance Backlog
See `memory/perf_backlog.md` for full backlog. See `memory/gap_analysis.md` for
revised gap analysis (corrects initial review assumptions after deep code read).

KEY CORRECTION: BF16 attention already uses Apple native SDPA
(`scaledDotProductAttentionWithQueryTensor` on macOS 14+). The "1024 seq_k limit"
was wrong — it's ~7552 for custom kernel, unlimited for MPSGraph path. The BF16
distilled model path is already state-of-the-art.

Real open items (priority order):
- BL-004: simdgroup_matrix for attention GEMM tiles (P3, M3+ only — NOT for M1 Max)
- BL-005: Native `bfloat` MSL type (P3, M3+ only — NOT for M1 Max)

Qwen3 performance insight (2026-03-02 deep profiling):
- Root cause of 11s first-encoding: MPSGraph lazy compilation of 5 linear graph shapes
- In mmap mode: GPU forward pass with use_bf16=1; in no-mmap mode: use_bf16=0 (CPU)
- BF16 cache (BF16_WEIGHT_CACHE_SIZE) shared between Qwen3 AND transformer
  - Mmap 4B: 243 Qwen3 slots + ~215 transformer slots = 458 < 1024 ✓
  - No-mmap 4B: 324 Qwen3 slots + ~215 transformer slots = 539 (was > 512, now OK)
- After warmup: encoding 11s→0.6s on warm OS page cache

COMPLETED items (committed):
- BL-001 ✓: Float32 attention path now uses MPSGraph native SDPA (Flash Attention) as primary route.
  `get_sdpa_graph_cache_f32` + `flux_gpu_attention_mpsgraph_f32` + separate g_sdpa_f32_graph_cache[32].
  `flux_gpu_attention_fused` tries MPSGraph first, falls back to custom kernel. Base models only.
- BL-007A ✓: SDPA graph cache 8→32
- BL-007B ✓: Linear graph cache 32→64
- BL-006 ✓: MTLResidencySet for weight buffers (macOS 15+); note: selector is `newResidencySetWithDescriptor:error:` not `newResidencySetWithDescriptor:`
- BL-002 ✓: simd_sum/simd_max in all 12 Metal kernels (8 barriers→2, 1KB threadgroup→32B)
- BL-003 ✓: GPU bias fusion via flux_metal_sgemm_cached_bias(); CPU bias loop eliminated for Metal path
- BL-008 ✓: MPSGraph causal SDPA fallback for Qwen3 seq>512 (GQA tiling + causal mask, cached by shape)
- BL-NEW ✓ (commit 750a24b): Qwen3 MPSGraph warmup at load time + BF16 cache 512→1024
  - Warmup runs dummy GPU forward pass in qwen3_model_load_mmap (mmap mode only)
  - Moves ~10s MPSGraph compilation from "Encoding text" to "Loading Qwen3 encoder"
  - BF16_WEIGHT_CACHE_SIZE 512→1024 prevents cache overflow with large model combos
  - Result: encoding time warm cache 12.2s→0.6s; total session 31.1s→17.2s

Verified unknowns (resolved 2026-03-02):
- Qwen3 causal attention: FIXED (BL-008) — MPSGraph fallback added for seq>512,
  both f32 and BF16 paths. BF16 path converts to f32, runs MPSGraph, converts back.
- BF16 GEMM: INTENTIONAL — native BF16 only when out_dim>=8192 or seq>=8192.
  FFN gate/up (9216+) gets native BF16. Q/K/V projections (3072/4096) use f32 cast.
  Threshold based on benchmarking; may be worth lowering to 4096 at high resolution.

## MLX Analysis
See `memory/mlx_analysis.md` for the full investigation.

Key findings:
- MLX has a C++ API (`#include <mlx/mlx.h>`) usable without Python
- `mlx::core::fast::scaled_dot_product_attention()` — Flash Attention, Metal-backed, GQA support
- `mlx::core::fast::rms_norm()` — fused single-pass Metal kernel
- `mlx::core::fast::rope()` — 1D only, not applicable to Flux's 4-axis RoPE
- No fused AdaLN; no ANE backend; no GGUF; no stable ABI
- Available via `brew install mlx` (dylib only, no static)
- MIT licensed — Metal kernel extraction into flux_shaders.metal is legal

"Cake and eat it" recommendation:
1. Pattern A (extract Metal kernels, zero runtime dep) — implement Flash Attention using
   MLX's `scaled_dot_product_attention.metal` as reference, embed in flux_shaders.metal.
   This resolves BL-001 with no new dependency.
2. Pattern B (#ifdef WITH_MLX optional backend) — viable for large-res users (>1024px)
   but has buffer-copy overhead and ABI pinning risk.

## Framework Notes (2026-03)
- MLX v0.31.0 is the current reference for Metal kernel patterns (MIT licensed)
- MLX Metal kernels live at: github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels
- Relevant kernels to study: `scaled_dot_product_attention.metal`, `rms_norm.metal`,
  `rope.metal`, `softmax.metal`, `layer_norm.metal`
- ANE: NOT accessible from raw Metal/MPS. Requires CoreML or MLX (M5+, macOS 26.2+)
- macOS 26.3: JACCL bandwidth improvement is multi-device only, not relevant here
- Metal 3 (M3+/Apple9): native `bfloat` type, `simdgroup_matrix` ops available

## User Preferences
- Concise responses, no emojis
- File references as markdown links with line numbers
- Commit after validated changes
- Never commit unrelated unstaged files
