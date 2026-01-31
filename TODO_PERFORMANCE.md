# Performance Optimization TODO

## Analysis Complete - 2024-01-31

Most optimizations in this list were based on assumptions that don't hold for this model/hardware. The bf16 GPU pipeline already handles the heavy lifting efficiently.

---

## Tested & Rejected

### [x] Lower SiLU GPU threshold - **SLOWER**
- Tested: 4M → 512K made generation 12% slower
- Reason: GPU kernel launch overhead dominates at these tensor sizes

### [x] Lower Softmax GPU threshold - **SLOWER**
- Tested: 4M → 64K made generation 11% slower
- Reason: Same - GPU sync overhead exceeds compute benefit

### [x] Qwen3 RMSNorm consolidation - **SKIP**
- Analysis: Would make things slower due to GPU overhead
- Text encoder deliberately kept on CPU (runs once per generation)

### [x] GroupNorm GPU - **NOT WORTH IT**
- Analysis: Hard to implement, ~1-2% gain
- GPU sync overhead (~500μs) eats most benefit

### [x] VAE attention buffers - **ALREADY OPTIMAL**
- Analysis: Buffers already allocated once outside loop
- No action needed

### [x] Attention transposes - **LOW PRIORITY**
- Analysis: bf16 pipeline handles this efficiently
- Hard refactor for minimal gain

---

## Completed

- [x] VAE dynamic buffer sizing (94% memory reduction for small images)
- [x] GPU bf16→f16 JIT conversion (450 MB/s, used by MPS matmul)
- [x] Transformer dynamic buffer allocation
- [x] SiLU consolidation - removed duplicate qwen3_silu(), now uses flux_silu()

---

## Notes

### Why Most Optimizations Don't Help

1. **bf16 pipeline dominates** - Transformer uses fused bf16 GPU kernels
2. **Text encoder runs once** - Optimizing it saves negligible time
3. **GPU overhead is real** - Kernel launch (~10-50μs) + sync adds up
4. **Memory bandwidth limited** - Many ops are bandwidth-bound, not compute-bound

### Actual Bottleneck

For 256x256, 4 steps:
- Text encoding: ~17-23s (runs once)
- Denoising: ~30s (4 steps × ~7.5s/step)
- VAE decode: ~0.6s

The transformer denoising is already GPU-accelerated via bf16 pipeline.
Further optimization would require:
- Smaller model (quantization)
- Faster text encoder (different architecture)
- Reduced step count (distillation)
