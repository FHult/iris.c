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
- [x] **Qwen3 async layer prefetching** - Uses GCD to prefetch next layer's weights
      while computing current layer. Overlaps mmap I/O with CPU computation.
      ~30% speedup on text encoding (17s → 12s). Set FLUX_NO_PREFETCH=1 to disable.
- [x] **Metal command batching** - Already implemented via flux_gpu_batch_begin/end.
      Queues all block operations without waiting, waits only at end.
- [x] **Transformer weight prefetch** - Already optimized by batching design.
      Weight loading overlaps with GPU computation naturally.

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

---

## Memory Analysis - 2024-01-31

### Memory Profile (256x256, 4-step generation)

| Component | Peak Usage | Notes |
|-----------|------------|-------|
| Model weights (mmap) | ~8.5 GB | Read-only, shared, doesn't count against RAM |
| GPU buffer pool | ~200 MB | 64 activation buffers, well-managed |
| Weight cache | ~100 MB | 512 entries, LRU eviction |
| VAE work buffers | ~50 MB | Adaptive sizing based on image resolution |
| Transformer work | ~30 MB | Pre-allocated once |
| Text embeddings | ~20 MB | Allocated per generation |

### What's Already Well-Designed

1. **Mmap weight loading** - Zero-copy access to model weights
   - Location: `flux_safetensors.c:224` - `mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0)`
   - Weights stay on disk, OS pages in on demand
   - Multiple processes can share same physical pages

2. **GPU buffer pooling** - 64-buffer activation pool with LRU
   - Location: `flux_metal.m:180-250` - `g_activation_buffers[]`
   - Buffers reused across operations
   - Avoids GPU allocation overhead

3. **Weight caching** - 512-entry cache for bf16→f16 conversions
   - Location: `flux_metal.m:130-178` - `g_weight_cache[]`
   - LRU eviction when full
   - Avoids repeated conversions

4. **Adaptive VAE buffers** - Size based on actual image dimensions
   - Location: `flux_vae.c:50-80` - `vae_alloc_work_buffers()`
   - 94% memory reduction for small images vs fixed allocation

5. **Transformer work buffers** - Pre-allocated once per context
   - Location: `flux_transformer.c:2800-2850`
   - Reused across all denoising steps

### Memory Inefficiencies Analyzed

#### [x] VAE Attention Allocations - **NOT WORTH IT**

**Location**: `flux_vae.c:292-296`

**Analysis**: Allocates ~400 MB for 1024×576 (16:9) per attention call.
Benchmarked malloc/free overhead vs pre-allocated buffers:

| Generation | Fresh malloc | Pre-allocated | Difference |
|------------|--------------|---------------|------------|
| First | 42 ms | 42 ms | 0 ms (page faults unavoidable) |
| 2nd+ | 1.2 ms | 0.6 ms | **0.6 ms** |

**Why skip**: OS caches freed pages. After warmup, malloc reuses same
physical pages with ~1 ms overhead. Pre-allocation saves only ~1 ms
per generation (0.2% of VAE decode time). Not worth the complexity.

#### [x] Transformer RoPE Per-Step - **NOT WORTH IT**

**Location**: `flux_transformer.c:2757-2765`

**Analysis**: ~1 MB per step × 4 steps = ~4 MB per generation.
Same page caching effect applies - negligible overhead after warmup.

### Conclusion

Memory allocation is already well-optimized. The OS page cache eliminates
most malloc overhead after the first generation. No changes needed.
