# Performance Backlog
*Generated 2026-03-02 from framework evolution review*

## Context

This project is a pure C/Metal implementation of Flux and Z-Image transformers.
Build targets: MPS (Apple Silicon GPU), BLAS (CPU), generic (pure C).
No external ML framework dependencies by policy (CLAUDE.md).

Framework baseline at time of review:
- MLX latest: v0.31.0 (Feb 2025)
- macOS: 26.3 / Darwin 25.3
- Metal: 3+ (M3 = Apple9 GPU family, M4 = Apple10+)
- MPS/MPSGraph: current macOS 15/26 versions

---

## BACKLOG ITEMS

### BL-001 — Float32 Attention Path: Add Native SDPA
**Priority:** P2 — Medium Impact, Low Effort
**Status:** DONE

**Correction from initial analysis:**
The BF16 attention path (`flux_gpu_attention_fused_bf16`, `flux_metal.m:4868`) already
uses Apple's native `MPSGraph -scaledDotProductAttentionWithQueryTensor:` (Flash
Attention on macOS 14+) as its primary route. The "1024 seq_k limit" from the
original analysis was incorrect — the real limit for the custom kernel fallback is
~7552 tokens (32KB threadgroup / 4 bytes per float). The BF16 path is already
state-of-the-art.

**Actual gap:**
The float32 attention path (`flux_gpu_attention_fused`, `flux_metal.m:4396`) does NOT
try native SDPA — it goes straight to the custom kernel. Base models with CFG
(flux-klein-4b-base, 9b-base) run in float32 and miss the native SDPA benefit.

**Solution:**
Add an MPSGraph native SDPA attempt to `flux_gpu_attention_fused` (mirroring what
`flux_gpu_attention_fused_bf16` already does) before falling through to the custom
kernel. Cast input float32 → BF16 for the SDPA call then cast output back.

**Files:**
- `flux_metal.m:4396-4446` — `flux_gpu_attention_fused()`
- `flux_metal.m:4134` — `flux_metal_attention_fused()` (CPU pointer version)

**Impact:** Base model CFG attention only. Distilled models already optimal.

---

### BL-002 — SIMD Group Reductions (simd_sum / simd_max)
**Priority:** P2 — Low Effort, Medium Impact
**Status:** Open

**Problem:**
Every reduction in the shaders (RMSNorm, attention max/sum, softmax) uses the
log-N threadgroup barrier pattern with `shared_max[256]` / `shared_sum[256]`.
`<metal_simdgroup>` is already included but SIMD primitives are never called.

**Solution:**
Replace the inner barrier loop with `simd_sum()` / `simd_max()` followed by a
single threadgroup barrier to merge simd-group results.

```metal
// Before (8 barriers for 256 threads):
for (uint stride = threads/2; stride > 0; stride >>= 1) {
    if (tid < stride) shared_max[tid] = max(shared_max[tid], shared_max[tid+stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// After (1 barrier):
float simd_result = simd_max(local_val);
if (simd_lane_id == 0) shared_max[simd_group_id] = simd_result;
threadgroup_barrier(mem_flags::mem_threadgroup);
float result = (tid < threads/32) ? simd_max(shared_max[tid]) : 0.0f;
```

**Applies to:** `rms_norm`, `qk_rms_norm`, `attention_fused` (phases 2+4),
`attention_causal` (phases 2+4), `softmax`, and all BF16 variants of each.

**Requirement:** Apple7+ GPU family (M1 and later). Already the project minimum
(`[g_device supportsFamily:MTLGPUFamilyApple7]` check in `flux_metal.m:448`).

**Files:** `flux_shaders.metal` — all reduction kernels

---

### BL-003 — GPU Bias Fusion (Remove CPU Roundtrip)
**Priority:** P2 — Medium Effort, Medium Impact
**Status:** Open

**Problem:**
After every GPU GEMM, bias is added on CPU (`flux_kernels.c:239-242`):
```c
for (int i = 0; i < seq; i++)
    for (int j = 0; j < out; j++)
        out_ptr[i*out + j] += bias[j];
```
This forces CPU-GPU sync → CPU write → next GPU op reads. Approximately
20 such roundtrips per denoising step (5 double blocks × 4 biased projections).

**Solution — Option A (preferred): MPSGraph bias epilogue**
Extend the existing `g_linear_graph_cache` MPSGraph to include bias as a
second input tensor:
```objc
MPSGraphTensor *biasT = [graph placeholderWithShape:@[@(out_dim)] ...];
MPSGraphTensor *result = [graph additionWithPrimaryTensor:mmOut
                                          secondaryTensor:biasT name:nil];
```

**Solution — Option B: Trivial broadcast kernel**
```metal
kernel void add_bias(device float *x, device const float *bias,
                     constant int &cols, uint2 pos [[thread_position_in_grid]]) {
    x[pos.y * cols + pos.x] += bias[pos.x];
}
// Dispatch: MTLSizeMake(out_dim, seq, 1)
```

**Files:**
- `flux_kernels.c:215-250` — `flux_linear()`
- `flux_metal.m` — extend MPSGraph linear cache

---

### BL-004 — simdgroup_matrix for Custom GEMM Tiles (M3+)
**Priority:** P3 — High Effort, Medium-High Impact (M3+ only)
**Status:** Open

**Problem:**
`batched_matmul_half_qkt` and `batched_matmul_half_sv` (`flux_shaders.metal:768-891`)
use a hand-written 16×16 shared-memory tile with manual FMA loop. M3+ (Apple9+)
has hardware 8×8 simdgroup matrix units that are not being used.

**Solution:**
Replace inner tile with `simdgroup_matrix` operations:
```metal
#if __METAL_VERSION__ >= 300
simdgroup_half8x8 a, b;
simdgroup_float8x8 c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
simdgroup_load(a, A_ptr, A_stride, ulong2(0,0));
simdgroup_load(b, B_ptr, B_stride, ulong2(0,0), true); // transposed
simdgroup_multiply_accumulate(c, a, b, c);
simdgroup_store(c, out_ptr, out_stride, ulong2(0,0));
#else
// existing 16x16 tile fallback
#endif
```

**Requirement:** Metal 3 / `__METAL_VERSION__ >= 300` / Apple9 GPU family (M3+).
Add runtime check: `[g_device supportsFamily:MTLGPUFamilyApple9]`.

**Files:** `flux_shaders.metal:753-891`, `flux_metal.m` (dispatch logic)

---

### BL-005 — Native `bfloat` MSL Type
**Priority:** P3 — Low Effort, Low-Medium Impact (M3+ only)
**Status:** Open

**Problem:**
BF16 ops use manual `ushort` bit manipulation with hand-rolled round-to-nearest-even
(`flux_shaders.metal:960-1020`). Metal 3+ supports `bfloat` as a first-class type.

**Solution:**
Gate on `__METAL_VERSION__ >= 300`:
```metal
#if __METAL_VERSION__ >= 300
// Use native bfloat type — hardware conversion on M3+
bfloat bf = bfloat(f32_val);
float f32 = float(bf_val);
#else
// existing ushort path
#endif
```

**Files:** `flux_shaders.metal:960-1020` — `bf16_to_f32()` / `f32_to_bf16()`
and all BF16 kernels that call them.

---

### BL-006 — MTLResidencySet for Weight Buffers
**Priority:** P4 — Low Effort, Low Impact
**Status:** DONE (commit f73940e)

**Problem:**
512 cached weight MTLBuffers are not declared resident before command encoding.
GPU MMU may page-fault them on first access, causing sporadic first-step jitter.

**Solution:**
Build a `MTLResidencySet` at model-load time (macOS 15+):
```objc
if (@available(macOS 15.0, *)) {
    MTLResidencySetDescriptor *desc = [MTLResidencySetDescriptor new];
    desc.initialCapacity = g_weight_cache_count;
    id<MTLResidencySet> rs = [g_device newResidencySetWithDescriptor:desc];
    for (int i = 0; i < g_weight_cache_count; i++)
        [rs addAllocation:g_weight_cache[i].buffer];
    [rs commit];
    g_residency_set = rs;
    // At command buffer encode time:
    // [cmdBuf useResidencySet:g_residency_set];
}
```

**Files:** `flux_metal.m` — model load path, command buffer encoding

---

### BL-007 — SDPA Graph Cache Too Small (8 → 32) and Linear Graph Cache (32 → 64)
**Priority:** P2 — Low Effort, Significant Impact in Interactive Mode
**Status:** DONE (commit f73940e)

**Problem A — SDPA cache:**
`MAX_SDPA_GRAPH_CACHE = 8` (`flux_metal.m:45`). Each unique `(seq_q, seq_k,
num_heads, head_dim)` combination requires a separate compiled MPSGraph. In
interactive CLI mode (flux_cli.c), users run multiple prompts: each different prompt
length changes seq_k (txt_seq + img_seq), creating a new graph. With only 8 slots,
the oldest compiled graph is silently overwritten, causing full MPSGraph recompilation
(~50-100ms per eviction) on the next generation with those dimensions.

Flux double blocks call SDPA twice per block (img_q + txt_q) × 5 blocks = 10
SDPA calls per step, each potentially with different (seq_q, seq_k) pairs.
8 slots is exhausted immediately on first generation.

**Solution A:** Increase `MAX_SDPA_GRAPH_CACHE` from 8 to 32. At ~few KB overhead
per entry (pointers + ObjC objects), 32 entries costs ~1KB of C struct overhead
(the graphs themselves are already ARC-managed).

**File:** `flux_metal.m:45`

**Problem B — Linear cache:**
`MAX_LINEAR_GRAPH_CACHE = 32` (`flux_metal.m:67`). Flux 9B with CFG may exceed 32
unique (seq, in_dim, out_dim) combinations.

**Solution B:** Increase to 64.

**File:** `flux_metal.m:67`

---

### BL-008 — Qwen3 Causal Attention: No MPSGraph Fallback After seq=512
**Priority:** P2 — Medium Impact, Low-Medium Effort
**Status:** Open

**Problem:**
Both causal attention functions hard-return 0 for `seq > 512`:
- `flux_metal_causal_attention()` (`flux_metal.m:6046`)
- `flux_gpu_causal_attention_bf16()` (`flux_metal.m:5341`)

When this triggers, the caller falls back to CPU for the entire attention computation
across all 36 Qwen3 layers. Unlike the transformer non-causal SDPA path (which chains
custom kernel → MPSGraph native SDPA), there is no MPSGraph causal SDPA fallback.

With Qwen3's chat template (~15-20 overhead tokens), prompts of ~380+ words (≈500
tokens) hit this limit. All 36 layers then run full CPU causal attention.

**Additional issue:** `flux_metal_causal_attention()` (float32 path) allocates fresh
buffers and creates a new command buffer outside batch mode every call
(`[g_queue commandBuffer]` at line 6083), bypassing the batch system entirely.

**Solution:**
Add an MPSGraph causal SDPA path to `flux_gpu_causal_attention_bf16()` as fallback
when `seq > 512`. `scaledDotProductAttentionWithQueryTensor` supports causal masking
via an additive mask tensor (fill upper triangle with -inf). This is the same
approach MPSGraph uses for GPT-style autoregressive attention.

Also integrate the float32 causal path into the tensor/batch system instead of
using raw `[g_queue commandBuffer]`.

**Files:**
- `flux_metal.m:5328-5403` — `flux_gpu_causal_attention_bf16()`
- `flux_metal.m:6036-6130` — `flux_metal_causal_attention()` (float32)
- `flux_qwen3.c:360-370` — call site

---

## COMPLETED

- **BL-001**: Float32 attention path now uses MPSGraph native SDPA (Flash Attention) as primary route,
  mirroring the BF16 path. `get_sdpa_graph_cache_f32` + `flux_gpu_attention_mpsgraph_f32` added;
  `flux_gpu_attention_fused` tries MPSGraph first, falls back to custom kernel. Benefits base models only
  (4B-base, 9B-base run float32 attention). Separate `g_sdpa_f32_graph_cache[32]` cache.
- **VAE-GPU-ATTN**: Mid-block attention moved to GPU (commit b09a206). Eliminates CPU stall in
  vae_decode_gpu(). New: `vae_attn_transpose` kernel + `flux_gpu_transpose_cs()` wrapper +
  `attnblock_forward_gpu()`. CPU fallback preserved for batch>1 or allocation failure.
- **BL-007A**: `MAX_SDPA_GRAPH_CACHE` 8 → 32 (commit f73940e)
- **BL-007B**: `MAX_LINEAR_GRAPH_CACHE` 32 → 64 (commit f73940e)
- **BL-006**: `MTLResidencySet` for weight buffers, macOS 15+, lazy init in `get_cached_weight_buffer()`, applied to batch/tensor/chain command buffers (commit f73940e)

---

## Notes

- MLX v0.31.0 (latest as of 2026-03) is reference for Metal kernel patterns.
  Its kernels are MIT-licensed and available at:
  `github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels`
- ANE (Apple Neural Engine) is NOT accessible via raw Metal/MPS.
  Only via CoreML or MLX `mx.neural_engine` (M5+, macOS 26.2+).
- macOS 26.3 JACCL bandwidth improvement is for distributed multi-device;
  irrelevant for single-device inference.
- BL-001 is the highest-leverage single change. Start there.
