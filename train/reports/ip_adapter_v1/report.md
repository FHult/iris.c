# IP-Adapter v1 Inference Report

**Date:** 2026-05-11  
**Checkpoint:** `best.safetensors` (EMA, chunk 4, step ~95 000)  
**Script:** `train/scripts/test_ip_adapter_inference.py`

---

## Setup

| Field | Value |
|---|---|
| Adapter type | `ip_adapter_klein` |
| Model target | `flux-klein-4b` |
| Quant | bfloat16 |
| Blocks | 25 (5 double-stream + 20 single-stream) |
| Hidden dim | 3072 |
| Image tokens | 128 |
| Perceiver heads | 16 |
| SigLIP | `google/siglip-so400m-patch14-384` |
| Prompt | "A pile of feathers depicted in a watercolour style, providing a simple and unadorned view with no intricate details." |
| Steps | 4 (distilled Flux Klein) |
| Seed | 42 |
| Resolution | 512 × 512 |
| Strength | 1.0 |

---

## ip_scale Diagnostics

| Stream | Blocks | Mean | Min | Max |
|---|---|---|---|---|
| Double-stream | 0–4 | 0.6973 | 0.4599 | 0.8446 |
| Single-stream | 5–24 | 0.5664 | 0.3113 | 0.6999 |

Per-block (double): `[0] 0.4599  [1] 0.8446  [2] 0.7350  [3] 0.7128  [4] 0.7341`

No anomalies (no NaN/Inf, no near-zero blocks, no saturation > 5.0).

---

## Token Norms

| | Mean | Std | Min | Max |
|---|---|---|---|---|
| Perceiver output (128 tokens, 3072d) | 47.35 | 1.65 | 42.52 | 50.74 |
| IP-K tokens (25 blocks × 128 tokens) | 501.97 | 88.60 | — | — |

IP-K norms are large (~500) relative to the Perceiver output (~47). This reflects the K projection weights having grown during training. The ip_scale values (~0.5–0.7) were learned alongside these norms and calibrate the effective injection magnitude at h_final.

---

## Timing

| | Time |
|---|---|
| Flux model load | 0.9s |
| Baseline (strength=0.0) | 10.2s |
| Adapter (strength=1.0) | 7.1s |

The adapter run is faster due to MLX graph compilation on the first (baseline) pass warming the kernel cache for the second.

---

## CLIP-I Score

**CLIP-I = 0.5291** (style image vs adapter output, SigLIP cosine similarity)

Above the 0.5 threshold. Moderate — consistent with the training approximation (IP summed at h_final rather than injected block-by-block; see TRAIN-6 in BACKLOG.md) and with training not fully converged at step ~95 000.

---

## Images

**Style reference** (`style_ref.jpg`):

![style ref](style_ref.jpg)

**Comparison — style ref | baseline | adapter** (`comparison.png`):

![comparison](comparison.png)

**Baseline only** (`baseline.png`) — Flux Klein 4B, no adapter, same seed:

![baseline](baseline.png)

**Adapter output** (`adapter.png`) — same seed with IP-Adapter at strength=1.0:

![adapter](adapter.png)

---

## Notes

- Inference uses the training-matching forward pass: `_flux_forward_no_ip` collects Q vectors and `h_final` from a clean Flux run; IP contributions are summed and added to `h_final` before `norm_out`/`proj_out`. This matches the training computation exactly.
- The block-by-block injection function `_flux_forward_with_ip` is NOT used — it was never part of the training graph and produces degenerate output with the current weights.
- To reach CLIP-I ~0.7–0.85, retrain with block-by-block injection (TRAIN-6).
