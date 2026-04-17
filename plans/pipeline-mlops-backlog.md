# Pipeline MLOps Backlog

Issues identified during pipeline audit (2026-04-01) that were not fixed in the initial pass.
Items are ordered by operational risk.

---

## MLX-10 — No model lineage in checkpoints ✅ DONE

**Problem:** Checkpoints contain only weights. There is no record of which config file,
data shards, git commit, or training args produced a given `.safetensors` file.
If you want to reproduce a checkpoint or understand why two checkpoints differ, you have no
reliable record beyond filename and wall-clock timestamp.

**Risk:** Medium. Silent config drift between chunks is invisible. Any ablation or
post-mortem is guesswork.

**Fix:** At checkpoint save time, write a sidecar `step_NNNNNN.json` with:
- git commit SHA (`git rev-parse HEAD`)
- config file path + full resolved YAML dict
- training args passed to the script
- timestamp, step, loss

Cost: negligible (one JSON write per checkpoint).

---

## MLX-11 — No pre-flight doctor check ✅ DONE

**Problem:** `run_training_pipeline.sh` starts the multi-day pipeline with no upfront
validation of environment, dependencies, or data layout.
Failures (missing Python packages, wrong MLX version, missing model weights, wrong shard
path) surface hours into the run, after non-resumable steps have already consumed time.

**Risk:** Medium. A single missing pip package can kill a 12-hour step.

**Fix:** Add a `doctor` subcommand (or run inline at startup) that checks:
- Python + MLX version meets minimum (`mlx >= 0.31.1`)
- Required Python packages importable (`turbojpeg`, `safetensors`, `yaml`, `wandb` if enabled)
- Model weight directory exists and contains expected files
- `DATA_ROOT` writable with ≥ 50 GB free
- Shard path non-empty if resuming past step 4
- Precompute cache non-empty if training is about to start
- `tmux` available if sessions are used

Exit with a clear per-check ✅/❌ table before doing anything destructive.

---

## MLX-12 — Hard example mining failure is silent in pipeline log ✅ DONE

**Problem:** If `mine_hard_examples.py` crashes, the pipeline logs a one-line warning and
continues. The Python traceback is swallowed. The operator has no way to diagnose the root
cause from the pipeline log alone.

**Status:** Partially fixed (fix 9 above now prints the re-run command and marks stderr).
Remaining gap: the Python traceback itself is still not surfaced because the process output
goes to the pipeline log only if it was captured with `tee`; bare `python ...` output
goes to stdout/stderr which is whatever the tmux session has.

**Fix:** Wrap the mining call in a subshell that captures both stdout+stderr and tees to a
dedicated log file `$DATA_ROOT/logs/mine_hard_chunk${CHUNK}.log`. On failure, print the
last 30 lines of that log to the pipeline log.

---

## MLX-13 — No alerting on training stall ✅ DONE

**Problem:** If training hangs (deadlock in prefetch thread, OOM kill, Metal timeout),
the heartbeat file goes stale. `pipeline_status.sh` already detects this (⚠️ stale
heartbeat) but only if you actively poll the status. There is no push notification.

**Risk:** Low–Medium. A stall costs wall-clock time proportional to how long until the
operator notices.

**Fix options (in order of effort):**
1. A cron job (every 5 min) that runs `pipeline_status.sh`, checks heartbeat age, and
   sends a macOS notification (`osascript -e 'display notification ...'`) if stale.
2. `wandb` alert (already supported in wandb; requires the wandb project to be configured).
3. A background watcher process launched by the pipeline that `kill -0` the training PID
   and sends a notification if it exits unexpectedly.

---

## MLX-14 — build_shards concurrent filter can miss final window ✅ DONE

**Problem:** The background `filter_shards` loop runs every 60 seconds. If
`build_shards` writes a shard in the last 59 seconds before it exits, that shard may not
be filtered until the explicit final pass. The final pass runs synchronously after
`build_shards` exits, so it will catch it — this is correct. But the log says
"background filter running" which implies continuous coverage, which is slightly misleading.

**Risk:** Low. The final pass is a hard guarantee; this is cosmetic.

**Fix:** Log the final-pass invocation as "Final filter pass (catching remaining shards)"
and suppress the "running" implication in background loop comments.

---

## MLX-22 — Post-training validation suite: `pipeline_validate.sh` (HIGH)

**Problem:** There is no automated way to verify that trained IP-Adapter weights are fit
for purpose before deploying them to the image app. The only current check is visual
inspection of training loss curves — which measures fitting, not style transfer quality.
`pipeline_validate.sh` is listed in DISPATCH.md as proposed but was never built.

**Goal:** A script that runs after each chunk completes and answers: "do these weights
actually transfer style correctly, and are they better than the previous chunk?"

---

### Components

#### V-01 — Weight integrity check (~10s)

Load the checkpoint safetensors, verify structure and numerics:

```python
# train/scripts/validate_weights.py --checkpoint path/to/step_NNNNN_ema.safetensors
import safetensors.numpy as st
import numpy as np

weights = st.load_file(checkpoint_path)
errors = []

# 1. Shape checks — compare against expected architecture from config
expected_shapes = {
    "image_proj.weight":    (hidden_dim, siglip_dim),   # Perceiver resampler
    "ip_scale":             (25,),                       # per-block scales
    # ... derive full list from train/ip_adapter/model.py IPAdapterKlein.__init__
}
for key, expected in expected_shapes.items():
    if key not in weights:
        errors.append(f"MISSING: {key}")
    elif weights[key].shape != expected:
        errors.append(f"SHAPE MISMATCH {key}: got {weights[key].shape}, expected {expected}")

# 2. Numerical health
for key, val in weights.items():
    if np.any(np.isnan(val)):  errors.append(f"NaN in {key}")
    if np.any(np.isinf(val)):  errors.append(f"Inf in {key}")

# 3. ip_scale sanity — values should be in [0.0, 3.0] after training
ip_scale = weights.get("ip_scale")
if ip_scale is not None:
    if ip_scale.min() < -0.5 or ip_scale.max() > 5.0:
        errors.append(f"ip_scale out of expected range: min={ip_scale.min():.3f} max={ip_scale.max():.3f}")
```

Emit: `{"v01_weight_integrity": {"ok": true/false, "errors": [...]}}`

---

#### V-02 — Style transfer smoke test (~3 min)

Run inference with each of the 5 (prompt, style_ref) pairs from
`train/configs/eval_prompts.txt` using the EMA checkpoint. Uses Python MLX inference —
does not require `--sref` to be implemented in iris.c.

```python
# train/scripts/run_inference.py  (new helper, called by validate script)
# Loads Flux + IP-Adapter weights in MLX, runs denoising loop with siglip features
# from the reference image, saves output PNG.

def run_sref_inference(flux, adapter, siglip, prompt, ref_image_path, seed, steps=4):
    siglip_feats = extract_siglip(siglip, ref_image_path)   # [1, 729, 1152]
    ip_embeds    = adapter.get_image_embeds(siglip_feats)    # [1, 128, 3072]
    k_ip, v_ip   = adapter.get_kv_all(ip_embeds)
    # ... standard Flux denoising loop with ip cross-attention injected
    return output_image
```

Output: `$CKPT_DIR/validation/step_NNNNN/` — one PNG per eval pair.
If any generation crashes or produces a blank/black image → FAIL.

---

#### V-03 — CLIP style similarity scoring (~2 min)

For each output image + its reference image, compute CLIP-I cosine similarity.
Uses `open_clip` which is already in the training venv.

```python
import open_clip, torch
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model = model.to('mps').eval()

def clip_image_similarity(img_a_path, img_b_path):
    with torch.no_grad():
        feat_a = model.encode_image(preprocess(Image.open(img_a_path)).unsqueeze(0).to('mps'))
        feat_b = model.encode_image(preprocess(Image.open(img_b_path)).unsqueeze(0).to('mps'))
        return float(torch.cosine_similarity(feat_a, feat_b).item())
```

Emit per eval pair: `clip_i_similarity` (expected range: 0.2–0.5 for good style transfer;
<0.15 suggests the adapter is not transferring style at all).

---

#### V-04 — CLIP prompt alignment scoring (~1 min)

For each output image + its text prompt, compute CLIP-T cosine similarity.
Ensures the model still follows the prompt despite style conditioning.

```python
def clip_text_similarity(img_path, prompt):
    tokens = open_clip.tokenize([prompt]).to('mps')
    with torch.no_grad():
        img_feat  = model.encode_image(preprocess(Image.open(img_path)).unsqueeze(0).to('mps'))
        txt_feat  = model.encode_text(tokens)
        return float(torch.cosine_similarity(img_feat, txt_feat).item())
```

Emit per eval pair: `clip_t_alignment` (expected range: 0.20–0.35; <0.15 suggests
the style conditioning is overriding the prompt entirely).

---

#### V-05 — No-adapter baseline delta (~3 min)

Generate the same 5 prompts without the IP-Adapter (ip_scale=0) to get a baseline.
Compute delta: `clip_i_with_adapter - clip_i_without_adapter`.

If delta ≤ 0 for all pairs, the adapter is not improving style transfer over the base
model — this is a strong signal of training failure or incorrect weight loading.

---

#### V-06 — EMA vs online weight comparison (~3 min)

Run inference with both `step_NNNNN_ema.safetensors` and `step_NNNNN.safetensors`
on the same eval pairs. Compare CLIP-I scores.

Expected: EMA scores ≥ online scores (EMA is the deployment weight; if online is
consistently better, the EMA decay may be too high and averaging away too much signal).

---

#### V-07 — Visual grid output

Save a structured grid image for human review:

```
┌──────────────────────────────────────────────────────────────┐
│  Checkpoint: step_0065000_ema   Scale: small   Chunk: 2      │
├───────────┬────────────────┬────────────────┬────────────────┤
│ Reference │ With adapter   │ No adapter     │ Prev chunk     │
├───────────┼────────────────┼────────────────┼────────────────┤
│ [sref 1]  │ [output 1]     │ [base 1]       │ [prev 1]       │
│           │ CLIP-I: 0.31   │ CLIP-I: 0.18   │ CLIP-I: 0.26   │
├───────────┼────────────────┼────────────────┼────────────────┤
│ [sref 2]  │ [output 2]     │ [base 2]       │ [prev 2]       │
...
```

Saved to: `$CKPT_DIR/validation/step_NNNNN/grid.png`

---

#### V-08 — Regression check against previous chunk checkpoint

Load the previous chunk's best EMA checkpoint, run the same eval pairs, compare
CLIP-I and CLIP-T scores. Emit: `{"regression": {"clip_i_delta": +0.04, "clip_t_delta": -0.01}}`.

Positive CLIP-I delta = style improved. Negative CLIP-T delta = some prompt following
sacrificed for style (acceptable within ±0.02; concerning if >0.05).

---

### Output format

All results written to `$CKPT_DIR/validation/step_NNNNN/results.json`:

```json
{
  "checkpoint": "step_0065000_ema.safetensors",
  "chunk": 2, "scale": "small",
  "timestamp": "2026-04-17T18:00:00Z",
  "v01_weight_integrity": { "ok": true, "errors": [] },
  "v02_smoke": { "ok": true, "generated": 5, "failed": 0 },
  "pairs": [
    {
      "prompt": "a portrait of a woman",
      "sref":   "eval_refs/watercolor_portrait.jpg",
      "output": "step_0065000/pair_0.png",
      "clip_i_with_adapter":    0.31,
      "clip_i_no_adapter":      0.17,
      "clip_i_delta":           0.14,
      "clip_t_alignment":       0.24,
      "clip_i_prev_chunk":      0.26
    }
    ...
  ],
  "summary": {
    "mean_clip_i":        0.29,
    "mean_clip_t":        0.23,
    "mean_adapter_delta": 0.11,
    "mean_chunk_delta":   0.03,
    "ema_vs_online_delta": 0.02,
    "verdict": "PASS"
  }
}
```

**Verdict logic:**
- `PASS`: mean_clip_i > 0.20 AND mean_adapter_delta > 0.05 AND no weight integrity errors
- `WARN`: mean_clip_i > 0.15 AND mean_adapter_delta > 0.0 (marginal style transfer)
- `FAIL`: any weight integrity error, OR mean_clip_i < 0.15, OR mean_adapter_delta ≤ 0

---

### Script interface

```bash
# Run full validation suite on latest EMA checkpoint:
bash train/scripts/pipeline_validate.sh

# Run on a specific checkpoint:
bash train/scripts/pipeline_validate.sh --checkpoint /path/to/step_65000_ema.safetensors

# Skip slow tiers (integrity + smoke only):
bash train/scripts/pipeline_validate.sh --fast

# Compare two specific checkpoints:
bash train/scripts/pipeline_validate.sh --compare step_50000_ema step_65000_ema
```

### Files to create

| File | Purpose |
|------|---------|
| `train/scripts/pipeline_validate.sh` | Orchestrator shell script |
| `train/scripts/validate_weights.py` | V-01 weight integrity |
| `train/scripts/run_inference.py` | V-02 Python MLX inference with IP-Adapter |
| `train/scripts/score_validation.py` | V-03/04/05/06/08 CLIP scoring + regression |
| `train/scripts/render_validation_grid.py` | V-07 grid image rendering |
| `train/configs/eval_prompts.txt` | Already exists — add more pairs (aim for 10) |
| `train/eval_refs/` | Reference images for eval pairs (populate manually, ~10 images) |

**Note on `run_inference.py`:** This is the Python-side inference path that enables
validation *before* `--sref` is implemented in iris.c (MLX-23). It reuses the forward
pass from `train_ip_adapter.py` but in no-grad eval mode. The key difference from
training: use `ip_scale` from the loaded weights unchanged (don't zero any blocks).

---

## MLX-23 — Implement `--sref` in iris.c binary (HIGH)

**Problem:** `--sref` is stubbed in `main.c:1273` with "not yet implemented (v2.6 target)".
The IP-Adapter weights are trained and the Python inference path works, but the shipped
binary cannot perform style-reference inference. This is the final step before the image
app can use the trained adapter.

---

### Architecture overview

The IP-Adapter adds cross-attention to 25 blocks of the Flux transformer:
- 5 double-stream blocks (indices controlled by `ip_scale[0:5]`)
- 20 single-stream blocks (indices controlled by `ip_scale[5:25]`)

At each injection point, the adapter injects additional K/V pairs derived from
SigLIP image features. The transformer's self-attention `Q` attends to both its own
`K/V` and the adapter `K/V` (weighted by `ip_scale[block_idx]`).

Adapter weight structure (from `train/ip_adapter/model.py`):
```
ip_adapter/
  image_proj.*          Perceiver resampler: siglip [B,729,1152] → ip_embeds [B,128,3072]
  to_k_ip_stacked       [25, 3072, head_dim×heads] = [25, 3072, 3072]  K projections
  to_v_ip_stacked       [25, 3072, head_dim×heads] = [25, 3072, 3072]  V projections
  ip_scale              [25]  per-block attention scale (trained, ~0.5–1.5)
```

---

### The SigLIP encoding problem

SigLIP-SO400M (400M params) is needed to extract `[B, 729, 1152]` features from a
reference image. Implementing a full ViT in C is disproportionate effort. The pragmatic
solution: **lazy Python extraction with a sidecar feature file**.

**User-facing API (unchanged):**
```bash
./iris -d flux-klein-4b --sref portrait.jpg --ip-adapter weights.safetensors -p "a cat" -o out.png
```

**Under the hood:**
1. iris.c checks for `portrait.jpg.sref` (pre-extracted feature file, binary float16 array)
2. If absent: executes `python3 train/scripts/extract_sref.py portrait.jpg portrait.jpg.sref`
3. Reads `portrait.jpg.sref` → `float16[729, 1152]` → promotes to float32 internally

`extract_sref.py` is a thin wrapper:
```python
# train/scripts/extract_sref.py  input.jpg  output.sref
import sys, numpy as np
from train.ip_adapter.utils import load_siglip, extract_features
feats = extract_features(sys.argv[1])   # [729, 1152] float32
feats.astype(np.float16).tofile(sys.argv[2])
```

**Caching:** `.sref` files are permanent sidecar files — computing SigLIP features once
per reference image (takes ~2s on M1 Max) is the right trade-off vs. re-extracting every
generation. Users can pre-extract with `--sref-precompute img1.jpg img2.jpg`.

**Multiple --sref flags (up to 4):** features are averaged:
```c
// Average features from multiple reference images
for (int i = 0; i < n_srefs; i++)
    add features[i] to accumulator
divide accumulator by n_srefs
```

---

### C implementation plan

#### Step 1 — New struct: `iris_ip_adapter_t` (in `iris.c` / new `iris_ip_adapter.c`)

```c
typedef struct {
    int     n_blocks;       // 25
    int     hidden;         // 3072 (4B) or 4096 (9B)
    int     ip_seq;         // 128  (resampler output tokens)
    int     siglip_seq;     // 729
    int     siglip_dim;     // 1152
    float  *ip_scale;       // [25]

    // Perceiver resampler weights
    float  *resampler_q;    // [ip_seq, hidden]
    float  *resampler_k;    // [siglip_dim, hidden]  (cross-attn over siglip)
    float  *resampler_v;    // [siglip_dim, hidden]
    float  *resampler_out;  // [hidden, hidden]
    float  *resampler_norm_q; // [hidden]
    float  *resampler_norm_k; // [hidden]
    float  *proj_w;         // [hidden, ip_seq×hidden] final projection

    // IP cross-attention projections (stacked over 25 blocks)
    float  *to_k_ip;        // [25, hidden, hidden]
    float  *to_v_ip;        // [25, hidden, hidden]
} iris_ip_adapter_t;
```

#### Step 2 — Weight loading: `iris_load_ip_adapter()`

Load from a safetensors file. Key name mapping:
```c
// safetensors key → struct field
"image_proj.latents"         → resampler_q (learned latent queries)
"image_proj.proj_in.weight"  → resampler_k / resampler_v
"image_proj.proj_out.weight" → proj_w
"to_k_ip_stacked"            → to_k_ip
"to_v_ip_stacked"            → to_v_ip
"ip_scale"                   → ip_scale
```

Weights are bfloat16 in the checkpoint — convert to float32 (or float16 for Metal path)
using the existing `iris_safetensors.c` loading infrastructure.

#### Step 3 — Perceiver resampler forward pass

The resampler maps `siglip_feats [729, 1152]` → `ip_embeds [128, 3072]`:

```
1. Linear project siglip_feats → K, V  (siglip_dim → hidden)
2. Learned queries Q [ip_seq, hidden]
3. RMSNorm Q and K separately
4. Multi-head cross-attention: Q attends to K/V
5. Linear output projection
Result: [128, 3072]
```

This is a single MHA forward pass — reuse `flux_attention()` from `iris_transformer_flux.c`.

#### Step 4 — IP K/V projection

From `ip_embeds [128, 3072]`, project to per-block K and V:
```c
// For each block b in 0..24:
// k_ip[b] = ip_embeds @ to_k_ip[b]   → [128, hidden]
// v_ip[b] = ip_embeds @ to_v_ip[b]   → [128, hidden]
```

Compute all 25 pairs once before the denoising loop (ip_embeds is timestep-independent).

#### Step 5 — Inject into transformer blocks

In `iris_transformer_flux.c`, each double/single block needs a new optional parameter:
`const float *k_ip, const float *v_ip, float ip_scale`.

In the attention computation, after computing the standard `K_self, V_self`, concatenate:
```c
// Attention over [self_seq + ip_seq] keys and values
K_full = concat(K_self [img_seq, hidden], K_ip [128, hidden])
V_full = concat(V_self [img_seq, hidden], V_ip [128, hidden])
// Q only attends over the first img_seq outputs (ip tokens are key/value only)
attn = softmax(Q @ K_full.T / sqrt(head_dim)) @ V_full
attn_ip_contribution = attn[:, img_seq:] @ V_ip  * ip_scale
output = attn[:, :img_seq] + ip_scale * attn_ip_contribution
```

**Note:** The ip_scale from the trained weights is applied per block. The `--sref-scale`
CLI flag (`0.0–1.0`) multiplies all ip_scale values as a global override:
`effective_ip_scale[b] = ip_scale[b] * sref_scale_cli`

#### Step 6 — CLI wiring (`main.c`)

```c
// Already stubbed:
fprintf(stderr, "      --sref PATH       Style reference image (up to 4 --sref flags)\n");
fprintf(stderr, "      --sref-scale N    Style influence 0.0-1.0 (default: 0.7)\n\n");

// Remove the "not yet implemented" error at line 1273.
// Add:
//   --ip-adapter PATH   IP-Adapter weights .safetensors
// Load order: after Flux model loads, call iris_load_ip_adapter(ctx, ip_adapter_path)
// Before generation: call iris_extract_sref_features(sref_paths, n_srefs) → siglip_feats
```

#### Step 7 — Metal (GPU) path

The IP cross-attention K/V projections and the resampler are small GEMMs (128×3072).
Use `flux_metal_sgemm_cached()` for `to_k_ip` / `to_v_ip` (static weights).
Use `flux_metal_sgemm()` for the resampler cross-attention (dynamic — siglip_feats change
per image). The K/V concatenation and extended attention can reuse the existing
`flux_gpu_attention_mpsgraph_f32()` path with extended sequence length.

---

### Implementation order

1. `extract_sref.py` — SigLIP feature extraction + `.sref` file format
2. `iris_ip_adapter.c/.h` — struct, weight loading, resampler forward
3. IP K/V precompute (called once before denoising loop)
4. Inject into double-stream blocks only first (5 blocks) — validate style transfer works
5. Extend to single-stream blocks (20 blocks)
6. Metal GPU path for resampler + K/V projections
7. CLI: `--sref`, `--ip-adapter`, `--sref-scale`, `--sref-precompute`
8. Integration with validation suite (MLX-22 V-07 binary path)

---

### Validation criteria (links to MLX-22)

Before merging, run `pipeline_validate.sh` and confirm:
- V-01: weight integrity PASS
- V-03: mean CLIP-I with iris binary ≥ Python inference mean CLIP-I ± 0.02
  (confirms C implementation matches Python reference)
- V-07: visual grid shows matching style character between Python and C outputs

---

## MLX-21 — Extended telemetry for performance and quality insights (MEDIUM)

**Problem:** Current telemetry (step, loss, lr, steps/s, ETA, disk) is enough to monitor
liveness but not enough to diagnose quality problems, predict failures, or make informed
data/architecture decisions. Ten specific gaps identified from past pipeline runs.

Each item below notes where to implement it, what it emits, and what decision it enables.

---

### T-01 — Gradient norm (TRAINING QUALITY, HIGH)

**Gap:** No visibility into gradient health. Training loss can look stable while gradients
are silently exploding or vanishing — only visible after a loss spike or NaN.

**Implement in:** `train_ip_adapter.py`, after `mx.eval(grads)`, before the optimizer step:
```python
grad_norm = float(mx.sqrt(sum(mx.sum(g**2) for g in mx.utils.tree_flatten(grads)[0])))
```
**Emit to:** heartbeat JSON as `"grad_norm": float` and `"grad_norm_smooth": float` (EMA).
**Threshold alert:** log `WARNING: grad_norm spike` if `grad_norm > 10 × grad_norm_smooth`.
**Decision enabled:** Catch instability before loss spikes; tune clipping threshold; validate
that chunk transitions (LR drops) don't destabilise gradients.

---

### T-02 — Per-bucket throughput and loss (TRAINING QUALITY + PERFORMANCE, HIGH)

**Gap:** We track overall steps/s but not per image-resolution bucket. If 512×768 takes 3×
longer than 512×512, bucket mix directly affects ETA accuracy and may indicate suboptimal
data sampling. Per-bucket loss reveals which resolutions the model struggles with most.

**Implement in:** `train_ip_adapter.py`, accumulate per-bucket step count, total time, total loss:
```python
bucket_key = f"{H}x{W}"
bucket_stats[bucket_key]["steps"] += 1
bucket_stats[bucket_key]["loss"]  += loss
bucket_stats[bucket_key]["time"]  += step_time
```
**Emit to:** heartbeat JSON as `"buckets": { "512x512": {"steps": N, "loss_avg": X, "secs_avg": X}, ... }`.
**Emit cadence:** update every step; write to heartbeat every `log_every` steps.
**Decision enabled:** Identify slow/hard buckets; adjust bucket sampling weights; fix ETA
estimates that are currently wrong if the active batch skews toward larger resolutions.

---

### T-03 — Memory pressure time-series (PERFORMANCE, HIGH)

**Gap:** The EMA OOM (chunk 1) was diagnosed reactively from crash logs. A running memory
sample in the heartbeat would make OOM prediction proactive and catch regressions if code
changes re-introduce lazy graph accumulation.

**Implement in:** `train_ip_adapter.py`, sample at each heartbeat write:
```python
import subprocess, json
vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
# parse "Pages wired down" and multiply by page size (16KB on Apple Silicon)
wired_gb = parse_vm_stat_wired(vm)
```
Or use `psutil.virtual_memory()` if psutil is already in the venv.
**Emit to:** heartbeat JSON as `"mem_wired_gb": float`, `"mem_used_gb": float`.
**Threshold alert:** log `WARNING: memory pressure high` if `mem_wired_gb > 26` (leaving
6 GB headroom on 32 GB systems).
**Decision enabled:** Proactive OOM prevention; validate that EMA fix holds across chunk
transitions; catch future regressions in memory-intensive code paths.

---

### T-04 — Per-dataset loss breakdown (TRAINING QUALITY, MEDIUM)

**Gap:** Training mixes 5 sources (LAION, COYO, JourneyDB, WikiArt, anchors, hard examples).
If one source dominates loss, it indicates either low data quality or insufficient
representation — but currently invisible.

**Implement in:** `train_ip_adapter.py` — each WebDataset sample already carries a `__source__`
or shard-name field. Accumulate loss per source tag:
```python
source = batch.get("source", "unknown")
source_loss[source] = source_loss.get(source, [])
source_loss[source].append(float(loss))
```
**Emit to:** heartbeat JSON as `"source_loss": { "journeydb": X, "laion": X, "wikiart": X, ... }`.
**Emit cadence:** rolling 100-step average per source, reset each log interval.
**Decision enabled:** Adjust per-source sampling weights; identify low-quality sources early;
validate that hard-example mixing (5%) is actually elevating loss for those records.

---

### T-05 — Validation loss on held-out set (TRAINING QUALITY, MEDIUM)

**Gap:** Training loss measures fitting, not generalisation. Loss can plateau or oscillate
while the model is actually improving (or degrading) on unseen data. This is the most
important quality signal missing from the current setup.

**Implement as:** A fixed held-out set of ~200 images (never in training shards), stored in
`$DATA_ROOT/validation/`. Curate once from LAION+WikiArt at diverse styles.
Run a no-grad forward pass on the held-out set every `val_every=1000` steps:
```python
if step % cfg["val_every"] == 0:
    val_loss = compute_val_loss(model, val_loader)
    heartbeat["val_loss"] = float(val_loss)
```
**Emit to:** heartbeat JSON as `"val_loss": float`. Also write to a separate
`$CKPT_DIR/val_loss.jsonl` (one line per eval: `{"step": N, "val_loss": X}`).
**Note:** Requires upfront curation of validation set (~1h one-time effort).
**Decision enabled:** Detect overfitting at chunk boundaries; choose best checkpoint by
val_loss rather than latest step; validate that LR decay schedule is working.

---

### T-06 — EMA vs. online weight divergence (TRAINING QUALITY, MEDIUM)

**Gap:** EMA weights are the deployment weights but we only monitor online weight loss.
Large divergence between EMA and online weights can signal training instability or a
misconfigured EMA decay. Currently invisible.

**Implement in:** `train_ip_adapter.py`, sample one layer's L2 distance every N steps:
```python
if step % 500 == 0:
    # compare one representative layer (e.g. first IP projection weight)
    online_w = adapter.ip_proj[0].weight
    ema_w    = ema_params["ip_proj"][0]["weight"]
    ema_drift = float(mx.mean((online_w - ema_w)**2)**0.5)
```
**Emit to:** heartbeat JSON as `"ema_drift": float`.
**Decision enabled:** Detect EMA misconfiguration; flag if decay is too high (weights drift
apart) or too low (EMA tracks too closely, losing smoothing benefit).

---

### T-07 — Data loader wait time (PERFORMANCE, MEDIUM)

**Gap:** We don't know if the GPU ever waits for data. If the data pipeline is the
bottleneck, adding prefetch workers would improve steps/s. If the GPU is always the
bottleneck, data loader tuning is wasted effort.

**Implement in:** `train_ip_adapter.py`, time the `next(data_iter)` call separately:
```python
t0 = time.perf_counter()
batch = next(data_iter)
loader_wait_ms = (time.perf_counter() - t0) * 1000

t1 = time.perf_counter()
# ... forward + backward ...
compute_ms = (time.perf_counter() - t1) * 1000
```
**Emit to:** heartbeat JSON as `"loader_wait_ms": float`, `"compute_ms": float`,
`"loader_pct": float` (loader_wait / total step time × 100).
**Decision enabled:** If `loader_pct > 20%`, investigate prefetch depth or worker count.
If `loader_pct < 5%`, data pipeline is not the bottleneck — focus elsewhere.

---

### T-08 — Filter rejection rate per source (DATA QUALITY, MEDIUM)

**Gap:** `filter_shards.py` logs total kept/dropped but not per source. JourneyDB, LAION,
COYO, and WikiArt may have very different rejection rates. If chunk 3 JourneyDB rejects
40% vs. 10% for chunk 2, the shard budget assumptions (shards needed per chunk) are wrong.

**Implement in:** `filter_shards.py` — already tracks kept/dropped per shard. Add source
tag parsing (from shard filename prefix or `__source__` field) and accumulate per source.
**Emit to:** a `$DATA_ROOT/logs/filter_stats_chunk{N}.json` after filter completes:
```json
{ "journeydb": {"kept": 42000, "dropped": 3100, "rejection_pct": 6.9},
  "laion":      {"kept": 91000, "dropped": 8200, "rejection_pct": 8.3} }
```
**Decision enabled:** Identify low-quality sources; adjust download budget for future chunks
(if JourneyDB rejects 30%, download 43% more tgzs to hit the target shard count).

---

### T-09 — Caption length distribution (DATA QUALITY, LOW)

**Gap:** Recaptioning short captions (<10 words) is backlogged (see `pipeline_recaption.sh`
in DISPATCH.md). Without knowing the distribution, we can't quantify how much training
signal is degraded by short captions or prioritise the recaption effort.

**Implement in:** `filter_shards.py` or as a one-shot `pipeline_doctor.sh` check.
Sample 10% of records, compute caption word count distribution.
**Emit to:** `$DATA_ROOT/logs/caption_stats_chunk{N}.json`:
```json
{ "median_words": 18, "pct_under_10": 0.31, "pct_under_5": 0.12,
  "histogram": [[0,5,8.2], [5,10,22.9], [10,20,41.3], [20,50,24.1], [50,999,3.5]] }
```
**Decision enabled:** Quantify recaption ROI before investing 2 days of compute;
if `pct_under_10 < 10%` the effort may not be worth it.

---

### T-10 — SigLIP coverage per batch (TRAINING QUALITY, LOW)

**Gap:** `pipeline_start.sh` warns if SigLIP cache coverage < 95% at startup but does not
track per-step how many batches actually ran with zero SigLIP conditioning vs. real features.
If 20% of batches silently use zero-vector conditioning, image style transfer quality
degrades for those steps — currently invisible.

**Implement in:** `train_ip_adapter.py` — the dataset loader already falls back to
`mx.zeros` when SigLIP cache misses. Count those fallbacks:
```python
siglip_miss_count += int(siglip_was_zero)
```
**Emit to:** heartbeat JSON as `"siglip_coverage_pct": float` (rolling 100-step window).
**Threshold alert:** log `WARNING: siglip coverage low` if coverage drops below 90%.
**Decision enabled:** Confirm SigLIP conditioning is working at the batch level, not just
at startup; detect mid-run cache corruption or shard/cache ID misalignment.

---

### Summary table

| ID | Name | Category | Priority | Implement in |
|----|------|----------|----------|--------------|
| T-01 | Gradient norm | Quality | HIGH | `train_ip_adapter.py` |
| T-02 | Per-bucket throughput + loss | Quality + Perf | HIGH | `train_ip_adapter.py` |
| T-03 | Memory pressure time-series | Performance | HIGH | `train_ip_adapter.py` |
| T-04 | Per-dataset loss breakdown | Quality | MEDIUM | `train_ip_adapter.py` |
| T-05 | Validation loss on held-out set | Quality | MEDIUM | `train_ip_adapter.py` + curation |
| T-06 | EMA vs. online weight divergence | Quality | MEDIUM | `train_ip_adapter.py` |
| T-07 | Data loader wait time | Performance | MEDIUM | `train_ip_adapter.py` |
| T-08 | Filter rejection rate per source | Data quality | MEDIUM | `filter_shards.py` |
| T-09 | Caption length distribution | Data quality | LOW | `filter_shards.py` / doctor |
| T-10 | SigLIP coverage per batch | Quality | LOW | `train_ip_adapter.py` |

T-01, T-02, T-03 are the highest leverage: implement these first as they feed directly
into decisions about training stability, ETA accuracy, and OOM prevention.
T-05 requires upfront curation of a held-out set (~1h one-time effort) before implementation.

---

## MLX-20 — Pipeline state file: single source of truth for current run (HIGH)

**Problem:** Every new Claude session must reverse-engineer the current run state from
logs, config files, checkpoint directories, and heartbeat JSON. This wastes time and
causes errors — e.g. reporting chunk 1 when chunk 2 is actually running, wrong scale,
wrong ETA. There is no single file that answers: "what is running, at what scale, on
which chunk, with what config, and what comes next?"

The existing `$DATA_ROOT/logs/pipeline_manifest.json` only records step completion events
(a history log), not the parameters of the current run.

**Fix:** Introduce `$DATA_ROOT/pipeline_state.json` — a single authoritative file written
and maintained by the pipeline scripts. Claude reads this file first in every session.

**Schema:**

```json
{
  "schema_version": 1,
  "last_updated": "2026-04-17T16:45:00Z",
  "updated_by": "pipeline_start.sh",

  "current_run": {
    "chunk": 2,
    "scale": "small",
    "steps": 65000,
    "lr": "3e-5",
    "config": "train/configs/stage1_512px.yaml",
    "checkpoint_dir": "/Volumes/2TBSSD/checkpoints/stage1",
    "latest_checkpoint": "step_0059000",
    "data_root": "/Volumes/2TBSSD",
    "shard_dir": "/Volumes/2TBSSD/shards",
    "precompute_dir": "/Volumes/2TBSSD/precomputed",
    "started_at": "2026-04-10T10:26:27Z",
    "training_pid": 12345,
    "notes": "resumed after OOM fix; EMA eval() added at step 0"
  },

  "next_chunk": {
    "chunk": 3,
    "scale": "small",
    "tgz_range": [100, 120],
    "download_done": false,
    "convert_done": false,
    "build_shards_done": false,
    "filter_shards_done": false,
    "clip_embed_done": false,
    "precompute_done": false
  },

  "history": [
    {
      "chunk": 1,
      "scale": "small",
      "steps": 50000,
      "lr": "1e-4",
      "final_checkpoint": "/Volumes/2TBSSD/checkpoints/stage1/step_0050000.safetensors",
      "completed_at": "2026-04-09T22:00:00Z"
    }
  ]
}
```

**Who writes it:**

| Event | Writer | Fields updated |
|-------|--------|----------------|
| `pipeline_start.sh` launches | `pipeline_start.sh` | `current_run.*`, `last_updated` |
| Training completes | `run_training_pipeline.sh` | `current_run.latest_checkpoint`, `history[]` |
| Checkpoint written | `train_ip_adapter.py` (optional, via heartbeat) | `current_run.latest_checkpoint` |
| Background data prep step done | prep script | `next_chunk.{step}_done` |
| Chunk N complete, chunk N+1 starts | `pipeline_start.sh` | rotate `next_chunk` → `current_run`, add to `history` |

**Stale detection:** `pipeline_status.sh --json` compares `last_updated` against
heartbeat timestamp. If `last_updated` is >6h older than the heartbeat, emit:
`"state_file_stale": true` in JSON output — signals the file was not updated by the
script that launched training.

**`notes` field:** Free-text string for human/Claude annotations. Invaluable for
cross-session context about why things are configured a certain way, what was tried, etc.
Example: `"skipped CLIP dedup — only 7 shards needed for small, dedup not worth GPU cost"`.
Updated manually or by scripts that know something notable happened.

**`pipeline_status.sh --json` integration:** Embed `pipeline_state.json` verbatim under
a `"state"` key so a single `pipeline_status.sh --json` call returns both live telemetry
(heartbeat, step count, disk) and the static run parameters (chunk, scale, config).

**DISPATCH.md update required:** Add as the first action in every new session:
> Read `$DATA_ROOT/pipeline_state.json` (or the `"state"` key from `pipeline_status.sh --json`).
> This file is the authoritative source of chunk, scale, LR, checkpoint dir, and next-chunk
> data prep status. Do not infer these values from logs or config files — the state file
> is always more current.

**Implementation order:**
1. Define schema (above) and write `write_pipeline_state()` bash function in a shared
   `pipeline_lib.sh` sourced by all scripts.
2. Call it from `pipeline_start.sh` at launch.
3. Call it from `run_training_pipeline.sh` at training-complete milestone.
4. Update `next_chunk.*` fields from prep scripts (or manually until MLX-18 orchestrator exists).
5. Add `"state"` key to `pipeline_status.sh --json` output.
6. Update DISPATCH.md.
7. Pre-populate the file for the current run (chunk 2, small) manually after implementation.

---

## MLX-19 — Producer-consumer download+conversion pipeline (LOW, refines MLX-17/18)

**Problem:** `run_chunk3_prep.sh` (and the equivalent logic in `run_training_pipeline.sh`)
downloads all N tgzs fully before starting conversion. Total wall clock = download_time +
convert_time. For 21 tgzs at ~16 GB each, conversion is ~10-20 min per tgz, so ~3-5h of
conversion work sits entirely outside the download window.

**Fix:** Producer-consumer pattern:
- **Downloader** (producer): calls `hf_hub_download()` once per tgz in sequence, writes
  `$DATA_ROOT/raw/journeydb/.tgz/{N:03d}.ready` sentinel after each file completes.
- **Converter** (consumer): polls for `.ready` sentinels every 30s, converts each new tgz
  immediately, deletes the raw tgz after writing WDS shards, writes `{N:03d}.converted`.
- **Main process**: waits for all `.converted` sentinels then exits.

**Why `hf_hub_download()` not `snapshot_download()`:** `snapshot_download` is all-or-nothing —
it returns only after all files are fetched. `hf_hub_download` downloads a single file and
returns immediately, enabling per-tgz signalling.

**Disk space benefit:** Raw tgz is deleted after conversion, so peak usage is ~1-2 tgzs
in-flight at a time (~32 GB) rather than all N tgzs simultaneously (~336 GB for 21 files).
This matters for chunk 4 (52 tgzs, ~832 GB if held in full).

**I/O contention:** HuggingFace downloads are network-bound and write to SSD at a fraction
of NVMe sequential write speed (~50-100 MB/s sustained vs. 5 GB/s NVMe peak). Converter
reads/writes at SSD speed. The two processes do not meaningfully compete.

**Wall clock saving:** Roughly `(N-1) × convert_time_per_tgz`. For 21 tgzs at 15 min each:
~5h saved. For chunk 4 (52 tgzs): ~12h saved.

**Resume behaviour:** Sentinels make the process fully resumable. On restart:
- Skip any tgz where `.converted` exists.
- Re-download any tgz where `.ready` exists but the actual `.tgz` file is absent
  (download interrupted mid-file — `hf_hub_download` handles this via ETag/cache).

**Implementation sketch:**

```python
# download_and_convert_chunk.py
import threading, time, os
from pathlib import Path
from huggingface_hub import hf_hub_download

def downloader(data_root, start_tgz, end_tgz, sentinel_dir):
    for i in range(start_tgz, end_tgz + 1):
        ready = sentinel_dir / f"{i:03d}.ready"
        converted = sentinel_dir / f"{i:03d}.converted"
        if converted.exists():
            continue
        if not ready.exists():
            print(f"[download] fetching {i:03d}.tgz ...", flush=True)
            hf_hub_download(
                repo_id="JourneyDB/JourneyDB",
                repo_type="dataset",
                filename=f"data/train/imgs/{i:03d}.tgz",
                local_dir=str(data_root / "raw" / "journeydb"),
            )
            ready.touch()

def converter(data_root, start_tgz, end_tgz, sentinel_dir, output_dir, poll=30):
    pending = set(range(start_tgz, end_tgz + 1))
    while pending:
        for i in sorted(pending):
            ready = sentinel_dir / f"{i:03d}.ready"
            converted = sentinel_dir / f"{i:03d}.converted"
            tgz_path = data_root / "raw" / "journeydb" / "data" / "train" / "imgs" / f"{i:03d}.tgz"
            if ready.exists() and not converted.exists():
                print(f"[convert] converting {i:03d}.tgz ...", flush=True)
                run_convert(data_root, output_dir, start_tgz=i, end_tgz=i, workers=1)
                tgz_path.unlink(missing_ok=True)   # free raw tgz immediately
                converted.touch()
            if converted.exists():
                pending.discard(i)
        if pending:
            time.sleep(poll)

sentinel_dir = Path(data_root) / "raw" / "journeydb" / ".tgz_state"
sentinel_dir.mkdir(parents=True, exist_ok=True)

dl = threading.Thread(target=downloader, args=(data_root, start, end, sentinel_dir), daemon=True)
cv = threading.Thread(target=converter,  args=(data_root, start, end, sentinel_dir, output_dir))
dl.start(); cv.start()
dl.join(); cv.join()
```

**Where to implement:** Replace the download+convert block in `run_training_pipeline.sh`
chunk 2/3/4 paths and in `run_chunk3_prep.sh` (or its successor under MLX-18 orchestrator)
with a call to `train/scripts/download_and_convert_chunk.py`.

**Also download annotation file first** (`train_anno_realease_repath.jsonl.tgz`) before
starting the producer loop — the converter needs it and it is small (~200 MB).

---

## MLX-17 — Full parallel data pipeline for next chunk during training (MEDIUM)

**Problem:** Between any two training chunks, the pipeline is fully serial:
chunk N training completes → download chunk N+1 tgzs → JDB conversion → build_shards →
filter_shards → precompute → chunk N+1 training starts.
With chunk 1 training taking ~3 days, all of these steps (except precompute) could be
running in the background and be fully complete before training ends, eliminating the
inter-chunk gap entirely.

**Steps that are safe to run alongside training (CPU/I/O only, no GPU, no significant RAM):**

| Step | Tool | RAM impact | I/O | Conflict risk |
|------|------|-----------|-----|---------------|
| Download tgzs | `curl`/`wget` | negligible | network | None |
| JDB conversion | `convert_journeydb.py --workers 1` | ~300MB | SSD write | Low |
| build_shards | `build_shards.py --workers 1` | ~500MB | SSD read+write | Low |
| filter_shards | `filter_shards.py` | ~200MB | SSD read+write | Low |

**Steps that are NOT safe to run alongside training:**

| Step | Reason |
|------|--------|
| CLIP dedup embed | Uses MPS GPU via open_clip — direct conflict with training |
| precompute qwen3/vae/siglip | Uses MPS GPU — direct conflict with training |

**Consequence for scheduling:** CLIP dedup embed and precompute are the only unavoidable
serial gap between chunk N training end and chunk N+1 training start. All other steps
can be eliminated from the critical path by running them in the background.

**Fix:** After chunk N training is launched, the pipeline should automatically check for
and start each preparatory step for chunk N+1:

```bash
# Pseudo-code for background pipeline after training launch
background_prepare_next_chunk() {
    local NEXT_CHUNK=$((CHUNK + 1))

    # 1. Download: launch in background if tgzs not yet present
    if ! next_chunk_tgzs_present "$NEXT_CHUNK"; then
        download_next_chunk "$NEXT_CHUNK" --limit-rate 5m &
        wait_for_download "$NEXT_CHUNK"   # wait before convert
    fi

    # 2. Convert: workers=1 to limit SSD contention
    if ! next_chunk_converted "$NEXT_CHUNK"; then
        python convert_journeydb.py \
            --start-tgz $(chunk_start_tgz $NEXT_CHUNK) \
            --end-tgz   $(chunk_end_tgz   $NEXT_CHUNK) \
            --workers 1
    fi

    # 3. Build shards: workers=1
    if ! next_chunk_shards_built "$NEXT_CHUNK"; then
        python train/scripts/build_shards.py --workers 1 \
            --output-dir "$DATA_ROOT/shards_chunk${NEXT_CHUNK}"
    fi

    # 4. Filter shards (quality only; skip CLIP dedup — needs GPU)
    if ! next_chunk_filtered "$NEXT_CHUNK"; then
        python train/scripts/filter_shards.py \
            --shard-dir "$DATA_ROOT/shards_chunk${NEXT_CHUNK}"
    fi

    echo "Chunk $NEXT_CHUNK data ready up to filter. CLIP dedup + precompute needed after training."
}
```

**Scheduling CLIP dedup:** Since embed requires MPS GPU, it must run either:
- Immediately after chunk N training ends (before precompute starts) — natural slot
- During a planned training pause (e.g., overnight gap)
- Skipped entirely for speed: CLIP dedup is a quality filter, not correctness-critical.
  Duplicate images dilute training signal but do not cause incorrect gradients.

**Sentinels needed:** Each background step should write a `.done` sentinel so the foreground
pipeline can gate on completion rather than re-running.

**Savings:** Removes download + conversion (~2-4h) + build_shards (~1-2h) + filter_shards (~30min)
from the critical path between chunks. At 4 chunks, this saves ~12-24h total wall-clock.

**Risk:** Low. Workers=1 caps CPU/I/O. All steps are idempotent. Sentinel files prevent
double-execution. Background jobs should log to dedicated files under `$DATA_ROOT/logs/`.

---

## MLX-18 — Pipeline orchestrator: concurrent multi-chunk execution (MEDIUM)

**Problem:** The current `run_training_pipeline.sh` is a monolithic sequential script.
It cannot safely run preparation for chunk N+1 while chunk N trains, because there is no
process supervisor — background `&` jobs are orphaned if the script dies, logs mix, and
there is no recovery model. MLX-17 identifies what to parallelise; this item specifies
the architecture for how.

---

### Design: dedicated orchestrator process in its own tmux window

The orchestrator is a new script `train/scripts/pipeline_orchestrator.sh`. It runs as a
polling state machine in a dedicated tmux window and owns the full multi-chunk lifecycle.
`pipeline_start.sh` launches it. `pipeline_stop.sh` can stop it. `pipeline_status.sh`
reads its sentinel files.

**Tmux session layout:**

```
tmux session: iris                    (already exists)
  window 0:  iris-orch    orchestrator loop (new — the conductor)
  window 1:  iris-train   GPU training      (already exists)
  window 2:  iris-prep    CPU/IO data prep  (new — one active step at a time)
  window 3:  iris-watchdog heartbeat monitor (already exists)
```

One GPU window (training), one CPU/IO window (data prep). Only one prep step runs at a
time to cap SSD I/O contention. The orchestrator serialises prep steps but runs them
concurrently with the training window.

---

### Sentinel file schema

All state lives in `$DATA_ROOT/pipeline/chunk{N}/`. The orchestrator is stateless — it
derives all state from these files on every poll cycle, so restarting it is always safe.

```
$DATA_ROOT/pipeline/
  chunk1/
    download.done         # JDB tgzs present (chunk 1 pre-existing, can be pre-touched)
    convert.done          # JDB → WebDataset conversion complete
    build_shards.done     # build_shards.py complete
    filter_shards.done    # filter_shards.py complete (quality filter only, no CLIP)
    clip_embed.done       # clip_dedup.py embed complete (GPU step)
    clip_index.done       # clip_dedup.py build-index complete (CPU)
    clip_dups.done        # clip_dedup.py find-dups complete (CPU)
    precompute.done       # precompute_all.py complete (GPU step)
    train.done            # training complete for this chunk
    train.ckpt            # single line: path to final checkpoint safetensors
    {step}.error          # written on failure; orchestrator pauses and notifies
  chunk2/
    ...same schema...
  chunk3/
    ...
  chunk4/
    ...
```

**Note:** Chunk 1 data is already complete. Before launching the orchestrator, pre-touch
the relevant sentinels to reflect reality:
```bash
mkdir -p $DATA_ROOT/pipeline/chunk1
touch $DATA_ROOT/pipeline/chunk1/download.done
touch $DATA_ROOT/pipeline/chunk1/convert.done
touch $DATA_ROOT/pipeline/chunk1/build_shards.done
touch $DATA_ROOT/pipeline/chunk1/filter_shards.done
# clip_embed/clip_index/clip_dups.done: only touch if CLIP dedup was completed
# precompute.done: touch if $DATA_ROOT/precomputed/.done exists
```

---

### JourneyDB chunk → tgz ranges (from run_training_pipeline.sh)

```
Chunk 1: tgz 000–049  lr=1e-4   steps=105000 (small=50000)
Chunk 2: tgz 050–099  lr=3e-5   steps=40000
Chunk 3: tgz 100–149  lr=1e-5   steps=40000
Chunk 4: tgz 150–201  lr=1e-5   steps=40000
```

Download size: ~800 GB per chunk (~16 GB per tgz × 50 tgzs).
These ranges are defined in `run_training_pipeline.sh` lines 42–50 and should not be
duplicated in the orchestrator — source that script's constants instead.

---

### GPU scheduling rule

Only one GPU consumer can run at a time. Priority: **training > precompute > clip_embed**.

```
GPU is FREE  ↔  iris-train tmux window does not exist
              AND $DATA_ROOT/pipeline/chunkN/train.done exists (or training not started)
```

CPU/IO steps (download, convert, build_shards, filter_shards, clip_index, clip_dups)
may always run — they do not touch MPS.

---

### State machine (per-chunk transitions)

```
State           Condition to advance              Action
─────────────────────────────────────────────────────────────────────────
WAITING         chunk N-1 train.done exists       begin download for chunk N
DOWNLOAD        download.done written             launch convert (workers=1)
CONVERT         convert.done written              launch build_shards (workers=1)
BUILD_SHARDS    build_shards.done written          launch filter_shards (workers=1)
FILTER_SHARDS   filter_shards.done written         → wait for GPU free
GPU_WAIT        GPU is free                       launch clip_embed (GPU)
  [optional: if --skip-clip-dedup: skip clip_embed/index/dups, go straight to precompute]
CLIP_EMBED      clip_embed.done written           launch clip_index (CPU, no GPU needed)
CLIP_INDEX      clip_index.done written           launch clip_dups (CPU)
CLIP_DUPS       clip_dups.done written            → wait for GPU free
PRECOMPUTE_WAIT GPU is free                       launch precompute_all.py (GPU)
PRECOMPUTE      precompute.done written           launch training for chunk N
TRAINING        train.done written; ckpt written  signal orchestrator: advance to chunk N+1
```

The orchestrator evaluates this state machine for both the current training chunk and the
next prep chunk on every poll cycle (every 60s).

---

### Orchestrator core loop (pseudocode)

```bash
POLL_INTERVAL=60
SKIP_CLIP_DEDUP=false   # set via --skip-clip-dedup flag

while true; do
    for CHUNK in 1 2 3 4; do
        STATE=$(get_chunk_state $CHUNK)          # derived from sentinel files
        GPU_FREE=$(gpu_is_free)                  # check iris-train tmux window

        case "$STATE" in
          WAITING)
            prev=$((CHUNK - 1))
            [[ -f "$SENTINEL/$prev/train.done" ]] && launch_download $CHUNK ;;

          DOWNLOAD)   check_window_done iris-prep "download" $CHUNK ;;
          CONVERT)    check_window_done iris-prep "convert"  $CHUNK ;;
          BUILD)      check_window_done iris-prep "build"    $CHUNK ;;
          FILTER)     check_window_done iris-prep "filter"   $CHUNK ;;

          GPU_WAIT_CLIP)
            $GPU_FREE && ! $SKIP_CLIP_DEDUP && launch_clip_embed $CHUNK
            $GPU_FREE &&   $SKIP_CLIP_DEDUP && touch_sentinel $CHUNK clip_embed
                                             && touch_sentinel $CHUNK clip_index
                                             && touch_sentinel $CHUNK clip_dups ;;

          CLIP_EMBED) check_window_done iris-prep "clip_embed" $CHUNK ;;
          CLIP_INDEX) check_window_done iris-prep "clip_index" $CHUNK ;;
          CLIP_DUPS)  check_window_done iris-prep "clip_dups"  $CHUNK ;;

          GPU_WAIT_PRECOMPUTE)
            $GPU_FREE && launch_precompute $CHUNK ;;

          PRECOMPUTE) check_window_done iris-prep "precompute" $CHUNK ;;

          READY_TO_TRAIN)
            PREV_CKPT=$(cat "$SENTINEL/$((CHUNK-1))/train.ckpt" 2>/dev/null || echo "")
            launch_training $CHUNK "$PREV_CKPT" ;;

          TRAINING)   check_training_done $CHUNK ;;
          DONE)       [[ $CHUNK -eq 4 ]] && { notify "All chunks complete"; exit 0; } ;;
        esac
    done
    sleep $POLL_INTERVAL
done
```

---

### Helper functions

**`gpu_is_free()`**
```bash
gpu_is_free() {
    ! tmux has-session -t iris:iris-train 2>/dev/null \
    || ! tmux list-windows -t iris -F '#{window_name}' 2>/dev/null | grep -q '^iris-train$'
}
```

**`launch_step NAME CMD LOG`** — creates/replaces the iris-prep window:
```bash
launch_step() {
    local name="$1" cmd="$2" log="$3"
    tmux new-window -t iris -n iris-prep \; \
         send-keys -t iris:iris-prep \
         "($cmd) >> '$log' 2>&1; echo EXIT_CODE=\$? >> '$log'" Enter
    log_orch "Launched $name → $log"
}
```

**`check_window_done STEP CHUNK`** — checks if iris-prep window exited; reads EXIT_CODE from log:
```bash
check_window_done() {
    local step="$1" chunk="$2"
    if ! tmux list-windows -t iris -F '#{window_name}' | grep -q '^iris-prep$'; then
        # window gone — check exit code from log
        local log="$DATA_ROOT/logs/orch_${step}_chunk${chunk}.log"
        if grep -q 'EXIT_CODE=0' "$log" 2>/dev/null; then
            touch "$SENTINEL/chunk${chunk}/${step}.done"
            log_orch "$step chunk $chunk: done"
        else
            touch "$SENTINEL/chunk${chunk}/${step}.error"
            notify_error "$step chunk $chunk FAILED — see $log"
        fi
    fi
}
```

**`notify_error MSG`** — macOS notification + log:
```bash
notify_error() {
    osascript -e "display notification \"$1\" with title \"iris pipeline ERROR\""
    log_orch "ERROR: $1"
}
```

---

### Worker command reference

Each step command, capped at workers=1 to limit SSD I/O during training:

| Step | Command |
|------|---------|
| Download | `python train/scripts/download_datasets.py --chunk $N --jdb-only` |
| Convert | `python train/scripts/build_shards.py --source journeydb --start-tgz $S --end-tgz $E --workers 1 --output $DATA_ROOT/raw/journeydb_wds_chunk$N` |
| Build shards | `python train/scripts/build_shards.py --workers 1 --output $SHARDS_DIR` |
| Filter shards | `python train/scripts/filter_shards.py --shard-dir $SHARDS_DIR --workers 1` |
| CLIP embed | `python train/scripts/clip_dedup.py embed --shards $SHARDS_DIR --embeddings $EMB_DIR` |
| CLIP index | `python train/scripts/clip_dedup.py build-index --embeddings $EMB_DIR --index $IDX` |
| CLIP find-dups | `python train/scripts/clip_dedup.py find-dups --index $IDX --out $DUPS_FILE` |
| Precompute | `python train/scripts/precompute_all.py --shard-dir $SHARDS_DIR --out-dir $PRECOMP_DIR` |
| Train | `bash train/scripts/pipeline_start.sh --chunk $N --resume $PREV_CKPT --scale $SCALE` |

*(Exact argument names must be verified against each script before implementation.)*

---

### Interface with existing scripts

- **`pipeline_start.sh`**: Add `--orchestrate` flag. When set, instead of running the
  pipeline inline, it creates the `iris-orch` tmux window and launches `pipeline_orchestrator.sh`
  with forwarded `--scale`, `--data-root`, `--skip-clip-dedup` flags.

- **`pipeline_stop.sh`**: Add logic to also kill `iris-orch` window and `iris-prep` window
  cleanly (SIGTERM, wait, then SIGKILL).

- **`pipeline_status.sh --json`**: Add `"orchestrator": { "running": true/false, "prep_chunk": N, "prep_step": "convert" }` to the JSON output by reading the sentinel files.

- **`DISPATCH.md`**: Add section describing orchestrator session layout and how to attach
  to individual windows.

---

### Failure model

- Any step failure: write `{step}.error` sentinel, send macOS notification, orchestrator
  pauses (stops advancing that chunk's state machine). Human intervention required.
- Orchestrator crash: restart `pipeline_orchestrator.sh` — it reads sentinel files and
  resumes from current state. No data is lost.
- Training crash: watchdog already handles notification. Orchestrator detects
  `iris-train` window gone without `train.done` → writes `train.error` → pauses.

---

### Implementation order

1. Define and document sentinel schema (above). Pre-touch chunk 1 sentinels.
2. Write `get_chunk_state()` — pure function, reads sentinel files, returns state string.
3. Write `launch_step()` and `check_window_done()` helpers.
4. Wire up state machine for chunk 1→2 transition only. Test with dry-run flag.
5. Extend to all 4 chunks.
6. Add `--orchestrate` to `pipeline_start.sh`.
7. Update `pipeline_status.sh --json` with orchestrator fields.
8. Update `DISPATCH.md`.

---

## MLX-16 — Front-run next chunk's JDB conversion during training (LOW)

**Problem:** Chunk N training and chunk N+1 data conversion are fully sequential.
After chunk 1 training completes (~3 days), the pipeline must wait for chunk 2 JDB
conversion before precompute and training can start. This is pure wall-clock waste —
conversion is CPU+I/O only with no GPU or RAM conflict with training.

**Observed:** Manually launching `convert_journeydb.py --workers 1` alongside chunk 1
training consumed ~200-300MB RAM (safe on 32GB) and ran without any impact on training
throughput (0.19 steps/s unchanged). Workers=1 preferred over 2 — training is the
timeline constraint anyway, so extra workers just add SSD I/O contention for no benefit.

**Fix:** After chunk 1 training is launched, automatically start the chunk 2 JDB
conversion in the background if the chunk 2 tgz files are already downloaded:
```bash
# After training starts, check if chunk 2 tgz are present
if [[ $(count_files "$JDB_IMGS_DIR" "0[5-9][0-9].tgz") -gt 0 ]]; then
    python convert_journeydb.py --start-tgz 50 --end-tgz 70 --workers 1 &
fi
```
Similarly, chunk 2 conversion could front-run chunk 3 precompute, etc.

**Savings:** Removes ~2-4h of serial conversion time per chunk from total wall-clock.
**Risk:** Low. Workers=2 cap keeps RAM impact negligible. Conversion is idempotent
(output tars are atomically renamed); a crash leaves no partial state.

---

## MLX-15 — Siglip precompute not in precompute_all done gate ✅ DONE

**Problem:** `precompute_all.py` writes `$PRECOMP_DIR/.done` when qwen3+vae are complete.
Siglip is optional and runs separately. The training script reads `siglip_cache_dir` from
config; if the cache exists but is incomplete (e.g., only 11/34 shards done when training
starts), some batches fall back to zero-feature mode silently.

**Risk:** Medium. Zero-feature batches produce incorrect gradients; the effect is diluted
by batch diversity but the training signal is degraded for those steps.

**Fix:** In `train_ip_adapter.py`, at startup when `siglip_cache_dir` is set, count the
cache files and compare against the shard count. Warn loudly (or assert) if coverage < 95%.
Optionally: add `--siglip-min-coverage 0.95` flag to fail fast.
