# Proxy VAE v3.18.0 → v3.19.0 Migration Guide

v3.19.0 hardens the proxy VAE introduced in v3.18.0: multi-metric confidence
scoring, automatic regression detection, quality modes, M1 Max optimisation,
model variants, a comprehensive evaluation suite, and campaign-level config.

This guide covers the breaking and behavioural changes. **If you never enabled
the proxy in v3.18.0 (the default), nothing changes — the real VAE is still
used until you explicitly opt in.**

---

## 1. Config schema changes (`v2_pipeline.yaml`)

**v3.18.0:**
```yaml
proxy_vae:
  proxy_path: null
  confidence_threshold: 0.75
```

**v3.19.0:**
```yaml
proxy_vae:
  proxy_path: null
  enabled: false                 # NEW — master switch (also requires proxy_path)
  default_mode: "balanced"       # NEW — speed | balanced | high_fidelity
  fallback_threshold: 0.75       # RENAMED from confidence_threshold
  regression_window:    100      # NEW
  regression_threshold: 0.60     # NEW
  campaigns: {}                  # NEW — per-campaign overrides
```

**Action required if you set `confidence_threshold` in v3.18.0:**
- Rename `confidence_threshold` → `fallback_threshold`.
- Add `enabled: true` (the proxy now requires both `proxy_path` AND `enabled`).

The old key is not read in v3.19.0 — a stale `confidence_threshold` is silently
ignored and the default 0.75 applies.

---

## 2. `ProxyVAE` API changes (`vae_distill/proxy.py`)

### Constructor

`confidence_threshold` → `fallback_threshold`, plus new parameters:

```python
# v3.18.0
ProxyVAE(student, channel_mean, channel_std, scaling_factor, shift_factor,
         confidence_threshold=0.75, teacher=None)

# v3.19.0
ProxyVAE(student, channel_mean, channel_std, scaling_factor, shift_factor,
         quality_mode="balanced",       # NEW
         fallback_threshold=0.75,        # renamed
         teacher=None,
         regression_window=100,          # NEW
         regression_threshold=0.60,      # NEW
         expected_decode_mse=None)       # NEW (for high_fidelity mode)
```

### `ProxyVAE.load()`

```python
# v3.18.0
ProxyVAE.load(ckpt_path, teacher=None, confidence_threshold=0.75, device_cfg=None)

# v3.19.0
ProxyVAE.load(ckpt_path, teacher=None, quality_mode="balanced",
              fallback_threshold=0.75, regression_window=100,
              regression_threshold=0.60)
```

`device_cfg` was unused and is removed.

### Confidence scoring

v3.18.0 used a single per-channel z-score outlier fraction. v3.19.0 uses:
- **balanced**: diagonal Mahalanobis distance (RMS of per-channel z-scores),
  mapped to confidence via `exp(-max(0, dist - 1))`.
- **high_fidelity**: balanced + a reconstruction check (decode one image per
  batch, compare MSE to `expected_decode_mse` baseline), taking the minimum.
- **speed**: no check; always returns the proxy output.

The score scale differs from v3.18.0, so a `fallback_threshold` tuned on the old
scoring needs re-tuning. Run `evaluate_vae_proxy.py` (see §5) and use the
`fallback_simulation` table to pick a threshold for your target fallback rate.

### Checkpoints

v3.18.0 proxy checkpoints load unchanged in v3.19.0. The new optional
`meta.expected_decode_mse` field is absent in old checkpoints, which means
high_fidelity mode falls back to balanced behaviour (no reconstruction check)
until the checkpoint is re-saved. Re-run training or re-save to populate it.

---

## 3. Model variants (`vae_distill/student.py`)

New `variant` config key selects a preset:

```yaml
student:
  variant: small     # ~3.4M params, maximum speed
  # variant: default # ~6.0M params (the v3.18.0 architecture)
  # variant: medium  # ~9.3M params, better fidelity
```

Explicit `channels` / `layers_per_block` / etc. still work and override the
preset. Omitting `variant` reproduces the v3.18.0 default architecture exactly,
so existing `vae_proxy_512px.yaml` configs are unaffected.

New helper functions: `build_student_small()`, `build_student_medium()`.

New `StudentEncoder` methods for M1 Max:
- `.to_bfloat16()` — cast Conv/Linear weights to bf16 (~2× conv speedup)
- `.make_compiled()` — return an `mx.compile`-wrapped forward pass
- `.quantize_attention(group_size, bits)` — INT-quantize attention Linears

---

## 4. precompute_all.py CLI

New flag:
```
--proxy-mode {speed,balanced,high_fidelity}   (default: balanced)
```
`--proxy-vae` and `--proxy-vae-threshold` are unchanged from v3.18.0.

The orchestrator now passes proxy args automatically when `proxy_vae.enabled`
and `proxy_vae.proxy_path` are set in the pipeline config, resolving per-campaign
overrides. No manual flag-passing is needed for orchestrated runs.

---

## 5. Evaluation suite (NEW)

`evaluate_vae_proxy.py` replaces the basic `eval_vae_proxy.py` from v3.18.0
(which is retained for backwards compatibility but superseded).

```bash
# Full evaluation with HTML report
python train/scripts/evaluate_vae_proxy.py \
    --proxy /Volumes/2TBSSD/checkpoints/vae_proxy/proxy_final.safetensors \
    --shards /Volumes/16TBCold/shards \
    --vae-cache /Volumes/16TBCold/precomputed/vae/current \
    --flux-model flux-klein-model \
    --n-images 2000 \
    --out  /Volumes/2TBSSD/proxy_vae_eval.json \
    --report /Volumes/2TBSSD/proxy_vae_eval.html
```

Writing `--out /Volumes/2TBSSD/proxy_vae_eval.json` lets `pipeline_doctor.py`
surface the proxy's quality status automatically (see §6).

Metrics: latent MSE, cosine similarity, per-channel mean/std, FFT magnitude
correlation, decoded PSNR (needs `--flux-model`), error distribution, and a
fallback-rate simulation across thresholds.

### Speed benchmark

`benchmark_vae_proxy.py` measures encode latency (ms/image) for the proxy
variants and, optionally, the real Flux VAE, then reports the speedup ratio.

```bash
# Proxy-only — safe to run during a live precompute (variants compared to each other)
python train/scripts/benchmark_vae_proxy.py --variants small,default,medium

# Full proxy-vs-teacher ratio — run ONLY when the GPU is idle (loads the teacher VAE)
python train/scripts/benchmark_vae_proxy.py \
    --with-teacher --flux-model flux-klein-model --variants default --batch 4
```

IMPORTANT: absolute ms/image is inflated under GPU contention, and proxy and
teacher are not under matched load unless both run in the same idle-GPU session.
The script prints a warning and the GPU-lock holder when contention is detected.
Trust the **speedup ratio** from a single idle-GPU run, not absolute latencies
measured while a flywheel/pipeline precompute is active.

### Downstream A/B (the definitive Tier-2 test)

`compare_downstream_quality.py` trains two short IP-Adapter runs that differ ONLY
in their VAE latents (real vs proxy), on identical shards/seed/steps, then compares
final `cond_gap` / `ref_gap`. This is the authoritative quality test — latent
metrics can pass while downstream training quality regresses.

```bash
python train/scripts/compare_downstream_quality.py \
    --proxy /Volumes/2TBSSD/checkpoints/vae_proxy/proxy_final.safetensors \
    --shards /Volumes/16TBCold/shards \
    --base-config train/configs/stage1_512px.yaml \
    --flux-model flux-klein-model \
    --n-shards 4 --steps 500 --seed 1234 \
    --workdir /Volumes/2TBSSD/proxy_ab \
    --out /Volumes/2TBSSD/proxy_ab/result.json
```

PASS = proxy-run final `cond_gap` is within `--tolerance` (default 5%) of the
real-run `cond_gap`. Refuses to run while the GPU lock is held (it launches
training) unless `--force` is passed — run it on an idle GPU.

---

## 6. pipeline_doctor.py

New `_check_proxy_vae` check reports:
- Misconfiguration (enabled but no path, or path missing)
- Missing evaluation report (INFO with the command to run)
- Failed quality gates from the latest eval (WARNING)
- Healthy status with headline metrics (INFO)

It reads the latest report from `{DATA_ROOT}/proxy_vae_eval.json`. No action
needed — it stays silent when the proxy is not configured.

---

## 7. Recommended rollout

1. Train a proxy on accumulated latents:
   `python train/scripts/train_vae_proxy.py --config train/configs/vae_proxy_512px.yaml`
2. Evaluate and write the report (§5). Confirm cosine > 0.95, std_ratio in
   0.95–1.05, decoded PSNR > 35 dB.
3. Pick `fallback_threshold` from the `fallback_simulation` table for your target
   fallback rate (start ~0.80 for balanced).
4. Set `proxy_vae.proxy_path`, `enabled: true`, `default_mode: balanced` in the
   pipeline config.
5. Start with one campaign in `speed` mode (warmup, where data quality is known)
   and reserve `high_fidelity` for fine-art / OOD campaigns.
6. Watch `pipeline_doctor.py --ai` for proxy health and fallback rate after the
   first precompute run.
