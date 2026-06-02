# PRECOMP-2: Proxy VAE Design — Deep Dive

**Status:** Pre-implementation design. Corpus prerequisite (VAE latents from iter 10,
~200K images) completes in ~13h. Implementation estimate 3–5 days.

**Problem:** The Flux VAE encoder is the dominant precompute bottleneck — ~185ms/image
at batch=4 on M1 Max MPS. With 5,000 images/shard and 40 shards/iter, VAE alone
accounts for ~925s/shard, gating the entire flywheel iteration at ~20h overhead per
training run.

**Goal:** A small proxy model that approximates the Flux VAE encoder at ~20ms/image
(≥9× speedup), used only at precompute time. Inference always uses the real VAE.

---

## Confidence Assessment

**Moderate confidence, not high.**

The concept is sound and the failure cost is low (fall back to real VAE, iterate). The
specific backlog spec — EfficientNet-B0 + MSE loss — has real problems that make a
first attempt likely to fall short of the LPIPS < 0.04 target without iteration.

### What makes it likely to work

The use case is tolerant. The proxy only produces precomputed training data; IP-Adapter
training averages gradients across thousands of batches, so per-sample noise in latents
doesn't destroy the learning signal. A proxy that is deterministic and consistent (same
image always → same proxy latent) gives a coherent training signal even if the absolute
latent values differ from the real VAE. The training process will adapt to the proxy
latent distribution — the adapter learns the right input→output mapping regardless.

The regression task is also well-scoped: [3, 512, 512] → [32, 64, 64] at fixed stride
8 is a standard dense prediction problem. 6M params is reasonable for it.

### What makes it risky

**MSE in latent space is a weak objective.** The 32 VAE channels carry structured spatial
information with specific semantics and channel correlation. MSE minimization produces
outputs that match channel statistics on average but smear high-frequency spatial
content. The proxy can achieve low mean-squared error while systematically losing
texture and edge structure that the Flux transformer depends on.

**The real criterion is not in the backlog.** LPIPS < 0.04 measures decoded image
reconstruction quality. The actual success criterion is: *does an IP-Adapter trained
on proxy latents produce outputs of comparable quality to one trained on real latents?*
This downstream training quality check is the definitive test and needs to be part of
the validation plan.

**Channel distribution mismatch.** Even if per-pixel error is low, if the proxy's
channel variance distribution differs from the real VAE's (e.g., proxy channels have
lower variance), the effective noise schedule during diffusion training shifts. The
model's learned noise level calibration assumes the real VAE's per-channel variance.

**Generalization risk.** Training on ~200K JourneyDB + LAION images from the flywheel
corpus covers a specific distribution. The proxy might underperform on out-of-distribution
inputs (different styles, unusual compositions) that appear in future training data.

---

## Architecture Options

### Option A: EfficientNet-B0 + 1×1 projection (backlog spec)

```
EfficientNet-B0 [stride-8 feature map: 40ch @ 64×64] → Conv1×1 → [32, 64, 64]
~5.3M params
```

**Problems:** EfficientNet MBConv blocks use inverted residuals designed for
classification — they deliberately compress spatial information into a compact
bottleneck for efficient classification. The depthwise separable convolutions are
optimised for speed, not spatial fidelity. Using the stride-8 feature map assumes
those features are spatially rich, but EfficientNet never needed them to be —
later classification layers process them further before the global pool. The 1×1
projection then has no mechanism to recover spatial structure that the backbone
discarded.

**Recommendation: do not use this for a spatial dense prediction task.**

---

### Option B: Task-specific stride-8 encoder CNN ★ Recommended first attempt

```
[3, 512, 512]
→ Conv(3→64, 3×3, stride=2, pad=1) + BN + ReLU   → [64, 256, 256]
→ Conv(64→128, 3×3, stride=2, pad=1) + BN + ReLU  → [128, 128, 128]
→ Conv(128→256, 3×3, stride=2, pad=1) + BN + ReLU → [256, 64, 64]
→ ResBlock(256→256) × 3                            → [256, 64, 64]
→ Conv(256→32, 1×1)                               → [32, 64, 64]
~6–8M params depending on ResBlock count
```

Each conv layer uses the full 3×3 spatial context. The three downsampling steps hit
stride 8 cleanly. Residual blocks at the final resolution let the network refine spatial
features before projection. GroupNorm rather than BatchNorm (avoids batch-size
sensitivity). This is architecturally similar to what fast semantic segmentation
backbones (FCN, DeepLab) use for the encoder portion.

**Pros:** Designed for the task, simple, fast, easy to debug, no classification-oriented
inductive bias. Straightforward MLX implementation.

**Cons:** No multi-scale context; stride-4 features are discarded after stride-8 stage.
May lose fine spatial detail that a skip-connection path would preserve.

---

### Option C: Lightweight FPN encoder

Extend Option B with a feature pyramid to recover fine spatial detail:

```
Encoder → features at [128ch @ 128×128] and [256ch @ 64×64]
Lateral: Conv(128→64, 1×1) → upsample → [64, 64, 64] is NOT needed (stride-4 not needed)
Refinement: concat stride-4 features downsampled to stride-8 into head
Head: Conv(320→32, 3×3) → [32, 64, 64]
```

The stride-4 feature map (128×128) contains higher-frequency detail. Downsampling it
and concatenating with the stride-8 features before the final projection gives the head
more information, similar to a UNet decoder step without a full decoder.

**When to use:** if Option B's spatial detail is insufficient (see validation tier 1 —
look for systematic blurring of high-frequency latent content).

---

### Option D: Shrunken Flux VAE encoder ★ Recommended second attempt

Take the Flux VAE encoder's architecture (conv blocks, residual blocks with GroupNorm,
same resolution stages) but reduce channel multipliers:

| Stage | Real VAE channels | Proxy channels |
|-------|-------------------|----------------|
| 128px | 256 | 64 |
| 64px  | 512 | 128 |
| 32px  | 512 | 256 |

This preserves the inductive bias of the real encoder (same spatial structure, same
normalisation, same residual pattern) while being ~10× smaller. The proxy is
structurally aligned with what it is trying to approximate — the latent space
it produces will likely have the correct channel covariance structure by construction.

**When to use:** if Option B fails due to channel distribution mismatch rather than
spatial quality issues. Requires porting the VAE encoder's architecture to Python
(it's currently in `iris_vae.c` / MLX).

---

### Option E: Full intermediate distillation

Run the real VAE encoder on the training corpus and extract activations at multiple
intermediate layers. Train the proxy to match both the final latent and intermediate
activations:

```
L = MSE(ẑ, z) + Σ_i λᵢ * MSE(proxy_feat_i, vae_feat_i)
```

The intermediate features provide a much richer training signal than the final latent
alone — the proxy learns to reproduce the internal representation, not just the output.

**When to use:** as a last resort if Options B and D both fail. Adds training complexity
and requires storing intermediate VAE activations (large). Not recommended for first
attempts.

---

## Loss Function Options

### Loss 1: Plain MSE (backlog spec) — insufficient alone

```python
L = mean((z_hat - z) ** 2)
```

Problems documented above. Do not use as the sole loss.

---

### Loss 2: Channel-normalised MSE ★ Baseline — always include

```python
sigma_c = std(z, dim=[0, 2, 3], keepdim=True)  # per-channel std over corpus
L = mean(((z_hat - z) / sigma_c) ** 2)
```

The 32 VAE channels have different variances. Plain MSE is dominated by
high-variance channels. Normalising by per-channel std gives each channel equal
weight in the loss and improves gradient flow to the lower-variance channels
that may carry critical structure. Compute σ_c once on a representative subset
of the training corpus and treat as a fixed constant.

---

### Loss 3: Channel-normalised MSE + decoded LPIPS ★ Recommended primary loss

```python
z_hat_decoded = vae_decode(z_hat)
z_decoded     = vae_decode(z)       # already computed — free
L = channel_norm_mse(z_hat, z) + λ * lpips(z_hat_decoded, z_decoded)
```

The VAE decoder is already loaded during precompute. Running it on the proxy's
output during training adds ~20ms/image forward overhead, but the LPIPS signal is
qualitatively better than any latent-space metric — it directly measures whether
the decoded images look the same. Start with λ=0.1 and tune.

Note: `z_decoded` is the image decoded from the real VAE latent. This is not the
original input image — it is the VAE's reconstruction of the input, which is what
the proxy is trying to match. Do not compare proxy-decoded images to original
input images; the real VAE itself introduces reconstruction error.

---

### Loss 4: Frequency-weighted MSE

Apply a spatial frequency weighting to penalise high-frequency errors more:

```python
z_hat_f = fft2(z_hat)
z_f     = fft2(z)
freq_weight = 1.0 + alpha * freq_magnitude  # higher frequencies weighted more
L = mean(freq_weight * |z_hat_f - z_f|^2)
```

Counteracts the MSE bias toward predicting the mean (which is smooth / low-frequency).
Can be combined with channel normalisation. `alpha=0.5` is a reasonable starting point.

Worth including if initial evaluations show the proxy is spatially blurry but
channel statistics match.

---

### Loss 5: Batch-level distribution matching

Add a per-batch distributional regulariser to ensure proxy latent channel statistics
match the real VAE's:

```python
L_dist = sum_c( (mean(z_hat_c) - mean(z_c))^2 + (std(z_hat_c) - std(z_c))^2 )
L = channel_norm_mse(z_hat, z) + λ_dist * L_dist
```

Cheap to compute, directly addresses the channel distribution mismatch risk. Use
λ_dist=0.01 to avoid it dominating. This is a regulariser, not a primary signal.

---

### Loss not recommended: adversarial

A GAN discriminator on latent vectors could enforce distributional matching across
all moments, not just mean/std. The training instability and complexity cost is not
justified here — the simpler distribution matching regulariser (Loss 5) achieves
the key goal without GAN training pathologies.

---

### Recommended loss combination

```python
L = channel_norm_mse(z_hat, z)
  + 0.10 * lpips(vae_decode(z_hat), vae_decode(z))
  + 0.01 * distribution_match(z_hat, z)
```

Start here. Add frequency weighting if visual inspection shows spatial blurring.

---

## Training Data

**Corpus:** VAE latents computed by the flywheel. After iter 10 completes (~13h),
~200K image-latent pairs will be on cold storage under `precomputed/vae/current/`.
The corresponding images are in the shard tars. 200K is 2× the minimum; use it all.

**Distribution concern:** JourneyDB is synthetic AI-generated images. The proxy
trained on a JourneyDB-heavy corpus may underperform on natural photographs. Ensure
the training corpus includes ≥30% LAION/COYO shards (natural images).

**Augmentation:**

- Horizontal flip: valid — flip image and correspondingly flip the spatial dimensions
  of the latent (`z[:, :, ::-1]`). Free 2× data.
- Vertical flip: valid by the same logic. Use cautiously (rare in natural images).
- **Do not use colour jitter** — colour is encoded in specific VAE channels and
  augmenting image colour without computing the corresponding real VAE latent would
  introduce corrupted training pairs. If colour augmentation is needed, re-run
  the real VAE on the augmented images.

**Train/validation split:** 90/10 by shard (not by image, to avoid data leakage).
Hold out 2 full shards (10K images) for all evaluations.

---

## Validation Methodology

### Tier 1: Latent quality (fast, run after every training checkpoint)

| Metric | Target | Failure signal |
|--------|--------|----------------|
| Decoded LPIPS | < 0.04 | Visual blurring or artefacts |
| Per-channel MSE (normalised) | < 0.01 | Channel structure mismatch |
| Per-channel mean error | < 0.001 | Systematic channel bias |
| Per-channel std ratio | 0.95–1.05 × real | Compression or inflation of variance |
| Decoded PSNR | > 35 dB | Gross reconstruction error |

The channel std ratio is the most important distribution metric. A proxy that
systematically produces lower-variance latents than the real VAE shifts the effective
SNR of the diffusion process — training on proxy latents would then diverge from
the model's noise schedule calibration.

### Tier 2: Downstream training quality (definitive, run once quality gate passed)

Train a 500-step IP-Adapter with real VAE latents and another 500-step run with proxy
latents. Compare:
- Held-out CLIP-I score on a fixed prompt+reference set
- Visual inspection of 10 generated images per run (same seeds)
- Training loss curves (should converge at similar rates)

If Tier 2 passes, the proxy is production-ready. If it fails despite Tier 1 passing,
the issue is channel structure (the per-image statistics look right but the spatial
semantics used by the transformer are wrong). In that case, upgrade to Option D
architecture or add intermediate distillation (Loss 4).

### Tier 3: Failure mode analysis

For the 100 held-out images where proxy latent MSE is highest, examine:
- Are failures clustered by image type (faces, text, low-light)?
- Are failures clustered by channel (some channels wrong, others fine)?
- Are failures spatial (edges/textures) or semantic (wrong colour/composition)?

This directs the next iteration: more training data from the failure distribution,
or a better architecture/loss.

---

## Alternative: Subsampled Precompute (No Proxy)

Before committing to PRECOMP-2, consider a simpler alternative: do not precompute
all 5,000 images per shard — subsample to 1,000.

- Precompute time: 1,000 images/shard × 30 min/shard × 1,000/5,000 = 6 min/shard
  → 40 shards × 6 min = 4h/iter (vs current 20h, vs proxy VAE target of ~3.3h)
- Training quality impact: unknown — depends on whether 1,000 images/shard provides
  sufficient diversity for 1,000 training steps. With batch=1 and 1,000 steps, you
  consume the full 1,000-image subsample once. Probably fine.

This is worth benchmarking before building the proxy. Implement as a
`--subsample-per-shard N` flag in `precompute_all.py`. If 1,000 images/shard
produces training quality within ~5% of 5,000 images/shard, the proxy becomes
less urgent.

---

## Integration Plan

**Precompute_all.py changes:**

```python
# New flag
parser.add_argument("--proxy-vae", default=None,
    help="Path to proxy VAE weights (.safetensors). Uses real VAE if not given.")

# In worker init
if args.proxy_vae:
    _W["vae_proxy"] = load_proxy_vae(args.proxy_vae)

# In _vae_gpu_encode
if _W.get("vae_proxy") is not None:
    return _proxy_vae_encode(batch_ids, batch_imgs, out_dir)
else:
    # existing fast batched path
```

The proxy runs as an MLX model (or exported weights loaded into MLX conv ops directly
— the architecture is simple enough). Keep the real VAE as fallback and for validation.

**Orchestrator change:** add `proxy_vae_path` to flywheel precompute config, passed
as `--proxy-vae` arg to `precompute_all.py`. No structural changes needed.

---

## Expected Impact

| Scenario | Precompute time / iter |
|----------|----------------------|
| Current (real VAE, no SigLIP cache) | ~20h |
| SigLIP cached after first run, real VAE | ~12h |
| Proxy VAE, SigLIP still running | ~9h |
| Proxy VAE + SigLIP cached | ~3.3h |
| Subsampled precompute (1K/shard) no proxy | ~4h |

The combination of proxy VAE + SigLIP caching (after the first full run per shard)
brings the per-iter overhead from 20h to ~3.3h, making the 15-iter warmup campaign
feasible in days rather than weeks.

---

## Recommended Implementation Order

1. **First: benchmark subsampled precompute** (1 day). Add `--subsample-per-shard`
   flag and run a 500-step comparison. If quality is within 5%, this alone might
   be sufficient and delays PRECOMP-2 until better hardware is available.

2. **If proxy needed: implement Option B** (simple encoder CNN) with the recommended
   loss (channel-norm MSE + decoded LPIPS + distribution matching). Train on the
   200K corpus from iter 10.

3. **Validate with Tier 1 then Tier 2.** Only promote to production after Tier 2 passes.

4. **If quality insufficient: upgrade to Option D** (shrunken Flux VAE encoder
   architecture), same loss function. This is more likely to produce the right
   channel statistics by construction.

5. **Document the proxy version in the precompute cache version hash** so that latents
   computed with the proxy are versioned separately from latents computed with the
   real VAE. The `encoder_config_subset` in `cache_manager.py` should include a
   `vae_proxy_version` key so mixed-proxy caches don't pollute real-VAE caches.
