# iris.c — IP-Adapter Training

> **Setup, pipeline operation, run scales, and monitoring** are documented in the root [README.md](../README.md#ip-adapter-training-pipeline).
> **Operational reference** (sentinels, heartbeats, tmux windows, recovery procedures) is in [DISPATCH.md](DISPATCH.md).

---

## Directory layout

```
train/
  train_ip_adapter.py         Main training loop (MLX)
  eval.py                     Checkpoint eval: generate images + CLIP-I/T metrics + HTML report
  start_pipeline.sh           Start or resume the pipeline (use this)
  ip_adapter/
    model.py                  IPAdapterKlein + PerceiverResampler
    loss.py                   Flow-matching loss
    ema.py                    Exponential moving average
    dataset.py                Async prefetch loader, multi-resolution bucketing
  scripts/
    orchestrator.py           State machine driving all pipeline steps end-to-end
    pipeline_ctl.py           Operator interface: pause / resume / abort / retry / status
    pipeline_doctor.py        Deep diagnostic: cross-checks sentinels, heartbeats, logs
    pipeline_status.py        Live progress view: step, loss, ETA, log tails
    pipeline_setup.py         First-run wizard: validates env, creates dirs, generates config
    pipeline_lib.py           Shared primitives: state I/O, sentinels, heartbeats, tmux
    precompute_all.py         Qwen3 + VAE + SigLIP precompute (restart-safe)
    mine_hard_examples.py     Loss-ranked hard example extraction
    clip_dedup.py             CLIP embedding + FAISS deduplication
    validate_shards.py        Shard integrity scan before training
    validator.py              Post-chunk validation: weight check + CLIP-I scoring
    download_convert.py       JourneyDB tgz download + image extraction
    build_shards.py           WebDataset shard assembly
    filter_shards.py          Drop corrupt / small / bad-caption records
  configs/
    v2_pipeline.yaml          Production pipeline config (chunks=4, scale=large default)
    v2_pipeline_smoke.yaml    Smoke test config (1 chunk, 100 steps, all quality features)
    v2_pipeline_dev.yaml      Dev config (1 chunk, 200 steps, no quality features)
    stage1_512px.yaml         Training hyperparameters (lr, batch, warmup, log_every, …)
    stage2_768px.yaml         Stage 2 config (768 px — future)
    eval_prompts.txt          Fixed (prompt, style_ref) pairs for checkpoint eval
```

---

## Model architecture

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| Flux Klein 4B | 4 B | Frozen |
| Qwen3 4B (Q4) | 2 GB | Frozen |
| VAE encoder | 0.3 GB | Frozen |
| SigLIP SO400M | 400 M | Frozen |
| PerceiverResampler | ~50 M | Yes |
| 25 × to_k_ip [3072, 3072] | ~236 M | Yes |
| 25 × to_v_ip [3072, 3072] | ~236 M | Yes |
| 25 × ip_scale (per block) | 25 | Yes |
| **Adapter total** | **~522 M** | **Yes** |

IP injection spans all 25 transformer blocks (5 double-stream + 20 single-stream). The PerceiverResampler compresses SigLIP SO400M features (729 tokens × 1152 dims) to 16 latent tokens before projection into each block's K/V space.

---

## Memory budget (M1 Max 32 GB)

| Component | Memory |
|-----------|--------|
| Flux Klein 4B (BF16, frozen) | 8.0 GB |
| Qwen3 4B (Q4, frozen) | 2.0 GB |
| VAE encoder (frozen) | 0.3 GB |
| SigLIP SO400M (BF16, frozen) | 0.8 GB |
| IP-Adapter (BF16, trainable) | 1.0 GB |
| AdamW optimizer state (FP32) | 2.0 GB |
| Activations at 512 px, batch=2 | 3.0 GB |
| OS + framework overhead | 2.0 GB |
| **Total** | **~19 GB** |

---

## Training observability

Two metrics distinguish "adapter is learning" from "adapter is doing nothing":

**`loss_cond` vs `loss_null`** — split loss logged every `log_every` steps. A zero gap after step ~1000 means the IP conditioning isn't contributing. Gap should grow positive (conditioned loss < null loss) as the adapter learns.

**`ip_scale` per block group** — 25 learnable scale scalars. Double-stream blocks (indices 0–4) control content, single-stream (5–24) control appearance. Healthy range 0.3–1.0; >2.0 risks content leakage; <0.05 means the block is inactive.

Both metrics surface in `pipeline_status.py` and `pipeline_doctor.py --ai`.

---

## Style Loss (optional)

An optional Gram matrix style loss can be added on top of the flow-matching objective to push the adapter toward capturing style statistics rather than content.

**How it works:** at each conditioned training step the predicted clean latent is reconstructed from the velocity prediction:

```
x0_pred = alpha_t * x_t - sigma_t * v_pred
```

The Gram matrix (channel cross-correlation) of `x0_pred` is compared to the Gram matrix of the ground-truth latent `x0_ref`. The MSE between them is added to the flow-matching loss:

```
total_loss = flow_matching_loss + style_loss_weight * gram_mse(x0_pred, x0_ref)
```

**Configuration** (`stage1_512px.yaml`):

```yaml
training:
  style_loss_weight: 0.0   # disabled by default; try 0.05–0.2
  style_loss_every: 1      # apply every N steps (set > 1 to reduce overhead)
```

**Properties:**
- Zero overhead when `style_loss_weight: 0.0` (default) — no extra computation.
- Memory cost is negligible: Gram matrix of `[B, 32, H/8, W/8]` is `[B, 32, 32]`.
- Only applied on conditioned steps (skipped when `image_dropout_prob` fires).
- Logged as `style_loss` in the training output and heartbeat.

**Tuning guidance:**
- Start with `style_loss_weight: 0.05` and monitor `loss_cond`/`loss_null` gap.
- If `ip_scale_double` rises above 1.0, the weight may be too high (content leakage).
- Combine with QUALITY-1 (cross-image reference permutation) for strongest style/content separation.

---

## Export

Convert a trained checkpoint to an iris.c-loadable bundle:

```bash
# BF16 export (recommended)
python train/export/export_adapter.py \
    --checkpoint /Volumes/2TBSSD/checkpoints/stage1/step_050000.safetensors \
    --output     /tmp/iris_ip_adapter/ \
    --use-ema    --validate

# INT8 export (~2× smaller)
python train/export/export_adapter.py --checkpoint ... --output ... --quant int8

# Style-only mode (zero double-stream blocks)
python train/export/export_adapter.py --checkpoint ... --output ... --style-only
```

Output bundle:

```
adapter_weights.safetensors   — mmap-ready weight tensors
adapter_meta.json             — dimensions, quant mode, provenance
```

See [export/README.md](export/README.md) for the full bundle format, quantisation
comparison table, and C integration guide. See [export/iris_ip_adapter.h](export/iris_ip_adapter.h)
for the C loader API.

---

## Evaluation

`train/eval.py` generates images for each entry in `configs/eval_prompts.txt`
using a checkpoint and computes CLIP-I (style fidelity) and CLIP-T (prompt
adherence) via SigLIP SO400M. Requires `transformers` + `torch`.

```bash
python train/eval.py \
    --checkpoint /Volumes/2TBSSD/checkpoints/stage1/step_050000.safetensors \
    --config     train/configs/stage1_512px.yaml

# Style-only mode (suppress layout injection)
python train/eval.py --checkpoint ... --config ... --style-only

# Dial conditioning strength down to 70%
python train/eval.py --checkpoint ... --config ... --sref-strength 0.7
```

Output per run: `eval_results.json` (CLIP scores per prompt) and `report.html`
(reference | generated image grid with scores).

**Automatic eval hook** — set `eval.enabled: true` in `stage1_512px.yaml` to
run eval every `eval.every_steps` (default 10 000) steps during training. Uses
the in-memory Flux model; no extra model reload. Results are also logged to
W&B when active.

---

## Inference-time style control

`effective_scale(style_only, sref_strength)` on `IPAdapterKlein` returns the
per-block scale without mutating the trained weights:

| Flag | Effect |
|------|--------|
| `style_only=True` | Zeros double-stream block scales (indices 0–4); keeps single-stream (5–24). Removes layout/content injection, leaving style-only conditioning. |
| `sref_strength=0.7` | Multiplies all active scales by 0.7. Range 0.0 (adapter off) – 2.0+ (over-conditioned). |

Both flags are available in `eval.py` (`--style-only`, `--sref-strength`) and
`run_inference.py`. Config defaults live in `stage1_512px.yaml` under
`adapter.style_only` / `adapter.sref_strength`; CLI flags override them.
