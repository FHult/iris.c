# SREF Joint-Backbone Style-Reference Adapter — Project Plan

Status: PROPOSED (2026-07-10). Owner: SREF workstream. Prereq reading: BACKLOG `SREF-FILM-1`,
`SREF-STYLE-CFG-PROBE`, `SREF-LEARNED-STAGE1`, and the retrieval-hybrid plan
(`plans/sref-retrieval-hybrid-project.md`). Memory: `project_sref_state.md`.

## 1. Why every prior adapter died (the one root cause)

A trained style-reference adapter has failed on this stack **three times, across three different
injection channels**:

| Attempt | Channel | Result |
|---|---|---|
| IP-Adapter K/V (2026-06) | per-block cross-attn `to_k_ip`/`to_v_ip` | collapse (`to_v_ip` rank ~6) |
| Learned tokens (2026-07-08) | in-sequence style tokens | inert / no binding |
| CSD→modulation FiLM (2026-07-10) | adaLN `temb` shift | collapse (cross-ref corr 0.9998) |

The velocity probe (`SREF-STYLE-CFG-PROBE`) and the FiLM diagnostic proved the collapse is **loss-bound,
not channel-bound**. Mechanism:

- Training is flow-matching reconstruction of a **noised target**. At low noise the model copies the
  (barely-noised) latent; at high noise it leans on the text prompt.
- **Style is a low-information, global signal.** Reconstruction loss weights everything by information
  content; content dominates and the model already gets it elsewhere, so a style hint contributes almost
  nothing to loss reduction.
- With the **backbone frozen**, the only trainable thing is the adapter, and the loss never rewards
  reference-*discrimination* → the easy minimum is a reference-*independent constant*. (FiLM even learned a
  large constant `temb` shift, norm ~44, identical across wildly different CSD inputs.)

The two things that DO work sidestep this loss: **in-context VAE-ref tokens** ARE the image (the loss
can't reconstruct without them at high noise), and **per-style LoRA** is direct weight surgery (it *is* one
style — no reference→style mapping to collapse).

**Corollary:** unfreezing the backbone is necessary but NOT sufficient. The Stage-2 probe added a small
r16 LoRA jointly and still collapsed — because at **batch=1** you cannot build the one term that directly
punishes a constant output: an **in-batch contrastive/discrimination loss**.

## 2. The design — what changes vs every prior attempt

Four levers, applied together. The first is the one we have literally never been able to try.

1. **In-batch contrastive loss (THE lever; requires batch > 1).** With B distinct references in a batch,
   reward "output conditioned on reference i matches reference i's style more than reference j's." A
   constant output scores catastrophically (equally (un)like everyone) → collapse is directly penalized.
   Batch=1 (our M1 ceiling at 512px) makes this term impossible; this is very plausibly *why* everything
   collapsed.
2. **Trainable backbone (LoRA r64–128, or full FT on cloud).** Gives the model the lever to actually
   change its style behaviour so the contrastive/style terms have something to push. (Not the frozen
   backbone; not the tiny r16 that failed.)
3. **High-noise-biased timestep sampling.** Global style is set at high noise, where the noised latent
   carries little info and conditioning MUST be used. Bias `t` sampling toward high noise so the
   style-use gradient lands where the model can't cheat.
4. **Style-space loss + CFG dropout.** A CSD/Gram/AdaIN-stats term alongside reconstruction (rewarded now
   because the backbone can move); style-condition dropout so inference-time CFG can amplify the (now
   organised) style signal — the amplifier experiment A found was missing.

### Architecture

```
reference image ─► CSD encoder (have it) ─► 768-d content-invariant style vector
                                              │
target image ─► VAE latent ─► noisy latent ─► DiT (LoRA, TRAINABLE) ─► v-pred
                                              ▲         ▲
                        CSDModulation(temb) ──┘         └── (optional) decoupled cross-attn tokens
                        [already built]                     [later; backbone learns to route it]

loss = L_recon  +  λ_c · L_contrastive(InfoNCE over batch)  +  λ_s · L_style(AdaIN/Gram)
       t ~ high-noise-biased            CFG: drop CSD with prob p
```

Conditioning descriptor = **CSD** (content-invariant by construction; already proven to discriminate style
in the retrieval-hybrid). Injection via the already-built `CSDModulation` (`temb += csd_mod(csd)`); a
decoupled cross-attention token path can be added later once the mechanism is proven.

### Loss math (v-prediction flow, linear schedule α=1−t/1000, σ=t/1000)

- `noisy = α·x0 + σ·noise`, `v_target = α·noise − σ·x0` (existing `fused_flow_noise`).
- `x0_pred = (α·noisy − σ·v_pred)/(α²+σ²)` (existing `predict_x0`, unbiased).
- **Style descriptor** `s(x) = concat(mean_HW(x), std_HW(x))` per channel → [B, 2C] (AdaIN stats,
  content-invariant to first order; existing `style_stats`). L2-normalise.
- **L_recon** = mean_i ‖v_pred_i − v_target_i‖².
- **L_contrastive (InfoNCE)**: `S[i,j] = s(x0_pred_i)·s(x0_target_j)/τ`; `L = CE(S, diag)`. Each
  prediction's style must match its OWN target's style more than the other batch members' → a constant
  output cannot classify → collapse punished.
- **Total** `L = L_recon + λ_c·L_contrastive` (+ optional `λ_s·gram_style_loss`).

### The primary anti-collapse diagnostic (in-loop, cheap)

Every N steps: take K fixed references, forward the **same** latent+text+noise+t with each, measure
cross-ref correlation of `x0_pred`. **Success = it drops below ~0.9** (references change the output).
The render+scorecard gate then confirms the change is STYLE (not content-copying).

## 3. Compute reality — train on cloud, infer in local C

- **Full FT of 4B does NOT fit 32 GB** (AdamW fp32 states alone ≈ 32 GB). On the M1 it is LoRA-only, and
  batch>1 at 512px does not fit — i.e. the M1 cannot run the recipe that matters at product resolution.
- **Training and inference separate cleanly.** A jointly-trained model is just *different weights + a
  conditioning path* — the C/Metal engine already runs that shape (LoRA + a small projector). So: heavy
  training on a rented **A100/H100 80 GB** (batch 16–64, high-rank/full FT, real throughput), then export
  → reimplement the small conditioning path in C → ship like the Style Library.
- **6.4M corpus** is the asset: diversity, not epochs. A serious run sees ~10⁵–10⁶ look-pairs total
  (each ~once → good generalisation), not many passes. Stage-1 cloud ≈ a few hundred GPU-hours ≈
  low-four-figures USD.

## 4. Staged plan (de-risk before spend; each stage gated)

- **Stage 0 — data.** Precompute CSD for the full corpus (only a 200/shard subset exists as
  `universe_csd`). Build the look-pair index at scale (method already in `neighbors.sqlite`).
- **Stage 0.5 — CHEAP LOCAL GO/NO-GO (this doc's immediate deliverable).** Test the ONE thing we've never
  tried: does **in-batch contrastive + trainable backbone** break the collapse? Run at **256 px** (¼ the
  tokens of 512 px) where **batch 4–8 + LoRA fits on the M1**. ~5–10k steps.
  **GO = in-loop cross-ref `x0` corr < 0.9 AND (render gate) styleCSD Δ beats band-control 0.009 with
  cross-ref output corr < 0.9.** NO-GO = collapses even with in-batch contrastive → the loss-bound verdict
  is ironclad, stop, and we spent ~a day not a cloud budget.
  - Config: 4B base, LoRA r64 (double+single attn) + `CSDModulation`, batch 4 @256px, high-noise-biased
    `t`, InfoNCE (τ=0.1) λ_c≈1.0, CSD-dropout 0.1, cosine LR 1e-4, grad-checkpointed, `cond_mode=csd`
    on `universe_csd` + look-pairing. Trainer: `train/lora/probe_joint_contrastive.py`.
- **Stage 1 — cloud proof.** A100/H100, full recipe @512px, batch 32, ~50–100k steps, ~500k look-pairs.
  Same gate. The real product go/no-go.
- **Stage 2 — scale + tune** (data mix, ranks, loss weights, add the decoupled cross-attn token path).
- **Stage 3 — ship.** Export weights + conditioning path; reimplement the conditioning path in C; wire
  into the web like the Style Library.

## 5. Risks / honest uncertainty

- The style-space + repulsion losses were GAMED on the FROZEN backbone at batch=1 (6-experiment campaign).
  The new variables are (trainable backbone) + (TRUE in-batch contrastive at batch>1) + (high-noise
  weighting). No prior attempt combined these; equally, none is guaranteed to work.
- If style is genuinely too low-information for this backbone to organise even under those pressures, it
  collapses again — but Stage 0.5 reveals that for ~a day of local compute before any cloud spend.
- Content confound: AdaIN-stat InfoNCE can be partly satisfied by content when texts differ. Stage 0.5's
  fixed-content in-loop collapse metric controls for this; the render+scorecard gate is the STYLE check.

## 6. Decision gates (summary)

1. Stage 0.5 in-loop cross-ref `x0` corr < 0.9 → proceed to render gate.
2. Render gate styleCSD Δ > 0.009 AND cross-ref output corr < 0.9 → GREENLIGHT cloud (Stage 1).
3. Either fails → the learned-adapter direction is closed with maximum confidence; retrieval-hybrid remains
   the shipped answer; revisit only C (hypernetwork→LoRA via weight distillation).
