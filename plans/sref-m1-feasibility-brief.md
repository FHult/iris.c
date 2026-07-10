# Can we train an "instant style-reference adapter" for Flux.2 on an M1 Max 32 GB? — External-review brief

Self-contained brief for a second opinion (Fable). **The question: is there genuinely NO way to run the
"expensive" joint-backbone training on consumer hardware (Apple M1 Max, 32 GB unified memory, MLX), or are
we missing a technique that would make it fit?** Everything needed to reason about it is below; no repo
access required.

## 1. The goal and what we've established

- **Product goal:** Midjourney-style `--sref` — user uploads a reference image; generations adopt its
  STYLE (not content), *instantly* (no per-style training at use time).
- **Model:** Flux.2 Klein **4B** (MMDiT: 5 double-stream + 20 single-stream blocks, hidden 3072, joint
  text+image token attention). Flow-matching (v-prediction). We use the "base" 50-step CFG variant for
  training (steerable); a 4-step guidance-distilled variant also exists.
- **Inference** is a bespoke **C/Metal** engine (no PyTorch at inference). **Training** is **MLX** on the
  M1 Max. So any trained conditioning path must be reimplementable in C (cheap for MLPs/LoRA; expensive for
  novel attention kernels).
- **What WORKS today:** (a) per-style **LoRA** (rank 16, trains the frozen backbone low-rank; strong style,
  but one LoRA per style — not instant); (b) **in-context** reference tokens (VAE-encode the reference,
  append as sequence tokens — the native img2img path; discriminates perfectly). Shipped product =
  "retrieval-hybrid": CSD-match the reference to the nearest pre-trained per-style LoRA.
- **What FAILED (3 times, decisively):** every attempt to train a *generic* adapter that maps an arbitrary
  reference → style, on a **frozen** backbone:
  - IP-Adapter K/V cross-attn injection → rank-collapse (`to_v_ip` stable-rank ~6); reference-independent
    constant injection.
  - Learned in-sequence style-token encoder (SigLIP→tokens) → inert; a joint r16 LoRA probe still didn't
    bind.
  - CSD→adaLN-modulation (FiLM into the timestep embedding) → collapse: cross-reference output correlation
    **0.9998**; the module learned to map *every* distinct reference to the *same* constant modulation shift.

## 2. The diagnosed root cause (why "just add an adapter" is dead)

The collapse is **loss-bound, not channel-bound** (proven across attention-token, K/V, and modulation
channels). Flow-matching trains on a **noised target**: at low noise the model copies the barely-noised
latent; at high noise it leans on the text prompt. **Style is a low-information, global signal, and
reconstruction loss weights everything by information content** — content dominates and the model already
gets it elsewhere, so a style hint contributes ~nothing to loss reduction and any module attached to it
decays to a reference-independent constant. Unfreezing the backbone is *necessary but not sufficient* (a
small joint LoRA still collapsed), because the loss still never *rewards telling references apart*.

**The fix we want to test** = train the backbone jointly AND add the one term that directly punishes a
constant output: an **in-batch contrastive / InfoNCE** loss (output conditioned on reference i must match
i's style more than the other batch members' j≠i). A constant output scores catastrophically on it. This
term **requires batch > 1** (the negatives must be in the same forward/graph). At batch=1 it does not exist
— which is very plausibly *why every prior attempt collapsed*, since batch=1 has been our hard ceiling.

## 3. The hardware reality (the crux of the question)

M1 Max, **32 GB unified memory**, MLX. Measured facts on THIS machine (our numbers):

- **Frozen base + LoRA, full backprop through the 4B transformer, batch 1:** 512px ≈ **19.3 GB**, 768px
  ≈ 19.3 GB, 1024px ≈ **21.3 GB** peak. So the usable working ceiling is ~**21.5 GB** before instability.
- **Full fine-tuning of 4B does NOT fit.** bf16 weights 8 GB + bf16 grads 8 GB + **AdamW moment states
  (m, v)** — in fp32 that's 2×16 = 32 GB, in bf16 still 2×8 = 16 GB → 8+8+16 = **32 GB before any
  activations**. Infeasible on 32 GB. So on the M1 it is **LoRA-only** (frozen base + low-rank trainable
  delta), by necessity.
- **The batch>1 wall (fresh, today).** We built the joint recipe (LoRA r64 on all attention + a small
  CSD→modulation MLP, ~79 M trainable) and tried the in-batch-contrastive run at **256px, batch 4**.
  Result: MLX **stalls compiling the first `value_and_grad` graph** — 25+ minutes at 95% CPU, all 32 GB
  consumed, **not one training step**. Root causes we found: (a) grad-checkpointing builds a per-block
  recompute subgraph × the autodiff tape → a pathologically large graph; (b) variable text-token length
  per batch → MLX recompiles the whole graph every step. We fixed (b) with a fixed text length and dropped
  checkpointing — and the batch-4 joint-grad graph through the 4B base **still** takes many minutes to
  compile / barely steps. In short: **the M1 strains to even build the batched-contrastive training graph**,
  before we get to throughput.
- **Throughput, when it does step:** frozen-LoRA batch-1 training is ~**0.13 it/s** at 512px. A batched
  joint-grad step (batch 4, 256px) measured **~0.03 it/s** in a synthetic smoke — i.e. ~30 s/step, so even
  10⁴ steps is days, and 6.4 M images means we'd see a small fraction of one epoch.

Our current conclusion (which we want challenged): the recipe that matters — **trainable backbone + a real
in-batch contrastive at batch ≥ 8–16, at 512px** — does **not** fit or run acceptably on M1 Max 32 GB, so
the honest path is **train on a rented A100/H100 80 GB, infer locally in C**. We have appetite to try
harder locally before accepting that.

## 4. The specific question for review

**Is there a credible way to run the expensive joint-backbone + in-batch-contrastive training on M1 Max
32 GB / MLX, or is cloud genuinely unavoidable?** In particular, are any of these viable, and what would
you prioritize:

1. **Decouple contrastive negatives from batch size (MoCo-style memory bank).** Keep a queue of recent
   style-stat embeddings as negatives, so even **batch 1–2** gets many InfoNCE negatives from the queue
   (embeddings are cheap to store; gradients only flow through the current-batch positives). Does this
   soundly replace a large true batch for *this* collapse problem, or does the lack of in-graph negatives
   defeat the anti-collapse pressure?
2. **QLoRA-style 4-bit frozen base.** MLX supports quantization; a 4-bit base is ~2 GB instead of 8 GB,
   freeing ~6 GB for a larger true batch and/or higher res. Any accuracy/■stability caveats for
   flow-matching + LoRA on a quantized DiT?
3. **Gradient accumulation** for the recon term to raise *effective* batch — while noting it does NOT give
   in-batch contrastive (negatives aren't co-resident). Is there a formulation that gets contrastive
   pressure under grad-accum?
4. **Resolution / sequence budget.** Train the style mechanism at 192–256px (style is low-frequency) and
   rely on it transferring to 512px+ at inference. Legitimate, or does style-at-low-res fail to transfer?
5. **MLX-specific pitfalls.** The graph-compile stall on large `value_and_grad` graphs (grad-checkpoint
   recompute subgraphs; per-shape recompilation). Known mitigations (fixed shapes, `mx.compile`, avoiding
   nn.utils.checkpoint, smaller sharded graphs) that would make a batched joint-grad step actually compile
   and run on M1?
6. **A different anti-collapse objective that doesn't need batch>1 at all** — e.g. a per-sample
   cross-reference *swap* consistency loss (predict with the correct vs a wrong reference in two forwards
   and require them to differ in style space), or a reward/So-style signal. Would any single-sample
   objective plausibly break the loss-bound collapse where in-batch contrastive would?
7. **Is the whole local-training premise wrong** — i.e. given the collapse is loss-bound and the M1 can't
   run the batched recipe, is the *only* honest instant-generic path a **hypernetwork→LoRA distilled from
   our per-style LoRAs** (supervised regression reference→weights, which sidesteps the flow loss entirely
   and is cheap to train), making the whole joint-backbone question moot?

### What "success" must clear (so feedback is grounded)

- **Anti-collapse gate:** with seed+prompt fixed and only the reference varied, cross-reference output
  correlation must drop **below ~0.90** (it is ~0.98–0.9998 in every collapsed run).
- **Style gate:** a CSD-based "styleCSD Δ" (output-vs-style-centroid cosine minus a no-reference baseline)
  must beat our training-free band-control rail's **0.009** on painterly refs — AND survive the collapse
  gate (a high styleCSD Δ with cross-ref corr ≥ 0.9 is a mode-collapse false positive, which we've been
  fooled by before).
