# SREF Architecture Options — Deep-Research Report (2026-07-05)

Deep research into architecture options for instant reference-style conditioning ("--sref") on our
stack, commissioned after SREF-DATA-TEST falsified the data-selection fix and confirmed the
IP-adapter K/V-injection collapse is structural (BACKLOG SREF-CHAMPION-COLLAPSE, SREF-DATA-TEST).
Method: 6-angle web sweep → 24 sources → 118 extracted claims → adversarial 3-vote verification
(9 confirmed / 3 refuted) + 6 targeted source-verification agents for the claims the first pass
could not finish. Every load-bearing claim below was verified against the primary source with
verbatim quotes.

## Headline findings

**1. The sequence path is the only conditioning mechanism validated on Flux-family DiTs.**
Every credible Flux conditioning method — OminiControl (arXiv 2411.15098), DreamO (2504.16915),
USO (2508.18966), omini-kontext, DreamOmni2, Flux Kontext itself — conditions by concatenating
reference tokens into the transformer sequence and letting native joint attention do the work,
with at most a small LoRA to teach the base to use them. A curated survey (Westlake-AGI-Lab
Awesome-Style-Transfer list) catalogs **no** style-transfer adapter validated on a Flux/DiT
backbone: the entire adapter family (InstantStyle, StyleShot, DEADiff, CSGO, StyleAligned,
StyleID) is SDXL/UNet-era. Our own empirical record matches exactly: in-context discriminates
perfectly; the K/V side-channel collapsed.

**2. Why our IP-adapter failed — literature triangulation.**
- InstantStyle (2404.02733, verified 3-0): CLIP-style global image features entangle style with
  content; tuning injection strength cannot fix the coupling — the failure is *where/what* is
  injected, i.e. architectural.
- DEADiff (2403.06951, verified 3-0): two causes — (a) reconstruction training is easier than
  text-following, so the model over-relies on the reference channel; (b) the encoder features
  couple style with semantics. Published fixes are architectural (dual Q-Former filters +
  mutually exclusive injection-layer subsets), not loss tweaks — consistent with our 6 failed
  loss-side fixes.
- The common published failure direction is content LEAKAGE (reference dominance): RB-Modulation's
  ICLR-25 eval shows IP-Adapter with the highest DINO style-sim (0.89) but catastrophic prompt
  alignment (ImageReward −1.99). Ours failed in the *opposite* direction (reference-inert) —
  best explained by the combination of style-only feature diets + leak-loss 0.5 + frozen
  double-stream scales pushing the adapter to the OTHER degenerate optimum ("apply the average
  style, ignore the reference"), and by:
- Distilling Diversity and Control (2503.10637, verified): **distilled few-step models commit
  final image structure within the FIRST denoising step** (base models take ~30% of steps).
  A weak side-channel signal on a 4-step guidance-distilled model has almost no room to steer —
  ignoring it is the easy optimum. This is the missing systemic explanation for why the same
  recipe that works elsewhere collapsed here.

**3. Base↔distilled control transfer works (SDXL evidence).**
2503.10637 (verified): Concept Sliders / DreamBooth / Custom Diffusion LoRAs trained on a base
model transfer to its distilled counterpart **without retraining, and vice-versa**. Also:
"diversity distillation" — run only the FIRST step with the base model, remaining steps
distilled — restores (even exceeds) base-model diversity at distilled cost (DreamSim 0.350 vs
base 0.337 vs distilled 0.264). Direct roadmap implication: once the 4B base is ported, train
conditioning against the base and ship it on the distilled model; consider hybrid first-step
sampling in iris.c (we will have both weights).

## Ranked shortlist

### #1 — Precision in-context: RoPE frequency-band control + KV-reuse (zero training)
Upgrade the shipped content-destruction+in-context rail from "destroy spatial content crudely
(patch-shuffle)" to "suppress positional copying surgically".
- **Evidence**: "Untwisting RoPE: Frequency Control for Shared Attention in DiTs"
  (arXiv 2602.05013, all claims verified): reference copying in DiT attention sharing is
  POSITIONAL — high-frequency RoPE bands dominate attention and force spatially-aligned copying;
  low bands carry global/semantic (style) interaction. Fix: per-chunk key-side scale
  `s_d = s_hf + (s_lf − s_hf)·d̃^β` (β=2), attenuate high bands / amplify low bands, applied only
  to reference-image KEYS, only in SINGLE-STREAM blocks, timestep-scheduled. Training-free, no
  gradients. User study: 2.40 vs StyleAligned 0.74 vs IP-Adapter 0.44. (Their single-stream-only
  finding independently matches our champion's freeze_double_stream_scales.)
- **Plus** OminiControl2 (2503.08280, verified) efficiency: reference-token K/V computed ONCE and
  reused across denoising steps (asymmetric masking: condition tokens do not attend to noisy
  tokens) = 84.7% of their conditioning-overhead reduction; optional 2× spatial pre-compression
  of the reference; strength control via attention BIAS γ on generation→reference attention
  (γ=0 off, γ>1 amplified) — a cleaner knob than latent blending.
- **Fit**: C-implementable in days (RoPE kernel: per-band scale for reference-token keys;
  attention: bias + optional K/V cache of reference tokens across 4 steps). No training. Improves
  the rail users already like. Sweep s_hf/s_lf with the discrimination gate + CSD score.
- **Risks**: paper validated on Flux.1-dev 50-step, NOT few-step distilled — must validate on
  4-step ourselves (cheap); exact s_hf/s_lf values unpublished (sweep).

### #2 — Learned in-sequence style tokens (USO-style projector) — the trainable rail
The Phase-2 "in-sequence encoder" from the pluggable-conditioning framework, now with verified
precedent (USO, ByteDance, arXiv 2508.18966 + code):
- **Architecture (verified in paper AND released code)**: style ref → SigLIP semantic encoder
  (NOT VAE) → lightweight "Hierarchical Projector" → **192 style tokens** (64/scale × SigLIP
  layers −2/−11/−20) → concatenated in-sequence with the SAME RoPE ids as text tokens (zeros).
  No K/V injection, no modulation path. Separate encoders for style (SigLIP) vs content (VAE)
  is their architectural anti-leakage measure.
- **Training (verified)**: Stage 1 trains ONLY the projector, DiT fully frozen → tiny trainable
  footprint, fits 32 GB MLX comfortably, and consumes exactly the SigLIP features we already
  precompute. Stage 2: DiT LoRA r128 (we have the trainer). USO's ablation says Style Reward
  Learning (CSD-similarity reward) is essential for their top scores — expensive; start without,
  approximate later (e.g. CSD-scored best-of-N data or reward-weighted samples).
- **Data**: 200k stylized pairs at USO scale; our 100k pool + neighbors_look.sqlite (look-similar/
  content-different pairs) is the right shape — i2L independently used the same
  "style-consistent, content-diverse" principle (= our DATA-SELECTION PRINCIPLE).
- **Fit**: C side = small projector (few matmuls/attention) emitting tokens into the existing
  in-context path; train↔infer parity protocol applies. Inference cost: +192 tokens ≈ +19% seq
  at 1024-token image vs +100% for a full 512px latent reference — and KV-reuse (#1) applies.
- **Risks**: an in-sequence signal is much harder to ignore than side-channel K/V (it sits in the
  proven-discriminative attention path), but ignoring is still possible → discrimination gate
  mandatory at every checkpoint; distilled-4-step trainability is unproven (USO trained on
  FLUX.1-dev 50-step) → if it stalls, train against the 4B base and transfer (finding 3).

### #3 — Base-model training + transfer (roadmap enabler)
Not an architecture, but a de-risking strategy now backed by evidence (2503.10637): train
conditioning modules (LoRAs, the #2 projector's Stage-2 LoRA) against the 4B BASE (CFG, gradual
structure formation = more steerable target), ship on distilled. Also implement hybrid sampling
(step 1 = base, steps 2–4 = distilled) as a diversity/steerability option in iris.c.

### #4 — Hypernetwork → instant LoRA (i2L) — long-term, now precedent-backed
i2L "Compressing Image Style Training into a Single Model Forward" (arXiv 2606.13809, verified):
image encoder (SigLIP2 patch embeddings) + LoRA-queries transformer + compressed decoding heads →
predicts explicit style-LoRA weights in ONE forward; trained end-to-end with plain flow-matching
through a frozen backbone (no per-style teacher LoRAs). **Backbones include FLUX.2-klein-base-4B
and Z-Image — our exact model families.** Output is a standard LoRA → consumed by our existing
iris_lora.c path with zero new inference code; composable.
- **Blockers**: their scale is 8×A100 × ~7 days per backbone on MegaStyle-1M; no per-style-LoRA
  quality baseline. Keep as Phase-4. Cheap intermediate: retrieval hybrid — CSD-nearest-neighbor
  over a library of our own trained style LoRAs (+ interpolation between the top-2) gives
  "instant LoRA" UX from pieces we already have (CSD index + LoRA trainer + loader).

## Avoid (with evidence)
- **Per-block K/V side-channel adapters** (any variant): our structural collapse + zero validated
  Flux ports + InstantStyle/DEADiff diagnoses + the whole Flux ecosystem converging on the
  sequence path.
- **Naive shared-attention (StyleAligned-style) on DiT**: verified 3-0 — collapses to
  near-identical images via content leakage unless the RoPE fix (#1) is applied.
- **RB-Modulation as a "modulation" rail**: verified 3-0 that despite the name it is NOT a learned
  AdaLN/FiLM injection — it is per-step test-time optimization (M=3 GD steps/denoise step,
  backprop through the CSD ViT). Impossible in the C engine. Also refuted (0-3): "a 768-d CSD
  vector alone is a sufficient style signal" — do not build the CSD→AdaLN FiLM rail on that
  assumption without our own capacity experiment; note our FiLM-style adapter experience already
  showed weak capacity.
- **CSGO-scale triplet pipelines**: verified — 8×H800, batch 20/GPU, 80k steps, 210k triplets
  built by training per-image LoRAs + B-LoRA decomposition. Two orders beyond budget; SDXL-only.
  (The B-LoRA decomposition trick itself is interesting for cheap triplet synthesis later.)
- **SDXL block-maps ported literally**: InstantStyle's style/layout block indices are UNet-
  specific. The Flux analogue is stream-level: single-stream blocks carry appearance/style
  (2602.05013 + our champion's frozen double-stream finding).

## Recommended sequencing
1. **Now (C work, no training)**: implement #1 — RoPE band-scaling for reference keys
   (single-stream blocks) + attention-bias strength + reference-KV reuse across steps. Validate
   with the discrimination gate + CSD; A/B vs patch-shuffle. This upgrades the shipped feature
   immediately.
2. **Next training cycle**: #2 Stage-1 (projector only, frozen DiT, cached SigLIP) on
   look-pairing data; gate; only if it discriminates, add Stage-2 LoRA.
3. **With the 4B base port**: retrain/finetune #2 against base, transfer to distilled; prototype
   hybrid first-step-from-base sampling.
4. **Later**: retrieval-hybrid instant LoRA; revisit i2L-style hypernetwork at reduced scale.

## Key sources
- InstantStyle — arxiv.org/html/2404.02733v1 · DEADiff — arxiv.org/html/2403.06951v1
- RB-Modulation — ICLR'25 (proceedings PDF) + arxiv.org/abs/2405.17401
- Untwisting RoPE — arxiv.org/html/2602.05013v1
- OminiControl — arxiv.org/html/2411.15098 · OminiControl2 — arxiv.org/abs/2503.08280
- DreamO — arxiv.org/html/2504.16915v1 · omini-kontext — github.com/Saquib764/omini-kontext
- USO — arxiv.org/abs/2508.18966 + github.com/bytedance/USO
- i2L — arxiv.org/html/2606.13809v1
- Distilling Diversity and Control — arxiv.org/html/2503.10637v1
- CSGO — arxiv.org/html/2408.16766v1
