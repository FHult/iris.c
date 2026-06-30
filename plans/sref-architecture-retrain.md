# SREF Architectural Retrain — Review & Plan

Status: **Architecture review IN PROGRESS** (2026-06-30). Charter in BACKLOG
"SREF ARCHITECTURAL RETRAIN — CHARTER & NEXT STEPS". This doc is the design + execution tracker
for the journey to a true `--sref` adapter (match THIS reference's specific style).

## Why we're here (the settled diagnosis)
The shipped IP-Adapter mode-collapses (every reference → ~same output; cross-ref output corr ≥0.92).
This was diagnosed exhaustively (see `plans/sref-retrain-diagnostic.md`):
- **Root cause:** the injection's `to_v_ip` learns rank ~6 → the injected V is near-constant across
  references. Universal across all cond_modes and 17 checkpoints.
- **Loss design is exhausted:** 6 training experiments (rank penalty; x0-style repulsion aggressive/gentle;
  longer rank-only; V-decorrelation; own-Q output repulsion) ALL failed the <0.90 gate. Each lever is
  gamed (V-space proxy), destabilizes (aggressive), or is overpowered (content anchor wins). Best is
  rank-only at 0.886/0.926 — still failing.
- **Conclusion:** the collapse is STRUCTURAL — injecting at scale ~0.38 into the FROZEN DISTILLED base via
  an additive K/V side-channel, the loss-minimum is a generic style push. Independently confirms the
  earlier "~0.70 mechanism-bound ceiling."

## The central architectural clue
Plain **in-context conditioning** (reference tokens placed IN the transformer sequence — the img2img path,
the shipped web default) **DISCRIMINATES references fine.** The frozen base already knows how to use
in-sequence tokens. The trained adapter's **separate additive K/V side-channel** is a mechanism the frozen
base was never trained for, and it collapses. → Make the adapter produce IN-SEQUENCE conditioning.

## Candidate architectures (to be detailed + sequenced)
1. **Learned in-context conditioning (LEADING).** Reference → small set of learned tokens CONCATENATED into
   the sequence (like the proven in-context path), trained to carry STYLE with a content term preserving the
   prompt's subject. Inherits in-context's discrimination; training learns compression + style-isolation.
2. **Higher-capacity CSD conditioning.** Replace the rank-1 global FiLM (`out=q*(1+scale)+shift`) with real
   cross-attention / higher-rank modulation so the content-invariant CSD signal carries style structure.
3. **Base-model adapter (highest impact, highest cost).** Train vs the undistilled base (CFG, 50 steps, more
   capacity). Distilled base may be too rigid to steer. Fresh train (distilled adapter doesn't transfer —
   SREF-BASE-1) + new CFG/inject code. Gate cheaply whether base also collapses.
4. **Different injection (AdaIN/stats or higher scale + content preservation).** Speculative; only if 1–3 stall.

## Architecture review — ANSWERS (Explore agent, 2026-06-30, with code refs)
- **Adapter injection (training):** separate per-block SDPA `ip_out = SDPA(image_Q, K_ip, V_ip)` then
  `hidden = hidden + scale[i]*ip_out` (train_ip_adapter.py:3447-3452, single-stream :3499-3507). IP tokens
  are NOT in the sequence; image Q is the frozen base's. A learned residual side-channel.
- **In-context (C inference):** reference VAE latents CONCATENATED into the sequence
  `[TEXT | TARGET_IMAGE | REFERENCE_IMAGE]` (iris_transformer_flux.c:4445), distinguished only by RoPE
  T-offset; native self-attention attends to them every block → different refs = different tokens =
  different output → DISCRIMINATES. No bottleneck projection, no scale knob.
- **Base-model adapter:** NO support — guidance hardcoded None, no CFG dual-pass, distilled 4-step only;
  needs substantial new code. (SREF-BASE-1; distilled adapter is OOD on base.)
- **Scale ~0.38:** learned per-block scalar (init 1.0), settles low as a SYMPTOM of the low-rank collapse
  (balances content destruction vs the constant style push), not an architectural requirement.
- ⚠️ The agent's *recommendation* ("full-rank K/V init + vproj_rank_penalty + vproj_decorr_loss") is
  EXACTLY the loss approach the 6 experiments already REFUTED (rank gamed/plateaus, decorr gamed,
  repulsion overpowered). DO NOT re-try loss-design fixes — that path is closed.

## Candidate 1 design — learned in-context, style-only tokens
THE INSIGHT, sharpened: in-context DISCRIMINATES (in-sequence) but LEAKS composition (the ref's VAE latents
carry content). The adapter DECOUPLES (style-only) but COLLAPSES (side-channel). Candidate 1 wants BOTH:
place STYLE-ONLY tokens IN the sequence so native attention discriminates them, while carrying no
composition. Encoder: reference → (content-invariant) style tokens in the base's img-embedding space →
concat into the single-stream sequence (like the reference image tokens, but learned + style-only).
**KEY RISK:** learned tokens in raw hidden space are OOD for the frozen base (in-context works because ref
tokens are real VAE latents through the SAME img embed). The tokens must live in an in-distribution
representation, or the base won't use them coherently. → De-risk BEFORE building a training pipeline.

## FIRST EXPERIMENT (cheap, NO training — de-risks the whole direction)
Probe the core hypothesis with the EXISTING in-context path + image preprocessing only:
**Feed CONTENT-DESTROYED references (patch-shuffled / heavily blurred) through in-context img2img and
measure whether STYLE transfers WITHOUT composition, and whether it DISCRIMINATES.**
- Patch-shuffle destroys composition but preserves local texture/colour/style; VAE-encode → in-sequence.
- Run `iris -d flux-klein-model -i <shuffled_ref> -p "a cat sitting on a chair" --img2img-strength 1.0`
  across several distinct refs (churchill, cyberfika, woodcut, flat_sticker), fixed seed.
- Measure: (a) style match (palette/texture vs ref), (b) composition leak (should be ~0 — scrambled),
  (c) DISCRIMINATION (cross-ref output corr — want < 0.90, like real in-context, unlike the adapter).
DECISION: if content-destroyed in-context still discriminates AND drops the composition leak → "in-sequence
style-only" is VIABLE → build a learned encoder that produces such tokens (Candidate 1 proper). If it
collapses or stays leaky → in-sequence can't be cleanly decoupled from content → pivot to base-model
adapter (Candidate 3) or higher-capacity CSD (Candidate 2). Either way this ~30-min, no-train probe
chooses the multi-week direction. Repro: `scratchpad/sref_arch_probe.*`.

## Execution gates (every architecture attempt)
- train↔infer PARITY fixture + prod-flag compile + `make mps` (AGENT protocol) before trusting any result.
- cached-mode only (live-encode segfaults MLX — BUGS MLX-1); never train from cold storage (AGENT #6).
- Promote ONLY on `sref_ref_discrimination.py` max cross-ref corr <0.90 AND a quality eyeball (corr can
  move while quality degrades — Option B showed this). Web stays on in-context until a ckpt PASSES.

## Log
- 2026-06-30: review opened; charter banked; Explore agent mapping injection paths + base support.
