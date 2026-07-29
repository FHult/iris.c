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

## FIRST EXPERIMENT RESULT (2026-06-30) — VALIDATED, and a NO-RETRAIN shortcut emerged
Patch-shuffled (6×6) references through the EXISTING in-context path (`-i shuf --img2img-strength 1.0`,
prompt "a cat sitting on a chair", seed 42, 512px):
- **Discrimination: cross-ref output corr mean 0.158 / max 0.379** (vs the adapter's 0.93–0.99 collapse).
  STRONGLY discriminates — passes the gate by a mile.
- **Style transfer WITHOUT composition leak:** churchill→clean line-art cat, woodcut→engraving/hatched cat,
  flat_sticker→flat-color sticker cat — all EXCELLENT, no trace of the reference's composition. cyberfika
  (busy paisley) only PARTIAL (style went to the background, cat stayed photoreal) — graphic/decorative
  patterns are harder to read as a global "style."
- CONCLUSION: "in-sequence style-only" is VALIDATED — and achievable TODAY with NO TRAINING by
  content-destroying the reference before the existing in-context path. 3/4 styles excellent.
Artifacts: scratchpad/arch_probe/{*_shuf.png, out_*.png}.

## REFRAMED PLAN — exploit the no-retrain shortcut first, learned encoder as a later quality lever
The multi-week "architectural retrain" is now potentially a preprocessing + wiring task:
1. **Tune content-destruction** (cheap, no train): grid size, multi-scale/averaged shuffles, blur, or
   frequency filtering — to (a) fix graphic styles like cyberfika and (b) maximize style fidelity. Measure
   on the simple-style eval set: discrimination (<0.90), style match, content-leak.
2. **Wire into the web "style" path**: preprocess the reference (content-destroy) → existing in-context.
   This UPGRADES the shipped in-context default from "style+composition" to true "style-only --sref" with
   no model change. Keep a strength/þabout knob.
3. **Validate + ship**: run the full eval (both eval sets) + the discrimination gate + a quality eyeball;
   compare to plain in-context (which leaks composition). If it wins, it's the new --sref.
4. **LATER quality lever (the original Candidate 1):** a LEARNED content-destroyer / style-token encoder,
   if crude preprocessing plateaus (esp. for graphic styles). Now de-risked — we know in-sequence
   style-only works; training just sharpens it. Base-model adapter (Candidate 3) only if even that stalls.

## SHIPPED (2026-06-30, commit 4b898b5) — style-only --sref via content-destruction
Steps 1–3 done. Tuned: grids 6/8/12 all discriminate (mean 0.158–0.175); grid 6 cleanest for rendering
styles (finer adds background fragments); cyberfika-type decorative patterns inherently partial. WIRED:
`content_destroy_png` (PIL patch-shuffle, deterministic) applied to STYLE-mode references before in-context
(web/server.py); composition-mode stays literal; default grid 6, `IRIS_SREF_SHUFFLE_GRID` override.
TESTED: unit test (style→destroyed, composition→not) + full web suite 136 pass. END-TO-END VALIDATED:
live web POST of churchill as a style ref → clean line-art cat, no composition leak (scratchpad/
web_sref_churchill.png). The web "style" upload is now true style-only --sref, no model change.
REMAINING (optional polish): reduce faint background fragments (e.g. mild blur after shuffle, or mask);
formal discrimination-gate run through the web path; the learned style-token encoder as a future quality
lever for graphic styles — now folded into the broader **Pluggable Conditioning Framework**
(`plans/pluggable-conditioning-framework.md`, BACKLOG "PLUGGABLE CONDITIONING FRAMEWORK") as Rail 2 /
Phase 2 alongside a LoRA training pipeline.

## Log
- 2026-06-30: review opened; charter banked; Explore confirmed injection mechanism (side-channel vs
  in-sequence) + no base support; FIRST EXPERIMENT (content-destroyed in-context) VALIDATED the direction
  (corr 0.158, 3/4 styles excellent, no composition leak) → no-retrain shortcut; plan reframed; SHIPPED the
  content-destruction in the web style path + end-to-end validated (commit 4b898b5).
