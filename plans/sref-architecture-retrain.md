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

## Architecture review — open questions (being answered now)
- [ ] EXACT adapter injection mechanism in training (`_flux_forward_with_ip*`, `get_kv_all`): additive
  per-block SDPA over IP tokens × scale? (Explore agent mapping.)
- [ ] EXACT in-context mechanism (C inference: where do reference tokens enter the sequence; concat order
  [TEXT,IMAGE]). Confirms the "in-sequence vs side-channel" distinction.
- [ ] Base-model adapter training support — exists or needs new code (CFG dual-pass, guidance).
- [ ] Why the injection must be low-scale (does high scale destroy content?).

## Candidate 1 design (DRAFT — fill after the review answers land)
_Token format, concat point, how the frozen base attends, content/style loss split, train↔infer parity
plan, first-experiment config + discrimination gate._

## Execution gates (every architecture attempt)
- train↔infer PARITY fixture + prod-flag compile + `make mps` (AGENT protocol) before trusting any result.
- cached-mode only (live-encode segfaults MLX — BUGS MLX-1); never train from cold storage (AGENT #6).
- Promote ONLY on `sref_ref_discrimination.py` max cross-ref corr <0.90 AND a quality eyeball (corr can
  move while quality degrades — Option B showed this). Web stays on in-context until a ckpt PASSES.

## Log
- 2026-06-30: review opened; charter banked; Explore agent mapping injection paths + base support.
