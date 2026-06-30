# SREF Adapter Retrain — Diagnostic-First Plan (Step 0 execution tracker)

Status: **Step 0 RUNNING** (2026-06-29). Full rationale + branch logic in
`BACKLOG.md` → "SREF ADAPTER RETRAIN — DIAGNOSTIC-FIRST PLAN". This doc tracks execution.

## Goal of Step 0
Decide whether the adapter mode-collapse (cross-ref output corr 0.983 on the champion) is
**conditioning-path bound (architectural)** or **recipe/data bound**, by running the
reference-discrimination gate (`debug/sref_ref_discrimination.py`) on every existing
checkpoint BEFORE spending any training time.

- PASS = max cross-ref output corr < 0.90 AND clearly above the no-adapter floor.
- FAIL (collapse) = max cross-ref corr ≥ 0.90 (reference-independent injection).

## Branch
- **ALL arms (incl. input-norm) collapse** → conditioning path is the suspect (perceiver /
  FiLM / ip-KV). Fix is architectural → Step 1A.
- **Some arms discriminate** → recipe/data/duration matters → Step 1B. Re-rank arms by
  DISCRIMINATION, not the (confounded) sref_score.

## Material on disk (verified 2026-06-29)
- **11 hybrid arms** (`/Volumes/2TBSSD/sref_eval/*/bundle`, cond_mode=hybrid): clean_base,
  clean_concentrate, clean_concentrate_leak (champion), clean_hier, clean_leak, clean_leak025,
  clean_pool9, clean_siglipdn, hybrid_arm, hybrid_hier, hybrid_siglipdown.
  Features: `refs_feat_hybrid_simple/` (8 bins).
- **1 CSD arm**: csd_arm (cond_mode=csd). Features: `refs_feat_csd_simple/` (8 bins).
- **3 legacy siglip arms** (cond_mode absent → siglip): leak1_pshuf, leak2_xref, style_arm.
- **2 input-norm bundles** (`/Volumes/2TBSSD/sref_sweep/{bundle_inputnorm,bundle_confirm_fix}`,
  siglip): the decisive test of whether the IP-ADAPTER-INFER-1 grid fix ever bought
  reference-specificity. Need siglip simple-set features (generated separately).
- Champion ckpts `clean_concentrate_leak/ckpt/step_{0002500,0003000}` are both LATE — no early
  snapshot (within-run early/late curve deferred to a fresh instrumented run, Step 2).

## Run config
4 maximally-distinct refs (churchill_lineart, cyberfika, woodcut, flat_sticker), scale 0.38,
seed 42, 512px, prompt "a cat sitting on a chair".

## Results (Step 0) — 2026-06-30

**VERDICT: ALL arms collapse — 0 discriminate. → conditioning-path / training-signal bound
(architectural), NOT recipe-bound. Proceed to Step 1A.**

Hybrid + CSD arms (scratchpad/sref_discrim_sweep.py; 4 refs, scale 0.38, seed 42, 512px):

| bundle | cond_mode | mean cross-corr | max cross-corr | vs-baseline | verdict |
|---|---|---|---|---|---|
| clean_base | hybrid | 0.991 | 0.997 | 0.551 | COLLAPSE |
| clean_concentrate | hybrid | 0.980 | 0.994 | 0.676 | COLLAPSE |
| clean_concentrate_leak (champion) | hybrid | 0.977 | 0.993 | 0.373 | COLLAPSE |
| clean_hier | hybrid | 0.964 | 0.987 | 0.557 | COLLAPSE |
| clean_leak | hybrid | 0.992 | 0.998 | 0.604 | COLLAPSE |
| clean_leak025 | hybrid | 0.975 | 0.990 | 0.576 | COLLAPSE |
| clean_pool9 | hybrid | 0.828 | 0.915 | 0.531 | COLLAPSE |
| clean_siglipdn | hybrid | 0.969 | 0.995 | 0.622 | COLLAPSE |
| hybrid_arm | hybrid | 0.986 | 0.995 | 0.600 | COLLAPSE |
| hybrid_hier | hybrid | 0.961 | 0.994 | 0.625 | COLLAPSE |
| hybrid_siglipdown | hybrid | 0.969 | 0.996 | 0.511 | COLLAPSE |
| csd_arm | csd | 0.966 | 0.980 | 0.223 | COLLAPSE |

Siglip arms incl. the input-norm grid-fix (scratchpad/sref_discrim_siglip.py):

| bundle | cond_mode | mean | max | vs-base | verdict |
|---|---|---|---|---|---|
| **bundle_inputnorm** (grid-fix) | siglip | 0.998 | **0.999** | 0.658 | COLLAPSE |
| bundle_confirm_fix | siglip | 0.995 | 0.997 | 0.392 | COLLAPSE |
| style_arm | siglip | 0.972 | 0.991 | 0.550 | COLLAPSE |
| leak1_pshuf | siglip | 0.972 | 0.983 | 0.496 | COLLAPSE |
| leak2_xref | siglip | 0.969 | 0.992 | 0.543 | COLLAPSE |

**TOTAL: 17/17 checkpoints collapse — across hybrid + CSD + siglip, every recipe, and BOTH
sides of the 2026-06-18 input-norm grid fix.** bundle_inputnorm at 0.999 is the smoking gun:
input-norm fixed cross-TOKEN collapse (the grid, within one ref) but NEVER touched cross-
REFERENCE collapse (different refs → same injection). Two different axes; only the first was
ever fixed.

Key reads:
- Every recipe lever (data concentration, leak penalty ×weights, hierarchical inject, pooling,
  SigLIP downscaling) collapses → recipe is NOT the cause; prior sref_score ranking is noise.
- **csd_arm collapses too (0.980)** — a different conditioning path (FiLM queries, no SigLIP
  perceiver) → the collapse is NOT perceiver-specific. Points at the K/V injection and/or the
  training signal (the loss rewards a generic injection; nothing punishes reference-agnostic output).
- vs-baseline 0.22–0.68 → the adapters DO transform strongly, just reference-independently
  (not inert; rules out "scale too low").

## Step 1A FIX CAMPAIGN (2026-06-30) — running C/A; repulsion was a dead end
Discrimination results (max cross-ref output corr; PASS <0.90):
- champion (collapsed) 0.993 · **rank-only @300 0.926 (best so far)** · aggressive repel 0.939 · gentle repel 0.945.
- BOTH x0-repulsion tunings made it WORSE → `style_repulsion_loss` mechanism is counterproductive
  (train/infer mismatch: ref-B's V in ref-A's shared Q context). ABANDONED.
- **C (longer rank-only) gate @600: V-injection cross-ref cosine still 0.965** (≈champion 0.953) →
  rank-only does NOT decorrelate V; it raises V *rank* (capacity) but the V *vectors* stay collinear.
  This EXPLAINS the rank-only plateau and motivates A. C stopped.
- **A (`vproj_decorr_loss`, commit 95c2c53): RUNNING** — rank 2.0 + decorr 1.0 (margin 0.5) directly
  penalizes that 0.965 V cosine. Watch: decorr_loss ↓ + offline V cosine ↓ + discrimination <0.90,
  loss STABLE (unlike the repulsion's divergence). `/Volumes/2TBSSD/sref_eval/run_A_decorr/`.

## Step 1A.1 RESULT (2026-06-30) — ROOT CAUSE: `to_v_ip` is low-rank → V near-constant → collapse
Tool: `debug/sref_kv_rank_audit.py` (offline; cond-encoder → ip_embeds → K/V on N refs; cross-ref
cosine per stage + SVD stable_rank of the weight matrices).

Stage cross-ref cosine (cos→1 = refs stop mattering):
- Champion (hybrid): raw SigLIP 0.348 → **ip_embeds SigLIP-half 0.407 (discriminates! perceiver is
  NOT the site)** | ip_embeds CSD-half 0.978 (FiLM rank-1) | K 0.864 | **V 0.953**.
- bundle_inputnorm (siglip): raw 0.332 → ip_embeds 0.916 | K 0.917 | **V 0.998** (var_ratio 0.035).

Weight stable_rank (3072=full, low=collapsing):
- Champion to_k_ip 104/255/311 (blocks 5/12/24) vs **to_v_ip 5.9/6.7/18.5**.
- bundle_inputnorm to_k_ip ~770 vs **to_v_ip ~25**. (Block 0 ~770 for both — double-block, never engages.)

Mechanism: inject = softmax(Q·Kᵀ)·V; a rank-~6 `to_v_ip` projects every ref onto the same few
directions → V reference-independent → output collapse regardless of perceiver/K. K stays full-rank
(adapter looks at refs differently) but V collapsed (injects the same thing). Universal across cond_modes
→ explains all 17 collapses. The easy minimum of a loss that never rewards reference-specific V.

## Step 1A IMPLEMENTATION

### Done (committed 30293b1) — loss primitives + hermetic tests (no training, reversible)
`train/ip_adapter/loss.py`:
- `style_repulsion_loss(x0_a, x0_b, margin)` — CAUSE fix. Two different-style refs at the SAME
  prompt/noise must produce different AdaIN per-channel style stats; hinge penalizes only collapse
  (d²<margin). Complements `content_leak_loss` (content via instance_norm; this = the style stats it
  leaves free).
- `vproj_rank_penalty(W, u)` — SYMPTOM fix. Spectral penalty σ1²/‖W‖_F² on `to_v_ip_stacked`; σ1 via one
  warm-started power-iteration step (u = persistent state). Minimizing flattens the spectrum → raises
  stable_rank.
- `style_stats` helper. 10 tests in `test_loss.py` (power-iter≡SVD; rank-1→1, orthogonal→1/n; gradients);
  full loss suite 49 passed.

### Done (commit bac44e5) — rank penalty WIRED + smoke-validated
`vproj_rank_penalty` wired into both trainer loss paths behind `training.vproj_rank_weight` (threads
persistent power-iter state `_rank_u`; no signature change). `sref_kv_rank_audit.py --ckpt` added.
SMOKE (warmstart from collapsed champion, rank_weight 2.0, 300 steps, hot cached data,
`/Volumes/2TBSSD/sref_eval/smoke_rank/`):
- Wiring VALID: trains clean, no MLX wedge, loss stable (avg 1.24→0.75), mlx_mem peak 24.8 GB.
- to_v_ip stable_rank ROSE: baseline 5.9/6.8/18.7 → step300 **15.1/24.1/43.7** (2.3–3.5×, still climbing).
- Discrimination (export→gen-gate): cross-ref corr mean 0.977→**0.886**, max 0.993→**0.926**. Moved the
  right way (~0.09) but STILL FAILS <0.90. → capacity↑ helps but doesn't FORCE use. Repulsion is next.

### Repulsion wired (commit 717a1f9) — but did NOT separate refs as tuned (2026-06-30)
Ring buffer of recent refs → cheap 2nd prediction via `_pred_from_embeds` on the shared Flux state →
`repel_loss_weight·style_repulsion_loss`. Combined smoke (warmstart champion, rank_w 2.0 + repel_w 0.5,
margin 1.0): repel was ACTIVE (repel_loss 0.98→~0.5–0.9) but **destabilized** — loss CLIMBED 0.6→2.0
(repulsion fights reconstruction with no content anchor). step150 discrimination cross-ref **0.904/0.939**
— WORSE than rank-only@300 (0.886/0.926); styled-vs-base 0.218 (transforms harder but all refs still
→ similar transform). So the repulsion disrupted WITHOUT disentangling. Killed (diverging).
LEADING HYPOTHESES for why it underperforms:
1. **Train/infer mismatch in the repel term:** x0_other = ref-B's V injected into ref-A's Q + h_final
   (shared precomputed Flux state). So we separate "B's V vs A's V WITHIN A's context" — which may not
   transfer to inference where B runs in its OWN context. (The leak null-pass reuses the state too, but
   null=zeros is benign; a real 2nd ref is not.)
2. Over-weight/margin (0.5/1.0) → degenerate disruption, not clean separation. Gentle run testing now.
3. style_stats(x0 latent) may be a weak repel target vs operating directly on the V injection (the
   measured collapse site).
NOW TESTING: gentle config (repel_w 0.1, margin 0.3, rank_w 2.0) — `/Volumes/2TBSSD/sref_eval/smoke_gentle/`.
Watch: loss STABLE (~0.7, unlike the diverging 2.0) AND cross-ref corr < rank-only's 0.886.
FALLBACKS if gentle also fails to pass <0.90:
- Redesign repel to act on V directly: decorrelate to_v_ip outputs across buffered refs (penalize
  cross-ref V cosine — the exact quantity Step 1A.1 measured at 0.95–0.998), no x0 round-trip.
- Give the 2nd ref its OWN Q context (extra correct-forward-Q pass) to kill the train/infer mismatch.
- Longer RANK-ONLY run (rank-only is the best so far at 0.886; it may cross 0.90 with more steps —
  cheap, stable, no repulsion risk).

### (superseded design note) wiring the CAUSE fix: `style_repulsion_loss`
batch_size=1 + the cheap `correct_forward_q` path (frozen-Flux state precomputed once, reused by
`_pred_from_embeds`). So a SECOND reference's prediction at the same noisy latent is nearly free. Plan:
1. **Second-reference source for repulsion (recommended: memory bank).** Keep a small ring buffer of the
   last K steps' reference features (cond_features). Each cond step, draw one buffered ref, compute its
   `x0_other = _pred_from_embeds(get_image_embeds(buffered))` on the SAME precomputed Flux state, add
   `repel_w * style_repulsion_loss(x0_pred, x0_other)`. No dataloader change; reuses the shared Flux work.
   (Alt: have the loader yield a random different-style 2nd ref per step — cleaner data, but a loader change.)
   NOTE: the buffered/2nd ref must be a RANDOM other image (≠ the dataset's same-style neighbor — that
   would punish same-style images for sharing style).
2. **Rank penalty:** thread persistent `u` [25,3072] through the compiled step; add `rank_w *
   vproj_rank_penalty(adapter.to_v_ip_stacked, u)`; return `u_next`.
3. **Hyperparameters (defaults to validate, then tune):** repel margin ~ (set from the latent's style-stat
   scale), repel_w ~0.1, rank_w ~0.01. Apply repulsion on COND steps only.
4. **Risk:** touches the compiled hot loop (MLX wedge history). Validate with a SHORT smoke run (~300 steps)
   watching: to_v_ip stable_rank (sref_kv_rank_audit.py) ↑ and cross-ref V cosine ↓ BEFORE a full run.

### Instrument every retrain
Per checkpoint: `sref_kv_rank_audit.py` (cheap, offline — leading indicator: to_v_ip stable_rank↑ + V
cosine↓) AND `sref_ref_discrimination.py` (gen-gate; promote ONLY on PASS, not sref_score). Web stays on
in-context (IRIS_SREF_ADAPTER off) until a checkpoint PASSES discrimination, then flip IRIS_SREF_ADAPTER=1.
