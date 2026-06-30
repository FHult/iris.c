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

## Next → Step 1A (architectural; see BACKLOG)
Cheapest-first, each gated by re-running this discrimination test before a full run:
1. K/V injection RANK audit (offline, no training): variance of to_k_ip/to_v_ip across refs —
   is diversity lost at the perceiver or the projection? csd collapse says check the projection.
2. Discrimination-aware TRAINING signal (contrastive/repulsion term punishing ref-agnostic output)
   — most likely missing ingredient; affects ALL cond_modes, consistent with universal collapse.
3. Per-query diversity regularization in the perceiver (hybrid path).
4. CSD/FiLM contribution check + rebalance.
Instrument the retrain with frequent checkpoints + per-checkpoint discrimination (Step 2).
