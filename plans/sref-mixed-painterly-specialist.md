# SREF Mixed-Data Painterly Specialist (SREF-STYLE-ROUTER step 3, attempt 3)

**Status:** RUN 1 LAUNCHED (2026-07-20). 40:60 mix, generic projector. ~11 h; gate at 7000.

## Run 1 — LAUNCHED 2026-07-20

- Config `train/configs/sref_painterly_specialist_v3_mixed.yaml`; caches built by
  `debug/build_mixed_painterly_caches.py` → `/Volumes/2TBSSD/mixed_painterly_v3/` (vae256/qwen3 58,444
  @ 40.1% WikiArt, 19 CSD bundles, merged neighbors 132,697). Projector = generic (see resolution below).
- **Smoke (100 steps, real config) PASSED:** reaches steps (no cache-miss death); peak **16.6 GB** (no
  swap); **0.18–0.21 it/s → ~11 h** for 8000 steps; pair term `loss_b` 0.75→0.27 dropping BELOW the
  0.6931 ref-blind floor (learning reference-discrimination, not collapsing); gnorm 6.3→0.6.
- Log `/Volumes/2TBSSD/logs/sref_painterly_v3_mixed.log`; checkpoints every 1000 →
  `checkpoints/sref_painterly_specialist_v3_mixed/`.
- **GATE at 7000/8000:** `debug/sref_gate_joint.py` (LoRA r64 + csd_mod render) →
  `debug/sref_scorecard.py --score-only`. PASS = painterly styleCSD Δ ≥ ~0.157 AND promptAdh recovers
  (≥ ~0.15) AND cross-ref output corr < 0.90. Fast-follow if WikiArt style is weak: train a multi-corpus
  projector (the generic holds only 76% of the WikiArt gap vs 84% diverse).

---

**Original scope (2026-07-20), for the trail:**

## Hypothesis

A painterly specialist can beat the generic v5.3.0 adapter on *painterly* refs **iff** it is
trained on WikiArt **mixed with diverse content**, not WikiArt alone.

- v2 (pure WikiArt, 23,444 pairs) got **strong style** (styleCSD Δ 0.157 vs generic 0.058 in that
  eval) but **overfit → destroyed prompt-following** (promptAdh 0.084; subject gone even at α=0).
  Lesson (SREF-STYLE-ROUTER): narrow-data specialization loses the subject; diverse data preserves it.
- The mixed run keeps v2's painterly-style lever (within-movement WikiArt pairs) while re-introducing
  diverse content so the LoRA does not collapse onto the painterly manifold.

## Decision criterion (the gate that decides go/no-go on the router)

Score with `debug/sref_scorecard.py --score-only` on the 32-ref eval set + the collapse gate:

1. **Painterly styleCSD Δ ≥ v2's ~0.157** (keep the style win), AND
2. **promptAdh recovers toward the generic's** (v2 was 0.084; target ≥ ~0.15 — subject legible), AND
3. **cross-ref output corr < 0.90** (not collapsed — SREF-CHAMPION-COLLAPSE gate, mandatory).

If (1)∧(2)∧(3): the specialist is real → wire the CSD-classify router (painterly → specialist,
else → generic). If style drops to generic levels: mixing over-diluted → the specialist is not the
lever, **generic stands** and the router is closed for good.

## Verified data inventory (on disk 2026-07-20)

WikiArt (style lever) — **fully intact from v2**:
- Raw: 41 tars `/Volumes/16TBCold/converted/wikiart/` (~54 GB, cold — source only).
- Staged hot: `/Volumes/2TBSSD/wikiart_pool_hot/` (12 shards).
- Caches: `precomputed/vae_wikiart256/` (23,444), `precomputed/qwen3_wikiart/` (23,444),
  `/Volumes/2TBSSD/wikiart_csd/` (14 shard-level), `/Volumes/2TBSSD/wikiart_neighbors.sqlite`
  (23,444 within-movement pairs).
- WikiArt latent→CSD projector: `checkpoints/latent_csd_wikiart/latent_csd_projector.safetensors`
  (val cos 0.83, geometry 98% retained). **Data-specific — must be the WikiArt projector, not the
  generic's** (reusing the generic's corrupted v1).

Diverse (content preservation) — the generic's data:
- Caches: `precomputed/vae_sref256px/` (109,254), `universe_csd_full/` (24 shard packs).
- Its within-look neighbor DB (generic joint recipe) — locate/confirm before merge.
- Diverse shard tars — the pool the generic trained from (confirm hot location before merge).

## Mix mechanism (no trainer code change needed)

`probe_joint_contrastive.py` accepts an explicit `data.shard_paths` **list** (else globs
`shard_path/*.tar`), so mixed shards are a config change. The caches are single dirs, so the build is:

1. **Unify caches** — one dir per encoder holding BOTH sources' records (symlink WikiArt + diverse
   npz into `precomputed/{vae256,qwen3,csd}_mixed/`; record ids must not collide — verify).
2. **Merge neighbor DBs** — `wikiart_neighbors.sqlite` (23,444) ∪ the diverse look-neighbor DB into
   one `mixed_neighbors.sqlite` (both are within-style pairs — the correct signal for each source).
3. **`data.shard_paths`** = 12 WikiArt tars + K diverse tars, K chosen to hit the target ratio.
4. **Projector — RESOLVED 2026-07-20: use the GENERIC projector (`checkpoints/latent_csd/`).**
   Cross-corpus geometry-gate test (`scratchpad/proj_geo_eval.py`, gap = cos(proj,neighbour) −
   cos(proj,foreign), true gap ~0.60–0.63):
   - generic proj: **diverse 84% retain (gap 0.53), WikiArt 76% retain (gap 0.46)** — usable on BOTH.
   - WikiArt proj: WikiArt 98% but **diverse only 45%** (gap 0.28) — fails the majority corpus.
   The generic projector is the best SINGLE projector for the mix (it is at home on the 60% diverse
   majority and still holds a strong WikiArt gap). The v1 "generic-proj-corrupts-WikiArt" failure was
   a *pure-WikiArt* run compounding the 24% gap loss; a diverse-majority mix is far more forgiving. No
   projector training needed for run 1. **Fast-follow only if** the WikiArt contrastive underperforms
   at the 7000-step gate → then train a multi-corpus projector (trainer needs a small multi-source
   `build_index` change; both corpora are 32×32 latents so batches stack cleanly).

## Mix ratio — THE decision

WikiArt has 23,444 pairs. Over-represent it vs its natural rarity, but keep enough diverse content
that prompt-following survives. Candidate starting points:

| Ratio (WikiArt : diverse) | Rationale |
|---|---|
| 50 : 50 | Aggressive style; risk of residual overfit. |
| 40 : 60 | Balanced — recommended first shot. |
| 30 : 70 | Content-safe; risk style dilutes to generic levels. |

Recommend **40:60** first (≈23k WikiArt + ≈35k diverse). One knob, one run; re-roll if the gate
misses on the style side (→ 50:50) or the content side (→ 30:70).

## Config (delta from `sref_painterly_specialist.yaml`)

Keep the **winning v1 recipe**: LoRA r64, t_range [700,950], contrastive_weight 1.0, infonce_tau 0.1,
foreign_queue 4096, null_style_prob 0.1, guidance 3.5, 8000 steps, bucket 256, grad_checkpoint.
Change: `shard_paths` (mixed list), the `*_mixed` cache dirs, `mixed_neighbors.sqlite`, the resolved
projector, `checkpoint_dir: checkpoints/sref_painterly_specialist_v3_mixed`.

## Cost & risk

- Build (cache unify + neighbor merge + projector check): ~1–2 h, mostly I/O; no new GPU precompute
  (all encoders already cached for both sources).
- Train: ~8000 steps at the v2 rate (~0.17 it/s → **~13 h**). caffeinate mandatory.
- Payoff **uncertain** — could confirm the router or definitively close it. Either is a real result.

## Launch sequence (once ratio is chosen)

1. Build `*_mixed` caches + `mixed_neighbors.sqlite`; verify no record-id collisions.
2. Projector fit/geometry-gate check on mixed latents.
3. Write `train/configs/sref_painterly_specialist_v3_mixed.yaml`.
4. `caffeinate -dimsu train/.venv/bin/python train/lora/probe_joint_contrastive.py --config ...`
5. Gate at 7000/8000 with `debug/sref_gate_joint.py` + `debug/sref_scorecard.py --score-only`.
