# SREF Retrieval-Hybrid — Phase 1 Data Scoping

Status (2026-07-08): **SCOPING — needs a data-source decision before any big job.** Phase 0 proved the
per-style LoRA mechanism transfers painterly style; the cluster-tightness A/B proved within-pool
selection is exhausted (top-100 no better than top-250). The bottleneck is **dense, movement-specific
style clusters**, which the current pool cannot supply. This note inventories the options and their cost.

## Inventory (measured 2026-07-08)
- **Existing corpus**: `/Volumes/16TBCold/shards` = **1280 shards × 5000 img = 6.4M images**;
  `/Volumes/16TBCold/converted/journeydb` = 200 shards (~1M, Midjourney — stylized but NOT paintings);
  `/Volumes/2TBSSD/baseline_pool_hot` = 22 shards (109k, already CSD-indexed, **painterly-sparse**:
  top-50 CSD cos only 0.53–0.59 for every painterly eval style).
- **No standalone art dataset** on cold (`coyo/`, `converted/`, `shards/`, `raw/` exist; no `wikiart/`).
- **Source labels are UNRELIABLE**: `shard_source.source_for_tar` returns "unknown" for the hot shards;
  `shard_index.db` / `shard_scores.db` on SSD are empty. WikiArt, if present, is mixed into the numbered
  shards with no cheap way to find it. → we cannot filter the corpus to "art shards" by label.
- **CSD throughput**: 41 img/s (encode only, warm). Cold reads are IO-bound (~31 s/shard just to
  enumerate members; ~2 TB of tars total, must be staged in batches — the SSD is 2 TB).

## Option A — Mine the existing corpus (CSD-index broadly)
CSD-index the 1280 cold shards, cluster across all, find the densest painterly clusters wherever they are.
- **Cost**: compute ~6.4M / 41 = ~43 h; but dominated by cold-HDD IO + batched staging of ~2 TB →
  realistically **multi-day**, babysat, resumable.
- **Yield**: LOW. The corpus is photo/digital-art heavy; painterly is genuinely sparse (the hot-pool
  measurement already showed this). We'd read millions of photos to harvest a little art, and the art we
  find is unlabeled (must name clusters by hand). High cost, low signal-to-noise.

## Option B — Acquire WikiArt directly (RECOMMENDED)
The public **WikiArt** dataset (HuggingFace, ~81k images, labeled by **~27 art movements** — Impressionism,
Cubism, Baroque, Expressionism, Art Nouveau, …) is exactly the content we lack, already
movement-labeled — which maps 1:1 onto the eval-set painterly styles.
- **Why it's better**: labels ARE the clusters → skip CSD-mining entirely; density is guaranteed (every
  Impressionism image is impressionism, cos ≫ 0.6); clean provenance; one download vs mining 6.4M images.
- **Cost**: download ~25–50 GB (curl, per the HF-download practice; needs approval); precompute 81k images
  (VAE + Qwen3 + CSD) at hot-SSD rates ≈ a few hours; then train one library LoRA per movement (~1.5 h
  each on M1, parallel/overnight, caffeinated). NEVER train from cold — stage the selected movement
  subsets hot first.
- **Deliverable**: a movement-labeled style-LoRA library + CSD centroids → the retrieval index the
  Phase-0 mechanism plugs straight into.

## Option C — Hybrid (belt-and-suspenders)
WikiArt for the painterly/fine-art movements (Option B) + keep the existing CSD pipeline for
graphic/photographic styles the corpus already covers well (band-control already handles graphic; a LoRA
could sharpen it). Defer the broad corpus mine (Option A) unless a wanted style is absent from WikiArt.

## Recommendation
**Option B.** The root cause is painterly sparsity; WikiArt fixes it at the source, cheaply and with clean
movement labels, instead of spending multi-days mining a photo-heavy corpus for sparse, unlabeled art.
It also lets us validate the retrieval-hybrid end-to-end fast: pick 3–4 movements matching the eval refs
(impressionism, cubism, baroque, expressionism), train a LoRA per movement, and re-gate — a *tight,
labeled* cluster is the exact condition the A/B showed we were missing.

## Immediate next action (pending approval)
1. Confirm the WikiArt source/URL + download (curl; ~25–50 GB; approval required — CLAUDE.md).
2. Convert to shards + precompute VAE/Qwen3/CSD (hot); verify per-movement CSD density ≫ the pool's 0.53.
3. Train one impressionism-movement LoRA and re-gate against the impressionism eval ref — the direct
   re-test of the A/B's missing condition. If styleCSD Δ jumps (tight labeled cluster → specific style),
   build out the library; if not, the mechanism/metric needs rethinking before scaling.
