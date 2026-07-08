# Shard Data Composition (reference — do not lose)

Durable notes on what the training corpus actually contains, measured 2026-07-08 from shard provenance
sidecars + the CSD style index. Written because this keeps getting re-derived.

## The unified corpus
- **`/Volumes/16TBCold/shards/` = 1280 shards × 5000 images ≈ 6.4M images.** This is the single source of
  truth for training data — the **unified build** (`build_shards.py --sources
  converted/{laion,journeydb,coyo,wikiart}`), NOT four separate pools.
- `/Volumes/16TBCold/converted/{laion,journeydb,coyo,wikiart}/` are the pre-build **source** tars
  (intermediate); `/Volumes/16TBCold/converted/journeydb` (200 tars) is one of these — it is **already
  folded into the 1280**, not a separate uncovered set. Do not treat converted/ as extra data.
- Provenance: each shard has a **sibling `<stem>.provenance.json`** (NOT inside the tar) listing the
  contributing source tars by `type` (jdb/journeydb, coyo, laion, wikiart). `shard_source.source_for_tar`
  reads it. **No per-source record counts** are stored — only which source tars fed the shard. Hot-pool
  copies (`baseline_pool_hot`) LOST their provenance sidecars → `source_for_tar` returns "unknown" there;
  the COLD shards have correct labels.

## Source distribution (all 1280 shards, exact)
| provenance tag | shards | note |
|---|---:|---|
| `journeydb` (pure) | **744** | Midjourney generations — highly stylized AI art |
| `coyo` (pure) | **413** | web image–text pairs (photos/graphics) |
| `coyo+laion+wikiart` (mixed) | **95** | laion-heavy in practice (provenance shows many laion source-tars per shard) |
| `journeydb+wikiart` (mixed) | **28** | journeydb + a wikiart minority |

Derived per-source presence (a shard can involve >1 source):
- **journeydb**: 772 shards (744 pure + 28 mixed) — the dominant source (~60%).
- **coyo**: 508 shards (413 pure + 95 mixed).
- **laion**: only the 95 `coyo+laion+wikiart` shards — **no pure laion shards** (but likely the bulk of
  those 95 by record count).
- **wikiart**: 123 shards (95 + 28) — **never pure, always a diluted minority** mixed with coyo/laion/jdb.
  Cannot be isolated at the record level (provenance is per-shard, combined-source, countless).

## Style signal (from the CSD index)
- **CSD style index**: `/Volumes/16TBCold/precomputed/style/v1_csd/` — a **200-records/shard subsample**
  over all 1280 shards = **255,987 CSD-ViT-L vectors** ([768] f16, L2-normalised), one npz per shard
  (`style_precompute.py`, encoder `csd_vit_l_style_v1`, 2026-06-11). Every shard covered (one at 187/200).
  Per-shard rankings: `v1_csd/shard_report.json` (from `style_shard_report.py`, tau 0.6).
- **Pool is broadly style-rich**: `pool_pair_rich 0.864` (86% of images have ≥3 strong same-style
  neighbors at cos ≥ 0.6). Every shard ≥ 0.635; none isolated.
- **Richest style-training shards = journeydb** (pair_rich ~0.99, high diversity) — Midjourney is highly
  stylized, so it yields the most same-style/content-diverse pairs. **Lowest = coyo** (~0.635, web photos).
- **Painterly / fine-art is the thin spot.** Corpus-wide CSD density to painterly eval refs (impressionism/
  baroque/cubism) tops out at only **~0.62 (top-50)** — barely above the 22-shard hot pool's 0.59 — because
  wikiart is diluted and unisolable. The corpus does **general/AI-art/illustration** styles well and **fine-
  art painting** poorly.

## Implications for style-transfer work (SREF)
- **General style transfer**: the corpus is a strong substrate — use the journeydb-rich shards + the
  style-neighbor graph (`shard_report.json` connectivity edges) for style-paired training. The per-style
  LoRA mechanism works (BACKLOG SREF-LEARNED-STAGE1 / retrieval-hybrid Phase 0).
- **Painterly specifically**: the corpus cannot supply dense, labeled painterly clusters (wikiart too thin).
  Acquiring the clean public **WikiArt** dataset (movement-labeled) remains the lever for those styles —
  see `plans/sref-phase1-data-scoping.md`.
- **No re-index needed**: the subset style compute of all shards is DONE. A FULL per-image CSD (5000/shard)
  would only be worth it to surface more painterly needles from the diluted wikiart — low yield; prefer
  clean WikiArt.

## Other data locations (for orientation)
- Hot working pool: `/Volumes/2TBSSD/baseline_pool_hot` (22 shards, provenance-less copies).
- Precompute caches (cold): `/Volumes/16TBCold/precomputed/{vae,qwen3,siglip,style}/`.
- Precompute caches (hot, active versions): `/Volumes/2TBSSD/precomputed/{vae/v_2232c1, qwen3/v_059443,
  siglip/v_336c6e, style/…}`.
