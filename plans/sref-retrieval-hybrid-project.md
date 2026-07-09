# SREF Retrieval-Hybrid Instant-LoRA — Project Scope

Status (2026-07-08): **PHASE 0 = QUALIFIED PASS (mechanism works; data-gated).** Trained one
impressionism-seeded per-style LoRA (distilled, rank 16, 400 steps, cluster cos 0.45;
`train/lora/curate_style_seeded.py` + `lora_painterly_v1.yaml` → `…/lora_lib/painterly_impressionism.safetensors`)
and gated it (`sref_scorecard.py --no-input-ref`). KEY RESULT: **the LoRA MECHANISM WORKS** — at
`--lora-scale 2.0` the output is CLEARLY painterly/digital-painting (soft brushed sky, illustrated,
painted signature), a capability band-control AND the learned encoder both lacked (weight-space edit is
always active → not subject to the "never learns the signal" failure). corr(out,baseline)=0.88 at scale 1.0
(NOT inert, unlike the encoder's 0.98–0.999). BUT it's GENERIC painterly, not impressionism SPECIFICALLY:
styleCSD Δ vs the impressionism ref = +0.011 @ scale1.0 (edges band-control's 0.009) but −0.040 @ scale2.0
(moves toward digital-painting, AWAY from impressionism's specific CSD). ROOT CAUSE: the 22-shard hot pool
is PAINTERLY-SPARSE (per-ref top-50 CSD cos only 0.53–0.59 for ALL painterly styles), so "250 nearest
impressionism" was a grab-bag of digital art averaging to a generic painterly look — the LoRA faithfully
learned its cluster; the cluster wasn't impressionism. → **The bottleneck is CLUSTER QUALITY / DATA, not
the mechanism** (vindicates the "style-compute more/art-rich shards" instinct). Montage artifact of the
result exists. The mechanism go/no-go is GREEN; the project is now a data-curation problem.

**CLUSTER-TIGHTNESS A/B (2026-07-08, done):** retrained the SAME LoRA on a TIGHTER top-100 cluster
(mean cos 0.53 vs 0.45), everything else identical (`lora_painterly_tight_v1.yaml`,
`…_tight_subset.json`). Re-gated impressionism at scales 1.0/1.5/2.0. Result: styleCSD Δ **+0.0038 @1.0 /
−0.051 @1.5 / −0.024 @2.0** — NO improvement over loose (+0.0114 / −0.040); if anything slightly LOWER,
and the ×2.0 image is *less* painterly (near-photoreal). CONCLUSION: **within-pool tightening is NOT the
lever.** You cannot select your way to an impressionism cluster that isn't in the pool — the 22
photo-heavy shards are painterly-sparse, so "nearest impressionism" = bluish landscape photos, and a
tighter selection just picks more of those. → **The lever is BROADER, ART-RICH DATA.** NEXT (the real
Phase-1): CSD-index art-rich shards (WikiArt especially; more of the corpus) to find DENSE, genuinely
specific style clusters, then stage+precompute the selected clusters and train library LoRAs from them
(the never-train-from-cold cost the user already flagged). Within-pool selection knobs are exhausted.

Original proposal below.

Status (2026-07-08, proposal): **PROPOSED.** The learned-encoder direction is KILLED (three decisive
negatives — BACKLOG `SREF-LEARNED-STAGE1`; the in-sequence style-token pathway never binds because
the flow-matching input is a noised target, so conditioning is never forced). This is the pragmatic
replacement: build the "instant style" UX out of the **two mechanisms that demonstrably work on this
stack** — per-style LoRA (weight-space) + the CSD style index. Written to be executed by a fresh
session and to be gated at every step by the EXISTING scorecard on the ACTUALLY-SHIPPED C path.

## 1. Thesis
No single universal style encoder. Instead: **retrieve** the nearest pre-trained style LoRA(s) for a
user's reference (by CSD style similarity), **blend** them, and **apply** via the existing
`iris --lora` path. Per-style LoRA already transfers painterly/graphic styles (weight-space, full
25-block coverage); CSD already discriminates style (index gate 0.569, well under 0.7). The only new
work is the library + retrieval + blend + web wiring. **No train↔infer gap** — the LoRA path is
already in the C binary, so `debug/sref_scorecard.py` gates the real thing.

## 2. Why this works where the learned encoder didn't
- Per-style LoRA edits WEIGHTS (always active, every denoising step, every noise level) — it does not
  depend on the model learning to attend to a weak in-sequence signal. This sidesteps the exact root
  cause that killed the learned encoder (SREF-LEARNED-STAGE1 root cause).
- CSD is content-invariant by construction → nearest-LoRA retrieval keys on STYLE, not content.
- The band-control rail (shipped) stacks on top for composition suppression; KV-reuse for speed.

## 3. Architecture
```
BUILD (offline, once):
  hot pool CSD (.npz, exists) ─► K diverse dense style clusters (curate_style_subset, generalised)
      └─► per cluster: train_lora → export_lora_diffusers → style_lib/<id>.safetensors
      └─► per cluster: CSD centroid (mean of cluster vectors) → style_lib/index.npz  (id → 768-d)

USE (per request):
  user ref image ─► CSD encode (csd_mlx) ─► cosine top-K' over index.npz
      └─► blend the K' LoRAs by softmax(sim/τ)  (rank-concat: merged_A=[wᵢ·Aᵢ], merged_B=[Bᵢ])
      └─► write merged.safetensors ─► iris --lora merged.safetensors --lora-scale s
      └─► (optionally stack band-control --sref-shf/--sref-slf for composition suppression)
```
Blend correctness: `Σ wᵢ·scaleᵢ·BᵢAᵢ` is represented EXACTLY by stacking rows of A (folding `wᵢ` in)
and cols of B → one rank-(Σrᵢ) LoRA. The existing single-LoRA `iris_load_lora` applies it unchanged.

## 4. What exists vs what's new
EXISTS (reuse): `train/style_encoder/csd_mlx.py` + `/Volumes/2TBSSD/models/csd_vit_l_style.safetensors`;
per-shard CSD `/Volumes/2TBSSD/sref_eval/style_cache/*.npz`; `train/lora/curate_style_subset.py`
(single densest cluster → rec_ids); `train/lora/train_lora.py` + `train/lora/export.py`
(`export_lora_diffusers`); C inference `iris --lora PATH --lora-scale N` (`iris_load_lora`, main.c)
and daemon JSON `lora`/`lora_scale`; web LoRA download/apply plumbing; the gate
`debug/sref_scorecard.py` (+ `sref_eval_set.json`).
NEW (build): (a) K-cluster curation (generalise curate_style_subset to emit K diverse clusters, not 1);
(b) a library builder that loops curate→train→export→centroid; (c) a retrieval+blend module
(`train/lora/style_retrieve.py`: CSD(ref) → top-K' → rank-concat blend → merged.safetensors);
(d) web wiring (ref upload → retrieve+blend → daemon lora path); (e) optional C multi-LoRA (only if
offline blend proves insufficient — MVP avoids it).

## 5. Phases + gates (kill early, gate on the scorecard's painterly number = the whole point)
| Phase | Deliverable | Gate |
|---|---|---|
| **0. Component proof (CHEAP, do FIRST)** | Train ONE painterly-style LoRA (curate densest painterly cluster → train_lora → export), apply `iris --lora`, run `sref_scorecard.py`. | A single trained LoRA's **painterly styleCSD Δ must beat band-control's 0.009** (and ideally approach graphic-level transfer) with leak held. If a hand-picked per-style LoRA can't beat the bar, the whole premise is wrong — STOP. |
| **1. Retrieval fidelity** | `style_retrieve.py`: CSD(ref) → nearest library LoRA. Build a small library (K≈8–16 clusters). | For held-out refs, the retrieved LoRA's style matches the ref (CSD sim of retrieved-centroid vs ref above a random-LoRA baseline); scorecard styleCSD Δ with the RETRIEVED (not oracle) LoRA still beats 0.009 on painterly. |
| **2. Blend** | Rank-concat blend of top-K'; τ + scale tuned. | Blended top-K' ≥ single-nearest on the scorecard (smoother, no worse); no NaN/over-saturation; leak held. |
| **3. Library scale + web** | Grow library to N styles; wire web (ref → retrieve+blend → generate); stack band-control. | End-to-end web path produces visible painterly/semi-real style on the eval set; latency acceptable (retrieval instant; blend + LoRA load one-time per request). |
| **4. Polish** | Library curation quality, dedup, per-style scale calibration, UX (show matched style). | Scorecard across the full eval set beats band-control on painterly + semi_real with leak low, prompt held. |

## 6. Risks / open questions
- **Does a single per-style LoRA actually beat band-control on painterly?** UNVERIFIED on the scorecard
  (the "per-style LoRA works" claim predates the scorecard). Phase 0 answers this cheaply and is the
  real go/no-go — do it before building any library/retrieval.
- **Library coverage vs size.** Too few styles → poor nearest match for out-of-distribution refs; too
  many → training cost. Start K≈8–16 dense clusters spanning the CSD space; grow guided by retrieval
  misses. Each LoRA is a ~1–2h M1 train (rank 16) — the library is the main compute cost (parallel/
  overnight; caffeinate).
- **Blend validity across dissimilar LoRAs.** Rank-concat is mathematically exact for the SUM of deltas,
  but summing two very different style deltas may muddy. Mitigate with softmax(sim/τ) weighting (τ small
  → mostly the nearest) and cap K'≈2–3.
- **Per-style scale calibration.** Different LoRAs need different `--lora-scale`; store a per-LoRA scale
  from Phase-0/1 tuning in the index.
- **C multi-LoRA not needed for MVP** (offline blend → single merged file). Add native stacking to
  `iris_lora.c` only if repeated per-request blends become a latency/UX problem.

## 7. Immediate next action
**Phase 0 only** — one painterly LoRA, one scorecard run. Concretely: `curate_style_subset.py` (already
targets the densest cluster; point/verify it lands a PAINTERLY cluster, or add a seed filter) → train a
rank-16 LoRA (`train_lora.py`, full double+single coverage, on hot SSD, caffeinated, ~1–2h) → `export.py`
→ `iris -d flux-klein-4b-base --lora <exported> ...` → `train/.venv/bin/python debug/sref_scorecard.py
--label lora_painterly --model flux-klein-4b-base --extra "--lora <exported> --lora-scale 1.0"`. If
painterly styleCSD Δ > 0.009, the project is greenlit; if not, reconsider (the stack may simply not do
painterly at 4B). Everything else (library, retrieval, blend, web) is downstream of that one number.

---

## PROVE-OUT RESULT (2026-07-09) — journeydb tight clusters + BASE model = the answer

Detoured from painterly (corpus can't supply dense painterly) to prove the mechanism on corpus-rich
styles with zero new data. Mined 3 distinct dense hot-pool clusters (`cluster_hot_styles.py`):
**c23 cyberpunk (intra_cos 0.673)**, c9 fantasy portrait (0.594), c8 graphic poster (0.559) — all far
tighter than painterly (0.45–0.53), confirming the corpus HAS dense coherent styles. Trained a per-cluster
LoRA (rank 16, 400 steps) on c23; gated on prompt "a lone figure standing on a city street at night", seed 7:

- **Distilled (4-step):** styleCSD Δ **+0.041** but VISUALLY only a tint (baseline ≈ LoRA) — the step-1
  structure-lock (SREF-STYLE-CEILING) caps expression even for a TIGHT cluster.
- **Base (50-step CFG):** styleCSD Δ **+0.176 (4× distilled)** with a STRONG cohesive visible restyle
  (deep teal cinematic grade, atmosphere, backlit silhouette). **BASE UNLOCKS the style distilled could
  only tint.** Base is far more LoRA-sensitive: ×1.0 sweet spot, ×1.5 overcooks (Δ −0.31).

CONCLUSIONS: (1) **tight cluster is the lever** — the Phase-0 painterly failure was cluster LOOSENESS, not
the mechanism; (2) corpus + per-style LoRA deliver strong visible style transfer **on the base**; (3) product
trade = **base for strong style (slow, 50-step + CFG, ~12–25× a distilled render), distilled for fast/graphic**;
(4) per-LoRA scale must be calibrated & stored in the retrieval manifest (base ~1.0). Both models are
`guidance_embeds=false` (true CFG), so `train_lora` `guidance=None` is the correct regime for base — no code
change. Tools: `cluster_hot_styles.py`, `style_retrieve.py` (rank-concat blend). Base-unlock montage artifact
exists. NEXT: train c9+c8 on base → full retrieval demo (query → CSD-nearest LoRA → apply on base → discriminate).

## END-TO-END DEMO (2026-07-09) — the full retrieval loop works

Trained 3 base LoRAs (cyberpunk/portrait/graphic hot-pool clusters), built `library.json` (centroids +
LoRA paths + scale 1.0), extracted HELD-OUT query images (ranked outside each training subset), and ran
the resolver end-to-end (`style_retrieve.py`):
- **Retrieval discrimination: 3-for-3** on held-out queries (diagonal-dominant CSD matrix: cyberpunk 0.82,
  portrait 0.79, graphic 0.73 to their own centroids, clear margins over the others).
- **Apply on base (50-step), fixed prompt "a portrait of a warrior", seed 7**: each retrieved LoRA produces
  a visibly distinct restyle vs the photoreal baseline — cyberpunk = strong painterly concept-art (best),
  fantasy = polished ornate character, graphic = monochrome painterly. All 4 outputs distinct (md5).
- CAVEAT: at a single global scale 1.0 the styles read as VARIATIONS (cyberpunk transfers most; fantasy/
  graphic subtler; graphic came out monochrome-painterly, not its bold flat-poster character). PROMPT
  CONGRUENCE matters (first demo used a night-scene prompt that only suited cyberpunk — re-ran neutral).
CONCLUSION: **the retrieval-hybrid product loop is proven** — arbitrary reference → CSD-nearest LoRA →
visible style on base, no per-request training. Remaining = per-LoRA scale calibration + broader/cleaner
library (WikiArt for painterly). Montage artifact exists. Tools: cluster_hot_styles.py, style_retrieve.py,
library.json. NEXT (product): wire into web (ref upload → CSD → retrieve+blend → base render), or broaden
the library, or add the blend-of-2 demo.

## PRODUCT ITEMS 1/2/3 (2026-07-09)

**Item 1 — web wiring (DONE, committed 79c04cc).** "Style Library" reference mode → server CSD-matches the
ref to the nearest trained per-style LoRA and applies it (retrieval-hybrid), no per-request training.
`resolve_style_lora()` (cached) + `/generate` library path (`job.style_lora`, abs path under STYLE_LIB_DIR)
+ `/sref/resolve` (preview) + `/style-library` (list) + UI mode option. Base-trained LoRAs
(STYLE_LIBRARY_MODEL=flux-klein-4b-base) → run the web on base for strong style; a warning fires otherwise.
Live-validated: /style-library lists 3 styles; /sref/resolve discriminates (cyberpunk 0.82, portrait 0.79).

**Item 3 — scale calibration + blend (in progress).** Base is LoRA-sensitive (c23: 1.0 good, 1.5 overcooks).
Sweeping c9/c8 at 0.8/1.0/1.2 → pick best styleCSD Δ vs centroid, write per-LoRA scale into library.json.
Blend-of-2 demo pending (rank-concat interpolation). (NOTE: bash `declare -A` needs bash4; macOS is 3.2 —
use parallel-array specs in scripts.)

**Item 2 — broaden library (KEY FINDING; training heat-deferred).** Naive CSD clustering on the MIXED
corpus surfaces PHOTOGRAPHIC CONTENT-GENRES as tightly as rendering styles: the "new distinct" clusters
were food photography (0.63), surreal-sculpture photography (0.62), interior/real-estate photography (0.51)
— from the coyo/laion stock sources, NOT useful transferable styles. The useful RENDERING styles concentrate
in the journeydb/Midjourney region, and the shipped 3 (cyberpunk/fantasy/graphic) already capture the
dominant ones (other tight clusters just overlap them). **Fix: journeydb-restricted finer clustering**
(shards <800, k=50) → real new styles: **line-art/coloring-book (c42, intra 0.666) is a clear add**; some
candidates are still content-ish (hyperreal 3D renders) → needs a VISUAL curation pass, not blind training.
So the library IS broadenable with real styles, via style-aware (journeydb-restricted + eyeballed) selection.
NEXT (cooler weather): curate the journeydb-restricted candidates (line-art + a few more) → train base LoRAs
→ add to library.json. WikiArt remains the lever for the painterly styles the corpus lacks.
