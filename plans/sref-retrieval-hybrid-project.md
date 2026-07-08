# SREF Retrieval-Hybrid Instant-LoRA — Project Scope

Status (2026-07-08): **PROPOSED.** The learned-encoder direction is KILLED (three decisive
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
