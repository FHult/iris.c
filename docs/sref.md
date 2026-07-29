# Style Reference (`--sref`) — the canonical guide

Transfer the *look* of a reference image onto a generated one (Midjourney `--sref`). This is the
single source of truth for how style reference works in iris; the `plans/sref-*` docs are the
research trail behind it (indexed at the bottom).

## TL;DR — which mode, when

| Your reference… | Use | Why |
|---|---|---|
| **anything** (general style transfer) | **Learned Style adapter** (default) | Works for any reference; the shipped generic answer (v5.3.0). |
| **bold / graphic / high-contrast** and you want it fast & free | **Band-control** | Training-free, zero extra cost. Strong on graphic, **weak on painterly**. |
| matches a **known trained style** | **Style Library** | CSD-retrieves the nearest trained per-style LoRA — strongest for those styles. |
| you want the **literal** reference look | **In-context** | Reference fed as image tokens; the original clean path. |

**All style transfer needs the BASE model** (`flux-klein-4b-base`), not the distilled model — the
distilled model commits structure at step 1, so it only *tints*; the base restyles. Run the web with
`--model-dir flux-klein-4b-base` (or set `STYLE_LIBRARY_MODEL`).

## The four modes

### 1. Learned Style adapter (default, v5.3.0)
The generic learned style adapter: `CSD(reference)` → a FiLM delta on the DiT's timestep-modulation
channel (`iris_csdmod.c`) + a LoRA on the base, amplified by style-CFG. Works for *any* reference.

```
./iris -d flux-klein-4b-base -p "a robot in a desert" \
  --lora <joint_lora.safetensors> \
  --sref-csdmod <csd_mod.safetensors> --sref-csd <ref.csd.f32> --sref-scale 0.4
```

- `--sref-scale` (α) is the style↔content slider; default **0.4** (style-CFG op point). Higher = more
  style, less prompt adherence.
- CSD is computed Python-side (`train/lora/dump_csd.py`); there is no C ViT-L. The web does this for you.
- Web: upload a reference → **"Learned Style"** mode. Needs the adapter export installed
  (`joint_lora.safetensors` + `csd_mod.safetensors`).

### 2. Band-control (training-free, v5.0.0)
Attenuate the reference's high-frequency RoPE bands (kills composition copying) and boost its
low-frequency bands (boosts style), on the in-context path. No adapter, no per-step cost.

```
./iris -d flux-klein-4b-base -p "..." -i ref.png --img2img-strength 1.0 \
  --sref-shf 0.0 --sref-slf 1.5          # 1.0/1.0 = OFF (bit-identical to plain in-context)
```

- `--sref-shf` (0–1, high-freq attenuation), `--sref-slf` (≥1, low-freq boost), `--sref-strength`
  (γ attention bias on reference-key columns, GPU-only).
- **Ceiling (SREF-STYLE-CEILING):** strong for graphic/high-contrast refs (woodcut → styleCSD Δ 0.096),
  but only ~0.009 on painterly. It is a mechanism ceiling, not a tuning problem — for painterly, use the
  learned adapter.
- Web: the default rail for a **Style**-mode reference; saved **style codes** replay on it.

### 3. Style Library (retrieval, v5.2/5.3)
Match `CSD(reference)` to the nearest trained per-style LoRA in a library and apply it on the base
(optionally rank-concat-blending the top-K). Strongest when the reference matches a style the library
was trained on.

- Web: **"Style Library"** mode; `/sref/resolve` previews the match, `/style-library` lists them.
- Library lives under `IRIS_STYLE_LIB` (default `/Volumes/2TBSSD/sref_eval/lora_lib`, `library.json`).
- Tools: `train/lora/{cluster_hot_styles,style_retrieve}.py`.

### 4. In-context (fallback)
Feed the reference as image tokens at `--img2img-strength 1.0` so the frozen base attends to it
natively (prompt drives content). Patch-shuffle the reference first to destroy composition while
keeping texture/style. This is what band-control refines.

## Conventions & caveats (don't relearn these)

- **BASE, not distilled** — see above; distilled only tints.
- **The collapse gate is mandatory.** Any adapter A/B must report cross-ref output corr **< 0.90**
  (`debug/sref_ref_discrimination.py` / the scorecard). A high style score with corr ≥ 0.9 is a
  mode-collapse false positive (the reference-inert failure that killed multiple attempts).
- **Eval** = `debug/sref_scorecard.py --score-only` over the 32-ref set (`debug/sref_eval_set.json`):
  styleCSD Δ (style transferred), promptAdh (subject kept), leak Δ (composition copied).
- **Guards** (so this can't silently rot): `make test-unit` runs the csd_mod parity guard; `web/tests`
  covers the Style Library / Learned Style routing; both run in the nightly health check.

## Not offered: per-type specialists / router — CLOSED

Routing a reference to a per-type specialist (painterly/graphic/…) was tested three times and is
**closed**: no specialist beats the generic at any data mix that preserves prompt-following (the last
attempt, a 40:60 WikiArt+diverse painterly specialist, diluted the style below generic levels). The
**generic Learned Style adapter is the robust answer.** Trail: `plans/archive/sref-mixed-painterly-specialist.md`,
BACKLOG `SREF-STYLE-ROUTER`.

## Research trail (`plans/archive/sref-*`) — by status

The SREF workstream is shipped and closed; its plan docs now live under `plans/archive/`.

**Shipped / current:**
- `sref-rope-band-control.md` — band-control (v5.0), shipped.
- `sref-retrieval-hybrid-project.md` — Style Library, shipped.
- `sref-joint-backbone-project.md` — the learned adapter (v5.3), shipped.

**Closed / superseded (kept as the trail):**
- `sref-mixed-painterly-specialist.md` — specialist/router, CLOSED (3rd negative).
- `sref-learned-encoder-project.md`, `sref-phase1-projector.md`, `sref-phase1-data-scoping.md` —
  learned-encoder direction, killed (noised-target root cause).
- `sref-architecture-retrain.md`, `sref-retrain-diagnostic.md`, `sref-architecture-options.md`,
  `sref-m1-feasibility-brief.md` — retrain/architecture exploration.
- `sref-execution-plan.md`, `sref-journey-retrospective.md` — earlier campaign planning/history.
