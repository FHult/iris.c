# iris SREF Style Library v1 — for Flux.2 Klein 4B (base)

A small **retrieval** style pack for [Flux.2 Klein 4B base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B):
three trained per-style LoRAs plus a CSD-centroid manifest. Give it a reference image; it picks the
CSD-nearest trained style and applies that LoRA on the base (your prompt drives the subject).

**Companion to the [Learned Style adapter](https://github.com/FHult/iris.c/releases/tag/sref-adapter-v1).**
Use *this* when your reference matches one of the trained styles (strongest, most consistent for those);
use the generic adapter for *any* reference. See [docs/sref.md](https://github.com/FHult/iris.c/blob/main/docs/sref.md).

## Contents

| File | Size | Style | Default scale |
|---|---|---|---|
| `cyberpunk_c23_base.safetensors` | 74.7 MB | cyberpunk / neon graphic | 1.0 |
| `fantasy_portrait_c9_base.safetensors` | 74.7 MB | fantasy portrait | 1.2 |
| `graphic_c8_base.safetensors` | 74.7 MB | flat graphic / illustration | 1.2 |
| `library.json` | — | CSD centroids + LoRA filenames + per-style scales (the retrieval index) | — |

Each LoRA is trained on the frozen 4B **base**; `library.json` holds each style's 768-d CSD centroid so a
reference can be matched by cosine. Scales are calibrated per style (the optimum differs).

## How it works

`CSD(reference)` → nearest centroid in `library.json` → apply that style's LoRA at its calibrated scale on
the base. When two styles are close, the resolver **rank-concatenates** the top-K into one merged LoRA
(exact weighted interpolation) that `iris --lora` loads unchanged.

## Usage

```bash
# reference image -> the iris command for the nearest style (needs CSD-ViT-L; see below)
python train/lora/style_retrieve.py --query ref.png --library library.json --top-k 1
# it prints, e.g.:
#   ./iris -d flux-klein-4b-base --lora cyberpunk_c23_base.safetensors --lora-scale 1.0 -p "..." -o out.png
```

`library.json` uses **relative filenames**, resolved against its own directory — unpack all files into one
folder and it works wherever you put it. Or point the iris web app at the folder (`IRIS_STYLE_LIB=<dir>`)
and use its **"Style Library"** reference mode, which does the CSD step for you.

## Dependencies (permissive; obtain from source)

1. **Flux.2 Klein 4B base** — Apache 2.0, ungated, commercial-OK. `black-forest-labs/FLUX.2-klein-base-4B`.
   Style transfer needs the **base** (the distilled model only tints).
2. **CSD-ViT-L** — CC-BY-4.0 (commercial-OK, attribution). `tomg-group-umd/CSD-ViT-L`. Cite Somepalli et
   al., *Measuring Style Similarity in Diffusion Models*, 2024. Only needed to match a reference; not to
   apply a style you already picked.

## Limitations (honest)

- **Three styles only.** Retrieval is nearest-of-three — a reference far from all three still matches the
  closest, with low confidence (check the printed cosine). For arbitrary references use the generic adapter.
- **Base model only** (distilled just tints); scales are per-style-calibrated (×1.5 can overcook).
- No painterly style in this pack; base 50-step renders are slower than distilled.

## License & attribution

- **These LoRAs + manifest:** MIT (same as the iris.c engine).
- Requires **Flux.2 Klein 4B base** (Apache 2.0) and, for matching, **CSD-ViT-L** (CC-BY-4.0) — obtain from
  their sources and follow their terms (CC-BY-4.0 requires attributing Somepalli et al.).

Built by `train/lora/cluster_hot_styles.py` (CSD k-means over a hot style corpus) → per-cluster LoRAs.
