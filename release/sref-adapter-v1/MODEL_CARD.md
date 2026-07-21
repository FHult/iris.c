# iris SREF Learned Style Adapter v1 — for Flux.2 Klein 4B (base)

A generic **style-reference** adapter for [Flux.2 Klein 4B base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B):
give it a reference image and generations adopt that image's **style** while your prompt drives the
**subject** (Midjourney `--sref`). Unlike a per-style LoRA, one adapter transfers *any* reference's
look — graphic or painterly.

Runs in the pure-C/Metal [iris](https://github.com/FHult/iris.c) engine (the `--sref-csdmod` path) or
its web app ("Learned Style" mode).

## What it is

Two small weight files that modulate the **frozen** Flux.2 Klein 4B base:

| File | Size | What |
|---|---|---|
| `joint_lora.safetensors` | 298.9 MB | LoRA (rank 64) on the DiT's double+single attention blocks. |
| `csd_mod.safetensors` | 15.7 MB | `CSDModulation`: a 2-layer FiLM (CSD 768 → 1024 → hidden 3072) that adds a style delta onto the timestep-modulation channel (`temb`). |

**How it works.** A CSD style descriptor of the reference image (768-d, L2-normalised) is mapped by
`csd_mod` to a delta on `temb` — the channel every block's adaLN and the final norm read, at every
noise level, so the model cannot ignore it. The LoRA restyles; **style-CFG** (inject the CSD delta only
in the *conditional* CFG forward) makes classifier-free guidance amplify the style instead of cancelling
it. A single scale `α` (default **0.4**) trades style strength against prompt adherence.

Trained on the frozen base with a **content-shared-pair InfoNCE contrastive** (same noised latent, correct
reference vs a foreign reference) in the noise band `t ∈ [700, 950]`, so the adapter is rewarded for
reference-*discrimination*, not a constant style — the failure mode that sank earlier attempts.

## Dependencies (both permissive; you download them from source)

This adapter is only weights for the frozen base — it needs two external models at inference, which you
obtain from their official sources (not redistributed here):

1. **Flux.2 Klein 4B base** — the frozen backbone. License **Apache 2.0**, ungated, commercial-OK.
   `black-forest-labs/FLUX.2-klein-base-4B`. Style transfer needs the **base** (not the distilled 4B):
   the distilled model commits structure at step 1, so it only *tints*.
2. **CSD-ViT-L** — computes the reference's style descriptor. License **CC-BY-4.0** (commercial-OK with
   attribution). `tomg-group-umd/CSD-ViT-L`. Cite: Somepalli et al., *Measuring Style Similarity in
   Diffusion Models*, 2024.

## Usage

```bash
# 1. Reference image -> CSD style vector (needs the CSD-ViT-L weights)
python train/lora/dump_csd.py --image ref.png --out ref.csd.f32 --weights <csd_vit_l_style.safetensors>

# 2. Generate on the 4B BASE with the adapter
./iris -d flux-klein-4b-base -p "a robot standing in a desert" \
  --lora joint_lora.safetensors \
  --sref-csdmod csd_mod.safetensors --sref-csd ref.csd.f32 --sref-scale 0.4 \
  -o out.png
```

`--sref-scale` (α) is the style↔content slider: higher = more style, less prompt adherence. Or use the
iris web app's **"Learned Style"** reference mode, which does the CSD step for you. Full guide:
[docs/sref.md](https://github.com/FHult/iris.c/blob/main/docs/sref.md).

## Evaluation & limitations (honest)

Scored with `debug/sref_scorecard.py` on a 32-reference held-out set (styleCSD Δ = style transferred,
promptAdh = subject kept, cross-ref corr < 0.90 = it actually discriminates references, not collapsed).

- **Base model only** — on the distilled model it only tints.
- **Style vs content is a slider** (`α`); at high `α` the subject can be over-styled. 0.4 is a balanced default.
- **Strongest on graphic/bold references** (line-art, comics, woodcut, flat illustration); painterly and
  photographic references transfer but are the hardest.
- **The generic adapter is the answer** — per-reference-type *specialists* were tested three times and none
  beat this generic adapter at any data mix that preserves prompt adherence (that research is closed).
- Not a content/composition copier — for that, use img2img/composition mode instead.

## License & attribution

- **This adapter:** MIT (same as the iris.c engine).
- Requires **Flux.2 Klein 4B base** (Apache 2.0) and **CSD-ViT-L** (CC-BY-4.0) — obtain from their
  sources and follow their terms (CC-BY-4.0 requires attributing Somepalli et al.).

Provenance: trained checkpoint `sref_joint_probe/joint_probe_0007000`, exported to C by
`train/lora/export_joint_to_c.py`. Parity-guarded in `make test-unit` (CSDModulation) — see the iris.c repo.
