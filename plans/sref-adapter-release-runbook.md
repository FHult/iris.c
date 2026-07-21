# SREF Learned Style Adapter v1 — open-weight release runbook

**Status: RELEASED on GitHub (2026-07-21).** https://github.com/FHult/iris.c/releases/tag/sref-adapter-v1
Scope shipped = the generic Learned Style adapter alone (`joint_lora` + `csd_mod` + model card +
manifest); `--latest=false` so it doesn't supersede the engine version release. HF upload intentionally
skipped (owner preference). Style Library bundle NOT released (optional follow-up). The BACKLOG
platform-vision endgame — open-weight release of the generic adapter (v5.3.0) — is DONE.

## Prerequisites — DONE

- [x] **License clearance (the gating item).** Verified against the HF model cards:
  - Flux.2 Klein 4B **base** = **Apache 2.0**, ungated, commercial-OK. The adapter trains on/requires
    this (NOT the non-commercial 9B). ⚠ the repo README license table says "MIT" for the 4B — HF says
    Apache 2.0; both permissive, but fix the README table when convenient.
  - CSD-ViT-L (the reference encoder) = **CC-BY-4.0**, commercial-OK with attribution.
  - Our adapter is ours → release **MIT** (matches the engine). No non-commercial blocker anywhere.
- [x] **Model card** — `release/sref-adapter-v1/MODEL_CARD.md` (what it is, deps + their licenses,
  usage, honest limitations, attribution). Ships as the release README.
- [x] **Bundle manifest + checksums** — `release/sref-adapter-v1/MANIFEST.txt` (sha256 pinned).
- [x] **Guards** — CSDModulation C parity in `make test-unit`; web routing covered; both in the nightly.

## Release artifacts (on 2TBSSD, not in git — ~314 MB)

`/Volumes/2TBSSD/sref_eval/joint_v1_c_export/{joint_lora.safetensors, csd_mod.safetensors}`
+ `release/sref-adapter-v1/{MODEL_CARD.md, MANIFEST.txt}` from git.

## Open decisions before the go

1. **Where.** GitHub Release on `FHult/iris.c` (attach the two files) and/or a Hugging Face model repo
   (better discoverability + the model-card renders natively). Recommend **both**: HF repo as the home,
   GitHub release for engine users.
2. **Scope.** Ship the generic adapter alone (recommended — it's the "answer"), or also the Style Library
   bundle (`/Volumes/2TBSSD/sref_eval/lora_lib/`, 3 base LoRAs + `library.json`) as a separate optional
   download. Keep them separate releases.
3. **Version tag.** Suggest `sref-adapter-v1` (independent of the engine's vX.Y.Z tags).

## One-step go (GitHub release), when approved

```bash
# from repo root; the two weight files staged locally as $BUNDLE
BUNDLE=/Volumes/2TBSSD/sref_eval/joint_v1_c_export
( cd "$BUNDLE" && shasum -a 256 -c "$OLDPWD/release/sref-adapter-v1/MANIFEST.txt" )   # verify first
gh release create sref-adapter-v1 \
  "$BUNDLE/joint_lora.safetensors" "$BUNDLE/csd_mod.safetensors" \
  release/sref-adapter-v1/MODEL_CARD.md release/sref-adapter-v1/MANIFEST.txt \
  --repo FHult/iris.c \
  --title "SREF Learned Style Adapter v1 (Flux.2 Klein 4B base)" \
  --notes-file release/sref-adapter-v1/MODEL_CARD.md
```

For a Hugging Face repo: `huggingface-cli repo create`, upload the two weights + `MODEL_CARD.md` as
`README.md`, set license `mit`. (Needs an HF token; do interactively.)

## Do NOT

- Do not publish until the owner says go (outward-facing, irreversible).
- Do not redistribute the Flux base or CSD-ViT-L weights — link to their sources (keeps their licenses
  the user's responsibility and our bundle clean).
