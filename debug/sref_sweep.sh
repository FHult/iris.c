#!/bin/bash
# debug/sref_sweep.sh — push-button visual sref evaluation sweep (SREF-2).
#
# Generates a prompts x ip-scales grid against each style reference with a
# trained adapter bundle, then scores it with the full sref_eval triad
# (style_sim / content_leak / prompt_adherence). The first OUTPUT-QUALITY
# evidence for a champion — held-out cond_gap says the adapter conditions;
# this says whether you can SEE it.
#
# GPU-exclusive: PAUSE THE FLYWHEEL FIRST —
#   train/.venv/bin/python train/scripts/pipeline_ctl.py pause --free-gpu
# and resume after:
#   train/.venv/bin/python train/scripts/pipeline_ctl.py resume
#
# Usage: ./debug/sref_sweep.sh [BUNDLE_DIR]   (default: iter0002 export)
# ~18 generations x ~35 s ≈ 12 min + eval. Output: /Volumes/2TBSSD/sref_sweep/

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SWEEP=/Volumes/2TBSSD/sref_sweep
BUNDLE="${1:-$SWEEP/bundle_iter0002}"
PY="$ROOT/train/.venv/bin/python"
GEN="$SWEEP/gen"
mkdir -p "$GEN"

PROMPTS=(
  "a cat sitting on a windowsill"
  "a sailboat on a calm sea at dawn"
  "a bowl of fruit on a wooden table"
)
SCALES=(0.4 0.8 1.2)

PAIRS="$SWEEP/pairs.json"
echo "[" > "$PAIRS.tmp"
first=1

for ref in "$SWEEP"/refs/*.jpg; do
  stem=$(basename "$ref" .jpg)
  feats="$SWEEP/refs/$stem.bin"
  if [ ! -f "$feats" ]; then
    echo "== SigLIP features: $stem"
    "$PY" "$ROOT/train/scripts/siglip_features.py" "$ref" --out "$feats"
  fi
  for pi in "${!PROMPTS[@]}"; do
    for s in "${SCALES[@]}"; do
      out="$GEN/${stem}_p${pi}_s${s}.png"
      if [ ! -f "$out" ]; then
        echo "== gen: ref=$stem prompt=$pi scale=$s"
        "$ROOT/iris" -d "$ROOT/flux-klein-model" -p "${PROMPTS[$pi]}" \
          --ip "$BUNDLE" --ip-features "$feats" --ip-scale "$s" \
          --seed 42 --steps 4 -W 512 -H 512 -o "$out"
      fi
      [ $first -eq 0 ] && echo "," >> "$PAIRS.tmp"
      first=0
      printf '{"ref": "%s", "gen": "%s", "prompt": "%s", "scale": %s}' \
        "$ref" "$out" "${PROMPTS[$pi]}" "$s" >> "$PAIRS.tmp"
    done
  done
done
echo "]" >> "$PAIRS.tmp"
mv "$PAIRS.tmp" "$PAIRS"

echo "== scoring with sref_eval (style_sim / content_leak / prompt_adherence)"
"$PY" "$ROOT/train/scripts/sref_eval.py" --pairs "$PAIRS" \
  --prompt-adherence --out "$SWEEP/report.json"
echo
echo "Report: $SWEEP/report.json   Images: $GEN/"
echo "Read:   per-scale style_sim should RISE with scale; content_leak should stay low;"
echo "        prompt_adherence should stay roughly flat (style transfers, prompt holds)."
