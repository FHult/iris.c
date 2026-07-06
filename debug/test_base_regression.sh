#!/bin/bash
# TB-002 — Base-model (Flux Klein 4B-base) regression test. On-demand (requires the base model
# weights; NOT part of the hermetic `make test`). Exercises the base-specific path — CFG double-pass,
# guidance, negative prompt — and guards determinism + basic image sanity.
#
#   debug/test_base_regression.sh                 # uses ./flux-klein-4b-base, 256px, 16 steps
#   MODEL=flux-klein-4b-base SIZE=256 STEPS=16 debug/test_base_regression.sh
#
# Run with the GPU free (stop the web server / other model jobs first — two 4B models may not
# co-reside in 32 GB). Exit 0 = pass, 1 = fail, 77 = skipped (model absent).
set -u
cd "$(dirname "$0")/.."
MODEL="${MODEL:-flux-klein-4b-base}"
SIZE="${SIZE:-256}"
STEPS="${STEPS:-16}"
SEED=42
PROMPT="a red apple on a wooden table"
OUT=$(mktemp -d /tmp/tb002.XXXXXX)
IRIS=./iris
PY=web/venv/bin/python

[ -x "$IRIS" ] || { echo "SKIP: ./iris not built"; exit 77; }
[ -d "$MODEL" ] || { echo "SKIP: base model '$MODEL' not present (download required)"; exit 77; }

run() { # run <outfile> <extra-args...>
  local out="$1"; shift
  caffeinate -dimsu "$IRIS" -d "$MODEL" -p "$PROMPT" -S "$SEED" --steps "$STEPS" \
    -W "$SIZE" -H "$SIZE" "$@" -o "$out" >"$OUT/log" 2>&1 || {
      echo "FAIL: iris run failed ($*)"; tail -5 "$OUT/log"; exit 1; }
  [ -f "$out" ] || { echo "FAIL: no output for run ($*)"; exit 1; }
}

echo "TB-002 base regression: model=$MODEL size=${SIZE} steps=${STEPS}"
echo "  [1/4] baseline (default CFG guidance)...";       run "$OUT/a.png"
echo "  [2/4] repeat (determinism)...";                  run "$OUT/a2.png"
echo "  [3/4] low guidance (--guidance 1.0)...";         run "$OUT/b.png" --guidance 1.0
echo "  [4/4] negative prompt...";                       run "$OUT/c.png" -N "blurry, deformed, low quality"

"$PY" - "$OUT" "$SIZE" <<'PY'
import sys, numpy as np
from PIL import Image
d, size = sys.argv[1], int(sys.argv[2])
def img(n): return np.asarray(Image.open(f"{d}/{n}").convert("RGB"), dtype=np.float64)
a, a2, b, c = img("a.png"), img("a2.png"), img("b.png"), img("c.png")
def corr(x, y): return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
fails = []
# sanity: right size, finite, non-degenerate (not a flat color)
for n, im in [("a", a), ("a2", a2), ("b", b), ("c", c)]:
    if im.shape != (size, size, 3): fails.append(f"{n}: wrong shape {im.shape}")
    if not np.isfinite(im).all(): fails.append(f"{n}: non-finite pixels")
    if float(im.std()) < 5.0:     fails.append(f"{n}: degenerate (std {im.std():.1f} — near-flat)")
# determinism
c_aa = corr(a, a2)
if c_aa < 0.999: fails.append(f"determinism: corr(a,a2)={c_aa:.4f} < 0.999")
# CFG guidance must change the image
c_ab = corr(a, b)
if c_ab > 0.995: fails.append(f"CFG: corr(a, low-guidance)={c_ab:.4f} > 0.995 (guidance had no effect)")
# negative prompt must change the image
c_ac = corr(a, c)
if c_ac > 0.995: fails.append(f"negative: corr(a, neg)={c_ac:.4f} > 0.995 (negative had no effect)")
print(f"  determinism corr(a,a2)={c_aa:.4f} | CFG corr(a,b)={c_ab:.4f} | negative corr(a,c)={c_ac:.4f}")
if fails:
    print("RESULT: FAIL"); [print("   -", f) for f in fails]; sys.exit(1)
print("RESULT: PASS — base model deterministic; CFG guidance and negative prompt both affect output")
PY
rc=$?
rm -rf "$OUT"
exit $rc
