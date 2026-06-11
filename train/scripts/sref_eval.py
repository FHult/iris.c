#!/usr/bin/env python
"""
sref_eval.py — style-reference quality evaluation (SREF-2 core).

CLIP-I conflates style and content, so it cannot judge an --sref model. This harness
scores generation/reference pairs with the metrics that define sref quality, using
BOTH heads of the CSD encoder (shared tower, two projections):

  style_sim     cos(style(gen), style(ref))      — want HIGH (style adopted)
  content_leak  cos(content(gen), content(ref))  — want LOW  (subject NOT copied)
  sref_score    style_sim − content_leak          — the single headline number

Prompt adherence (CLIP-T) is a planned third axis — stubbed until a text tower is
wired; style_sim/content_leak already separate "transfers style" from "copies image",
which is the failure mode CLIP-I cannot see.

Input layouts (auto-detected per pair):
  --pairs FILE     json list of {"ref": path, "gen": path, ["prompt": str]}
  --ref-dir/--gen-dir   matched by filename stem

Output: per-pair scores + aggregates (+ optional baseline comparison) to stdout and
--out JSON (consumed later by quality_gate / shippable-champion checks).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from style_encoder.csd_mlx import CSDStyleEncoder, preprocess


def load_pairs(args) -> list[dict]:
    if args.pairs:
        return json.loads(Path(args.pairs).read_text())
    refs = {p.stem: p for p in Path(args.ref_dir).iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    pairs = []
    for g in sorted(Path(args.gen_dir).iterdir()):
        if g.suffix.lower() in (".jpg", ".jpeg", ".png") and g.stem in refs:
            pairs.append({"ref": str(refs[g.stem]), "gen": str(g)})
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--ref-dir", default=None)
    ap.add_argument("--gen-dir", default=None)
    ap.add_argument("--weights", default="/Volumes/2TBSSD/models/csd_vit_l_style.safetensors")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.pairs and not (args.ref_dir and args.gen_dir):
        ap.error("need --pairs or --ref-dir + --gen-dir")

    pairs = load_pairs(args)
    if not pairs:
        print("no pairs found", file=sys.stderr)
        return 1

    enc = CSDStyleEncoder(args.weights)
    rows = []
    for p in pairs:
        try:
            imgs = np.stack([preprocess(Image.open(p["ref"])),
                             preprocess(Image.open(p["gen"]))])
        except Exception as exc:
            rows.append({**p, "error": str(exc)})
            continue
        style, content = enc.encode_both(imgs)
        s = float(style[0] @ style[1])
        c = float(content[0] @ content[1])
        rows.append({**p, "style_sim": round(s, 4), "content_leak": round(c, 4),
                     "sref_score": round(s - c, 4)})

    ok = [r for r in rows if "error" not in r]
    agg = {}
    if ok:
        for k in ("style_sim", "content_leak", "sref_score"):
            v = np.array([r[k] for r in ok])
            agg[k] = {"mean": round(float(v.mean()), 4),
                      "p10": round(float(np.percentile(v, 10)), 4),
                      "p90": round(float(np.percentile(v, 90)), 4)}
    report = {"n_pairs": len(ok), "n_errors": len(rows) - len(ok),
              "aggregate": agg, "pairs": rows}

    print(json.dumps({"n_pairs": report["n_pairs"], "aggregate": agg}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
