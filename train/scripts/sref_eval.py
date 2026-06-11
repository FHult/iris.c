#!/usr/bin/env python
"""
sref_eval.py — style-reference quality evaluation (SREF-2 core).

CLIP-I conflates style and content, so it cannot judge an --sref model. This harness
scores generation/reference pairs with the metrics that define sref quality, using
BOTH heads of the CSD encoder (shared tower, two projections):

  style_sim     cos(style(gen), style(ref))      — want HIGH (style adopted)
  content_leak  cos(content(gen), content(ref))  — want LOW  (subject NOT copied)
  sref_score    style_sim − content_leak          — the single headline number

Third axis (optional, --prompt-adherence): prompt_adherence = cos(SigLIP-text(prompt),
SigLIP-image(gen)) — did the generation still follow the PROMPT while adopting the
style? Uses the same google/siglip-so400m the training stack uses (text tower via
transformers/torch; no new weights). Only computed for pairs that carry a "prompt".

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
    # setdefault on a sorted listing: when a stem exists with multiple
    # extensions (x.jpg + x.png), the alphabetically first wins, deterministically.
    refs: dict = {}
    for p in sorted(Path(args.ref_dir).iterdir()):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            refs.setdefault(p.stem, p)
    pairs = []
    for g in sorted(Path(args.gen_dir).iterdir()):
        if g.suffix.lower() in (".jpg", ".jpeg", ".png") and g.stem in refs:
            pairs.append({"ref": str(refs[g.stem]), "gen": str(g)})
    return pairs


class _SiglipAdherence:
    """Lazy SigLIP text+image scorer (transformers/torch; mps when available)."""

    MODEL = "google/siglip-so400m-patch14-384"

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor
        self._torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(self.MODEL).eval().to(self.dev)
        self.proc = AutoProcessor.from_pretrained(self.MODEL)

    def score(self, prompt: str, gen_img: Image.Image) -> float:
        torch = self._torch
        with torch.no_grad():
            ti = self.proc(text=[prompt], padding="max_length", truncation=True,
                           return_tensors="pt").to(self.dev)
            ii = self.proc(images=[gen_img.convert("RGB")],
                           return_tensors="pt").to(self.dev)
            t = self.model.get_text_features(**ti)
            v = self.model.get_image_features(**ii)
            # transformers 5.x returns output objects; 4.x returned tensors.
            if hasattr(t, "pooler_output"):
                t = t.pooler_output
            if hasattr(v, "pooler_output"):
                v = v.pooler_output
            t = t / t.norm(dim=-1, keepdim=True)
            v = v / v.norm(dim=-1, keepdim=True)
            return float((t * v).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--ref-dir", default=None)
    ap.add_argument("--gen-dir", default=None)
    ap.add_argument("--weights", default="/Volumes/2TBSSD/models/csd_vit_l_style.safetensors")
    ap.add_argument("--prompt-adherence", action="store_true",
                    help="add cos(SigLIP-text(prompt), SigLIP-image(gen)) for "
                         "pairs that carry a 'prompt' field")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.pairs and not (args.ref_dir and args.gen_dir):
        ap.error("need --pairs or --ref-dir + --gen-dir")

    pairs = load_pairs(args)
    if not pairs:
        print("no pairs found", file=sys.stderr)
        return 1

    enc = CSDStyleEncoder(args.weights)
    adher = None
    if args.prompt_adherence and any(p.get("prompt") for p in pairs):
        adher = _SiglipAdherence()
    rows = []
    for p in pairs:
        try:
            gen_img = Image.open(p["gen"])
            imgs = np.stack([preprocess(Image.open(p["ref"])),
                             preprocess(gen_img)])
        except Exception as exc:
            rows.append({**p, "error": str(exc)})
            continue
        style, content = enc.encode_both(imgs)
        s = float(style[0] @ style[1])
        c = float(content[0] @ content[1])
        row = {**p, "style_sim": round(s, 4), "content_leak": round(c, 4),
               "sref_score": round(s - c, 4)}
        if adher is not None and p.get("prompt"):
            try:
                row["prompt_adherence"] = round(adher.score(p["prompt"], gen_img), 4)
            except Exception as exc:
                row["prompt_adherence_error"] = str(exc)
        rows.append(row)

    ok = [r for r in rows if "error" not in r]
    agg = {}
    if ok:
        for k in ("style_sim", "content_leak", "sref_score", "prompt_adherence"):
            vals = [r[k] for r in ok if k in r]
            if not vals:
                continue
            v = np.array(vals)
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
