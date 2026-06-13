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
import os
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
    ap.add_argument("--report", default=None,
                    help="write an HTML contact sheet (ref + gens + scores) — the "
                         "readable visual verdict for a champion sweep")
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
    if args.report:
        _write_html_report(Path(args.report), rows, agg)
        print(f"wrote {args.report}")
    return 0


def _write_html_report(path: Path, rows: list[dict], agg: dict) -> None:
    """Contact sheet grouped by reference: each row is ref thumbnail + the gens
    (ascending ip-scale when present) with per-image scores. Image paths are
    relative to the report so it's portable next to the gen/ dir.

    Colour cues encode the sref reading: style_sim green-high, content_leak
    red-high (want LOW), prompt_adherence green-high."""
    import html as _html
    base = path.parent

    def _rel(p):
        try:
            return os.path.relpath(p, base)
        except Exception:
            return p

    def _chip(label, val, good_high=True):
        if val is None:
            return f'<span class="chip na">{label} —</span>'
        # 0..1 cosine → hue: good=green(120), bad=red(0)
        g = val if good_high else (1.0 - val)
        hue = max(0.0, min(120.0, g * 120.0))
        return (f'<span class="chip" style="background:hsl({hue:.0f},65%,88%)">'
                f'{label} {val:+.3f}</span>')

    from collections import OrderedDict
    by_ref: "OrderedDict[str, list]" = OrderedDict()
    for r in rows:
        by_ref.setdefault(r.get("ref", "?"), []).append(r)

    cards = []
    for ref, items in by_ref.items():
        items = sorted(items, key=lambda r: r.get("scale", 0))
        gens = []
        for r in items:
            if "error" in r:
                gens.append(f'<div class="gen err">{_html.escape(r["error"][:120])}</div>')
                continue
            sc = r.get("scale")
            scl = f'scale {sc}' if sc is not None else ''
            gens.append(
                f'<div class="gen"><img src="{_rel(r["gen"])}" loading="lazy">'
                f'<div class="cap">{scl}'
                f'<div class="prompt">{_html.escape((r.get("prompt") or "")[:48])}</div>'
                f'{_chip("style", r.get("style_sim"), True)}'
                f'{_chip("leak", r.get("content_leak"), False)}'
                f'{_chip("prompt", r.get("prompt_adherence"), True)}'
                f'</div></div>')
        cards.append(
            f'<section class="card"><div class="ref">'
            f'<img src="{_rel(ref)}" loading="lazy"><div>REFERENCE</div></div>'
            f'<div class="gens">{"".join(gens)}</div></section>')

    def _agg_line(k, label, good_high=True):
        a = agg.get(k)
        if not a:
            return ""
        return (f'<div class="agg">{label}: mean <b>{a["mean"]:+.3f}</b> '
                f'(p10 {a["p10"]:+.3f} / p90 {a["p90"]:+.3f}) '
                f'{"↑higher=better" if good_high else "↓lower=better"}</div>')

    summary = (
        _agg_line("style_sim", "style_sim", True) +
        _agg_line("content_leak", "content_leak", False) +
        _agg_line("sref_score", "sref_score (style − leak)", True) +
        _agg_line("prompt_adherence", "prompt_adherence", True))

    css = (
        "body{font:13px -apple-system,sans-serif;margin:18px;background:#fafafa;color:#222}"
        "h1{font-size:18px} .agg{margin:2px 0} .card{display:flex;gap:14px;align-items:flex-start;"
        "background:#fff;border:1px solid #e2e2e2;border-radius:8px;padding:12px;margin:12px 0}"
        ".ref{flex:0 0 180px;text-align:center;font-weight:600;color:#666}"
        ".ref img,.gen img{width:180px;height:180px;object-fit:cover;border-radius:6px}"
        ".gens{display:flex;flex-wrap:wrap;gap:12px} .gen{width:180px} .gen.err{color:#b00}"
        ".cap{font-size:11px;margin-top:3px} .prompt{color:#888;margin:2px 0}"
        ".chip{display:inline-block;padding:1px 5px;margin:1px 2px 1px 0;border-radius:4px;"
        "font-size:10px} .chip.na{background:#eee;color:#999}")
    body = (f"<h1>SREF evaluation — visual verdict</h1><div class='summary'>{summary}</div>"
            f"<p style='color:#888'>style high + leak low + prompt steady = clean style "
            f"transfer. {len(by_ref)} references.</p>{''.join(cards)}")
    path.write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
                    f"<style>{css}</style></head><body>{body}</body></html>")


if __name__ == "__main__":
    raise SystemExit(main())
