#!/usr/bin/env python
"""
sref_regrade.py — re-judge saved SREF sweep results under the CONTENT GATE (no GPU, no re-gen).

The injection ratio (Δstyle/Δleak) is gameable by content-washing: a gen that collapses into a
ref-matching texture scores high style_sim + low content → high ratio while depicting nothing.
prompt_adherence (CLIP gen↔prompt) is the guard — it sits at the no-adapter null (~0.15) when
content is preserved and collapses toward 0 when washed. content_retain = prompt_adherence /
null_prompt; a scale is a REAL win only if it retains content. This reads each arm's saved
scores.json and prints the honest "best content-preserving ratio".

Usage:
  train/.venv/bin/python train/scripts/sref_regrade.py [--retain 0.75] [arm ...]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sref_sweep_eval import frontier

ROOT = Path("/Volumes/2TBSSD/sref_eval")
DEFAULT_ARMS = ["clean_base", "clean_leak025", "clean_leak",
                "hybrid_arm", "csd_arm", "hybrid_siglipdown", "hybrid_hier"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*", default=None, help="arm names (default: the campaign set)")
    ap.add_argument("--retain", type=float, default=0.75, help="content-retain floor (frac of null)")
    ap.add_argument("--null", default=str(ROOT / "noadapter/scores.json"))
    args = ap.parse_args()

    na = json.loads(Path(args.null).read_text())["aggregate"]
    NS, NL = na["style_sim"]["mean"], na["content_leak"]["mean"]
    NP = na.get("prompt_adherence", {}).get("mean")
    print(f"null: style={NS:+.4f}  leak={NL:+.4f}  prompt_adherence={NP:.4f}   "
          f"| CONTENT GATE: retain >= {args.retain:.0%} of null prompt-adherence\n")

    summary = []
    for arm in (args.arms or DEFAULT_ARMS):
        p = ROOT / arm / "scores.json"
        if not p.exists():
            continue
        fr = frontier(json.loads(p.read_text()))
        print(f"=== {arm} ===")
        best = None
        for sc, a in fr.items():
            if a.get("style_sim") is None or a.get("content_leak") is None:
                continue
            ds = a["style_sim"] - NS
            dl = a["content_leak"] - NL
            ratio = ds / dl if dl else None
            pa = a.get("prompt_adherence")
            retain = (pa / NP) if (pa is not None and NP) else None
            ok = retain is not None and retain >= args.retain
            rs = f"{ratio:.3f}" if ratio is not None else "  —  "
            rt = f"{retain:.2f}" if retain is not None else " — "
            print(f"  s={sc}: ratio={rs:>6}  retain={rt}  {'✓' if ok else '✗ WASH':<6}"
                  f"  (Δstyle={ds:+.3f} Δleak={dl:+.3f} prompt={pa:+.3f})")
            if ok and ratio is not None and (best is None or ratio > best[0]):
                best = (ratio, sc)
        if best:
            print(f"  → best CONTENT-PRESERVING ratio: {best[0]:.3f} @ scale {best[1]}\n")
            summary.append((arm, best[0], best[1]))
        else:
            print(f"  → ALL SCALES WASH (no content-preserving win)\n")
            summary.append((arm, None, None))

    print("=== HONEST RANKING (best content-preserving inj_ratio) ===")
    for arm, r, sc in sorted(summary, key=lambda x: (x[1] is None, -(x[1] or 0))):
        print(f"  {arm:<18} {('%.3f @ %s' % (r, sc)) if r is not None else 'all wash'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
