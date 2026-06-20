#!/usr/bin/env python
"""
sref_sweep_eval.py — scale-sweep sref evaluation for an IP-Adapter bundle.

The sref eval outcome is dominated by inference-time --ip-scale, which is
ORTHOGONAL to the training recipe (BACKLOG SREF-EVAL-PARAMS): a single fixed
scale can mis-rank recipes because the coherent operating point is reference-
and prompt-dependent. So this driver evaluates an adapter as a FRONTIER over
ip-scale, not a scalar:

  for each (prompt, style-ref) pair × each ip-scale × each seed:
      iris -d MODEL --ip BUNDLE --ip-features REF.bin --ip-scale S -p PROMPT -> gen.png
  then sref_eval.py scores every (ref, gen) pair (CSD style_sim / content_leak /
  sref_score + SigLIP prompt_adherence), and we aggregate PER-SCALE.

Output (under OUT_ROOT/<name>/):
  gens/...                 generated pngs (resumable — existing pngs are reused)
  pairs.json               flat pair list fed to sref_eval (carries 'scale')
  scores.json              sref_eval per-pair + overall aggregate
  report.html              CSD contact sheet (grouped by ref, sorted by scale)
  frontier.json            per-scale aggregates (the headline for recipe compares)

To compare two recipes fairly: run this for each, then compare style_sim at a
MATCHED content_leak budget (or prompt_adherence floor) off the frontiers —
never a single fixed scale unless every arm is coherent there.

Usage:
  train/.venv/bin/python train/scripts/sref_sweep_eval.py \
      --bundle /Volumes/2TBSSD/sref_sweep/bundle_inputnorm \
      --name v4_baseline \
      --scales 0.3,0.5,0.7 --seeds 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _iris_bin() -> str:
    cand = REPO / "iris"
    return str(cand) if cand.exists() else "iris"


def generate(spec: dict, bundle: str, scales: list[float], seeds: list[int],
             out_dir: Path, log_dir: Path) -> list[dict]:
    """Generate every (pair × scale × seed) gen, skipping existing files.
    Returns the flat pair list for sref_eval (each carries ref/gen/prompt/scale)."""
    refs_dir = Path(spec["refs_dir"])
    feat_dir = Path(spec["feat_dir"])
    model = spec.get("model_dir", "flux-klein-model")
    W = int(spec.get("width", 512))
    H = int(spec.get("height", 512))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[dict] = []
    n_gen = n_skip = 0
    for pi, p in enumerate(spec["pairs"]):
        style = p["style"]
        prompt = p["prompt"]
        ref_img = refs_dir / f"{style}.jpg"
        ref_feat = feat_dir / f"{style}.bin"
        if not ref_img.exists() or not ref_feat.exists():
            print(f"  MISSING ref for style={style} ({ref_img} / {ref_feat}) — skipping",
                  file=sys.stderr)
            continue
        for sc in scales:
            for sd in seeds:
                stem = f"{style}__p{pi:02d}__s{sc}__seed{sd}"
                gen = out_dir / f"{stem}.png"
                if not gen.exists():
                    cmd = [
                        _iris_bin(), "-d", model,
                        "--ip", bundle, "--ip-features", str(ref_feat),
                        "--ip-scale", str(sc),
                        "-p", prompt, "-S", str(sd),
                        "-W", str(W), "-H", str(H),
                        "-o", str(gen),
                    ]
                    lf = log_dir / f"{stem}.log"
                    with open(lf, "w") as fh:
                        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
                    if r.returncode != 0 or not gen.exists():
                        print(f"  GEN FAIL {stem} (rc={r.returncode}) — see {lf}",
                              file=sys.stderr)
                        continue
                    n_gen += 1
                    print(f"  gen {stem}", flush=True)
                else:
                    n_skip += 1
                pairs.append({
                    "ref": str(ref_img), "gen": str(gen), "prompt": prompt,
                    "scale": sc, "seed": sd, "style": style,
                })
    print(f"generation: {n_gen} new, {n_skip} reused, {len(pairs)} pairs total")
    return pairs


def frontier(scores: dict) -> dict:
    """Per-scale aggregates from sref_eval's per-pair rows."""
    rows = [r for r in scores.get("pairs", []) if "error" not in r]
    by_scale: dict = {}
    for r in rows:
        by_scale.setdefault(r.get("scale"), []).append(r)
    out = {}
    for sc, rs in sorted(by_scale.items(), key=lambda kv: (kv[0] is None, kv[0])):
        def m(k):
            vals = [x[k] for x in rs if k in x]
            return round(sum(vals) / len(vals), 4) if vals else None
        out[str(sc)] = {
            "n": len(rs),
            "style_sim": m("style_sim"),
            "content_leak": m("content_leak"),
            "sref_score": m("sref_score"),
            "prompt_adherence": m("prompt_adherence"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="IP-Adapter bundle dir (export_adapter.py)")
    ap.add_argument("--name", required=True, help="run name (subdir under --out-root)")
    ap.add_argument("--eval-set", default="/Volumes/2TBSSD/sref_eval/eval_set.json")
    ap.add_argument("--out-root", default="/Volumes/2TBSSD/sref_eval")
    ap.add_argument("--scales", default="0.3,0.5,0.7")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--csd-weights", default="/Volumes/2TBSSD/models/csd_vit_l_style.safetensors")
    ap.add_argument("--null", default="/Volumes/2TBSSD/sref_eval/noadapter/scores.json",
                    help="no-adapter null-baseline scores.json. Raw sref_score is FLOOR-"
                         "confounded (content_leak ~0.32 even for unrelated images); deltas vs "
                         "this null are the real metric (BACKLOG NULL-FLOOR REFRAME).")
    args = ap.parse_args()

    spec = json.loads(Path(args.eval_set).read_text())
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    run_dir = Path(args.out_root) / args.name
    gens = run_dir / "gens"
    logs = run_dir / "logs"
    pairs = generate(spec, args.bundle, scales, seeds, gens, logs)
    if not pairs:
        print("no pairs generated", file=sys.stderr)
        return 1

    pairs_path = run_dir / "pairs.json"
    pairs_path.write_text(json.dumps(pairs, indent=2))

    scores_path = run_dir / "scores.json"
    report_path = run_dir / "report.html"
    cmd = [sys.executable, str(HERE / "sref_eval.py"),
           "--pairs", str(pairs_path), "--prompt-adherence",
           "--out", str(scores_path), "--report", str(report_path)]
    if Path(args.csd_weights).exists():
        cmd += ["--weights", args.csd_weights]
    print("running sref_eval ...", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("sref_eval failed", file=sys.stderr)
        return r.returncode

    scores = json.loads(scores_path.read_text())
    fr = frontier(scores)

    # Null-relative deltas: raw sref_score is floor-confounded (content_leak ~0.32 even for
    # unrelated images). The real signal is Δstyle − Δleak vs the no-adapter null, and the
    # injection ratio Δstyle/Δleak (need > 1.0 to beat the null). See BACKLOG NULL-FLOOR REFRAME.
    null = None
    if args.null and Path(args.null).exists():
        try:
            _na = json.loads(Path(args.null).read_text()).get("aggregate", {})
            null = {"style_sim": _na["style_sim"]["mean"],
                    "content_leak": _na["content_leak"]["mean"]}
        except Exception as _e:
            print(f"  (null baseline unreadable: {_e})", file=sys.stderr)
    if null is not None:
        for sc, a in fr.items():
            if a.get("style_sim") is None or a.get("content_leak") is None:
                continue
            d_style = round(a["style_sim"] - null["style_sim"], 4)
            d_leak = round(a["content_leak"] - null["content_leak"], 4)
            a["d_style"] = d_style
            a["d_leak"] = d_leak
            a["d_sref"] = round(d_style - d_leak, 4)            # the real objective
            a["inj_ratio"] = round(d_style / d_leak, 3) if d_leak else None  # want > 1.0

    # Keep frontier.json a FLAT scale->agg dict (delta fields added in-place per scale) for
    # backward compatibility with readers that iterate scales; null baseline goes to a sibling.
    (run_dir / "frontier.json").write_text(json.dumps(fr, indent=2))
    if null is not None:
        (run_dir / "null_baseline.json").write_text(json.dumps(null, indent=2))

    print(f"\n=== sref frontier: {args.name} ===")
    if null is not None:
        print(f"null baseline: style_sim={null['style_sim']:+.4f}  content_leak={null['content_leak']:+.4f}")
        print(f"{'scale':>6} {'n':>3} {'Δstyle':>8} {'Δleak':>8} {'Δsref':>8} {'ratio':>6} "
              f"{'(style':>8} {'leak':>7} {'sref)':>8} {'prompt':>8}")
        for sc, a in fr.items():
            def f(x):
                return f"{x:+.4f}" if isinstance(x, (int, float)) else "   —  "
            def g(x):
                return f"{x:.3f}" if isinstance(x, (int, float)) else "  —  "
            print(f"{sc:>6} {a['n']:>3} {f(a.get('d_style')):>8} {f(a.get('d_leak')):>8} "
                  f"{f(a.get('d_sref')):>8} {g(a.get('inj_ratio')):>6} {f(a.get('style_sim')):>8} "
                  f"{f(a.get('content_leak')):>7} {f(a.get('sref_score')):>8} {f(a.get('prompt_adherence')):>8}")
        print("  (Δsref = Δstyle − Δleak is the real objective; inj_ratio = Δstyle/Δleak, want > 1.0)")
    else:
        print(f"{'scale':>6} {'n':>3} {'style_sim':>10} {'leak':>8} {'sref':>8} {'prompt':>8}")
        for sc, a in fr.items():
            def f(x):
                return f"{x:+.4f}" if isinstance(x, (int, float)) else "   —  "
            print(f"{sc:>6} {a['n']:>3} {f(a['style_sim']):>10} {f(a['content_leak']):>8} "
                  f"{f(a['sref_score']):>8} {f(a['prompt_adherence']):>8}")
        print("  (no --null baseline → raw scores only; these are FLOOR-confounded, see BACKLOG)")
    print(f"\nwrote {scores_path}\nwrote {report_path}\nwrote {run_dir/'frontier.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
