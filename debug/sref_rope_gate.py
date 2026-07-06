#!/usr/bin/env python3
"""sref_rope_gate.py — the IN-CONTEXT style-transfer gate for Phase-1 RoPE band-control
(plans/sref-rope-band-control.md). The in-context analogue of debug/sref_ref_discrimination.py.

Holds prompt + seed fixed, varies ONLY the style reference IMAGE, runs the shipped in-context path
(`./iris -i <ref>`), and measures three things per config (one set of iris flags / shuffle setting):

  DISCRIMINATION  max pairwise pixel corr across the styled outputs  — must stay LOW (< 0.90).
                  in-context already discriminates; band-scaling must not destroy that.
  STYLE ADHERENCE mean CSD cosine(output_i, reference_i)             — higher = more style transfer.
  COPYING         mean pixel corr(output_i, reference_i)             — lower = less composition copy.

One config per invocation → one JSON row on stdout (append to a sweep file). Saves an output grid.

  debug/sref_rope_gate.py --label bands_shf0.2_slf1.5 --out-dir /tmp/gate \
      --extra "--sref-shf 0.2 --sref-slf 1.5"            # a band-control cell
  debug/sref_rope_gate.py --label patchshuffle --shuffle-grid 6   # baseline A
  debug/sref_rope_gate.py --label incontext_raw                   # baseline B (no shuffle, no bands)

RUN WITH THE TRAIN VENV: the CSD style encoder is MLX-based, so invoke via
`train/.venv/bin/python debug/sref_rope_gate.py ...` (web/venv lacks mlx → CSD import fails).
"""
import argparse, io, json, os, random, shlex, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CSD_WEIGHTS = "/Volumes/2TBSSD/models/csd_vit_l_style.safetensors"

# 8 diverse refs the adapter collapsed across: 5 WikiArt paintings + line-art/sticker/woodcut.
DEFAULT_REFS = [
    "/Volumes/2TBSSD/sref_eval/refs/artnouveau.jpg",
    "/Volumes/2TBSSD/sref_eval/refs/baroque_portrait.jpg",
    "/Volumes/2TBSSD/sref_eval/refs/cubism_stilllife.jpg",
    "/Volumes/2TBSSD/sref_eval/refs/expressionism_portrait.jpg",
    "/Volumes/2TBSSD/sref_eval/refs/impressionism_landscape.jpg",
    "/Volumes/2TBSSD/sref_eval/refs_simple/churchill_lineart.jpg",
    "/Volumes/2TBSSD/sref_eval/refs_simple/flat_sticker.jpg",
    "/Volumes/2TBSSD/sref_eval/refs_simple/woodcut.jpg",
]


def patch_shuffle(img: Image.Image, grid: int) -> Image.Image:
    """Replica of web/server.py content_destroy_png (deterministic grid-shuffle)."""
    if grid <= 1:
        return img
    img = img.convert("RGB").resize((512, 512))
    ph = pw = 512 // grid
    boxes = [(j * pw, i * ph, (j + 1) * pw, (i + 1) * ph) for i in range(grid) for j in range(grid)]
    patches = [img.crop(b) for b in boxes]
    order = list(range(len(patches)))
    random.Random(1234).shuffle(order)
    canvas = Image.new("RGB", (512, 512))
    for dst, src in zip(boxes, order):
        canvas.paste(patches[src], (dst[0], dst[1]))
    return canvas


def flat(p):
    return np.asarray(Image.open(p).convert("RGB").resize((512, 512)), dtype=np.float64).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="config name (row id)")
    ap.add_argument("--out-dir", default="/tmp/sref_rope_gate")
    ap.add_argument("--model", default="flux-klein-model")
    ap.add_argument("--prompt", default="a cat sitting on a chair")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--shuffle-grid", type=int, default=0, help="patch-shuffle refs (0=off); baseline A uses 6")
    ap.add_argument("--extra", default="", help="extra iris flags, e.g. '--sref-shf 0.2 --sref-slf 1.5'")
    ap.add_argument("--refs", nargs="+", default=DEFAULT_REFS)
    ap.add_argument("--max-corr", type=float, default=0.90)
    ap.add_argument("--jsonl", default=None, help="append the row to this file too")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = shlex.split(args.extra)

    # 1. Prepare (optionally shuffled) reference images and run iris per ref.
    names, ref_used, out_used = [], [], []
    for rp in args.refs:
        name = Path(rp).stem
        names.append(name)
        src = Image.open(rp).convert("RGB")
        ref_path = out_dir / f"ref_{name}.png"
        (patch_shuffle(src, args.shuffle_grid) if args.shuffle_grid > 1 else src.resize((512, 512))).save(ref_path)
        out_path = out_dir / f"out_{name}.png"
        cmd = [str(ROOT / "iris"), "-d", args.model, "-p", args.prompt, "-i", str(ref_path),
               "-S", str(args.seed), "--steps", str(args.steps),
               "-W", str(args.size), "-H", str(args.size), "-o", str(out_path)] + extra
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out_path.exists():
            sys.stderr.write(f"gen failed ({name}):\n{r.stderr[-600:]}\n")
            return 1
        ref_used.append(ref_path); out_used.append(out_path)

    # 2. Pixel-space metrics.
    outs = [flat(p) for p in out_used]
    import itertools
    cross = [float(np.corrcoef(outs[i], outs[j])[0, 1]) for i, j in itertools.combinations(range(len(outs)), 2)]
    mean_cross, max_cross = float(np.mean(cross)), float(np.max(cross))
    # copying: output vs its OWN reference (shuffled ref if shuffle on — use the ORIGINAL ref for a fair
    # composition-copy measure); use the pre-shuffle original.
    orig = [flat(rp) for rp in args.refs]
    copy_corr = float(np.mean([np.corrcoef(outs[i], orig[i])[0, 1] for i in range(len(outs))]))

    # 3. CSD style adherence: cosine(output_i, ORIGINAL reference_i) in CSD space.
    sys.path.insert(0, str(ROOT / "train"))
    from style_encoder.csd_mlx import CSDStyleEncoder, preprocess
    enc = CSDStyleEncoder(CSD_WEIGHTS)
    def csd(paths):
        x = np.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])
        return np.asarray(enc.encode(x)).astype(np.float32)  # [n,768] L2-normalised
    csd_out = csd(out_used)
    csd_ref = csd(args.refs)  # style target = the UNSHUFFLED reference's style
    style_adh = float(np.mean([float(np.dot(csd_out[i], csd_ref[i])) for i in range(len(outs))]))

    # 4. Save grid (ref | out per row) for eyeballing.
    cell = 224
    grid = Image.new("RGB", (cell * 2, cell * len(names)), "white")
    for i, name in enumerate(names):
        grid.paste(Image.open(orig_p := args.refs[i]).convert("RGB").resize((cell, cell)), (0, i * cell))
        grid.paste(Image.open(out_used[i]).convert("RGB").resize((cell, cell)), (cell, i * cell))
    grid.save(out_dir / "grid.png")

    row = {"label": args.label, "shuffle_grid": args.shuffle_grid, "extra": args.extra,
           "n_refs": len(names), "max_cross": round(max_cross, 4), "mean_cross": round(mean_cross, 4),
           "style_adh": round(style_adh, 4), "copy_corr": round(copy_corr, 4),
           "discriminates": max_cross < args.max_corr}
    line = json.dumps(row)
    print(line)
    if args.jsonl:
        with open(args.jsonl, "a") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
