#!/usr/bin/env python3
"""SREF scorecard (Phase 0 of plans/sref-learned-encoder-project.md).

The gate every prior SREF effort lacked: it separates the axes that actually matter and reports
them PER REFERENCE TYPE, because a single mean hid the band-control painterly failure
(BACKLOG SREF-STYLE-CEILING). For a given method (a set of `iris` flags) it renders one output per
reference at a FIXED, UNRELATED prompt+seed, then scores each output against its reference:

  STYLE ADHERENCE   how much the reference's LOOK transferred, measured two independent ways:
                      style_csd   = CSD cosine(out, ref)                (content-invariant style)
                      style_gram  = -VGG Gram distance(out, ref)        (texture/brushwork)
                    reported as the DELTA vs a no-reference baseline (the reference's contribution).
  COMPOSITION LEAK  structural SSIM(out, ref) on grayscale (NOT pixel corr — captures LAYOUT copy),
                    as the delta vs no-ref (how much the reference pulled structure toward itself).
                    With an unrelated prompt, positive leak = the reference imposed its composition.
  PROMPT ADHERENCE  CLIP cosine(out, prompt) — did we keep the prompt's subject?

A method WINS (project gate) only if it raises style adherence on PAINTERLY + SEMI_REAL references
while composition-leak stays near zero and prompt-adherence holds. Phase-0 validation: band-control
should reproduce the known ceiling — style delta clearly higher on `graphic` than `painterly`.

Run with the train venv (CSD is MLX; VGG/CLIP are torch):
  train/.venv/bin/python debug/sref_scorecard.py --label bands_default --extra "--sref-shf 0.0 --sref-slf 1.5"
  train/.venv/bin/python debug/sref_scorecard.py --label raw_incontext                # no bands
  # add --refs woodcut baroque_portrait   to score a fast subset.
Requires the GPU relatively free (each render loads the distilled model). Outputs cached under
--out-dir; re-runs reuse them unless --regen. Appends a JSON summary row to --jsonl if given.
"""
import argparse, json, subprocess, sys, os
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CSD_WEIGHTS = "/Volumes/2TBSSD/models/csd_vit_l_style.safetensors"
EVAL_SET = ROOT / "debug" / "sref_eval_set.json"


# ── generation ───────────────────────────────────────────────────────────────
def render(out_path, prompt, seed, size, steps, model, ref=None, extra=(), regen=False):
    if out_path.exists() and not regen:
        return True
    cmd = [str(ROOT / "iris"), "-d", model, "-p", prompt, "-S", str(seed), "--steps", str(steps),
           "-W", str(size), "-H", str(size), "-o", str(out_path)]
    if ref:
        cmd += ["-i", str(ref)]
    cmd += list(extra)
    r = subprocess.run(["caffeinate", "-dimsu"] + cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out_path.exists():
        sys.stderr.write(f"render failed ({out_path.name}): {r.stderr[-400:]}\n")
        return False
    return True


# ── metrics ──────────────────────────────────────────────────────────────────
def load_rgb(p, size=512):
    return Image.open(p).convert("RGB").resize((size, size))


class Metrics:
    def __init__(self):
        import torch, torchvision, open_clip
        sys.path.insert(0, str(ROOT / "train"))
        from style_encoder.csd_mlx import CSDStyleEncoder, preprocess
        self.torch = torch
        self._csd = CSDStyleEncoder(CSD_WEIGHTS)
        self._csd_pre = preprocess
        # VGG16 feature maps for Gram (texture/style)
        w = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        self._vgg = torchvision.models.vgg16(weights=w).features.eval()
        for p in self._vgg.parameters():
            p.requires_grad_(False)
        self._vgg_layers = {3: "r1_2", 8: "r2_2", 15: "r3_3"}  # relu blocks
        self._imnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._imnet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # CLIP for prompt adherence
        self._clip, _, self._clip_pre = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self._clip.eval()
        self._clip_tok = open_clip.get_tokenizer("ViT-B-32")

    # --- style: CSD cosine ---
    def csd(self, imgs):
        x = np.stack([self._csd_pre(im) for im in imgs])
        return np.asarray(self._csd.encode(x)).astype(np.float32)  # [n,768] L2

    def csd_cos(self, a_vec, b_vec):
        return float(np.dot(a_vec, b_vec))

    # --- style: VGG Gram distance ---
    def _vgg_gram(self, im):
        t = self.torch
        x = t.from_numpy(np.asarray(im.resize((224, 224)), dtype=np.float32) / 255.0)
        x = x.permute(2, 0, 1).unsqueeze(0)
        x = (x - self._imnet_mean) / self._imnet_std
        grams = {}
        h = x
        with t.no_grad():
            for i, layer in enumerate(self._vgg):
                h = layer(h)
                if i in self._vgg_layers:
                    b, c, hh, ww = h.shape
                    f = h.reshape(c, hh * ww)
                    grams[i] = (f @ f.t()) / (c * hh * ww)
                    if i == max(self._vgg_layers):
                        break
        return grams

    def gram_dist(self, im_a, im_b):
        ga, gb = self._vgg_gram(im_a), self._vgg_gram(im_b)
        t = self.torch
        return float(np.mean([t.mean((ga[i] - gb[i]) ** 2).item() for i in ga]))

    # --- composition leak: structural SSIM on grayscale ---
    def ssim(self, im_a, im_b, size=256):
        from scipy.ndimage import gaussian_filter
        a = np.asarray(im_a.convert("L").resize((size, size)), dtype=np.float64)
        b = np.asarray(im_b.convert("L").resize((size, size)), dtype=np.float64)
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        g = lambda x: gaussian_filter(x, 1.5)
        mu_a, mu_b = g(a), g(b)
        va, vb = g(a * a) - mu_a ** 2, g(b * b) - mu_b ** 2
        vab = g(a * b) - mu_a * mu_b
        s = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
        return float(np.clip(s.mean(), -1, 1))

    # --- prompt adherence: CLIP image-text cosine ---
    def clip_prompt(self, im, prompt):
        t = self.torch
        with t.no_grad():
            iv = self._clip.encode_image(self._clip_pre(im).unsqueeze(0))
            tv = self._clip.encode_text(self._clip_tok([prompt]))
            iv = iv / iv.norm(dim=-1, keepdim=True)
            tv = tv / tv.norm(dim=-1, keepdim=True)
            return float((iv @ tv.t()).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--extra", default="", help="extra iris flags, e.g. '--sref-shf 0.0 --sref-slf 1.5'")
    ap.add_argument("--model", default="flux-klein-model")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out-dir", default="/tmp/sref_scorecard")
    ap.add_argument("--refs", nargs="*", default=None, help="subset of ref ids (default: all)")
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--score-only", action="store_true",
                    help="do NOT render (no C iris) — score PNGs pre-rendered by another path "
                         "(e.g. debug/sref_infer_style.py, the learned-projector gate). Fails loudly "
                         "if the expected PNGs are missing.")
    ap.add_argument("--no-input-ref", action="store_true",
                    help="render WITHOUT -i (method style is in the flags, e.g. --lora), scoring the "
                         "output against each ref. For the retrieval-hybrid per-style LoRA gate.")
    args = ap.parse_args()

    import shlex
    spec = json.loads(EVAL_SET.read_text())
    prompt, seed = spec["eval_prompt"], spec["eval_seed"]
    refs = spec["refs"]
    if args.refs:
        refs = [r for r in refs if r["id"] in args.refs]
    extra = shlex.split(args.extra)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / args.label; md.mkdir(exist_ok=True)
    base_png = out_dir / f"_baseline_s{seed}.png"

    if args.score_only:
        # score PNGs already rendered by an external path (the learned rail isn't in the C binary yet)
        if not base_png.exists():
            sys.exit(f"--score-only: missing baseline {base_png} (run debug/sref_infer_style.py first)")
        rendered = [(r, md / f"{r['id']}.png") for r in refs if (md / f"{r['id']}.png").exists()]
        if not rendered:
            sys.exit(f"--score-only: no per-ref PNGs under {md} (run debug/sref_infer_style.py first)")
    else:
        # 1. no-reference baseline (shared across methods; the reference's effect is measured vs this)
        if not render(base_png, prompt, seed, args.size, args.steps, args.model, ref=None, regen=args.regen):
            sys.exit("baseline render failed")
        # 2. one render per reference with the method's flags. --no-input-ref: the method's style comes
        #    from the flags (e.g. --lora), NOT an in-context image, so render WITHOUT -i; the ref is used
        #    only for scoring. (For a fixed-weight LoRA the per-ref renders are identical — gate a subset.)
        rendered = []
        for r in refs:
            op = md / f"{r['id']}.png"
            rref = None if args.no_input_ref else r["path"]
            if render(op, prompt, seed, args.size, args.steps, args.model, ref=rref, extra=extra, regen=args.regen):
                rendered.append((r, op))
        if not rendered:
            sys.exit("no renders")

    # 3. score
    M = Metrics()
    base_img = load_rgb(base_png)
    base_csd = M.csd([base_img])[0]
    rows = []
    for r, op in rendered:
        ref_img, out_img = load_rgb(r["path"]), load_rgb(op)
        cs = M.csd([ref_img, out_img])
        ref_csd, out_csd = cs[0], cs[1]
        style_csd_delta = M.csd_cos(out_csd, ref_csd) - M.csd_cos(base_csd, ref_csd)
        style_gram_delta = M.gram_dist(base_img, ref_img) - M.gram_dist(out_img, ref_img)  # >0 = closer
        leak_delta = M.ssim(out_img, ref_img) - M.ssim(base_img, ref_img)
        prompt_adh = M.clip_prompt(out_img, prompt)
        rows.append({"id": r["id"], "type": r["type"], "style_csd_delta": style_csd_delta,
                     "style_gram_delta": style_gram_delta, "leak_delta": leak_delta,
                     "prompt_adh": prompt_adh})

    # 4. aggregate per type + overall
    def agg(subset):
        n = len(subset)
        return {k: round(float(np.mean([r[k] for r in subset])), 4)
                for k in ("style_csd_delta", "style_gram_delta", "leak_delta", "prompt_adh")} | {"n": n}
    types = sorted(set(r["type"] for r in rows))
    per_type = {t: agg([r for r in rows if r["type"] == t]) for t in types}
    overall = agg(rows)

    print(f"\n=== SREF scorecard: {args.label}   (prompt: {prompt!r}, seed {seed}) ===")
    print(f"{'type':<11}{'n':>3}  {'styleCSD Δ':>11}{'styleGram Δ':>12}{'leak Δ':>9}{'promptAdh':>11}")
    print("-" * 58)
    for t in types + ["OVERALL"]:
        a = per_type[t] if t != "OVERALL" else overall
        print(f"{t:<11}{a['n']:>3}  {a['style_csd_delta']:>11.4f}{a['style_gram_delta']:>12.5f}"
              f"{a['leak_delta']:>9.4f}{a['prompt_adh']:>11.4f}")
    print("\nstyleCSD Δ / styleGram Δ > 0 = style transferred (bigger = more); leak Δ ~0 good, >0 = "
          "composition copied; promptAdh higher = kept the subject.")

    summary = {"label": args.label, "extra": args.extra, "prompt": prompt, "seed": seed,
               "per_type": per_type, "overall": overall, "per_ref": rows}
    if args.jsonl:
        with open(args.jsonl, "a") as f:
            f.write(json.dumps(summary) + "\n")
    (out_dir / f"{args.label}_scorecard.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out_dir / (args.label + '_scorecard.json')}")


if __name__ == "__main__":
    main()
