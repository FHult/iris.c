"""train/lora/style_retrieve.py — the retrieval-hybrid resolver (plans/sref-retrieval-hybrid-project.md).

Given a user reference image, pick the CSD-nearest style LoRA(s) from a trained library and (if blending
more than one) rank-concatenate them into a single merged LoRA the existing `iris --lora` path applies
unchanged. This is the "instant style" resolver: reference -> style, no per-request training.

Library manifest (JSON): a list of {"name", "centroid":[768], "lora":<path>, "scale":<float>}.
Build it from the trained per-cluster LoRAs + their CSD centroids (train/lora/cluster_hot_styles.py).

Blend math (exact): the exported LoRA is delta = B @ A with (alpha/rank) baked into B. For weights wᵢ,
Σ wᵢ·BᵢAᵢ is represented EXACTLY by  merged_A = vstack(Aᵢ)  and  merged_B = hstack(wᵢ·Bᵢ)  per key —
a single rank-(Σrᵢ) LoRA that iris_load_lora reads as-is (it detects rank from tensor shapes).

  train/.venv/bin/python train/lora/style_retrieve.py --query REF.jpg --library lib.json \
      --top-k 2 --tau 0.15 --out /tmp/merged.safetensors
Prints the LoRA path + `iris --lora ...` to run. Run with the train venv (CSD is MLX).
"""
import argparse, json, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mlx.core as mx
from style_encoder.csd_mlx import CSDStyleEncoder, preprocess
from PIL import Image

CSD_WEIGHTS = "/Volumes/2TBSSD/models/csd_vit_l_style.safetensors"


def csd_vec(enc, image_path):
    im = Image.open(image_path).convert("RGB")
    v = np.asarray(enc.encode(np.stack([preprocess(im)])))[0].astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def blend_loras(entries, weights, out_path):
    """Rank-concat blend: merged_A=vstack(A), merged_B=hstack(w*B) per key. entries: [(lora_path, ...)],
    weights aligned. Writes a single merged Diffusers LoRA to out_path."""
    loaded = [dict(mx.load(e["lora"]).items()) for e in entries]
    keys = set(loaded[0])
    for d in loaded[1:]:
        keys &= set(d)
    merged = {}
    for k in keys:
        if k.endswith(".lora_A.weight"):
            merged[k] = mx.concatenate([d[k] for d in loaded], axis=0)              # [Σrank, in]
        elif k.endswith(".lora_B.weight"):
            merged[k] = mx.concatenate(
                [(loaded[i][k] * float(weights[i])) for i in range(len(loaded))], axis=1)  # [out, Σrank]
    mx.save_safetensors(out_path, merged)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="reference image")
    ap.add_argument("--library", required=True, help="library manifest JSON")
    ap.add_argument("--top-k", type=int, default=2, help="max LoRAs to blend")
    ap.add_argument("--tau", type=float, default=0.15, help="softmax temperature on CSD cosine")
    ap.add_argument("--single-thresh", type=float, default=0.9,
                    help="if the top weight exceeds this, use that LoRA alone (no merge)")
    ap.add_argument("--out", default="/tmp/style_merged.safetensors")
    ap.add_argument("--model", default="flux-klein-model")
    args = ap.parse_args()

    lib = json.loads(Path(args.library).read_text())
    C = np.stack([np.asarray(e["centroid"], dtype=np.float32) for e in lib])
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)

    enc = CSDStyleEncoder(CSD_WEIGHTS)
    q = csd_vec(enc, args.query)
    sim = C @ q
    order = np.argsort(-sim)[:args.top_k]
    w = np.exp((sim[order] - sim[order].max()) / args.tau); w /= w.sum()

    print(f"query: {os.path.basename(args.query)}")
    for r, i in enumerate(order):
        print(f"  #{r+1}  {lib[i]['name']:<16} cos {sim[i]:.3f}  weight {w[r]:.3f}  ({lib[i]['lora']})")

    top = order[0]
    if len(order) == 1 or w[0] >= args.single_thresh:
        lora, scale = lib[top]["lora"], lib[top].get("scale", 1.0)
        print(f"\n=> single style: {lib[top]['name']}  (weight {w[0]:.3f} >= {args.single_thresh})")
    else:
        entries = [lib[i] for i in order]
        # fold each LoRA's own calibrated scale into its blend weight
        wsc = [float(w[r]) * float(entries[r].get("scale", 1.0)) for r in range(len(order))]
        blend_loras(entries, wsc, args.out)
        lora, scale = args.out, 1.0
        print(f"\n=> blended {len(order)} styles -> {args.out}  (rank-concat, weights {[round(x,3) for x in wsc]})")

    print(f"\nrun:\n  ./iris -d {args.model} --lora {lora} --lora-scale {scale} -p \"<prompt>\" -o out.png")
    print(json.dumps({"lora": str(lora), "scale": float(scale),
                      "matched": [lib[int(i)]["name"] for i in order],
                      "cos": [round(float(sim[int(i)]), 4) for i in order]}))


if __name__ == "__main__":
    main()
