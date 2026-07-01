"""Curate a CONTENT-AGNOSTIC look cluster: images that share a low-level LOOK (palette, tone,
contrast, grain) but span DIVERSE content — so a LoRA trained on it learns the transferable look,
not the subject. This is the fix for LORA-TRAIN-3: the CSD-far cluster was coherent BY being
subject-specific (fantasy portraits) → the LoRA entangled style with content.

Method: decode a sample of pool images → a ~17-d content-agnostic LOOK vector (RGB/HSV/luminance
stats, warmth, colorfulness, grain). Standardise. For each candidate seed, take its top-N LOOK
neighbours and score = look_coherence − μ·csd_coherence, where csd_coherence (mean cached-CSD cosine
among the cluster) is LOW when the cluster is content-DIVERSE. So we maximise "same look, varied
subject". Writes {"rec_ids":[...]} for the trainer.

  train/.venv/bin/python curate_lookstyle_subset.py --out subset.json [--per-shard 1000] [--mu 1.0]
"""
import argparse, glob, io, json, os, tarfile
import numpy as np
from PIL import Image

HOT = "/Volumes/2TBSSD/baseline_pool_hot"
CSD = "/Volumes/2TBSSD/sref_eval/style_cache"


def look_vec(im: Image.Image) -> np.ndarray:
    """~17-d content-agnostic 'look': palette, saturation, tonal range/contrast, warmth,
    colorfulness, grain. All statistics over the whole frame → subject-independent."""
    a = np.asarray(im.convert("RGB").resize((96, 96)), dtype=np.float32) / 255.0
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    mx_, mn_ = a.max(-1), a.min(-1)
    sat = np.where(mx_ > 1e-6, (mx_ - mn_) / (mx_ + 1e-6), 0.0)     # HSV saturation
    val = mx_                                                       # HSV value (brightness)
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    p10, p50, p90 = np.percentile(lum, [10, 50, 90])
    rg, yb = R - G, 0.5 * (R + G) - B                              # Hasler-Süsstrunk colorfulness
    colorful = np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)
    grain = np.sqrt(np.diff(lum, axis=0).var() + np.diff(lum, axis=1).var())  # high-freq energy
    return np.array([
        R.mean(), G.mean(), B.mean(), R.std(), G.std(), B.std(),
        sat.mean(), sat.std(), val.mean(), val.std(),
        p10, p50, p90, p90 - p10, R.mean() - B.mean(), colorful, grain], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--per-shard", type=int, default=1000, help="images sampled per shard")
    ap.add_argument("--mu", type=float, default=1.0, help="content-diversity weight (penalise csd coherence)")
    ap.add_argument("--look-floor", type=float, default=0.80, help="min look coherence")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    ids, looks, csds = [], [], []
    for tp in sorted(glob.glob(f"{HOT}/*.tar")):
        sh = os.path.basename(tp)[:-4]
        z = np.load(f"{CSD}/{sh}.npz")
        cache = {k: z[k] for k in z.files}
        with tarfile.open(tp) as t:
            members = [m for m in t.getmembers() if m.name.endswith(".jpg")
                       and m.name[:-4] in cache]
            pick = rng.choice(len(members), size=min(args.per_shard, len(members)), replace=False)
            for i in pick:
                m = members[i]; rid = m.name[:-4]
                try:
                    im = Image.open(io.BytesIO(t.extractfile(m).read()))
                    looks.append(look_vec(im)); csds.append(cache[rid]); ids.append(rid)
                except Exception:
                    continue
        print(f"  {sh}: {len(ids)} total", flush=True)

    L = np.stack(looks).astype(np.float32)
    L = (L - L.mean(0)) / (L.std(0) + 1e-6)                        # z-score per dim
    L /= (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)        # → cosine
    V = np.stack(csds).astype(np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    M, N = len(ids), args.n
    print(f"sampled {M} images; look-dim {L.shape[1]}", flush=True)

    cand = rng.choice(M, size=min(500, M), replace=False)
    lsim = L[cand] @ L.T                                          # [C,M] look similarity
    topk = np.argpartition(-lsim, N, axis=1)[:, :N]
    look_coh = np.take_along_axis(lsim, topk, axis=1).mean(1)
    csd_coh = np.array([ (V[topk[i]] @ V[cand[i]]).mean() for i in range(len(cand)) ])  # content coherence
    score = np.where(look_coh >= args.look_floor, look_coh - args.mu * csd_coh, -1e9)
    bi = int(np.argmax(score)); best = cand[bi]
    top = np.argsort(-(L[best] @ L.T))[:N]
    rec_ids = [ids[i] for i in top]
    lc = float((L[best] @ L[top].T).mean()); cc = float((V[best] @ V[top].T).mean())

    json.dump({"rec_ids": rec_ids, "seed": ids[best], "look_coh": lc, "csd_coh": cc}, open(args.out, "w"))
    print(f"SELECTED seed {ids[best]}  N={N}  (μ={args.mu}, look-floor {args.look_floor})", flush=True)
    print(f"  look coherence {lc:.3f}  (shared low-level look)", flush=True)
    print(f"  csd  coherence {cc:.3f}  (LOWER = more content-diverse; far cluster was ~0.75)", flush=True)
    print(f"  shard spread: {len(set(r.split('_')[0] for r in rec_ids))} of 22 shards", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
