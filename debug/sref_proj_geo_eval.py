"""Cross-corpus geometry-gate eval for latent->CSD projectors.

Does a projector PRESERVE the separation the pair loss needs, on a given corpus?
  cos(proj(target), CSD(style-neighbour))  vs  cos(proj(target), CSD(foreign)).
Ground truth (true CSD both sides): ~0.75 vs ~0.12. A projector that flattens the gap
gives the contrastive nothing to push on -> unusable on that corpus.

Usage: proj_geo_eval.py --proj CKPT --vae-dir DIR --csd-dir DIR --neighbors DB [--n 384]
"""
import argparse, glob, json, os, sqlite3, sys
from pathlib import Path
import numpy as np
import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from ip_adapter.latent_csd import LatentCSDProjector, load_vae_bn_stats, bn_pack

MODEL_DIR = str(ROOT / "flux-klein-4b-base")


def _l2(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    ap.add_argument("--vae-dir", required=True)
    ap.add_argument("--csd-dir", required=True, help="dir of *.npz CSD bundles {rec_id: [768]}")
    ap.add_argument("--neighbors", required=True, help="neighbors.sqlite for this corpus")
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    proj = LatentCSDProjector(width=a.width)
    proj.load_weights(a.proj)
    proj.freeze(); proj.eval()
    bn_m, bn_s = load_vae_bn_stats(MODEL_DIR)

    tab = {}
    for p in sorted(glob.glob(os.path.join(a.csd_dir, "*.npz"))):
        tab.update({k: v.astype(np.float32) for k, v in np.load(p).items()})
    have_lat = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.vae_dir, "*.npz"))}

    con = sqlite3.connect(a.neighbors)
    pairs = []
    for rec, nids, _ in con.execute("select rec_id, neighbor_ids, neighbor_cos from neighbors"):
        if rec not in tab or rec not in have_lat:
            continue
        ns = [x for x in json.loads(nids) if x in tab and x != rec]
        if ns:
            pairs.append((rec, ns[0]))
    if not pairs:
        print(f"[{a.label}] NO usable pairs (tab={len(tab)} have_lat={len(have_lat)})")
        return
    rng = np.random.default_rng(0)
    rng.shuffle(pairs)
    pairs = pairs[:a.n]
    keys = list(tab)
    tgt_ids = [t for t, _ in pairs]
    A = _l2(np.stack([tab[b] for _, b in pairs]))                                    # neighbour CSD
    B = _l2(np.stack([tab[keys[i]] for i in rng.integers(0, len(keys), len(pairs))]))  # foreign CSD
    T = _l2(np.stack([tab[t] for t in tgt_ids]))                                     # true target CSD

    def rd(r):
        return np.load(os.path.join(a.vae_dir, f"{r}.npz"))["latent"]
    zs = []
    for i in range(0, len(tgt_ids), 64):
        lat = mx.array(np.stack([rd(r) for r in tgt_ids[i:i + 64]]))
        zs.append(np.array(proj(bn_pack(lat, bn_m, bn_s))))
    Z = np.concatenate(zs)

    pn = float(np.mean(np.sum(Z * A, 1)))
    pf = float(np.mean(np.sum(Z * B, 1)))
    tn = float(np.mean(np.sum(T * A, 1)))
    tf = float(np.mean(np.sum(T * B, 1)))
    pt = float(np.mean(np.sum(Z * T, 1)))
    gap = pn - pf
    true_gap = tn - tf
    retain = gap / true_gap if true_gap else 0.0
    print(f"[{a.label}] n={len(pairs)}  proj_vs_neighbour={pn:.4f} proj_vs_foreign={pf:.4f} "
          f"GAP={gap:.4f}  (true gap {true_gap:.4f}, RETAIN {retain*100:.0f}%)  proj_vs_true={pt:.4f}")


if __name__ == "__main__":
    main()
