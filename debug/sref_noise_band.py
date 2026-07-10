"""Which timestep band can the SREF content-shared-pair contrastive actually operate in?

Two competing forces on the style descriptor `z = proj(x0_pred)`:

  x0_pred = (alpha*noisy - sigma*v_pred) / (alpha^2 + sigma^2)

  LEAK      alpha^2/(alpha^2+sigma^2)  -- the fraction of x0_pred fixed by the model's INPUT.
                                          High at low t: the model CANNOT move style there, so the
                                          contrastive has no lever (and a ref-blind model wins free).
  AUTHORITY sigma/(alpha^2+sigma^2)    -- d x0_pred / d v_pred. Peaks at t~700.
  READABILITY                          -- but at high t a REAL v_pred carries error, and x0_pred
                                          inherits it amplified by AUTHORITY. If proj(x0_pred) stops
                                          tracking the true style, the contrastive gradient is noise.

LEAK and AUTHORITY are analytic (printed below). READABILITY is what this probe MEASURES, by feeding
proj a simulated x0_pred from a v_pred with realistic error and asking whether it still separates the
target's own style from a foreign style.

The external review (Fable) recommended t < 0.7 ("x0_pred is mush at high noise"); the leak/authority
algebra says the opposite. This settles it with the projector in the loop.

  train/.venv/bin/python debug/sref_noise_band.py
"""
import glob, json, os, sqlite3, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))

import numpy as np
import mlx.core as mx

from ip_adapter.latent_csd import LatentCSDProjector, load_vae_bn_stats, bn_pack

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAE_DIR = "/Volumes/2TBSSD/precomputed/vae/v_2232c1"
EVAL_CSD_DIR = "/Volumes/2TBSSD/sref_eval/style_cache"
CKPT = "/Volumes/2TBSSD/checkpoints/latent_csd/latent_csd_projector.safetensors"
N = 256
V_ERRS = (0.0, 0.25, 0.5, 0.75)     # std of per-element v_pred error; our trained runs sit near ~0.55


def _l2(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9)


def main():
    if not os.path.exists(CKPT):
        sys.exit(f"missing projector checkpoint: {CKPT}")
    rng = np.random.default_rng(0)
    mx.random.seed(0)

    tab = {}
    for p in sorted(glob.glob(os.path.join(EVAL_CSD_DIR, "*.npz"))):
        tab.update({k: v.astype(np.float32) for k, v in np.load(p).items()})
    have_lat = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(VAE_DIR, "*.npz"))}
    con = sqlite3.connect(os.path.join(EVAL_CSD_DIR, "neighbors.sqlite"))
    ids = [r for (r,) in con.execute("select rec_id from neighbors") if r in tab and r in have_lat]
    rng.shuffle(ids)
    ids = ids[:N]

    bn_m, bn_s = load_vae_bn_stats(os.path.join(ROOT, "flux-klein-4b-base"))
    proj = LatentCSDProjector()
    proj.load_weights(CKPT)
    proj.eval()

    x0 = mx.array(np.stack([np.load(os.path.join(VAE_DIR, f"{r}.npz"))["latent"] for r in ids]))
    x0 = bn_pack(x0, bn_m, bn_s)
    S_self = _l2(np.stack([tab[r] for r in ids]))
    keys = list(tab)
    S_frgn = _l2(np.stack([tab[keys[i]] for i in rng.integers(0, len(keys), len(ids))]))

    z_clean = np.array(proj(x0))
    print(f"n={len(ids)}   ceiling: cos(proj(clean x0), CSD(self)) = "
          f"{np.mean(np.sum(z_clean*S_self,1)):.4f}  vs foreign "
          f"{np.mean(np.sum(z_clean*S_frgn,1)):.4f}\n")

    hdr = "  ".join(f"e={e:<4}" for e in V_ERRS)
    print(f"{'t':>5} {'leak':>6} {'gain':>6} |  separation  cos(proj(x0_pred),self) - cos(...,foreign)")
    print(f"{'':>5} {'':>6} {'':>6} |  {hdr}")
    print("-" * 78)
    for t in (300, 400, 500, 600, 700, 800, 900, 950):
        a, s = 1.0 - t / 1000.0, t / 1000.0
        den = a * a + s * s
        noise = mx.random.normal(x0.shape)
        noisy = a * x0 + s * noise
        v_tgt = a * noise - s * x0
        cells = []
        for e in V_ERRS:
            v_pred = v_tgt + (e * mx.random.normal(x0.shape) if e > 0 else 0.0)
            x0p = (a * noisy - s * v_pred) / den
            z = np.array(proj(x0p))
            sep = float(np.mean(np.sum(z * S_self, 1)) - np.mean(np.sum(z * S_frgn, 1)))
            cells.append(f"{sep:>6.4f}")
        print(f"{t:>5} {a*a/den:>5.1%} {s/den:>6.3f} |  " + "  ".join(cells), flush=True)

    print("\nleak = fraction of x0_pred fixed by the INPUT (contrastive has no lever where this is high)")
    print("gain = d x0_pred / d v_pred (the model's authority over the style it emits)")
    print("separation must stay well above 0 at the realistic error level for the band to be usable.")


if __name__ == "__main__":
    main()
