#!/usr/bin/env python3
"""sref_kv_rank_audit.py — Step 1A.1 of the SREF retrain diagnostic (BACKLOG SREF ADAPTER RETRAIN).

Step 0 proved every adapter mode-collapses (different refs -> ~identical OUTPUT, corr 0.99) while
the INPUT features differ (ref-vs-ref feat corr 0.30-0.42). This probe runs the conditioning ->
injection pipeline OFFLINE (no generation, no training) on several distinct references and measures
cross-reference similarity at EACH stage, to locate WHERE the diversity dies:

    raw features  ->  ip_embeds (perceiver/FiLM out)  ->  K/V = to_k_ip/to_v_ip(ip_embeds)

For hybrid the perceiver (SigLIP) and CSD halves are reported separately. Reports, per stage:
  - mean pairwise cross-ref cosine of the flattened stage tensor
  - the same after centering (removing the across-ref MEAN), i.e. how much of the signal is the
    SHARED constant vs the ref-specific residual: var_ratio = ||residual|| / ||full||.
A stage where cross-ref cosine jumps to ~1 (and var_ratio collapses to ~0) is where references stop
mattering — the architectural fault site.

Usage:
  debug/sref_kv_rank_audit.py --bundle BUNDLE --feat r1.bin r2.bin r3.bin ... [--cond-mode hybrid]
"""
import argparse, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
import mlx.core as mx
from mlx.utils import tree_unflatten
from ip_adapter.model import IPAdapterKlein

# export-key -> training-param-key (superset; includes input-norm affine the fixture map omits)
INV = {
    "perceiver.query_tokens": "image_proj.query_tokens",
    "perceiver.query_proj":   "image_proj.cross_attn.query_proj.weight",
    "perceiver.key_proj":     "image_proj.cross_attn.key_proj.weight",
    "perceiver.value_proj":   "image_proj.cross_attn.value_proj.weight",
    "perceiver.out_proj":     "image_proj.cross_attn.out_proj.weight",
    "perceiver.norm_weight":  "image_proj.norm.weight",
    "perceiver.norm_bias":    "image_proj.norm.bias",
    "perceiver.in_gamma":     "image_proj.in_gamma",
    "perceiver.in_beta":      "image_proj.in_beta",
    "csd.query_tokens":       "csd_proj.query_tokens",
    "csd.film_weight":        "csd_proj.film.weight",
    "csd.film_bias":          "csd_proj.film.bias",
    "csd.norm_weight":        "csd_proj.norm.weight",
    "csd.norm_bias":          "csd_proj.norm.bias",
    "group_gate":             "group_gate",
    "ip_k_stacked":           "to_k_ip_stacked",
    "ip_v_stacked":           "to_v_ip_stacked",
    "ip_scale":               "scale",
}


def reload_bundle(model, bundle_dir):
    w = mx.load(str(Path(bundle_dir) / "adapter_weights.safetensors"))
    upd = {}
    for ek, tk in INV.items():
        if ek not in w:
            continue
        sk = ek + ".scale"
        if sk in w:                                    # int8 per-row dequant
            q = w[ek].astype(mx.float32)
            scale = w[sk].astype(mx.float32).reshape(-1, 1)
            upd[tk] = (q.reshape(-1, q.shape[-1]) * scale).reshape(q.shape)
        else:
            upd[tk] = w[ek].astype(mx.float32)
    model.update(tree_unflatten(list(upd.items())))
    mx.eval(model.parameters())
    return set(upd.keys())


def cross_ref_stats(mat):
    """mat: [n_refs, D] numpy. Returns (mean_pairwise_cosine, var_ratio)."""
    X = mat.astype(np.float64)
    # pairwise cosine
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    U = X / norm
    C = U @ U.T
    n = X.shape[0]
    cos = C[np.triu_indices(n, k=1)]
    # how much of each ref is the shared mean vs the ref-specific residual
    mu = X.mean(axis=0, keepdims=True)
    resid = X - mu
    var_ratio = float(np.linalg.norm(resid) / (np.linalg.norm(X) + 1e-12))
    return float(np.mean(cos)), float(np.min(cos)), float(np.max(cos)), var_ratio


def _stable_rank_report(Wk, Wv, blocks=(0, 5, 12, 24)):
    """Print stable_rank / top-1 energy of stacked K and V weight matrices [N,d,e]."""
    print("\nweight-matrix rank of to_k_ip / to_v_ip (stable_rank: 3072=full, low=collapsing; "
          "top1=σ1²/Σσ² energy in the #1 direction):")
    for label, W in (("to_k_ip", np.array(Wk.astype(mx.float32))),
                     ("to_v_ip", np.array(Wv.astype(mx.float32)))):
        for n in blocks:
            sv = np.linalg.svd(W[n].astype(np.float64), compute_uv=False)
            sr = float((sv ** 2).sum() / (sv[0] ** 2 + 1e-30))
            t1 = float(sv[0] ** 2 / ((sv ** 2).sum() + 1e-30))
            print(f"  {label} block {n:2d}: stable_rank {sr:8.1f}  top1 {t1:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", help="exported adapter bundle dir (full stage audit)")
    ap.add_argument("--ckpt", help="training checkpoint .safetensors (weight-rank only — for "
                                   "per-checkpoint instrumentation; no refs/model needed)")
    ap.add_argument("--feat", nargs="+", help="reference feature .bin files (>=3; bundle mode)")
    ap.add_argument("--cond-mode", default="hybrid", choices=["hybrid", "siglip", "csd"])
    ap.add_argument("--siglip-dim", type=int, default=1152)
    ap.add_argument("--csd-dim", type=int, default=768)
    ap.add_argument("--num-image-tokens", type=int, default=256)
    ap.add_argument("--perceiver-heads", type=int, default=16)
    args = ap.parse_args()

    # --ckpt: weight-rank only, straight from a training checkpoint (the leading indicator the
    # SREF retrain watches per checkpoint — to_v_ip stable_rank should RISE as collapse is fixed).
    if args.ckpt:
        w = mx.load(args.ckpt)
        for pref in ("", "ema."):
            if f"{pref}to_v_ip_stacked" in w:
                print(f"=== {args.ckpt}  [{pref or 'online'}] ===")
                _stable_rank_report(w[f"{pref}to_k_ip_stacked"], w[f"{pref}to_v_ip_stacked"])
        return 0

    if not (args.bundle and args.feat):
        ap.error("bundle mode needs --bundle and --feat (or use --ckpt for weight-rank only)")

    model = IPAdapterKlein(num_blocks=25, hidden_dim=3072,
                           num_image_tokens=args.num_image_tokens, siglip_dim=args.siglip_dim,
                           cond_mode=args.cond_mode, csd_dim=args.csd_dim,
                           perceiver_heads=args.perceiver_heads)
    loaded = reload_bundle(model, args.bundle)
    print(f"loaded {len(loaded)} param groups from {args.bundle}")

    names = [Path(f).stem for f in args.feat]
    rows = args.siglip_dim
    # collect per-ref stage tensors
    raw_sig, raw_csd, ip_emb, sig_tok, csd_tok, K, V = [], [], [], [], [], [], []
    for f in args.feat:
        arr = np.fromfile(f, dtype=np.float32)
        if args.cond_mode == "hybrid":
            arr = arr.reshape(730, args.siglip_dim)
            cond = mx.array(arr[None])                          # [1,730,1152]
            raw_sig.append(arr[:729].ravel())
            raw_csd.append(arr[729, :args.csd_dim].ravel())
        elif args.cond_mode == "siglip":
            arr = arr.reshape(729, args.siglip_dim)
            cond = mx.array(arr[None]); raw_sig.append(arr.ravel())
        else:
            cond = mx.array(arr.reshape(1, -1)); raw_csd.append(arr.ravel())
        emb = model.get_image_embeds(cond)                      # [1, ntok, 3072]
        k, v = model.get_kv_all(emb)                            # [1,25,ntok,3072]
        mx.eval(emb, k, v)
        e = np.array(emb)[0]
        ip_emb.append(e.ravel())
        if args.cond_mode == "hybrid":
            half = args.num_image_tokens // 2
            sig_tok.append(e[:half].ravel()); csd_tok.append(e[half:].ravel())
        K.append(np.array(k)[0].ravel()); V.append(np.array(v)[0].ravel())

    def report(label, lst):
        if not lst:
            return
        m = np.stack(lst)
        mean_c, min_c, max_c, vr = cross_ref_stats(m)
        print(f"  {label:<28} cos mean {mean_c:6.3f}  min {min_c:6.3f}  max {max_c:6.3f}   "
              f"var_ratio {vr:6.3f}")

    print(f"\nrefs ({len(names)}): {names}")
    print("stage cross-reference similarity (cos→1 & var_ratio→0 = references stop mattering):")
    report("raw SigLIP features", raw_sig)
    report("raw CSD vector", raw_csd)
    report("ip_embeds (all tokens)", ip_emb)
    report("  ip_embeds SigLIP half", sig_tok)
    report("  ip_embeds CSD half", csd_tok)
    report("K injection (25 blocks)", K)
    report("V injection (25 blocks)", V)

    # Weight-matrix rank: does to_k_ip/to_v_ip intrinsically collapse inputs toward a few
    # dominant directions? stable_rank = ||W||_F^2 / ||W||_2^2 (= Σσ²/σ1²); 3072 = full,
    # low = dominated by its top singular direction (collapses diverse inputs to ~one output).
    _stable_rank_report(model.to_k_ip_stacked, model.to_v_ip_stacked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
