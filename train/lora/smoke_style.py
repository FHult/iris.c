"""train/lora/smoke_style.py — SREF Phase-1 mechanics smoke (plans/sref-phase1-projector.md).

Overfit a single random batch with the StyleProjector as the ONLY trainable module and the DiT frozen.
If the in-sequence style-token path is wired correctly, the flow-matching loss FALLS toward 0 and the
projector's params change — i.e. gradients flow THROUGH the frozen DiT back to the projector (the whole
premise: you CAN train an in-sequence encoder by backprop through the frozen backbone). Random data is
fine here — this validates the MECHANICS, not real style learning (that's the next step, real data +
the scorecard gate). Run: train/.venv/bin/python train/lora/smoke_style.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mlx.utils import tree_flatten

from ip_adapter.model import StyleProjector
from lora.style_step import style_loss

MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "flux-klein-model")  # distilled: mechanics


def main():
    print("loading Flux2Klein ...", flush=True)
    t0 = time.time()
    flux = Flux2Klein(model_path=os.path.abspath(MODEL), quantize=None)
    flux.freeze()
    print(f"  loaded + frozen in {time.time()-t0:.1f}s", flush=True)

    projector = StyleProjector(hidden_dim=3072, num_heads=24, num_style_tokens=192, siglip_dim=1152)
    n = sum(v.size for _, v in tree_flatten(projector.trainable_parameters()))
    print(f"StyleProjector trainable params: {n/1e6:.1f} M (DiT frozen)", flush=True)

    mx.random.seed(0)
    B, C, Lh, Lw, seq_txt, text_dim = 1, 32, 16, 16, 64, 7680     # 128px, tiny — mechanics only
    latent = (mx.random.normal((B, C, Lh, Lw)) * 0.5).astype(mx.bfloat16)
    text_embeds = (mx.random.normal((B, seq_txt, text_dim)) * 0.1).astype(mx.bfloat16)
    siglip = (mx.random.normal((B, 729, 1152)) * 1.0).astype(mx.bfloat16)
    t_int = mx.array([500], dtype=mx.int32)
    noise = mx.random.normal((B, C, Lh, Lw))

    # snapshot a projector param to confirm it actually updates (grads reached it through the DiT)
    p0 = mx.array(projector.query_tokens)

    opt = optim.AdamW(learning_rate=1e-3)
    def step_loss(p):
        return style_loss(flux, p, latent, text_embeds, siglip, t_int, noise)
    lg = nn.value_and_grad(projector, step_loss)

    print("overfitting one batch (loss should FALL; projector should update) ...", flush=True)
    first, lv = None, None
    for s in range(60):
        loss, grads = lg(projector)
        gflat = [v for _, v in tree_flatten(grads)]
        gn = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for g in gflat))
        grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
        opt.update(projector, grads)
        mx.eval(projector.trainable_parameters(), opt.state, loss)
        lv = float(loss)
        if first is None:
            first = lv
        if s % 5 == 0 or s == 59:
            print(f"  step {s:2d}  loss {lv:.5f}  gnorm {float(gn):.4f}", flush=True)
    dparam = float(mx.max(mx.abs(projector.query_tokens - p0)))
    print(f"RESULT: loss {first:.5f} -> {lv:.5f}  ({'FELL' if lv < first*0.5 else 'did NOT fall enough'})")
    print(f"        projector query_tokens max |Δ| = {dparam:.5f}  "
          f"({'UPDATED (grads reached it through the frozen DiT)' if dparam > 1e-4 else 'NOT updated — BUG'})")
    print(f"        loss finite: {bool(mx.isfinite(mx.array(lv)).item())}")
    print("SMOKEDONE", flush=True)


if __name__ == "__main__":
    main()
