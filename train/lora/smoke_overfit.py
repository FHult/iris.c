"""train/lora/smoke_overfit.py — piece-2 mechanics smoke (loads the real Flux model).

Overfit a SINGLE random batch: if the LoRA train step is wired correctly (forward + flow-matching loss
+ LoRA-only gradient + optimizer, base frozen), the loss falls toward 0. Random data is fine here — this
validates the MECHANICS, not the latent-space correctness (that's piece 3, with the real data loader).
Run: train/.venv/bin/python train/lora/smoke_overfit.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

from lora.lora import inject_lora_double_blocks, lora_param_count
from lora.train_step import lora_loss

MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "flux-klein-model")

print("loading Flux2Klein ...", flush=True)
t0 = time.time()
flux = Flux2Klein(model_path=os.path.abspath(MODEL), quantize=None)
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

inj = inject_lora_double_blocks(flux, rank=8, alpha=8)
print(f"injected {len(inj)} LoRA modules; {lora_param_count(flux):,} trainable params", flush=True)

mx.random.seed(0)
B, C, Lh, Lw, seq_txt, text_dim = 1, 32, 16, 16, 64, 7680
latent = (mx.random.normal((B, C, Lh, Lw)) * 0.5).astype(mx.bfloat16)
text_embeds = (mx.random.normal((B, seq_txt, text_dim)) * 0.1).astype(mx.bfloat16)
t_int = mx.array([500], dtype=mx.int32)
noise = mx.random.normal((B, C, Lh, Lw))

opt = optim.AdamW(learning_rate=1e-3)

def step_loss(m):
    return lora_loss(m, latent, text_embeds, t_int, noise)

lg = nn.value_and_grad(flux, step_loss)
print("overfitting one batch (loss should FALL) ...", flush=True)
first = None
for s in range(60):
    loss, grads = lg(flux)
    grads, gn = optim.clip_grad_norm(grads, max_norm=1.0)
    opt.update(flux, grads)
    mx.eval(flux.trainable_parameters(), opt.state, loss)
    lv = float(loss)
    if first is None:
        first = lv
    if s % 5 == 0 or s == 59:
        print(f"  step {s:2d}  loss {lv:.5f}  gnorm {float(gn):.3f}", flush=True)
print(f"RESULT: loss {first:.5f} -> {lv:.5f}  ({'FELL' if lv < first*0.5 else 'did NOT fall enough'})", flush=True)
print("SMOKEDONE", flush=True)
