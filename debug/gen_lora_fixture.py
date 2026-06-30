#!/usr/bin/env python3
"""debug/gen_lora_fixture.py — generate the LoRA train↔infer parity fixture (AGENT protocol).

The MLX `LoRALinear` (training) is the golden; the C `lora_apply` (inference) must reproduce it.
Writes a Diffusers-format safetensors (the exact keys/scale-baking train/lora/export.py produces) +
the input x + the golden delta (LoRALinear(x) - base(x)), all f32. Committed under debug/fixtures/lora/
so `make test-unit` needs no Python/MLX at test time. Regenerate: train/.venv/bin/python debug/gen_lora_fixture.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
import mlx.core as mx
import mlx.nn as nn
from lora.lora import LoRALinear

OUT = os.path.join(os.path.dirname(__file__), "fixtures", "lora")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(0)
RANK, DIM, SEQ, ALPHA = 4, 8, 5, 8.0           # scale = alpha/rank = 2.0 (baked into B at export)
A = rng.standard_normal((RANK, DIM)).astype(np.float32)
B = rng.standard_normal((DIM, RANK)).astype(np.float32)   # non-zero → exercises the matmul (not identity)
x = rng.standard_normal((SEQ, DIM)).astype(np.float32)

lin = nn.Linear(DIM, DIM)
lora = LoRALinear(lin, rank=RANK, alpha=ALPHA)
lora.lora_A = mx.array(A)
lora.lora_B = mx.array(B)
# golden delta = LoRALinear(x) - base(x) = (alpha/rank) * (x A^T) B^T  — what C lora_apply must reproduce
golden = np.array(lora(mx.array(x)) - lin(mx.array(x))).astype(np.float32)

# Diffusers safetensors (mirrors train/lora/export.py: to_q keys, B baked with alpha/rank)
mx.save_safetensors(os.path.join(OUT, "lora.safetensors"), {
    "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": mx.array(A),
    "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": mx.array(B * (ALPHA / RANK)),
})
x.tofile(os.path.join(OUT, "x.bin"))
golden.tofile(os.path.join(OUT, "golden.bin"))
with open(os.path.join(OUT, "shapes.txt"), "w") as f:
    f.write(f"rank={RANK} dim={DIM} seq={SEQ}\n")
print(f"fixture written to {OUT}  (rank {RANK}, dim {DIM}, seq {SEQ}; golden |max| {np.abs(golden).max():.3f})")
