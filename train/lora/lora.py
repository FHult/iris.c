"""train/lora/lora.py — LoRALinear module + injection/freeze for Flux Klein.

Framework Phase 1 (plans/lora-training-pipeline.md). Wraps frozen nn.Linear layers with a trainable
low-rank delta and trains ONLY the delta (base frozen). The math mirrors the C engine exactly so the
trained LoRA round-trips through the Diffusers export → iris_lora.c:

    LoRALinear(x) = base(x) + (alpha/rank) * (x @ A^T) @ B^T      A:[rank,in]  B:[out,rank]
    iris_lora.c:  out += scale * (x @ A^T) @ B^T                  (same; scale = --lora-scale)

B is zero-initialised → the delta is 0 at init (a clean identity start; gradients still flow through B
from step 0). At export the (alpha/rank) factor is baked into B so the trained strength is realised at
--lora-scale 1.0 (the loader has no alpha tensor — iris_lora.c reads only lora_A/lora_B).
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

# Double-stream attention Linears to wrap (the MVP target set). All [3072,3072] in Klein 4B.
# Names match the MLX mflux module attrs AND map 1:1 to the Diffusers export keys.
DOUBLE_ATTN_TARGETS = [
    "to_q", "to_k", "to_v", "to_out",                 # image stream (to_out → diffusers to_out.0)
    "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",  # text stream
]

# Single-stream (Flux2ParallelSelfAttention) fused Linears. mflux fuses QKV+MLP into one input
# projection and attn+MLP into one output projection — these map 1:1 to the C single_linear1/2.
SINGLE_ATTN_TARGETS = [
    "to_qkv_mlp_proj",   # norm → [Q,K,V,gate,up]  → C single_linear1
    "to_out",            # [attn_out, mlp_out] → hidden → C single_linear2
]


class LoRALinear(nn.Module):
    """Frozen `linear` + trainable low-rank delta. forward = base + scale*(x A^T) B^T."""

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float | None = None):
        super().__init__()
        self.linear = linear                          # frozen base (weight [out,in], optional bias)
        out_dim, in_dim = linear.weight.shape
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scale = self.alpha / self.rank
        # A: small normal (fan-in scaled); B: zero → delta 0 at init.
        self.lora_A = mx.random.normal((self.rank, in_dim)) * (in_dim ** -0.5)
        self.lora_B = mx.zeros((out_dim, self.rank))

    def __call__(self, x: mx.array) -> mx.array:
        base = self.linear(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T   # [..., out]
        return base + self.scale * delta


def inject_lora_double_blocks(flux, rank: int = 16, alpha: float | None = None,
                              targets: list[str] | None = None) -> list[tuple[int, str]]:
    """Replace each double block's target attention Linears with LoRALinear, freeze the whole model,
    then unfreeze ONLY the LoRA params. After this, nn.value_and_grad(flux, ...) yields gradients for
    the LoRA alone. Returns the list of (block_index, attr_name) injected.
    Targets default to DOUBLE_ATTN_TARGETS; a name absent on a block is skipped (no error)."""
    targets = targets or DOUBLE_ATTN_TARGETS
    injected: list[tuple[int, str]] = []
    blocks = flux.transformer.transformer_blocks
    for bi, block in enumerate(blocks):
        attn = block.attn
        for name in targets:
            lin = getattr(attn, name, None)
            if not isinstance(lin, nn.Linear):
                continue                              # absent or not a plain Linear → skip
            setattr(attn, name, LoRALinear(lin, rank=rank, alpha=alpha))
            injected.append((bi, name))
    flux.freeze(recurse=True)
    flux.unfreeze(recurse=True, keys=["lora_A", "lora_B"])
    return injected


def inject_lora_single_blocks(flux, rank: int = 16, alpha: float | None = None,
                              targets: list[str] | None = None) -> list[tuple[int, str]]:
    """Wrap each single block's fused attention Linears (to_qkv_mlp_proj, to_out) with LoRALinear,
    then freeze the model and unfreeze ONLY the LoRA params. Recursive unfreeze(keys=[lora_A,lora_B])
    covers any double-block LoRA injected earlier too, so call order doesn't matter. Returns the list
    of (block_index, attr_name) injected."""
    targets = targets or SINGLE_ATTN_TARGETS
    injected: list[tuple[int, str]] = []
    blocks = flux.transformer.single_transformer_blocks
    for bi, block in enumerate(blocks):
        attn = block.attn
        for name in targets:
            lin = getattr(attn, name, None)
            if not isinstance(lin, nn.Linear):
                continue
            setattr(attn, name, LoRALinear(lin, rank=rank, alpha=alpha))
            injected.append((bi, name))
    flux.freeze(recurse=True)
    flux.unfreeze(recurse=True, keys=["lora_A", "lora_B"])
    return injected


def lora_param_count(flux) -> int:
    """Total trainable (LoRA) parameter count — sanity/telemetry."""
    from mlx.utils import tree_flatten
    return sum(p.size for _, p in tree_flatten(flux.trainable_parameters()))
