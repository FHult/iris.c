"""train/lora/export.py — export trained LoRA to the Diffusers safetensors format iris_lora.c loads.

Maps each injected LoRALinear to the Diffusers keys (1:1 with the MLX attr names; `to_out` → `to_out.0`).
Bakes the (alpha/rank) scale into lora_B so the trained strength is realised at `--lora-scale 1.0` (the
C loader has no alpha tensor — it reads only lora_A/lora_B and multiplies by the inference scale). Tensors
are written as float32 (iris_lora.c reads f32 via safetensors_get_f32).
"""
from __future__ import annotations

import mlx.core as mx

from .lora import LoRALinear

# MLX attr name → Diffusers key suffix under `transformer.transformer_blocks.{i}.attn.`
# (matches iris_lora.c load_diffusers: note to_out → "to_out.0").
_DIFFUSERS_SUFFIX = {
    "to_q": "to_q", "to_k": "to_k", "to_v": "to_v", "to_out": "to_out.0",
    "add_q_proj": "add_q_proj", "add_k_proj": "add_k_proj",
    "add_v_proj": "add_v_proj", "to_add_out": "to_add_out",
}


def collect_lora_tensors(flux) -> dict:
    """{diffusers_key: f32 array} for every injected LoRALinear in the double blocks. lora_A [rank,in];
    lora_B [out,rank] with (alpha/rank) baked in."""
    out: dict = {}
    for i, block in enumerate(flux.transformer.transformer_blocks):
        attn = block.attn
        for name, suffix in _DIFFUSERS_SUFFIX.items():
            mod = getattr(attn, name, None)
            if not isinstance(mod, LoRALinear):
                continue
            key = f"transformer.transformer_blocks.{i}.attn.{suffix}"
            out[f"{key}.lora_A.weight"] = mod.lora_A.astype(mx.float32)
            out[f"{key}.lora_B.weight"] = (mod.lora_B * mod.scale).astype(mx.float32)  # bake scale
    return out


def export_lora_diffusers(flux, path: str) -> int:
    """Write the LoRA to `path` (Diffusers safetensors). Returns the number of adapters (key pairs)."""
    tensors = collect_lora_tensors(flux)
    if not tensors:
        raise ValueError("no LoRALinear modules found — inject_lora_double_blocks first")
    mx.save_safetensors(path, tensors)
    return len(tensors) // 2
