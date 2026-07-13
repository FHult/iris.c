#!/usr/bin/env python3
"""export_joint_to_c.py — export a joint-backbone SREF checkpoint (LoRA r64 + CSDModulation) into
the two artifacts the C/Metal iris engine consumes:

  1. <out>/joint_lora.safetensors   — the LoRA in canonical Diffusers naming that iris_lora.c loads
     via `--lora`. Rename rule vs the MLX checkpoint keys:
       - strip the leading `flux.`
       - append `.weight` (MLX params are `...lora_A`, iris expects `...lora_A.weight`)
       - DOUBLE-block `to_out` → `to_out.0` (HF ModuleList); single-block `to_out` stays bare
     lora_A=[rank,in], lora_B=[out,rank] already match iris load_adapter — no transpose.
  2. <out>/csd_mod.safetensors       — the CSDModulation MLP (fc1/fc2 weight+bias) for the new C
     csd_mod path (temb += scale * fc2(silu(fc1(csd)))).

Usage: train/.venv/bin/python train/lora/export_joint_to_c.py \
         --ckpt /Volumes/2TBSSD/checkpoints/sref_joint_probe/joint_probe_0007000.safetensors \
         --out  /Volumes/2TBSSD/sref_eval/joint_v1_c_export
"""
import argparse, os
import mlx.core as mx


def rename_lora_key(k: str) -> str | None:
    # k like: flux.transformer.transformer_blocks.4.attn.to_v.lora_B
    if "lora_" not in k:
        return None
    assert k.startswith("flux."), k
    body = k[len("flux."):]                       # transformer.(single_)transformer_blocks.N.attn.PROJ.lora_X
    is_single = "single_transformer_blocks" in body
    # double-block to_out gets the HF ModuleList index .0
    if (not is_single) and ".attn.to_out.lora_" in body:
        body = body.replace(".attn.to_out.lora_", ".attn.to_out.0.lora_")
    return body + ".weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = mx.load(args.ckpt)

    lora, csd = {}, {}
    for k, v in d.items():
        if k.startswith("csd_mod."):
            csd[k[len("csd_mod."):]] = v.astype(mx.float32)   # fc1.weight/bias, fc2.weight/bias
        elif "lora_" in k:
            nk = rename_lora_key(k)
            lora[nk] = v.astype(mx.float32)
    assert len(csd) == 4, sorted(csd)
    lora_path = os.path.join(args.out, "joint_lora.safetensors")
    csd_path = os.path.join(args.out, "csd_mod.safetensors")
    mx.save_safetensors(lora_path, lora)
    mx.save_safetensors(csd_path, csd)
    n_dbl = sum(1 for k in lora if "single_transformer_blocks" not in k and k.endswith("lora_A.weight"))
    n_sgl = sum(1 for k in lora if "single_transformer_blocks" in k and k.endswith("lora_A.weight"))
    print(f"  LoRA: {len(lora)} tensors ({n_dbl} double + {n_sgl} single adapters) -> {lora_path}")
    print(f"  csd_mod: {sorted(csd)} -> {csd_path}")
    for k in sorted(csd):
        print(f"    {k}: {tuple(csd[k].shape)}")
    # sanity: a couple renamed LoRA keys
    for k in sorted(lora)[:2] + [x for x in sorted(lora) if "to_out" in x][:2]:
        print(f"    lora key: {k}  {tuple(lora[k].shape)}")


if __name__ == "__main__":
    main()
