"""train/tests/test_lora_module.py — LoRALinear math + injection/freeze (framework Phase 1).

Hermetic (toy nn.Modules; no Flux load). Verifies the trained delta matches the C engine's apply
math, the identity start (B=0), and that injection freezes the base so gradients flow ONLY to LoRA.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.requires_mps  # exercises MLX arrays

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lora.lora import LoRALinear, inject_lora_double_blocks, DOUBLE_ATTN_TARGETS, lora_param_count
from lora.train_step import patchify_pack, unpatchify, make_position_ids


# ---- toy model mirroring flux.transformer.transformer_blocks[i].attn.<target> ----
class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        for n in DOUBLE_ATTN_TARGETS:
            setattr(self, n, nn.Linear(d, d))

class _Block(nn.Module):
    def __init__(self, d):
        super().__init__(); self.attn = _Attn(d)

class _Transformer(nn.Module):
    def __init__(self, d, n):
        super().__init__(); self.transformer_blocks = [_Block(d) for _ in range(n)]

class _Flux(nn.Module):
    def __init__(self, d, n):
        super().__init__(); self.transformer = _Transformer(d, n)


class TestLoRALinearMath:
    def test_zero_init_is_identity(self):
        base = nn.Linear(8, 8)
        lora = LoRALinear(base, rank=4, alpha=4)
        x = mx.random.normal((3, 8))
        y0, y1 = base(x), lora(x)
        mx.eval(y0, y1)
        assert np.allclose(np.array(y0), np.array(y1), atol=1e-6)   # B=0 → delta 0

    def test_matches_c_apply_math(self):
        # forward - base == scale * (x A^T) B^T  (the exact iris_lora.c lora_apply)
        base = nn.Linear(8, 6)
        lora = LoRALinear(base, rank=4, alpha=8)                    # scale = 2.0
        lora.lora_B = mx.random.normal(lora.lora_B.shape)          # perturb B so delta != 0
        x = mx.random.normal((5, 8))
        delta = np.array(lora(x) - base(x))
        A, B = np.array(lora.lora_A), np.array(lora.lora_B)
        expect = 2.0 * (np.array(x) @ A.T) @ B.T
        assert np.allclose(delta, expect, atol=1e-4)

    def test_scale_is_alpha_over_rank(self):
        lora = LoRALinear(nn.Linear(4, 4), rank=8, alpha=16)
        assert abs(lora.scale - 2.0) < 1e-9


class TestInjectAndFreeze:
    def test_injection_replaces_targets(self):
        flux = _Flux(8, 2)
        injected = inject_lora_double_blocks(flux, rank=4, alpha=4)
        assert len(injected) == 2 * len(DOUBLE_ATTN_TARGETS)
        for b in flux.transformer.transformer_blocks:
            for n in DOUBLE_ATTN_TARGETS:
                assert isinstance(getattr(b.attn, n), LoRALinear)

    def test_only_lora_is_trainable(self):
        flux = _Flux(8, 2)
        inject_lora_double_blocks(flux, rank=4, alpha=4)
        keys = [k for k, _ in tree_flatten(flux.trainable_parameters())]
        assert keys, "no trainable params"
        assert all(("lora_A" in k or "lora_B" in k) for k in keys), keys
        # base linear weights must NOT be trainable
        assert not any(k.endswith("linear.weight") for k in keys)

    def test_gradients_flow_only_to_lora(self):
        flux = _Flux(8, 2)
        inject_lora_double_blocks(flux, rank=4, alpha=4)
        def loss_fn(m, x):
            out = x
            for b in m.transformer.transformer_blocks:
                out = b.attn.to_q(out) + b.attn.to_v(out)
            return mx.sum(out ** 2)
        x = mx.random.normal((3, 8))
        l, g = nn.value_and_grad(flux, loss_fn)(flux, x)
        mx.eval(l, g)
        gkeys = [k for k, _ in tree_flatten(g)]
        assert gkeys and all(("lora_A" in k or "lora_B" in k) for k in gkeys), gkeys

    def test_param_count_matches_rank(self):
        flux = _Flux(8, 1)
        inject_lora_double_blocks(flux, rank=4, alpha=4)
        # per target: A[4,8] + B[8,4] = 64; × 8 targets
        assert lora_param_count(flux) == 8 * (4 * 8 + 8 * 4)


class TestPatchifyRoundTrip:
    def test_patchify_unpatchify_identity(self):
        # the train<->infer-critical reshape must round-trip exactly
        B, C, Lh, Lw = 2, 32, 8, 6
        x = mx.random.normal((B, C, Lh, Lw))
        packed = patchify_pack(x)
        assert packed.shape == (B, (Lh // 2) * (Lw // 2), 128)
        back = unpatchify(packed, B, C, Lh, Lw)
        mx.eval(back)
        assert np.allclose(np.array(x), np.array(back), atol=1e-5)

    def test_position_id_shapes(self):
        pH, pW, seq_txt = 4, 3, 7
        img_ids, txt_ids = make_position_ids(pH, pW, seq_txt)
        mx.eval(img_ids, txt_ids)
        assert img_ids.shape == (pH * pW, 4) and txt_ids.shape == (seq_txt, 4)
        # img: T=0, L=0; txt: first 3 axes 0, L = arange
        assert int(img_ids[:, 0].sum()) == 0 and int(img_ids[:, 3].sum()) == 0
        assert int(txt_ids[:, 3].max()) == seq_txt - 1


class TestExport:
    def test_export_keys_shapes_and_baked_scale(self, tmp_path):
        from lora.export import export_lora_diffusers
        flux = _Flux(8, 2)
        inject_lora_double_blocks(flux, rank=4, alpha=8)         # scale = 2.0
        # set a known B on block0 to_q so we can verify the baked scale
        mod = flux.transformer.transformer_blocks[0].attn.to_q
        mod.lora_B = mx.ones(mod.lora_B.shape)
        p = str(tmp_path / "lora.safetensors")
        n = export_lora_diffusers(flux, p)
        assert n == 2 * len(DOUBLE_ATTN_TARGETS)                 # adapters = blocks*targets
        w = mx.load(p)
        # Diffusers keys present, with to_out -> to_out.0
        assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in w
        assert "transformer.transformer_blocks.0.attn.to_out.0.lora_B.weight" in w
        assert "transformer.transformer_blocks.1.attn.add_v_proj.lora_A.weight" in w
        A = w["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"]
        B = w["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"]
        assert A.shape == (4, 8) and B.shape == (8, 4)           # [rank,in], [out,rank]
        assert A.dtype == mx.float32 and B.dtype == mx.float32
        assert float(B[0, 0]) == 2.0                            # 1.0 * scale(2.0) baked in
