"""train/lora/train_step.py — the LoRA flow-matching training step (framework Phase 1, piece 2).

Calls the model's STANDARD forward (`flux.transformer(...)`, which runs all blocks with the now
LoRA-wrapped Linears and applies x_embedder/context_embedder/norm_out/proj_out internally) and computes
the v-prediction flow-matching loss. The patchify/unpatchify + position-id prep is replicated EXACTLY
from the (validated) IP-adapter trainer's `_flux_forward_no_ip` / `_pred_from_embeds` so the train↔infer
latent space is inherited, not reinvented. gradients flow ONLY to the LoRA (inject_lora froze the base).
"""
from __future__ import annotations

import mlx.core as mx

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ip_adapter.loss import fused_flow_noise, get_schedule_values


def patchify_pack(latent: mx.array) -> mx.array:
    """[B, C, Lh, Lw] (C=32 VAE latent) → [B, seq_img, 128] packed tokens. Mirrors
    _flux_forward_no_ip Step 1 exactly (the x_embedder input form)."""
    B, C, Lh, Lw = latent.shape
    pH, pW = Lh // 2, Lw // 2
    h = latent.reshape(B, C, pH, 2, pW, 2)
    h = h.transpose(0, 1, 3, 5, 2, 4)
    h = h.reshape(B, C * 4, pH, pW)
    return h.reshape(B, C * 4, pH * pW).transpose(0, 2, 1)        # [B, seq_img, 128]


def unpatchify(pred_seq: mx.array, B: int, C: int, Lh: int, Lw: int) -> mx.array:
    """[B, seq_img, 128] → [B, C, Lh, Lw]. Inverse of patchify_pack; mirrors _pred_from_embeds."""
    pH, pW = Lh // 2, Lw // 2
    p = pred_seq.transpose(0, 2, 1).reshape(B, C * 4, pH, pW)
    p = p.reshape(B, C, 2, 2, pH, pW).transpose(0, 1, 4, 2, 5, 3).reshape(B, C, Lh, Lw)
    return p


def make_position_ids(pH: int, pW: int, seq_txt: int):
    """img_ids [seq_img,4]=(T0,H,W,L0), txt_ids [seq_txt,4]=(0,0,0,L). Mirrors _flux_forward_no_ip Step 2."""
    seq_img = pH * pW
    h_grid = mx.broadcast_to(mx.arange(pH, dtype=mx.int32)[:, None], (pH, pW)).reshape(-1)
    w_grid = mx.broadcast_to(mx.arange(pW, dtype=mx.int32)[None, :], (pH, pW)).reshape(-1)
    z = mx.zeros(seq_img, dtype=mx.int32)
    img_ids = mx.stack([z, h_grid, w_grid, z], axis=1)
    zt = mx.zeros(seq_txt, dtype=mx.int32)
    txt_ids = mx.stack([zt, zt, zt, mx.arange(seq_txt, dtype=mx.int32)], axis=1)
    return img_ids, txt_ids


def lora_loss(flux, latent: mx.array, text_embeds: mx.array, t_int: mx.array,
              noise: mx.array, guidance=None) -> mx.array:
    """v-prediction flow-matching MSE for a LoRA step. latent/noise: [B,32,Lh,Lw]; text_embeds:
    [B,seq_txt,text_dim]; t_int: [B] int timesteps. Frozen base + LoRA forward; loss in latent space."""
    B, C, Lh, Lw = latent.shape
    alpha, sigma = get_schedule_values(t_int)
    noisy, v_target = fused_flow_noise(latent, noise, alpha, sigma)        # latent space [B,C,Lh,Lw]
    hidden = patchify_pack(noisy.astype(text_embeds.dtype))                # [B, seq, 128]
    img_ids, txt_ids = make_position_ids(Lh // 2, Lw // 2, text_embeds.shape[1])
    pred_seq = flux.transformer(hidden, text_embeds, t_int, img_ids, txt_ids, guidance)  # [B,seq,128]
    pred = unpatchify(pred_seq, B, C, Lh, Lw).astype(mx.float32)
    return mx.mean((pred - v_target.astype(mx.float32)) ** 2)
