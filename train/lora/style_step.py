"""train/lora/style_step.py — SREF Phase-1 in-sequence style-token training step.

Shared by the mechanics smoke (smoke_style.py) and the real trainer (train_style_projector.py).
`flux_forward_style` = the LoRA trainer's faithful forward (train_step.flux_forward_lora) + style
tokens concatenated onto the context stream (TEXT | STYLE | IMAGE) with text-like RoPE, so gradients
flow through the frozen DiT back to the StyleProjector (plans/sref-phase1-projector.md). Supports the
same per-block gradient checkpointing as the LoRA path (make_ckpt_blocks) for 512px memory.
"""
from __future__ import annotations

import mlx.core as mx

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ip_adapter.loss import fused_flow_noise, get_schedule_values
from lora.train_step import patchify_pack, unpatchify, make_position_ids


def flux_forward_style(tr, hidden_states, text_embeds, style_tokens, t_int, img_ids, txt_ids,
                       style_ids, ckpt_double=None, ckpt_single=None, guidance=None):
    """flux_forward_lora + in-sequence style tokens. style_tokens: [B, n_style, hidden] (already at
    hidden_dim from the projector — NOT run through context_embedder). Sequence order TEXT|STYLE|IMAGE;
    the line-95-equivalent strip uses encoder_hidden_states.shape[1] (= seq_txt + n_style) so it keeps
    exactly the image tokens. ckpt lists None → raw blocks (parity with flux.transformer)."""
    dt = hidden_states.dtype
    ts = t_int if isinstance(t_int, mx.array) else mx.array(t_int, dtype=dt)
    if ts.ndim == 0:
        ts = mx.full((hidden_states.shape[0],), ts, dtype=dt)
    ts = ts.astype(dt)
    ts = ts * mx.where(mx.max(ts) <= 1.0, 1000.0, 1.0).astype(dt)
    temb = tr.time_guidance_embed(ts, guidance).astype(mx.bfloat16)
    hidden_states = tr.x_embedder(hidden_states)
    encoder_hidden_states = tr.context_embedder(text_embeds)
    encoder_hidden_states = mx.concatenate(
        [encoder_hidden_states, style_tokens.astype(encoder_hidden_states.dtype)], axis=1)
    if img_ids.ndim == 3: img_ids = img_ids[0]
    if txt_ids.ndim == 3: txt_ids = txt_ids[0]
    txt_ids_ext = mx.concatenate([txt_ids, style_ids], axis=0)
    ir = tr.pos_embed(img_ids); trp = tr.pos_embed(txt_ids_ext)
    rope = (mx.concatenate([trp[0], ir[0]], axis=0), mx.concatenate([trp[1], ir[1]], axis=0))
    mod_img = tr.double_stream_modulation_img(temb)
    mod_txt = tr.double_stream_modulation_txt(temb)
    blocks_d = ckpt_double if ckpt_double is not None else tr.transformer_blocks
    for blk in blocks_d:
        encoder_hidden_states, hidden_states = blk(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
            temb_mod_params_img=mod_img, temb_mod_params_txt=mod_txt, image_rotary_emb=rope)
    hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
    mod_single = tr.single_stream_modulation(temb)[0]
    blocks_s = ckpt_single if ckpt_single is not None else tr.single_transformer_blocks
    for blk in blocks_s:
        hidden_states = blk(hidden_states=hidden_states, temb_mod_params=mod_single, image_rotary_emb=rope)
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
    hidden_states = tr.norm_out(hidden_states, temb)
    return tr.proj_out(hidden_states)


def _style_ids(seq_txt: int, n_style: int):
    zt = mx.zeros(n_style, dtype=mx.int32)
    return mx.stack([zt, zt, zt, mx.arange(seq_txt, seq_txt + n_style, dtype=mx.int32)], axis=1)


def style_loss(flux, projector, latent, text_embeds, siglip, t_int, noise,
               ckpt_double=None, ckpt_single=None, guidance=None):
    """v-prediction flow-matching MSE for a StyleProjector step. latent/noise [B,32,Lh,Lw] (BN-packed
    to C's inference space by the caller — VAE-Q1), text_embeds [B,seq_txt,text_dim], siglip
    [B,729,siglip_dim]. Null-style (CFG) = pass siglip already zeroed for dropped samples."""
    B, C, Lh, Lw = latent.shape
    alpha, sigma = get_schedule_values(t_int)
    noisy, v_target = fused_flow_noise(latent, noise, alpha, sigma)
    hidden = patchify_pack(noisy.astype(text_embeds.dtype))
    img_ids, txt_ids = make_position_ids(Lh // 2, Lw // 2, text_embeds.shape[1])
    style_tokens = projector(siglip)
    style_ids = _style_ids(text_embeds.shape[1], style_tokens.shape[1])
    pred_seq = flux_forward_style(flux.transformer, hidden, text_embeds, style_tokens, t_int,
                                  img_ids, txt_ids, style_ids, ckpt_double, ckpt_single, guidance)
    pred = unpatchify(pred_seq, B, C, Lh, Lw).astype(mx.float32)
    return mx.mean((pred - v_target.astype(mx.float32)) ** 2)
