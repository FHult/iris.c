"""train/lora/train_step.py — the LoRA flow-matching training step (framework Phase 1, piece 2).

Calls the model's STANDARD forward (`flux.transformer(...)`, which runs all blocks with the now
LoRA-wrapped Linears and applies x_embedder/context_embedder/norm_out/proj_out internally) and computes
the v-prediction flow-matching loss. The patchify/unpatchify + position-id prep is replicated EXACTLY
from the (validated) IP-adapter trainer's `_flux_forward_no_ip` / `_pred_from_embeds` so the train↔infer
latent space is inherited, not reinvented. gradients flow ONLY to the LoRA (inject_lora froze the base).
"""
from __future__ import annotations

import mlx.core as mx
from mlx.nn.utils import checkpoint as _nn_checkpoint

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


def make_ckpt_blocks(flux):
    """Per-block gradient checkpointing — frees a block's activations after forward and recomputes
    them block-by-block in the backward, so the LoRA backprops through the full frozen transformer at
    512² without the huge-graph stall / memory blowup. Uses `nn.utils.checkpoint` (MODULE-AWARE) NOT
    bare `mx.checkpoint`: the latter only differentiates the function INPUTS, giving ZERO gradient to
    the LoRA params CAPTURED inside the block (verified) — the IP-adapter got away with mx.checkpoint
    only because its blocks were frozen. Returns (ckpt_double, ckpt_single)."""
    tr = flux.transformer
    return ([_nn_checkpoint(b) for b in tr.transformer_blocks],
            [_nn_checkpoint(b) for b in tr.single_transformer_blocks])


def flux_forward_lora(tr, hidden_states, text_embeds, t_int, img_ids, txt_ids,
                      ckpt_double=None, ckpt_single=None, guidance=None):
    """Faithful reimpl of Flux2Transformer.__call__ that loops the (optionally checkpointed) blocks,
    so the LoRA backprops through the full transformer cheaply. Mirrors mflux transformer.__call__ +
    the IP-adapter's _flux_forward_with_ip block-call convention (call_block = ckpt[i] or block).
    hidden_states: patchified [B,seq_img,128]; text_embeds raw [B,seq_txt,dim]. Returns velocity
    [B,seq_img,128]. ckpt lists None → use the raw blocks (parity with flux.transformer)."""
    dt = hidden_states.dtype
    ts = t_int if isinstance(t_int, mx.array) else mx.array(t_int, dtype=dt)
    if ts.ndim == 0:
        ts = mx.full((hidden_states.shape[0],), ts, dtype=dt)
    ts = ts.astype(dt)
    ts = ts * mx.where(mx.max(ts) <= 1.0, 1000.0, 1.0).astype(dt)
    temb = tr.time_guidance_embed(ts, guidance).astype(mx.bfloat16)
    hidden_states = tr.x_embedder(hidden_states)
    encoder_hidden_states = tr.context_embedder(text_embeds)
    if img_ids.ndim == 3: img_ids = img_ids[0]
    if txt_ids.ndim == 3: txt_ids = txt_ids[0]
    ir = tr.pos_embed(img_ids); trp = tr.pos_embed(txt_ids)
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
        hidden_states = blk(hidden_states=hidden_states, temb_mod_params=mod_single,
                            image_rotary_emb=rope)
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
    hidden_states = tr.norm_out(hidden_states, temb)
    return tr.proj_out(hidden_states)


def lora_loss(flux, latent: mx.array, text_embeds: mx.array, t_int: mx.array, noise: mx.array,
              ckpt_double=None, ckpt_single=None, guidance=None) -> mx.array:
    """v-prediction flow-matching MSE for a LoRA step. latent/noise: [B,32,Lh,Lw]; text_embeds:
    [B,seq_txt,text_dim]; t_int: [B] int timesteps. Frozen base + LoRA forward (checkpointed when
    ckpt lists are passed); loss in latent space."""
    B, C, Lh, Lw = latent.shape
    alpha, sigma = get_schedule_values(t_int)
    noisy, v_target = fused_flow_noise(latent, noise, alpha, sigma)        # latent space [B,C,Lh,Lw]
    hidden = patchify_pack(noisy.astype(text_embeds.dtype))                # [B, seq, 128]
    img_ids, txt_ids = make_position_ids(Lh // 2, Lw // 2, text_embeds.shape[1])
    if ckpt_double is None and ckpt_single is None:
        pred_seq = flux.transformer(hidden, text_embeds, t_int, img_ids, txt_ids, guidance)
    else:
        pred_seq = flux_forward_lora(flux.transformer, hidden, text_embeds, t_int, img_ids, txt_ids,
                                     ckpt_double, ckpt_single, guidance)
    pred = unpatchify(pred_seq, B, C, Lh, Lw).astype(mx.float32)
    return mx.mean((pred - v_target.astype(mx.float32)) ** 2)
