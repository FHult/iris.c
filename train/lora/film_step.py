"""train/lora/film_step.py — SREF experiment B: CSD→timestep-modulation (FiLM) training step.

Companion to lora/style_step.py (the in-sequence style-token path that was CLOSED by
SREF-STYLE-CFG-PROBE). Instead of concatenating style tokens onto the context stream — an
OPTIONAL attention channel the frozen DiT down-weights at high noise — this injects the
content-invariant CSD style vector into the DiT's timestep/guidance modulation embedding `temb`
BEFORE the (frozen) adaLN modulation MLPs. `temb` feeds every block's scale/shift/gate and the
final norm at every noise level, so the model cannot ignore it. Only the small CSDModulation MLP
is trainable; the DiT is frozen and gradients flow back through the frozen modulation MLPs.

Shares the exact latent space (VAE-Q1 BN-packed), scheduler, and grad-checkpointing as the LoRA /
style-token paths so train↔infer is inherited, not reinvented.
"""
from __future__ import annotations

import mlx.core as mx

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ip_adapter.loss import fused_flow_noise, get_schedule_values
from lora.train_step import patchify_pack, unpatchify, make_position_ids


def flux_forward_film(tr, hidden_states, text_embeds, csd_mod, csd_vec, t_int, img_ids, txt_ids,
                      ckpt_double=None, ckpt_single=None, guidance=None):
    """flux_forward_lora with the CSD style vector FiLM'd into `temb`. csd_vec: [B, csd_dim]
    (L2-normalised). No sequence extension — the only change vs the plain forward is one add on
    temb (`temb += csd_mod(csd_vec)`), so mod_img/mod_txt/mod_single and norm_out all see the
    style-shifted embedding. ckpt lists None → raw blocks (parity with flux.transformer)."""
    dt = hidden_states.dtype
    ts = t_int if isinstance(t_int, mx.array) else mx.array(t_int, dtype=dt)
    if ts.ndim == 0:
        ts = mx.full((hidden_states.shape[0],), ts, dtype=dt)
    ts = ts.astype(dt)
    ts = ts * mx.where(mx.max(ts) <= 1.0, 1000.0, 1.0).astype(dt)
    temb = tr.time_guidance_embed(ts, guidance).astype(mx.bfloat16)
    temb = temb + csd_mod(csd_vec).astype(temb.dtype)          # ← THE injection (unignorable channel)
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
        hidden_states = blk(hidden_states=hidden_states, temb_mod_params=mod_single, image_rotary_emb=rope)
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
    hidden_states = tr.norm_out(hidden_states, temb)
    return tr.proj_out(hidden_states)


def film_loss(flux, csd_mod, latent, text_embeds, csd_vec, t_int, noise,
              ckpt_double=None, ckpt_single=None, guidance=None):
    """v-prediction flow-matching MSE for a CSDModulation step. latent/noise [B,32,Lh,Lw] (BN-packed
    to C's inference space — VAE-Q1), text_embeds [B,seq_txt,text_dim], csd_vec [B,csd_dim]
    (L2-normalised). Null-style (CFG) = pass csd_vec already zeroed for dropped samples."""
    B, C, Lh, Lw = latent.shape
    alpha, sigma = get_schedule_values(t_int)
    noisy, v_target = fused_flow_noise(latent, noise, alpha, sigma)
    hidden = patchify_pack(noisy.astype(text_embeds.dtype))
    img_ids, txt_ids = make_position_ids(Lh // 2, Lw // 2, text_embeds.shape[1])
    pred_seq = flux_forward_film(flux.transformer, hidden, text_embeds, csd_mod, csd_vec, t_int,
                                 img_ids, txt_ids, ckpt_double, ckpt_single, guidance)
    pred = unpatchify(pred_seq, B, C, Lh, Lw).astype(mx.float32)
    return mx.mean((pred - v_target.astype(mx.float32)) ** 2)
