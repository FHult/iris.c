"""train/lora/smoke_style.py — SREF Phase-1 mechanics smoke (plans/sref-phase1-projector.md).

Overfit a single random batch with the StyleProjector as the ONLY trainable module and the DiT frozen.
If the in-sequence style-token path is wired correctly, the flow-matching loss FALLS toward 0 and the
projector's params change — i.e. gradients flow THROUGH the frozen DiT back to the projector (the whole
premise: you CAN train an in-sequence encoder by backprop through the frozen backbone). Random data is
fine here — this validates the MECHANICS, not real style learning (that's the next step, real data +
the scorecard gate). Run: train/.venv/bin/python train/lora/smoke_style.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mlx.utils import tree_flatten

from ip_adapter.model import StyleProjector
from ip_adapter.loss import fused_flow_noise, get_schedule_values
from lora.train_step import patchify_pack, unpatchify, make_position_ids

MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "flux-klein-model")  # distilled: mechanics


def flux_forward_style(tr, hidden_states, text_embeds, style_tokens, t_int, img_ids, txt_ids,
                       style_ids, guidance=None):
    """flux_forward_lora + in-sequence style tokens: concat the projector's tokens onto the context
    stream after context_embedder (TEXT | STYLE), extend the text RoPE with text-like style ids, and
    let the rest of the standard forward run. Sequence order: TEXT | STYLE | IMAGE. Raw blocks (no
    checkpointing) — fine at the tiny smoke resolution."""
    dt = hidden_states.dtype
    ts = t_int if isinstance(t_int, mx.array) else mx.array(t_int, dtype=dt)
    if ts.ndim == 0:
        ts = mx.full((hidden_states.shape[0],), ts, dtype=dt)
    ts = ts.astype(dt)
    ts = ts * mx.where(mx.max(ts) <= 1.0, 1000.0, 1.0).astype(dt)
    temb = tr.time_guidance_embed(ts, guidance).astype(mx.bfloat16)
    hidden_states = tr.x_embedder(hidden_states)
    encoder_hidden_states = tr.context_embedder(text_embeds)
    # ── SREF: inject style tokens into the context stream (already at hidden_dim; NOT via embedder) ──
    encoder_hidden_states = mx.concatenate(
        [encoder_hidden_states, style_tokens.astype(encoder_hidden_states.dtype)], axis=1)
    if img_ids.ndim == 3: img_ids = img_ids[0]
    if txt_ids.ndim == 3: txt_ids = txt_ids[0]
    txt_ids_ext = mx.concatenate([txt_ids, style_ids], axis=0)                # text + style RoPE ids
    ir = tr.pos_embed(img_ids); trp = tr.pos_embed(txt_ids_ext)
    rope = (mx.concatenate([trp[0], ir[0]], axis=0), mx.concatenate([trp[1], ir[1]], axis=0))
    mod_img = tr.double_stream_modulation_img(temb)
    mod_txt = tr.double_stream_modulation_txt(temb)
    for blk in tr.transformer_blocks:
        encoder_hidden_states, hidden_states = blk(
            hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
            temb_mod_params_img=mod_img, temb_mod_params_txt=mod_txt, image_rotary_emb=rope)
    hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
    mod_single = tr.single_stream_modulation(temb)[0]
    for blk in tr.single_transformer_blocks:
        hidden_states = blk(hidden_states=hidden_states, temb_mod_params=mod_single, image_rotary_emb=rope)
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]    # strip text+style, keep img
    hidden_states = tr.norm_out(hidden_states, temb)
    return tr.proj_out(hidden_states)


def style_loss(flux, projector, latent, text_embeds, siglip, t_int, noise, guidance=None):
    B, C, Lh, Lw = latent.shape
    alpha, sigma = get_schedule_values(t_int)
    noisy, v_target = fused_flow_noise(latent, noise, alpha, sigma)
    hidden = patchify_pack(noisy.astype(text_embeds.dtype))
    img_ids, txt_ids = make_position_ids(Lh // 2, Lw // 2, text_embeds.shape[1])
    style_tokens = projector(siglip)                                          # [B, n_style, hidden]
    n_style, seq_txt = style_tokens.shape[1], text_embeds.shape[1]
    zt = mx.zeros(n_style, dtype=mx.int32)
    style_ids = mx.stack([zt, zt, zt, mx.arange(seq_txt, seq_txt + n_style, dtype=mx.int32)], axis=1)
    pred_seq = flux_forward_style(flux.transformer, hidden, text_embeds, style_tokens, t_int,
                                  img_ids, txt_ids, style_ids, guidance)
    pred = unpatchify(pred_seq, B, C, Lh, Lw).astype(mx.float32)
    return mx.mean((pred - v_target.astype(mx.float32)) ** 2)


def main():
    print("loading Flux2Klein ...", flush=True)
    t0 = time.time()
    flux = Flux2Klein(model_path=os.path.abspath(MODEL), quantize=None)
    flux.freeze()
    print(f"  loaded + frozen in {time.time()-t0:.1f}s", flush=True)

    projector = StyleProjector(hidden_dim=3072, num_heads=24, num_style_tokens=192, siglip_dim=1152)
    n = sum(v.size for _, v in tree_flatten(projector.trainable_parameters()))
    print(f"StyleProjector trainable params: {n/1e6:.1f} M (DiT frozen)", flush=True)

    mx.random.seed(0)
    B, C, Lh, Lw, seq_txt, text_dim = 1, 32, 16, 16, 64, 7680     # 128px, tiny — mechanics only
    latent = (mx.random.normal((B, C, Lh, Lw)) * 0.5).astype(mx.bfloat16)
    text_embeds = (mx.random.normal((B, seq_txt, text_dim)) * 0.1).astype(mx.bfloat16)
    siglip = (mx.random.normal((B, 729, 1152)) * 1.0).astype(mx.bfloat16)
    t_int = mx.array([500], dtype=mx.int32)
    noise = mx.random.normal((B, C, Lh, Lw))

    # snapshot a projector param to confirm it actually updates (grads reached it through the DiT)
    p0 = mx.array(projector.query_tokens)

    opt = optim.AdamW(learning_rate=1e-3)
    def step_loss(p):
        return style_loss(flux, p, latent, text_embeds, siglip, t_int, noise)
    lg = nn.value_and_grad(projector, step_loss)

    print("overfitting one batch (loss should FALL; projector should update) ...", flush=True)
    first, lv = None, None
    for s in range(60):
        loss, grads = lg(projector)
        gflat = [v for _, v in tree_flatten(grads)]
        gn = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for g in gflat))
        grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
        opt.update(projector, grads)
        mx.eval(projector.trainable_parameters(), opt.state, loss)
        lv = float(loss)
        if first is None:
            first = lv
        if s % 5 == 0 or s == 59:
            print(f"  step {s:2d}  loss {lv:.5f}  gnorm {float(gn):.4f}", flush=True)
    dparam = float(mx.max(mx.abs(projector.query_tokens - p0)))
    print(f"RESULT: loss {first:.5f} -> {lv:.5f}  ({'FELL' if lv < first*0.5 else 'did NOT fall enough'})")
    print(f"        projector query_tokens max |Δ| = {dparam:.5f}  "
          f"({'UPDATED (grads reached it through the frozen DiT)' if dparam > 1e-4 else 'NOT updated — BUG'})")
    print(f"        loss finite: {bool(mx.isfinite(mx.array(lv)).item())}")
    print("SMOKEDONE", flush=True)


if __name__ == "__main__":
    main()
