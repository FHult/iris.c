"""Does the SREF content-shared-pair step fit an M1 Max 32 GB? Measure, don't assume.

The old batch>1 in-batch-contrastive recipe wedged MLX: batch 2 AND batch 4 consumed all 32 GB and
never reached step 0 while compiling the first value_and_grad graph. The corrected pair loss runs the
two branches as SEQUENTIAL batch-1 passes with summed grad trees, so the graph MLX builds is the
batch-1 graph that already worked. This walks the ladder on the REAL 4B base and prints peak memory
and wall time for each rung, so the claim is a measurement rather than an argument.

Rungs: forward-only, one batch-1 backward (recon), then the full two-pass pair step.
Working ceiling is ~21.5 GB (BACKLOG TRAIN-7); above that the machine gets unstable.

  train/.venv/bin/python debug/sref_memory_ladder.py            # 256px (the probe's resolution)
  train/.venv/bin/python debug/sref_memory_ladder.py --px 512
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

from ip_adapter.model import CSDModulation
from ip_adapter.latent_csd import LatentCSDProjector, load_vae_bn_stats
from ip_adapter.loss import get_schedule_values, fused_flow_noise, reconstruct_x0, pair_row_ce
from lora.lora import inject_lora_double_blocks, inject_lora_single_blocks
from lora.train_step import patchify_pack, unpatchify, make_position_ids, make_ckpt_blocks
from lora.film_step import flux_forward_film

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GB = 2 ** 30
CEILING_GB = 21.5


class _Joint(nn.Module):
    def __init__(self, flux, csd_mod):
        super().__init__()
        self.flux, self.csd_mod = flux, csd_mod


def _l2(x):
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", type=int, default=256)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--text-seq", type=int, default=128)
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute block activations in the backward (exact math, less memory). The "
                         "old wedge was checkpointing x a BATCHED graph; at batch 1 it may be fine.")
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "flux-klein-4b-base"))
    args = ap.parse_args()

    L = args.px // 8
    print(f"loading Flux2Klein ({args.model_dir}) @ {args.px}px (latent {L}x{L}) ...", flush=True)
    flux = Flux2Klein(model_path=args.model_dir, quantize=None)
    inj = inject_lora_double_blocks(flux, rank=args.rank)
    inj += inject_lora_single_blocks(flux, rank=args.rank)
    csd_mod = CSDModulation(hidden_dim=3072, csd_dim=768, mlp_dim=1024)
    joint = _Joint(flux, csd_mod)
    n = sum(v.size for _, v in tree_flatten(joint.trainable_parameters()))
    proj = LatentCSDProjector()
    proj.freeze(); proj.eval()
    load_vae_bn_stats(args.model_dir)
    ckpt_d, ckpt_s = make_ckpt_blocks(flux) if args.grad_checkpoint else (None, None)
    print(f"  {len(inj)} LoRA modules (r{args.rank}), {n/1e6:.2f} M trainable"
          f"{'  [grad checkpointing ON]' if args.grad_checkpoint else ''}", flush=True)

    mx.random.seed(0)
    latent = (mx.random.normal((1, 32, L, L)) * 0.5).astype(mx.bfloat16)
    text = (mx.random.normal((1, args.text_seq, 7680)) * 0.1).astype(mx.bfloat16)
    s_a, s_b = _l2(mx.random.normal((1, 768))), _l2(mx.random.normal((1, 768)))
    t_int = mx.full((1,), 850, dtype=mx.int32)
    alpha, sigma = get_schedule_values(t_int)
    noisy, v_target = fused_flow_noise(latent, mx.random.normal(latent.shape), alpha, sigma)
    mx.eval(noisy, v_target)

    def fwd(J, csd):
        hidden = patchify_pack(noisy.astype(text.dtype))
        img_ids, txt_ids = make_position_ids(L // 2, L // 2, text.shape[1])
        seq = flux_forward_film(J.flux.transformer, hidden, text, J.csd_mod, csd, t_int,
                                img_ids, txt_ids, ckpt_d, ckpt_s, 3.5)
        return unpatchify(seq, 1, 32, L, L)

    def z(v):
        return _l2(proj(reconstruct_x0(noisy, v, alpha, sigma).astype(mx.float32)))

    def loss_a(J):
        v = fwd(J, s_a)
        return mx.mean((v.astype(mx.float32) - v_target.astype(mx.float32)) ** 2) \
            + 0.5 * pair_row_ce(z(v), s_a, s_b, 0.1)

    def loss_b(J):
        return 0.5 * pair_row_ce(z(fwd(J, s_b)), s_b, s_a, 0.1)

    def loss_recon(J):
        v = fwd(J, s_a)
        return mx.mean((v.astype(mx.float32) - v_target.astype(mx.float32)) ** 2)

    def rung(name, fn):
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.time()
        out = fn()
        mx.eval(out)
        dt = time.time() - t0
        peak = mx.get_peak_memory() / GB
        ok = "OK " if peak < CEILING_GB else "OVER"
        print(f"  [{ok}] {name:<44} peak {peak:6.2f} GB   {dt:6.2f} s", flush=True)
        return peak

    print(f"\nceiling {CEILING_GB} GB (BACKLOG TRAIN-7). rungs:", flush=True)
    rung("forward only (weights resident)", lambda: fwd(joint, s_a))
    rung("forward only (2nd call, warm)", lambda: fwd(joint, s_a))
    rung("backward: recon only, batch 1", lambda: nn.value_and_grad(joint, loss_recon)(joint))
    rung("backward: row A (recon + pair), batch 1", lambda: nn.value_and_grad(joint, loss_a)(joint))
    rung("backward: row B (pair only),   batch 1", lambda: nn.value_and_grad(joint, loss_b)(joint))

    # The real step: two sequential passes, eval between, grads summed.
    def full_step():
        _, g_a = nn.value_and_grad(joint, loss_a)(joint)
        mx.eval(g_a)                                   # free pass A's graph before pass B
        _, g_b = nn.value_and_grad(joint, loss_b)(joint)
        mx.eval(g_b)
        return tree_map(lambda x, y: x + y, g_a, g_b)

    print()
    p = rung("FULL STEP: two sequential passes + sum", full_step)
    t0 = time.time()
    for _ in range(3):
        mx.eval(full_step())
    print(f"\n  steady-state: {3/(time.time()-t0):.3f} it/s "
          f"({(time.time()-t0)/3:.1f} s/step), peak {mx.get_peak_memory()/GB:.2f} GB", flush=True)
    print(f"\nVERDICT: {'FITS' if p < CEILING_GB else 'DOES NOT FIT'} "
          f"({p:.2f} GB vs {CEILING_GB} GB ceiling)", flush=True)


if __name__ == "__main__":
    main()
