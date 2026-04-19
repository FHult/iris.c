"""
train/scripts/precompute_all.py — Unified single-pass precompute.

Reads each shard once and writes all requested outputs:
  - Qwen3 4-bit quantised text embeddings  → {qwen3_output}/{id}.npz
  - VAE int8 quantised latents             → {vae_output}/{id}.npz
  - SigLIP 4-bit quantised features        → {siglip_output}/{id}.npz  (--siglip)

Saves ~10-12h vs running the three scripts sequentially (~22h total):
each shard's tar is opened once instead of twice or three times, and all
models are loaded a single time at worker startup via Pool initializer.

Within each shard:
  Phase 1  — sequential tar read → list of (rec_id, jpg_bytes, caption)
  Phase 2a — Qwen3 pass: tokenise+sort all captions, batched forward
  Phase 2b — VAE pass:   1-ahead image decode prefetch while GPU encodes
  Phase 2c — SigLIP pass (optional): same prefetch pattern as VAE

Default workers=1 (GPU-bound; multiple workers contend for Metal).

Reference: plans/ip-adapter-training.md §2.7
"""

import argparse
import glob
import io
import multiprocessing
import os
import sys
import tarfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import subprocess
    _PERF_CORES = int(subprocess.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def _quantize_int8(arr: np.ndarray):
    """Per-channel absmax int8 quantisation for VAE latents [32, H, W]."""
    scale = np.abs(arr).max(axis=(1, 2), keepdims=True) / 127.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return q, scale.astype(np.float16)


def _quantize_4bit(arr: np.ndarray):
    """Per-token absmax 4-bit quantisation (nibble-packed) for [..., dim] arrays."""
    scale = np.abs(arr).max(axis=-1, keepdims=True) / 7.0
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.round(arr / scale), -8, 7).astype(np.int8)
    q_packed = ((q[..., 0::2] & 0x0F) | ((q[..., 1::2] & 0x0F) << 4)).astype(np.uint8)
    return q_packed, scale.astype(np.float16)


# ---------------------------------------------------------------------------
# Shard iteration
# ---------------------------------------------------------------------------

def iter_shard(shard_path: str):
    """Yield (id, jpg_bytes, caption) for records that have both image and text."""
    try:
        with tarfile.open(shard_path) as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            keys = {}
            for name in members:
                stem, _, ext = name.rpartition(".")
                keys.setdefault(stem, {})[ext.lower()] = name

            for stem, exts in keys.items():
                jpg_key = exts.get("jpg") or exts.get("jpeg") or exts.get("png")
                txt_key = exts.get("txt") or exts.get("caption")
                if not jpg_key or not txt_key:
                    continue
                jpg = tar.extractfile(members[jpg_key]).read()
                txt = tar.extractfile(members[txt_key]).read().decode(
                    "utf-8", errors="replace"
                ).strip()
                yield stem, jpg, txt
    except Exception as e:
        print(f"Warning: {shard_path}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

_SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_SIGLIP_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _decode_jpg(jpg_bytes: bytes, tj=None) -> np.ndarray:
    """Decode JPEG/PNG bytes to HWC uint8 RGB. Reuse TurboJPEG instance if provided."""
    if tj is not None:
        from turbojpeg import TJPF_RGB
        return tj.decode(jpg_bytes, pixel_format=TJPF_RGB)
    try:
        from turbojpeg import TurboJPEG, TJPF_RGB
        return TurboJPEG().decode(jpg_bytes, pixel_format=TJPF_RGB)
    except ImportError:
        from PIL import Image as PilImage
        return np.array(PilImage.open(io.BytesIO(jpg_bytes)).convert("RGB"), dtype=np.uint8)


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    from PIL import Image as PilImage
    return np.array(
        PilImage.fromarray(img).resize((size, size), PilImage.LANCZOS), dtype=np.uint8
    )


def _preprocess_vae(jpg_bytes: bytes, image_size: int, tj=None) -> np.ndarray:
    """Returns float32 [1, 3, H, W] in [-1, 1]."""
    img = _resize(_decode_jpg(jpg_bytes, tj), image_size)
    img_f = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
    return img_f[np.newaxis]


def _preprocess_siglip(jpg_bytes: bytes, tj=None) -> np.ndarray:
    """Returns float32 [1, 3, 384, 384] normalised for SigLIP."""
    img = _resize(_decode_jpg(jpg_bytes, tj), 384)
    img_f = (img.astype(np.float32) / 255.0 - _SIGLIP_MEAN) / _SIGLIP_STD
    return img_f.transpose(2, 0, 1)[np.newaxis]


# ---------------------------------------------------------------------------
# Per-worker model state (populated once via Pool initializer)
# ---------------------------------------------------------------------------

_W: dict = {}


def _load_flux_vae_only(flux_model_path: str):
    """Load only the Flux2 VAE weights, skipping the 4B transformer and text encoder.

    Flux2Klein.__init__ loads all three components (VAE + transformer + text_encoder),
    consuming ~16 GB of Metal memory that is never used here. This function loads only
    the ~500 MB VAE safetensors, freeing that memory for compute buffers.
    """
    import mlx.core as mx
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.common.weights.loading.weight_applier import WeightApplier

    vae_comp = next(c for c in Flux2KleinWeightDefinition.get_components() if c.name == "vae")

    class _VaeOnly:
        @staticmethod
        def get_components():
            return [vae_comp]
        @staticmethod
        def get_download_patterns():
            return ["vae/*.safetensors", "vae/*.json"]

    weights = WeightLoader.load(weight_definition=_VaeOnly, model_path=flux_model_path)
    vae = Flux2VAE()
    WeightApplier.apply_and_quantize_single(
        weights=weights, model=vae, component=vae_comp, quantize_arg=None,
    )
    vae.eval()
    mx.eval(vae.parameters())
    return vae


def _worker_init(qwen3_model_path: str, flux_model_path: str,
                 enable_siglip: bool, image_size: int) -> None:
    """Load all models once per worker process."""
    global _W
    import mlx.core as mx  # noqa: F401 (ensures Metal context is initialised)

    from mlx_lm import load as mlx_lm_load
    model, tokenizer = mlx_lm_load(qwen3_model_path)
    model.eval()
    _W["qwen3"] = model
    _W["tokenizer"] = tokenizer

    vae = _load_flux_vae_only(flux_model_path)
    _W["vae"] = vae

    if enable_siglip:
        try:
            from mlx_vlm import load as vlm_load
            siglip_model, _ = vlm_load("google/siglip-so400m-patch14-384")
            siglip_model.eval()
            _W["siglip"] = (True, siglip_model)
        except Exception:
            from transformers import AutoModel
            hf_model = AutoModel.from_pretrained(
                "google/siglip-so400m-patch14-384"
            ).vision_model.eval()
            _W["siglip"] = (False, hf_model)
    else:
        _W["siglip"] = None

    try:
        from turbojpeg import TurboJPEG
        _W["tj"] = TurboJPEG()
    except ImportError:
        _W["tj"] = None

    _W["image_size"] = image_size


# ---------------------------------------------------------------------------
# Encode helpers (operate on records already filtered to missing-output only)
# ---------------------------------------------------------------------------

def _qwen3_hidden_states(model, tokens, target=(9, 18, 27)):
    """
    Manual forward pass through Qwen3 that captures intermediate layer outputs.

    mlx_lm 0.31+ does not expose output_hidden_states in Model.__call__.
    We replicate the inner Qwen3Model forward pass (embed → layers → collect).
    """
    import mlx.core as mx
    from mlx_lm.models.qwen3 import create_attention_mask
    inner = model.model
    h = inner.embed_tokens(tokens)
    cache = [None] * len(inner.layers)
    mask = create_attention_mask(h, cache[0])
    collected = {}
    target_set = set(target)
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        h = layer(h, mask, c)
        if i in target_set:
            collected[i] = h
    # Evaluate the full forward pass so MLX can free the unevaluated graph
    # nodes for layers after the last collected layer (28-35). Without this,
    # those pending nodes accumulate across batches and cause Metal buffer
    # pressure that grows with each shard (~625 batches/shard × 36 layers).
    mx.eval(h)
    return collected


def _encode_qwen3(records: list, out_dir: str, batch_size: int) -> int:
    """
    Encode captions through Qwen3, save 4-bit quantised embeddings.
    records: list of (rec_id, jpg_bytes, caption)
    Returns number successfully saved.
    """
    import mlx.core as mx
    model = _W["qwen3"]
    tokenizer = _W["tokenizer"]
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)

    tokenized = []
    for rec_id, _, caption in records:
        try:
            chat = [{"role": "user", "content": caption}]
            text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer.encode(text)
            tokenized.append((rec_id, caption, ids))
        except Exception as e:
            print(f"  Skipping {rec_id} (tokenize): {e}", file=sys.stderr)
    tokenized.sort(key=lambda x: len(x[2]))

    import time as _time
    total_seq = 0
    max_seq = 0
    t_q_start = _time.time()
    written = 0
    for i in range(0, len(tokenized), batch_size):
        batch = tokenized[i:i + batch_size]
        seq_lens = [len(ids) for _, _, ids in batch]
        max_len = max(seq_lens)
        total_seq += max_len
        max_seq = max(max_seq, max_len)
        padded = mx.array(np.array([
            ids + [pad_id] * (max_len - len(ids)) for _, _, ids in batch
        ]))
        batch_idx = i // batch_size
        t_b = _time.time()
        try:
            h = _qwen3_hidden_states(model, padded)
            mx.eval(h[9], h[18], h[27])
            h9_np  = np.array(h[9].astype(mx.float32))
            h18_np = np.array(h[18].astype(mx.float32))
            h27_np = np.array(h[27].astype(mx.float32))
            t_b_end = _time.time()
            if batch_idx < 3 or batch_idx % 100 == 0:
                print(f"  [qwen3 batch {batch_idx}] seq={max_len} dt={t_b_end-t_b:.2f}s",
                      file=sys.stderr, flush=True)
            for j, (rec_id, _, _) in enumerate(batch):
                sl = seq_lens[j]
                emb = np.concatenate(
                    [h9_np[j, :sl], h18_np[j, :sl], h27_np[j, :sl]], axis=-1
                )
                q, scale = _quantize_4bit(emb)
                np.savez(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
                written += 1
        except Exception as e:
            print(f"  Qwen3 batch {i // batch_size} failed ({e}), retrying single",
                  file=sys.stderr)
            _batch_skipped = 0
            for rec_id, caption, ids in batch:
                try:
                    sl = len(ids)
                    h = _qwen3_hidden_states(model, mx.array([ids]))
                    mx.eval(h[9], h[18], h[27])
                    emb = np.concatenate([
                        np.array(h[9][0, :sl].astype(mx.float32)),
                        np.array(h[18][0, :sl].astype(mx.float32)),
                        np.array(h[27][0, :sl].astype(mx.float32)),
                    ], axis=-1)
                    q, scale = _quantize_4bit(emb)
                    np.savez(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
                    written += 1
                except Exception as e2:
                    print(f"  Skipping {rec_id}: {e2}", file=sys.stderr)
                    _batch_skipped += 1
            if _batch_skipped:
                print(f"  Qwen3 batch {i // batch_size}: {_batch_skipped}/{len(batch)} records skipped after single-record retry",
                      file=sys.stderr, flush=True)
    n_batches = len(tokenized) // batch_size + 1
    elapsed = _time.time() - t_q_start
    avg_dt = elapsed / max(n_batches, 1)
    avg_seq = total_seq // max(n_batches, 1)
    print(f"  [qwen3 summary] {n_batches} batches  avg_seq={avg_seq}  max_seq={max_seq}"
          f"  avg_dt={avg_dt:.2f}s/batch  total={elapsed:.1f}s", file=sys.stderr, flush=True)
    return written


def _encode_with_prefetch(records: list, out_dir: str, batch_size: int,
                          preprocess_fn, gpu_encode_fn) -> int:
    """
    Generic image encode with 1-ahead prefetch.

    preprocess_fn(jpg_bytes, tj) → float32 array | None on error
    gpu_encode_fn(batch_ids, batch_imgs, out_dir) → int (count saved)

    Phase: batch decode across P-cores while GPU encodes previous batch.
    """
    raw = [(rec_id, jpg) for rec_id, jpg, _ in records]
    if not raw:
        return 0

    tj = _W["tj"]
    batches = [raw[i:i + batch_size] for i in range(0, len(raw), batch_size)]

    def _decode_batch(items):
        def _one(item):
            rec_id, jpg_bytes = item
            try:
                return rec_id, preprocess_fn(jpg_bytes, tj)
            except Exception as e:
                print(f"  Skipping {rec_id}: {e}", file=sys.stderr)
                return rec_id, None
        n = min(_PERF_CORES, len(items))
        with ThreadPoolExecutor(max_workers=n) as pool:
            return list(pool.map(_one, items))

    written = 0
    prefetch_q: deque = deque()
    with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
        if batches:
            prefetch_q.append(prefetch_pool.submit(_decode_batch, batches[0]))

        for idx in range(len(batches)):
            preprocessed = prefetch_q.popleft().result()
            if idx + 1 < len(batches):
                prefetch_q.append(prefetch_pool.submit(_decode_batch, batches[idx + 1]))

            batch_ids  = [r for r, img in preprocessed if img is not None]
            batch_imgs = [img for r, img in preprocessed if img is not None]
            if batch_imgs:
                written += gpu_encode_fn(batch_ids, batch_imgs, out_dir)
            # Flush Metal buffer free-list every 32 batches to limit pool growth.
            if idx > 0 and idx % 32 == 0:
                import mlx.core as mx
                mx.clear_cache()

    return written


def _vae_gpu_encode(batch_ids, batch_imgs, out_dir) -> int:
    import mlx.core as mx
    import time as _time
    vae = _W["vae"]
    _n = _W.get("_vae_batch_n", 0)
    _W["_vae_batch_n"] = _n + 1
    try:
        stacked = np.concatenate(batch_imgs, axis=0)
        _t0 = _time.time()
        latents = vae.encode(mx.array(stacked))
        mx.eval(latents)
        _dt = _time.time() - _t0
        if _n < 3 or _n % 50 == 0:
            print(f"  [vae batch {_n}] n={len(batch_ids)} dt={_dt:.2f}s ({_dt/len(batch_ids):.3f}s/img)",
                  file=sys.stderr, flush=True)
        latents_np = np.array(latents.astype(mx.float32))
        for k, rec_id in enumerate(batch_ids):
            q, scale = _quantize_int8(latents_np[k])
            np.savez(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
        return len(batch_ids)
    except Exception as e:
        print(f"  VAE batch failed ({e}), retrying single", file=sys.stderr)
        saved = 0
        for rec_id, img_np in zip(batch_ids, batch_imgs):
            try:
                latent = vae.encode(mx.array(img_np))
                mx.eval(latent)
                q, scale = _quantize_int8(np.array(latent[0].astype(mx.float32)))
                np.savez(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
                saved += 1
            except Exception as e2:
                print(f"  Skipping {rec_id}: {e2}", file=sys.stderr)
        return saved


def _siglip_gpu_encode(batch_ids, batch_imgs, out_dir) -> int:
    import mlx.core as mx
    _use_mlx_vlm, model_obj = _W["siglip"]
    try:
        stacked = np.concatenate(batch_imgs, axis=0)
        if _use_mlx_vlm:
            feats = model_obj.vision_model(mx.array(stacked))
            mx.eval(feats)
            feats_np = np.array(feats)
        else:
            import torch
            with torch.no_grad():
                out = model_obj(pixel_values=torch.from_numpy(stacked))
            feats_np = out.last_hidden_state.float().numpy()
        for k, rec_id in enumerate(batch_ids):
            q, scale = _quantize_4bit(feats_np[k].astype(np.float32))
            np.savez(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
        return len(batch_ids)
    except Exception as e:
        print(f"  SigLIP batch failed ({e}), retrying single", file=sys.stderr)
        saved = 0
        for rec_id, img_np in zip(batch_ids, batch_imgs):
            try:
                if _use_mlx_vlm:
                    feats = model_obj.vision_model(mx.array(img_np))
                    mx.eval(feats)
                    feat_np = np.array(feats[0])
                else:
                    import torch
                    with torch.no_grad():
                        out = model_obj(pixel_values=torch.from_numpy(img_np))
                    feat_np = out.last_hidden_state[0].float().numpy()
                q, scale = _quantize_4bit(feat_np.astype(np.float32))
                np.savez(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
                saved += 1
            except Exception as e2:
                print(f"  Skipping {rec_id}: {e2}", file=sys.stderr)
        return saved


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def process_shard(args) -> dict:
    """
    Worker: open each shard once and compute all requested outputs.

    Phase 1  — sequential tar read → list of (rec_id, jpg_bytes, caption)
    Phase 2a — Qwen3:  tokenise+sort all pending captions, batched forward
    Phase 2b — VAE:    1-ahead image decode while GPU encodes
    Phase 2c — SigLIP: 1-ahead image decode while GPU encodes (optional)
    """
    shard_path, qwen3_out, vae_out, siglip_out, \
        qwen3_batch, vae_batch, siglip_batch = args

    try:
        return _process_shard_inner(
            shard_path, qwen3_out, vae_out, siglip_out,
            qwen3_batch, vae_batch, siglip_batch,
        )
    except Exception as e:
        print(f"Error processing {os.path.basename(shard_path)}: {e}", file=sys.stderr)
        return {"shard": shard_path, "wq": 0, "wv": 0, "ws": 0, "error": True}


def _process_shard_inner(shard_path, qwen3_out, vae_out, siglip_out,
                          qwen3_batch, vae_batch, siglip_batch) -> dict:
    for d in filter(None, [qwen3_out, vae_out, siglip_out]):
        os.makedirs(d, exist_ok=True)

    import time as _time
    import mlx.core as _mx_gc
    # Phase 1: read tar once — partition records by which outputs are still needed
    t1 = _time.time()
    wq = wv = ws = 0          # already-written counts (resume)
    pending_q: list = []      # (rec_id, jpg_bytes, caption) needing Qwen3
    pending_v: list = []      # needing VAE
    pending_s: list = []      # needing SigLIP

    def _npz_valid(path: str) -> bool:
        """Return True iff path exists and is non-empty (guards against partial writes)."""
        try:
            return os.path.getsize(path) > 0
        except OSError:
            return False

    for rec_id, jpg_bytes, caption in iter_shard(shard_path):
        need_q = bool(qwen3_out) and not _npz_valid(
            os.path.join(qwen3_out, f"{rec_id}.npz"))
        need_v = bool(vae_out)   and not _npz_valid(
            os.path.join(vae_out, f"{rec_id}.npz"))
        need_s = bool(siglip_out) and not _npz_valid(
            os.path.join(siglip_out, f"{rec_id}.npz"))

        if not need_q and qwen3_out:
            wq += 1
        if not need_v and vae_out:
            wv += 1
        if not need_s and siglip_out:
            ws += 1

        rec = (rec_id, jpg_bytes, caption)
        if need_q:
            pending_q.append(rec)
        if need_v:
            pending_v.append(rec)
        if need_s:
            pending_s.append(rec)

    t2 = _time.time()
    shard_name = os.path.basename(shard_path)
    print(f"  [{shard_name}] phase1={t2-t1:.1f}s  pending_q={len(pending_q)}  pending_v={len(pending_v)}",
          file=sys.stderr, flush=True)

    # Phase 2a: Qwen3
    if pending_q:
        wq += _encode_qwen3(pending_q, qwen3_out, qwen3_batch)
    t3 = _time.time()
    print(f"  [{shard_name}] qwen3={t3-t2:.1f}s", file=sys.stderr, flush=True)
    # Clear Metal free-list between Qwen3 and VAE phases to avoid buffer
    # fragmentation from Qwen3 attention masks interfering with large VAE convs.
    _mx_gc.clear_cache()

    # Phase 2b: VAE (1-ahead prefetch)
    if pending_v:
        _W["_vae_batch_n"] = 0
        image_size = _W["image_size"]
        wv += _encode_with_prefetch(
            pending_v, vae_out, vae_batch,
            lambda jpg, tj: _preprocess_vae(jpg, image_size, tj),
            _vae_gpu_encode,
        )

    # Phase 2c: SigLIP (1-ahead prefetch, optional)
    if pending_s and _W.get("siglip") is not None:
        ws += _encode_with_prefetch(
            pending_s, siglip_out, siglip_batch,
            _preprocess_siglip,
            _siglip_gpu_encode,
        )

    t4 = _time.time()
    print(f"  [{shard_name}] vae={t4-t3:.1f}s  total={t4-t1:.1f}s", file=sys.stderr, flush=True)

    # Release Python objects and Metal buffer cache between shards.
    import gc
    import mlx.core as mx
    gc.collect()
    mx.clear_cache()

    return {
        "shard": shard_path,
        "wq": wq, "wv": wv, "ws": ws,
        "error": False,
        "skipped_q": len(pending_q) - wq,
        "skipped_v": len(pending_v) - wv,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified single-pass precompute: Qwen3 + VAE [+ SigLIP]"
    )
    parser.add_argument("--shards", required=True,
                        help="Directory containing .tar shards")
    parser.add_argument("--qwen3-output", default="data/precomputed/qwen3",
                        help="Output dir for Qwen3 .npz files")
    parser.add_argument("--vae-output", default="data/precomputed/vae",
                        help="Output dir for VAE .npz files")
    parser.add_argument("--siglip-output", default="data/precomputed/siglip",
                        help="Output dir for SigLIP .npz files (only used with --siglip)")
    parser.add_argument("--siglip", action="store_true",
                        help="Also compute SigLIP features (~420 GB)")
    parser.add_argument("--qwen3-model", default="Qwen/Qwen3-4B",
                        help="Qwen3 model ID or local path")
    # Auto-detect local Flux Klein model; fall back to HF ID if not found locally.
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_dir   = os.path.dirname(os.path.dirname(_script_dir))
    _flux_default = next(
        (p for p in [
            os.path.join(_repo_dir, "flux-klein-model"),
            os.path.join(_repo_dir, "flux-klein-4b"),
            os.path.join(_repo_dir, "flux-klein-4b-base"),
        ] if os.path.isdir(os.path.join(p, "vae"))),
        "black-forest-labs/FLUX.2-Klein",
    )
    parser.add_argument("--flux-model", default=_flux_default,
                        help="Flux model local path or HF repo ID (VAE is extracted from it)")
    parser.add_argument("--image-size", type=int, default=512,
                        help="Image size for VAE (default 512; SigLIP always 384)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel processes (default 1; GPU-bound)")
    parser.add_argument("--qwen3-batch", type=int, default=8,
                        help="Captions per Qwen3 forward pass (default 8)")
    parser.add_argument("--vae-batch", type=int, default=16,
                        help="Images per VAE encode call (default 16)")
    parser.add_argument("--siglip-batch", type=int, default=8,
                        help="Images per SigLIP forward pass (default 8)")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Randomly select N shards from the pool instead of all "
                             "(use with --scale to limit precompute to training budget)")
    parser.add_argument("--new-shards-first", type=int, default=0,
                        help="Reserve this many slots for shards not yet in any precomputed "
                             "output dir, filling remaining slots from the full pool. "
                             "Use for chunks 2+ to prioritise newly downloaded data. "
                             "Set to 0 (default) for uniform random selection.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for shard selection (default 1; use chunk number "
                             "for reproducible but distinct selections per chunk)")
    parser.add_argument("--match-dir", default=None,
                        help="Run only on shards already covered by this precomputed output "
                             "directory. Shard coverage is detected from npz filenames "
                             "(format: SHARDID_RECID.npz). Use to align siglip with an "
                             "existing qwen3 or vae run instead of a new random selection.")
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(args.shards, "*.tar")))
    if not shards:
        print(f"No .tar files in {args.shards} — nothing to precompute.", flush=True)
        sys.exit(0)

    if args.match_dir is not None:
        # Select only shards that have at least one record in the reference dir.
        # Record files are named SHARDID_RECID.npz — the prefix before the first
        # underscore is the shard stem (e.g. "000014" → "000014.tar").
        covered_ids: set[str] = set()
        for fname in os.listdir(args.match_dir):
            if fname.endswith(".npz"):
                covered_ids.add(fname.split("_")[0])
        before = len(shards)
        shards = [s for s in shards if os.path.splitext(os.path.basename(s))[0] in covered_ids]
        print(f"  --match-dir: {len(shards)} shards matched from {before} available  "
              f"(reference: {args.match_dir})")
        if not shards:
            print("No shards matched — check that --match-dir contains npz files.", file=sys.stderr)
            sys.exit(1)
    elif args.max_shards is not None and args.max_shards < len(shards):
        import random as _random
        _rng = _random.Random(args.seed)
        n_total = len(shards)

        if args.new_shards_first > 0:
            # Identify shards that have no precomputed output yet (new data).
            # A shard is "new" if no .npz exists for it in any output directory.
            out_dirs = [d for d in [args.qwen3_output, args.vae_output] if d]
            def _has_output(shard_path: str) -> bool:
                stem = os.path.splitext(os.path.basename(shard_path))[0]
                return any(
                    os.path.exists(os.path.join(d, f"{stem}.npz")) or
                    # per-record npz files use record IDs not shard stem; check dir non-empty
                    (os.path.isdir(d) and bool(os.listdir(d)))
                    for d in out_dirs
                )
            new_shards = [s for s in shards if not _has_output(s)]
            old_shards = [s for s in shards if _has_output(s)]
            n_new = min(args.new_shards_first, len(new_shards), args.max_shards)
            n_old = args.max_shards - n_new
            selected  = _rng.sample(new_shards, n_new) if n_new > 0 else []
            selected += _rng.sample(old_shards, min(n_old, len(old_shards))) if n_old > 0 else []
            print(f"  --max-shards: {len(selected)} selected "
                  f"({n_new} new + {len(selected)-n_new} old) of {n_total} available  "
                  f"[seed={args.seed}]")
        else:
            selected = _rng.sample(shards, args.max_shards)
            print(f"  --max-shards: selected {len(selected)} of {n_total} available shards  "
                  f"[seed={args.seed}]")

        shards = sorted(selected)  # stable order for reproducible ETA display

    siglip_out = args.siglip_output if args.siglip else None

    for d in filter(None, [args.qwen3_output, args.vae_output, siglip_out]):
        os.makedirs(d, exist_ok=True)

    qwen3_gb  = len(shards) * 0.46
    latent_h  = args.image_size // 8
    vae_gb    = len(shards) * 5000 * 32 * latent_h * latent_h / 1e9
    siglip_gb = len(shards) * 5000 * (729 * 576 + 729 * 2) / 1e9

    print(f"Unified precompute: {len(shards)} shards")
    print(f"  Qwen3 model:  {args.qwen3_model}  → {args.qwen3_output}  (~{qwen3_gb:.0f} GB)")
    print(f"  Flux model:   {args.flux_model}  → {args.vae_output}  (~{vae_gb:.0f} GB)")
    if siglip_out:
        print(f"  SigLIP:       google/siglip-so400m-patch14-384  → {siglip_out}  (~{siglip_gb:.0f} GB)")
    print(f"  Workers: {args.workers}   "
          f"Qwen3 batch: {args.qwen3_batch}   "
          f"VAE batch: {args.vae_batch}"
          + (f"   SigLIP batch: {args.siglip_batch}" if siglip_out else ""))
    print()

    work_items = [
        (s, args.qwen3_output, args.vae_output, siglip_out,
         args.qwen3_batch, args.vae_batch, args.siglip_batch)
        for s in shards
    ]

    import time as _time
    from collections import deque as _deque
    results = []
    t_start = t_last = _time.time()
    # Keep recent shard times for windowed ETA (last 10 shards).
    # Avoids the harmonic-mean trap where fast cache-hit shards mask growing compute time.
    recent_dts: _deque = _deque(maxlen=10)

    with multiprocessing.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.qwen3_model, args.flux_model, args.siglip, args.image_size),
    ) as pool:
        for done, result in enumerate(
            pool.imap_unordered(process_shard, work_items, chunksize=1), 1
        ):
            results.append(result)
            t_now = _time.time()
            dt = t_now - t_last
            t_last = t_now
            if dt > 0:
                recent_dts.append(dt)
            avg_dt = sum(recent_dts) / len(recent_dts) if recent_dts else dt
            eta = (len(work_items) - done) * avg_dt

            errs = sum(1 for r in results if r["error"])
            ws   = sum(r["ws"] for r in results)
            pct  = 100 * done // len(work_items)
            err_str    = f"  errors={errs}" if errs else ""
            siglip_str = f" +siglip" if siglip_out and ws > 0 else ""
            print(
                f"  [{done}/{len(work_items)}] {pct}%{siglip_str}"
                f"{err_str}  {dt:.1f} s/shard  ETA {int(eta//3600)}h {int((eta%3600)//60)}m",
                flush=True,
            )

    wq = sum(r["wq"] for r in results)
    wv = sum(r["wv"] for r in results)
    ws = sum(r["ws"] for r in results)
    sq = sum(r.get("skipped_q", 0) for r in results)
    sv = sum(r.get("skipped_v", 0) for r in results)
    errs = sum(1 for r in results if r["error"])
    elapsed = (_time.time() - t_start) / 3600
    print(f"\nDone in {elapsed:.1f}h.")
    print(f"  Qwen3:  {wq:,} embeddings  → {args.qwen3_output}/" + (f"  ({sq:,} skipped after retry)" if sq else ""))
    print(f"  VAE:    {wv:,} latents      → {args.vae_output}/" + (f"  ({sv:,} skipped after retry)" if sv else ""))
    if siglip_out:
        print(f"  SigLIP: {ws:,} features    → {siglip_out}/")
    if errs:
        print(f"  {errs} shards had errors (check stderr)")
    if sq or sv:
        print(f"  WARNING: {sq + sv:,} records were skipped after single-record retry — check stderr for details")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
