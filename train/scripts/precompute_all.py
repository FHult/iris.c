"""
train/scripts/precompute_all.py — Unified single-pass precompute.

Reads each shard once and writes all requested outputs:
  - Qwen3 4-bit quantised text embeddings  → {qwen3_output}/{id}.npz
  - VAE int8 quantised latents             → {vae_output}/{id}.npz
  - SigLIP 4-bit quantised features        → {siglip_output}/{id}.npz  (--siglip)

Saves ~10-12h vs running the three scripts sequentially (~22h total):
each shard's tar is opened once instead of twice or three times, and all
models are loaded a single time at startup.

Within each shard:
  Phase 1  — sequential tar read → list of (rec_id, jpg_bytes, caption)
  Phase 2a — Qwen3 pass: tokenise+sort all captions, batched forward
  Phase 2b — VAE pass:   1-ahead image decode prefetch while GPU encodes
  Phase 2c — SigLIP pass (optional): same prefetch pattern as VAE

IO prefetch: the next shard's tar is read in a background thread while the
GPU processes the current shard (Phase 2a/2b), hiding most of the tar I/O
latency.  Output directories are scanned once per shard (set lookup O(1))
rather than one stat() call per record per directory.

Reference: plans/ip-adapter-training.md §2.7
"""

import argparse
import glob
import io
import json
import multiprocessing
import os
import queue as _queue_mod
import sys
import tarfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import write_heartbeat

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


def _quantize_int8_batch(arr: np.ndarray):
    """Batch per-channel absmax int8 quantisation for VAE latents [B, 32, H, W]."""
    scale = np.abs(arr).max(axis=(2, 3), keepdims=True) / 127.0
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
# File helpers
# ---------------------------------------------------------------------------

def _scan_existing(out_dir: str | None) -> set[str]:
    """One-time directory scan; returns rec_id stems of all .npz files present.

    Called once per shard at Phase 1 start.  Replaces per-record os.stat()
    calls (which were ~15,000 syscalls/shard with 3 outputs) with a single
    directory listing followed by O(1) set lookups.

    Atomic writes (see _save_npz_atomic) ensure every listed file is complete,
    so no size check is needed here.
    """
    if not out_dir or not os.path.isdir(out_dir):
        return set()
    try:
        return {f[:-4] for f in os.listdir(out_dir) if f.endswith(".npz")}
    except OSError:
        return set()


def _save_npz_atomic(path: str, **arrays) -> None:
    """Write arrays to a .tmp file then atomically rename to path.

    Prevents partial writes from being mistaken for valid outputs on resume —
    a crash mid-write leaves a .tmp file (ignored by _scan_existing) rather
    than a truncated .npz that would pass an existence check.

    np.savez appends ".npz" to the filename if it doesn't already end with it,
    so the actual file created is path + ".tmp.npz", not path + ".tmp".
    """
    tmp_stem = path + ".tmp"
    tmp_actual = tmp_stem + ".npz"  # what np.savez actually creates
    try:
        np.savez(tmp_stem, **arrays)
        os.replace(tmp_actual, path)
    except Exception:
        try:
            os.unlink(tmp_actual)
        except OSError:
            pass
        raise


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
# Per-process model state (populated once via _worker_init)
# ---------------------------------------------------------------------------

_W: dict = {}
_progress_q = None
_BLOCKLIST: frozenset = frozenset()  # record IDs to skip; set by main() from --blocklist


def _report_progress(shard: str, phase: str, done: int, total: int) -> None:
    """Non-blocking push to the progress queue (no-op if queue absent)."""
    q = _progress_q
    if q is not None:
        try:
            q.put_nowait({"shard": shard, "phase": phase, "done": done, "total": total})
        except Exception:
            pass


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
                 enable_siglip: bool, image_size: int, progress_q,
                 load_qwen3: bool = True, load_vae: bool = True) -> None:
    """Load all models once.

    load_qwen3/load_vae: set False when the caller has confirmed all records
    are already cached, avoiding loading 8+ GB of model weights that would be
    immediately discarded.  This prevents jetsam killing the subsequent training
    process when training starts shortly after a skip-only precompute run.
    """
    global _W, _progress_q
    _progress_q = progress_q
    import mlx.core as mx  # noqa: F401 (ensures Metal context is initialised)

    # Cap MLX GPU memory to 14 GB.  Qwen3-4B (~8 GB) + VAE (~0.5 GB) + SigLIP (~1.7 GB)
    # fit comfortably within this limit.  Keeping the cap lower (vs 20 GB) reduces
    # memory pressure on 32 GB systems and leaves more headroom for the OS and other
    # processes running concurrently with precompute.
    try:
        mx.set_memory_limit(14 * 1024 ** 3)
    except AttributeError:
        pass  # MLX version does not expose set_memory_limit; safe to ignore

    if load_qwen3:
        from mlx_lm import load as mlx_lm_load
        model, tokenizer = mlx_lm_load(qwen3_model_path)
        model.eval()
        _W["qwen3"] = model
        _W["tokenizer"] = tokenizer
    else:
        _W["qwen3"] = None
        _W["tokenizer"] = None

    if load_vae:
        vae = _load_flux_vae_only(flux_model_path)
        _W["vae"] = vae
    else:
        _W["vae"] = None

    if enable_siglip:
        try:
            from mlx_vlm import load as vlm_load
            siglip_model, _ = vlm_load("google/siglip-so400m-patch14-384")
            siglip_model.eval()
            _W["siglip"] = (True, siglip_model, "mlx")
        except Exception:
            import torch
            from transformers import AutoModel
            hf_model = AutoModel.from_pretrained(
                "google/siglip-so400m-patch14-384"
            ).vision_model.eval()
            _siglip_device = "mps" if torch.backends.mps.is_available() else "cpu"
            hf_model = hf_model.to(_siglip_device)
            _W["siglip"] = (False, hf_model, _siglip_device)
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

def _qwen3_hidden_states(model, tokens, target=(8, 17, 26)):
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
    last_target = max(target_set)
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        h = layer(h, mask, c)
        if i in target_set:
            collected[i] = h
        if i == last_target:
            break
    # Evaluate only the layers we actually ran (0..last_target).
    # Breaking early means layers after last_target are never added to the MLX
    # computation graph, so there are no pending nodes to accumulate.
    mx.eval(*collected.values())
    return collected


def _encode_qwen3(records: list, out_dir: str, batch_size: int, shard_name: str = "") -> int:
    """
    Encode captions through Qwen3, save 4-bit quantised embeddings.
    records: list of (rec_id, caption)
    Returns number successfully saved.
    """
    import mlx.core as mx
    model = _W["qwen3"]
    tokenizer = _W["tokenizer"]
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)

    tokenized = []
    for rec_id, caption in records:
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
    n_batches = (len(tokenized) + batch_size - 1) // batch_size
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
        _report_progress(shard_name, "qwen3", batch_idx, n_batches)
        t_b = _time.time()
        try:
            h = _qwen3_hidden_states(model, padded)
            mx.eval(h[8], h[17], h[26])
            h8_np  = np.array(h[8].astype(mx.float32))
            h17_np = np.array(h[17].astype(mx.float32))
            h26_np = np.array(h[26].astype(mx.float32))
            t_b_end = _time.time()
            if batch_idx < 3 or batch_idx % 100 == 0:
                print(f"  [qwen3 batch {batch_idx}] seq={max_len} dt={t_b_end-t_b:.2f}s",
                      file=sys.stderr, flush=True)
            for j, (rec_id, _, _) in enumerate(batch):
                sl = seq_lens[j]
                emb = np.concatenate(
                    [h8_np[j, :sl], h17_np[j, :sl], h26_np[j, :sl]], axis=-1
                )
                q, scale = _quantize_4bit(emb)
                _save_npz_atomic(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
                written += 1
            if batch_idx % 32 == 31:
                mx.clear_cache()
        except Exception as e:
            print(f"  Qwen3 batch {i // batch_size} failed ({e}), retrying single",
                  file=sys.stderr)
            _batch_skipped = 0
            for rec_id, _, ids in batch:
                try:
                    sl = len(ids)
                    h = _qwen3_hidden_states(model, mx.array([ids]))
                    mx.eval(h[8], h[17], h[26])
                    emb = np.concatenate([
                        np.array(h[8][0, :sl].astype(mx.float32)),
                        np.array(h[17][0, :sl].astype(mx.float32)),
                        np.array(h[26][0, :sl].astype(mx.float32)),
                    ], axis=-1)
                    q, scale = _quantize_4bit(emb)
                    _save_npz_atomic(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
                    written += 1
                except Exception as e2:
                    print(f"  Skipping {rec_id}: {e2}", file=sys.stderr)
                    _batch_skipped += 1
            if _batch_skipped:
                print(f"  Qwen3 batch {i // batch_size}: {_batch_skipped}/{len(batch)} records skipped after single-record retry",
                      file=sys.stderr, flush=True)
    mx.clear_cache()
    n_batches = len(tokenized) // batch_size + 1
    elapsed = _time.time() - t_q_start
    avg_dt = elapsed / max(n_batches, 1)
    avg_seq = total_seq // max(n_batches, 1)
    print(f"  [qwen3 summary] {n_batches} batches  avg_seq={avg_seq}  max_seq={max_seq}"
          f"  avg_dt={avg_dt:.2f}s/batch  total={elapsed:.1f}s", file=sys.stderr, flush=True)
    return written


def _encode_with_prefetch(records: list, out_dir: str, batch_size: int,
                          preprocess_fn, gpu_encode_fn,
                          phase: str = "vae", shard_name: str = "") -> int:
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
            _report_progress(shard_name, phase, idx, len(batches))
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
        q_batch, scale_batch = _quantize_int8_batch(latents_np)
        for k, rec_id in enumerate(batch_ids):
            _save_npz_atomic(os.path.join(out_dir, f"{rec_id}.npz"), q=q_batch[k], scale=scale_batch[k])
        return len(batch_ids)
    except Exception as e:
        print(f"  VAE batch failed ({e}), retrying single", file=sys.stderr)
        saved = 0
        for rec_id, img_np in zip(batch_ids, batch_imgs):
            try:
                latent = vae.encode(mx.array(img_np))
                mx.eval(latent)
                q, scale = _quantize_int8(np.array(latent[0].astype(mx.float32)))
                _save_npz_atomic(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
                saved += 1
            except Exception as e2:
                print(f"  Skipping {rec_id}: {e2}", file=sys.stderr)
        return saved


def _siglip_gpu_encode(batch_ids, batch_imgs, out_dir) -> int:
    import mlx.core as mx
    siglip_entry = _W["siglip"]
    _use_mlx_vlm = siglip_entry[0]
    model_obj     = siglip_entry[1]
    _dev          = siglip_entry[2] if len(siglip_entry) > 2 else "cpu"
    try:
        stacked = np.concatenate(batch_imgs, axis=0)
        if _use_mlx_vlm:
            feats = model_obj.vision_model(mx.array(stacked))
            mx.eval(feats)
            feats_np = np.array(feats)
        else:
            import torch
            with torch.no_grad():
                pv = torch.from_numpy(stacked).to(_dev)
                out = model_obj(pixel_values=pv)
            feats_np = out.last_hidden_state.float().cpu().numpy()
        for k, rec_id in enumerate(batch_ids):
            q, scale = _quantize_4bit(feats_np[k].astype(np.float32))
            _save_npz_atomic(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
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
                        pv = torch.from_numpy(img_np).to(_dev)
                        out = model_obj(pixel_values=pv)
                    feat_np = out.last_hidden_state[0].float().cpu().numpy()
                q, scale = _quantize_4bit(feat_np.astype(np.float32))
                _save_npz_atomic(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
                saved += 1
            except Exception as e2:
                print(f"  Skipping {rec_id}: {e2}", file=sys.stderr)
        return saved


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def process_shard(args) -> dict:
    """
    Process one shard: compute all requested outputs.

    Phase 1  — partition records by which outputs are still needed
    Phase 2a — Qwen3:  tokenise+sort all pending captions, batched forward
    Phase 2b — VAE:    1-ahead image decode while GPU encodes
    Phase 2c — SigLIP: 1-ahead image decode while GPU encodes (optional)
    """
    shard_path, qwen3_out, vae_out, siglip_out, \
        qwen3_batch, vae_batch, siglip_batch, pre_records = args

    try:
        return _process_shard_inner(
            shard_path, qwen3_out, vae_out, siglip_out,
            qwen3_batch, vae_batch, siglip_batch,
            pre_records=pre_records,
        )
    except Exception as e:
        print(f"Error processing {os.path.basename(shard_path)}: {e}", file=sys.stderr)
        return {"shard": shard_path, "wq": 0, "wv": 0, "ws": 0, "error": True}


def _process_shard_inner(shard_path, qwen3_out, vae_out, siglip_out,
                          qwen3_batch, vae_batch, siglip_batch,
                          pre_records=None) -> dict:
    for d in filter(None, [qwen3_out, vae_out, siglip_out]):
        os.makedirs(d, exist_ok=True)

    import time as _time
    import mlx.core as _mx_gc

    # Phase 1: scan output dirs once (O(1) set lookup per record vs per-record stat).
    # pre_records is a list pre-read by the IO prefetch thread during the previous
    # shard's GPU phases; falls back to reading from disk if not provided.
    t1 = _time.time()
    existing_q = _scan_existing(qwen3_out)
    existing_v = _scan_existing(vae_out)
    existing_s = _scan_existing(siglip_out)

    wq = wv = ws = 0
    pending_q: list = []
    pending_v: list = []
    pending_s: list = []

    records_iter = pre_records if pre_records is not None else iter_shard(shard_path)
    for rec_id, jpg_bytes, caption in records_iter:
        if rec_id in _BLOCKLIST:
            continue
        need_q = bool(qwen3_out)   and rec_id not in existing_q
        need_v = bool(vae_out)     and rec_id not in existing_v
        need_s = bool(siglip_out)  and rec_id not in existing_s

        if not need_q and qwen3_out:
            wq += 1
        if not need_v and vae_out:
            wv += 1
        if not need_s and siglip_out:
            ws += 1

        if need_q:
            pending_q.append((rec_id, caption))
        if need_v:
            pending_v.append((rec_id, jpg_bytes, caption))
        if need_s:
            pending_s.append((rec_id, jpg_bytes, caption))

    t2 = _time.time()
    shard_name = os.path.basename(shard_path)
    print(f"  [{shard_name}] phase1={t2-t1:.1f}s  pending_q={len(pending_q)}  pending_v={len(pending_v)}",
          file=sys.stderr, flush=True)

    # Phase 2a: Qwen3
    n_pending_q = len(pending_q)
    wq_before = wq
    if pending_q:
        _report_progress(shard_name, "qwen3", 0, n_pending_q)
        wq += _encode_qwen3(pending_q, qwen3_out, qwen3_batch, shard_name=shard_name)
    pending_q.clear()
    t3 = _time.time()
    print(f"  [{shard_name}] qwen3={t3-t2:.1f}s", file=sys.stderr, flush=True)
    # Clear Metal free-list between Qwen3 and VAE phases to avoid buffer
    # fragmentation from Qwen3 attention masks interfering with large VAE convs.
    _mx_gc.clear_cache()

    # Phase 2b: VAE (1-ahead prefetch)
    n_pending_v = len(pending_v)
    wv_before = wv
    if pending_v:
        _W["_vae_batch_n"] = 0
        image_size = _W["image_size"]
        _report_progress(shard_name, "vae", 0, n_pending_v)
        wv += _encode_with_prefetch(
            pending_v, vae_out, vae_batch,
            lambda jpg, tj: _preprocess_vae(jpg, image_size, tj),
            _vae_gpu_encode,
            phase="vae", shard_name=shard_name,
        )
    pending_v.clear()

    # Phase 2c: SigLIP (1-ahead prefetch, optional)
    if pending_s and _W.get("siglip") is not None:
        _report_progress(shard_name, "siglip", 0, len(pending_s))
        ws += _encode_with_prefetch(
            pending_s, siglip_out, siglip_batch,
            _preprocess_siglip,
            _siglip_gpu_encode,
            phase="siglip", shard_name=shard_name,
        )

    t4 = _time.time()
    print(f"  [{shard_name}] vae={t4-t3:.1f}s  total={t4-t1:.1f}s", file=sys.stderr, flush=True)

    # Release Python objects and Metal buffer cache between shards.
    # torch.mps.empty_cache() is required when siglip runs via PyTorch MPS:
    # mx.clear_cache() only frees the MLX Metal allocator; PyTorch has a separate
    # MPS cache that grows across shards and is never returned to the OS otherwise,
    # eventually exhausting swap and causing a kernel panic.
    import gc
    import mlx.core as mx
    gc.collect()
    mx.clear_cache()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    return {
        "shard": shard_path,
        "wq": wq, "wv": wv, "ws": ws,
        "error": False,
        "skipped_q": n_pending_q - (wq - wq_before),
        "skipped_v": n_pending_v - (wv - wv_before),
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
                        help="Ignored (GPU precompute is single-threaded); kept for CLI compat")
    parser.add_argument("--qwen3-batch", type=int, default=16,
                        help="Captions per Qwen3 forward pass (default 16)")
    parser.add_argument("--vae-batch", type=int, default=4,
                        help="Images per VAE encode call (default 4; profiled optimum on M1 Max at 512px)")
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
    parser.add_argument("--chunk", type=int, default=None,
                        help="Pipeline chunk number (for heartbeat naming)")
    parser.add_argument("--ai", action="store_true",
                        help="Emit compact JSON summary to stdout at completion; progress to stderr")
    parser.add_argument("--list-cache", default=None, metavar="PRECOMP_ROOT",
                        help="List versioned cache versions under PRECOMP_ROOT and exit")
    parser.add_argument("--clear-stale", default=None, metavar="PRECOMP_ROOT",
                        help="Delete non-current cache versions under PRECOMP_ROOT and exit")
    parser.add_argument("--blocklist", default=None, metavar="PATH",
                        help="File of record IDs to skip (one per line); produced by clip_dedup find-dups")
    args = parser.parse_args()

    # Cache management shortcuts — these exit immediately without doing any precompute.
    if args.list_cache or args.clear_stale:
        from cache_manager import PrecomputeCache, ENCODERS
        import json as _json
        _precomp_root = Path(args.list_cache or args.clear_stale)
        if args.clear_stale:
            for _enc in ENCODERS:
                _del = PrecomputeCache.clear(_precomp_root, _enc, stale_only=True)
                if _del:
                    print(f"{_enc}: deleted {_del}")
        else:
            for _enc in ENCODERS:
                _vs = PrecomputeCache.list_versions(_precomp_root, _enc)
                if not _vs:
                    print(f"{_enc}: (no versioned cache)")
                    continue
                print(f"{_enc}:")
                for _v in _vs:
                    _tag  = " [current]" if _v["current"] else ""
                    _done = "complete" if _v["complete"] else "incomplete"
                    print(f"  {_v['version']}{_tag}  {_done}  "
                          f"{_v.get('record_count', 0):,} records  "
                          f"{_v.get('config', {})}")
        return

    global _BLOCKLIST
    if args.blocklist:
        _BLOCKLIST = frozenset(Path(args.blocklist).read_text().splitlines())
        print(f"Blocklist: {len(_BLOCKLIST):,} record IDs will be skipped", flush=True)

    # Block manual runs when GPU is already in use by training or the pipeline.
    # Orchestrated runs (PIPELINE_ORCHESTRATED=1) skip this — the orchestrator
    # manages GPU serialisation via GPU_TOKEN and the file lock itself.
    if not os.environ.get("PIPELINE_ORCHESTRATED"):
        import atexit
        from pipeline_lib import (
            gpu_is_free, tmux_window_exists, TMUX_PREP_WIN,
            acquire_gpu_lock, release_gpu_lock, gpu_lock_holder,
        )
        if not gpu_is_free():
            print("ERROR: iris-train is running. GPU is in use by training.", file=sys.stderr)
            sys.exit(1)
        if tmux_window_exists(TMUX_PREP_WIN):
            print("ERROR: iris-prep is running. GPU is in use by the pipeline.", file=sys.stderr)
            sys.exit(1)
        _lock_info = gpu_lock_holder()
        if _lock_info is not None:
            print(
                f"ERROR: GPU lock held by '{_lock_info.get('label', '?')}' "
                f"(PID {_lock_info.get('pid', '?')}). Exiting.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not acquire_gpu_lock("precompute (manual)"):
            print("ERROR: GPU lock acquire race — try again.", file=sys.stderr)
            sys.exit(1)
        atexit.register(release_gpu_lock)

    if args.workers > 1:
        print(f"  Note: --workers={args.workers} ignored; GPU precompute is single-threaded.")

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
                    os.path.exists(os.path.join(d, f"{stem}.npz"))
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

    # Write incomplete manifests at startup so the orchestrator can detect
    # in-progress runs and the config that generated them.
    _active_caches: dict = {}
    try:
        from cache_manager import PrecomputeCache, get_git_sha
        _git_sha = get_git_sha(Path(os.path.dirname(os.path.abspath(__file__))))
        _cache_configs = {
            "qwen3":  {"qwen3_model": args.qwen3_model, "layers": [9, 18, 27]},
            "vae":    {"flux_model": Path(args.flux_model).name,
                       "image_size": args.image_size},
            "siglip": {"siglip_model": "google/siglip-so400m-patch14-384",
                       "image_size": 384},
        }
        for _enc, _out in [("qwen3", args.qwen3_output),
                           ("vae",   args.vae_output),
                           ("siglip", siglip_out)]:
            if not _out:
                continue
            _enc_dir = Path(_out).parent
            _cache = PrecomputeCache(_enc_dir, _enc, _cache_configs[_enc], _git_sha)
            # Only write a manifest if this output dir IS the versioned cache dir
            # (i.e. the path ends with the version hash the cache would produce).
            if Path(_out).name == _cache.version():
                _cache.write_manifest_incomplete()
                _active_caches[_enc] = _cache
    except Exception:
        pass  # manifest writing is best-effort

    qwen3_gb  = len(shards) * 0.46
    latent_h  = args.image_size // 8
    vae_gb    = len(shards) * 5000 * 32 * latent_h * latent_h / 1e9
    siglip_gb = len(shards) * 5000 * (729 * 576 + 729 * 2) / 1e9

    print(f"Unified precompute: {len(shards)} shards")
    print(f"  Qwen3 model:  {args.qwen3_model}  → {args.qwen3_output}  (~{qwen3_gb:.0f} GB)")
    print(f"  Flux model:   {args.flux_model}  → {args.vae_output}  (~{vae_gb:.0f} GB)")
    if siglip_out:
        print(f"  SigLIP:       google/siglip-so400m-patch14-384  → {siglip_out}  (~{siglip_gb:.0f} GB)")
    print(f"  Workers: 1   Qwen3 batch: {args.qwen3_batch}   "
          f"VAE batch: {args.vae_batch}"
          + (f"   SigLIP batch: {args.siglip_batch}" if siglip_out else ""))
    print()

    # Sample the LAST shard to check whether qwen3/vae are already fully cached.
    # Avoids loading 8+ GB of model weights that would be immediately discarded —
    # which caused jetsam to kill the subsequent training process on 32 GB systems
    # when precompute and training started within seconds of each other.
    #
    # We sample the last shard (not the first) because precompute processes shards
    # in sorted order; after a partial run the first N shards are cached but later
    # shards are not.  Sampling shards[0] incorrectly concludes "all done" when only
    # a prefix is cached, causing all remaining shards to be silently skipped.
    _load_qwen3 = True
    _load_vae   = True
    if shards and args.qwen3_output and args.vae_output:
        try:
            import itertools as _itools
            _sample_shard = shards[-1]  # last shard = least likely cached in a partial run
            _sample = [(r, None, None) for r, _, _ in _itools.islice(iter_shard(_sample_shard), 20)]
            _existing_q_sample = _scan_existing(args.qwen3_output)
            _existing_v_sample = _scan_existing(args.vae_output)
            _load_qwen3 = any(r not in _existing_q_sample for r, _, _ in _sample)
            _load_vae   = any(r not in _existing_v_sample for r, _, _ in _sample)
        except Exception:
            pass  # can't sample → load both models to be safe
    if not _load_qwen3:
        print("  Qwen3: sample shard fully cached — skipping model load")
    if not _load_vae:
        print("  VAE:   sample shard fully cached — skipping model load")

    # When a model is skipped, also omit its output path from work_items so
    # _process_shard_inner doesn't attempt encoding with a None model.
    _qwen3_work = args.qwen3_output if _load_qwen3 else None
    _vae_work   = args.vae_output   if _load_vae   else None

    # ── Resume state (PIPELINE-13) ─────────────────────────────────────────
    # A JSON file lists shard basenames that completed with no errors in a
    # previous run.  On restart those shards are skipped entirely, avoiding
    # the tar-open and IO prefetch for already-done work.
    _resume_state_path = None
    _done_shards: set[str] = set()
    _state_dir = args.qwen3_output or args.vae_output or siglip_out
    if _state_dir:
        _resume_state_path = os.path.join(_state_dir, ".precompute_done.json")
        try:
            with open(_resume_state_path) as _rf:
                _done_shards = set(json.load(_rf))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    if _done_shards:
        shards_before = len(shards)
        shards = [s for s in shards if os.path.basename(s) not in _done_shards]
        print(f"  Resume: skipping {shards_before - len(shards)} fully-done shards "
              f"({len(shards)} remaining)", flush=True)

    def _append_done_shard(shard_path: str) -> None:
        if _resume_state_path is None:
            return
        _done_shards.add(os.path.basename(shard_path))
        try:
            with open(_resume_state_path, "w") as _wf:
                json.dump(sorted(_done_shards), _wf)
        except OSError:
            pass
    # ── end resume state ───────────────────────────────────────────────────

    work_items = [
        (s, _qwen3_work, _vae_work, siglip_out,
         args.qwen3_batch, args.vae_batch, args.siglip_batch, None)
        for s in shards
    ]

    import threading as _threading
    import time as _time
    from collections import deque as _deque
    results = []
    t_start = t_last = _time.time()
    # Keep recent shard times for windowed ETA (last 10 shards).
    # Avoids the harmonic-mean trap where fast cache-hit shards mask growing compute time.
    recent_dts: _deque = _deque(maxlen=10)

    # Thread-safe progress queue (no inter-process serialisation needed).
    progress_q = _queue_mod.Queue()

    # Shared state updated by main loop and drained from progress_q; read by heartbeat thread.
    _hb_state = {"done": 0, "total": len(work_items), "pct": 0, "eta_sec": 0,
                 "shard": None, "phase": None, "phase_done": 0, "phase_total": 0}
    _hb_stop  = _threading.Event()

    def _drain_progress_q():
        """Drain all pending worker progress messages into _hb_state (non-blocking)."""
        while True:
            try:
                msg = progress_q.get_nowait()
                _hb_state.update(
                    shard=msg.get("shard"),
                    phase=msg.get("phase"),
                    phase_done=msg.get("done", 0),
                    phase_total=msg.get("total", 0),
                )
            except Exception:
                break

    def _hb_thread():
        while not _hb_stop.wait(60):
            _drain_progress_q()
            s = _hb_state
            phase_str = f"{s['phase']} {s['phase_done']}/{s['phase_total']}" if s["phase"] else None
            write_heartbeat("precompute", args.chunk,
                            done=s["done"], total=s["total"],
                            pct=s["pct"], eta_sec=s["eta_sec"],
                            current_shard=s["shard"], current_phase=phase_str)

    _threading.Thread(target=_hb_thread, daemon=True).start()

    # Load models once in the main process (no subprocess fork / pickle overhead).
    _worker_init(args.qwen3_model, args.flux_model, args.siglip, args.image_size, progress_q,
                 _load_qwen3, _load_vae)

    def _read_shard_records(shard_path: str) -> list:
        """Read all shard records into memory for IO prefetch."""
        return list(iter_shard(shard_path))

    # IO prefetch: submit next shard's tar read to a background thread so it
    # overlaps with the current shard's GPU phases (Qwen3 ~5-10 min, VAE ~5-10 min).
    # Peak extra RAM: one shard's raw JPEGs (~500 MB) held while the current shard
    # processes — well within budget on a 32 GB system.
    with ThreadPoolExecutor(max_workers=1) as io_exec:
        prefetch = io_exec.submit(_read_shard_records, work_items[0][0]) if work_items else None

        for seq_idx, base_item in enumerate(work_items):
            pre_records = prefetch.result() if prefetch is not None else []
            prefetch = None
            if seq_idx + 1 < len(work_items):
                prefetch = io_exec.submit(_read_shard_records, work_items[seq_idx + 1][0])

            result = process_shard(base_item[:7] + (pre_records,))
            if not result.get("error"):
                _append_done_shard(base_item[0])
            done = seq_idx + 1
            results.append(result)
            t_now = _time.time()
            dt = t_now - t_last
            t_last = t_now
            if dt > 0:
                recent_dts.append(dt)
            avg_dt = sum(recent_dts) / len(recent_dts) if recent_dts else dt
            eta = (len(work_items) - done) * avg_dt

            _drain_progress_q()
            errs = sum(1 for r in results if r["error"])
            ws   = sum(r["ws"] for r in results)
            pct  = 100 * done // len(work_items)
            _hb_state.update(done=done, pct=pct, eta_sec=round(eta),
                             shard=None, phase=None, phase_done=0, phase_total=0)
            err_str    = f"  errors={errs}" if errs else ""
            siglip_str = f" +siglip" if siglip_out and ws > 0 else ""
            print(
                f"  [{done}/{len(work_items)}] {pct}%{siglip_str}"
                f"{err_str}  {dt:.1f} s/shard  ETA {int(eta//3600)}h {int((eta%3600)//60)}m",
                flush=True,
            )
            write_heartbeat("precompute", args.chunk,
                            done=done, total=len(work_items), pct=pct,
                            eta_sec=round(eta))

    _hb_stop.set()

    wq = sum(r["wq"] for r in results)
    wv = sum(r["wv"] for r in results)
    ws = sum(r["ws"] for r in results)
    sq = sum(r.get("skipped_q", 0) for r in results)
    sv = sum(r.get("skipped_v", 0) for r in results)
    errs = sum(1 for r in results if r["error"])
    elapsed = (_time.time() - t_start) / 3600
    _summary_out = sys.stderr if args.ai else sys.stdout
    print(f"\nDone in {elapsed:.1f}h.", file=_summary_out)
    print(f"  Qwen3:  {wq:,} embeddings  → {args.qwen3_output}/" + (f"  ({sq:,} skipped after retry)" if sq else ""), file=_summary_out)
    print(f"  VAE:    {wv:,} latents      → {args.vae_output}/" + (f"  ({sv:,} skipped after retry)" if sv else ""), file=_summary_out)
    if siglip_out:
        print(f"  SigLIP: {ws:,} features    → {siglip_out}/", file=_summary_out)
    if errs:
        print(f"  {errs} shards had errors (check stderr)", file=_summary_out)
    if sq or sv:
        print(f"  WARNING: {sq + sv:,} records were skipped after single-record retry — check stderr for details", file=_summary_out)

    # Post-run coverage verification: count .npz files per shard and flag any
    # short shards before the orchestrator marks precompute.done.  A partial
    # shard (e.g. from a mid-shard kill) would otherwise only be caught 12+h
    # later at promotion time, causing a 200+ escalation storm.
    _check_dirs = [d for d in [args.qwen3_output, args.vae_output] if d]
    if _check_dirs and work_items:
        print("\nVerifying per-shard .npz coverage ...",
              file=sys.stderr if args.ai else sys.stdout, flush=True)
        _short: list[str] = []
        for shard_path, *_ in work_items:
            stem = os.path.splitext(os.path.basename(shard_path))[0]
            # Count records in the shard (fast header-only pass).
            try:
                expected = sum(1 for _ in iter_shard(shard_path))
            except Exception:
                continue
            if expected == 0:
                continue
            for out_dir in _check_dirs:
                actual = sum(1 for f in os.listdir(out_dir)
                             if f.startswith(stem + "_") and f.endswith(".npz"))
                if actual < expected:
                    _short.append(
                        f"  {os.path.basename(out_dir)}/{stem}: {actual}/{expected} records"
                    )
        if _short:
            print(f"  COVERAGE GAPS ({len(_short)} shard(s)):",
                  file=sys.stderr if args.ai else sys.stdout, flush=True)
            for s in _short:
                print(s, file=sys.stderr if args.ai else sys.stdout, flush=True)
            print("  Rerun precompute to fill gaps — exiting with error.",
                  file=sys.stderr if args.ai else sys.stdout, flush=True)
            if args.ai:
                import json as _json
                print(_json.dumps({"ok": False, "error": "coverage gaps",
                                   "gap_count": len(_short),
                                   "done": len(work_items), "total": len(work_items),
                                   "pct": 100, "errors": _short[:10]}))
            sys.exit(1)
        else:
            print(f"  Coverage OK — all {len(work_items)} shard(s) complete.",
                  file=sys.stderr if args.ai else sys.stdout, flush=True)

    # Mark versioned cache dirs complete (best-effort; only when the output path
    # matched the version hash written at startup).
    if _active_caches:
        try:
            _total_shards = len(work_items)
            for _enc, _cache in _active_caches.items():
                _rec_count = _cache.record_count()
                _cache.mark_complete(record_count=_rec_count, shard_count=_total_shards)
        except Exception:
            pass

    if args.ai:
        import json as _json
        error_list = [r.get("error") for r in results if r.get("error")]
        print(_json.dumps({
            "ok": errs == 0,
            "done": len(work_items),
            "total": len(work_items),
            "pct": 100,
            "eta_sec": 0,
            "elapsed_hours": round(elapsed, 2),
            "qwen3_written": wq,
            "vae_written": wv,
            "siglip_written": ws if siglip_out else None,
            "errors": error_list[:10],
        }))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
