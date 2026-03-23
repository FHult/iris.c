"""
train/ip_adapter/dataset.py — Two-level async prefetch loader for IP-Adapter training.

Two-level prefetch pipeline (plans/ip-adapter-training.md §3.4):
  Level 1 (shard thread)  — pre-decompresses the next tar into a memory dict
                            while the current shard is being consumed. Eliminates
                            the 0.5–2s pause at each of ~310 shard boundaries.
  Level 2 (sample thread) — decodes JPEG (turbojpeg) + dequantises pre-computed
                            embeds + builds batches. Overlap with GPU step.
  GPU (main thread)       — training step; never sees shard boundary stalls.

CPU allocation:
  Prefetch threads: always 2 (GPU is the bottleneck — more adds noise).
  JPEG decode: turbojpeg (2–4× faster than Pillow). Falls back to Pillow if unavailable.
  Shard workers target P-cores via daemon threads (OS routes to P-cores at user QoS).

Multi-resolution bucketing (§3.9):
  BUCKETS = [(512,512), (512,768), (768,512), (640,640), (512,896), (896,512)]
  Each batch uses one bucket; images are resized to that bucket's (H,W).
  No 256px — degenerate for Flux's patchification.

GPU augmentation (§3.10):
  augment_mlx(): random horizontal flip + random crop (pad to +32px, crop to target).
  Runs in MLX on GPU after the batch is transferred — no CPU→GPU copy penalty.

Pre-computed embed loading:
  If data/qwen3_q4/{id}.npz exists → dequantise and return (no Qwen3 forward pass).
  If data/vae_int8/{id}.npz exists → dequantise and return (no VAE encode).
  If data/siglip_q4/{id}.npz exists → dequantise and return (no SigLIP forward pass).
  Falls back to None for each encoder when pre-computed cache is absent.
"""

import io
import os
import queue
import random
import sys
import tarfile
import threading
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    from turbojpeg import TurboJPEG as _TurboJPEG, TJPF_RGB as _TJPF_RGB
    _HAS_TURBOJPEG = True
except ImportError:
    _HAS_TURBOJPEG = False
    _TJPF_RGB = None

from .utils import PERF_CORES as _PERF_CORES


# ---------------------------------------------------------------------------
# Multi-resolution training buckets (§3.9)
# ---------------------------------------------------------------------------

BUCKETS: List[Tuple[int, int]] = [
    (512, 512),
    (512, 768),
    (768, 512),
    (640, 640),
    (512, 896),
    (896, 512),
]

# Do NOT add 256px — degenerate for Flux's patchification.


def _select_bucket(w: int, h: int) -> Tuple[int, int]:
    """Return the bucket (bH, bW) closest in aspect ratio to the source image."""
    if w == 0 or h == 0:
        return BUCKETS[0]
    aspect = w / h
    best = min(BUCKETS, key=lambda b: abs(b[1] / b[0] - aspect))
    return best  # (H, W)


# ---------------------------------------------------------------------------
# JPEG decode / encode helpers
# ---------------------------------------------------------------------------

def _make_jpeg():
    """Return a per-thread TurboJPEG instance (not process-safe to pass across forks)."""
    if _HAS_TURBOJPEG:
        return _TurboJPEG()
    return None


def _decode_jpeg(raw: bytes, tj=None) -> Optional[np.ndarray]:
    """
    Decode JPEG bytes → HWC uint8 RGB numpy array.
    Uses turbojpeg (2–4× faster) when available, falls back to Pillow.
    """
    try:
        if tj is not None:
            return tj.decode(raw, pixel_format=_TJPF_RGB)
        from PIL import Image
        return np.array(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def _resize_to_bucket(img: np.ndarray, bucket_h: int, bucket_w: int) -> np.ndarray:
    """Resize HWC numpy image to (bucket_h, bucket_w) using Pillow LANCZOS."""
    h, w = img.shape[:2]
    if h == bucket_h and w == bucket_w:
        return img
    from PIL import Image
    return np.array(
        Image.fromarray(img).resize((bucket_w, bucket_h), Image.LANCZOS),
        dtype=np.uint8,
    )


def _normalize(img: np.ndarray) -> np.ndarray:
    """uint8 HWC → float32 CHW in [-1, 1]."""
    return (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)


# ---------------------------------------------------------------------------
# GPU augmentation (§3.10)
# ---------------------------------------------------------------------------

def augment_mlx(img, bucket_h: int, bucket_w: int):
    """
    MLX GPU augmentation: random horizontal flip + random crop.
    img: MLX array [B, C, H_pad, W_pad] — BCHW format from the prefetch thread.
    Images are pre-padded to (bucket_h + 32, bucket_w + 32).

    Uses Python random (not mx.random) to avoid GPU→CPU sync stalls.
    """
    import mlx.core as mx

    # Random horizontal flip (axis=-1 = W axis in BCHW)
    if random.random() > 0.5:
        img = img[..., ::-1]

    # Random crop offsets — Python random avoids GPU sync
    h_off = random.randint(0, 31)
    w_off = random.randint(0, 31)

    # BCHW crop: keep all channels, slice H and W
    if img.ndim == 4:  # [B, C, H_pad, W_pad]
        return img[:, :, h_off:h_off + bucket_h, w_off:w_off + bucket_w]
    else:  # [C, H_pad, W_pad]
        return img[:, h_off:h_off + bucket_h, w_off:w_off + bucket_w]


# ---------------------------------------------------------------------------
# Pre-computed embed loaders
# ---------------------------------------------------------------------------

def _load_qwen3_embed(rec_id: str, qwen3_dir: Optional[str]) -> Optional[np.ndarray]:
    """
    Load 4-bit quantised Qwen3 text embedding.
    Returns float16 [seq, 7680] or None if not available.
    """
    if not qwen3_dir:
        return None
    path = os.path.join(qwen3_dir, f"{rec_id}.npz")
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path)
        q, scale = d["q"], d["scale"]
        # Use explicit out= buffers to prevent numpy's refcount-based in-place
        # ufunc optimisation from reusing q's memory for the first operation,
        # which would corrupt q before the second nibble can be extracted.
        lo = np.empty(q.shape, dtype=np.int8)
        hi = np.empty(q.shape, dtype=np.int8)
        np.bitwise_and(q, np.int8(0x0F), out=lo)
        np.right_shift(q, 4, out=hi)
        np.bitwise_and(hi, np.int8(0x0F), out=hi)
        full = np.empty((q.shape[0], q.shape[1] * 2), dtype=np.int8)
        full[:, 0::2] = lo
        full[:, 1::2] = hi
        return (full.astype(np.float32) * scale.astype(np.float32)).astype(np.float16)
    except Exception:
        return None


def _load_vae_latent(
    rec_id: str,
    vae_dir: Optional[str],
    expected_hw: Optional[Tuple[int, int]] = None,
) -> Optional[np.ndarray]:
    """
    Load int8 quantised VAE latent.
    Returns float16 [32, H/8, W/8] or None if not available.

    expected_hw: (H//8, W//8) of the current training bucket.  If the stored
    latent has a different spatial shape the cache entry is from a different
    resolution and we return None so the caller falls back to inline VAE encode.
    """
    if not vae_dir:
        return None
    path = os.path.join(vae_dir, f"{rec_id}.npz")
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path)
        latent = (d["q"].astype(np.float32) * d["scale"].astype(np.float32)).astype(np.float16)
        if expected_hw is not None:
            if latent.shape[1] != expected_hw[0] or latent.shape[2] != expected_hw[1]:
                return None
        return latent
    except Exception:
        return None


def _load_siglip_embed(rec_id: str, siglip_dir: Optional[str]) -> Optional[np.ndarray]:
    """
    Load 4-bit quantised SigLIP features.
    Returns float16 [729, 1152] or None if not available.
    """
    if not siglip_dir:
        return None
    path = os.path.join(siglip_dir, f"{rec_id}.npz")
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path)
        q, scale = d["q"], d["scale"]
        lo = np.empty(q.shape, dtype=np.int8)
        hi = np.empty(q.shape, dtype=np.int8)
        np.bitwise_and(q, np.int8(0x0F), out=lo)
        np.right_shift(q, 4, out=hi)
        np.bitwise_and(hi, np.int8(0x0F), out=hi)
        full = np.empty((q.shape[0], q.shape[1] * 2), dtype=np.int8)
        full[:, 0::2] = lo
        full[:, 1::2] = hi
        return (full.astype(np.float32) * scale.astype(np.float32)).astype(np.float16)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shard content iteration
# ---------------------------------------------------------------------------

def _iter_shard_contents(contents: Dict[str, bytes]) -> list:
    """
    Parse {filename: bytes} dict (pre-decompressed shard) into list of record dicts.
    Returns [{"id": str, "jpg": bytes, "txt": str}, ...]
    """
    keys: Dict[str, Dict[str, str]] = {}
    for name in contents:
        stem, _, ext = name.rpartition(".")
        if stem not in keys:
            keys[stem] = {}
        keys[stem][ext.lower()] = name

    records = []
    for stem, exts in keys.items():
        jpg_key = exts.get("jpg") or exts.get("jpeg") or exts.get("png")
        txt_key = exts.get("txt") or exts.get("caption")
        if not jpg_key or not txt_key:
            continue
        txt = contents[txt_key].decode("utf-8", errors="replace").strip()
        records.append({"id": stem, "jpg": contents[jpg_key], "txt": txt})
    return records


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def make_prefetch_loader(
    shard_paths: List[str],
    batch_size: int = 2,
    image_dropout_prob: float = 0.30,
    text_dropout_prob: float = 0.10,
    sample_buffer: int = 6,
    bucket: Optional[Tuple[int, int]] = None,
    qwen3_cache_dir: Optional[str] = None,
    vae_cache_dir: Optional[str] = None,
    siglip_cache_dir: Optional[str] = None,
    anchor_shard_dir: Optional[str] = None,
    anchor_mix_ratio: float = 0.20,
) -> Iterator:
    """
    Two-level prefetch pipeline (§3.4).

    shard_paths:          list of .tar shard file paths (in any order; shuffled internally)
    anchor_shard_dir:     path to anchor shards mixed in at anchor_mix_ratio (default 20%).
                          Use during incremental/chunked training to prevent distribution shift.
    anchor_mix_ratio:     fraction of batches drawn from anchor set (default 0.20)
    batch_size:           images per batch (default 2)
    image_dropout_prob:   null image conditioning dropout rate (default 0.30)
    text_dropout_prob:    null text conditioning dropout rate (default 0.10)
    sample_buffer:        max batches in the level-2 queue (default 6)
    bucket:               (H, W) training resolution; if None, randomly selects from BUCKETS
    qwen3_cache_dir:      path to pre-computed Qwen3 .npz files (data/qwen3_q4/)
    vae_cache_dir:        path to pre-computed VAE .npz files (data/vae_int8/)
    siglip_cache_dir:     path to pre-computed SigLIP .npz files (data/siglip_q4/)

    Yields batches of:
      images:      float32 numpy [B, C, H+32, W+32] — padded for GPU crop augmentation
      captions:    list of str, len B (empty string if text dropout)
      style_refs:  float32 numpy [B, C, H+32, W+32] — same image (self-supervised)
      text_embeds: float16 numpy [B, seq, 7680] or None (if no Qwen3 cache)
      vae_latents: float16 numpy [B, 32, H/8, W/8] or None (if no VAE cache)
      siglip_feats:float16 numpy [B, 729, 1152] or None (if no SigLIP cache)
      bucket_hw:   (H, W) tuple for this batch
    """
    paths = list(shard_paths)
    anchor_paths: List[str] = []
    if anchor_shard_dir and os.path.isdir(anchor_shard_dir):
        anchor_paths = sorted(
            os.path.join(anchor_shard_dir, f)
            for f in os.listdir(anchor_shard_dir)
            if f.endswith(".tar")
        )

    shard_q: queue.Queue = queue.Queue(maxsize=2)
    sample_q: queue.Queue = queue.Queue(maxsize=sample_buffer)

    # ---- Level 1: shard decompressor thread --------------------------------
    def shard_loader():
        rng = random.Random()
        consecutive_errors = 0
        while True:
            # Build this epoch's shard list: mix anchor shards at anchor_mix_ratio
            epoch_paths = list(paths)
            if anchor_paths:
                n_anchor = max(1, int(len(epoch_paths) * anchor_mix_ratio / (1 - anchor_mix_ratio)))
                epoch_paths += rng.choices(anchor_paths, k=n_anchor)
            rng.shuffle(epoch_paths)
            for path in epoch_paths:
                try:
                    with tarfile.open(path) as tar:
                        contents = {
                            m.name: tar.extractfile(m).read()
                            for m in tar.getmembers()
                            if m.isfile()
                        }
                    shard_q.put(contents)
                    consecutive_errors = 0
                except Exception as e:
                    print(f"Shard error {path}: {e}", file=sys.stderr)
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        # Too many consecutive failures — signal consumer to stop
                        shard_q.put(None)
                        print("FATAL: 10 consecutive shard errors — stopping loader",
                              file=sys.stderr)
                        return

    # ---- Level 2: sample decoder + batch builder thread -------------------
    def sample_decoder():
        tj = _make_jpeg()
        rng = random.Random()

        while True:
            contents = shard_q.get()
            records = _iter_shard_contents(contents)
            rng.shuffle(records)

            # Pick bucket for this shard
            if bucket is not None:
                bH, bW = bucket
            else:
                bH, bW = rng.choice(BUCKETS)

            # Pad target size by 32px for GPU random-crop augmentation
            pad_h, pad_w = bH + 32, bW + 32

            imgs_buf, caps_buf, refs_buf = [], [], []
            temb_buf, vlat_buf, sfeat_buf = [], [], []

            for rec in records:
                # Decode image
                img = _decode_jpeg(rec["jpg"], tj)
                if img is None:
                    continue

                # Resize to padded bucket size
                img = _resize_to_bucket(img, pad_h, pad_w)

                # Null conditioning dropout (decided here, outside MLX compiled region)
                if rng.random() < image_dropout_prob:
                    style_ref = np.zeros_like(img)
                else:
                    style_ref = img.copy()

                caption = rec["txt"]
                if rng.random() < text_dropout_prob:
                    caption = ""

                # Pre-computed embeds (CPU-trivial dequant)
                text_emb = _load_qwen3_embed(rec["id"], qwen3_cache_dir)
                # VAE cache was encoded at a fixed resolution; reject entries whose
                # spatial shape doesn't match the current training bucket.
                vae_lat = _load_vae_latent(
                    rec["id"], vae_cache_dir, expected_hw=(bH // 8, bW // 8)
                )
                siglip_feat = _load_siglip_embed(rec["id"], siglip_cache_dir)

                imgs_buf.append(_normalize(img))
                refs_buf.append(_normalize(style_ref))
                caps_buf.append(caption)
                temb_buf.append(text_emb)
                vlat_buf.append(vae_lat)
                sfeat_buf.append(siglip_feat)

                if len(imgs_buf) == batch_size:
                    # Stack arrays; keep None for missing caches
                    t_arr = np.stack(temb_buf) if temb_buf[0] is not None else None
                    v_arr = np.stack(vlat_buf) if vlat_buf[0] is not None else None
                    s_arr = np.stack(sfeat_buf) if sfeat_buf[0] is not None else None

                    sample_q.put((
                        np.stack(imgs_buf),     # [B, C, H+32, W+32]
                        list(caps_buf),
                        np.stack(refs_buf),
                        t_arr,
                        v_arr,
                        s_arr,
                        (bH, bW),
                    ))
                    imgs_buf, caps_buf, refs_buf = [], [], []
                    temb_buf, vlat_buf, sfeat_buf = [], [], []

    threading.Thread(target=shard_loader, daemon=True).start()
    threading.Thread(target=sample_decoder, daemon=True).start()

    while True:
        try:
            item = sample_q.get(timeout=120)
        except queue.Empty:
            raise RuntimeError(
                "sample_q timeout (120s) — shard_loader or sample_decoder likely crashed"
            )
        yield item
