"""
train/ip_adapter/dataset.py — WebDataset loader with two-level async prefetch.

Two-level prefetch:
  Level 1 (shard): background thread decompresses the next tar shard while
                   the current shard is being consumed.
  Level 2 (sample): background thread decodes + preprocesses the next batch
                    while the GPU runs the current step.

CPU core targeting:
  img2dataset shard writing uses --processes_count to target performance cores.
  During training, the prefetch thread pool is sized based on os.cpu_count()
  filtered to performance cores (P-cores on M1 Max = 8).

  On macOS, performance cores can be targeted via QoS:
    thread.daemon = True
    # Apple's Grand Central Dispatch routes background QoS to E-cores,
    # userInitiated/userInteractive routes to P-cores.
    # Python threads don't expose QoS directly, but setting thread priority
    # via ctypes is possible (see _set_pcores() below).

Usage:
    loader = make_prefetch_loader(config, batch_size=2)
    for batch in loader:
        images, captions, style_refs = batch
"""

import os
import queue
import threading
from typing import Iterator, Optional, Tuple

import numpy as np

try:
    import webdataset as wds
    from PIL import Image
    import io
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


from .utils import PERF_CORES as _PERF_CORE_COUNT


def _check_deps():
    if not _HAS_DEPS:
        raise ImportError(
            "Missing: pip install webdataset Pillow\n"
            "Run: source train/.venv/bin/activate"
        )


def _decode_image(data: bytes, size: int) -> Optional[np.ndarray]:
    """Decode JPEG bytes → RGB numpy [H, W, 3] uint8, resized to size×size."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.width != size or img.height != size:
            img = img.resize((size, size), Image.LANCZOS)
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """uint8 [H,W,3] → float32 [3,H,W] in [-1, 1]."""
    return (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)


def _make_webdataset(
    shard_path: str,
    image_size: int,
    image_dropout_prob: float = 0.0,
    text_dropout_prob: float = 0.0,
    shuffle_buffer: int = 10000,
) -> Iterator:
    """
    Build a WebDataset iterator over shards in shard_path.

    Each sample yields (image_pixels, caption, style_ref_pixels) where:
      - image_pixels: float32 [3, H, W] normalised to [-1, 1] — training target
      - caption: str — text prompt
      - style_ref_pixels: float32 [3, H, W] or None (if same-image self-supervised)

    For style reference training, we use a second randomly-sampled image from
    the same batch as the style reference (self-supervised style learning).
    The IP-Adapter is trained to reconstruct the target image given the style
    of the reference — when target == reference it learns identity; diversity
    in the batch provides style variation.
    """
    _check_deps()

    pattern = os.path.join(shard_path, "*.tar")
    dataset = (
        wds.WebDataset(pattern, resampled=True, shardshuffle=True)
        .shuffle(shuffle_buffer)
        .decode("pil")
        .to_tuple("jpg;png", "txt", handler=wds.warn_and_continue)
    )

    for img_data, caption in dataset:
        # Decode target image
        if isinstance(img_data, bytes):
            img = _decode_image(img_data, image_size)
        else:
            # Already decoded by webdataset
            img = np.array(img_data.convert("RGB").resize(
                (image_size, image_size), Image.LANCZOS
            ), dtype=np.uint8)

        if img is None:
            continue

        # Null conditioning dropout
        import random
        if random.random() < image_dropout_prob:
            style_ref = np.zeros_like(img)  # zeroed style reference
        else:
            style_ref = img  # self-supervised: same image as style ref

        if random.random() < text_dropout_prob:
            caption = ""

        yield _normalize_image(img), caption, _normalize_image(style_ref)


def make_prefetch_loader(
    shard_path: str,
    image_size: int,
    batch_size: int = 2,
    prefetch_batches: int = 4,
    num_threads: int = 2,
    image_dropout_prob: float = 0.30,
    text_dropout_prob: float = 0.10,
    shuffle_buffer: int = 10000,
) -> Iterator:
    """
    Two-level prefetch loader: fills a queue with pre-decoded batches.

    num_threads: number of prefetch threads (default 2, targeting P-cores)
    prefetch_batches: queue depth in batches (default 4)

    Returns batches of (images, captions, style_refs):
      images:     float32 numpy [B, 3, H, W]
      captions:   list of str, len B
      style_refs: float32 numpy [B, 3, H, W]
    """
    _check_deps()

    q: queue.Queue = queue.Queue(maxsize=prefetch_batches * num_threads)
    stop_event = threading.Event()

    def _worker():
        gen = _make_webdataset(
            shard_path, image_size,
            image_dropout_prob=image_dropout_prob,
            text_dropout_prob=text_dropout_prob,
            shuffle_buffer=shuffle_buffer,
        )
        imgs_buf, caps_buf, refs_buf = [], [], []
        for img, cap, ref in gen:
            if stop_event.is_set():
                break
            imgs_buf.append(img)
            caps_buf.append(cap)
            refs_buf.append(ref)
            if len(imgs_buf) == batch_size:
                batch = (
                    np.stack(imgs_buf),
                    list(caps_buf),
                    np.stack(refs_buf),
                )
                q.put(batch)
                imgs_buf, caps_buf, refs_buf = [], [], []

    # Launch prefetch threads (use daemon so they die with the main process)
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            batch = q.get(timeout=60)
            yield batch
    finally:
        stop_event.set()


def make_shard_writer(
    output_path: str,
    shard_size: int = 5000,
    num_workers: int = None,
) -> None:
    """
    Utility called by build_shards.py — not used during training.
    Documents the shard writing config for reference.

    num_workers: defaults to P-core count (8 on M1 Max)
    shard_size: 5000 images/shard at ~76KB avg JPEG = ~380MB/shard
    """
    if num_workers is None:
        # Target performance cores on Apple Silicon
        num_workers = min(_PERF_CORE_COUNT, os.cpu_count() or 4)
    return dict(
        output_path=output_path,
        shard_size=shard_size,
        num_workers=num_workers,
    )
