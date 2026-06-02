"""
train/vae_distill/dataset.py — Training data pipeline for proxy VAE distillation.

Pairs source images (from WebDataset .tar shards) with pre-computed teacher
latents (.npz files from the VAE precompute cache).

Data flow:
  1. Scan shard_dirs for *.tar files whose stems have ≥1 npz in vae_cache_dir.
  2. Optionally filter shards by unified_score_threshold (reads .unified_score.json).
  3. Build a flat index of (shard_path, record_id) pairs with both image + latent.
  4. Yield batches of (image [B,3,H,W] float32 in [-1,1], latent [B,32,H/8,W/8]).

Augmentation:
  Horizontal flip (safe — VAE is symmetric; flip image and flip latent width-dim).
  No colour jitter (colour is encoded in VAE channels; augmenting image without
  re-running teacher would produce mismatched pairs).
"""

from __future__ import annotations

import glob
import os
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# Reuse the JPEG decode + resize from precompute_all
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from precompute_all import _decode_jpg, _resize


# ---------------------------------------------------------------------------
# Shard index
# ---------------------------------------------------------------------------

def _load_unified_score(shard_path: Path) -> Optional[float]:
    score_file = shard_path.with_suffix(".unified_score.json")
    if not score_file.exists():
        return None
    try:
        import json
        return float(json.loads(score_file.read_text()).get("final_value_score", 0.0))
    except Exception:
        return None


def build_index(
    shard_dirs: list[str],
    vae_cache_dir: str,
    min_unified_score: float = 0.0,
    max_shards: Optional[int] = None,
) -> list[tuple[str, str]]:
    """
    Build a list of (shard_tar_path, record_id) pairs for training.

    Only includes records that have both a source image (in the tar) and a
    pre-computed teacher latent (as <record_id>.npz in vae_cache_dir).

    Parameters
    ----------
    shard_dirs          : directories containing *.tar shard files
    vae_cache_dir       : directory (or symlink target) containing *.npz files
    min_unified_score   : skip shards with final_value_score below this
    max_shards          : if set, sample at most this many shards

    Returns list of (shard_path, record_id) pairs.
    """
    cache_dir = Path(vae_cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"VAE cache dir not found: {vae_cache_dir}")

    # Build set of record IDs that have a cached latent
    cached_ids: set[str] = {
        p.stem for p in cache_dir.glob("*.npz")
    }
    if not cached_ids:
        raise RuntimeError(f"No .npz files found in {vae_cache_dir}")
    print(f"  [dataset] {len(cached_ids):,} cached latents in {vae_cache_dir}")

    # Group cached IDs by shard prefix (first 6 chars = shard ID zero-padded)
    shard_to_ids: dict[str, list[str]] = defaultdict(list)
    for rec_id in cached_ids:
        shard_prefix = rec_id.split("_")[0] if "_" in rec_id else rec_id[:6]
        shard_to_ids[shard_prefix].append(rec_id)

    # Find tar files
    tar_paths: dict[str, Path] = {}
    for sdir in shard_dirs:
        for tar in Path(sdir).glob("*.tar"):
            tar_paths[tar.stem] = tar

    # Build index with optional score filtering
    all_shards: list[Path] = []
    for stem, tar_path in sorted(tar_paths.items()):
        if stem not in shard_to_ids:
            continue
        if min_unified_score > 0.0:
            score = _load_unified_score(tar_path)
            if score is not None and score < min_unified_score:
                continue
        all_shards.append(tar_path)

    if max_shards is not None and len(all_shards) > max_shards:
        random.shuffle(all_shards)
        all_shards = all_shards[:max_shards]

    print(f"  [dataset] {len(all_shards):,} shards pass filter")

    # Build flat (shard_path, record_id) index
    index: list[tuple[str, str]] = []
    for tar_path in all_shards:
        stem = tar_path.stem
        for rec_id in shard_to_ids.get(stem, []):
            index.append((str(tar_path), rec_id))

    random.shuffle(index)
    print(f"  [dataset] {len(index):,} training pairs indexed")
    return index


# ---------------------------------------------------------------------------
# Batch iterator
# ---------------------------------------------------------------------------

class VAEDistillDataset:
    """
    Iterable dataset yielding (image_batch, latent_batch) pairs.

    Loads all images for a shard up front (one tar.open per shard), then
    pairs them with their pre-loaded NPZ latents for fast random access.

    Parameters
    ----------
    index       : output of build_index()
    vae_cache   : path to directory containing <record_id>.npz files
    image_size  : resize short edge to this, then center-crop
    batch_size  : images per batch
    hflip_prob  : probability of horizontal flip augmentation
    seed        : RNG seed for reproducibility
    """

    def __init__(
        self,
        index: list[tuple[str, str]],
        vae_cache: str,
        image_size: int = 512,
        batch_size: int = 8,
        hflip_prob: float = 0.5,
        seed: int = 42,
    ):
        self._index      = index
        self._cache_dir  = Path(vae_cache)
        self._image_size = image_size
        self._batch_size = batch_size
        self._hflip_prob = hflip_prob
        self._rng        = random.Random(seed)
        self._np_rng     = np.random.default_rng(seed)

        try:
            from turbojpeg import TurboJPEG
            self._tj = TurboJPEG()
        except Exception:
            self._tj = None

    def __len__(self) -> int:
        return len(self._index) // self._batch_size

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        idx = list(range(len(self._index)))
        self._rng.shuffle(idx)

        # Group consecutive indices by shard to amortise tar.open() cost
        shard_groups: dict[str, list[int]] = defaultdict(list)
        for i in idx:
            shard_path, _ = self._index[i]
            shard_groups[shard_path].append(i)

        # Load shard images on demand, emit batches
        image_buf:  list[np.ndarray] = []
        latent_buf: list[np.ndarray] = []

        for shard_path, indices in shard_groups.items():
            # Load all images from this shard into memory
            try:
                shard_imgs = self._load_shard_images(shard_path)
            except Exception as e:
                print(f"  [dataset] skip {shard_path}: {e}", flush=True)
                continue

            for i in indices:
                _, rec_id = self._index[i]
                jpg_bytes = shard_imgs.get(rec_id)
                if jpg_bytes is None:
                    continue

                # Load precomputed latent
                npz_path = self._cache_dir / f"{rec_id}.npz"
                if not npz_path.exists():
                    continue
                try:
                    latent_f32 = np.load(str(npz_path))["latent"].astype(np.float32)
                except Exception:
                    continue

                # Decode and resize image
                try:
                    img = _decode_jpg(jpg_bytes, self._tj)       # [H, W, 3] uint8
                    img = _resize(img, self._image_size)          # [S, S, 3] uint8
                    img_f = (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
                    # [3, H, W] float32 in [-1, 1]
                except Exception:
                    continue

                # Horizontal flip augmentation (flip both image and latent)
                if self._np_rng.random() < self._hflip_prob:
                    img_f    = img_f[:, :, ::-1].copy()
                    latent_f32 = latent_f32[:, :, ::-1].copy()

                image_buf.append(img_f)
                latent_buf.append(latent_f32)

                if len(image_buf) == self._batch_size:
                    yield (
                        np.stack(image_buf,  axis=0),   # [B, 3, H, W]
                        np.stack(latent_buf, axis=0),   # [B, 32, H/8, W/8]
                    )
                    image_buf.clear()
                    latent_buf.clear()

    def _load_shard_images(self, shard_path: str) -> dict[str, bytes]:
        """Return {stem: jpg_bytes} for all image members in the tar."""
        out: dict[str, bytes] = {}
        with tarfile.open(shard_path) as tar:
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                stem, _, ext = m.name.rpartition(".")
                if ext.lower() in ("jpg", "jpeg", "png"):
                    out[stem] = tar.extractfile(m).read()
        return out
