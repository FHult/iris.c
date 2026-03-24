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

    from mflux.models.flux2 import Flux2Klein
    flux = Flux2Klein(model_path=flux_model_path, quantize=None)
    flux.vae.freeze()
    _W["vae"] = flux.vae

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

    written = 0
    for i in range(0, len(tokenized), batch_size):
        batch = tokenized[i:i + batch_size]
        seq_lens = [len(ids) for _, _, ids in batch]
        max_len = max(seq_lens)
        padded = mx.array(np.array([
            ids + [pad_id] * (max_len - len(ids)) for _, _, ids in batch
        ]))
        try:
            outputs = model(padded, output_hidden_states=True)
            h = outputs.hidden_states
            h9_np  = np.array(h[9])
            h18_np = np.array(h[18])
            h27_np = np.array(h[27])
            for j, (rec_id, _, _) in enumerate(batch):
                sl = seq_lens[j]
                emb = np.concatenate(
                    [h9_np[j, :sl], h18_np[j, :sl], h27_np[j, :sl]], axis=-1
                )
                q, scale = _quantize_4bit(emb.astype(np.float32))
                np.savez(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
                written += 1
        except Exception as e:
            print(f"  Qwen3 batch {i // batch_size} failed ({e}), retrying single",
                  file=sys.stderr)
            for rec_id, caption, ids in batch:
                try:
                    sl = len(ids)
                    out = model(mx.array([ids]), output_hidden_states=True)
                    h = out.hidden_states
                    emb = np.concatenate([
                        np.array(h[9][0, :sl]),
                        np.array(h[18][0, :sl]),
                        np.array(h[27][0, :sl]),
                    ], axis=-1)
                    q, scale = _quantize_4bit(emb.astype(np.float32))
                    np.savez(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
                    written += 1
                except Exception as e2:
                    print(f"  Skipping {rec_id}: {e2}", file=sys.stderr)
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

    return written


def _vae_gpu_encode(batch_ids, batch_imgs, out_dir) -> int:
    import mlx.core as mx
    vae = _W["vae"]
    try:
        stacked = np.concatenate(batch_imgs, axis=0)
        latents = vae.encode(mx.array(stacked))
        mx.eval(latents)
        latents_np = np.array(latents)
        for k, rec_id in enumerate(batch_ids):
            q, scale = _quantize_int8(latents_np[k].astype(np.float32))
            np.savez(os.path.join(out_dir, f"{rec_id}.npz"), q=q, scale=scale)
        return len(batch_ids)
    except Exception as e:
        print(f"  VAE batch failed ({e}), retrying single", file=sys.stderr)
        saved = 0
        for rec_id, img_np in zip(batch_ids, batch_imgs):
            try:
                latent = vae.encode(mx.array(img_np))
                mx.eval(latent)
                q, scale = _quantize_int8(np.array(latent[0]).astype(np.float32))
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

    # Phase 1: read tar once — partition records by which outputs are still needed
    wq = wv = ws = 0          # already-written counts (resume)
    pending_q: list = []      # (rec_id, jpg_bytes, caption) needing Qwen3
    pending_v: list = []      # needing VAE
    pending_s: list = []      # needing SigLIP

    for rec_id, jpg_bytes, caption in iter_shard(shard_path):
        need_q = bool(qwen3_out) and not os.path.exists(
            os.path.join(qwen3_out, f"{rec_id}.npz"))
        need_v = bool(vae_out)   and not os.path.exists(
            os.path.join(vae_out, f"{rec_id}.npz"))
        need_s = bool(siglip_out) and not os.path.exists(
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

    # Phase 2a: Qwen3
    if pending_q:
        wq += _encode_qwen3(pending_q, qwen3_out, qwen3_batch)

    # Phase 2b: VAE (1-ahead prefetch)
    if pending_v:
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

    return {
        "shard": shard_path,
        "wq": wq, "wv": wv, "ws": ws,
        "error": False,
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
    args = parser.parse_args()

    shards = sorted(glob.glob(os.path.join(args.shards, "*.tar")))
    if not shards:
        print(f"No .tar files in {args.shards}", file=sys.stderr)
        sys.exit(1)

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
    results = []
    t_start = t_last = _time.time()
    interval_rates = []

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
            if dt > 0:
                interval_rates.append(1.0 / dt)
            avg_rate = sum(interval_rates) / len(interval_rates) if interval_rates else 0
            eta = (len(work_items) - done) / avg_rate if avg_rate > 0 else 0
            t_last = t_now

            errs = sum(1 for r in results if r["error"])
            wq   = sum(r["wq"] for r in results)
            wv   = sum(r["wv"] for r in results)
            ws   = sum(r["ws"] for r in results)
            err_str = f"  errors={errs}" if errs else ""
            siglip_str = f"  siglip={ws:,}" if siglip_out else ""
            print(
                f"  [{done}/{len(work_items)}]"
                f"  qwen3={wq:,}  vae={wv:,}{siglip_str}"
                f"{err_str}  {1/avg_rate:.1f} s/shard  ETA {eta/60:.0f}m",
                flush=True,
            )

    wq = sum(r["wq"] for r in results)
    wv = sum(r["wv"] for r in results)
    ws = sum(r["ws"] for r in results)
    errs = sum(1 for r in results if r["error"])
    elapsed = (_time.time() - t_start) / 3600
    print(f"\nDone in {elapsed:.1f}h.")
    print(f"  Qwen3:  {wq:,} embeddings  → {args.qwen3_output}/")
    print(f"  VAE:    {wv:,} latents      → {args.vae_output}/")
    if siglip_out:
        print(f"  SigLIP: {ws:,} features    → {siglip_out}/")
    if errs:
        print(f"  {errs} shards had errors (check stderr)")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
