"""
train/scripts/clip_dedup.py — CLIP embedding + FAISS deduplication.

Three subcommands called sequentially by the orchestrator per chunk:

  embed       — compute CLIP image embeddings for all shards in --shards,
                write one {stem}.npz per shard into --embeddings
  build-index — load .npz files from --embeddings, extend (or create) the
                cumulative FAISS IndexFlatIP at --index; writes a
                {index}.ids sidecar (one record ID per line, matching add order)
  find-dups   — KNN search over the full index, write near-duplicate IDs
                to --out (the build_shards blocklist); append-safe

The index is cumulative: each chunk's build-index call extends it so
cross-chunk near-duplicates are caught when later chunks run find-dups.

Usage (called by orchestrator):
    clip_dedup.py embed       --shards <dir> --embeddings <dir>
    clip_dedup.py build-index --embeddings <dir> --index <path.faiss>
    clip_dedup.py find-dups   --index <path.faiss> --out <blocklist.txt>
"""

import argparse
import glob
import io
import os
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Must be set before numpy/FAISS import on macOS to prevent libOMP crash.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    LOG_DIR, write_heartbeat, log_event, log_orch, now_iso,
)

try:
    import subprocess as _sp
    _PERF_CORES = int(_sp.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4

DUP_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Device + CLIP loading
# ---------------------------------------------------------------------------

def _clip_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


_clip_model      = None
_clip_preprocess = None
_clip_backend    = None


def _load_clip() -> None:
    global _clip_model, _clip_preprocess, _clip_backend
    if _clip_model is not None:
        return
    device = _clip_device()
    try:
        import torch, open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14-quickgelu", pretrained="openai"
        )
        model = model.to(device).half().eval()
        _clip_model, _clip_preprocess, _clip_backend = model, preprocess, "open_clip"
        log_orch(f"CLIP: ViT-L-14 via open_clip on {device}")
        return
    except ImportError:
        pass
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model = model.to(device).eval()
        proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model, _clip_preprocess, _clip_backend = model, proc, "transformers"
        log_orch(f"CLIP: ViT-B-32 via transformers on {device}")
        return
    except ImportError:
        pass
    raise RuntimeError(
        "CLIP requires open_clip or transformers+torch: "
        "pip install open-clip-torch  OR  pip install transformers torch"
    )


def _embed_batch(images: list) -> np.ndarray:
    """Embed PIL images; return L2-normalised float32 [N, D]."""
    import torch
    device = next(_clip_model.parameters()).device
    if _clip_backend == "open_clip":
        batch = torch.stack([_clip_preprocess(img) for img in images]).to(device)
        with torch.no_grad():
            feats = _clip_model.encode_image(batch.half()).float()
    else:
        inputs = _clip_preprocess(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = _clip_model.get_image_features(**inputs).float()
    arr   = feats.cpu().numpy()
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return (arr / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Shard decoding (parallelised, TurboJPEG when available)
# ---------------------------------------------------------------------------

def _decode_shard(tar_path: str, preprocess) -> tuple:
    """
    Open one WDS .tar shard, decode all images.
    Returns (ids: list[str], tensors: list[Tensor]) or (None, None) on error.
    Phase 1 is sequential (tarfile not thread-safe); phase 2 decodes in parallel.
    """
    from PIL import Image as _PIL
    try:
        from turbojpeg import TurboJPEG as _TJ, TJPF_RGB as _RGB
        _tj = _TJ()
        def _decode(data): return _PIL.fromarray(_tj.decode(data, pixel_format=_RGB))
    except ImportError:
        def _decode(data): return _PIL.open(io.BytesIO(data)).convert("RGB")

    raw_items = []
    try:
        with tarfile.open(tar_path) as tf:
            members = {m.name: m for m in tf.getmembers() if m.isfile()}
            keys: dict = {}
            for name in members:
                stem, _, ext = name.rpartition(".")
                if ext.lower() in ("jpg", "jpeg", "png"):
                    keys.setdefault(stem, name)
            for stem in sorted(keys):
                f = tf.extractfile(members[keys[stem]])
                if f:
                    raw_items.append((stem, f.read()))
    except Exception as e:
        log_orch(f"embed: skipping {os.path.basename(tar_path)}: {e}", level="warning")
        return None, None

    if not raw_items:
        return [], []

    def _proc(item):
        stem, raw = item
        try:
            return stem, preprocess(_decode(raw))
        except Exception:
            return None, None

    n_threads = min(_PERF_CORES, len(raw_items))
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        results = list(pool.map(_proc, raw_items))

    ids     = [k for k, t in results if k is not None]
    tensors = [t for k, t in results if t is not None]
    return ids, tensors


# ---------------------------------------------------------------------------
# embed subcommand
# ---------------------------------------------------------------------------

def cmd_embed(args) -> int:
    import torch
    shard_dir = Path(args.shards)
    embed_dir = Path(args.embeddings)
    embed_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(shard_dir.glob("*.tar"))
    if not shards:
        log_orch(f"embed: no .tar files in {shard_dir}")
        return 0

    _load_clip()
    batch_size = args.batch_size

    total = len(shards)
    done  = 0
    done_event = threading.Event()

    def _heartbeat():
        while not done_event.is_set():
            write_heartbeat("clip_dedup", phase="embed",
                            done=done, total=total,
                            pct=round(done / total * 100, 1) if total else 100)
            time.sleep(30)

    hb = threading.Thread(target=_heartbeat, daemon=True)
    hb.start()

    try:
        # Prefetch next shard while GPU processes current one
        pending = [(i, p) for i, p in enumerate(shards)
                   if not (embed_dir / (p.stem + ".npz")).exists()]

        log_orch(f"embed: {total} shards total, {len(pending)} to process")

        with ThreadPoolExecutor(max_workers=2) as loader:
            from collections import deque
            q: deque = deque()
            for k in range(min(2, len(pending))):
                q.append(loader.submit(_decode_shard, str(pending[k][1]), _clip_preprocess))

            for batch_i, (shard_idx, shard_path) in enumerate(pending):
                out_npz = embed_dir / (shard_path.stem + ".npz")
                ids, tensors = q.popleft().result()

                # Enqueue shard 2 ahead
                nxt = batch_i + 2
                if nxt < len(pending):
                    q.append(loader.submit(_decode_shard, str(pending[nxt][1]), _clip_preprocess))

                if ids is None or not tensors:
                    done += 1
                    continue

                device = next(_clip_model.parameters()).device
                all_embs = []
                with torch.no_grad():
                    for i in range(0, len(tensors), batch_size):
                        b = torch.stack(tensors[i:i + batch_size]).to(device)
                        if _clip_backend == "open_clip":
                            e = _clip_model.encode_image(b.half()).float()
                        else:
                            e = _clip_model.get_image_features(pixel_values=b).float()
                        e = e / e.norm(dim=-1, keepdim=True)
                        all_embs.append(e.cpu().numpy())

                emb_arr = np.concatenate(all_embs, axis=0).astype(np.float32)
                np.savez(out_npz, ids=np.array(ids), embeddings=emb_arr)

                log_event("clip_dedup", "embed_shard",
                          shard=shard_path.name, n=len(ids))
                done += 1

    finally:
        done_event.set()

    write_heartbeat("clip_dedup", phase="embed", done=total, total=total, pct=100)
    log_orch(f"embed: complete — embeddings in {embed_dir}")
    return 0


# ---------------------------------------------------------------------------
# build-index subcommand
# ---------------------------------------------------------------------------

def cmd_build_index(args) -> int:
    try:
        import faiss
        faiss.omp_set_num_threads(1)
    except ImportError:
        raise RuntimeError("build-index requires faiss-cpu: pip install faiss-cpu")

    embed_dir  = Path(args.embeddings)
    index_path = Path(args.index)
    ids_path   = index_path.with_suffix(".ids")

    npz_files = sorted(embed_dir.glob("*.npz"))
    if not npz_files:
        log_orch(f"build-index: no .npz files in {embed_dir}")
        return 0

    # Load existing state — cumulative across chunks
    if index_path.exists() and ids_path.exists():
        log_orch(f"build-index: extending existing index ({index_path.name})")
        index       = faiss.read_index(str(index_path))
        indexed_ids = set(ids_path.read_text().splitlines())
    else:
        index       = None
        indexed_ids = set()

    new_vecs: list = []
    new_ids:  list = []
    for npz_path in npz_files:
        data      = np.load(npz_path, allow_pickle=False)
        file_ids  = data["ids"].tolist()
        file_embs = data["embeddings"].astype(np.float32)
        for i, fid in enumerate(file_ids):
            if fid not in indexed_ids:
                new_ids.append(fid)
                new_vecs.append(file_embs[i])

    if not new_ids:
        log_orch("build-index: all embeddings already indexed")
        return 0

    vecs = np.stack(new_vecs, axis=0).astype(np.float32)
    dim  = vecs.shape[1]
    if index is None:
        index = faiss.IndexFlatIP(dim)

    index.add(vecs)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with open(ids_path, "a") as f:
        for fid in new_ids:
            f.write(fid + "\n")

    log_event("clip_dedup", "index_built",
              added=len(new_ids), total=index.ntotal, dim=dim)
    log_orch(f"build-index: +{len(new_ids)} vectors; index total={index.ntotal}")
    return 0


# ---------------------------------------------------------------------------
# find-dups subcommand
# ---------------------------------------------------------------------------

def cmd_find_dups(args) -> int:
    try:
        import faiss
        faiss.omp_set_num_threads(1)
    except ImportError:
        raise RuntimeError("find-dups requires faiss-cpu: pip install faiss-cpu")

    index_path = Path(args.index)
    ids_path   = index_path.with_suffix(".ids")
    out_path   = Path(args.out)
    threshold  = args.threshold

    if not index_path.exists():
        log_orch(f"find-dups: index not found ({index_path}) — skipping", level="warning")
        return 0

    index   = faiss.read_index(str(index_path))
    all_ids = ids_path.read_text().splitlines()
    n       = index.ntotal

    if n == 0:
        log_orch("find-dups: empty index")
        return 0

    # Reconstruct stored vectors from the flat index
    d   = index.d
    xb  = faiss.vector_float_to_array(index.get_xb())
    all_vecs = np.array(xb, dtype=np.float32).reshape(n, d)

    log_orch(f"find-dups: {n} vectors, threshold={threshold}")

    existing_dups: set = set()
    if out_path.exists():
        existing_dups = set(out_path.read_text().splitlines())

    k         = min(5, n)
    new_dups: list = []
    kept_set: set  = set()
    batch_size = 4096

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        chunk = all_vecs[start:end]
        D, I  = index.search(chunk, k)

        for local_i in range(end - start):
            global_i = start + local_i
            fid = all_ids[global_i]
            if fid in existing_dups:
                continue
            kept_set.add(global_i)
            for rank in range(1, k):
                neighbor = int(I[local_i, rank])
                if neighbor < 0:
                    break
                if float(D[local_i, rank]) >= threshold:
                    nid = all_ids[neighbor]
                    if nid not in existing_dups and neighbor not in kept_set:
                        existing_dups.add(nid)
                        new_dups.append(nid)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if new_dups:
        with open(out_path, "a") as f:
            for fid in new_dups:
                f.write(fid + "\n")

    log_event("clip_dedup", "dups_found",
              new_dups=len(new_dups), total_dups=len(existing_dups), index_size=n)
    log_orch(f"find-dups: {len(new_dups)} new, {len(existing_dups)} total in blocklist")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap  = argparse.ArgumentParser(description="CLIP deduplication pipeline (V2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_emb = sub.add_parser("embed", help="Compute CLIP embeddings per shard")
    p_emb.add_argument("--shards",     required=True)
    p_emb.add_argument("--embeddings", required=True)
    p_emb.add_argument("--batch-size", dest="batch_size", type=int, default=512)

    p_idx = sub.add_parser("build-index", help="Build/extend cumulative FAISS index")
    p_idx.add_argument("--embeddings", required=True)
    p_idx.add_argument("--index",      required=True)

    p_dup = sub.add_parser("find-dups", help="Find near-duplicates, write blocklist")
    p_dup.add_argument("--index",     required=True)
    p_dup.add_argument("--out",       required=True)
    p_dup.add_argument("--threshold", type=float, default=DUP_THRESHOLD)

    args = ap.parse_args()
    handlers = {
        "embed":       cmd_embed,
        "build-index": cmd_build_index,
        "find-dups":   cmd_find_dups,
    }
    sys.exit(handlers[args.cmd](args))


if __name__ == "__main__":
    main()
