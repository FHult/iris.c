#!/usr/bin/env python3
"""
train/scripts/download_convert.py — Combined JDB download+convert worker (MLX-19).

Downloads JDB tgzs one at a time (producer thread), converts each one immediately
when ready (consumer thread), deletes raw tgz after convert. Peak disk = ~2 tgzs
in-flight instead of all N simultaneously.

Also downloads WikiArt and verifies LAION/COYO manifest presence.

On success: exits 0. Orchestrator marks both download.done and convert.done.

Usage:
    python train/scripts/download_convert.py --chunk 1 --config train/configs/v2_pipeline.yaml
"""

import argparse
import io
import json
import os
import queue
import sys
import tarfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    DATA_ROOT, STAGING_DIR, TRAIN_DIR, LOG_DIR, COLD_ROOT,
    write_heartbeat, log_event, log_orch, load_config, now_iso,
)
from downloader import (
    jdb_tgz_ranges, jdb_tgz_filename, download_jdb_annotation,
    run_wikiart_download, check_laion, check_coyo,
    JDB_REPO_ID, _hf_download_file_guarded as _hf_download_file,
)
from data_stager import _same_device, _atomic_copy_file


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------

def _pool_link_or_copy(pool_file: Path, staging_file: Path, use_symlinks: bool) -> None:
    """Create staging_file pointing at pool_file (symlink or copy). No-op if exists."""
    if staging_file.exists() or staging_file.is_symlink():
        return
    staging_file.parent.mkdir(parents=True, exist_ok=True)
    if use_symlinks:
        os.symlink(pool_file.resolve(), staging_file)
    else:
        _atomic_copy_file(pool_file, staging_file)


# ---------------------------------------------------------------------------
# Tgz priority ordering
# ---------------------------------------------------------------------------

def _prioritised_tgz_range(start: int, end: int, config: dict) -> list[int]:
    """Return a download-priority-ordered list of tgz indices.

    If cold_root/metadata/tgz_scores.json exists, sort tgzs with the highest
    quality scores first so higher-signal data is available earliest.
    Falls back to sequential order when no scores file is present (cold start).
    """
    sequential = list(range(start, end + 1))

    storage  = config.get("storage", {})
    cold_root = Path(storage.get("cold_root", COLD_ROOT))
    scores_file = cold_root / "metadata" / "tgz_scores.json"
    if not scores_file.exists():
        return sequential

    try:
        data = json.loads(scores_file.read_text())
        tgz_scores_raw = data.get("tgz_scores", {})
        # tgz_scores keys are strings like "12" → {"score": 0.7, ...}
        scores: dict[int, float] = {int(k): v["score"] for k, v in tgz_scores_raw.items()
                                    if isinstance(v, dict) and "score" in v}
    except (ValueError, OSError, KeyError):
        return sequential

    # Tgzs in range: sort scored ones desc by score, append unscored at end (sequential)
    scored = [i for i in sequential if i in scores]
    unscored = [i for i in sequential if i not in scores]
    scored.sort(key=lambda i: scores[i], reverse=True)
    prioritised = scored + unscored

    if prioritised != sequential:
        log_orch(
            f"tgz priority: {len(scored)} scored (top={prioritised[0]:03d} "
            f"score={scores.get(prioritised[0], 0):.4f}), {len(unscored)} unscored",
            tgz_priority="scored",
        )
    else:
        log_orch("tgz priority: sequential (all unscored or scores match range)",
                 tgz_priority="sequential")

    return prioritised


# ---------------------------------------------------------------------------
# JDB converter
# ---------------------------------------------------------------------------

def _load_annotation_index(anno_path: Path) -> dict:
    """Load JDB annotation JSONL.tgz → {stem: caption} mapping."""
    captions: dict = {}
    with tarfile.open(anno_path, "r:gz") as atf:
        for m in atf.getmembers():
            if not m.name.endswith(".jsonl"):
                continue
            with atf.extractfile(m) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        stem = Path(rec["img_path"]).stem
                        cap = (rec.get("Task2", {}) or {}).get("Caption") or rec.get("prompt", "")
                        cap = cap.strip()
                        if cap:
                            captions[stem] = cap
                    except Exception:
                        pass
            break
    return captions


def _convert_tgz(tgz_path: Path, out_dir: Path, anno_path: Path, chunk: int, idx: int) -> None:
    """Extract one JDB tgz and write a WebDataset tar with jpg+txt pairs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log_event("download_convert", "convert_start", chunk=chunk, tgz=idx)

    captions = _load_annotation_index(anno_path)

    out_tar = out_dir / f"{idx:03d}.tar"
    written = 0
    with tarfile.open(tgz_path, "r:gz") as src, tarfile.open(out_tar, "w") as dst:
        for member in src.getmembers():
            if not member.isfile():
                continue
            if not member.name.lower().endswith((".jpg", ".jpeg")):
                continue
            stem = Path(member.name).stem
            caption = captions.get(stem)
            if not caption:
                continue
            img_data = src.extractfile(member).read()
            info_jpg = tarfile.TarInfo(name=f"{stem}.jpg")
            info_jpg.size = len(img_data)
            dst.addfile(info_jpg, io.BytesIO(img_data))
            txt_data = caption.encode()
            info_txt = tarfile.TarInfo(name=f"{stem}.txt")
            info_txt.size = len(txt_data)
            dst.addfile(info_txt, io.BytesIO(txt_data))
            written += 1

    elapsed = time.time() - t0
    print(f"  JDB tgz {idx:03d}: {written} jpg+txt pairs → {out_tar}", flush=True)
    log_event("download_convert", "convert_done", chunk=chunk, tgz=idx,
              elapsed_sec=round(elapsed, 1), written=written)


def run_jdb_download_convert(chunk: int, config: dict, scale: str = "all-in") -> None:
    """Producer-consumer: download one tgz, signal consumer, consumer converts and deletes.

    Three-level cache hierarchy when pool keys are configured in storage:
      Level 0: converted pool hit  → symlink/copy .tar to staging; skip download+conversion
      Level 1: raw pool hit        → symlink/copy .tgz to staging; convert; write to conv pool
      Level 2: no pool hit         → download to raw pool (or staging); convert; write to conv pool
    """
    ranges = jdb_tgz_ranges(config, scale)
    start, end = ranges[chunk]
    tgz_range = _prioritised_tgz_range(start, end, config)
    total = len(tgz_range)

    raw_dir = STAGING_DIR / f"chunk{chunk}" / "raw" / "journeydb"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = STAGING_DIR / f"chunk{chunk}" / "converted" / "journeydb"
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel_dir = raw_dir / ".tgz_state"
    sentinel_dir.mkdir(exist_ok=True)

    # --- Pool configuration ---
    storage = config.get("storage", {})
    raw_pool_dir  = Path(storage["raw_pool_root"])       if "raw_pool_root"       in storage else None
    conv_pool_dir = Path(storage["converted_pool_root"]) if "converted_pool_root" in storage else None

    raw_pool_enabled  = raw_pool_dir is not None
    conv_pool_enabled = conv_pool_dir is not None

    if raw_pool_enabled:
        raw_pool_sentinels = raw_pool_dir / ".downloaded"
        raw_pool_sentinels.mkdir(parents=True, exist_ok=True)
        # Determine transfer mode: symlink (same device) or copy (cross-device).
        # raw_dir may not exist yet on first run; fall back to parent or DATA_ROOT.
        _ref_raw = raw_dir if raw_dir.exists() else raw_dir.parent
        use_symlinks_raw = _same_device(raw_pool_dir, _ref_raw) if raw_pool_dir.exists() and _ref_raw.exists() else False

    if conv_pool_enabled:
        conv_pool_sentinels = conv_pool_dir / ".converted"
        conv_pool_sentinels.mkdir(parents=True, exist_ok=True)
        _ref_out = out_dir if out_dir.exists() else out_dir.parent
        use_symlinks_conv = _same_device(conv_pool_dir, _ref_out) if conv_pool_dir.exists() and _ref_out.exists() else False

    # --- Level 0: resolve converted pool hits synchronously before threads start ---
    already_done: set = set()
    if conv_pool_enabled:
        for i in tgz_range:
            converted_sentinel = sentinel_dir / f"{i:03d}.converted"
            if converted_sentinel.exists():
                already_done.add(i)
                continue
            pool_tar      = conv_pool_dir / f"{i:03d}.tar"
            pool_sentinel = conv_pool_sentinels / f"{i:03d}"
            if pool_sentinel.exists() and pool_tar.exists():
                staging_tar = out_dir / f"{i:03d}.tar"
                _pool_link_or_copy(pool_tar, staging_tar, use_symlinks_conv)
                converted_sentinel.touch()
                already_done.add(i)
                log_event("download_convert", "conv_pool_hit", chunk=chunk, tgz=i)

    remaining = [i for i in tgz_range if i not in already_done]

    # --- Annotation: download to raw pool (or staging), symlink into staging if needed ---
    anno_target = raw_pool_dir if raw_pool_enabled else raw_dir
    download_jdb_annotation(anno_target)
    if raw_pool_enabled:
        anno_pool    = raw_pool_dir / "data" / "train" / "train_anno_realease_repath.jsonl.tgz"
        anno_staging = raw_dir     / "data" / "train" / "train_anno_realease_repath.jsonl.tgz"
        _pool_link_or_copy(anno_pool, anno_staging, use_symlinks_raw)

    log_orch(f"JDB chunk {chunk} scale={scale}: {len(already_done)} conv-pool hits; "
             f"download+convert {len(remaining)} tgzs ({start:03d}–{end:03d})")

    ready_q: queue.Queue = queue.Queue()
    error_event = threading.Event()
    done_event  = threading.Event()
    cur_tgz     = [None]  # shared: producer writes, heartbeat reads

    def producer():
        try:
            for i in remaining:
                if error_event.is_set():
                    break
                ready = sentinel_dir / f"{i:03d}.ready"

                if raw_pool_enabled:
                    pool_tgz     = raw_pool_dir / "data" / "train" / "imgs" / f"{i:03d}.tgz"
                    pool_sentinel = raw_pool_sentinels / f"{i:03d}"
                    if not pool_sentinel.exists():
                        cur_tgz[0] = i
                        log_event("download_convert", "download_start", chunk=chunk, tgz=i)
                        t0 = time.time()
                        try:
                            _hf_download_file(JDB_REPO_ID, jdb_tgz_filename(i), str(raw_pool_dir))
                            pool_sentinel.touch()
                            elapsed = time.time() - t0
                            log_event("download_convert", "download_done", chunk=chunk, tgz=i,
                                      elapsed_sec=round(elapsed, 1))
                        except Exception as e:
                            log_event("download_convert", "download_error", chunk=chunk, tgz=i,
                                      error=str(e))
                            error_event.set()
                            raise
                    tgz_staging = raw_dir / "data" / "train" / "imgs" / f"{i:03d}.tgz"
                    _pool_link_or_copy(pool_tgz, tgz_staging, use_symlinks_raw)
                else:
                    if not ready.exists():
                        cur_tgz[0] = i
                        log_event("download_convert", "download_start", chunk=chunk, tgz=i)
                        t0 = time.time()
                        try:
                            _hf_download_file(JDB_REPO_ID, jdb_tgz_filename(i), str(raw_dir))
                            elapsed = time.time() - t0
                            log_event("download_convert", "download_done", chunk=chunk, tgz=i,
                                      elapsed_sec=round(elapsed, 1))
                        except Exception as e:
                            log_event("download_convert", "download_error", chunk=chunk, tgz=i,
                                      error=str(e))
                            error_event.set()
                            raise

                cur_tgz[0] = None
                ready.touch()
                ready_q.put(i)
        finally:
            ready_q.put(None)  # sentinel to stop consumer

    def consumer():
        anno_path = raw_dir / "data" / "train" / "train_anno_realease_repath.jsonl.tgz"
        try:
            while True:
                idx = ready_q.get()
                if idx is None:
                    break
                if error_event.is_set():
                    continue
                tgz_staging = raw_dir / "data" / "train" / "imgs" / f"{idx:03d}.tgz"
                converted   = sentinel_dir / f"{idx:03d}.converted"
                staging_tar = out_dir / f"{idx:03d}.tar"
                try:
                    _convert_tgz(tgz_staging, out_dir, anno_path, chunk, idx)
                    converted.touch()

                    # Write-through to converted pool
                    if conv_pool_enabled:
                        pool_tar     = conv_pool_dir / f"{idx:03d}.tar"
                        conv_sentinel = conv_pool_sentinels / f"{idx:03d}"
                        if not pool_tar.exists():
                            pool_tar.parent.mkdir(parents=True, exist_ok=True)
                            if use_symlinks_conv:
                                # Same device: move tar to pool; replace staging with symlink
                                os.replace(staging_tar, pool_tar)
                                os.symlink(pool_tar.resolve(), staging_tar)
                            else:
                                _atomic_copy_file(staging_tar, pool_tar)
                        conv_sentinel.touch()
                        log_event("download_convert", "conv_pool_write", chunk=chunk, tgz=idx)

                    # Remove raw tgz staging path; never delete pool file
                    if tgz_staging.is_symlink() or tgz_staging.exists():
                        tgz_staging.unlink()
                        log_event("download_convert",
                                  "staging_unlinked" if raw_pool_enabled else "raw_deleted",
                                  chunk=chunk, tgz=idx)
                except Exception as e:
                    log_event("download_convert", "convert_error", chunk=chunk, tgz=idx,
                              error=str(e))
                    error_event.set()
                    raise
        finally:
            done_event.set()

    def heartbeat_loop():
        from downloader import _incomplete_bytes
        hf_cache   = (raw_pool_dir if raw_pool_enabled else raw_dir) / ".cache" / "huggingface" / "download"
        prev_bytes = 0
        prev_ts    = time.time()
        while not done_event.is_set():
            done_count = sum(1 for i in tgz_range
                             if (sentinel_dir / f"{i:03d}.converted").exists())
            now_bytes  = _incomplete_bytes(hf_cache)
            now_ts     = time.time()
            dt         = now_ts - prev_ts
            delta      = now_bytes - prev_bytes
            speed_mbps = round(delta / 1e6 / dt, 1) if dt > 0 and delta > 0 else 0.0
            prev_bytes, prev_ts = now_bytes, now_ts
            write_heartbeat("download_convert", chunk=chunk,
                            done=done_count, total=total,
                            pct=round(done_count / total * 100, 1),
                            in_flight_gb=round(now_bytes / 1e9, 2),
                            current_tgz=cur_tgz[0],
                            dl_speed_mbps=speed_mbps)
            time.sleep(30)

    if remaining:
        prod_thread = threading.Thread(target=producer, daemon=True)
        cons_thread = threading.Thread(target=consumer, daemon=True)
        hb_thread   = threading.Thread(target=heartbeat_loop, daemon=True)

        hb_thread.start()
        prod_thread.start()
        cons_thread.start()

        prod_thread.join()
        cons_thread.join()
        done_event.set()
        hb_thread.join(timeout=5)

        if error_event.is_set():
            raise RuntimeError(f"JDB download+convert failed for chunk {chunk}")

    done_count = sum(1 for i in tgz_range if (sentinel_dir / f"{i:03d}.converted").exists())
    write_heartbeat("download_convert", chunk=chunk, done=done_count, total=total, pct=100)
    log_orch(f"JDB chunk {chunk}: all {total} tgzs downloaded and converted")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="V2 combined download+convert worker")
    ap.add_argument("--chunk",    type=int, required=True)
    ap.add_argument("--config",   default=str(TRAIN_DIR / "configs" / "v2_pipeline.yaml"))
    ap.add_argument("--scale",    default="all-in",
                    choices=["smoke", "small", "medium", "large", "all-in"])
    ap.add_argument("--jdb-only", action="store_true")
    ap.add_argument("--no-jdb",   action="store_true")
    args = ap.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        config = {}

    chunk = args.chunk
    scale = args.scale
    log_orch(f"download_convert starting for chunk {chunk} scale={scale}")

    if not args.no_jdb:
        run_jdb_download_convert(chunk, config, scale)

    if not args.jdb_only:
        run_wikiart_download(chunk, config)
        check_laion(config)
        check_coyo(config)

    log_orch(f"download_convert complete for chunk {chunk}")


if __name__ == "__main__":
    main()
