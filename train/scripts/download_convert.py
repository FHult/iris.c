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
import queue
import sys
import tarfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    DATA_ROOT, STAGING_DIR, TRAIN_DIR, LOG_DIR,
    write_heartbeat, log_event, log_orch, load_config, now_iso,
)
from downloader import (
    jdb_tgz_ranges, jdb_tgz_filename, download_jdb_annotation,
    run_wikiart_download, check_laion, check_coyo,
    JDB_REPO_ID, _hf_download_file_guarded as _hf_download_file,
)


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
    """Producer-consumer: download one tgz, signal consumer, consumer converts and deletes."""
    ranges = jdb_tgz_ranges(config, scale)
    start, end = ranges[chunk]
    tgz_range = range(start, end + 1)
    total = len(tgz_range)

    raw_dir = STAGING_DIR / f"chunk{chunk}" / "raw" / "journeydb"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = STAGING_DIR / f"chunk{chunk}" / "converted" / "journeydb"
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel_dir = raw_dir / ".tgz_state"
    sentinel_dir.mkdir(exist_ok=True)

    download_jdb_annotation(raw_dir)

    log_orch(f"JDB chunk {chunk} scale={scale}: download+convert {total} tgzs ({start:03d}–{end:03d})")

    ready_q: queue.Queue = queue.Queue()
    error_event = threading.Event()
    done_event = threading.Event()

    def producer():
        try:
            for i in tgz_range:
                if error_event.is_set():
                    break
                converted = sentinel_dir / f"{i:03d}.converted"
                ready     = sentinel_dir / f"{i:03d}.ready"
                if converted.exists():
                    log_event("download_convert", "skip", chunk=chunk, tgz=i,
                              reason="already_converted")
                    continue
                if not ready.exists():
                    log_event("download_convert", "download_start", chunk=chunk, tgz=i)
                    t0 = time.time()
                    try:
                        _hf_download_file(JDB_REPO_ID, jdb_tgz_filename(i), str(raw_dir))
                        ready.touch()
                        elapsed = time.time() - t0
                        log_event("download_convert", "download_done", chunk=chunk, tgz=i,
                                  elapsed_sec=round(elapsed, 1))
                    except Exception as e:
                        log_event("download_convert", "download_error", chunk=chunk, tgz=i,
                                  error=str(e))
                        error_event.set()
                        raise
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
                tgz_path = raw_dir / "data" / "train" / "imgs" / f"{idx:03d}.tgz"
                converted = sentinel_dir / f"{idx:03d}.converted"
                try:
                    _convert_tgz(tgz_path, out_dir, anno_path, chunk, idx)
                    converted.touch()
                    # Delete raw tgz immediately to free disk
                    if tgz_path.exists():
                        tgz_path.unlink()
                        log_event("download_convert", "raw_deleted", chunk=chunk, tgz=idx)
                except Exception as e:
                    log_event("download_convert", "convert_error", chunk=chunk, tgz=idx,
                              error=str(e))
                    error_event.set()
                    raise
        finally:
            done_event.set()

    def heartbeat_loop():
        hf_cache = raw_dir / ".cache" / "huggingface" / "download"
        while not done_event.is_set():
            done_count = sum(
                1 for i in tgz_range
                if (sentinel_dir / f"{i:03d}.converted").exists()
            )
            from downloader import _incomplete_bytes
            in_flight_gb = round(_incomplete_bytes(hf_cache) / 1e9, 2)
            write_heartbeat("download_convert", chunk=chunk,
                            done=done_count, total=total,
                            pct=round(done_count / total * 100, 1),
                            in_flight_gb=in_flight_gb)
            time.sleep(30)

    prod_thread = threading.Thread(target=producer, daemon=True)
    cons_thread = threading.Thread(target=consumer, daemon=True)
    hb_thread   = threading.Thread(target=heartbeat_loop, daemon=True)

    hb_thread.start()
    prod_thread.start()
    cons_thread.start()

    prod_thread.join()
    cons_thread.join()
    done_event.set()

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
