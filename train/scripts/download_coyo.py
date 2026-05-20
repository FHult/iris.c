#!/usr/bin/env python3
"""
train/scripts/download_coyo.py — Standalone COYO image downloader.

Fetches a COYO-shaped parquet metadata repo from HuggingFace, downloads the
referenced images, and packs them into WebDataset tar shards under the
project cold pool. Self-contained (no pipeline_lib dependency), resumable
across crashes, and runnable as a detached daemon.

Pipeline:
  1. metadata — stream the HF parquet(s), filter (image ≥256px, caption
     ≥5 words), insert into a SQLite index at
     COLD_ROOT/coyo/coyo_index.db.
  2. download — for every row still marked 'pending', fetch the image via
     img2dataset (subprocess) if available, otherwise a ThreadPoolExecutor
     HTTP loop. Successful jpegs land at COLD_ROOT/coyo/raw_images/{id}.jpg
     and the DB is updated.
  3. pack — read 'downloaded' rows in id order, pack 10_000 image+caption
     pairs per tar at COLD_ROOT/raw/coyo/shard-NNNNNN.tar (atomic .tar.tmp
     → rename). Existing finished shards are skipped.

Output WDS layout matches download_laion_coyo.py: each shard contains
paired {id}.jpg + {id}.txt entries, ready to be consumed by build_shards.py
which already classifies any source dir containing "coyo" as type=coyo.

Usage:
    train/.venv/bin/python train/scripts/download_coyo.py            # full run (sequential)
    train/.venv/bin/python train/scripts/download_coyo.py --parallel # metadata + download in parallel
    train/.venv/bin/python train/scripts/download_coyo.py --phase metadata
    train/.venv/bin/python train/scripts/download_coyo.py --phase download          # one-shot
    train/.venv/bin/python train/scripts/download_coyo.py --phase download --follow # poll until metadata_done
    train/.venv/bin/python train/scripts/download_coyo.py --phase pack
    train/.venv/bin/python train/scripts/download_coyo.py --detach   # daemonize

Heartbeat: /Volumes/2TBSSD/.heartbeat/download_coyo.json (count/total/rate).
Log when --detach: /tmp/download_coyo.log
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HF_REPO    = "recastai/coyo-10m-aesthetic"
DEFAULT_COLD_ROOT  = Path("/Volumes/16TBCold")
HEARTBEAT_PATH     = Path("/Volumes/2TBSSD/.heartbeat/download_coyo.json")
DAEMON_LOG_PATH    = Path("/tmp/download_coyo.log")
DEFAULT_WORKERS    = 8
DEFAULT_SHARD_SIZE = 10_000

MIN_IMAGE_PX      = 256
MIN_CAPTION_WORDS = 5
HTTP_TIMEOUT      = 20
HTTP_UA           = "Mozilla/5.0 (Macintosh) iris-coyo-fetcher/1.0"
COMMIT_BATCH      = 500
SUBMIT_CHUNK      = 2_000   # futures submitted at a time; limits peak memory
LOG_INTERVAL      = 500     # completions between progress log lines
FAIL_RATE_WARN    = 0.40    # warn when sustained fail rate exceeds this fraction
FOLLOW_BATCH      = 10_000  # max pending rows fetched per iteration in follow mode
FOLLOW_POLL_S     = 30      # seconds to wait between polls when pending is empty

URL_COLS    = ("url", "URL", "image_url", "img_url")
CAP_COLS    = ("caption", "text", "TEXT", "long_caption", "description")
WIDTH_COLS  = ("width", "image_width", "w")
HEIGHT_COLS = ("height", "image_height", "h")
ID_COLS     = ("id", "image_id", "sample_id")

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY,
    url         TEXT NOT NULL,
    caption     TEXT NOT NULL,
    width       INTEGER,
    height      INTEGER,
    status      TEXT NOT NULL DEFAULT 'pending',
    local_path  TEXT,
    error       TEXT,
    updated_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
"""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def write_heartbeat(**fields) -> None:
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "process": "download_coyo"}
    data.update(fields)
    tmp = HEARTBEAT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(HEARTBEAT_PATH)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def cold_paths(cold_root: Path) -> dict:
    base    = cold_root / "coyo"
    raw_img = base / "raw_images"
    db      = base / "coyo_index.db"
    shards  = cold_root / "raw" / "coyo"
    return {"base": base, "raw_images": raw_img, "db": db, "shards": shards}


def ensure_dirs(paths: dict) -> None:
    for key in ("base", "raw_images", "shards"):
        paths[key].mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=60.0)
    conn.executescript(SQL_SCHEMA)
    # WAL + sync NORMAL is the right trade-off for many small commits.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.commit()
    return conn


def db_counts(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM images GROUP BY status"
    ).fetchall()
    out = {"pending": 0, "downloaded": 0, "failed": 0, "total": 0}
    for status, n in rows:
        out[status] = n
        out["total"] += n
    return out


# ---------------------------------------------------------------------------
# Phase 1 — metadata
# ---------------------------------------------------------------------------

def _pick(colnames: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    s = set(colnames)
    for c in candidates:
        if c in s:
            return c
    return None


def _hash_id(url: str) -> int:
    # Stable 63-bit positive hash so it fits SQLite INTEGER PRIMARY KEY.
    import hashlib
    h = hashlib.blake2b(url.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & 0x7FFFFFFFFFFFFFFF


def phase_metadata(hf_repo: str, paths: dict,
                   max_parquets: Optional[int] = None,
                   min_aesthetic: Optional[float] = None) -> None:
    """Stream parquet files from `hf_repo`, filter, insert into SQLite."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise SystemExit(f"huggingface_hub is required: {e}")
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit(f"pyarrow is required: {e}")

    log(f"metadata: discovering parquet files in {hf_repo}")
    files = [f for f in list_repo_files(hf_repo, repo_type="dataset")
             if f.endswith(".parquet")]
    if not files:
        raise SystemExit(f"No parquet files found in {hf_repo}")
    files.sort()
    if max_parquets is not None:
        files = files[:max_parquets]
    log(f"metadata: {len(files)} parquet file(s) to scan")

    conn = open_db(paths["db"])
    existing = db_counts(conn)
    log(f"metadata: DB has {existing['total']:,} rows already "
        f"(pending={existing['pending']:,}, downloaded={existing['downloaded']:,}, "
        f"failed={existing['failed']:,})")

    total_seen = 0
    total_kept = 0
    total_skipped = 0
    total_dup = 0
    t0 = time.monotonic()
    last_hb_time = t0

    for fi, fname in enumerate(files):
        log(f"metadata: [{fi+1}/{len(files)}] downloading {fname}")
        try:
            local = hf_hub_download(repo_id=hf_repo, filename=fname,
                                    repo_type="dataset")
        except Exception as e:
            log(f"metadata: WARN failed to download {fname}: {e}")
            continue

        try:
            pf = pq.ParquetFile(local)
        except Exception as e:
            log(f"metadata: WARN failed to open {fname}: {e}")
            continue

        names = pf.schema.names
        url_c    = _pick(names, URL_COLS)
        cap_c    = _pick(names, CAP_COLS)
        width_c  = _pick(names, WIDTH_COLS)
        height_c = _pick(names, HEIGHT_COLS)
        id_c     = _pick(names, ID_COLS)
        aes_c    = "aesthetic_score_laion_v2" if "aesthetic_score_laion_v2" in names else None
        if url_c is None or cap_c is None:
            log(f"metadata: WARN {fname} has no url/caption column "
                f"(found: {names}) — skipping")
            continue
        if min_aesthetic is not None and aes_c is None:
            log(f"metadata: WARN {fname} has no aesthetic_score_laion_v2 column "
                f"— all rows will be skipped under --min-aesthetic")

        cols = [c for c in (id_c, url_c, cap_c, width_c, height_c, aes_c) if c is not None]

        batch_rows: list[tuple] = []
        for batch in pf.iter_batches(batch_size=8192, columns=cols):
            d = batch.to_pydict()
            urls = d[url_c]
            caps = d[cap_c]
            ws   = d[width_c]  if width_c  else [None] * len(urls)
            hs   = d[height_c] if height_c else [None] * len(urls)
            ids  = d[id_c]     if id_c     else [None] * len(urls)
            aess = d[aes_c]    if aes_c    else [None] * len(urls)
            n = len(urls)
            total_seen += n
            for i in range(n):
                url = urls[i]
                cap = caps[i]
                if not url or not cap:
                    total_skipped += 1
                    continue
                aes = aess[i]
                if min_aesthetic is not None and (aes is None or aes < min_aesthetic):
                    total_skipped += 1
                    continue
                try:
                    w = int(ws[i]) if ws[i] is not None else 0
                    h = int(hs[i]) if hs[i] is not None else 0
                except (TypeError, ValueError):
                    w = h = 0
                if width_c and height_c and (w < MIN_IMAGE_PX or h < MIN_IMAGE_PX):
                    total_skipped += 1
                    continue
                cap_str = str(cap).strip()
                if len(cap_str.split()) < MIN_CAPTION_WORDS:
                    total_skipped += 1
                    continue
                try:
                    rid = int(ids[i]) if ids[i] is not None else _hash_id(str(url))
                except (TypeError, ValueError):
                    rid = _hash_id(str(url))
                batch_rows.append((rid, str(url), cap_str, w or None, h or None))

            if len(batch_rows) >= COMMIT_BATCH:
                kept, dup = _insert_metadata_batch(conn, batch_rows)
                total_kept += kept
                total_dup  += dup
                batch_rows.clear()

            now = time.monotonic()
            if now - last_hb_time > 30:
                rate = total_seen / max(now - t0, 1e-3)
                write_heartbeat(phase="metadata", file=fname,
                                seen=total_seen, kept=total_kept,
                                skipped=total_skipped, dup=total_dup,
                                rate_per_sec=round(rate, 1))
                log(f"metadata: seen={total_seen:,} kept={total_kept:,} "
                    f"skipped={total_skipped:,} dup={total_dup:,} "
                    f"({rate:.0f}/s)")
                last_hb_time = now

        if batch_rows:
            kept, dup = _insert_metadata_batch(conn, batch_rows)
            total_kept += kept
            total_dup  += dup
            batch_rows.clear()

    final = db_counts(conn)
    write_heartbeat(phase="metadata", done=True,
                    seen=total_seen, kept=total_kept,
                    skipped=total_skipped, dup=total_dup,
                    db_total=final["total"], db_pending=final["pending"])
    log(f"metadata: complete — seen={total_seen:,} kept={total_kept:,} "
        f"skipped={total_skipped:,} dup={total_dup:,}")
    log(f"metadata: DB totals — total={final['total']:,} pending={final['pending']:,} "
        f"downloaded={final['downloaded']:,} failed={final['failed']:,}")
    conn.close()
    # Signal that all parquet files have been scanned; follow-mode download uses this.
    try:
        (paths["base"] / "metadata_done").touch()
    except OSError:
        pass


def _insert_metadata_batch(conn: sqlite3.Connection,
                           rows: list[tuple]) -> tuple[int, int]:
    """Insert a batch with INSERT OR IGNORE. Returns (inserted, duplicates)."""
    before = conn.total_changes
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO images (id, url, caption, width, height, "
            "status, updated_at) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            [(r[0], r[1], r[2], r[3], r[4], time.time()) for r in rows],
        )
    inserted = conn.total_changes - before
    return inserted, len(rows) - inserted


# ---------------------------------------------------------------------------
# Phase 2 — download
# ---------------------------------------------------------------------------

def _img2dataset_available() -> bool:
    return shutil.which("img2dataset") is not None


def _mem_gb() -> str:
    """Return a brief memory usage string, or '' if psutil unavailable."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return f" [RAM used={vm.used/1e9:.1f}GB avail={vm.available/1e9:.1f}GB]"
    except Exception:
        return ""


def phase_download(paths: dict, workers: int) -> None:
    conn = open_db(paths["db"])
    pending = conn.execute(
        "SELECT id, url FROM images WHERE status='pending'"
    ).fetchall()
    if not pending:
        log("download: nothing pending")
        conn.close()
        return

    total = db_counts(conn)["total"]
    log(f"download: {len(pending):,} pending / {total:,} total{_mem_gb()}")
    if workers > 8:
        log(f"download: WARNING — {workers} workers is aggressive; "
            f"consider --workers 8 to reduce OOM risk")

    raw_dir = paths["raw_images"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    use_i2d = _img2dataset_available()
    log(f"download: backend = {'img2dataset' if use_i2d else 'urllib threadpool'} "
        f"workers={workers} submit_chunk={SUBMIT_CHUNK}")

    state = {"done": 0, "ok": 0, "fail": 0, "t0": time.monotonic(),
             "total": len(pending)}
    stop_ev = threading.Event()

    def _hb_and_log():
        while not stop_ev.wait(60):
            elapsed = max(time.monotonic() - state["t0"], 1e-3)
            rate = state["done"] / elapsed
            eta = (state["total"] - state["done"]) / rate if rate > 0 else 0
            fail_rate = state["fail"] / max(state["done"], 1)
            write_heartbeat(
                phase="download", backend="img2dataset" if use_i2d else "urllib",
                done=state["done"], total=state["total"],
                ok=state["ok"], fail=state["fail"],
                fail_rate=round(fail_rate, 3),
                pct=round(state["done"] * 100.0 / max(state["total"], 1), 2),
                rate_per_sec=round(rate, 2),
                eta_sec=int(eta),
            )
            warn = f" *** FAIL RATE {fail_rate:.0%}" if fail_rate > FAIL_RATE_WARN else ""
            log(f"download: heartbeat — {state['done']:,}/{state['total']:,} "
                f"ok={state['ok']:,} fail={state['fail']:,} "
                f"({rate:.1f}/s ETA {int(eta)//3600}h{(int(eta)%3600)//60}m)"
                f"{warn}{_mem_gb()}")

    # Write initial heartbeat immediately so pipeline_status shows the phase.
    write_heartbeat(phase="download", backend="img2dataset" if use_i2d else "urllib",
                    done=0, total=len(pending), ok=0, fail=0,
                    pct=0.0, rate_per_sec=0.0, eta_sec=0)

    hb = threading.Thread(target=_hb_and_log, daemon=True)
    hb.start()

    try:
        if use_i2d:
            _download_img2dataset(conn, pending, raw_dir, workers, state)
        else:
            _download_urllib(conn, pending, raw_dir, workers, state)
    except Exception as exc:
        log(f"download: FATAL — phase_download raised: {exc}")
        raise
    finally:
        stop_ev.set()
        conn.commit()
        conn.close()

    final = db_counts(open_db(paths["db"]))
    write_heartbeat(phase="download", done=True,
                    ok=state["ok"], fail=state["fail"],
                    db_downloaded=final["downloaded"], db_failed=final["failed"],
                    db_pending=final["pending"])
    log(f"download: complete — ok={state['ok']:,} fail={state['fail']:,} "
        f"(db_pending now {final['pending']:,}){_mem_gb()}")


def _update_row(conn: sqlite3.Connection, lock: threading.Lock,
                rid: int, status: str, local_path: Optional[str],
                error: Optional[str]) -> None:
    with lock:
        conn.execute(
            "UPDATE images SET status=?, local_path=?, error=?, updated_at=? "
            "WHERE id=?",
            (status, local_path, error, time.time(), rid),
        )


def _fetch_url(url: str) -> bytes:
    from urllib.request import Request, urlopen
    req = Request(url, headers={"User-Agent": HTTP_UA})
    with urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return r.read()


def _save_jpeg(raw: bytes, dst: Path) -> Optional[str]:
    """Decode + re-encode to JPEG; return None on success, error string on failure."""
    try:
        from PIL import Image
    except ImportError:
        return "Pillow not installed"
    try:
        import warnings
        img = Image.open(io.BytesIO(raw))
        img.load()
        if img.mode not in ("RGB", "L", "RGBA"):
            # Suppress the palette-transparency UserWarning: it's expected for P-mode images
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w < MIN_IMAGE_PX or h < MIN_IMAGE_PX:
            return f"too small ({w}x{h})"
        tmp = dst.with_suffix(".jpg.tmp")
        img.save(tmp, format="JPEG", quality=90)
        tmp.replace(dst)
        return None
    except Exception as e:
        return f"decode/encode failed: {e}"


def _download_one(rid: int, url: str, raw_dir: Path) -> tuple[int, str, Optional[str], Optional[str]]:
    dst = raw_dir / f"{rid}.jpg"
    if dst.exists() and dst.stat().st_size > 0:
        return rid, "downloaded", str(dst), None
    try:
        raw = _fetch_url(url)
    except Exception as e:
        return rid, "failed", None, f"http: {e}"
    err = _save_jpeg(raw, dst)
    if err:
        return rid, "failed", None, err
    return rid, "downloaded", str(dst), None


def _download_urllib(conn: sqlite3.Connection,
                     pending: list[tuple],
                     raw_dir: Path,
                     workers: int,
                     state: dict) -> None:
    lock = threading.Lock()
    batch: list[tuple] = []
    recent_errors: list[str] = []  # rolling window of recent error messages
    last_log_done = 0

    def _flush(force: bool = False) -> None:
        if not batch:
            return
        if not force and len(batch) < COMMIT_BATCH:
            return
        with lock:
            with conn:
                conn.executemany(
                    "UPDATE images SET status=?, local_path=?, error=?, "
                    "updated_at=? WHERE id=?",
                    batch,
                )
        batch.clear()

    # Submit in fixed-size chunks to keep peak Future/memory bounded.
    # Submitting all 2M+ futures at once allocates hundreds of MB upfront
    # and was the likely cause of prior OOM crashes.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk_start in range(0, len(pending), SUBMIT_CHUNK):
            chunk = pending[chunk_start:chunk_start + SUBMIT_CHUNK]
            futures = [pool.submit(_download_one, rid, url, raw_dir)
                       for rid, url in chunk]
            for f in as_completed(futures):
                try:
                    rid, status, local_path, error = f.result()
                except Exception as exc:
                    log(f"download: unhandled worker exception: {exc}")
                    state["done"] += 1
                    state["fail"] += 1
                    continue
                batch.append((status, local_path, error, time.time(), rid))
                state["done"] += 1
                if status == "downloaded":
                    state["ok"] += 1
                else:
                    state["fail"] += 1
                    if error:
                        recent_errors.append(error)
                        if len(recent_errors) > 30:
                            recent_errors.pop(0)
                _flush()

                if state["done"] - last_log_done >= LOG_INTERVAL:
                    elapsed = max(time.monotonic() - state["t0"], 1e-3)
                    rate = state["done"] / elapsed
                    eta_s = int((state["total"] - state["done"]) / rate) if rate > 0 else 0
                    fail_rate = state["fail"] / max(state["done"], 1)
                    warn = f" *** HIGH FAIL RATE {fail_rate:.0%}" if fail_rate > FAIL_RATE_WARN else ""
                    log(f"download: {state['done']:,}/{state['total']:,} "
                        f"ok={state['ok']:,} fail={state['fail']:,} "
                        f"({rate:.0f}/s ETA {eta_s//3600}h{(eta_s%3600)//60}m)"
                        f"{warn}")
                    if fail_rate > FAIL_RATE_WARN and recent_errors:
                        samples = list(dict.fromkeys(recent_errors[-5:]))[:3]
                        log(f"download: recent error samples: "
                            + " | ".join(samples))
                    last_log_done = state["done"]

    _flush(force=True)


def _download_img2dataset(conn: sqlite3.Connection,
                          pending: list[tuple],
                          raw_dir: Path,
                          workers: int,
                          state: dict) -> None:
    """Run img2dataset in chunks so a crash mid-run doesn't lose all progress."""
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas required for img2dataset backend")

    CHUNK = 100_000
    for start in range(0, len(pending), CHUNK):
        chunk = pending[start:start + CHUNK]
        log(f"download/i2d: chunk {start:,}–{start + len(chunk):,} of {len(pending):,}")
        results = _img2dataset_one_chunk(chunk, raw_dir, workers)
        batch = []
        for rid, (status, local_path, error) in results.items():
            batch.append((status, local_path, error, time.time(), rid))
            state["done"] += 1
            if status == "downloaded":
                state["ok"] += 1
            else:
                state["fail"] += 1
        with conn:
            conn.executemany(
                "UPDATE images SET status=?, local_path=?, error=?, "
                "updated_at=? WHERE id=?",
                batch,
            )


def _img2dataset_one_chunk(rows: list[tuple],
                           raw_dir: Path,
                           workers: int) -> dict[int, tuple[str, Optional[str], Optional[str]]]:
    import pandas as pd

    results: dict[int, tuple[str, Optional[str], Optional[str]]] = {}
    with tempfile.TemporaryDirectory(prefix="coyo_i2d_") as tmpdir:
        tmp = Path(tmpdir)
        url_list = tmp / "urls.parquet"
        out_dir  = tmp / "out"

        df = pd.DataFrame({
            "id":  [r[0] for r in rows],
            "url": [r[1] for r in rows],
        })
        df.to_parquet(url_list, index=False)

        cmd = [
            "img2dataset",
            "--url_list", str(url_list),
            "--input_format", "parquet",
            "--url_col", "url",
            "--output_folder", str(out_dir),
            "--output_format", "files",
            "--processes_count", "1",
            "--thread_count", str(workers),
            "--image_size", str(MIN_IMAGE_PX),
            "--resize_mode", "no",
            "--encode_format", "jpg",
            "--encode_quality", "90",
            "--save_additional_columns", '["id"]',
            "--enable_wandb", "False",
            "--retries", "1",
            "--timeout", str(HTTP_TIMEOUT),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            log(f"download/i2d: subprocess failed rc={proc.returncode}")
            if proc.stderr:
                log(f"download/i2d stderr: {proc.stderr[-2000:]}")
            for r in rows:
                results[r[0]] = ("failed", None, "img2dataset subprocess failed")
            return results

        for meta_path in sorted(out_dir.rglob("*.parquet")):
            try:
                meta = pd.read_parquet(meta_path)
            except Exception as e:
                log(f"download/i2d: cannot read {meta_path}: {e}")
                continue
            shard_dir = meta_path.parent
            for _, row in meta.iterrows():
                try:
                    rid = int(row["id"])
                except (KeyError, ValueError, TypeError):
                    continue
                status = str(row.get("status", "")).lower()
                if status == "success":
                    key = row.get("key", "")
                    src_jpg = shard_dir / f"{key}.jpg"
                    if not src_jpg.exists():
                        results[rid] = ("failed", None,
                                        "img2dataset success but file missing")
                        continue
                    dst = raw_dir / f"{rid}.jpg"
                    try:
                        shutil.move(str(src_jpg), str(dst))
                    except OSError as e:
                        results[rid] = ("failed", None, f"move: {e}")
                        continue
                    results[rid] = ("downloaded", str(dst), None)
                else:
                    err = row.get("error_message") or row.get("status") or "unknown"
                    results[rid] = ("failed", None, str(err))

        for r in rows:
            results.setdefault(r[0], ("failed", None, "no img2dataset output"))
    return results


# ---------------------------------------------------------------------------
# Phase 2b — follow-mode download (for parallel / continuous operation)
# ---------------------------------------------------------------------------

def phase_download_follow(paths: dict, workers: int,
                          meta_done_ev: Optional[threading.Event] = None) -> None:
    """Download in follow mode: poll the DB for pending rows in batches.

    Runs until the DB has no pending rows AND metadata is complete.
    Metadata completion is detected via either a threading.Event (parallel mode)
    or the sentinel file `paths["base"]/metadata_done` (standalone follow mode).
    """
    meta_sentinel = paths["base"] / "metadata_done"
    conn = open_db(paths["db"])
    raw_dir = paths["raw_images"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    use_i2d = _img2dataset_available()
    log(f"download/follow: starting — backend={'img2dataset' if use_i2d else 'urllib'} "
        f"workers={workers} batch={FOLLOW_BATCH:,}")

    state = {"done": 0, "ok": 0, "fail": 0, "t0": time.monotonic(), "total": 0}
    stop_ev = threading.Event()

    def _hb_and_log():
        while not stop_ev.wait(60):
            elapsed = max(time.monotonic() - state["t0"], 1e-3)
            rate = state["done"] / elapsed
            fail_rate = state["fail"] / max(state["done"], 1)
            write_heartbeat(
                phase="download_follow",
                backend="img2dataset" if use_i2d else "urllib",
                done=state["done"], ok=state["ok"], fail=state["fail"],
                fail_rate=round(fail_rate, 3),
                rate_per_sec=round(rate, 2),
            )
            warn = f" *** FAIL RATE {fail_rate:.0%}" if fail_rate > FAIL_RATE_WARN else ""
            log(f"download/follow: {state['done']:,} done "
                f"ok={state['ok']:,} fail={state['fail']:,} "
                f"({rate:.1f}/s){warn}{_mem_gb()}")

    write_heartbeat(phase="download_follow", backend="img2dataset" if use_i2d else "urllib",
                    done=0, ok=0, fail=0, rate_per_sec=0.0)
    hb = threading.Thread(target=_hb_and_log, daemon=True)
    hb.start()

    try:
        while True:
            batch = conn.execute(
                "SELECT id, url FROM images WHERE status='pending' LIMIT ?",
                (FOLLOW_BATCH,)
            ).fetchall()

            if not batch:
                meta_done = (
                    meta_sentinel.exists()
                    or (meta_done_ev is not None and meta_done_ev.is_set())
                )
                if not meta_done:
                    log(f"download/follow: no pending rows yet; sleeping {FOLLOW_POLL_S}s ...")
                    time.sleep(FOLLOW_POLL_S)
                    continue
                # Metadata finished — one final query to catch rows inserted between
                # the empty fetch above and the done signal (race window).
                batch = conn.execute(
                    "SELECT id, url FROM images WHERE status='pending' LIMIT ?",
                    (FOLLOW_BATCH,)
                ).fetchall()
                if not batch:
                    log("download/follow: no pending rows and metadata complete — done")
                    break
                # Found rows in the drain pass; fall through to download them.

            state["total"] += len(batch)
            log(f"download/follow: got {len(batch):,} pending rows "
                f"(cumulative total {state['total']:,}){_mem_gb()}")
            if use_i2d:
                _download_img2dataset(conn, batch, raw_dir, workers, state)
            else:
                _download_urllib(conn, batch, raw_dir, workers, state)
    finally:
        stop_ev.set()
        conn.commit()
        conn.close()

    final = db_counts(open_db(paths["db"]))
    write_heartbeat(phase="download_follow", done=True,
                    ok=state["ok"], fail=state["fail"],
                    db_downloaded=final["downloaded"], db_failed=final["failed"],
                    db_pending=final["pending"])
    log(f"download/follow: complete — ok={state['ok']:,} fail={state['fail']:,} "
        f"(db_pending now {final['pending']:,}){_mem_gb()}")


def phase_all_parallel(hf_repo: str, paths: dict, workers: int,
                       max_parquets: Optional[int],
                       min_aesthetic: Optional[float]) -> None:
    """Run metadata and download concurrently.

    Metadata thread streams HF parquet files and inserts pending rows.
    Download thread polls the DB for pending rows in FOLLOW_BATCH-sized batches.
    Both threads run to completion before returning.
    """
    meta_done_ev = threading.Event()
    meta_exc: list[BaseException] = []
    dl_exc: list[BaseException] = []

    def _meta():
        try:
            phase_metadata(hf_repo, paths,
                           max_parquets=max_parquets,
                           min_aesthetic=min_aesthetic)
        except BaseException as exc:
            meta_exc.append(exc)
        finally:
            meta_done_ev.set()

    def _download():
        try:
            phase_download_follow(paths, workers, meta_done_ev=meta_done_ev)
        except BaseException as exc:
            dl_exc.append(exc)

    log("parallel: starting metadata + download threads concurrently")
    meta_t = threading.Thread(target=_meta, name="metadata", daemon=False)
    dl_t   = threading.Thread(target=_download, name="download", daemon=False)
    meta_t.start()
    dl_t.start()
    meta_t.join()
    dl_t.join()
    log("parallel: both threads complete")

    if meta_exc:
        log(f"parallel: metadata thread raised: {meta_exc[0]}")
        raise meta_exc[0]
    if dl_exc:
        log(f"parallel: download thread raised: {dl_exc[0]}")
        raise dl_exc[0]


# ---------------------------------------------------------------------------
# Phase 3 — pack
# ---------------------------------------------------------------------------

def _existing_shards(shard_dir: Path) -> set[int]:
    out = set()
    for p in shard_dir.glob("shard-*.tar"):
        try:
            out.add(int(p.stem.split("-", 1)[1]))
        except (IndexError, ValueError):
            pass
    return out


def _add_to_tar(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = int(time.time())
    tf.addfile(info, io.BytesIO(data))


def phase_pack(paths: dict, shard_size: int) -> None:
    conn = open_db(paths["db"])
    shard_dir = paths["shards"]
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any orphaned tmp files from a prior interrupted run.
    for tmp in shard_dir.glob("shard-*.tar.tmp"):
        try:
            tmp.unlink()
        except OSError:
            pass

    rows_iter = conn.execute(
        "SELECT id, caption, local_path FROM images "
        "WHERE status='downloaded' ORDER BY id"
    )

    existing = _existing_shards(shard_dir)
    log(f"pack: {len(existing)} shard(s) already on disk")

    buf: list[tuple[int, str, str]] = []
    next_shard = 0
    written_shards = 0
    written_records = 0
    skipped_missing = 0
    t0 = time.monotonic()

    def _flush_shard():
        nonlocal next_shard, written_shards, written_records
        if not buf:
            return
        if next_shard in existing:
            log(f"pack: shard-{next_shard:06d} already complete — skipping")
            buf.clear()
            next_shard += 1
            return

        tmp_path   = shard_dir / f"shard-{next_shard:06d}.tar.tmp"
        final_path = shard_dir / f"shard-{next_shard:06d}.tar"
        records_in = 0
        with tarfile.open(tmp_path, "w") as tf:
            for rid, caption, local_path in buf:
                p = Path(local_path)
                if not p.exists() or p.stat().st_size == 0:
                    skipped_missing_inner.append(rid)
                    continue
                try:
                    data = p.read_bytes()
                except OSError:
                    skipped_missing_inner.append(rid)
                    continue
                stem = f"{rid:012d}"
                _add_to_tar(tf, f"{stem}.jpg", data)
                _add_to_tar(tf, f"{stem}.txt", caption.encode("utf-8"))
                records_in += 1
        if records_in == 0:
            tmp_path.unlink(missing_ok=True)
            log(f"pack: shard-{next_shard:06d} empty after filtering — dropped")
        else:
            tmp_path.replace(final_path)
            written_shards  += 1
            written_records += records_in
            elapsed = max(time.monotonic() - t0, 1e-3)
            log(f"pack: wrote shard-{next_shard:06d} ({records_in} records, "
                f"{written_shards} shards in {elapsed:.0f}s)")
            write_heartbeat(phase="pack",
                            shards_written=written_shards,
                            records_written=written_records,
                            current_shard=next_shard,
                            elapsed_sec=int(elapsed))
        buf.clear()
        next_shard += 1

    skipped_missing_inner: list[int] = []

    for rid, caption, local_path in rows_iter:
        if not local_path:
            skipped_missing += 1
            continue
        buf.append((rid, caption, local_path))
        if len(buf) >= shard_size:
            _flush_shard()

    # The trailing partial shard is intentionally NOT written — it would
    # rewrite itself on the next run as new downloads arrive. Only full
    # shards are atomic-published.
    leftover = len(buf)

    skipped_missing += len(skipped_missing_inner)
    write_heartbeat(phase="pack", done=True,
                    shards_written=written_shards,
                    records_written=written_records,
                    leftover_records=leftover,
                    skipped_missing=skipped_missing)
    log(f"pack: complete — {written_shards} new shards, "
        f"{written_records:,} records written, "
        f"{leftover:,} held back in trailing partial shard, "
        f"{skipped_missing:,} rows skipped (file missing)")
    conn.close()


# ---------------------------------------------------------------------------
# Daemon helper
# ---------------------------------------------------------------------------

def detach_and_exec(argv: list[str]) -> None:
    """Double-fork into the background; reopen stdio onto DAEMON_LOG_PATH."""
    if os.fork() != 0:
        os._exit(0)
    os.setsid()
    if os.fork() != 0:
        os._exit(0)

    DAEMON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_fd = os.open(str(DAEMON_LOG_PATH),
                     os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 0)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    # Re-exec without --detach so the child runs the actual phases.
    clean = [a for a in argv if a != "--detach"]
    os.execv(sys.executable, [sys.executable] + clean)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Standalone COYO downloader: HF parquet → SQLite → "
                    "raw_images → WDS tar shards.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--phase", choices=["all", "metadata", "download", "pack"],
                    default="all")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                    help=f"HF dataset repo (default: {DEFAULT_HF_REPO})")
    ap.add_argument("--cold-root", type=Path, default=DEFAULT_COLD_ROOT,
                    help=f"Cold storage root (default: {DEFAULT_COLD_ROOT})")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"Parallel download threads (default: {DEFAULT_WORKERS})")
    ap.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
                    help=f"Records per WDS shard (default: {DEFAULT_SHARD_SIZE})")
    ap.add_argument("--max-parquets", type=int, default=None,
                    help="Limit metadata phase to first N parquet files")
    ap.add_argument("--min-aesthetic", type=float, default=None,
                    help="Skip rows with aesthetic_score_laion_v2 below this threshold")
    ap.add_argument("--detach", action="store_true",
                    help=f"Double-fork into background, log to {DAEMON_LOG_PATH}")
    ap.add_argument("--follow", action="store_true",
                    help="download phase: poll DB continuously until metadata sentinel exists")
    ap.add_argument("--parallel", action="store_true",
                    help="all phase: run metadata and download in concurrent threads")
    args = ap.parse_args(sys.argv[1:])

    if args.detach:
        detach_and_exec(sys.argv)
        return 0

    if not args.cold_root.exists():
        log(f"ERROR: cold root not mounted: {args.cold_root}")
        return 1

    paths = cold_paths(args.cold_root)
    ensure_dirs(paths)

    log(f"download_coyo: phase={args.phase} repo={args.hf_repo} "
        f"cold={args.cold_root} workers={args.workers} "
        f"shard_size={args.shard_size}")

    if args.phase == "all" and args.parallel:
        phase_all_parallel(args.hf_repo, paths, args.workers,
                           args.max_parquets, args.min_aesthetic)
        phase_pack(paths, args.shard_size)
    else:
        if args.phase in ("all", "metadata"):
            phase_metadata(args.hf_repo, paths,
                           max_parquets=args.max_parquets,
                           min_aesthetic=args.min_aesthetic)

        if args.phase in ("all", "download"):
            if args.follow:
                phase_download_follow(paths, args.workers)
            else:
                phase_download(paths, args.workers)

        if args.phase in ("all", "pack"):
            phase_pack(paths, args.shard_size)

    return 0


if __name__ == "__main__":
    sys.exit(main())
