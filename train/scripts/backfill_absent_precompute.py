#!/usr/bin/env python3
"""Backfill precompute for the corpus shards that have NO precompute, at the
existing subsample level (~200 records/shard, deterministic tar-order first-N).

Honors the invariants:
  - Stages each batch cold->hot BEFORE precompute (never precompute from cold).
  - Writes qwen3/vae/siglip into the authoritative COLD caches (extends the
    existing versions v_059443 / v_2232c1 / v_336c6e; precompute_all re-finalizes
    each version's manifest per invocation).
  - Updates the precompute_coverage table after each batch (metadata source of
    truth) and frees hot tars, so the run is resumable and bounded in hot space.

Resumable: 'absent' is recomputed from precompute_coverage on each start, so
already-done shards are skipped. Caffeinate the launch (long, staging-dominated).

Usage: backfill_absent_precompute.py [--batch 60] [--subsample 200] [--limit N]
"""
import argparse
import glob
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import datetime

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLD_SHARDS = "/Volumes/16TBCold/shards"
COLD_PRE = "/Volumes/16TBCold/precomputed"
HOT_POOL = "/Volumes/2TBSSD/staging/backfill_pool"
DBS = ["/Volumes/16TBCold/metadata/shard_scores.db", "/Volumes/2TBSSD/shard_scores.db"]
FLUX = os.path.join(REPO, "flux-klein-4b")   # symlink -> flux-klein-model (VAE parity verified)
VENV_PY = os.path.join(REPO, "train/.venv/bin/python")
VER = {"qwen3": "v_059443", "vae": "v_2232c1", "siglip": "v_336c6e"}
IMG = (".jpg", ".jpeg", ".png", ".webp")
MIN_FREE_GB = 150   # keep this much hot headroom before staging a batch


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _log(msg):
    print(f"[{_now()}] {msg}", flush=True)


def _absent_shards():
    """Corpus shards with zero records in ANY encoder (per the coverage table)."""
    db = next(d for d in DBS if os.path.exists(d))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.cursor()
    covered = {}
    for e in VER:
        cur.execute("SELECT shard_id FROM precompute_coverage WHERE encoder=? AND n_records>0", (e,))
        covered[e] = set(r[0] for r in cur.fetchall())
    con.close()
    corpus = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{COLD_SHARDS}/*.tar"))
    return [s for s in corpus if any(s not in covered[e] for e in VER)]


def _free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def _tar_images(path):
    try:
        with tarfile.open(path) as t:
            return sum(1 for m in t.getnames() if m.lower().endswith(IMG))
    except Exception:
        return None


def _update_coverage(shard_ids, subsample):
    ts = _now()
    rows = []
    for sid in shard_ids:
        nimg = _tar_images(os.path.join(HOT_POOL, f"{sid}.tar"))
        nrec = min(subsample, nimg) if nimg else subsample
        for e in VER:
            rows.append((e, VER[e], sid, nrec, nimg, None, ts))
    for db in DBS:
        if os.path.isdir(os.path.dirname(db)):
            con = sqlite3.connect(db, timeout=60)
            con.executemany("INSERT OR REPLACE INTO precompute_coverage VALUES (?,?,?,?,?,?,?)", rows)
            con.commit()
            con.close()


def _stage(shard_ids):
    os.makedirs(HOT_POOL, exist_ok=True)
    for sid in shard_ids:
        dst = os.path.join(HOT_POOL, f"{sid}.tar")
        if os.path.exists(dst):
            continue
        src = os.path.join(COLD_SHARDS, f"{sid}.tar")
        shutil.copyfile(src, dst)
        _log(f"  staged {sid} ({os.path.getsize(dst)} bytes)")


def _precompute(subsample):
    cmd = [
        VENV_PY, os.path.join(REPO, "train/scripts/precompute_all.py"),
        "--shards", HOT_POOL,
        # NOTE: output must be the VERSION dir — precompute_all writes npz flat to
        # {output}/{id}.npz, so pointing at the encoder dir orphans them beside the
        # version dir (validated 2026-07-30). PrecomputeCache derives enc_dir = parent.
        "--qwen3-output", os.path.join(COLD_PRE, "qwen3", VER["qwen3"]),
        "--vae-output", os.path.join(COLD_PRE, "vae", VER["vae"]),
        "--siglip-output", os.path.join(COLD_PRE, "siglip", VER["siglip"]),
        "--siglip", "--image-size", "512",
        "--subsample-per-shard", str(subsample), "--skip-light-scores",
        "--flux-model", FLUX, "--seed", "1", "--ai",
    ]
    return subprocess.run(cmd, cwd=REPO).returncode


def _cleanup(shard_ids):
    for sid in shard_ids:
        p = os.path.join(HOT_POOL, f"{sid}.tar")
        if os.path.exists(p):
            os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=60)
    ap.add_argument("--subsample", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    absent = _absent_shards()
    if a.limit:
        absent = absent[: a.limit]
    _log(f"backfill: {len(absent)} absent shards, batch={a.batch}, subsample={a.subsample}")
    if not absent:
        _log("nothing to do — all corpus shards covered.")
        return 0

    done = 0
    for i in range(0, len(absent), a.batch):
        batch = absent[i : i + a.batch]
        free = _free_gb("/Volumes/2TBSSD")
        _log(f"batch {i // a.batch + 1}: {len(batch)} shards ({batch[0]}..{batch[-1]}), hot free={free:.0f} GB")
        if free < MIN_FREE_GB + len(batch) * 4:
            _log(f"ABORT: insufficient hot space ({free:.0f} GB) for batch + {MIN_FREE_GB} GB margin.")
            return 2
        _stage(batch)
        rc = _precompute(a.subsample)
        if rc != 0:
            _log(f"ABORT: precompute_all returned {rc} on batch starting {batch[0]}.")
            return rc
        _update_coverage(batch, a.subsample)
        _cleanup(batch)
        done += len(batch)
        _log(f"batch done ({done}/{len(absent)} shards); coverage updated + hot freed.")

    _log(f"BACKFILL COMPLETE: {done} shards precomputed at subsample {a.subsample}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
