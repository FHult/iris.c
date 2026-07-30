#!/usr/bin/env python3
"""Census precompute coverage per shard/encoder and PERSIST it to shard_scores.db.

Implements the AGENT.md "metadata stores are the source of truth" rule: walking the
cold `.npz` dirs (1.15M files each, spinning HDD) costs minutes and recurs every
session. Scan ONCE, write a `precompute_coverage` table, read it thereafter.

    precompute_coverage.py scan     # slow: walks cold dirs, upserts the table, prints report
    precompute_coverage.py query    # fast: reads the table, prints coverage (no fs walk)

Table: precompute_coverage(encoder, version, shard_id, n_records, n_images, is_full, ts)
PK (encoder, version, shard_id). n_images/is_full known only for shards we can cheaply
count (hot-staged tars); NULL otherwise (fill as shards are staged/indexed).
"""
import argparse, os, sqlite3, json, glob, tarfile, datetime

COLD = {"qwen3": "v_059443", "vae": "v_2232c1", "siglip": "v_336c6e"}
COLD_ROOT = "/Volumes/16TBCold/precomputed"
HOT_POOLS = [
    "/Volumes/2TBSSD/baseline_pool_hot",
    "/Volumes/2TBSSD/wikiart_pool_hot",
    "/Volumes/2TBSSD/validation/held_out",
    "/Volumes/2TBSSD/anchor_shards",
]
DBS = ["/Volumes/16TBCold/metadata/shard_scores.db", "/Volumes/2TBSSD/shard_scores.db"]
IMG = (".jpg", ".jpeg", ".png", ".webp")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _ensure_table(con):
    con.execute(
        """CREATE TABLE IF NOT EXISTS precompute_coverage(
            encoder TEXT, version TEXT, shard_id TEXT,
            n_records INTEGER, n_images INTEGER, is_full INTEGER, ts TEXT,
            PRIMARY KEY(encoder, version, shard_id))"""
    )


def _scan_encoder(enc, ver):
    d = os.path.join(COLD_ROOT, enc, ver)
    hist = {}
    for e in os.scandir(d):
        n = e.name
        if n.endswith(".npz"):
            sid = n.split("_", 1)[0]
            hist[sid] = hist.get(sid, 0) + 1
    return hist


def _hot_image_counts():
    out = {}
    for pool in HOT_POOLS:
        for f in glob.glob(os.path.join(pool, "*.tar")):
            sid = os.path.basename(f)[:-4]
            try:
                with tarfile.open(f) as t:
                    n = sum(1 for m in t.getnames() if m.lower().endswith(IMG))
            except Exception:
                n = None
            out[sid] = (n, os.path.basename(pool))
    return out


def scan():
    hot = _hot_image_counts()
    hists = {enc: _scan_encoder(enc, ver) for enc, ver in COLD.items()}
    ts = _now()
    rows = []
    for enc, ver in COLD.items():
        h = hists[enc]
        for sid in sorted(set(h) | set(hot)):
            nrec = h.get(sid, 0)
            nimg = hot.get(sid, (None, None))[0]
            is_full = None if not nimg else (1 if nrec >= nimg else 0)
            rows.append((enc, ver, sid, nrec, nimg, is_full, ts))
    for db in DBS:
        if not os.path.isdir(os.path.dirname(db)):
            continue
        con = sqlite3.connect(db, timeout=30)
        _ensure_table(con)
        con.executemany(
            "INSERT OR REPLACE INTO precompute_coverage VALUES (?,?,?,?,?,?,?)", rows
        )
        con.commit()
        con.close()
    rep = {"ts": ts, "per_enc": {}, "hot_shards": {}}
    for enc, ver in COLD.items():
        h = hists[enc]
        rep["per_enc"][enc] = {"version": ver, "n_shards": len(h), "n_records": sum(h.values())}
    for sid, (nimg, pool) in sorted(hot.items()):
        cov = {enc: hists[enc].get(sid, 0) for enc in COLD}
        full = nimg is not None and all(cov[enc] >= nimg for enc in COLD)
        rep["hot_shards"][sid] = {"pool": pool, "n_images": nimg, **cov, "full_all_enc": full}
    rep["hot_summary"] = {
        "n_hot_shards": len(hot),
        "n_full_all_enc": sum(1 for v in rep["hot_shards"].values() if v["full_all_enc"]),
    }
    print(json.dumps(rep, indent=1))


def query():
    db = next((d for d in DBS if os.path.exists(d)), None)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute(
        "SELECT encoder, version, COUNT(*), SUM(n_records), "
        "SUM(CASE WHEN is_full=1 THEN 1 ELSE 0 END) "
        "FROM precompute_coverage GROUP BY encoder, version"
    )
    print(f"db={db}")
    for enc, ver, nsh, nrec, nfull in cur.fetchall():
        print(f"  {enc} {ver}: shards={nsh} records={nrec} full_hot={nfull}")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "query"])
    a = ap.parse_args()
    (scan if a.cmd == "scan" else query)()
