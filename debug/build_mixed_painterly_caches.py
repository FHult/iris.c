"""Build unified caches for the 40:60 mixed-data painterly specialist (SREF-STYLE-ROUTER step 3).

WikiArt (all 23,444) + 7 diverse shards (35,000) -> 40.1% WikiArt. Record ids don't collide
(wikiart_* vs NNNNNN_NNNN), so caches are a UNION; ratio is controlled by shard_paths in the config.
Symlinks only (no copies). Merges the two within-style neighbor DBs.
"""
import os, glob, sqlite3, sys

BASE = "/Volumes/2TBSSD"
OUT = os.path.join(BASE, "mixed_painterly_v3")
DIVERSE_SHARDS = ["000000", "000004", "000008", "000012", "000026", "000061", "000112"]  # 7 x 5000

SRC = {
    "vae256": [
        (os.path.join(BASE, "precomputed/vae_wikiart256"), "wikiart_*.npz"),   # all WikiArt
    ] + [(os.path.join(BASE, "precomputed/vae_sref256px"), f"{s}_*.npz") for s in DIVERSE_SHARDS],
    "qwen3": [
        (os.path.join(BASE, "precomputed/qwen3_wikiart"), "wikiart_*.npz"),
    ] + [(os.path.join(BASE, "precomputed/qwen3/v_059443"), f"{s}_*.npz") for s in DIVERSE_SHARDS],
    "csd": [
        (os.path.join(BASE, "wikiart_csd"), "*.npz"),                          # 14 WikiArt bundles
    ] + [(os.path.join(BASE, "universe_csd_full"), f"{s}.npz") for s in DIVERSE_SHARDS],  # 7 bundles
}
NBR_SRCS = [os.path.join(BASE, "wikiart_neighbors.sqlite"),
            os.path.join(BASE, "sref_eval/style_cache/neighbors.sqlite")]


def link_group(name, specs):
    dst = os.path.join(OUT, name)
    os.makedirs(dst, exist_ok=True)
    n = 0
    for src_dir, pat in specs:
        for p in glob.glob(os.path.join(src_dir, pat)):
            link = os.path.join(dst, os.path.basename(p))
            if not os.path.islink(link) and not os.path.exists(link):
                os.symlink(p, link)
            n += 1
    print(f"  {name}: {n} files linked -> {dst}", flush=True)
    return n


def merge_neighbors():
    dst = os.path.join(OUT, "neighbors.sqlite")
    if os.path.exists(dst):
        os.remove(dst)
    out = sqlite3.connect(dst)
    out.execute("CREATE TABLE neighbors (rec_id TEXT PRIMARY KEY, neighbor_ids TEXT, neighbor_cos TEXT)")
    total = 0
    for src in NBR_SRCS:
        c = sqlite3.connect(src)
        schema = c.execute("select sql from sqlite_master where type='table' and name='neighbors'").fetchone()[0]
        rows = c.execute("select rec_id, neighbor_ids, neighbor_cos from neighbors").fetchall()
        out.executemany("INSERT OR IGNORE INTO neighbors VALUES (?,?,?)", rows)
        print(f"  neighbors from {os.path.basename(src)}: {len(rows)} rows  (schema ok: {'neighbors' in schema})", flush=True)
        total += len(rows)
        c.close()
    out.commit()
    merged = out.execute("select count(*) from neighbors").fetchone()[0]
    out.close()
    print(f"  merged neighbors.sqlite: {merged} rows (src total {total})", flush=True)
    return merged


def verify():
    print("=== VERIFY ===", flush=True)
    # 1. counts
    vae = len(os.listdir(os.path.join(OUT, "vae256")))
    qw = len(os.listdir(os.path.join(OUT, "qwen3")))
    csd = len(os.listdir(os.path.join(OUT, "csd")))
    n_wiki = len(glob.glob(os.path.join(OUT, "vae256", "wikiart_*.npz")))
    n_div = vae - n_wiki
    ratio = n_wiki / vae if vae else 0
    print(f"  vae256={vae} (wikiart={n_wiki}, diverse={n_div}, WikiArt frac={ratio*100:.1f}%)  qwen3={qw}  csd_bundles={csd}", flush=True)
    # 2. a WikiArt + a diverse rec resolve in all 3 caches
    import numpy as np
    ok = True
    for rid in ("wikiart_000000", "000000_0000"):
        vpath = os.path.join(OUT, "vae256", f"{rid}.npz")
        qpath = os.path.join(OUT, "qwen3", f"{rid}.npz")
        vae_ok = os.path.exists(vpath)
        qw_ok = os.path.exists(qpath)
        # csd: rec must be a key in SOME bundle
        csd_ok = False
        for b in glob.glob(os.path.join(OUT, "csd", "*.npz")):
            if rid in np.load(b).files:
                csd_ok = True; break
        print(f"  {rid}: vae={vae_ok} qwen3={qw_ok} csd={csd_ok}", flush=True)
        ok = ok and vae_ok and qw_ok and csd_ok
    print(f"  RESULT: {'PASS' if (ok and 0.38 <= ratio <= 0.42) else 'CHECK'}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(f"building mixed caches -> {OUT}", flush=True)
    for name, specs in SRC.items():
        link_group(name, specs)
    merge_neighbors()
    verify()
