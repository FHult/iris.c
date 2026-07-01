"""Build neighbors_look.sqlite — the look-stratified, content-decorrelated pairing DB for the
SREF-DATA-TEST. Same schema as the champion's CSD neighbors.sqlite (rec_id -> neighbor_ids/cos),
so it drops into train_ip_adapter via data.style_neighbors_db with NO trainer change.

For each anchor record: neighbours = records that SHARE a low-level LOOK (17-d look vector cosine
>= LOOK_MIN) but DIFFER in content (rank by cached-CSD cosine ASC = most content-different first).
So each training pair is look-shared / subject-varied — the DATA-SELECTION PRINCIPLE applied to the
IP-adapter (BACKLOG SREF-DATA-TEST). neighbor_cos stores the LOOK cosine (>= LOOK_MIN passes the
loader's _STYLE_NBR_MIN_COS 0.6 gate).

  train/.venv/bin/python build_look_neighbors.py --allowlist pool_top25.json --out neighbors_look.sqlite
"""
import argparse, glob, io, json, os, sqlite3, tarfile, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # train/ (for lora.*)
from lora.curate_lookstyle_subset import look_vec       # reuse the validated look descriptor

HOT = "/Volumes/2TBSSD/baseline_pool_hot"
CSD = "/Volumes/2TBSSD/sref_eval/style_cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allowlist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--look-min", type=float, default=0.70, help="min look cosine to be a neighbour")
    args = ap.parse_args()

    allow = set(json.load(open(args.allowlist))["rec_ids"])
    print(f"allowlist: {len(allow)} rec_ids", flush=True)

    ids, looks, csds = [], [], []
    for tp in sorted(glob.glob(f"{HOT}/*.tar")):
        sh = os.path.basename(tp)[:-4]
        cache = dict(np.load(f"{CSD}/{sh}.npz"))
        with tarfile.open(tp) as t:
            for m in t.getmembers():
                if not m.name.endswith(".jpg"):
                    continue
                rid = m.name[:-4]
                if rid not in allow or rid not in cache:
                    continue
                try:
                    im = Image.open(io.BytesIO(t.extractfile(m).read()))
                    looks.append(look_vec(im)); csds.append(cache[rid]); ids.append(rid)
                except Exception:
                    continue
        print(f"  {sh}: {len(ids)} decoded", flush=True)

    L = np.stack(looks).astype(np.float32)
    L = (L - L.mean(0)) / (L.std(0) + 1e-6)
    L /= (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
    V = np.stack(csds).astype(np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    M = len(ids)
    idpos = {r: i for i, r in enumerate(ids)}
    print(f"decoded {M} records; building k={args.k} look-neighbours (look>= {args.look_min}, "
          f"content-different)", flush=True)

    con = sqlite3.connect(args.out)
    con.execute("CREATE TABLE IF NOT EXISTS neighbors (rec_id TEXT PRIMARY KEY, neighbor_ids TEXT, neighbor_cos TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    for k, v in {"k": str(args.k), "look_min": str(args.look_min), "kind": "look_stratified_content_decorrelated"}.items():
        con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))

    CH = 1000
    n_with = 0
    for s in range(0, M, CH):
        e = min(s + CH, M)
        lsim = L[s:e] @ L.T                       # [chunk, M]
        csim = V[s:e] @ V.T                       # [chunk, M]
        for r in range(e - s):
            i = s + r
            elig = np.where((lsim[r] >= args.look_min) & (np.arange(M) != i))[0]
            if len(elig) == 0:
                continue
            # most content-different first (lowest CSD cosine), among strong-look matches
            order = elig[np.argsort(csim[r][elig])][: args.k]
            nb_ids = [ids[j] for j in order]
            nb_cos = [round(float(lsim[r][j]), 4) for j in order]   # store LOOK cosine (>= look_min)
            con.execute("INSERT OR REPLACE INTO neighbors VALUES (?,?,?)",
                        (ids[i], json.dumps(nb_ids), json.dumps(nb_cos)))
            n_with += 1
        print(f"  neighbours {e}/{M}  ({n_with} with >=1)", flush=True)
    con.commit()
    # quick stats
    row = con.execute("SELECT neighbor_ids, neighbor_cos FROM neighbors LIMIT 1").fetchone()
    con.close()
    print(f"DONE: {n_with}/{M} records have >=1 look-neighbour  ({100*n_with/M:.0f}% coverage)", flush=True)
    print(f"  sample neighbours: {row[0][:80]}  cos {row[1][:40]}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
