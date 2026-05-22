#!/usr/bin/env python3
"""
train/scripts/pack_wikiart.py — Convert WikiArt HuggingFace dataset → WDS tar pool.

Reads the HF-cached huggan/wikiart dataset and writes one or more WebDataset tars
to an output directory. Each record becomes a stem.jpg + stem.txt pair.

Caption format:
    "A [style] [genre] by [artist]"
    e.g. "A Impressionism landscape by claude-monet"

The script writes tars of --shard-size records each (default 2000), then touches
a .tar.deduped sentinel so clean_wds_pool.py skips these tars (WikiArt is small
enough that cross-dataset dedup via build_shards blocklist is sufficient).

Usage:
    train/.venv/bin/python train/scripts/pack_wikiart.py \\
        [--out-dir /Volumes/16TBCold/converted/wikiart] \\
        [--shard-size 2000] \\
        [--min-size 256]
"""

import argparse
import io
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import COLD_ROOT, write_heartbeat, log_orch

HF_REPO = "huggan/wikiart"
HF_CACHE = Path("/Users/fredrikhult/.cache/huggingface/datasets")

DEFAULT_OUT_DIR  = COLD_ROOT / "converted" / "wikiart"
DEFAULT_SHARD    = 2000
DEFAULT_MIN_SIZE = 256


def _artist_label(ds, idx: int) -> str:
    name = ds.features["artist"].int2str(idx)
    return name.replace("-", " ").title()


def _genre_label(ds, idx: int) -> str:
    name = ds.features["genre"].int2str(idx)
    return name.replace("_", " ").replace("Unknown Genre", "painting")


def _style_label(ds, idx: int) -> str:
    name = ds.features["style"].int2str(idx)
    return name.replace("_", " ")


def _make_caption(ds, row: dict) -> str:
    artist = _artist_label(ds, row["artist"])
    genre  = _genre_label(ds, row["genre"])
    style  = _style_label(ds, row["style"])
    return f"A {style} {genre} by {artist}"


def _encode_jpeg(pil_img, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    rgb = pil_img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack WikiArt HF dataset → WDS tars")
    ap.add_argument("--out-dir",    default=str(DEFAULT_OUT_DIR),
                    help="Output directory for WDS tars")
    ap.add_argument("--shard-size", type=int, default=DEFAULT_SHARD,
                    help="Records per output tar (default: 2000)")
    ap.add_argument("--min-size",   type=int, default=DEFAULT_MIN_SIZE,
                    help="Skip images smaller than this on either axis (default: 256)")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Print what would be written but do not create files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    log_orch(f"pack_wikiart: loading {HF_REPO} from HF cache")
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE))

    from datasets import load_dataset
    ds = load_dataset(HF_REPO, split="train", streaming=False)
    total = len(ds)
    log_orch(f"pack_wikiart: {total:,} records, shard_size={args.shard_size}, min_size={args.min_size}")

    shard_idx  = 0
    written    = 0
    skipped    = 0
    buf_count  = 0
    tmp_path: Path | None = None
    dst_tar = None

    def _close_shard():
        nonlocal dst_tar, tmp_path, shard_idx, buf_count
        if dst_tar is None:
            return
        dst_tar.close()
        final = out_dir / f"{shard_idx:03d}.tar"
        if not args.dry_run:
            os.replace(str(tmp_path), str(final))
            # Touch sentinel so clean_wds_pool skips dedup on these tars
            final.with_suffix(".tar.deduped").touch()
        log_orch(f"  shard {shard_idx:03d}: {buf_count} records → {final.name}")
        shard_idx += 1
        dst_tar  = None
        tmp_path = None
        buf_count = 0

    def _open_shard():
        nonlocal dst_tar, tmp_path
        tmp_path = out_dir / f"{shard_idx:03d}.tar.tmp"
        if args.dry_run:
            dst_tar = None
        else:
            dst_tar = tarfile.open(str(tmp_path), "w")  # noqa: SIM115

    _open_shard()

    for i, row in enumerate(ds):
        img = row["image"]
        w, h = img.size
        if w < args.min_size or h < args.min_size:
            skipped += 1
            continue

        caption = _make_caption(ds, row)
        stem    = f"wikiart_{i:06d}"

        try:
            jpg_bytes = _encode_jpeg(img)
        except Exception as e:
            log_orch(f"  skipping record {i}: encode error: {e}", level="warning")
            skipped += 1
            continue

        txt_bytes = caption.encode()

        if not args.dry_run and dst_tar is not None:
            info_jpg = tarfile.TarInfo(name=f"{stem}.jpg")
            info_jpg.size = len(jpg_bytes)
            dst_tar.addfile(info_jpg, io.BytesIO(jpg_bytes))

            info_txt = tarfile.TarInfo(name=f"{stem}.txt")
            info_txt.size = len(txt_bytes)
            dst_tar.addfile(info_txt, io.BytesIO(txt_bytes))

        buf_count += 1
        written   += 1

        if buf_count >= args.shard_size:
            _close_shard()
            _open_shard()

        if (i + 1) % 500 == 0 or i + 1 == total:
            pct = round(written / total * 100, 1)
            write_heartbeat("pack_wikiart",
                            done=i + 1, total=total, pct=pct,
                            written=written, skipped=skipped)
            print(f"  [{i+1}/{total}] written={written:,} skipped={skipped:,}", flush=True)

    _close_shard()

    print()
    print(f"Done. {written:,} records → {shard_idx} shards in {out_dir}")
    print(f"Skipped: {skipped:,} (below {args.min_size}px)")
    if args.dry_run:
        print("[dry-run] no files written")


if __name__ == "__main__":
    main()
