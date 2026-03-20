"""
train/scripts/recaption.py — Re-caption short captions with Moondream (style-focused).

~20% of LAION images have captions under 10 words. Re-caption these with a
style-focused prompt to generate the vocabulary the adapter needs to learn:

  "Describe this image's visual style, colors, lighting, and artistic technique."

Generates captions like:
  "painterly impressionist style with warm amber tones, soft bokeh, and textured brushwork"

These are directly useful training signal for a style adapter. Generic "describe this
image" prompts describe objects, not style.

Run two parallel processes (covering ~475 shards each) to finish in ~2 days:

    # Terminal 1
    python train/scripts/recaption.py \\
        --shards /Volumes/IrisData/shards \\
        --shard_start 0 --shard_end 474

    # Terminal 2
    python train/scripts/recaption.py \\
        --shards /Volumes/IrisData/shards \\
        --shard_start 475 --shard_end 949

CPU allocation: PERF_CORES // 2 per process (2 processes share the GPU evenly).

Reference: plans/ip-adapter-training.md §2.4
"""

import argparse
import glob
import io
import os
import sys
import tarfile

try:
    import subprocess
    _PERF_CORES = int(subprocess.check_output(
        ["sysctl", "-n", "hw.perflevel0.logicalcpu"], text=True).strip())
except Exception:
    _PERF_CORES = os.cpu_count() or 4

_STYLE_PROMPT = (
    "Describe this image's visual style, colors, lighting, and artistic technique."
)


def _word_count(text: str) -> int:
    return len(text.strip().split()) if text and text.strip() else 0


def _decode_image(jpg_bytes: bytes):
    """Decode JPEG bytes to PIL Image for VLM input."""
    from PIL import Image as PilImage
    return PilImage.open(io.BytesIO(jpg_bytes)).convert("RGB")


def iter_shard_raw(shard_path: str):
    """Yield (stem, jpg_bytes, txt) from a tar shard."""
    try:
        with tarfile.open(shard_path) as tar:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            keys = {}
            for name in members:
                stem, _, ext = name.rpartition(".")
                if stem not in keys:
                    keys[stem] = {}
                keys[stem][ext.lower()] = name

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


def rewrite_shard(shard_path: str, records: list) -> None:
    """Rewrite shard in place with updated captions."""
    tmp_path = shard_path + ".recaption.tmp"
    try:
        with tarfile.open(tmp_path, "w") as out_tar:
            for rec in records:
                for ext, data in [("jpg", rec["jpg"]), ("txt", rec["txt"].encode("utf-8"))]:
                    name = f"{rec['key']}.{ext}"
                    info = tarfile.TarInfo(name=name)
                    info.size = len(data)
                    out_tar.addfile(info, io.BytesIO(data))
        os.replace(tmp_path, shard_path)
    except Exception as e:
        print(f"Error rewriting {shard_path}: {e}", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def process_shard(shard_path: str, model, tokenizer, min_words: int = 10) -> dict:
    """
    Re-caption all records in one shard whose captions are under min_words words.
    Rewrites the shard in place.
    Returns counts.
    """
    records = []
    recaptioned = 0
    kept = 0

    for stem, jpg, txt in iter_shard_raw(shard_path):
        if _word_count(txt) < min_words:
            # Re-caption with style-focused prompt
            try:
                img = _decode_image(jpg)
                new_caption = _run_vlm(model, tokenizer, img, _STYLE_PROMPT)
                if new_caption and _word_count(new_caption) >= min_words:
                    txt = new_caption
                    recaptioned += 1
            except Exception as e:
                print(f"  VLM failed for {stem}: {e}", file=sys.stderr)

        records.append({"key": stem, "jpg": jpg, "txt": txt})
        kept += 1

    if recaptioned > 0:
        rewrite_shard(shard_path, records)

    return {"shard": shard_path, "kept": kept, "recaptioned": recaptioned}


def _run_vlm(model, tokenizer, img, prompt: str) -> str:
    """Run Moondream VLM inference on one image."""
    try:
        import mlx.core as mx
        # mlx-vlm generate API
        from mlx_vlm.utils import load_image
        from mlx_vlm import generate as vlm_generate
        result = vlm_generate(
            model, tokenizer,
            prompt=prompt,
            image=img,
            max_tokens=128,
            verbose=False,
        )
        return result.strip()
    except Exception as e:
        print(f"  VLM generate error: {e}", file=sys.stderr)
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Re-caption short captions with Moondream (style-focused)"
    )
    parser.add_argument("--shards", required=True,
                        help="Directory containing .tar shards")
    parser.add_argument("--shard_start", type=int, default=0,
                        help="First shard index to process (inclusive)")
    parser.add_argument("--shard_end", type=int, default=None,
                        help="Last shard index to process (inclusive)")
    parser.add_argument("--model", default="mlx-community/moondream2-4bit",
                        help="VLM model (default: mlx-community/moondream2-4bit)")
    parser.add_argument("--min_caption_words", type=int, default=10,
                        help="Re-caption if caption has fewer words (default 10)")
    args = parser.parse_args()

    all_shards = sorted(glob.glob(os.path.join(args.shards, "*.tar")))
    if not all_shards:
        print(f"No .tar files in {args.shards}", file=sys.stderr)
        sys.exit(1)

    end = args.shard_end if args.shard_end is not None else len(all_shards) - 1
    shards = all_shards[args.shard_start : end + 1]

    print(f"Loading VLM: {args.model}")
    try:
        from mlx_vlm import load as vlm_load
        model, tokenizer = vlm_load(args.model)
    except Exception as e:
        print(f"Failed to load VLM {args.model}: {e}", file=sys.stderr)
        print("Run: pip install mlx-vlm", file=sys.stderr)
        sys.exit(1)

    print(f"Re-captioning shards {args.shard_start}–{end} "
          f"({len(shards)} shards, min_words={args.min_caption_words})")

    total_kept = 0
    total_recaptioned = 0

    for i, shard_path in enumerate(shards):
        result = process_shard(shard_path, model, tokenizer, args.min_caption_words)
        total_kept += result["kept"]
        total_recaptioned += result["recaptioned"]
        if (i + 1) % 10 == 0 or i == len(shards) - 1:
            print(
                f"  [{i+1}/{len(shards)}] recaptioned={total_recaptioned:,} "
                f"total={total_kept:,}"
            )

    print(f"\nDone. Re-captioned {total_recaptioned:,} / {total_kept:,} records.")


if __name__ == "__main__":
    main()
