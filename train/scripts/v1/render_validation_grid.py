#!/usr/bin/env python3
"""
train/scripts/render_validation_grid.py — V-07 visual validation grid.

Assembles a grid showing per-pair: reference | no_adapter | with_adapter | (prev_chunk).

Usage:
    python train/scripts/render_validation_grid.py \
        --images /tmp/val_chunk1/ \
        --prompts train/configs/eval_prompts.txt \
        --output /tmp/val_chunk1/grid.png \
        [--prev-images /tmp/val_chunk0/]   # optional regression column
"""

import argparse
import os
import sys
from pathlib import Path


def _open_or_placeholder(path: str, size: tuple[int, int], label: str = ""):
    from PIL import Image, ImageDraw
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    # Grey placeholder with label
    img = Image.new("RGB", size, (180, 180, 180))
    draw = ImageDraw.Draw(img)
    draw.text((4, size[1] // 2 - 8), label or "N/A", fill=(80, 80, 80))
    return img


def render_grid(
    images_dir: str,
    prompts_path: str,
    output_path: str,
    prev_images_dir: str = None,
    thumb_size: tuple[int, int] = (256, 256),
) -> None:
    from PIL import Image, ImageDraw

    # Parse eval prompts
    prompt_pairs: list[tuple[str, str]] = []
    with open(prompts_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                prompt_pairs.append((parts[0].strip(), parts[1].strip()))

    n_rows = len(prompt_pairs)
    has_prev = prev_images_dir is not None
    n_cols = 4 if has_prev else 3  # ref | no_adapter | with_adapter | [prev]

    col_labels = ["reference", "no_adapter", "with_adapter"]
    if has_prev:
        col_labels.append("prev_chunk")

    label_h = 20
    cell_w, cell_h = thumb_size
    grid_w = n_cols * cell_w
    grid_h = n_rows * (cell_h + label_h) + label_h  # extra row for col headers

    grid = Image.new("RGB", (grid_w, grid_h), (240, 240, 240))
    draw = ImageDraw.Draw(grid)

    # Column headers
    for ci, lbl in enumerate(col_labels):
        draw.text((ci * cell_w + 4, 4), lbl, fill=(40, 40, 40))

    for ri, (prompt, ref_path) in enumerate(prompt_pairs):
        slug = prompt[:40].replace(" ", "_").replace("/", "-")
        y = label_h + ri * (cell_h + label_h)

        # Row label (truncated prompt)
        draw.text((2, y + cell_h // 2), prompt[:30], fill=(60, 60, 60))

        ref_full = ref_path if os.path.isabs(ref_path) else os.path.join("train", ref_path)
        img_with = os.path.join(images_dir, f"{ri:02d}_{slug}_with_adapter.png")
        img_no   = os.path.join(images_dir, f"{ri:02d}_{slug}_no_adapter.png")

        cells = [
            _open_or_placeholder(ref_full,  thumb_size, "ref"),
            _open_or_placeholder(img_no,    thumb_size, "no_adapter"),
            _open_or_placeholder(img_with,  thumb_size, "with_adapter"),
        ]
        if has_prev:
            img_prev = os.path.join(prev_images_dir, f"{ri:02d}_{slug}_with_adapter.png")
            cells.append(_open_or_placeholder(img_prev, thumb_size, "prev"))

        for ci, cell in enumerate(cells):
            grid.paste(cell, (ci * cell_w, y))

        # Separator line between rows
        if ri < n_rows - 1:
            row_y = y + cell_h + label_h - 1
            draw.line([(0, row_y), (grid_w, row_y)], fill=(200, 200, 200), width=1)

    grid.save(output_path)
    print(f"Grid saved: {output_path}  ({grid_w}x{grid_h}px, {n_rows} rows, {n_cols} cols)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images",      required=True)
    ap.add_argument("--prompts",     default="train/configs/eval_prompts.txt")
    ap.add_argument("--output",      required=True)
    ap.add_argument("--prev-images", default=None)
    ap.add_argument("--thumb-size",  type=int, default=256, help="Cell width/height in pixels")
    args = ap.parse_args()

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("Pillow required: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    render_grid(
        images_dir=args.images,
        prompts_path=args.prompts,
        output_path=args.output,
        prev_images_dir=args.prev_images,
        thumb_size=(args.thumb_size, args.thumb_size),
    )


if __name__ == "__main__":
    main()
