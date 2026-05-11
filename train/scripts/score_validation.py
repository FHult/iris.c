#!/usr/bin/env python3
"""
train/scripts/score_validation.py — CLIP-I / CLIP-T scoring for MLX-22.

Given a directory of generated images (from run_inference.py) and a prompts file,
computes:
  CLIP-I: cosine similarity between generated-with-adapter and reference image
  CLIP-T: cosine similarity between generated-with-adapter and prompt text
  adapter_delta: CLIP-I(with_adapter) - CLIP-I(no_adapter)

Verdict:
  PASS: mean_clip_i > 0.20 AND mean_adapter_delta > 0.05 AND no weight errors
  WARN: mean_clip_i > 0.15 AND mean_adapter_delta > 0.0
  FAIL: otherwise

Usage:
    python train/scripts/score_validation.py \
        --images /tmp/val_chunk1/ \
        --prompts train/configs/eval_prompts.txt \
        --output /tmp/val_chunk1/scores.json
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLIP loading (lazy — only if scoring is needed)
# ---------------------------------------------------------------------------

_clip_model = None
_clip_processor = None
_clip_name = "openai/clip-vit-base-patch32"


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return
    try:
        from transformers import CLIPModel, CLIPProcessor
        print(f"Loading CLIP ({_clip_name})...")
        _clip_processor = CLIPProcessor.from_pretrained(_clip_name)
        _clip_model = CLIPModel.from_pretrained(_clip_name).eval()
    except ImportError as e:
        raise RuntimeError(
            "transformers and torch required for CLIP scoring. "
            "pip install transformers torch"
        ) from e


def _img_embedding(image) -> "np.ndarray":
    import torch
    inputs = _clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vision_out = _clip_model.vision_model(pixel_values=inputs["pixel_values"])
        feat = _clip_model.visual_projection(vision_out.pooler_output)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].numpy()


def _txt_embedding(text: str) -> "np.ndarray":
    import torch
    inputs = _clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_out = _clip_model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        feat = _clip_model.text_projection(text_out.pooler_output)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].numpy()


def cosine(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Score one pair
# ---------------------------------------------------------------------------

def score_pair(
    img_with_path: str,
    img_no_path: str,
    ref_path: str,
    prompt: str,
) -> dict:
    """
    Returns per-pair scores dict. Missing files produce None scores.
    """
    from PIL import Image
    import numpy as np

    result: dict = {
        "prompt": prompt,
        "ref_path": ref_path,
        "clip_i": None,
        "clip_t": None,
        "adapter_delta": None,
    }

    if not os.path.exists(img_with_path):
        result["error"] = f"missing: {img_with_path}"
        return result

    img_with = Image.open(img_with_path).convert("RGB")
    emb_with = _img_embedding(img_with)

    # CLIP-I vs reference
    emb_ref = None
    ref_full = ref_path if os.path.isabs(ref_path) else os.path.join("train", ref_path)
    if os.path.exists(ref_full):
        ref_img = Image.open(ref_full).convert("RGB")
        emb_ref = _img_embedding(ref_img)
        result["clip_i"] = round(cosine(emb_with, emb_ref), 4)
    else:
        result["warn_clip_i"] = f"ref not found: {ref_full}"

    # CLIP-T vs prompt
    emb_txt = _txt_embedding(prompt)
    result["clip_t"] = round(cosine(emb_with, emb_txt), 4)

    # Adapter delta: CLIP-I(with) - CLIP-I(no)
    if os.path.exists(img_no_path) and result["clip_i"] is not None and emb_ref is not None:
        img_no = Image.open(img_no_path).convert("RGB")
        emb_no = _img_embedding(img_no)
        clip_i_no = cosine(emb_no, emb_ref)
        result["adapter_delta"] = round(result["clip_i"] - clip_i_no, 4)

    return result


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def compute_verdict(pairs: list[dict], weight_errors: list[str] = None) -> dict:
    """
    PASS: mean_clip_i > 0.20 AND mean_adapter_delta > 0.05 AND no weight errors
    WARN: mean_clip_i > 0.15 AND mean_adapter_delta > 0.0
    FAIL: otherwise
    """
    if weight_errors:
        return {"verdict": "FAIL", "reason": f"weight errors: {weight_errors}"}

    clip_i_vals     = [p["clip_i"] for p in pairs if p.get("clip_i") is not None]
    delta_vals      = [p["adapter_delta"] for p in pairs if p.get("adapter_delta") is not None]

    if not clip_i_vals:
        clip_t_vals = [p["clip_t"] for p in pairs if p.get("clip_t") is not None]
        return {
            "verdict": "WARN",
            "reason": "no ref images — CLIP-I skipped",
            "mean_clip_t": round(sum(clip_t_vals) / len(clip_t_vals), 4) if clip_t_vals else None,
            "n_pairs": len(pairs),
        }

    mean_clip_i     = sum(clip_i_vals) / len(clip_i_vals)
    mean_delta      = sum(delta_vals) / len(delta_vals) if delta_vals else None

    summary = {
        "mean_clip_i": round(mean_clip_i, 4),
        "mean_adapter_delta": round(mean_delta, 4) if mean_delta is not None else None,
        "n_pairs": len(pairs),
    }

    if mean_clip_i > 0.20 and (mean_delta is None or mean_delta > 0.05):
        verdict = "PASS"
    elif mean_clip_i > 0.15 and (mean_delta is None or mean_delta > 0.0):
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {"verdict": verdict, **summary}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images",  required=True, help="Directory with generated images")
    ap.add_argument("--prompts", default="train/configs/eval_prompts.txt")
    ap.add_argument("--output",  default=None, help="Write scores JSON to this path")
    ap.add_argument("--weight-errors", nargs="*", default=None,
                    help="Pass weight integrity errors to include in verdict")
    args = ap.parse_args()

    _load_clip()

    # Parse eval prompts
    prompt_pairs: list[tuple[str, str]] = []
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                prompt_pairs.append((parts[0].strip(), parts[1].strip()))

    pair_results: list[dict] = []
    for idx, (prompt, ref_path) in enumerate(prompt_pairs):
        slug = prompt[:40].replace(" ", "_").replace("/", "-")
        img_with = os.path.join(args.images, f"{idx:02d}_{slug}_with_adapter.png")
        img_no   = os.path.join(args.images, f"{idx:02d}_{slug}_no_adapter.png")

        print(f"  [{idx+1}/{len(prompt_pairs)}] scoring: {prompt[:50]}")
        result = score_pair(img_with, img_no, ref_path, prompt)
        pair_results.append(result)

        print(f"    CLIP-I={result.get('clip_i', 'n/a')}  "
              f"CLIP-T={result.get('clip_t', 'n/a')}  "
              f"delta={result.get('adapter_delta', 'n/a')}")

    verdict = compute_verdict(pair_results, args.weight_errors or [])
    print(f"\nVerdict: {verdict['verdict']}")
    print(f"  mean_clip_i={verdict.get('mean_clip_i')}  "
          f"mean_delta={verdict.get('mean_adapter_delta')}")

    output = {
        "verdict": verdict,
        "pairs": pair_results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Scores written to {args.output}")

    sys.exit(0)


if __name__ == "__main__":
    main()
