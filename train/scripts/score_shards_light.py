"""
train/scripts/score_shards_light.py — Lightweight per-shard quality scoring.

Runs after shard building, before expensive precompute (Qwen3 + VAE + SigLIP).
Scores each shard on four signals and writes a JSON sidecar so the flywheel
shard selector can prioritise high-quality shards from the first iteration.

Scoring signals (in priority order):
  1. CLIP image-text alignment   — cosine similarity, ViT-L/14 on MPS
  2. Aesthetic score             — LAION aesthetic predictor v2 linear head
  3. Sharpness                   — Laplacian variance (no ML)
  4. Caption quality             — length, repetition, lexical diversity

Per-source thresholds and weights account for dataset heterogeneity:
  journeydb  — AI-generated art; rich prompts; strict aesthetic, lenient CLIP
  laion      — noisy web data; moderate all signals
  coyo       — similar to laion
  wikiart    — traditional painting; strict aesthetic, lenient CLIP alignment

Output per shard:
  {shard_stem}.light_scores.json  — scoring result and metadata
  {shard_stem}.scored             — sentinel (skip on re-run)

Usage:
    train/.venv/bin/python train/scripts/score_shards_light.py \\
        --shards /Volumes/16TBCold/shards \\
        --workers 1

    # Force re-score all shards:
    train/.venv/bin/python train/scripts/score_shards_light.py \\
        --shards /Volumes/16TBCold/shards --force

    # Score specific subset:
    train/.venv/bin/python train/scripts/score_shards_light.py \\
        --shards /Volumes/16TBCold/shards \\
        --shard-pattern "00[0-2]*.tar"

Integration:
  - Run after build_shards.py, before precompute_all.py.
  - pipeline_doctor.py reads .light_scores.json sidecars for pool health checks.
  - Flywheel shard selector can read combined_score as a prior before any
    training-derived attribution signal exists (cold start).
"""

import argparse
import glob
import io
import json
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_lib import (
    now_iso, write_heartbeat, log_event,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERSION = "v1"
SAMPLE_N = 256          # images sampled per shard (enough for stable statistics)
BATCH_SIZE = 32         # images per GPU forward pass
AESTHETIC_WEIGHTS_URL = (
    "https://huggingface.co/laion/aesthetics_predictor_v2_linear"
    "/resolve/main/linear.pth"
)
AESTHETIC_CACHE_PATH = Path.home() / ".cache" / "iris_pipeline" / "aesthetic_v2_linear.pth"

# Per-source scoring weights: (clip_alignment, aesthetic, sharpness, caption)
SOURCE_WEIGHTS = {
    "journeydb": (0.25, 0.40, 0.15, 0.20),
    "laion":     (0.35, 0.30, 0.15, 0.20),
    "coyo":      (0.35, 0.30, 0.15, 0.20),
    "wikiart":   (0.20, 0.45, 0.15, 0.20),
    "unknown":   (0.30, 0.35, 0.15, 0.20),
}

# Per-source combined-score thresholds for keep/discard decision
SOURCE_THRESHOLDS = {
    "journeydb": 0.42,
    "laion":     0.38,
    "coyo":      0.38,
    "wikiart":   0.40,
    "unknown":   0.40,
}


# ---------------------------------------------------------------------------
# Source classification (mirrors build_shards._classify_source)
# ---------------------------------------------------------------------------

def _detect_source(shard_path: str) -> str:
    """
    Detect the primary source dataset for a shard.

    For single-source shards (or source dirs named by dataset), the path
    keyword match is sufficient.  For mixed-source output shards (the common
    case from build_shards.py) the provenance sidecar is read.  If multiple
    source types are present, "unknown" is returned so that the neutral
    mixed-source weights are applied rather than biasing toward whichever
    source type happens to sort first.
    """
    p = shard_path.lower()
    if "journeydb" in p or "jdb" in p:
        return "journeydb"
    if "wikiart" in p:
        return "wikiart"
    if "laion" in p:
        return "laion"
    if "coyo" in p:
        return "coyo"
    # Read provenance sidecar written by build_shards.py
    prov = Path(shard_path).with_suffix(".provenance.json")
    if prov.exists():
        try:
            data = json.loads(prov.read_text())
            type_map = {"jdb": "journeydb"}
            types = {
                type_map.get(s.get("type", ""), s.get("type", "unknown"))
                for s in data.get("sources", [])
            } - {"unknown"}
            if len(types) == 1:
                return types.pop()
            # Multiple source types → treat as mixed; caller uses "unknown" weights
        except Exception:
            pass
    return "unknown"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_clip(device: str):
    """Load OpenCLIP ViT-L/14 on the given device. Returns (model, preprocess, tokenizer)."""
    import open_clip  # type: ignore
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device
        )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return model, preprocess, tokenizer


def _load_aesthetic_predictor(device: str):
    """
    Load the LAION aesthetic predictor v2 linear head (Linear(768, 1)).
    Downloads weights from HuggingFace on first use and caches locally.
    Returns None if download fails — aesthetic scoring is then skipped.
    """
    import torch  # type: ignore

    AESTHETIC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not AESTHETIC_CACHE_PATH.exists():
        print(f"  Downloading aesthetic predictor weights → {AESTHETIC_CACHE_PATH}")
        tmp = AESTHETIC_CACHE_PATH.with_suffix(".pth.tmp")
        try:
            urllib.request.urlretrieve(AESTHETIC_WEIGHTS_URL, str(tmp))
            tmp.rename(AESTHETIC_CACHE_PATH)
        except Exception as e:
            tmp.unlink(missing_ok=True)
            print(f"  WARNING: could not download aesthetic predictor: {e}")
            print("  Aesthetic scores will be omitted.")
            return None

    try:
        state = torch.load(str(AESTHETIC_CACHE_PATH), map_location="cpu", weights_only=True)
        linear = torch.nn.Linear(768, 1)
        linear.load_state_dict(state)
        linear.eval()
        linear.to(device)
        return linear
    except Exception as e:
        print(f"  WARNING: failed to load aesthetic predictor: {e}")
        print(f"  Delete {AESTHETIC_CACHE_PATH} to trigger a fresh download.")
        return None


# ---------------------------------------------------------------------------
# Signal extractors
# ---------------------------------------------------------------------------

def _clip_scores(model, preprocess, tokenizer,
                 images_pil: list, captions: list,
                 device: str) -> tuple[list[float], list[np.ndarray]]:
    """
    Returns (alignment_scores, image_features_list).
    alignment_scores: cosine similarity between each image and its caption.
    image_features_list: raw L2-normalised CLIP image features (for aesthetic).
    """
    import torch  # type: ignore

    all_align: list[float] = []
    all_feats: list[np.ndarray] = []

    for i in range(0, len(images_pil), BATCH_SIZE):
        batch_imgs = images_pil[i : i + BATCH_SIZE]
        batch_caps = captions[i : i + BATCH_SIZE]

        img_tensors = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
        txt_tokens  = tokenizer(batch_caps).to(device)

        with torch.no_grad():
            img_feats = model.encode_image(img_tensors)
            txt_feats = model.encode_text(txt_tokens)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

        sims = (img_feats * txt_feats).sum(dim=-1).cpu().tolist()
        all_align.extend(sims)
        all_feats.extend(img_feats.cpu().numpy())

    return all_align, all_feats


def _aesthetic_scores(predictor, image_features: list[np.ndarray],
                      device: str) -> list[float]:
    """Run aesthetic predictor on pre-extracted CLIP image features."""
    import torch  # type: ignore

    scores: list[float] = []
    for i in range(0, len(image_features), BATCH_SIZE):
        batch = np.stack(image_features[i : i + BATCH_SIZE])
        t = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            raw = predictor(t).squeeze(-1).cpu().tolist()
        # Predictor output is on a 1-10 scale; normalise to [0, 1]
        scores.extend([max(0.0, min(1.0, (v - 1.0) / 9.0)) for v in raw])
    return scores


def _sharpness_score(img_pil) -> float:
    """Laplacian variance as a sharpness proxy, normalised to [0, 1].
    Uses pure numpy to avoid Pillow version dependencies (Kernel API changed in 12.x)."""
    grey = img_pil.convert("L").resize((256, 256))
    arr  = np.array(grey, dtype=np.float32)
    # 4-connected discrete Laplacian (avoids diagonal coupling noise)
    lap = (
        - arr[:-2, 1:-1]   # top
        - arr[2:,  1:-1]   # bottom
        - arr[1:-1, :-2]   # left
        - arr[1:-1, 2:]    # right
        + 4.0 * arr[1:-1, 1:-1]
    )
    var = float(np.var(lap))
    # Empirically: very blurry ≈ 10, sharp ≈ 3000+
    return float(min(1.0, var / 2000.0))


def _caption_quality_score(caption: str) -> float:
    """
    Heuristic caption quality on [0, 1]:
      - Length score: prefer 10-100 words
      - Repetition penalty: repeated trigrams
      - Lexical diversity: unique/total words ratio
      - Artifact penalty: URLs, file extensions, excessive punctuation
    """
    if not caption or not caption.strip():
        return 0.0

    words = caption.strip().lower().split()
    n = len(words)
    if n < 5:
        return 0.1

    # Length score: peaks at 20-80 words
    length_score = min(1.0, n / 20.0) if n < 20 else (1.0 if n <= 80 else max(0.3, 1.0 - (n - 80) / 200.0))

    # Lexical diversity
    diversity = len(set(words)) / max(n, 1)

    # Trigram repetition
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    rep_penalty = 0.0
    if trigrams:
        unique_tri = len(set(trigrams)) / len(trigrams)
        rep_penalty = max(0.0, 1.0 - unique_tri) * 0.5

    # Artifact detection
    artifact = 0.0
    cap_lower = caption.lower()
    if cap_lower.startswith("http") or "www." in cap_lower:
        artifact += 0.4
    if re.search(r'\.(jpg|jpeg|png|gif|webp|bmp|tiff?)(\s|$)', cap_lower):
        artifact += 0.3
    if caption.count("|") > 3 or caption.count("_") > 5:
        artifact += 0.2

    score = length_score * 0.4 + diversity * 0.4 - rep_penalty - artifact
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# Per-shard scoring
# ---------------------------------------------------------------------------

def _read_shard_samples(shard_path: str, n: int = SAMPLE_N) -> tuple[list, list]:
    """
    Read up to n image-caption pairs from a shard tar using streaming mode.

    Uses tarfile 'r|' (pipe/streaming) to avoid a full sequential scan of
    all tar headers — critical when shards live on a spinning HDD.  Streaming
    reads headers sequentially and stops as soon as n complete pairs are found,
    accessing only the first ~n/shard_size fraction of the file.

    WDS shards interleave image/caption files for the same record ID
    consecutively, so streaming naturally collects complete pairs.
    """
    from PIL import Image  # type: ignore

    # partial_records: stem → {"jpg": bytes} or {"txt": str} accumulated
    # as the stream passes through each file.
    partial: dict[str, dict] = {}
    pairs: list[tuple[bytes, str]] = []

    try:
        with tarfile.open(shard_path, mode="r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                stem, _, ext = member.name.rpartition(".")
                ext = ext.lower()

                if ext in ("jpg", "jpeg", "png"):
                    data = tf.extractfile(member)
                    if data is None:
                        continue
                    partial.setdefault(stem, {})["jpg"] = data.read()
                elif ext in ("txt", "caption"):
                    data = tf.extractfile(member)
                    if data is None:
                        continue
                    partial.setdefault(stem, {})["txt"] = (
                        data.read().decode("utf-8", errors="replace").strip()
                    )

                rec = partial.get(stem, {})
                if "jpg" in rec and "txt" in rec:
                    pairs.append((rec["jpg"], rec["txt"]))
                    del partial[stem]
                    if len(pairs) >= n:
                        break  # stop early — no need to scan the rest
    except Exception as e:
        print(f"  WARNING: failed to read {shard_path}: {e}")
        return [], []

    images, captions = [], []
    for jpg_bytes, caption in pairs:
        try:
            img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
            images.append(img)
            captions.append(caption)
        except Exception:
            continue

    return images, captions


def score_shard(shard_path: str, model, preprocess, tokenizer,
                aesthetic_predictor, device: str,
                sample_n: int = SAMPLE_N) -> dict:
    """Score a single shard. Returns the full result dict."""
    source = _detect_source(shard_path)
    shard_id = Path(shard_path).stem  # e.g. "000042"

    images, captions = _read_shard_samples(shard_path, n=sample_n)
    num_samples = len(images)

    if num_samples == 0:
        return {
            "shard_id": shard_id,
            "source": source,
            "num_samples": 0,
            "light_scores": {
                "clip_alignment_avg": 0.0,
                "aesthetic_avg": 0.0,
                "blur_avg": 0.0,
                "caption_quality": 0.0,
                "combined_score": 0.0,
            },
            "decision": "discard",
            "threshold_used": SOURCE_THRESHOLDS.get(source, 0.40),
            "timestamp": now_iso(),
            "version": VERSION,
            "error": "no_samples",
        }

    # CLIP alignment + features
    align_scores, img_feats = _clip_scores(model, preprocess, tokenizer,
                                           images, captions, device)
    clip_avg = float(np.mean(align_scores)) if align_scores else 0.0

    # Aesthetic score
    has_aesthetic = aesthetic_predictor is not None and bool(img_feats)
    if has_aesthetic:
        aesth_scores = _aesthetic_scores(aesthetic_predictor, img_feats, device)
        aesth_avg = float(np.mean(aesth_scores))
    else:
        aesth_avg = 0.0  # unused when weight is redistributed below

    # Sharpness (CPU, fast)
    blur_scores = [_sharpness_score(img) for img in images]
    blur_avg = float(np.mean(blur_scores)) if blur_scores else 0.0

    # Caption quality (CPU, fast)
    cap_scores = [_caption_quality_score(cap) for cap in captions]
    cap_avg = float(np.mean(cap_scores)) if cap_scores else 0.0

    # Combined weighted score.
    # When aesthetic predictor is unavailable, redistribute its weight
    # proportionally to the remaining signals so the combined score stays
    # on a comparable scale instead of receiving a fixed 0.5 bonus.
    w_clip, w_aesth, w_blur, w_cap = SOURCE_WEIGHTS.get(source, SOURCE_WEIGHTS["unknown"])
    if not has_aesthetic:
        # Redistribute aesthetic weight to the other three signals
        remaining = w_clip + w_blur + w_cap
        if remaining > 0:
            scale = (w_clip + w_aesth + w_blur + w_cap) / remaining
            w_clip *= scale
            w_blur *= scale
            w_cap  *= scale
        w_aesth = 0.0
    combined = (
        w_clip  * clip_avg +
        w_aesth * aesth_avg +
        w_blur  * blur_avg +
        w_cap   * cap_avg
    )

    threshold = SOURCE_THRESHOLDS.get(source, 0.40)
    decision  = "keep" if combined >= threshold else "discard"

    return {
        "shard_id":    shard_id,
        "source":      source,
        "num_samples": num_samples,
        "light_scores": {
            "clip_alignment_avg": round(clip_avg, 4),
            "aesthetic_avg":      round(aesth_avg, 4),
            "blur_avg":           round(blur_avg, 4),
            "caption_quality":    round(cap_avg, 4),
            "combined_score":     round(combined, 4),
        },
        "decision":       decision,
        "threshold_used": threshold,
        "timestamp":      now_iso(),
        "version":        VERSION,
    }


# ---------------------------------------------------------------------------
# Sidecar / sentinel helpers
# ---------------------------------------------------------------------------

def _sidecar_path(shard_path: str) -> Path:
    return Path(shard_path).with_suffix(".light_scores.json")


def _sentinel_path(shard_path: str) -> Path:
    return Path(shard_path).with_suffix(".scored")


def _is_scored(shard_path: str) -> bool:
    return _sentinel_path(shard_path).exists()


def _write_result(shard_path: str, result: dict) -> None:
    sidecar = _sidecar_path(shard_path)
    sidecar.write_text(json.dumps(result, indent=2))
    _sentinel_path(shard_path).touch()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight per-shard quality scoring (CLIP + aesthetic + sharpness + caption)"
    )
    parser.add_argument(
        "--shards", required=True,
        help="Directory containing shard .tar files"
    )
    parser.add_argument(
        "--shard-pattern", default="*.tar",
        help="Glob pattern for shards within --shards (default: *.tar)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-score shards that already have a .scored sentinel"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Unused (reserved for future parallel mode; GPU is the bottleneck)"
    )
    parser.add_argument(
        "--sample-n", type=int, default=SAMPLE_N,
        help=f"Images sampled per shard (default {SAMPLE_N})"
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device override (default: mps if available, else cpu)"
    )
    parser.add_argument(
        "--no-aesthetic", action="store_true",
        help="Skip loading the aesthetic predictor (faster, less accurate)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override combined-score threshold for all sources"
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="Pipeline YAML config (train/configs/v2_pipeline.yaml). Loads "
             "light_scoring.thresholds_per_source, .default_threshold, and .weights "
             "— command-line --threshold takes precedence over config values."
    )
    args = parser.parse_args()

    import torch  # type: ignore
    from PIL import Image  # type: ignore  # noqa: F401 (ensure PIL available early)

    # Device selection
    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load config-driven thresholds/weights (before CLI override so --threshold wins)
    if args.config:
        try:
            import yaml  # type: ignore
            with open(args.config) as _f:
                _cfg = yaml.safe_load(_f)
            _ls = _cfg.get("light_scoring", {})
            if _ls.get("default_threshold") is not None:
                for _k in SOURCE_THRESHOLDS:
                    SOURCE_THRESHOLDS[_k] = float(_ls["default_threshold"])
            for _src, _t in _ls.get("thresholds_per_source", {}).items():
                SOURCE_THRESHOLDS[_src] = float(_t)
            _wc = _ls.get("weights", {})
            if _wc:
                _w = (
                    float(_wc.get("clip_alignment", SOURCE_WEIGHTS["unknown"][0])),
                    float(_wc.get("aesthetic",      SOURCE_WEIGHTS["unknown"][1])),
                    float(_wc.get("blur",           SOURCE_WEIGHTS["unknown"][2])),
                    float(_wc.get("caption_quality",SOURCE_WEIGHTS["unknown"][3])),
                )
                for _k in SOURCE_WEIGHTS:
                    SOURCE_WEIGHTS[_k] = _w
            print(f"  Config: loaded light_scoring from {args.config}")
        except Exception as _e:
            print(f"  WARNING: could not load --config {args.config}: {_e}")

    # CLI --threshold overrides everything (applied last)
    if args.threshold is not None:
        for k in SOURCE_THRESHOLDS:
            SOURCE_THRESHOLDS[k] = args.threshold

    shards_dir = Path(args.shards)
    if not shards_dir.exists():
        print(f"ERROR: --shards dir not found: {shards_dir}", file=sys.stderr)
        sys.exit(1)

    all_shards = sorted(glob.glob(str(shards_dir / args.shard_pattern)))
    # Exclude .provenance.json, .light_scores.json, .scored etc. that glob might catch
    all_shards = [s for s in all_shards if s.endswith(".tar") and not s.endswith(".tar.tmp")]

    if not all_shards:
        print(f"No shards found matching {shards_dir / args.shard_pattern}")
        sys.exit(0)

    pending = [s for s in all_shards if args.force or not _is_scored(s)]
    already_done = len(all_shards) - len(pending)

    print(f"score_shards_light: {len(all_shards)} total shards, "
          f"{already_done} already scored, {len(pending)} pending")
    print(f"  Device:  {device}")
    print(f"  Samples: {args.sample_n} per shard")

    if not pending:
        print("All shards already scored. Use --force to re-score.")
        sys.exit(0)

    # Load models
    print("Loading CLIP ViT-L/14 ...", flush=True)
    t0 = time.time()
    model, preprocess, tokenizer = _load_clip(device)
    print(f"  Loaded in {time.time() - t0:.1f}s", flush=True)

    aesthetic_predictor = None
    if not args.no_aesthetic:
        print("Loading aesthetic predictor ...", flush=True)
        aesthetic_predictor = _load_aesthetic_predictor(device)

    # Score loop
    n_keep = n_discard = n_error = 0
    start_ts = now_iso()

    for i, shard_path in enumerate(pending):
        shard_name = Path(shard_path).name
        print(f"[{i+1}/{len(pending)}] {shard_name} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            result = score_shard(
                shard_path, model, preprocess, tokenizer,
                aesthetic_predictor, device,
                sample_n=args.sample_n,
            )

            _write_result(shard_path, result)

            scores = result["light_scores"]
            print(
                f"{result['decision'].upper()} | "
                f"clip={scores['clip_alignment_avg']:.3f} "
                f"aesth={scores['aesthetic_avg']:.3f} "
                f"sharp={scores['blur_avg']:.3f} "
                f"cap={scores['caption_quality']:.3f} "
                f"→ {scores['combined_score']:.3f} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

            if result["decision"] == "keep":
                n_keep += 1
            else:
                n_discard += 1

        except Exception as e:
            import traceback
            print(f"ERROR: {e}", flush=True)
            traceback.print_exc()
            n_error += 1
            log_event("score_shards_light", "shard_error",
                      shard=shard_name, error=str(e))
            continue

        # Heartbeat every 10 shards
        if (i + 1) % 10 == 0:
            write_heartbeat(
                "score_shards_light", chunk=None,
                done=i + 1,
                total=len(pending),
                pct=round(100 * (i + 1) / len(pending)),
                n_keep=n_keep,
                n_discard=n_discard,
                n_error=n_error,
                start_ts=start_ts,
            )

    # Final heartbeat
    write_heartbeat(
        "score_shards_light", chunk=None,
        done=len(pending),
        total=len(pending),
        pct=100,
        n_keep=n_keep,
        n_discard=n_discard,
        n_error=n_error,
        start_ts=start_ts,
    )

    print(f"\nDone. keep={n_keep}  discard={n_discard}  error={n_error}")
    if n_keep + n_discard > 0:
        keep_pct = round(100 * n_keep / (n_keep + n_discard))
        print(f"  Keep rate: {keep_pct}%")
    print(f"  Sidecars: {shards_dir}/*.light_scores.json")


if __name__ == "__main__":
    main()
