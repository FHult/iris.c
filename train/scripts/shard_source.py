"""
shard_source.py — authoritative shard source label from provenance.json.

The per-shard ``<shard>.provenance.json`` (written at shard-build time) is the
ground truth for which dataset(s) a shard's images came from: it lists every
contributing source tgz with a ``type`` tag (jdb / coyo / laion / wikiart).

Earlier source labeling was wrong in two places, both fixed to use this module:
  - shard_index.py copied the light-scorer's ``source``, which is "unknown" for
    any multi-source shard (the scorer can't pick one) — 123/1280 shards unknown.
  - shard_selector._infer_source() used a shard-ID-range heuristic
    (``id < 400000 -> journeydb``) that mislabeled every shard as "journeydb".

Labeling convention (operator-chosen):
  - Pure shard            -> its single canonical type, e.g. "journeydb".
  - Multi-source shard    -> sorted unique canonical types joined by "+", e.g.
                            "journeydb+wikiart", "coyo+laion+wikiart".

This is deterministic and lossless: a mix becomes its own diversity bucket.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# provenance `type` -> canonical source label. Matches the single-source labels
# the light scorer already wrote for pure shards (jdb -> journeydb; rest pass
# through). Unknown/blank types collapse to "unknown".
_TYPE_TO_SOURCE = {
    "jdb": "journeydb",
    "journeydb": "journeydb",
    "coyo": "coyo",
    "laion": "laion",
    "wikiart": "wikiart",
}


def normalize_source_type(t: Optional[str]) -> str:
    """Map a raw provenance `type` to its canonical source label."""
    key = (t or "").lower().strip()
    if not key:
        return "unknown"
    return _TYPE_TO_SOURCE.get(key, key)


def source_from_provenance(provenance_path) -> Optional[str]:
    """Canonical source label for a shard from its provenance.json.

    Returns None when the file is missing/unreadable or lists no source types
    (callers decide the fallback). Pure shard -> single type; multi-source ->
    sorted unique types joined by "+".
    """
    try:
        data = json.loads(Path(provenance_path).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    srcs = data.get("sources") or []
    types = {normalize_source_type(s.get("type")) for s in srcs if s.get("type")}
    types.discard("unknown")
    types.discard("")
    if not types:
        return None
    return "+".join(sorted(types))


def source_for_tar(tar_path) -> Optional[str]:
    """Derive source from a shard tar's sibling ``<stem>.provenance.json``."""
    return source_from_provenance(Path(tar_path).with_suffix(".provenance.json"))
