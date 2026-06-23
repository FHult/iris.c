#!/usr/bin/env python
"""
cond_attrib_report.py — aggregate the trainer's per-record cond-attrib JSONL into a
per-shard (and overall) conditioned-vs-null loss readout.

The trainer writes cond_attrib.jsonl (one line per step: {step, id, shard, null, cross_ref,
loss}) when training.log_cond_attrib is true. cond and null steps fall on DIFFERENT samples
(the image_dropout coin flip), so this is NOT a paired per-record cond_gap — it's a per-shard
mean conditioned-loss vs mean null-loss signal: lower conditioned loss (relative to that shard's
null loss, or the global null) ⇒ the adapter reconstructs that shard's data better WHEN
conditioned ⇒ that data is "teaching" the conditioning more. Use it to spot shards worth
keeping/dropping; it is a data-quality proxy, not the eval metric (null-relative style/leak).

Usage:
  train/.venv/bin/python train/scripts/cond_attrib_report.py CKPT_DIR/cond_attrib.jsonl [--top 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", help="path to cond_attrib.jsonl")
    ap.add_argument("--top", type=int, default=20, help="show top/bottom N shards by cond-benefit")
    ap.add_argument("--min-cond", type=int, default=3,
                    help="min conditioned samples per shard to rank it (noise floor)")
    args = ap.parse_args()

    p = Path(args.jsonl)
    if not p.exists():
        print(f"not found: {p}", file=sys.stderr)
        return 1

    # per-shard accumulators: [cond_sum, cond_n, null_sum, null_n]
    sh = defaultdict(lambda: [0.0, 0, 0.0, 0])
    g_cond_sum = g_null_sum = 0.0
    g_cn = g_nn = 0
    n_lines = 0
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        n_lines += 1
        s = sh[r["shard"]]
        if r["null"]:
            s[2] += r["loss"]; s[3] += 1
            g_null_sum += r["loss"]; g_nn += 1
        else:
            s[0] += r["loss"]; s[1] += 1
            g_cond_sum += r["loss"]; g_cn += 1

    if not n_lines:
        print("empty JSONL (no steps logged yet)")
        return 0

    g_cond = g_cond_sum / g_cn if g_cn else float("nan")
    g_null = g_null_sum / g_nn if g_nn else float("nan")
    print(f"lines={n_lines}  shards={len(sh)}  global: cond={g_cond:.4f} (n={g_cn})  "
          f"null={g_null:.4f} (n={g_nn})  cond_gap(null-cond)={g_null - g_cond:+.4f}")

    # Per-shard: conditioned-loss benefit vs the GLOBAL null (robust when a shard has few/no
    # null samples of its own). Lower cond loss ⇒ more positive benefit.
    rows = []
    for name, (cs, cn, ns, nn) in sh.items():
        if cn < args.min_cond:
            continue
        cond = cs / cn
        benefit = g_null - cond          # how far below the global null this shard's cond loss sits
        rows.append((benefit, name, cond, cn, (ns / nn if nn else None), nn))
    rows.sort(reverse=True)

    def show(rs, label):
        print(f"\n{label} (cond-benefit = global_null − shard_cond_loss):")
        print(f"  {'shard':<24} {'benefit':>9} {'cond':>8} {'n_cond':>7} {'null':>8} {'n_null':>7}")
        for benefit, name, cond, cn, nullv, nn in rs:
            nulls = f"{nullv:.4f}" if nullv is not None else "   -  "
            print(f"  {name:<24} {benefit:>+9.4f} {cond:>8.4f} {cn:>7d} {nulls:>8} {nn:>7d}")

    show(rows[: args.top], f"TOP {args.top} shards (adapter conditions BEST)")
    if len(rows) > args.top:
        show(rows[-args.top:], f"BOTTOM {args.top} shards (adapter conditions WORST)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
