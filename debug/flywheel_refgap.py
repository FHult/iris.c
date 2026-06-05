#!/usr/bin/env python
"""
flywheel_refgap.py — per-iteration ref_gap / cond_gap progression for a flywheel
campaign (read-only on the telemetry DB).

The campaign's health signal is not the noisy flow loss but whether ref_gap
(self-ref vs cross-ref loss) and cond_gap (conditioned vs null) trend more
positive across iterations as the warm-started checkpoint accumulates signal.

Usage:
    flywheel_refgap.py [campaign_name] [db_path]
    (defaults: warmup-run2  /Volumes/2TBSSD/flywheel_history.db)
"""

import sqlite3
import sys


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "warmup-run2"
    db = sys.argv[2] if len(sys.argv) > 2 else "/Volumes/2TBSSD/flywheel_history.db"
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = c.execute(
        "SELECT iteration, status, ref_gap, cond_gap, train_loss, steps, elapsed_secs "
        "FROM iterations WHERE flywheel_name=? ORDER BY iteration", (name,)).fetchall()
    if not rows:
        print(f"no iterations recorded for '{name}' yet")
        return 0

    def f(x):
        return f"{x:>9.4f}" if x is not None else f"{'—':>9}"

    print(f"=== {name}: per-iteration signal ===")
    print(f"{'it':>3} {'status':<8} {'ref_gap':>9} {'cond_gap':>9} {'loss':>9} {'steps':>6}  trend")
    prev = None
    for it, st, rg, cg, loss, steps, _el in rows:
        trend = ""
        if rg is not None and prev is not None:
            trend = "▲" if rg > prev else ("▼" if rg < prev else "=")
        if rg is not None:
            prev = rg
        print(f"{it:>3} {st:<8} {f(rg)} {f(cg)} {f(loss)} {steps or 0:>6}  {trend}")

    done = [(it, rg) for it, st, rg, *_ in rows if st == "done" and rg is not None]
    if len(done) >= 2:
        a, b = done[0], done[-1]
        print(f"\nref_gap: iter {a[0]}={a[1]:+.4f} → iter {b[0]}={b[1]:+.4f}  "
              f"(Δ {b[1]-a[1]:+.4f} over {len(done)} done iters) — "
              f"{'strengthening ✓' if b[1] > a[1] else 'NOT improving ⚠'}")
    elif done:
        print(f"\nref_gap baseline: iter {done[0][0]} = {done[0][1]:+.4f} (need ≥2 done iters for a trend)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
