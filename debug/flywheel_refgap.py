#!/usr/bin/env python
"""
flywheel_refgap.py — per-iteration signal progression + champion for a flywheel
campaign (read-only on the telemetry DB).

The campaign's health signal is not the noisy flow loss but whether cond_gap
(conditioned vs null loss — the adapter actually using the reference) and ref_gap
(self-ref vs cross-ref — style/content separation) trend more positive across
iterations as the warm-started checkpoint accumulates signal. cond_gap is the
metric FlywheelDB.get_best() ranks by, so the highest-cond_gap done iteration is
the recorded champion (the training-internal one — a real champion still needs a
golden-set output-quality win; see plans/c-ip-adapter.md).

Usage:
    flywheel_refgap.py [campaign_name] [db_path]
    (defaults: warmup-run2  /Volumes/2TBSSD/flywheel_history.db)
"""

import os
import sqlite3
import sys


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "warmup-run2"
    db = sys.argv[2] if len(sys.argv) > 2 else "/Volumes/2TBSSD/flywheel_history.db"
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = c.execute(
        "SELECT iteration, status, ref_gap, cond_gap, train_loss, steps, checkpoint "
        "FROM iterations WHERE flywheel_name=? ORDER BY iteration", (name,)).fetchall()
    if not rows:
        print(f"no iterations recorded for '{name}' yet")
        return 0

    def f(x):
        return f"{x:>9.4f}" if x is not None else f"{'—':>9}"

    def trend(cur, prev):
        if cur is None or prev is None:
            return " "
        return "▲" if cur > prev else ("▼" if cur < prev else "=")

    # champion = highest-cond_gap *done* iter (matches get_best's ORDER BY cond_gap DESC)
    done = [r for r in rows if r[1] == "done" and r[3] is not None]
    champ = max(done, key=lambda r: r[3]) if done else None
    champ_it = champ[0] if champ else None

    print(f"=== {name}: per-iteration signal ===")
    print(f"{'it':>3} {'status':<8} {'cond_gap':>9}{'':1} {'ref_gap':>9}{'':1} {'loss':>9} {'steps':>6}  ")
    pcg = prg = None
    for it, st, rg, cg, loss, steps, _ck in rows:
        star = " ★" if it == champ_it else "  "
        print(f"{it:>3} {st:<8} {f(cg)}{trend(cg,pcg)} {f(rg)}{trend(rg,prg)} {f(loss)} {steps or 0:>6}{star}")
        if cg is not None: pcg = cg
        if rg is not None: prg = rg

    # cond_gap trajectory + champion verdict
    cdone = [(r[0], r[3]) for r in done]
    if len(cdone) >= 2:
        a, b = cdone[0], cdone[-1]
        print(f"\ncond_gap: iter {a[0]}={a[1]:+.4f} → iter {b[0]}={b[1]:+.4f}  "
              f"(Δ {b[1]-a[1]:+.4f} over {len(cdone)} done iters) — "
              f"{'strengthening ✓' if b[1] > a[1] else 'NOT improving ⚠'}")
    if champ:
        ck = champ[6]
        cg = champ[3]
        flag = "below the 1%-of-null guard ⚠" if (cg is not None and 0 <= cg < 0.01) else "above the learning floor ✓"
        print(f"champion (max cond_gap, = get_best): iter {champ_it}  cond_gap={cg:+.4f}  ({flag})")
        if ck:
            print(f"  checkpoint: {ck}  ({'present' if os.path.exists(ck) else 'missing'})")
        print("  NOTE: this is the training-internal champion. A shippable champion also "
              "needs a golden-set output-quality win (clip_i/aesthetic vs baseline).")

    # Attribution warmth — is the shard bandit warm enough to ablate a recipe against?
    # Mirrors shard_selector: a shard's attribution becomes usable once attr_confidence
    # reaches 1.0 (≥MIN_ATTR_OBS=3 observations in BOTH the included and excluded roles).
    # The branch trigger for run3-with-ablation is "stall detector fires AND this warm".
    shard_db = os.path.join(os.path.dirname(os.path.abspath(db)), "shard_scores.db")
    if os.path.exists(shard_db):
        total = scored = attr = None
        try:
            sc = sqlite3.connect(f"file:{shard_db}?mode=ro", uri=True)
            total  = sc.execute("SELECT COUNT(*) FROM shards").fetchone()[0]
            scored = sc.execute("SELECT COUNT(*) FROM shards WHERE n_scored>0").fetchone()[0]
            attr   = sc.execute("SELECT COUNT(*) FROM shards WHERE attr_confidence>=1.0").fetchone()[0]
            sc.close()
        except sqlite3.Error:
            total = None
        if total is not None:
            # Readiness floor = 2 iterations' worth of fully-attributed shards, using the
            # campaign's own median per-iter shard count (falls back to 42).
            try:
                ns = [r[0] for r in c.execute(
                    "SELECT n_shards FROM iterations WHERE flywheel_name=? AND status='done'",
                    (name,)).fetchall() if r[0]]
            except sqlite3.Error:
                ns = []
            per_iter = sorted(ns)[len(ns) // 2] if ns else 42
            floor = 2 * per_iter
            ready = attr >= floor
            pct = 100.0 * attr / total if total else 0.0
            print(f"\nattribution: {attr}/{total} shards fully attributed ({pct:.1f}%) "
                  f"[attr_confidence≥1.0 = ≥3 incl & ≥3 excl obs]; {scored} touched")
            print(f"  ablation-ready: {'YES ✓' if ready else 'no — keep warming'} "
                  f"({attr} {'≥' if ready else '<'} {floor} floor = 2×~{per_iter} shards/iter)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
