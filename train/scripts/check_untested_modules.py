#!/usr/bin/env python3
"""
check_untested_modules.py — flag Python source modules that have no test.

Part of GROK-TEST-8. Greps the main Python source dirs for modules that lack a
corresponding test, so test coverage gaps surface in CI instead of silently
accumulating. Pragmatic by design:

  * Only modules above a line threshold (default 50) are flagged — tiny helper
    files and thin shims are not worth a dedicated test file.
  * A module counts as "tested" if EITHER a `test_<stem>.py` exists OR the module
    stem is imported/referenced anywhere under the tests dir. The reference check
    catches the common case where one test file exercises a module whose name it
    does not mirror (e.g. test_golden_gate.py tests evaluate_golden_set.py).

Exit status:
  0  — no untested modules (or warn-only mode).
  1  — untested modules found AND --strict was passed.

Usage:
  python scripts/check_untested_modules.py            # warn, exit 0
  python scripts/check_untested_modules.py --strict   # fail CI on gaps
  python scripts/check_untested_modules.py --min-lines 80
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# Resolve repo layout relative to this file (train/scripts/check_untested_modules.py).
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.dirname(SCRIPTS_DIR)
REPO_DIR = os.path.dirname(TRAIN_DIR)
TESTS_DIR = os.path.join(TRAIN_DIR, "tests")

# Source dirs scanned for modules. (No top-level data/ dir in this repo.)
SOURCE_DIRS = [
    os.path.join(TRAIN_DIR, "scripts"),
    os.path.join(TRAIN_DIR, "ip_adapter"),
    os.path.join(REPO_DIR, "debug"),
]

# Modules that are never expected to carry their own test file.
SKIP_NAMES = {"__init__", "conftest", "setup"}


def _iter_modules(source_dirs):
    """Yield (path, stem, line_count) for non-test .py files in source_dirs."""
    for d in source_dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".py"):
                continue
            stem = name[:-3]
            if stem in SKIP_NAMES or stem.startswith("test_"):
                continue
            path = os.path.join(d, name)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    line_count = sum(1 for _ in fh)
            except OSError:
                continue
            yield path, stem, line_count


def _collect_test_text(tests_dir):
    """Return (set_of_test_stems, concatenated_test_source)."""
    test_stems = set()
    blobs = []
    for root, _dirs, files in os.walk(tests_dir):
        for name in files:
            if not name.endswith(".py"):
                continue
            if name.startswith("test_"):
                test_stems.add(name[len("test_"):-len(".py")])
            try:
                with open(os.path.join(root, name), "r", encoding="utf-8",
                          errors="replace") as fh:
                    blobs.append(fh.read())
            except OSError:
                continue
    return test_stems, "\n".join(blobs)


def _is_referenced(stem, test_text):
    """True if `stem` is imported or named as a whole word in any test file."""
    return re.search(r"\b" + re.escape(stem) + r"\b", test_text) is not None


def find_untested(source_dirs, tests_dir, min_lines):
    test_stems, test_text = _collect_test_text(tests_dir)
    untested = []
    for path, stem, line_count in _iter_modules(source_dirs):
        if line_count < min_lines:
            continue
        if stem in test_stems:
            continue
        if _is_referenced(stem, test_text):
            continue
        untested.append((os.path.relpath(path, REPO_DIR), line_count))
    return untested


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-lines", type=int, default=50,
                    help="only flag modules with more than this many lines (default 50)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any untested module is found")
    args = ap.parse_args(argv)

    untested = find_untested(SOURCE_DIRS, TESTS_DIR, args.min_lines)

    if not untested:
        print(f"check_untested_modules: OK — every source module (>{args.min_lines} "
              f"lines) has a test or is referenced from a test.")
        return 0

    label = "ERROR" if args.strict else "WARNING"
    print(f"check_untested_modules: {label} — {len(untested)} module(s) "
          f"(>{args.min_lines} lines) with no corresponding test:")
    for relpath, line_count in sorted(untested, key=lambda x: -x[1]):
        print(f"  {relpath}  ({line_count} lines)")
    print("\nAdd a tests/test_<module>.py, reference the module from an existing "
          "test, or (if it is genuinely untestable glue) raise --min-lines.")

    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
