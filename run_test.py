#!/usr/bin/env python3
"""
Iris test runner - verifies inference correctness against reference images.
Usage: python3 run_test.py [--iris-binary PATH] [--full]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Test cases: uses mean_diff threshold to allow bf16 precision differences
# while still catching actual bugs. Observed mean_diff values:
#   - 64x64: ~3.4, 512x512: ~1.7, img2img: ~6-17 (varies due to GPU non-determinism)
# Threshold of 20 accounts for GPU floating-point non-determinism while
# still catching catastrophic failures (wrong image would have mean_diff > 50).
# Optional: "input" for img2img tests
TESTS = [
    {
        "name": "64x64 quick test (2 steps)",
        "prompt": "A fluffy orange cat sitting on a windowsill",
        "seed": 42,
        "steps": 2,
        "width": 64,
        "height": 64,
        "reference": "test_vectors/reference_2step_64x64_seed42.png",
        "mean_diff_threshold": 20,
    },
    {
        "name": "512x512 full test (4 steps)",
        "prompt": "A red apple on a wooden table",
        "seed": 123,
        "steps": 4,
        "width": 512,
        "height": 512,
        "reference": "test_vectors/reference_4step_512x512_seed123.png",
        "mean_diff_threshold": 20,
    },
    {
        "name": "256x256 img2img test (4 steps)",
        "prompt": "A colorful oil painting of a cat",
        "seed": 456,
        "steps": 4,
        "width": 256,
        "height": 256,
        "input": "test_vectors/img2img_input_256x256.png",
        "reference": "test_vectors/reference_img2img_256x256_seed456.png",
        "mean_diff_threshold": 20,
    },
    {
        "name": "256x256 img2img strength=0.35 test (4 steps)",
        "prompt": "A colorful oil painting of a cat",
        "seed": 456,
        "steps": 4,
        "width": 256,
        "height": 256,
        "input": "test_vectors/img2img_input_256x256.png",
        "img2img_strength": 0.35,
        "reference": "test_vectors/reference_img2img_strength035_256x256_seed456.png",
        "mean_diff_threshold": 20,
    },
    {
        # strength=0.0 in iris means full denoising from scratch (no img2img
        # conditioning), equivalent to txt2img. Tests that the binary handles
        # this boundary value without crashing and produces a valid output.
        "name": "256x256 img2img strength=0.0 boundary (no crash, valid output)",
        "prompt": "A colorful oil painting of a cat",
        "seed": 456,
        "steps": 4,
        "width": 256,
        "height": 256,
        "input": "test_vectors/img2img_input_256x256.png",
        "img2img_strength": 0.0,
    },
]

# Full-only tests: these are slow and require visual inspection.
FULL_TESTS = [
    {
        "name": "1024x1024 img2img with attention budget shrinking (4 steps)",
        "prompt": "A blue sports car parked on a rainy city street at night",
        "seed": 99,
        "steps": 4,
        "width": 1024,
        "height": 1024,
        "expect_stderr": "reference image resized",
        "visual_check": "a blue sports car on a rainy city street at night, "
                        "output is 1024x1024",
    },
]

# Optional Z-Image smoke test.
# This runs only if a Z-Image model directory is auto-detected (or provided).
ZIMAGE_SMOKE_TEST = {
    "name": "Z-Image smoke test (2 steps, 256x256)",
    "prompt": "A simple geometric logo",
    "seed": 7,
    "steps": 2,
    "width": 256,
    "height": 256,
}


def is_zimage_model_dir(model_dir: Path) -> bool:
    """Return True if model_dir looks like a Z-Image model directory."""
    if not model_dir.is_dir():
        return False

    model_index = model_dir / "model_index.json"
    if not model_index.exists():
        return False

    try:
        text = model_index.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    return ("ZImagePipeline" in text) or ("Z-Image" in text)


def detect_zimage_model_dir(explicit_dir: Optional[str]) -> Optional[Path]:
    """Find a Z-Image model directory."""
    if explicit_dir:
        p = Path(explicit_dir)
        return p if is_zimage_model_dir(p) else None

    # Common local names first.
    for p in (Path("zimage-turbo"), Path("zimage"), Path("Z-Image-Turbo")):
        if is_zimage_model_dir(p):
            return p

    # Fallback: scan direct subdirectories.
    for p in sorted(Path(".").iterdir()):
        if p.is_dir() and is_zimage_model_dir(p):
            return p

    return None


def run_test(iris_binary: str, test: dict, model_dir: str) -> tuple[bool, str]:
    """Run a single test case. Returns (passed, message)."""
    if "output" in test:
        output_path = test["output"]
    else:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

    cmd = [
        iris_binary,
        "-d", model_dir,
        "-p", test["prompt"],
        "--seed", str(test["seed"]),
        "--steps", str(test["steps"]),
        "-W", str(test["width"]),
        "-H", str(test["height"]),
        "-o", output_path,
    ]

    # Add input image for img2img tests
    if "input" in test:
        cmd.extend(["-i", test["input"]])

    # Add img2img strength if specified
    if "img2img_strength" in test:
        cmd.extend(["--img2img-strength", str(test["img2img_strength"])])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return False, f"flux exited with code {result.returncode}: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "timeout (300s)"
    except FileNotFoundError:
        return False, f"binary not found: {iris_binary}"

    # If the test has no reference image, it's a visual-check-only test.
    if "reference" not in test:
        # Verify output was created and has expected dimensions.
        try:
            out = Image.open(output_path)
        except Exception as e:
            return False, f"failed to load output: {e}"
        if out.width != test["width"] or out.height != test["height"]:
            return False, (f"wrong output size: {out.width}x{out.height}, "
                           f"expected {test['width']}x{test['height']}")

        # Check expected stderr substring (e.g. the resize note).
        if "expect_stderr" in test:
            if test["expect_stderr"] not in result.stderr:
                return False, (f"expected '{test['expect_stderr']}' in "
                               f"stderr but not found")

        return True, f"output saved to {output_path}"

    # Compare images against reference
    try:
        ref = np.array(Image.open(test["reference"]))
        out = np.array(Image.open(output_path))
    except Exception as e:
        return False, f"failed to load images: {e}"

    if ref.shape != out.shape:
        return False, f"shape mismatch: ref={ref.shape}, out={out.shape}"

    diff = np.abs(ref.astype(float) - out.astype(float))
    max_diff = diff.max()
    mean_diff = diff.mean()

    threshold = test["mean_diff_threshold"]
    if mean_diff <= threshold:
        return True, f"mean_diff={mean_diff:.2f}, max_diff={max_diff:.0f}"
    else:
        return False, f"mean_diff={mean_diff:.2f} > {threshold} (max={max_diff:.0f})"


def run_server_mode_test(iris_binary: str, model_dir: str) -> tuple[bool, str]:
    """
    Smoke-test the --server JSON IPC mode.

    Spawns the binary with --server, sends one generation request via stdin,
    reads stdout events until 'complete' or 'error', then verifies the output.
    """
    output_path = "/tmp/iris_server_mode_test.png"
    request = {
        "prompt": "A simple blue circle",
        "output": output_path,
        "seed": 1,
        "steps": 2,
        "width": 64,
        "height": 64,
        "show_steps": False,
    }

    try:
        proc = subprocess.Popen(
            [iris_binary, "-d", model_dir, "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except FileNotFoundError:
        return False, f"binary not found: {iris_binary}"

    try:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        proc.stdin.close()

        events = []
        last_event = None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                events.append(ev)
                last_event = ev
                if ev.get("event") in ("complete", "error"):
                    break
            except json.JSONDecodeError:
                pass

        proc.wait(timeout=60)

    except subprocess.TimeoutExpired:
        proc.kill()
        return False, "timeout (60s)"
    except Exception as e:
        proc.kill()
        return False, str(e)

    if not events:
        return False, "no events received from server"

    if last_event is None or last_event.get("event") != "complete":
        err = last_event.get("message", "unknown error") if last_event else "no complete event"
        return False, f"server error: {err}"

    if not Path(output_path).exists():
        return False, f"output file not created: {output_path}"

    try:
        img = Image.open(output_path)
        if img.width != 64 or img.height != 64:
            return False, f"wrong output size: {img.width}x{img.height}"
    except Exception as e:
        return False, f"output is not a valid image: {e}"

    return True, (f"server emitted {len(events)} events, "
                  f"output={output_path} ({img.width}x{img.height})")


FLUX_MODEL_SEARCH_ORDER = [
    "flux-klein-4b", "flux-klein-model", "flux-klein-9b",
    "flux-klein-4b-base", "flux-klein-9b-base",
]
FLUX_BASE_MODEL_SEARCH_ORDER = ["flux-klein-4b-base", "flux-klein-9b-base"]

# Optional base-model smoke test: exercises the CFG code path (double forward
# pass per step). Runs only when a base-model directory is detected.
BASE_MODEL_SMOKE_TEST = {
    "name": "Base model smoke test (2 steps, 64x64, CFG path)",
    "prompt": "A simple mountain landscape",
    "seed": 1,
    "steps": 2,
    "width": 64,
    "height": 64,
}


def detect_flux_model_dir() -> Optional[str]:
    for name in FLUX_MODEL_SEARCH_ORDER:
        if Path(name).is_dir():
            return name
    return None


def detect_base_model_dir() -> Optional[str]:
    for name in FLUX_BASE_MODEL_SEARCH_ORDER:
        if Path(name).is_dir():
            return name
    return None


def main():
    parser = argparse.ArgumentParser(description="Run iris inference tests")
    parser.add_argument("--iris-binary", default="./iris", help="Path to iris binary")
    parser.add_argument("--model-dir", default=None, help="Path to model (auto-detected if omitted)")
    parser.add_argument("--zimage-model-dir", default=None,
                        help="Optional Z-Image model dir (auto-detected if omitted)")
    parser.add_argument("--quick", action="store_true", help="Run only the quick 64x64 test")
    parser.add_argument("--full", action="store_true",
                        help="Also run slow tests that require visual inspection")
    args = parser.parse_args()

    if args.model_dir is None:
        args.model_dir = detect_flux_model_dir()
        if args.model_dir is None:
            print("Error: no Flux model dir found. Pass --model-dir or download a model.")
            sys.exit(1)
        print(f"Detected Flux model dir: {args.model_dir}")

    if args.quick:
        tests_to_run = TESTS[:1]
    else:
        tests_to_run = list(TESTS)
    full_tests_to_run = list(FULL_TESTS) if args.full else []

    # Optional zImage coverage: run only in non-quick mode.
    scheduled_tests: list[tuple[dict, str]] = [(t, args.model_dir) for t in tests_to_run]
    zimage_dir = detect_zimage_model_dir(args.zimage_model_dir)
    if not args.quick:
        if zimage_dir:
            print(f"Detected Z-Image model dir: {zimage_dir}")
            scheduled_tests.append((ZIMAGE_SMOKE_TEST, str(zimage_dir)))
        elif args.zimage_model_dir:
            print(f"Warning: --zimage-model-dir '{args.zimage_model_dir}' is not a valid Z-Image model dir")
            print("Skipping optional Z-Image smoke test.")
        else:
            print("No Z-Image model dir detected; skipping optional Z-Image smoke test.")

    # Optional base-model coverage (CFG path): run only in non-quick mode.
    base_model_dir = detect_base_model_dir()
    if not args.quick:
        if base_model_dir:
            print(f"Detected base model dir: {base_model_dir}")
            scheduled_tests.append((BASE_MODEL_SMOKE_TEST, base_model_dir))
        else:
            print("No base model dir detected; skipping optional base model smoke test.")

    # Server-mode IPC test runs in non-quick mode.
    run_server_test = not args.quick

    total = len(scheduled_tests) + len(full_tests_to_run) + (1 if run_server_test else 0)
    print(f"Running {total} test(s)...\n")

    passed = 0
    failed = 0
    visual_checks = []

    for i, (test, model_dir) in enumerate(scheduled_tests, 1):
        print(f"[{i}/{total}] {test['name']}...")
        ok, msg = run_test(args.iris_binary, test, model_dir)

        if ok:
            print(f"    PASS: {msg}")
            passed += 1
        else:
            print(f"    FAIL: {msg}")
            failed += 1

    for j, test in enumerate(full_tests_to_run, len(scheduled_tests) + 1):
        print(f"[{j}/{total}] {test['name']}...")

        # Step 1: Generate a reference image to use as img2img input.
        ref_path = "/tmp/iris_test_ref_1024.png"
        print(f"    Step 1: Generating 1024x1024 reference image...")
        ref_cmd = [
            args.iris_binary, "-d", args.model_dir,
            "-p", "A red sports car parked on a sunny city street",
            "--seed", "42", "--steps", "4",
            "-W", "1024", "-H", "1024", "-o", ref_path,
        ]
        try:
            r = subprocess.run(ref_cmd, capture_output=True, text=True,
                               timeout=300)
            if r.returncode != 0:
                print(f"    FAIL: could not generate reference: {r.stderr}")
                failed += 1
                continue
        except Exception as e:
            print(f"    FAIL: {e}")
            failed += 1
            continue
        print(f"    Step 1: Done ({ref_path})")

        # Step 2: Run img2img with the reference — this should trigger
        # the attention budget shrinking and print a resize note.
        output_path = "/tmp/iris_test_img2img_1024.png"
        print(f"    Step 2: Running img2img with attention budget "
              f"shrinking (reference should be auto-resized)...")
        test_with_input = dict(test)
        test_with_input["input"] = ref_path
        test_with_input["output"] = output_path
        ok, msg = run_test(args.iris_binary, test_with_input, args.model_dir)

        if ok:
            print(f"    Step 2: Done ({output_path})")
            print(f"    PASS: {msg}")
            passed += 1
            if "visual_check" in test:
                visual_checks.append((test["name"], output_path,
                                      test["visual_check"]))
        else:
            print(f"    FAIL: {msg}")
            failed += 1

    if run_server_test:
        server_idx = len(scheduled_tests) + len(full_tests_to_run) + 1
        print(f"[{server_idx}/{total}] Server-mode IPC smoke test...")
        ok, msg = run_server_mode_test(args.iris_binary, args.model_dir)
        if ok:
            print(f"    PASS: {msg}")
            passed += 1
        else:
            print(f"    FAIL: {msg}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")

    if visual_checks:
        print("\n--- Visual verification needed ---")
        for name, out_file, description in visual_checks:
            print(f"  {name}:")
            print(f"    Open: {out_file}")
            print(f"    Check: {description}")

    if not args.full and not args.quick:
        print("\nTo run a more complete (and slow) test, run with --full")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
