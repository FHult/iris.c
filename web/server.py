#!/usr/bin/env python3
"""
FLUX.2 Web UI - Simple Flask server for image generation.

Uses flux binary in server mode for persistent model (faster subsequent generations).

Usage:
    python web/server.py [--port PORT] [--model-dir PATH] [--flux-binary PATH]

Requirements:
    pip install flask
"""

import argparse
import base64
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "flux-klein-model"
DEFAULT_FLUX_BINARY = PROJECT_DIR / "flux"
OUTPUT_DIR = SCRIPT_DIR / "output"
THUMB_DIR = SCRIPT_DIR / "output" / "thumbs"
HISTORY_FILE = OUTPUT_DIR / "history.json"
THUMB_SIZE = 200  # Max dimension for thumbnails

app = Flask(__name__, static_folder="static")


def convert_image_to_png(image_data: bytes) -> bytes:
    """
    Convert any image format to PNG for the C code.
    Supports JPEG, WebP, GIF, BMP, TIFF, and any other PIL-supported format.
    Handles animated images (uses first frame) and transparency.
    """
    # Quick check: if already PNG, return as-is
    if image_data[:8] == b'\x89PNG\r\n\x1a\n':
        return image_data

    try:
        from PIL import Image
        import io

        # Try to open the image with PIL
        img = Image.open(io.BytesIO(image_data))

        # Force load the image data (important for some formats)
        img.load()

        # Handle animated images (GIF, WebP) - use first frame
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            img.seek(0)
            img.load()

        # Convert palette and other modes to RGBA or RGB
        if img.mode == 'P':
            img = img.convert('RGBA')
        elif img.mode == 'LA':
            img = img.convert('RGBA')
        elif img.mode not in ('RGB', 'RGBA'):
            img = img.convert('RGB')

        # Save as PNG
        png_buffer = io.BytesIO()
        img.save(png_buffer, format='PNG')
        return png_buffer.getvalue()

    except ImportError:
        raise ValueError("PIL/Pillow required for non-PNG image formats. Install with: pip install Pillow")
    except Exception as e:
        raise ValueError(f"Failed to convert image: {e}")


# Store active jobs and their progress
jobs = {}
jobs_lock = threading.Lock()

# Store history of completed generations
history = []
history_lock = threading.Lock()
MAX_HISTORY = 10000  # Effectively unlimited; frontend handles pagination

# Flux server process manager
flux_server = None
flux_server_lock = threading.Lock()


def save_history():
    """Save history to JSON file (must be called with history_lock held)."""
    try:
        data = [job.to_history_dict() for job in history]
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Warning: Failed to save history: {e}")


def load_history_from_disk():
    """Load history from JSON file on startup."""
    if not HISTORY_FILE.exists():
        return
    try:
        with open(HISTORY_FILE) as f:
            data = json.load(f)
        for item in data:
            job_id = item["id"]
            output_path = OUTPUT_DIR / f"{job_id}.png"
            if not output_path.exists():
                continue
            job = Job(job_id, prompt=item.get("prompt", ""),
                      width=item.get("width", 512), height=item.get("height", 512),
                      steps=item.get("steps", 4))
            job.seed = item.get("seed")
            job.created_at = item.get("created_at", 0)
            job.status = "complete"
            job.output_path = output_path
            history.append(job)
        print(f"Loaded {len(history)} history items from disk")
    except Exception as e:
        print(f"Warning: Failed to load history: {e}")


class Job:
    def __init__(self, job_id, prompt="", width=512, height=512, steps=4):
        self.id = job_id
        self.status = "pending"
        self.progress = 0
        self.total_steps = steps
        self.phase = "Starting"
        self.error = None
        self.output_path = None
        self.seed = None
        self.queue = queue.Queue()
        # Store generation parameters for history/remix
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.created_at = time.time()
        # Temp files to clean up after job completes
        self.temp_files = []

    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "total_steps": self.total_steps,
            "phase": self.phase,
            "error": self.error,
            "seed": self.seed,
        }

    def to_history_dict(self):
        return {
            "id": self.id,
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "seed": self.seed,
            "created_at": self.created_at,
            "image_url": f"/image/{self.id}",
            "thumb_url": f"/thumb/{self.id}",
        }

    def cleanup_temp_files(self):
        """Clean up any temp files associated with this job."""
        for path in self.temp_files:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass
        self.temp_files = []


class FluxServer:
    """Manages a persistent flux process in server mode."""

    def __init__(self, flux_binary, model_dir):
        self.flux_binary = flux_binary
        self.model_dir = model_dir
        self.process = None
        self.ready = False
        self.lock = threading.Lock()
        self.current_job = None

    def start(self):
        """Start the flux server process."""
        cmd = [
            str(self.flux_binary),
            "-d", str(self.model_dir),
            "--server",
        ]

        print(f"Starting flux server: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # Unbuffered
        )

        # Start a thread to read stderr (model loading messages)
        stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        stderr_thread.start()

        # Wait for ready signal from stdout
        for line in iter(self.process.stdout.readline, b''):
            line = line.decode('utf-8').strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("event") == "ready":
                    self.ready = True
                    print("Flux server ready")
                    break
            except json.JSONDecodeError:
                print(f"Flux server: {line}")

        # Start output reader thread
        reader_thread = threading.Thread(target=self._read_output, daemon=True)
        reader_thread.start()

    def _read_stderr(self):
        """Read stderr and print loading messages."""
        for line in iter(self.process.stderr.readline, b''):
            line = line.decode('utf-8', errors='replace').strip()
            if line:
                print(f"Flux: {line}")

    def _read_output(self):
        """Read JSON events from stdout and route to current job."""
        for line in iter(self.process.stdout.readline, b''):
            line = line.decode('utf-8').strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Route event to current job
            job = self.current_job
            if not job:
                continue

            event_type = event.get("event")

            if event_type == "status":
                seed = event.get("seed")
                if seed:
                    job.seed = seed
                    job.queue.put(("status", job.to_dict()))

            elif event_type == "phase":
                phase = event.get("phase", "")
                elapsed = event.get("elapsed", 0)
                # Capitalize first letter
                if phase:
                    phase = phase[0].upper() + phase[1:]
                job.phase = phase
                data = job.to_dict()
                data["elapsed"] = elapsed
                job.queue.put(("status", data))

            elif event_type == "phase_done":
                phase = event.get("phase", "")
                phase_time = event.get("phase_time", 0)
                elapsed = event.get("elapsed", 0)
                if phase:
                    phase = phase[0].upper() + phase[1:]
                data = job.to_dict()
                data["phase_done"] = phase
                data["phase_time"] = phase_time
                data["elapsed"] = elapsed
                job.queue.put(("status", data))

            elif event_type == "progress":
                step = event.get("step", 0)
                total = event.get("total", job.total_steps)
                step_time = event.get("step_time", 0)
                elapsed = event.get("elapsed", 0)
                job.progress = step
                job.total_steps = total
                job.phase = "Denoising"
                data = job.to_dict()
                data["step_time"] = step_time
                data["elapsed"] = elapsed
                job.queue.put(("progress", data))

            elif event_type == "complete":
                job.status = "complete"
                job.output_path = OUTPUT_DIR / f"{job.id}.png"
                job.progress = job.total_steps
                total_time = event.get("total_time", 0)
                job.queue.put(("complete", {
                    "image_url": f"/image/{job.id}",
                    "total_time": total_time
                }))
                job.queue.put(("done", None))
                # Clean up temp files now that generation is complete
                job.cleanup_temp_files()
                # Add to history
                with history_lock:
                    history.insert(0, job)
                    if len(history) > MAX_HISTORY:
                        history.pop()
                    save_history()
                self.current_job = None

            elif event_type == "error":
                job.status = "error"
                job.error = event.get("message", "Unknown error")
                job.queue.put(("error", {"message": job.error}))
                job.queue.put(("done", None))
                # Clean up temp files on error
                job.cleanup_temp_files()
                self.current_job = None

    def generate(self, job, prompt, width, height, steps, seed, input_image_path, reference_image_paths=None):
        """Send a generation request to the flux server."""
        with self.lock:
            if not self.ready or self.process.poll() is not None:
                raise RuntimeError("Flux server not running")

            self.current_job = job
            job.status = "running"
            job.queue.put(("status", job.to_dict()))

            # Build request
            output_path = OUTPUT_DIR / f"{job.id}.png"
            request_data = {
                "prompt": prompt,
                "output": str(output_path),
                "width": width,
                "height": height,
                "steps": steps,
            }

            if seed is not None and seed != "":
                request_data["seed"] = int(seed)

            # Multi-reference mode (new) takes precedence
            if reference_image_paths and len(reference_image_paths) > 0:
                request_data["reference_images"] = [str(p) for p in reference_image_paths]
            elif input_image_path:
                # Backwards compatibility: single image mode
                request_data["input_image"] = str(input_image_path)

            # Send request
            request_line = json.dumps(request_data) + "\n"
            try:
                self.process.stdin.write(request_line.encode('utf-8'))
                self.process.stdin.flush()
            except BrokenPipeError:
                raise RuntimeError("Flux server crashed")

    def stop(self):
        """Stop the flux server process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            self.ready = False


def run_generation_server_mode(job, prompt, width, height, steps, seed, input_image_path, reference_image_paths=None):
    """Run generation using the persistent flux server."""
    global flux_server
    try:
        flux_server.generate(job, prompt, width, height, steps, seed, input_image_path, reference_image_paths)
        # Request sent successfully - temp files will be cleaned up in _read_output
        # when the job completes or errors
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.queue.put(("error", {"message": str(e)}))
        job.queue.put(("done", None))
        # Clean up temp files immediately on send failure
        job.cleanup_temp_files()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Start a new image generation job."""
    global flux_server

    data = request.json or {}

    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    width = int(data.get("width", 512))
    height = int(data.get("height", 512))
    steps = int(data.get("steps", 4))
    seed = data.get("seed")

    # Validate dimensions
    if width < 64 or width > 1792 or width % 16 != 0:
        return jsonify({"error": "Width must be 64-1792 and divisible by 16"}), 400
    if height < 64 or height > 1792 or height % 16 != 0:
        return jsonify({"error": "Height must be 64-1792 and divisible by 16"}), 400
    if steps < 1 or steps > 256:
        return jsonify({"error": "Steps must be 1-256"}), 400

    # Handle reference images
    reference_image_paths = []
    input_image_path = None
    reference_images_b64 = data.get("reference_images", [])

    # Filter out None/empty values
    reference_images_b64 = [img for img in reference_images_b64 if img] if reference_images_b64 else []

    if len(reference_images_b64) == 1:
        # Single image: use img2img mode for better results
        try:
            image_data = base64.b64decode(reference_images_b64[0].split(",")[-1])
            # Convert any format to PNG
            image_data = convert_image_to_png(image_data)
            input_image_path = OUTPUT_DIR / f"temp_{uuid.uuid4().hex}.png"
            with open(input_image_path, "wb") as f:
                f.write(image_data)
        except Exception as e:
            return jsonify({"error": f"Invalid input image: {e}"}), 400
    elif len(reference_images_b64) > 1:
        # Multiple images: use multiref mode
        for i, img_b64 in enumerate(reference_images_b64[:2]):
            try:
                image_data = base64.b64decode(img_b64.split(",")[-1])
                # Convert any format to PNG
                image_data = convert_image_to_png(image_data)
                ref_path = OUTPUT_DIR / f"temp_{uuid.uuid4().hex}_ref{i}.png"
                with open(ref_path, "wb") as f:
                    f.write(image_data)
                reference_image_paths.append(ref_path)
            except Exception as e:
                # Clean up any already saved reference images
                for p in reference_image_paths:
                    try:
                        os.unlink(p)
                    except:
                        pass
                return jsonify({"error": f"Invalid reference image {i+1}: {e}"}), 400

    # Also check for legacy input_image parameter (backwards compatibility)
    if not input_image_path and not reference_image_paths:
        input_image_b64 = data.get("input_image")
        if input_image_b64:
            try:
                image_data = base64.b64decode(input_image_b64.split(",")[-1])
                # Convert any format to PNG
                image_data = convert_image_to_png(image_data)
                input_image_path = OUTPUT_DIR / f"temp_{uuid.uuid4().hex}.png"
                with open(input_image_path, "wb") as f:
                    f.write(image_data)
            except Exception as e:
                return jsonify({"error": f"Invalid input image: {e}"}), 400

    # Create job
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, prompt=prompt, width=width, height=height, steps=steps)

    # Store temp file paths in job for cleanup after generation completes
    if input_image_path:
        job.temp_files.append(str(input_image_path))
    if reference_image_paths:
        job.temp_files.extend([str(p) for p in reference_image_paths])

    with jobs_lock:
        jobs[job_id] = job

    # Start generation (server mode handles this via the persistent process)
    thread = threading.Thread(
        target=run_generation_server_mode,
        args=(job, prompt, width, height, steps, seed, input_image_path, reference_image_paths),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "started"})


@app.route("/progress/<job_id>")
def progress(job_id):
    """SSE endpoint for job progress updates."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    def generate_events():
        while True:
            try:
                event_type, data = job.queue.get(timeout=30)
                if event_type == "done":
                    break
                yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            except queue.Empty:
                # Send keepalive
                yield ": keepalive\n\n"

    return Response(
        generate_events(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/image/<job_id>")
def get_image(job_id):
    """Serve a generated image."""
    # Check active jobs first
    with jobs_lock:
        job = jobs.get(job_id)

    # Fall back to history
    if not job:
        with history_lock:
            for h in history:
                if h.id == job_id:
                    job = h
                    break

    # Fall back to file on disk
    if not job:
        path = OUTPUT_DIR / f"{job_id}.png"
        if path.exists():
            return send_file(path, mimetype="image/png")

    if not job or not job.output_path or not job.output_path.exists():
        return jsonify({"error": "Image not found"}), 404

    return send_file(job.output_path, mimetype="image/png")


@app.route("/thumb/<job_id>")
def get_thumb(job_id):
    """Serve a thumbnail for a generated image."""
    # Ensure thumb directory exists
    THUMB_DIR.mkdir(exist_ok=True)

    thumb_path = THUMB_DIR / f"{job_id}.jpg"

    # If thumbnail exists, serve it
    if thumb_path.exists():
        return send_file(thumb_path, mimetype="image/jpeg")

    # Otherwise, generate it from the original image
    original_path = OUTPUT_DIR / f"{job_id}.png"
    if not original_path.exists():
        return jsonify({"error": "Image not found"}), 404

    try:
        from PIL import Image
        img = Image.open(original_path)
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        # Convert to RGB for JPEG (handles RGBA)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (0, 0, 0))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(thumb_path, "JPEG", quality=80)
        return send_file(thumb_path, mimetype="image/jpeg")
    except ImportError:
        # PIL not available, serve original
        return send_file(original_path, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"Failed to generate thumbnail: {e}"}), 500


@app.route("/status/<job_id>")
def get_status(job_id):
    """Get current job status."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    result = job.to_dict()
    if job.status == "complete":
        result["image_url"] = f"/image/{job_id}"
    return jsonify(result)


@app.route("/history")
def get_history():
    """Get list of recent generations."""
    with history_lock:
        return jsonify([job.to_history_dict() for job in history])


@app.route("/history/<job_id>", methods=["DELETE"])
def delete_history(job_id):
    """Delete a history item."""
    with history_lock:
        for i, job in enumerate(history):
            if job.id == job_id:
                history.pop(i)
                # Delete the image file too
                if job.output_path and job.output_path.exists():
                    try:
                        os.unlink(job.output_path)
                    except OSError:
                        pass
                # Delete thumbnail too
                thumb_path = THUMB_DIR / f"{job_id}.jpg"
                if thumb_path.exists():
                    try:
                        os.unlink(thumb_path)
                    except OSError:
                        pass
                save_history()
                return jsonify({"status": "deleted"})
    return jsonify({"error": "Not found"}), 404


@app.route("/save-crop", methods=["POST"])
def save_crop():
    """Save a cropped image to history."""
    data = request.json or {}

    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "Image data required"}), 400

    # Get metadata from original image
    original_id = data.get("original_id")
    prompt = data.get("prompt", "Cropped image")
    seed = data.get("seed")
    width = data.get("width")
    height = data.get("height")

    # Create new job entry for the cropped image
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, prompt=prompt, width=width or 512, height=height or 512, steps=0)
    job.seed = seed
    job.status = "complete"
    job.output_path = OUTPUT_DIR / f"{job_id}.png"

    # Decode and save the cropped image
    try:
        image_data = base64.b64decode(image_b64.split(",")[-1])
        with open(job.output_path, "wb") as f:
            f.write(image_data)
    except Exception as e:
        return jsonify({"error": f"Failed to save image: {e}"}), 500

    # Add to history
    with history_lock:
        history.insert(0, job)
        if len(history) > MAX_HISTORY:
            history.pop()
        save_history()

    return jsonify({
        "job_id": job_id,
        "image_url": f"/image/{job_id}",
    })


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    """Cancel a running job."""
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.status == "complete":
        return jsonify({"error": "Job already complete"}), 400

    # Mark job as cancelled and send event
    job.status = "cancelled"
    job.error = "Cancelled by user"
    job.queue.put(("error", {"message": "Cancelled"}))
    job.queue.put(("done", None))

    return jsonify({"status": "cancelled"})


def main():
    global flux_server

    parser = argparse.ArgumentParser(description="FLUX.2 Web UI Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Path to model directory")
    parser.add_argument("--flux-binary", type=Path, default=DEFAULT_FLUX_BINARY,
                        help="Path to flux binary")
    args = parser.parse_args()

    # Validate paths
    if not args.flux_binary.exists():
        print(f"Error: flux binary not found at {args.flux_binary}")
        print("Build it first with: make mps (or make blas)")
        return 1

    if not args.model_dir.exists():
        print(f"Error: Model directory not found at {args.model_dir}")
        print("Download it first with: ./download_model.sh")
        return 1

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load history from disk
    load_history_from_disk()

    # Start flux server (persistent model)
    print("Starting flux server with persistent model...")
    flux_server = FluxServer(args.flux_binary, args.model_dir)
    flux_server.start()

    print(f"\nFLUX.2 Web UI")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Model: {args.model_dir}")
    print(f"  Binary: {args.flux_binary}")
    print(f"  Mode: Server (persistent model)")
    print(f"  Press Ctrl+C to stop\n")

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        print("\nStopping flux server...")
        flux_server.stop()


if __name__ == "__main__":
    main()
