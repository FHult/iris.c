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
_MODEL_SEARCH_ORDER = ["flux-klein-4b", "flux-klein-model", "flux-klein-9b",
                       "flux-klein-4b-base", "flux-klein-9b-base"]
DEFAULT_MODEL_DIR = next(
    (PROJECT_DIR / d for d in _MODEL_SEARCH_ORDER if (PROJECT_DIR / d).is_dir()),
    PROJECT_DIR / "flux-klein-4b"  # fallback for error message
)
DEFAULT_FLUX_BINARY = PROJECT_DIR / "flux"
OUTPUT_DIR = SCRIPT_DIR / "output"
THUMB_DIR = SCRIPT_DIR / "output" / "thumbs"
HISTORY_FILE = OUTPUT_DIR / "history.json"
THUMB_SIZE = 200  # Max dimension for thumbnails

app = Flask(__name__, static_folder="static")


def generate_thumbnail(job_id: str, source_path: Path) -> bool:
    """Generate thumbnail for an image in background. Returns True on success."""
    THUMB_DIR.mkdir(exist_ok=True)
    thumb_path = THUMB_DIR / f"{job_id}.jpg"
    if thumb_path.exists():
        return True
    try:
        from PIL import Image
        img = Image.open(source_path)
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
        return True
    except Exception as e:
        print(f"Warning: Failed to generate thumbnail for {job_id}: {e}")
        return False


def convert_image_to_png(image_data: bytes) -> bytes:
    """
    Convert any image format to PNG for the C code.
    Supports JPEG, WebP, GIF, BMP, TIFF, and any other PIL-supported format.
    Handles animated images (uses first frame) and transparency.
    """
    # Always convert through PIL to normalize bit depth, color type, etc.
    # User-uploaded PNGs may be 16-bit, palette-indexed, interlaced, or have
    # other features the C PNG loader doesn't support.
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
history_by_id = {}  # O(1) lookup by job_id
history_lock = threading.Lock()
MAX_HISTORY = 10000  # Effectively unlimited; frontend handles pagination

# Job queue for pending generations
job_queue = []
job_queue_lock = threading.Lock()

# Batched history save
history_dirty = False
history_save_interval = 5.0  # seconds

# Job cleanup settings
JOB_CLEANUP_AGE = 3600  # Remove completed jobs after 1 hour
JOB_CLEANUP_INTERVAL = 300  # Check every 5 minutes

# Flux server process manager
flux_server = None
flux_server_lock = threading.Lock()


def save_history():
    """Save history to JSON file (must be called with history_lock held)."""
    global history_dirty
    try:
        data = [job.to_history_dict() for job in history]
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f)
        history_dirty = False
    except Exception as e:
        print(f"Warning: Failed to save history: {e}")


def mark_history_dirty():
    """Mark history as needing save (called with history_lock held)."""
    global history_dirty
    history_dirty = True


def periodic_history_saver():
    """Background thread that saves history periodically when dirty."""
    global history_dirty
    while True:
        time.sleep(history_save_interval)
        with history_lock:
            if history_dirty:
                save_history()


def periodic_job_cleanup():
    """Background thread that removes old completed jobs from memory."""
    while True:
        time.sleep(JOB_CLEANUP_INTERVAL)
        now = time.time()
        to_remove = []
        with jobs_lock:
            for job_id, job in jobs.items():
                if job.status in ("complete", "error", "cancelled"):
                    if now - job.created_at > JOB_CLEANUP_AGE:
                        to_remove.append(job_id)
            for job_id in to_remove:
                del jobs[job_id]
        if to_remove:
            print(f"Cleaned up {len(to_remove)} old jobs from memory")


def load_history_from_disk():
    """Load history from JSON file on startup."""
    global history_by_id
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
            job.generation_time = item.get("generation_time")
            job.batch_id = item.get("batch_id")
            job.style = item.get("style")
            job.status = "complete"
            job.output_path = output_path
            history.append(job)
            history_by_id[job_id] = job
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
        # Subscriber-based event broadcasting (supports multiple SSE connections)
        self._subscribers = []
        self._subscribers_lock = threading.Lock()
        # Store generation parameters for history/remix
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.created_at = time.time()
        # Temp files to clean up after job completes
        self.temp_files = []
        self.generation_time = None
        self.batch_id = None
        self.style = None

    def subscribe(self):
        """Create a new subscriber queue for an SSE connection."""
        q = queue.Queue()
        with self._subscribers_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        """Remove a subscriber queue."""
        with self._subscribers_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def put_event(self, event):
        """Broadcast an event to all subscribers. Event is a (type, data) tuple."""
        with self._subscribers_lock:
            for q in self._subscribers:
                q.put(event)

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
            "generation_time": self.generation_time,
            "batch_id": self.batch_id,
            "style": self.style,
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
        # Clean up step images
        step_images = getattr(self, 'step_images', {})
        for path in step_images.values():
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass
        self.step_images = {}


class FluxServer:
    """Manages a persistent flux process in server mode."""

    def __init__(self, flux_binary, model_dir):
        self.flux_binary = flux_binary
        self.model_dir = model_dir
        self.process = None
        self.ready = False
        self.lock = threading.Lock()
        self.current_job = None
        self.model_info = None      # e.g. "FLUX.2-klein-4B v1.0 (distilled, 4 steps, guidance 1.0)"
        self.is_distilled = True
        self._intentional_stop = False
        self.generation_count = 0

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
                    self.model_info = event.get("model")
                    self.is_distilled = event.get("is_distilled", True)
                    print(f"Flux server ready: {self.model_info}")
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
                    job.put_event(("status", job.to_dict()))

            elif event_type == "phase":
                phase = event.get("phase", "")
                elapsed = event.get("elapsed", 0)
                # Capitalize first letter
                if phase:
                    phase = phase[0].upper() + phase[1:]
                job.phase = phase
                data = job.to_dict()
                data["elapsed"] = elapsed
                job.put_event(("status", data))

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
                job.put_event(("status", data))

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
                job.put_event(("progress", data))

            elif event_type == "step_image":
                step = event.get("step", 0)
                total = event.get("total", job.total_steps)
                path = event.get("path", "")
                elapsed = event.get("elapsed", 0)
                # Store step image path for serving
                if not hasattr(job, 'step_images'):
                    job.step_images = {}
                job.step_images[step] = path
                job.put_event(("step_image", {
                    "step": step,
                    "total": total,
                    "image_url": f"/step-image/{job.id}/{step}",
                    "elapsed": elapsed
                }))

            elif event_type == "complete":
                job.status = "complete"
                job.output_path = OUTPUT_DIR / f"{job.id}.png"
                job.progress = job.total_steps
                total_time = event.get("total_time", 0)
                job.generation_time = total_time
                # Clean up temp files now that generation is complete
                job.cleanup_temp_files()
                # Generate thumbnail in background thread
                thumb_thread = threading.Thread(
                    target=generate_thumbnail,
                    args=(job.id, job.output_path),
                    daemon=True
                )
                thumb_thread.start()
                # Add to history BEFORE sending complete event to avoid race condition
                # (frontend calls loadHistory immediately on receiving complete)
                with history_lock:
                    history.insert(0, job)
                    history_by_id[job.id] = job
                    if len(history) > MAX_HISTORY:
                        old_job = history.pop()
                        history_by_id.pop(old_job.id, None)
                    mark_history_dirty()  # Batched save instead of immediate
                # Now send the complete event
                job.put_event(("complete", {
                    "image_url": f"/image/{job.id}",
                    "total_time": total_time
                }))
                job.put_event(("done", None))
                self.current_job = None
                # Process next queued job if any
                process_job_queue()

            elif event_type == "error":
                job.status = "error"
                job.error = event.get("message", "Unknown error")
                job.put_event(("error", {"message": job.error}))
                job.put_event(("done", None))
                # Clean up temp files on error
                job.cleanup_temp_files()
                self.current_job = None
                # Process next queued job if any
                process_job_queue()

        # Process ended — check if unexpected crash
        if not self._intentional_stop:
            exit_code = self.process.returncode if self.process else None
            print(f"Flux server process died unexpectedly (exit code: {exit_code})")
            job = self.current_job
            if job:
                job.status = "error"
                job.error = "Server process crashed"
                job.put_event(("error", {"message": "Server process crashed, restarting..."}))
                job.put_event(("done", None))
                job.cleanup_temp_files()
                self.current_job = None
            # Auto-restart
            print("Auto-restarting flux server...")
            try:
                self._intentional_stop = False
                self.ready = False
                self.generation_count = 0
                self.start()
                print("Flux server restarted successfully")
                process_job_queue()
            except Exception as e:
                print(f"Failed to restart flux server: {e}")

    def generate(self, job, prompt, width, height, steps, seed, input_image_path, reference_image_paths=None, show_steps=True, guidance=None, schedule=None):
        """Send a generation request to the flux server."""
        with self.lock:
            if not self.ready or self.process.poll() is not None:
                raise RuntimeError("Flux server not running")

            self.current_job = job
            job.status = "running"
            self.generation_count += 1
            is_cold = self.generation_count == 1
            if is_cold:
                job.phase = "Loading models (first run, may take a minute)"
            job.put_event(("status", job.to_dict()))

            # Build request
            output_path = OUTPUT_DIR / f"{job.id}.png"
            request_data = {
                "prompt": prompt,
                "output": str(output_path),
                "width": width,
                "height": height,
                "steps": steps,
                "show_steps": show_steps,
            }

            if seed is not None and seed != "":
                request_data["seed"] = int(seed)

            if guidance is not None and guidance > 0:
                request_data["guidance"] = float(guidance)

            if schedule and schedule != "sigmoid":
                request_data["schedule"] = schedule

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
        self._intentional_stop = True
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            self.ready = False

    def restart(self):
        """Restart the flux server (used for cancellation)."""
        print("Restarting flux server...")
        self.stop()
        self._intentional_stop = False
        self.generation_count = 0
        self.start()
        print("Flux server restarted")


def process_job_queue():
    """Process the next job in the queue if the server is free."""
    global flux_server
    with job_queue_lock:
        if not job_queue:
            return
        # Check if server is busy
        if flux_server and flux_server.current_job:
            return
        # Get next job
        queued = job_queue.pop(0)
        # Update queue positions for remaining jobs
        for i, q in enumerate(job_queue):
            q['job'].put_event(("queue_position", {"position": i + 1, "total": len(job_queue) + 1}))

    # Start the job
    job = queued['job']
    job.status = "running"
    job.put_event(("status", job.to_dict()))

    try:
        flux_server.generate(
            job,
            queued['prompt'],
            queued['width'],
            queued['height'],
            queued['steps'],
            queued['seed'],
            queued['input_image_path'],
            queued['reference_image_paths'],
            queued['show_steps'],
            queued.get('guidance'),
            queued.get('schedule'),
        )
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.put_event(("error", {"message": str(e)}))
        job.put_event(("done", None))
        job.cleanup_temp_files()
        # Try next job
        process_job_queue()


def queue_generation(job, prompt, width, height, steps, seed, input_image_path, reference_image_paths=None, show_steps=True, guidance=None, schedule=None):
    """Queue a generation job. Starts immediately if server is free, otherwise queues."""
    global flux_server

    queued_job = {
        'job': job,
        'prompt': prompt,
        'width': width,
        'height': height,
        'steps': steps,
        'seed': seed,
        'input_image_path': input_image_path,
        'reference_image_paths': reference_image_paths,
        'show_steps': show_steps,
        'guidance': guidance,
        'schedule': schedule,
    }

    with job_queue_lock:
        # Check if server is busy
        if flux_server and flux_server.current_job:
            # Queue the job
            job_queue.append(queued_job)
            position = len(job_queue)
            job.status = "queued"
            job.put_event(("queued", {"position": position, "total": position}))
            return

    # Server is free, start immediately
    job.status = "running"
    job.put_event(("status", job.to_dict()))

    try:
        flux_server.generate(job, prompt, width, height, steps, seed, input_image_path, reference_image_paths, show_steps, guidance, schedule)
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.put_event(("error", {"message": str(e)}))
        job.put_event(("done", None))
        job.cleanup_temp_files()


def run_generation_server_mode(job, prompt, width, height, steps, seed, input_image_path, reference_image_paths=None, show_steps=True, guidance=None, schedule=None):
    """Run generation using the persistent flux server (legacy, now uses queue)."""
    queue_generation(job, prompt, width, height, steps, seed, input_image_path, reference_image_paths, show_steps, guidance, schedule)


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

    # Apply style preset suffix for generation (keep base prompt for history)
    style = data.get("style")
    generation_prompt = prompt
    if style and style in STYLE_PRESETS:
        generation_prompt = prompt + STYLE_PRESETS[style]["suffix"]

    width = int(data.get("width", 512))
    height = int(data.get("height", 512))
    steps = int(data.get("steps", 4))
    seed = data.get("seed")
    show_steps = bool(data.get("show_steps", True))
    guidance = data.get("guidance")  # None = auto (1.0 distilled, 4.0 base)
    schedule = data.get("schedule")  # "sigmoid" (default), "linear", "power"
    batch_id = data.get("batch_id")  # Groups variation batches

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
        for i, img_b64 in enumerate(reference_images_b64[:4]):
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

    # Create job (store base prompt; style applied separately for generation)
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, prompt=prompt, width=width, height=height, steps=steps)
    job.batch_id = batch_id
    job.style = style if (style and style in STYLE_PRESETS) else None

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
        args=(job, generation_prompt, width, height, steps, seed, input_image_path, reference_image_paths, show_steps, guidance, schedule),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id, "status": "started"})


@app.route("/progress/<job_id>")
def progress(job_id):
    """SSE endpoint for job progress updates.

    Uses per-connection subscriber queues so multiple SSE connections
    (e.g. after page reload) each get their own independent event stream.
    """
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    subscriber_queue = job.subscribe()

    def generate_events():
        try:
            while True:
                try:
                    event_type, data = subscriber_queue.get(timeout=30)
                    if event_type == "done":
                        break
                    yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"
        finally:
            job.unsubscribe(subscriber_queue)

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

    # Fall back to history (O(1) lookup)
    if not job:
        with history_lock:
            job = history_by_id.get(job_id)

    # Fall back to file on disk
    if not job:
        path = OUTPUT_DIR / f"{job_id}.png"
        if path.exists():
            return send_file(path, mimetype="image/png")

    if not job or not job.output_path or not job.output_path.exists():
        return jsonify({"error": "Image not found"}), 404

    response = send_file(job.output_path, mimetype="image/png")
    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response


@app.route("/step-image/<job_id>/<int:step>")
def get_step_image(job_id, step):
    """Serve a step image (intermediate denoising result)."""
    # Check active jobs
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    step_images = getattr(job, 'step_images', {})
    path = step_images.get(step)

    if not path or not Path(path).exists():
        return jsonify({"error": "Step image not found"}), 404

    return send_file(path, mimetype="image/png")


@app.route("/thumb/<job_id>")
def get_thumb(job_id):
    """Serve a thumbnail for a generated image."""
    # Ensure thumb directory exists
    THUMB_DIR.mkdir(exist_ok=True)

    thumb_path = THUMB_DIR / f"{job_id}.jpg"

    # If thumbnail exists, serve it
    if thumb_path.exists():
        response = send_file(thumb_path, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

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
        response = send_file(thumb_path, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response
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
        job = history_by_id.get(job_id)
        if job:
            history.remove(job)
            del history_by_id[job_id]
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
            mark_history_dirty()
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
        history_by_id[job_id] = job
        if len(history) > MAX_HISTORY:
            old_job = history.pop()
            history_by_id.pop(old_job.id, None)
        mark_history_dirty()

    # Generate thumbnail in background
    thumb_thread = threading.Thread(
        target=generate_thumbnail,
        args=(job_id, job.output_path),
        daemon=True
    )
    thumb_thread.start()

    return jsonify({
        "job_id": job_id,
        "image_url": f"/image/{job_id}",
    })


@app.route("/server-status")
def server_status():
    """Get server status including ready state and current job."""
    global flux_server
    status = {
        "ready": False,
        "busy": False,
        "current_job": None,
        "queue_length": 0,
    }

    if flux_server:
        status["ready"] = flux_server.ready
        if flux_server.current_job:
            status["busy"] = True
            status["current_job"] = flux_server.current_job.id

    with job_queue_lock:
        status["queue_length"] = len(job_queue)

    return jsonify(status)


@app.route("/model-info")
def model_info():
    """Get loaded model information."""
    global flux_server
    if not flux_server or not flux_server.ready:
        return jsonify({"error": "Server not ready"}), 503
    return jsonify({
        "model": flux_server.model_info,
        "is_distilled": flux_server.is_distilled,
    })


@app.route("/active-jobs")
def active_jobs():
    """Get all active (running + queued) jobs for recovery after page reload."""
    result = {
        "running": None,
        "queued": [],
    }

    # Get currently running job
    if flux_server and flux_server.current_job:
        job = flux_server.current_job
        d = job.to_dict()
        d["prompt"] = job.prompt
        d["width"] = job.width
        d["height"] = job.height
        d["steps"] = job.steps
        result["running"] = d

    # Get queued jobs
    with job_queue_lock:
        for i, queued in enumerate(job_queue):
            job = queued["job"]
            d = job.to_dict()
            d["prompt"] = job.prompt
            d["width"] = job.width
            d["height"] = job.height
            d["steps"] = job.steps
            d["queue_position"] = i + 1
            result["queued"].append(d)

    return jsonify(result)


# Style presets for prompt enhancement
STYLE_PRESETS = {
    # Photography
    "photo_realistic": {
        "name": "Photo Realistic",
        "category": "Photography",
        "description": "Photorealistic style with natural lighting",
        "suffix": ", photorealistic, high resolution, natural lighting, sharp focus, detailed textures, professional photography",
        "recommended_steps": 4,
    },
    "cinematic": {
        "name": "Cinematic",
        "category": "Photography",
        "description": "Movie-like dramatic lighting and composition",
        "suffix": ", cinematic lighting, dramatic composition, film grain, anamorphic lens, movie still, atmospheric",
        "recommended_steps": 4,
    },
    "portrait": {
        "name": "Portrait",
        "category": "Photography",
        "description": "Professional portrait photography",
        "suffix": ", portrait photography, studio lighting, shallow depth of field, professional headshot, detailed face, sharp focus",
        "recommended_steps": 4,
    },
    "landscape": {
        "name": "Landscape",
        "category": "Photography",
        "description": "Epic landscape photography",
        "suffix": ", landscape photography, golden hour, dramatic sky, wide angle, national geographic style, breathtaking scenery",
        "recommended_steps": 4,
    },
    "street_photo": {
        "name": "Street Photography",
        "category": "Photography",
        "description": "Candid urban documentary feel",
        "suffix": ", street photography, candid, urban, documentary style, natural light, film grain, authentic moment",
        "recommended_steps": 4,
    },
    "macro": {
        "name": "Macro",
        "category": "Photography",
        "description": "Extreme close-up with crisp detail",
        "suffix": ", macro photography, extreme close-up, shallow depth of field, crisp detail, studio lighting, fine textures",
        "recommended_steps": 4,
    },
    "long_exposure": {
        "name": "Long Exposure",
        "category": "Photography",
        "description": "Motion blur, light trails, dreamy",
        "suffix": ", long exposure photography, motion blur, light trails, smooth water, dreamy atmosphere, silky textures",
        "recommended_steps": 4,
    },
    "film_noir": {
        "name": "Film Noir",
        "category": "Photography",
        "description": "High contrast B&W, dramatic shadows",
        "suffix": ", film noir, black and white, high contrast, dramatic shadows, moody lighting, 1940s aesthetic, venetian blinds",
        "recommended_steps": 4,
    },
    "vintage": {
        "name": "Vintage",
        "category": "Photography",
        "description": "Faded colors, 70s film stock feel",
        "suffix": ", vintage photography, faded colors, light leaks, 70s film stock, warm tones, vignette, retro",
        "recommended_steps": 4,
    },
    # Art Styles
    "digital_art": {
        "name": "Digital Art",
        "category": "Art",
        "description": "Polished digital illustration style",
        "suffix": ", digital art, highly detailed, vibrant colors, sharp lines, trending on artstation, professional illustration",
        "recommended_steps": 4,
    },
    "concept_art": {
        "name": "Concept Art",
        "category": "Art",
        "description": "Professional concept art for games/films",
        "suffix": ", concept art, detailed environment, atmospheric perspective, professional, matte painting, epic scale",
        "recommended_steps": 6,
    },
    "anime": {
        "name": "Anime",
        "category": "Art",
        "description": "Japanese anime/manga style",
        "suffix": ", anime style, cel shaded, vibrant colors, clean lines, studio ghibli inspired, detailed",
        "recommended_steps": 4,
    },
    "oil_painting": {
        "name": "Oil Painting",
        "category": "Art",
        "description": "Classical oil painting aesthetic",
        "suffix": ", oil painting, textured brush strokes, classical art, rich colors, museum quality, masterpiece",
        "recommended_steps": 6,
    },
    "watercolor": {
        "name": "Watercolor",
        "category": "Art",
        "description": "Soft watercolor painting style",
        "suffix": ", watercolor painting, soft edges, flowing colors, paper texture, artistic, delicate details",
        "recommended_steps": 4,
    },
    "impressionist": {
        "name": "Impressionist",
        "category": "Art",
        "description": "Loose brushwork, light and color focus",
        "suffix": ", impressionist painting, loose brushwork, visible brush strokes, light and color, Monet inspired, plein air",
        "recommended_steps": 6,
    },
    "art_nouveau": {
        "name": "Art Nouveau",
        "category": "Art",
        "description": "Ornate organic curves, Mucha-inspired",
        "suffix": ", art nouveau, ornate, organic curves, Alphonse Mucha inspired, decorative borders, flowing lines, elegant",
        "recommended_steps": 6,
    },
    "ukiyo_e": {
        "name": "Ukiyo-e",
        "category": "Art",
        "description": "Japanese woodblock print style",
        "suffix": ", ukiyo-e, Japanese woodblock print, flat colors, bold outlines, traditional Japanese art, Hokusai inspired",
        "recommended_steps": 4,
    },
    "surrealist": {
        "name": "Surrealist",
        "category": "Art",
        "description": "Dreamlike, impossible geometry",
        "suffix": ", surrealist art, dreamlike, impossible geometry, Dali inspired, melting forms, subconscious imagery, bizarre",
        "recommended_steps": 6,
    },
    "pop_art": {
        "name": "Pop Art",
        "category": "Art",
        "description": "Bold colors, halftone dots, graphic",
        "suffix": ", pop art, bold colors, halftone dots, Warhol inspired, graphic design, screen print, high contrast",
        "recommended_steps": 4,
    },
    "pixel_art": {
        "name": "Pixel Art",
        "category": "Art",
        "description": "Retro 8/16-bit game aesthetic",
        "suffix": ", pixel art, retro 8-bit, 16-bit, clean pixels, limited color palette, game aesthetic, nostalgic",
        "recommended_steps": 4,
    },
    # Design & Technical
    "minimalist": {
        "name": "Minimalist",
        "category": "Design",
        "description": "Clean, simple, minimal design",
        "suffix": ", minimalist, clean design, simple composition, negative space, modern aesthetic, elegant",
        "recommended_steps": 2,
    },
    "isometric": {
        "name": "Isometric",
        "category": "Design",
        "description": "3D isometric view, clean geometry",
        "suffix": ", isometric view, 3D isometric, clean geometry, technical illustration, precise angles, detailed miniature",
        "recommended_steps": 4,
    },
    "flat_design": {
        "name": "Flat Design",
        "category": "Design",
        "description": "Clean vectors, modern UI illustration",
        "suffix": ", flat design, clean vector art, solid colors, modern UI illustration, geometric shapes, simple",
        "recommended_steps": 2,
    },
    "blueprint": {
        "name": "Blueprint",
        "category": "Design",
        "description": "Technical drawing, schematic style",
        "suffix": ", blueprint, technical drawing, white lines on blue background, schematic, engineering diagram, precise",
        "recommended_steps": 2,
    },
    # Mood & Genre
    "fantasy": {
        "name": "Fantasy",
        "category": "Mood",
        "description": "Epic fantasy art style",
        "suffix": ", fantasy art, magical atmosphere, ethereal lighting, detailed, epic composition, mythical, enchanted",
        "recommended_steps": 6,
    },
    "scifi": {
        "name": "Sci-Fi",
        "category": "Mood",
        "description": "Futuristic science fiction style",
        "suffix": ", science fiction, futuristic, cyberpunk, neon lights, high tech, detailed machinery, atmospheric",
        "recommended_steps": 6,
    },
    "neon_glow": {
        "name": "Neon Glow",
        "category": "Mood",
        "description": "Dark background with vivid neon colors",
        "suffix": ", neon glow, dark background, vivid neon colors, cyberpunk lighting, glow effects, luminous, electric",
        "recommended_steps": 4,
    },
    "dark_gothic": {
        "name": "Dark Gothic",
        "category": "Mood",
        "description": "Moody, dark palette, fog and shadows",
        "suffix": ", dark gothic, moody atmosphere, dark palette, cathedral architecture, fog, dramatic shadows, ominous",
        "recommended_steps": 6,
    },
    "cozy": {
        "name": "Cozy",
        "category": "Mood",
        "description": "Warm lighting, soft textures, inviting",
        "suffix": ", cozy atmosphere, warm lighting, soft textures, hygge, inviting, intimate setting, comfortable",
        "recommended_steps": 4,
    },
    "ethereal": {
        "name": "Ethereal",
        "category": "Mood",
        "description": "Soft glow, pastel colors, dreamy",
        "suffix": ", ethereal, soft glow, pastel colors, dreamy atmosphere, angelic, luminous, otherworldly beauty",
        "recommended_steps": 4,
    },
}

# Step count guidance
STEP_GUIDANCE = {
    1: {"label": "Fastest", "description": "Very quick, loose interpretation", "quality": "draft"},
    2: {"label": "Fast", "description": "Quick with decent coherence", "quality": "good"},
    3: {"label": "Balanced", "description": "Good balance of speed and quality", "quality": "good"},
    4: {"label": "Standard", "description": "Recommended default, good prompt adherence", "quality": "high"},
    5: {"label": "Quality", "description": "Higher fidelity, slower", "quality": "high"},
    6: {"label": "High Quality", "description": "Detailed output, recommended for complex prompts", "quality": "very_high"},
    7: {"label": "Maximum", "description": "Best quality, slowest", "quality": "very_high"},
    8: {"label": "Ultra", "description": "Maximum detail and coherence", "quality": "maximum"},
}


@app.route("/style-presets")
def get_style_presets():
    """Get available style presets for prompt enhancement."""
    return jsonify({
        "presets": {k: {
            "name": v["name"],
            "category": v.get("category", ""),
            "description": v["description"],
            "recommended_steps": v["recommended_steps"],
        } for k, v in STYLE_PRESETS.items()},
        "step_guidance": STEP_GUIDANCE,
    })


@app.route("/enhance-prompt", methods=["POST"])
def enhance_prompt():
    """Enhance a prompt with style modifiers and detail expansion."""
    data = request.json or {}
    prompt = data.get("prompt", "").strip()
    style = data.get("style")  # Optional style preset key
    auto_enhance = data.get("auto_enhance", True)  # Add quality modifiers

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    enhanced = prompt
    recommended_steps = 4

    # Apply style preset if specified
    if style and style in STYLE_PRESETS:
        preset = STYLE_PRESETS[style]
        enhanced = prompt + preset["suffix"]
        recommended_steps = preset["recommended_steps"]
    elif auto_enhance:
        # Auto-enhance: add generic quality modifiers if prompt is short/simple
        word_count = len(prompt.split())
        if word_count < 10:
            # Short prompt - add quality modifiers
            quality_suffixes = [
                ", high quality",
                ", detailed",
                ", well-composed",
            ]
            # Only add if not already present
            for suffix in quality_suffixes:
                keyword = suffix.replace(", ", "").strip()
                if keyword.lower() not in prompt.lower():
                    enhanced += suffix
                    break

    return jsonify({
        "original": prompt,
        "enhanced": enhanced,
        "style": style,
        "recommended_steps": recommended_steps,
    })


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    """Cancel a running or queued job."""
    global flux_server

    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.status == "complete":
        return jsonify({"error": "Job already complete"}), 400

    # Check if job is queued (not yet running)
    with job_queue_lock:
        for i, queued in enumerate(job_queue):
            if queued['job'].id == job_id:
                job_queue.pop(i)
                job.status = "cancelled"
                job.error = "Cancelled by user"
                job.put_event(("error", {"message": "Cancelled"}))
                job.put_event(("done", None))
                job.cleanup_temp_files()
                # Update queue positions for remaining jobs
                for j, q in enumerate(job_queue):
                    q['job'].put_event(("queue_position", {"position": j + 1, "total": len(job_queue)}))
                return jsonify({"status": "cancelled"})

    # Job is currently running - need to restart flux server
    if flux_server and flux_server.current_job and flux_server.current_job.id == job_id:
        job.status = "cancelled"
        job.error = "Cancelled by user"
        job.put_event(("error", {"message": "Cancelled"}))
        job.put_event(("done", None))
        job.cleanup_temp_files()

        # Restart the flux server to stop the current generation
        # Do this in a background thread to not block the response
        def restart_and_process_queue():
            flux_server.restart()
            process_job_queue()

        restart_thread = threading.Thread(target=restart_and_process_queue, daemon=True)
        restart_thread.start()

        return jsonify({"status": "cancelled", "server_restarting": True})

    # Job already finished or errored
    job.status = "cancelled"
    job.error = "Cancelled by user"
    job.put_event(("error", {"message": "Cancelled"}))
    job.put_event(("done", None))

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

    # Start background threads
    history_saver_thread = threading.Thread(target=periodic_history_saver, daemon=True)
    history_saver_thread.start()

    job_cleanup_thread = threading.Thread(target=periodic_job_cleanup, daemon=True)
    job_cleanup_thread.start()

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
