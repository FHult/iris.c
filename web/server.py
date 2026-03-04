#!/usr/bin/env python3
"""
FLUX.2 Web UI - Simple Flask server for image generation.

Uses flux binary in server mode for persistent model (faster subsequent generations).

Usage:
    python web/server.py [--port PORT] [--model-dir PATH] [--iris-binary PATH]

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
import urllib.request
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
DEFAULT_IRIS_BINARY = PROJECT_DIR / "iris"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Known model slots. The startup model dir is prepended dynamically in main().
# sh_arg: argument to pass to download_model.sh; None means not downloadable via UI.
MODEL_SLOTS = [
    {
        "key": "flux-klein-model",
        "label": "4B Distilled",
        "description": "4 steps, fast generation",
        "sh_arg": None,
        "expected_files": [
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "tokenizer/tokenizer.json",
        ],
    },
    {
        "key": "flux-klein-4b-base",
        "label": "4B Base",
        "description": "50 steps, CFG, higher quality",
        "sh_arg": "4b-base",
        "expected_files": [
            "model_index.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/tokenizer.json",
        ],
    },
    {
        "key": "zimage-turbo",
        "label": "Z-Image Turbo",
        "description": "8 steps, 6B, Apache 2.0",
        "sh_arg": "zimage-turbo",
        "expected_files": [
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "tokenizer/tokenizer.json",
        ],
    },
]
_model_download_progress = {}  # key -> {done, error}
THUMB_DIR = SCRIPT_DIR / "output" / "thumbs"
HISTORY_FILE = OUTPUT_DIR / "history.json"
CONFIG_FILE = None          # set in main() after OUTPUT_DIR is known
_hf_token = os.environ.get("HF_TOKEN", "")
THUMB_SIZE = 200  # Max dimension for thumbnails

app = Flask(__name__, static_folder="static")


def _is_valid_thumb(path: Path) -> bool:
    """Return True if path exists and is a non-empty file."""
    return path.exists() and path.stat().st_size > 0


def generate_thumbnail(job_id: str, source_path: Path) -> bool:
    """Generate thumbnail for an image. Returns True on success.

    Writes atomically (temp file + rename) to avoid corrupt thumbnails from
    concurrent calls or mid-write interruptions.
    """
    THUMB_DIR.mkdir(exist_ok=True)
    thumb_path = THUMB_DIR / f"{job_id}.jpg"
    if _is_valid_thumb(thumb_path):
        return True
    try:
        from PIL import Image
        img = Image.open(source_path)
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        # Convert to RGB for JPEG (handles RGBA/palette modes)
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (0, 0, 0))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        # Write to a temp file then rename atomically — prevents corrupt
        # thumbnails if two threads race or the process is interrupted.
        tmp_path = thumb_path.with_suffix('.tmp')
        img.save(tmp_path, "JPEG", quality=80)
        tmp_path.replace(thumb_path)
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
# True while a job has been popped from the queue but generate() hasn't set
# current_job yet.  Prevents a second concurrent process_job_queue() call from
# also popping a job in that brief window.
job_queue_dispatching = False

# Batched history save
history_dirty = False
history_save_interval = 5.0  # seconds

# Job cleanup settings
JOB_CLEANUP_AGE = 3600  # Remove completed jobs after 1 hour
JOB_CLEANUP_INTERVAL = 300  # Check every 5 minutes

# Flux server process manager
iris_server = None
iris_server_lock = threading.Lock()

# Global model dir (set in main(), used by LoRA endpoints)
model_dir_path = None

# LoRA download progress tracking
_download_progress = {}  # {dl_id: {percent, done, error}}

# Curated list of Klein-compatible LoRA adapters
CURATED_LORAS = [
    # ── HuggingFace ──────────────────────────────────────────────────────────
    {
        "id": "fal-outpaint",
        "name": "Outpaint",
        "source": "huggingface",
        "description": "Extends images beyond their borders with natural scene continuation",
        "repo": "fal/flux-2-klein-4B-outpaint-lora",
        "filename": "flux-outpaint-lora.safetensors",
        "models": ["4b"],
        "trigger": None,
        "strength": "1.0",
    },
    {
        "id": "fal-zoom",
        "name": "Zoom",
        "source": "huggingface",
        "description": "Zooms into red-highlighted regions and generates an enlarged detailed view",
        "repo": "fal/flux-2-klein-4B-zoom-lora",
        "filename": "flux-red-zoom-lora.safetensors",
        "models": ["4b"],
        "trigger": None,
        "strength": "1.0",
    },
    {
        "id": "fal-sprite",
        "name": "Spritesheet",
        "source": "huggingface",
        "description": "Turns a single object into a 2×2 sprite sheet with 4 camera angles",
        "repo": "fal/flux-2-klein-4b-spritesheet-lora",
        "filename": "flux-spritesheet-lora.safetensors",
        "models": ["4b"],
        "trigger": None,
        "strength": "1.0",
    },
    {
        "id": "dever-arcane",
        "name": "Arcane Style",
        "source": "huggingface",
        "description": "Painterly Arcane animated-series aesthetic",
        "repo": "DeverStyle/Flux.2-Klein-Loras",
        "filename": "dever_arcane_flux2_klein_9b.safetensors",
        "models": ["9b"],
        "trigger": "arcane_visual_style",
        "strength": "1.0",
    },
    {
        "id": "dever-cyanide",
        "name": "Cyanide & Happiness",
        "source": "huggingface",
        "description": "Dark absurdist stick-figure comic style",
        "repo": "DeverStyle/Flux.2-Klein-Loras",
        "filename": "dever_cyanide_and_happiness_flux2_klein_9b.safetensors",
        "models": ["9b"],
        "trigger": None,
        "strength": "1.0",
    },
    {
        "id": "dever-dmc",
        "name": "Devil May Cry Style",
        "source": "huggingface",
        "description": "High-contrast action game illustration aesthetic",
        "repo": "DeverStyle/Flux.2-Klein-Loras",
        "filename": "dever_devil_may_cry_flux2_klein_9b.safetensors",
        "models": ["9b"],
        "trigger": None,
        "strength": "1.0",
    },
    {
        "id": "valiant-ac",
        "name": "AC Comic Style",
        "source": "huggingface",
        "description": "American comic + pop art + cyber neon illustration blend",
        "repo": "valiantcat/FLUX.2-klein-AC-Style-LORA",
        "filename": "flux2_klein_lowres.safetensors",
        "models": ["4b", "9b"],
        "trigger": None,
        "strength": "1.0",
    },
    # ── Civitai ───────────────────────────────────────────────────────────────
    {
        "id": "civitai-hayley",
        "name": "Hayley (Influencer)",
        "source": "civitai",
        "description": "Photoreal influencer character with excellent facial likeness, expressions and casual photography styles. Works on both distilled and base.",
        "civitai_model_id": 2399494,
        "civitai_version_filter": None,
        "filename": "hayley-flux2-klein.safetensors",
        "models": ["4b"],
        "trigger": "hayleymodel",
        "strength": "1.0",
    },
    {
        "id": "civitai-selfies",
        "name": "Ultra Real Amateur Selfies",
        "source": "civitai",
        "description": "Smartphone-style realistic selfies with natural poses and a wide variety of looks.",
        "civitai_model_id": 2233658,
        "civitai_version_filter": None,
        "filename": "ultra-real-selfies-klein4b.safetensors",
        "models": ["4b"],
        "trigger": None,
        "strength": "1.0–1.25",
    },
    {
        "id": "civitai-mecha",
        "name": "New Mecha Style",
        "source": "civitai",
        "description": "Anime-inspired semi-realistic mecha and detailed digital illustrations. Works great on both base and distilled.",
        "civitai_model_id": 2227157,
        "civitai_version_filter": None,
        "filename": "new-mecha-style-klein4b.safetensors",
        "models": ["4b"],
        "trigger": None,
        "strength": "0.8",
    },
    {
        "id": "civitai-photostyle",
        "name": "Photostyle by Sean Archer",
        "source": "civitai",
        "description": "High-end fashion and glamour photography aesthetic. Best on base model.",
        "civitai_model_id": 1632416,
        "civitai_version_filter": "klein",
        "filename": "photostyle-sean-archer-klein4b.safetensors",
        "models": ["4b"],
        "trigger": None,
        "strength": "0.7–1.0",
    },
    {
        "id": "civitai-blahaj",
        "name": "Blahaj Shark",
        "source": "civitai",
        "description": "Cute IKEA shark plush / mascot character consistency.",
        "civitai_model_id": 646253,
        "civitai_version_filter": "klein4b",
        "filename": "blahaj-v9-klein4b.safetensors",
        "models": ["4b"],
        "trigger": None,
        "strength": "0.7",
    },
    {
        "id": "civitai-quilm",
        "name": "QuilM Style",
        "source": "civitai",
        "description": "Distinctive illustrative art style. Works best on distilled.",
        "civitai_model_id": 2324191,
        "civitai_version_filter": None,
        "filename": "quilm-style-f2-klein4b.safetensors",
        "models": ["4b"],
        "trigger": "quilm style",
        "strength": "0.8–1.0",
    },
    {
        "id": "civitai-anatomy",
        "name": "Anatomy / Quality Fixer",
        "source": "civitai",
        "description": "Improves hands, anatomy and overall coherence. Use at low strength for subtle correction.",
        "civitai_model_id": 2324991,
        "civitai_version_filter": "4b",
        "filename": "klein-anatomy-fixer-4b.safetensors",
        "models": ["4b"],
        "trigger": None,
        "strength": "0.6–1.0",
    },
]


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
            job.lora = item.get("lora")
            job.lora_scale = item.get("lora_scale", 1.0)
            job.favorited = item.get("favorited", False)
            job.guidance = item.get("guidance")
            job.schedule = item.get("schedule")
            job.img2img_strength = item.get("img2img_strength", 1.0)
            job.negative_prompt = item.get("negative_prompt", '')
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
        self.lora = None        # LoRA filename (relative, inside loras/ dir)
        self.lora_scale = 1.0
        self.favorited = False
        self.guidance = None    # None = auto
        self.schedule = None    # None = default (sigmoid)
        self.img2img_strength = 1.0
        self.negative_prompt = ''  # empty = no negative conditioning

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
            "lora": self.lora,
            "lora_scale": self.lora_scale,
            "favorited": self.favorited,
            "guidance": self.guidance,
            "schedule": self.schedule,
            "img2img_strength": self.img2img_strength,
            "negative_prompt": self.negative_prompt,
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


class IrisServer:
    """Manages a persistent flux process in server mode."""

    def __init__(self, iris_binary, model_dir):
        self.iris_binary = iris_binary
        self.model_dir = model_dir
        self.process = None
        self.ready = False
        self.lock = threading.Lock()
        self.current_job = None
        self.model_info = None      # e.g. "FLUX.2-klein-4B v1.0 (distilled, 4 steps, guidance 1.0)"
        self.is_distilled = True
        self.is_zimage = False
        self._intentional_stop = False
        self.generation_count = 0

    def start(self):
        """Start the flux server process."""
        cmd = [
            str(self.iris_binary),
            "-d", str(self.model_dir),
            "--server",
        ]

        print(f"Starting iris server: {' '.join(cmd)}")

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
                    self.is_zimage = event.get("is_zimage", False)
                    print(f"Iris server ready: {self.model_info}")
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

    def generate(self, job, prompt, width, height, steps, seed, input_image_path, reference_image_paths=None, show_steps=True, guidance=None, schedule=None, lora_name=None, lora_scale=1.0, img2img_strength=1.0, negative_prompt=''):
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

            if lora_name and model_dir_path:
                lora_full_path = model_dir_path / "loras" / lora_name
                request_data["lora"] = str(lora_full_path)
                request_data["lora_scale"] = float(lora_scale)

            if img2img_strength and float(img2img_strength) < 1.0:
                request_data["img2img_strength"] = float(img2img_strength)

            if negative_prompt:
                request_data["negative_prompt"] = negative_prompt

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
        self.current_job = None  # intentional stop doesn't clear this in the event loop
        self._intentional_stop = False
        self.generation_count = 0
        self.start()
        print("Flux server restarted")

    def switch(self, new_model_dir):
        """Switch to a different model directory.
        stop() terminates the flux process and waits for it to exit, which causes the OS
        to reclaim all Metal buffers and mmap regions — the old model is fully unloaded
        before start() loads the new one. The two models are never in memory simultaneously.
        """
        print(f"Switching model to {new_model_dir}...")
        self.stop()  # old process exits → OS reclaims all GPU memory / mmap regions
        self.model_dir = new_model_dir
        self._intentional_stop = False
        self.generation_count = 0
        self.start()
        print("Model switch complete")


def process_job_queue():
    """Process the next job in the queue if the server is free."""
    global iris_server, job_queue_dispatching
    with job_queue_lock:
        if not job_queue:
            return
        # Check if server is busy or a dispatch is already in progress.
        # job_queue_dispatching covers the window between popping the job and
        # generate() setting current_job, preventing a second concurrent caller
        # from also popping a job in that gap.
        if job_queue_dispatching:
            return
        if iris_server and iris_server.current_job:
            return
        # Claim the next job atomically while still holding the lock
        queued = job_queue.pop(0)
        job_queue_dispatching = True
        # Update queue positions for remaining jobs
        for i, q in enumerate(job_queue):
            q['job'].put_event(("queue_position", {"position": i + 1, "total": len(job_queue) + 1}))

    # Start the job
    job = queued['job']
    job.status = "running"
    job.put_event(("status", job.to_dict()))

    try:
        iris_server.generate(
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
            queued.get('lora_name'),
            queued.get('lora_scale', 1.0),
            queued.get('img2img_strength', 1.0),
            queued.get('negative_prompt', ''),
        )
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.put_event(("error", {"message": str(e)}))
        job.put_event(("done", None))
        job.cleanup_temp_files()
        # Try next job
        process_job_queue()
    finally:
        with job_queue_lock:
            job_queue_dispatching = False


def queue_generation(job, prompt, width, height, steps, seed, input_image_path, reference_image_paths=None, show_steps=True, guidance=None, schedule=None, lora_name=None, lora_scale=1.0, img2img_strength=1.0, negative_prompt=''):
    """Queue a generation job. Starts immediately if server is free, otherwise queues."""
    global iris_server

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
        'lora_name': lora_name,
        'lora_scale': lora_scale,
        'img2img_strength': img2img_strength,
        'negative_prompt': negative_prompt,
    }

    with job_queue_lock:
        # Check if server is busy
        if iris_server and iris_server.current_job:
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
        iris_server.generate(job, prompt, width, height, steps, seed, input_image_path, reference_image_paths, show_steps, guidance, schedule, lora_name, lora_scale, img2img_strength, negative_prompt)
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.put_event(("error", {"message": str(e)}))
        job.put_event(("done", None))
        job.cleanup_temp_files()


def run_generation_server_mode(job, prompt, width, height, steps, seed, input_image_path, reference_image_paths=None, show_steps=True, guidance=None, schedule=None, lora_name=None, lora_scale=1.0, img2img_strength=1.0, negative_prompt=''):
    """Run generation using the persistent flux server (legacy, now uses queue)."""
    queue_generation(job, prompt, width, height, steps, seed, input_image_path, reference_image_paths, show_steps, guidance, schedule, lora_name, lora_scale, img2img_strength, negative_prompt)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Start a new image generation job."""
    global iris_server

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
    lora_name = data.get("lora") or None       # LoRA filename (relative, inside loras/ dir)
    lora_scale = float(data.get("lora_scale", 1.0))
    img2img_strength = float(data.get("img2img_strength", 1.0))
    negative_prompt = (data.get("negative_prompt") or "").strip()

    # Validate dimensions
    if width < 64 or width > 1792 or width % 16 != 0:
        return jsonify({"error": "Width must be 64-1792 and divisible by 16"}), 400
    if height < 64 or height > 1792 or height % 16 != 0:
        return jsonify({"error": "Height must be 64-1792 and divisible by 16"}), 400
    if steps < 1 or steps > 256:
        return jsonify({"error": "Steps must be 1-256"}), 400

    # Handle reference images — accept both bare base64 strings and
    # per-slot objects: {"data": "<base64>", "strength": 0.85, "mode": "composition"|"style"}
    reference_image_paths = []
    input_image_path = None
    raw_refs = data.get("reference_images", []) or []

    # Normalise to list of dicts; bare strings are backward-compat
    def _normalise_ref(r):
        if isinstance(r, dict):
            return r
        return {"data": r, "strength": 1.0, "mode": "composition"}

    reference_slots = [_normalise_ref(r) for r in raw_refs if r]

    # Derive effective img2img_strength: take the minimum slot strength so the
    # most-constrained slot drives the global noise level (C backend uses one value).
    # Style-mode slots get half-weight to reduce composition influence.
    if reference_slots:
        effective_strengths = []
        for s in reference_slots:
            w = float(s.get("strength", 1.0))
            if s.get("mode") == "style":
                w *= 0.5
            effective_strengths.append(w)
        # Use minimum — preserves the strongest constraint; override only when
        # caller explicitly set img2img_strength at the request level.
        if "img2img_strength" not in data:
            img2img_strength = min(effective_strengths)

    if len(reference_slots) == 1:
        # Single image: use img2img mode for better results
        try:
            image_data = base64.b64decode(reference_slots[0]["data"].split(",")[-1])
            image_data = convert_image_to_png(image_data)
            input_image_path = OUTPUT_DIR / f"temp_{uuid.uuid4().hex}.png"
            with open(input_image_path, "wb") as f:
                f.write(image_data)
        except Exception as e:
            return jsonify({"error": f"Invalid input image: {e}"}), 400
    elif len(reference_slots) > 1:
        # Multiple images: use multiref mode
        for i, slot in enumerate(reference_slots[:4]):
            try:
                image_data = base64.b64decode(slot["data"].split(",")[-1])
                image_data = convert_image_to_png(image_data)
                ref_path = OUTPUT_DIR / f"temp_{uuid.uuid4().hex}_ref{i}.png"
                with open(ref_path, "wb") as f:
                    f.write(image_data)
                reference_image_paths.append(ref_path)
            except Exception as e:
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
    job.lora = lora_name
    job.lora_scale = lora_scale
    job.guidance = guidance
    job.schedule = schedule
    job.img2img_strength = img2img_strength
    job.negative_prompt = negative_prompt

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
        args=(job, generation_prompt, width, height, steps, seed, input_image_path, reference_image_paths, show_steps, guidance, schedule, lora_name, lora_scale, img2img_strength, negative_prompt),
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
    THUMB_DIR.mkdir(exist_ok=True)
    thumb_path = THUMB_DIR / f"{job_id}.jpg"

    # Serve existing valid thumbnail directly (immutable cache — content never changes)
    if _is_valid_thumb(thumb_path):
        response = send_file(thumb_path, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    # No valid thumbnail — generate on demand from the original
    original_path = OUTPUT_DIR / f"{job_id}.png"
    if not original_path.exists():
        return jsonify({"error": "Image not found"}), 404

    if generate_thumbnail(job_id, original_path):
        response = send_file(thumb_path, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    # PIL unavailable or generation failed — fall back to serving the original
    return send_file(original_path, mimetype="image/png")


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
        return jsonify([
            job.to_history_dict() for job in history
            if job.output_path and job.output_path.exists()
        ])


@app.route("/history/<job_id>/favorite", methods=["POST"])
def toggle_favorite(job_id):
    """Toggle the favorited state of a history item."""
    with history_lock:
        job = history_by_id.get(job_id)
        if not job:
            return jsonify({"error": "Not found"}), 404
        job.favorited = not job.favorited
        mark_history_dirty()
        return jsonify({"favorited": job.favorited})


@app.route("/queue/reorder", methods=["POST"])
def reorder_queue():
    """Reorder pending jobs in the queue."""
    data = request.json or {}
    new_order = data.get("order", [])
    with job_queue_lock:
        by_id = {q["job"].id: q for q in job_queue}
        reordered = []
        seen = set()
        for job_id in new_order:
            if job_id in by_id:
                reordered.append(by_id[job_id])
                seen.add(job_id)
        for q in job_queue:
            if q["job"].id not in seen:
                reordered.append(q)
        job_queue[:] = reordered
        for i, q in enumerate(job_queue):
            q["job"].put_event(("queue_position", {"position": i + 1, "total": len(job_queue)}))
    return jsonify({"status": "ok"})


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


@app.route("/outpaint-prep", methods=["POST"])
def outpaint_prep():
    """Pad an image with neutral grey fill for outpainting.

    Request JSON:
        image   – base64 data-URI of the source image
        top     – extra pixels to add above (snapped to 16px grid)
        bottom  – extra pixels to add below
        left    – extra pixels to add to the left
        right   – extra pixels to add to the right

    Response JSON:
        padded_image  – base64 data-URI of the padded PNG
        width         – new canvas width
        height        – new canvas height
        orig_x        – x offset of original image in canvas
        orig_y        – y offset of original image in canvas
    """
    from PIL import Image as PILImage
    import io as _io

    data = request.json or {}
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "Image data required"}), 400

    try:
        raw = base64.b64decode(image_b64.split(",")[-1])
        src = PILImage.open(_io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    orig_w, orig_h = src.size

    # Parse and snap padding amounts to 16-pixel grid
    def _snap(v):
        v = max(0, int(v or 0))
        return round(v / 16) * 16

    pad_top    = _snap(data.get("top",    0))
    pad_bottom = _snap(data.get("bottom", 0))
    pad_left   = _snap(data.get("left",   0))
    pad_right  = _snap(data.get("right",  0))

    new_w = orig_w + pad_left + pad_right
    new_h = orig_h + pad_top  + pad_bottom

    if new_w > 1792 or new_h > 1792:
        return jsonify({"error": "Padded size exceeds maximum (1792 px)"}), 400

    if new_w == orig_w and new_h == orig_h:
        return jsonify({"error": "No expansion selected"}), 400

    # Create padded canvas with neutral grey fill (128,128,128)
    canvas = PILImage.new("RGB", (new_w, new_h), (128, 128, 128))
    canvas.paste(src, (pad_left, pad_top))

    # Mirror-blend a thin strip from each edge into the padding zone so the
    # seam blends more gracefully (8 px fade using Image.blend)
    FADE = min(8, pad_top, pad_bottom, pad_left, pad_right,
               orig_w // 4, orig_h // 4)
    if FADE > 0:
        # Top edge
        if pad_top > 0:
            strip = src.crop((0, 0, orig_w, FADE)).transpose(PILImage.FLIP_TOP_BOTTOM)
            grey  = PILImage.new("RGB", (orig_w, FADE), (128, 128, 128))
            blend = PILImage.blend(strip, grey, 0.5)
            canvas.paste(blend, (pad_left, pad_top - FADE))
        # Bottom edge
        if pad_bottom > 0:
            strip = src.crop((0, orig_h - FADE, orig_w, orig_h)).transpose(PILImage.FLIP_TOP_BOTTOM)
            grey  = PILImage.new("RGB", (orig_w, FADE), (128, 128, 128))
            blend = PILImage.blend(strip, grey, 0.5)
            canvas.paste(blend, (pad_left, pad_top + orig_h))
        # Left edge
        if pad_left > 0:
            strip = src.crop((0, 0, FADE, orig_h)).transpose(PILImage.FLIP_LEFT_RIGHT)
            grey  = PILImage.new("RGB", (FADE, orig_h), (128, 128, 128))
            blend = PILImage.blend(strip, grey, 0.5)
            canvas.paste(blend, (pad_left - FADE, pad_top))
        # Right edge
        if pad_right > 0:
            strip = src.crop((orig_w - FADE, 0, orig_w, orig_h)).transpose(PILImage.FLIP_LEFT_RIGHT)
            grey  = PILImage.new("RGB", (FADE, orig_h), (128, 128, 128))
            blend = PILImage.blend(strip, grey, 0.5)
            canvas.paste(blend, (pad_left + orig_w, pad_top))

    buf = _io.BytesIO()
    canvas.save(buf, format="PNG")
    encoded = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "padded_image": encoded,
        "width":  new_w,
        "height": new_h,
        "orig_x": pad_left,
        "orig_y": pad_top,
    })


@app.route("/server-status")
def server_status():
    """Get server status including ready state and current job."""
    global iris_server
    status = {
        "ready": False,
        "busy": False,
        "current_job": None,
        "queue_length": 0,
    }

    if iris_server:
        status["ready"] = iris_server.ready
        if iris_server.current_job:
            status["busy"] = True
            status["current_job"] = iris_server.current_job.id

    with job_queue_lock:
        status["queue_length"] = len(job_queue)

    return jsonify(status)


@app.route("/model-info")
def model_info():
    """Get loaded model information."""
    global iris_server
    if not iris_server or not iris_server.ready:
        return jsonify({"error": "Server not ready"}), 503
    return jsonify({
        "model": iris_server.model_info,
        "is_distilled": iris_server.is_distilled,
        "is_zimage": iris_server.is_zimage,
    })


@app.route("/active-jobs")
def active_jobs():
    """Get all active (running + queued) jobs for recovery after page reload."""
    result = {
        "running": None,
        "queued": [],
    }

    # Get currently running job
    if iris_server and iris_server.current_job:
        job = iris_server.current_job
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


@app.route("/available-loras")
def available_loras():
    """List LoRA adapters available in the model's loras/ directory, plus the curated registry."""
    loras = []
    if model_dir_path:
        lora_dir = model_dir_path / "loras"
        if lora_dir.exists():
            for f in sorted(lora_dir.glob("*.safetensors")):
                size_mb = round(f.stat().st_size / 1_048_576, 1)
                loras.append({"name": f.stem, "filename": f.name, "size_mb": size_mb})

    # Annotate curated list with downloaded state
    downloaded_filenames = {item["filename"] for item in loras}
    curated = [
        {**entry, "downloaded": entry["filename"] in downloaded_filenames}
        for entry in CURATED_LORAS
    ]

    return jsonify({"loras": loras, "curated": curated})


@app.route("/available-models")
def available_models_endpoint():
    """List known model slots with download and current status."""
    slots = []
    for slot in MODEL_SLOTS:
        slot_dir = PROJECT_DIR / slot["key"]
        downloaded = (slot_dir / "transformer").is_dir()
        is_current = (iris_server is not None and
                      Path(iris_server.model_dir).resolve() == slot_dir.resolve())
        slots.append({
            "key": slot["key"],
            "label": slot["label"],
            "description": slot["description"],
            "downloaded": downloaded,
            "current": is_current,
            "downloadable": slot.get("sh_arg") is not None,
        })
    current_name = Path(iris_server.model_dir).name if iris_server else None
    return jsonify({"current_model_dir": current_name, "slots": slots})


@app.route("/switch-model", methods=["POST"])
def switch_model():
    """Switch the loaded model to a different directory (restarts the flux process)."""
    global iris_server, model_dir_path
    data = request.json or {}
    key = data.get("key", "").strip()
    slot = next((s for s in MODEL_SLOTS if s["key"] == key), None)
    if not slot:
        return jsonify({"error": "Unknown model key"}), 400
    new_dir = PROJECT_DIR / key
    if not (new_dir / "transformer").is_dir():
        return jsonify({"error": "Model not downloaded yet"}), 400
    if iris_server and Path(iris_server.model_dir).resolve() == new_dir.resolve():
        return jsonify({"error": "Already loaded"}), 400
    with job_queue_lock:
        busy = bool(iris_server and iris_server.current_job) or bool(job_queue)
    if busy:
        return jsonify({"error": "Cannot switch while jobs are running or queued"}), 409

    def do_switch():
        global model_dir_path
        iris_server.switch(new_dir)
        model_dir_path = new_dir

    threading.Thread(target=do_switch, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/download-model", methods=["POST"])
def download_model_endpoint():
    """Start downloading a model via download_model.sh."""
    data = request.json or {}
    key = data.get("key", "").strip()
    slot = next((s for s in MODEL_SLOTS if s["key"] == key), None)
    if not slot or not slot.get("sh_arg"):
        return jsonify({"error": "Unknown or non-downloadable model key"}), 400
    # Idempotent: if already running, return ok
    state = _model_download_progress.get(key, {})
    if state and not state.get("done") and not state.get("error"):
        return jsonify({"ok": True, "already_running": True})

    _model_download_progress[key] = {"done": False, "error": None}

    def do_download():
        script = PROJECT_DIR / "download_model.sh"
        cmd = ["bash", str(script), slot["sh_arg"]]
        token = _hf_token
        if token:
            cmd += ["--token", token]
        try:
            proc = subprocess.run(cmd, cwd=str(PROJECT_DIR), capture_output=True, text=True)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "Download failed")[-500:]
                _model_download_progress[key]["error"] = err
            else:
                _model_download_progress[key]["done"] = True
        except Exception as e:
            _model_download_progress[key]["error"] = str(e)

    threading.Thread(target=do_download, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/download-model/progress/<key>")
def download_model_progress_endpoint(key):
    """Poll download progress for a model download."""
    state = _model_download_progress.get(key)
    if state is None:
        return jsonify({"error": "Not started"}), 404
    slot = next((s for s in MODEL_SLOTS if s["key"] == key), None)
    files_done = 0
    files_total = 0
    if slot:
        out_dir = PROJECT_DIR / key
        files_total = len(slot.get("expected_files", []))
        files_done = sum(1 for f in slot.get("expected_files", []) if (out_dir / f).exists())
    return jsonify({**state, "files_done": files_done, "files_total": files_total})


@app.route("/delete-model", methods=["POST"])
def delete_model_endpoint():
    """Delete a downloaded model directory from disk."""
    import shutil
    data = request.json or {}
    key = data.get("key", "").strip()
    slot = next((s for s in MODEL_SLOTS if s["key"] == key), None)
    if not slot:
        return jsonify({"error": "Unknown model key"}), 400
    model_dir = PROJECT_DIR / key
    if not model_dir.is_dir():
        return jsonify({"error": "Model not downloaded"}), 400
    if iris_server and Path(iris_server.model_dir).resolve() == model_dir.resolve():
        return jsonify({"error": "Cannot delete the currently loaded model"}), 400
    try:
        shutil.rmtree(model_dir)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True})


def _do_download(dl_id, url, dest, extra_headers=None):
    """Background thread: download a file from url to dest, tracking progress."""
    try:
        headers = {"User-Agent": "flux2.c/1.0"}
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        _download_progress[dl_id]["percent"] = int(downloaded * 100 / total)
        _download_progress[dl_id]["done"] = True
        _download_progress[dl_id]["percent"] = 100
    except Exception as e:
        _download_progress[dl_id]["error"] = str(e)
        # Remove partial file on error
        try:
            if dest.exists():
                dest.unlink()
        except Exception:
            pass


@app.route("/download-lora", methods=["POST"])
def download_lora():
    """Start downloading a LoRA (HuggingFace or Civitai) to the model's loras/ directory."""
    data = request.json or {}
    dl_id = data.get("id", "").strip()
    source = data.get("source", "huggingface")
    filename = data.get("filename", "").strip()

    if not dl_id or not filename:
        return jsonify({"error": "id and filename are required"}), 400

    # Basic path safety: filename must be a plain filename with no slashes
    if "/" in filename or "\\" in filename or ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400

    if not model_dir_path:
        return jsonify({"error": "Model directory not configured"}), 500

    lora_dir = model_dir_path / "loras"
    lora_dir.mkdir(exist_ok=True)
    dest = lora_dir / filename

    if dest.exists():
        return jsonify({"ok": True, "already_exists": True})

    extra_headers = {}
    civitai_token = os.environ.get("CIVITAI_TOKEN")

    if source == "civitai":
        civitai_model_id = data.get("civitai_model_id")
        version_filter = (data.get("civitai_version_filter") or "").strip().lower()
        if not civitai_model_id:
            return jsonify({"error": "civitai_model_id required for civitai source"}), 400
        try:
            api_url = f"https://civitai.com/api/v1/models/{civitai_model_id}"
            api_headers = {"User-Agent": "flux2.c/1.0"}
            if civitai_token:
                api_headers["Authorization"] = f"Bearer {civitai_token}"
            api_req = urllib.request.Request(api_url, headers=api_headers)
            with urllib.request.urlopen(api_req, timeout=15) as resp:
                model_data = json.loads(resp.read())
            versions = model_data.get("modelVersions", [])
            if not versions:
                return jsonify({"error": "No versions found for this Civitai model"}), 400
            if version_filter:
                matching = [v for v in versions if version_filter in v.get("name", "").lower()]
                version = matching[0] if matching else versions[0]
            else:
                version = versions[0]
            url = f"https://civitai.com/api/download/models/{version['id']}"
            if civitai_token:
                extra_headers["Authorization"] = f"Bearer {civitai_token}"
        except Exception as e:
            return jsonify({"error": f"Failed to resolve Civitai model: {e}"}), 500
    else:
        repo = data.get("repo", "").strip()
        if not repo:
            return jsonify({"error": "repo is required for huggingface source"}), 400
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        if _hf_token:
            extra_headers["Authorization"] = f"Bearer {_hf_token}"

    _download_progress[dl_id] = {"percent": 0, "done": False, "error": None}
    threading.Thread(
        target=_do_download, args=(dl_id, url, dest), kwargs={"extra_headers": extra_headers}, daemon=True
    ).start()
    return jsonify({"ok": True})


@app.route("/download-lora/progress/<dl_id>")
def download_lora_progress(dl_id):
    """Poll download progress for a given download id."""
    state = _download_progress.get(dl_id)
    if state is None:
        return jsonify({"error": "Unknown download id"}), 404
    return jsonify(state)


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    """Cancel a running or queued job."""
    global iris_server

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
    if iris_server and iris_server.current_job and iris_server.current_job.id == job_id:
        job.status = "cancelled"
        job.error = "Cancelled by user"
        job.put_event(("error", {"message": "Cancelled"}))
        job.put_event(("done", None))
        job.cleanup_temp_files()

        # Restart the flux server to stop the current generation
        # Do this in a background thread to not block the response
        def restart_and_process_queue():
            iris_server.restart()
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


def _load_config():
    global _hf_token
    if CONFIG_FILE and CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            if not os.environ.get("HF_TOKEN"):
                _hf_token = data.get("hf_token", "")
        except Exception:
            pass


def _save_config():
    if CONFIG_FILE:
        try:
            CONFIG_FILE.write_text(json.dumps({"hf_token": _hf_token}))
        except Exception:
            pass


@app.route("/settings", methods=["GET"])
def get_settings():
    return jsonify({"hf_token_set": bool(_hf_token)})


@app.route("/settings", methods=["POST"])
def post_settings():
    global _hf_token
    data = request.json or {}
    if "hf_token" in data:
        _hf_token = data["hf_token"].strip()
        _save_config()
    return jsonify({"ok": True})


def main():
    global iris_server, model_dir_path

    parser = argparse.ArgumentParser(description="FLUX.2 Web UI Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Path to model directory")
    parser.add_argument("--iris-binary", type=Path, default=DEFAULT_IRIS_BINARY,
                        help="Path to flux binary")
    args = parser.parse_args()

    # Validate paths
    if not args.iris_binary.exists():
        print(f"Error: iris binary not found at {args.iris_binary}")
        print("Build it first with: make mps (or make blas)")
        return 1

    if not args.model_dir.exists():
        print(f"Error: Model directory not found at {args.model_dir}")
        print("Download it first with: ./download_model.sh")
        return 1

    model_dir_path = args.model_dir

    # Ensure the startup model dir is listed as a switchable slot so users can
    # switch back to it after switching to another model.
    startup_key = args.model_dir.resolve().name
    if not any(s["key"] == startup_key for s in MODEL_SLOTS):
        MODEL_SLOTS.insert(0, {
            "key": startup_key,
            "label": startup_key,
            "description": "Startup model",
            "sh_arg": None,
            "expected_files": ["model_index.json"],
        })

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    global CONFIG_FILE
    CONFIG_FILE = OUTPUT_DIR / "config.json"
    _load_config()

    # Load history from disk
    load_history_from_disk()

    # Start background threads
    history_saver_thread = threading.Thread(target=periodic_history_saver, daemon=True)
    history_saver_thread.start()

    job_cleanup_thread = threading.Thread(target=periodic_job_cleanup, daemon=True)
    job_cleanup_thread.start()

    # Start flux server (persistent model)
    print("Starting iris server with persistent model...")
    iris_server = IrisServer(args.iris_binary, args.model_dir)
    iris_server.start()

    print(f"\nFLUX.2 Web UI")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Model: {args.model_dir}")
    print(f"  Binary: {args.iris_binary}")
    print(f"  Mode: Server (persistent model)")
    print(f"  Press Ctrl+C to stop\n")

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        print("\nStopping flux server...")
        iris_server.stop()


if __name__ == "__main__":
    main()
