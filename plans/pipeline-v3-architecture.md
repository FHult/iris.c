# Pipeline V3 Architecture — Container-Native, Multi-Node

**Status:** Research / Pre-design  
**Scope:** IP-Adapter training pipeline, generalisable to other training recipes  
**Target hardware:** Apple Silicon cluster (M-series Mac Minis / Mac Studios) + optional Linux nodes for CPU-only steps  
**Date:** 2026-04-29  
**Prerequisite:** V2 pipeline fully operational (see `plans/pipeline-v2-architecture.md`)

---

## 1. Motivation

V2 runs on a single M1 Max with a local 2 TB NVMe. It handles crashes well, produces good
checkpoints, and the orchestrator manages the full pipeline autonomously. The limits of V2 are:

- **Single-node throughput ceiling.** Training is ~0.226 steps/s on one M1 Max. A 225 K-step
  full run takes ~11 days. Adding a second Apple Silicon node could halve that for
  embarrassingly parallel steps (precompute, shard building) and enable data parallelism during training.
- **No remote observability.** Status requires SSH + CLI. There is no web dashboard or
  mobile-accessible control plane.
- **tmux-coupled process management.** All process lifecycle is managed via tmux windows.
  Scaling this to multiple machines means SSH + tmux sessions per node, which doesn't compose.
- **Local disk dependency.** All intermediate artefacts live on one NVMe. Any single-disk
  failure loses everything. A distributed store adds resilience and enables multi-node data access.

V3 addresses these by containerising the pipeline, adopting Kubernetes as the scheduler,
replacing file-based state with a shared store, and adding a web control plane.

---

## 2. Design Goals

| Goal | Constraint |
|------|-----------|
| Scale data prep across Linux + Apple Silicon nodes | GPU steps (precompute, train) remain Apple Silicon; CPU/IO steps run anywhere |
| Single orchestrator with a web GUI | REST API; React or HTMX frontend; mobile-accessible |
| All state survives pod/node failure | Redis or Postgres replaces sentinel files; object storage replaces local NVMe for artefacts |
| Keep MLX/Metal for training | V3 does not port training code; see V4 for CUDA path |
| Deployment is reproducible | All containers built from Dockerfiles in this repo; Helm chart or Kustomize for K8s manifests |
| Observable without SSH | Structured logs → centralised log sink; metrics → Prometheus/Grafana; alerts → push notification |

---

## 3. The Critical Decision: MLX Constraint

MLX only runs on Apple Silicon. This is the single biggest architectural constraint in V3.

**What it means in practice:**

| Step | Compute requirement | Node type |
|------|--------------------|-----------| 
| download / convert | CPU + network I/O | Any Linux or Apple Silicon |
| build_shards / filter | CPU + disk I/O | Any Linux or Apple Silicon |
| clip_embed / dedup | CPU or CUDA (CLIP) | Any Linux or Apple Silicon |
| precompute (Qwen3 + VAE) | MLX GPU required | **Apple Silicon only** |
| train | MLX GPU required | **Apple Silicon only** |
| mine / validate | Light CPU | Any node |
| orchestrator / web GUI | CPU only | Any node |

A V3 cluster could be heterogeneous:
- 1–3 Apple Silicon nodes (M4 Mac Mini or Mac Studio) for GPU work
- Optional Linux x86 nodes for bulk data download and shard building

The Apple Silicon nodes run Darwin. Kubernetes on Darwin requires either:
- **OrbStack** (lightweight Linux VM per node, shares Metal access via virtio-gpu; experimental for ML)
- **Native Darwin K8s** (e.g. k3s compiled for arm64-darwin; containers are Darwin processes, not Linux)
- **VMs with GPU passthrough** (not supported on Apple Silicon; Metal is not passthrough-capable)

The practical answer for a small home cluster: **k3s on each Mac, containers are Docker/OCI
images built for arm64-darwin, shared NFS or object storage for artefacts.** This is less
"pure Kubernetes" than a cloud deployment but fully functional and keeps Metal access.

---

## 4. Container Architecture

### 4.1 Container Inventory

```
┌─────────────────────────────────────────────────────────┐
│  iris-orchestrator                                       │
│  Python FastAPI service                                  │
│  - State machine (ported from orchestrator.py)          │
│  - REST API for pipeline_ctl operations                  │
│  - Webhook / push notification sender                    │
│  - Schedules Jobs onto K8s via python-kubernetes client  │
└──────────────────┬──────────────────────────────────────┘
                   │ creates/monitors Jobs
        ┌──────────┼──────────┬──────────────┬────────────┐
        ▼          ▼          ▼              ▼            ▼
  iris-download  iris-prep  iris-precompute  iris-train  iris-web
  (Job)          (Job)      (Job, GPU node)  (Job, GPU)  (Deployment)
```

**iris-orchestrator** (Deployment, 1 replica)
- Replaces `orchestrator.py`; exposes REST API at `/api/v1/`
- Reads/writes state from Redis (replaces `pipeline_state.json` + sentinel files)
- Uses the Kubernetes Jobs API to launch pipeline steps
- Writes structured events to a log stream (replaces `orchestrator.jsonl`)
- Sends push notifications on escalation

**iris-download** (Job)
- Replaces `download_convert.py` + `downloader.py`
- Downloads and converts one chunk's worth of raw data
- Writes converted WDS tars to shared object storage
- Runs on any node (no GPU required)

**iris-prep** (Job)
- Replaces `build_shards.py`, `filter_shards.py`, `clip_dedup.py`
- Reads converted tars from object storage; writes unified shards
- Runs on any node; can be parallelised across shard ranges

**iris-precompute** (Job, nodeAffinity: apple-silicon)
- Replaces `precompute_all.py`
- Reads shards from object storage; writes Qwen3 + VAE + SigLIP `.npz` caches
- Must run on an Apple Silicon node (MLX dependency)
- Writes precomputed caches back to shared storage

**iris-train** (Job, nodeAffinity: apple-silicon)
- Replaces `train_ip_adapter.py`
- Reads shards + precomputed caches from shared storage; writes checkpoints
- Must run on an Apple Silicon node (MLX dependency)
- Exposes `/health` liveness endpoint (replaces heartbeat file)
- Resumable: detects latest checkpoint on startup (already implemented)
- PodDisruptionBudget prevents eviction during training

**iris-web** (Deployment, 1 replica)
- New: web UI + mobile dashboard
- Reads pipeline state from Redis via orchestrator REST API
- Shows live training metrics, chunk progress, log tails, ETA
- Exposes pipeline controls: pause / resume / retry / abort
- Replaces `pipeline_status.py` + `pipeline_ctl.py` as primary interface

### 4.2 State Store Migration

| V2 mechanism | V3 replacement | Notes |
|---|---|---|
| `{step}.done` sentinel files | Redis key `chunk:{N}:step:{step}:done` (bool) | Atomic, survives orchestrator restart |
| `{step}.error` sentinel files | Redis key `chunk:{N}:step:{step}:error` (string) | Error message as value |
| `pipeline_state.json` | Redis hash `pipeline:state` | Written by orchestrator on every transition |
| Heartbeat `.json` files | K8s liveness probe + Redis key `hb:{process}:{chunk}` (JSON) | Written by worker, TTL-expired for staleness |
| `dispatch_queue.jsonl` | Redis list `dispatch:issues` (JSON entries) | RPUSH on new issue; LRANGE for status |
| `orchestrator.jsonl` | Structured log stream (stdout → log sink) | Parsed by web UI |
| `pipeline_control.json` | Redis pub/sub channel `pipeline:control` | Orchestrator subscribes; ctl publishes |

**Why Redis over Postgres:** Pipeline state has low data volume, high write frequency (one write per
step transition, ~12 transitions per chunk), and simple key-value semantics. Redis TTL handles
heartbeat staleness natively. Postgres is a better fit if query patterns become complex.

### 4.3 Artefact Storage Migration

| V2 path | V3 replacement | Access pattern |
|---|---|---|
| `/Volumes/2TBSSD/staging/chunk{N}/raw/` | Ephemeral pod volume (deleted after convert) | Write-once by download job |
| `/Volumes/2TBSSD/staging/chunk{N}/converted/` | Object storage bucket `converted/chunk{N}/` | Write by download, read by prep |
| `/Volumes/2TBSSD/shards/` | Object storage bucket `shards/` | Write by prep, read by precompute + train |
| `/Volumes/2TBSSD/precomputed/` | Object storage bucket `precomputed/` | Write by precompute, read by train |
| `/Volumes/2TBSSD/hard_examples/` | Object storage bucket `hard_examples/` | Write by mine, read by train |
| `/Volumes/2TBSSD/checkpoints/` | Object storage bucket `checkpoints/` | Write + read by train |

**Object storage options:**
- **MinIO** (self-hosted, S3-compatible) — runs as a K8s StatefulSet on the cluster; no cloud
  dependency; S3 API means code changes are minimal (swap `open(path)` for `boto3.get_object`)
- **NFS shared volume** — simpler; lower bandwidth than MinIO for concurrent access; fine for
  a 2-node cluster; poor scalability
- **S3/GCS** — right answer for cloud deployment (V4); latency for small `.npz` reads at
  training-time is the main concern (mitigated by local caching in the training container)

**Key performance concern:** The training step reads thousands of small `.npz` files per epoch
from precomputed caches. Over NFS or S3 this will be slow. The training container should
pre-cache the relevant shard's precomputed files to a local emptyDir volume at startup
(replicate the current `--precomputed-*` directory structure locally before training begins).

---

## 5. Orchestrator Refactor

The current `orchestrator.py` is a monolithic polling loop. V3 splits it into two concerns:

**State machine logic** (keep, refactor surface only)
- `derive_chunk_state()`, `_handle_error()`, `_advance_chunk()` all port directly
- Replace sentinel file reads with Redis calls
- Replace `mark_done()` / `mark_error()` with Redis writes
- Replace `tmux_new_window()` calls with `kubernetes.client.BatchV1Api.create_namespaced_job()`

**Scheduling loop** (replace polling with event-driven)
- K8s Jobs emit completion/failure events; orchestrator watches via `list_namespaced_job(watch=True)`
- No more 60s polling loop; state transitions are event-driven
- Retry logic, backoff, and escalation remain in Python (no change in semantics)

**REST API surface** (new)
```
GET  /api/v1/status          → full pipeline status JSON (replaces pipeline_status.py --json)
POST /api/v1/control/pause   → pause orchestrator
POST /api/v1/control/resume  → resume
POST /api/v1/control/retry   → {chunk, step} retry
GET  /api/v1/logs/{step}/{chunk}  → log tail (SSE stream)
GET  /api/v1/metrics         → Prometheus text format
```

---

## 6. Web GUI

Single-page app (or server-rendered HTMX) consuming the orchestrator REST API.

**Dashboard view:**
- Per-chunk progress: step name, % complete, ETA, loss/grad_norm for training chunks
- Resource view: which nodes are active, what's running where
- Log tail: live SSE stream for active step
- Dispatch issues panel: unresolved alerts with resolve button

**Control panel:**
- Pause / Resume
- Retry failed step
- Force-advance chunk (operator override)
- Trigger manual checkpoint

**Mobile:** The REST API + SSE log stream are mobile-browser compatible by construction.
No native app required.

---

## 7. Kubernetes Specifics

### 7.1 Node labelling

```yaml
# Apple Silicon nodes
kubectl label node mac-mini-1 iris/compute=apple-silicon
kubectl label node mac-studio-1 iris/compute=apple-silicon

# Linux nodes (optional, for CPU steps)
kubectl label node linux-worker-1 iris/compute=cpu
```

### 7.2 Training job affinity + disruption budget

```yaml
# iris-train Job spec excerpt
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: iris/compute
          operator: In
          values: [apple-silicon]
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: iris-train-pdb
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: iris-train
```

### 7.3 Liveness probe replacing heartbeat file

```yaml
# iris-train container spec excerpt
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 300   # allow model load + warmup
  periodSeconds: 60
  failureThreshold: 15       # 15 min unresponsive → restart
```

The training script exposes a minimal HTTP server in a background thread:
`GET /health` returns 200 + JSON `{step, loss, eta_sec}` if the training loop
is alive. This replaces `write_heartbeat()` entirely.

---

## 8. Migration Path from V2

V3 is a non-trivial refactor. Suggested phasing:

**Phase 1 — Containerise without K8s (Docker Compose)**
- Package each pipeline step as a Docker image (or Darwin-native OCI image)
- Replace sentinel files with Redis (drop-in via `pipeline_lib.py` Redis backend)
- Keep tmux for process launch (Docker Compose replaces tmux)
- Add minimal REST API to orchestrator (10 endpoints)
- Goal: same pipeline, same hardware, state now in Redis

**Phase 2 — Add web GUI**
- Build web dashboard consuming the REST API
- Deploy as a container alongside the orchestrator
- Validate mobile access

**Phase 3 — Move to K8s (single node first)**
- Deploy k3s on the M1 Max
- Port tmux/Docker Compose process launch to K8s Jobs
- Validate that the training Job lifecycle works correctly (startup, checkpoint, restart)

**Phase 4 — Add second Apple Silicon node**
- Add Mac Mini M4 to the cluster
- Precompute runs on node 2 while training runs on node 1
- Validate shared storage (MinIO or NFS)

---

## 9. Open Questions

1. **Darwin containers on K8s**: OrbStack's K8s support is the most mature path for running
   OCI containers that access Metal on macOS. Needs evaluation for the MLX use case specifically.
   If it doesn't work, the fallback is native Darwin processes managed by K8s as systemd units
   via a custom DaemonSet + exec model.

2. **Precomputed cache locality**: How much local cache pre-warming is needed in the training
   container before the forward pass begins? Benchmark: read 17 shards × ~4970 records × 2 files
   (qwen3 + vae) = ~170K file reads. Over NFS this is the main startup cost.

3. **Checkpoint storage**: 4 GB per checkpoint, every 500 steps. At 0.226 steps/s that's one
   checkpoint every ~37 minutes. Over MinIO (local NVMe backed) this should be fast. Over NFS,
   page-cache purge behaviour may differ from local disk.

4. **Single-node K8s overhead**: For a 1-node deployment, K8s adds scheduling overhead vs
   running Python scripts directly. Quantify before committing to K3s as the sole deployment
   model.

---

## 10. V4 Roadmap Item: PyTorch/CUDA Backend

Tracked separately in `plans/pipeline-v4-roadmap.md` when that document is created.

The key insight: V3 containerises the current MLX/Metal training code and targets Apple
Silicon clusters. A parallel V4 workstream would:

1. **Port the training backend to PyTorch** alongside the MLX backend. The IP-Adapter model
   (`IPAdapterKlein`), loss function, EMA, and dataset loader are all framework-agnostic in
   design. The main porting work is replacing `mx.*` calls with `torch.*` and rewriting the
   Metal-specific paths in `iris_transformer_flux.c` (not relevant for training — training is
   pure Python/MLX).

2. **Introduce a backend abstraction** in `train_ip_adapter.py`:
   ```
   --backend mlx    (default, current path)
   --backend torch  (new, CUDA/ROCm)
   ```
   Both backends produce identical checkpoints (safetensors format, same weight names).

3. **Enable cloud GPU training** once the PyTorch backend is functional:
   - `iris-train` container image gains a CUDA variant (`iris-train:cuda`)
   - K8s node selector `iris/compute=nvidia-gpu` targets cloud GPU nodes
   - Same orchestrator, same state machine, same checkpoint format
   - Apple Silicon nodes handle precompute (MLX-only); cloud GPU nodes handle training

4. **Cost model**: Spot A100/H100 instances cost ~$1-3/hr. A 22h training run at $2/hr = $44
   vs ~10 days of wall-clock on a single M1 Max. For iteration speed this is compelling once
   the dataset pipeline is stable.

The PyTorch/CUDA refactor is scoped as V4 because it requires:
- Re-implementing and validating all MLX training ops in PyTorch (risk of numerical divergence)
- Building and maintaining a second set of container images
- Cloud account setup, cost controls, and IAM for the training cluster

This is substantial work. It makes most sense after V3 is stable and the training recipe
is validated end-to-end (i.e. after a complete 4-chunk run produces a good model).
