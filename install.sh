#!/usr/bin/env bash
#
# iris.c installer — builds the engine, sets up the web UI, and checks for a model.
# Safe to re-run. After this finishes, run ./run.sh to open the app.
#
set -euo pipefail
cd "$(dirname "$0")"

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
info() { printf "  \033[34m•\033[0m %s\n" "$1"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$1"; }
die()  { printf "  \033[31m✗ %s\033[0m\n" "$1"; exit 1; }

bold "iris.c — installer"
echo

# ── 1. Detect the fastest backend this machine supports ──────────────────────
OS="$(uname -s)"; ARCH="$(uname -m)"
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    BACKEND="mps";   BACKEND_DESC="Apple Silicon GPU (Metal) — fastest"
elif command -v cc >/dev/null 2>&1 && { [ "$OS" = "Darwin" ] || ldconfig -p 2>/dev/null | grep -qi openblas; }; then
    BACKEND="blas";  BACKEND_DESC="BLAS-accelerated CPU"
else
    BACKEND="generic"; BACKEND_DESC="portable C (slow — no GPU/BLAS found)"
fi
info "Backend: $BACKEND ($BACKEND_DESC)"

# ── 2. Prerequisites ─────────────────────────────────────────────────────────
command -v make >/dev/null 2>&1 || die "'make' not found."
command -v cc   >/dev/null 2>&1 || command -v clang >/dev/null 2>&1 || die "No C compiler (cc/clang) found."
if [ "$OS" = "Darwin" ] && ! xcode-select -p >/dev/null 2>&1; then
    warn "Xcode Command Line Tools are required. Installing them now (a dialog may appear)…"
    xcode-select --install || true
    die "Re-run ./install.sh once the Command Line Tools finish installing."
fi
PY="$(command -v python3 || true)"
[ -n "$PY" ] || die "python3 not found — install Python 3 (e.g. 'brew install python') and re-run."
ok "Build tools and Python found."

# ── 3. Build the engine ──────────────────────────────────────────────────────
bold "Building the engine (make $BACKEND)…"
make "$BACKEND" >/tmp/iris_build.log 2>&1 || { tail -20 /tmp/iris_build.log; die "Build failed (full log: /tmp/iris_build.log)."; }
[ -x ./iris ] || die "Build finished but ./iris is missing."
ok "Built ./iris"

# ── 4. Web UI Python environment ─────────────────────────────────────────────
bold "Setting up the web UI…"
if [ ! -x web/venv/bin/python ]; then
    "$PY" -m venv web/venv || die "Could not create the Python virtualenv (web/venv)."
fi
web/venv/bin/pip install --quiet --upgrade pip >/dev/null 2>&1 || true
web/venv/bin/pip install --quiet -r web/requirements.txt || die "Could not install web dependencies."
ok "Web UI ready (web/venv)"

# ── 5. Model weights ─────────────────────────────────────────────────────────
have_model() { [ -d "$1/transformer" ] && [ -d "$1/vae" ] && [ -d "$1/text_encoder" ]; }
MODEL_DIR=""
for d in flux-klein-model flux-klein-4b flux-klein-4b-base flux-klein-9b flux-klein-9b-base zimage-turbo; do
    if have_model "$d"; then MODEL_DIR="$d"; break; fi
done

echo
if [ -n "$MODEL_DIR" ]; then
    ok "Found a model: ./$MODEL_DIR"
else
    warn "No model weights yet. The 4B model is recommended for a first install (~16 GB download)."
    if [ -t 0 ]; then
        read -r -p "  Download the 4B model now? [y/N] " ans
        if [ "${ans:-N}" = "y" ] || [ "${ans:-N}" = "Y" ]; then
            ./download_model.sh 4b || die "Download failed. FLUX.2 may require accepting its license on Hugging Face and an access token — see ./download_model.sh for --token."
        else
            info "Skipped. Download later with:  ./download_model.sh 4b"
        fi
    else
        info "Download a model with:  ./download_model.sh 4b   (then run ./run.sh)"
    fi
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo
bold "Install complete."
echo "  Start the app:   ./run.sh"
echo "  Then open:       http://localhost:8080"
echo
echo "  (Style references need a trained adapter bundle — drop it in ./sref-bundle"
echo "   or set IRIS_IP_BUNDLE before ./run.sh. Without one, normal generation still works.)"
