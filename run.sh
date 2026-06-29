#!/usr/bin/env bash
#
# Start the iris.c web app and open it in your browser.
# Run ./install.sh first if you haven't.
#
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8080}"

die() { printf "\033[31m%s\033[0m\n" "$1"; exit 1; }

[ -x ./iris ]                 || die "Engine not built. Run ./install.sh first."
[ -x web/venv/bin/python ]    || die "Web UI not set up. Run ./install.sh first."

# Find a model directory (downloaded by ./download_model.sh).
have_model() { [ -d "$1/transformer" ] && [ -d "$1/vae" ] && [ -d "$1/text_encoder" ]; }
MODEL_DIR=""
for d in flux-klein-model flux-klein-4b flux-klein-4b-base flux-klein-9b flux-klein-9b-base zimage-turbo; do
    if have_model "$d"; then MODEL_DIR="$d"; break; fi
done
[ -n "$MODEL_DIR" ] || die "No model found. Download one first:  ./download_model.sh 4b"

# Optional: a style-reference adapter bundle enables the --sref feature.
# Use IRIS_IP_BUNDLE if already set, else a ./sref-bundle directory if present.
if [ -z "${IRIS_IP_BUNDLE:-}" ] && [ -d ./sref-bundle ]; then
    export IRIS_IP_BUNDLE="$PWD/sref-bundle"
fi

echo "Model:  ./$MODEL_DIR"
[ -n "${IRIS_IP_BUNDLE:-}" ] && echo "Style references: enabled ($IRIS_IP_BUNDLE)" || echo "Style references: off (no adapter bundle)"
echo "Opening http://localhost:$PORT  (first image loads the model; it'll take ~30–60s)"

# Open the browser shortly after the server starts.
( sleep 3
  URL="http://localhost:$PORT"
  if command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"
  fi ) >/dev/null 2>&1 &

exec web/venv/bin/python web/server.py --model-dir "$MODEL_DIR" --port "$PORT"
