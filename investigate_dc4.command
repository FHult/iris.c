#!/bin/bash
# Diagnostic + restart script for download_convert chunk 4
# Output is written to a results file so Claude can read it

RESULTS="/Users/fredrikhult/src/iris.c/dc4_check_results.txt"
WORKSPACE="/Users/fredrikhult/src/iris.c"
DATA_ROOT="/Volumes/2TBSSD"

exec > "$RESULTS" 2>&1
echo "=== download_convert chunk 4 investigation: $(date) ==="

# 1. Check if PID 19548 is still alive
echo ""
echo "--- PID 19548 status ---"
ps aux | grep 19548 | grep -v grep || echo "PID 19548 is NOT running (already dead)"

# 2. All download_convert processes
echo ""
echo "--- All download_convert processes ---"
ps aux | grep download_convert | grep -v grep || echo "No download_convert processes running"

# 3. Heartbeat file
echo ""
echo "--- Heartbeat: download_convert_chunk4.json ---"
HB="$DATA_ROOT/.heartbeat/download_convert_chunk4.json"
if [ -f "$HB" ]; then
    python3 -c "
import json, time
data = json.load(open('$HB'))
print(json.dumps(data, indent=2))
# Check age
mtime = $(python3 -c "import os; print(int(os.path.getmtime('$HB')))" 2>/dev/null || echo 0)
age = int(time.time()) - mtime
print(f'Heartbeat age: {age}s ({age//3600}h {(age%3600)//60}m)')
" 2>/dev/null || python3 -c "
import json
print(json.dumps(json.load(open('$HB')), indent=2))
"
    # Age
    python3 -c "
import os, time
mtime = os.path.getmtime('$HB')
age = int(time.time() - mtime)
print(f'Heartbeat file age: {age}s ({age//3600}h {(age%3600)//60}m ago)')
"
else
    echo "Heartbeat file not found at $HB"
    echo "All heartbeat files:"
    ls -la "$DATA_ROOT/.heartbeat/" 2>/dev/null || echo "Cannot list heartbeat dir"
fi

# 4. Sentinel files for chunk 4
echo ""
echo "--- Sentinel files for chunk 4 ---"
ls -la "$DATA_ROOT/pipeline/chunk4/" 2>/dev/null || echo "Cannot access chunk4 sentinel dir"

# 5. Log file content
echo ""
echo "--- Log file for download_convert chunk4 ---"
LOG_FILE="$DATA_ROOT/logs/download_convert_chunk4.log"
if [ -f "$LOG_FILE" ]; then
    echo "Log file: $LOG_FILE"
    python3 -c "
f = open('$LOG_FILE', 'rb')
f.seek(0, 2)
size = f.tell()
print(f'Total log size: {size} bytes')
f.seek(max(0, size - 6000))
print('--- Last 6KB ---')
print(f.read().decode('utf-8', 'replace'))
"
else
    echo "Log not found at $LOG_FILE"
    echo "Available logs:"
    ls -la "$DATA_ROOT/logs/" 2>/dev/null | grep -i "download\|convert" || echo "No matching logs"
fi

# 6. Check orchestrator log for context
echo ""
echo "--- Recent orchestrator events for chunk4 download ---"
ORCH_LOG="$DATA_ROOT/logs/orchestrator.jsonl"
if [ -f "$ORCH_LOG" ]; then
    python3 -c "
import json
lines = open('$ORCH_LOG').readlines()
relevant = [l for l in lines if 'chunk4' in l.lower() and ('download' in l.lower() or 'convert' in l.lower())]
for l in relevant[-20:]:
    try:
        d = json.loads(l)
        print(json.dumps(d))
    except:
        print(l.rstrip())
" 2>/dev/null || echo "Could not parse orchestrator log"
else
    echo "No orchestrator.jsonl found"
fi

# 7. Decide what to do and restart if needed
echo ""
echo "--- DECISION ---"

# Check if already complete
DONE_SENTINEL="$DATA_ROOT/pipeline/chunk4/download.done"
CONV_SENTINEL="$DATA_ROOT/pipeline/chunk4/convert.done"
if [ -f "$DONE_SENTINEL" ] || [ -f "$CONV_SENTINEL" ]; then
    echo "SENTINEL EXISTS — chunk4 download/convert already marked done. No restart needed."
    ls -la "$DATA_ROOT/pipeline/chunk4/"*.done 2>/dev/null
    echo "=== No action taken (already complete) ==="
    exit 0
fi

# PID not running and no done sentinel: restart
echo "PID 19548 not running and no done sentinel found."
echo "Restarting download_convert for chunk 4..."

VENV_PYTHON="$WORKSPACE/train/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON=$(which python3)
    echo "Venv not found, using: $VENV_PYTHON"
fi

CONFIG="$WORKSPACE/train/configs/v2_pipeline.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    ls "$WORKSPACE/train/configs/"
    echo "=== RESTART FAILED: config missing ==="
    exit 1
fi

echo "Command: $VENV_PYTHON $WORKSPACE/train/scripts/download_convert.py --chunk 4 --config $CONFIG"
echo "Starting in background, logging to: $DATA_ROOT/logs/download_convert_chunk4.log"

# Run in background, redirect to log
nohup "$VENV_PYTHON" "$WORKSPACE/train/scripts/download_convert.py" \
    --chunk 4 \
    --config "$CONFIG" \
    >> "$DATA_ROOT/logs/download_convert_chunk4.log" 2>&1 &

NEW_PID=$!
echo "New PID: $NEW_PID"
sleep 2
if kill -0 "$NEW_PID" 2>/dev/null; then
    echo "Process is alive after 2s. Restart successful."
else
    echo "WARNING: Process died within 2s. Check log for errors."
fi

echo ""
echo "=== Investigation complete: $(date) ==="
echo "Results saved to: $RESULTS"
