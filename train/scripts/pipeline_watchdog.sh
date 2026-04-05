#!/bin/bash
# train/scripts/pipeline_watchdog.sh — Training stall detector.
#
# Polls the training heartbeat file every INTERVAL seconds (default 5 min).
# Sends a macOS notification if:
#   - heartbeat goes stale (age > STALE_THRESHOLD_S, default 5 min)
#   - training process exits unexpectedly
#
# Usage:
#   # Start in background (pipeline_start.sh does this automatically):
#   bash train/scripts/pipeline_watchdog.sh --heartbeat PATH --pid TRAIN_PID &
#
#   # Manual one-shot check:
#   bash train/scripts/pipeline_watchdog.sh --check-only
#
# Options:
#   --heartbeat PATH    Path to heartbeat.json (auto-detected if omitted)
#   --pid PID           PID of training process to monitor (optional)
#   --interval N        Poll interval in seconds (default: 300)
#   --stale N           Stale threshold in seconds (default: 300)
#   --data-root PATH    DATA_ROOT for auto-detecting heartbeat (default: /Volumes/2TBSSD)
#   --check-only        Run one check and exit (no loop)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────────────
HEARTBEAT_PATH=""
TRAIN_PID=""
INTERVAL=300
STALE_THRESHOLD=300
DATA_ROOT=""
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --heartbeat)  HEARTBEAT_PATH="$2"; shift 2 ;;
        --pid)        TRAIN_PID="$2";      shift 2 ;;
        --interval)   INTERVAL="$2";       shift 2 ;;
        --stale)      STALE_THRESHOLD="$2"; shift 2 ;;
        --data-root)  DATA_ROOT="$2";      shift 2 ;;
        --check-only) CHECK_ONLY=true;     shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Auto-detect DATA_ROOT ─────────────────────────────────────────────────────
if [[ -z "$DATA_ROOT" ]]; then
    for candidate in /Volumes/2TBSSD /Volumes/IrisData /Volumes/TrainData "$TRAIN_DIR/data"; do
        if [[ -d "$candidate" ]]; then DATA_ROOT="$candidate"; break; fi
    done
    DATA_ROOT="${DATA_ROOT:-$TRAIN_DIR/data}"
fi

# ── Auto-detect heartbeat path ────────────────────────────────────────────────
if [[ -z "$HEARTBEAT_PATH" ]]; then
    HEARTBEAT_PATH=$(ls -t \
        "$TRAIN_DIR/checkpoints/stage1/heartbeat.json" \
        "$TRAIN_DIR/checkpoints/stage2/heartbeat.json" \
        "$TRAIN_DIR/checkpoints/heartbeat.json" \
        "$DATA_ROOT/logs/heartbeat.json" \
        2>/dev/null | head -1 || true)
fi

# ── Notification helper ───────────────────────────────────────────────────────
notify() {
    local title="$1" msg="$2"
    # macOS notification via osascript
    osascript -e "display notification \"$msg\" with title \"$title\" sound name \"Basso\"" \
        2>/dev/null || true
    echo "[watchdog $(date '+%H:%M:%S')] ALERT: $title — $msg"
}

# ── Single check function ─────────────────────────────────────────────────────
_last_alert=""  # debounce: only alert once per stale period

do_check() {
    # 1. If we have a PID, check if it's still alive
    if [[ -n "$TRAIN_PID" ]]; then
        if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
            if [[ "$_last_alert" != "dead" ]]; then
                notify "Training stopped" "PID $TRAIN_PID is no longer running"
                _last_alert="dead"
            fi
            return
        fi
    fi

    # 2. Check heartbeat staleness
    if [[ -z "$HEARTBEAT_PATH" || ! -f "$HEARTBEAT_PATH" ]]; then
        echo "[watchdog $(date '+%H:%M:%S')] No heartbeat file found — waiting for training to start"
        return
    fi

    AGE_S=$(python3 - "$HEARTBEAT_PATH" "$STALE_THRESHOLD" 2>/dev/null <<'PYEOF'
import json, sys, time
try:
    d = json.load(open(sys.argv[1]))
    ts = d.get("timestamp", "")
    threshold = int(sys.argv[2])
    if not ts:
        print(-1); sys.exit()
    age = int(time.time()) - int(time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S")))
    print(age)
except Exception:
    print(-1)
PYEOF
)

    if [[ "$AGE_S" -lt 0 ]]; then
        echo "[watchdog $(date '+%H:%M:%S')] Could not parse heartbeat — skipping"
        return
    fi

    if [[ "$AGE_S" -ge "$STALE_THRESHOLD" ]]; then
        # Only alert once per stale period — debounce on "stale", not on age
        # (age increments every second; using it as the key would re-fire every loop).
        if [[ "$_last_alert" != "stale" ]]; then
            STEP=$(python3 -c "import json; d=json.load(open('$HEARTBEAT_PATH')); \
print(f\"step {d.get('step',0):,}/{d.get('total_steps',0):,}\")" 2>/dev/null || echo "unknown step")
            notify "Training stall detected" "Heartbeat ${AGE_S}s stale — $STEP"
            _last_alert="stale"
        fi
    else
        # Heartbeat is fresh — clear alert state so next stale period re-alerts
        _last_alert=""
        echo "[watchdog $(date '+%H:%M:%S')] OK — heartbeat ${AGE_S}s old"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "[watchdog] Started  interval=${INTERVAL}s  stale_threshold=${STALE_THRESHOLD}s"
echo "[watchdog] Heartbeat: ${HEARTBEAT_PATH:-auto-detect}"
[[ -n "$TRAIN_PID" ]] && echo "[watchdog] Monitoring PID: $TRAIN_PID"

if $CHECK_ONLY; then
    do_check
    exit 0
fi

while true; do
    do_check
    sleep "$INTERVAL"
done
