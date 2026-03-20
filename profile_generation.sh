#!/bin/bash
# Profile a single generation and summarize timing

PROMPT="${1:-A fluffy orange cat}"
OUTPUT="/tmp/profile_test_$$.png"

echo "Profiling generation: '$PROMPT'"
echo "============================================"

# Run flux and capture JSON output, parsing timing
./flux --server <<< "{\"prompt\": \"$PROMPT\", \"output\": \"$OUTPUT\", \"width\": 512, \"height\": 512, \"steps\": 4, \"show_steps\": false}" 2>/dev/null | \
while IFS= read -r line; do
    event=$(echo "$line" | grep -o '"event":"[^"]*"' | cut -d'"' -f4)
    case "$event" in
        ready)
            echo "Server ready"
            ;;
        phase)
            phase=$(echo "$line" | grep -o '"phase":"[^"]*"' | cut -d'"' -f4)
            elapsed=$(echo "$line" | grep -o '"elapsed":[0-9.]*' | cut -d: -f2)
            printf "  %-25s started at %.2fs\n" "$phase" "$elapsed"
            ;;
        phase_done)
            phase=$(echo "$line" | grep -o '"phase":"[^"]*"' | cut -d'"' -f4)
            phase_time=$(echo "$line" | grep -o '"phase_time":[0-9.]*' | cut -d: -f2)
            printf "  %-25s took %.2fs\n" "$phase" "$phase_time"
            ;;
        progress)
            step=$(echo "$line" | grep -o '"step":[0-9]*' | cut -d: -f2)
            total=$(echo "$line" | grep -o '"total":[0-9]*' | cut -d: -f2)
            step_time=$(echo "$line" | grep -o '"step_time":[0-9.]*' | cut -d: -f2)
            printf "  Step %d/%d               took %.2fs\n" "$step" "$total" "$step_time"
            ;;
        complete)
            total_time=$(echo "$line" | grep -o '"total_time":[0-9.]*' | cut -d: -f2)
            echo "============================================"
            printf "TOTAL: %.2fs\n" "$total_time"
            rm -f "$OUTPUT"
            exit 0
            ;;
        error)
            msg=$(echo "$line" | grep -o '"message":"[^"]*"' | cut -d'"' -f4)
            echo "ERROR: $msg"
            exit 1
            ;;
    esac
done
