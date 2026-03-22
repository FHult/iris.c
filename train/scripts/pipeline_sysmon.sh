#!/bin/bash
# train/scripts/pipeline_sysmon.sh — System observability snapshot.
#
# One-shot report of: CPU utilisation by core class, memory pressure and swap,
# GPU memory allocation, disk I/O throughput, network bytes in/out, and a
# per-process breakdown for all active pipeline processes.
#
# No flags required. Designed for remote monitoring via Claude CoWork Dispatch.
#
# Usage:
#   bash train/scripts/pipeline_sysmon.sh

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "================================================================="
echo "  System snapshot — $(ts)"
echo "================================================================="
echo ""

# ── CPU ───────────────────────────────────────────────────────────────────────
echo "── CPU ──────────────────────────────────────────────────────────"

# Core counts
P_CORES=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || echo "?")
E_CORES=$(sysctl -n hw.perflevel1.logicalcpu 2>/dev/null || echo "?")
TOTAL_CORES=$(sysctl -n hw.logicalcpu 2>/dev/null || echo "?")
printf "  Cores: %s total  (%s P-cores + %s E-cores)\n" "$TOTAL_CORES" "$P_CORES" "$E_CORES"

# CPU utilisation: take two top samples 1s apart so we get a real percentage
CPU_LINE=$(top -l 2 -n 0 -s 1 2>/dev/null | grep "^CPU" | tail -1)
if [[ -n "$CPU_LINE" ]]; then
    printf "  %-60s\n" "$CPU_LINE"
fi

# Load average
LOAD=$(sysctl -n vm.loadavg 2>/dev/null | tr -d '{}')
printf "  Load avg: %s\n" "$LOAD"
echo ""

# ── Memory ────────────────────────────────────────────────────────────────────
echo "── Memory ───────────────────────────────────────────────────────"

TOTAL_MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
TOTAL_MEM_GB=$(( TOTAL_MEM_BYTES / 1024 / 1024 / 1024 ))
printf "  Physical RAM: %d GB\n" "$TOTAL_MEM_GB"

# Memory pressure via vm_stat (page size on Apple Silicon = 16 KB)
PAGE_SIZE=$(pagesize 2>/dev/null || echo 16384)
vm_stat 2>/dev/null | awk -v ps="$PAGE_SIZE" '
    /Pages free/          { free=$NF+0 }
    /Pages active/        { active=$NF+0 }
    /Pages inactive/      { inactive=$NF+0 }
    /Pages wired/         { wired=$NF+0 }
    /Pages occupied by compressor/ { compressed=$NF+0 }
    END {
        total = free + active + inactive + wired + compressed
        used  = active + inactive + wired + compressed
        gb    = 1024*1024*1024 / ps
        printf "  Free:       %5.1f GB\n", free/gb
        printf "  Active:     %5.1f GB\n", active/gb
        printf "  Wired:      %5.1f GB\n", wired/gb
        printf "  Compressed: %5.1f GB\n", compressed/gb
    }
'

# Swap
SWAP=$(sysctl vm.swapusage 2>/dev/null | sed 's/vm.swapusage: //')
printf "  Swap: %s\n" "$SWAP"
echo ""

# ── GPU ───────────────────────────────────────────────────────────────────────
echo "── GPU (Apple Silicon unified memory) ───────────────────────────"

# GPU memory via IOKit — works without sudo
GPU_MEM=$(ioreg -r -c AGXAccelerator -d 2 2>/dev/null \
    | grep -E '"DeviceMemoryUsed"|"IOGPUCurrentOccupancy"' \
    | awk -F'= ' '{sum += $2} END {printf "%.1f MB\n", sum/1024/1024}' 2>/dev/null || echo "n/a")
printf "  GPU memory allocated: %s\n" "$GPU_MEM"

# Metal command queue activity (shows if GPU is busy with compute)
METAL_PROCS=$(pgrep -l -f "train_ip_adapter\|precompute_qwen3\|precompute_vae\|build_shards" 2>/dev/null \
    | awk '{print $2}' | xargs basename 2>/dev/null | sort -u | tr '\n' ' ')
if [[ -n "$METAL_PROCS" ]]; then
    printf "  Metal processes: %s\n" "$METAL_PROCS"
else
    printf "  Metal processes: (none)\n"
fi
echo ""

# ── Disk I/O ──────────────────────────────────────────────────────────────────
echo "── Disk I/O (1-second delta sample) ────────────────────────────"

# iostat enumerates physical disks (disk0, disk4, ...) independently of APFS
# volume names (/dev/disk5s1, etc.). Show all disks; second sample = 1-sec delta.
iostat -d -c 2 -w 1 2>/dev/null | awk '
    NR==1 { printf "  disks:  %s\n", $0; next }
    NR==2 { printf "  cols:   %s\n", $0; next }
    NR==4 { printf "  delta:  %s\n", $0 }
'

echo ""
echo "── Disk usage ───────────────────────────────────────────────────"
for vol in /Volumes/2TBSSD /Volumes/IrisData /Volumes/TrainData /; do
    [[ -d "$vol" ]] || continue
    df -h "$vol" 2>/dev/null | awk -v v="$vol" 'NR==2{
        printf "  %-20s used=%-8s avail=%-8s capacity=%s\n", v, $3, $4, $5}'
done
echo ""

# ── Network ───────────────────────────────────────────────────────────────────
echo "── Network ──────────────────────────────────────────────────────"
netstat -ib 2>/dev/null | awk '
    NR==1 { next }
    /^en/ && !seen[$1]++ && ($7+0 > 0 || $10+0 > 0) {
        ib=$7+0; ob=$10+0
        iu="B"; ou="B"
        if (ib>1073741824){ib=ib/1073741824; iu="GB"} else if (ib>1048576){ib=ib/1048576; iu="MB"} else if (ib>1024){ib=ib/1024; iu="KB"}
        if (ob>1073741824){ob=ob/1073741824; ou="GB"} else if (ob>1048576){ob=ob/1048576; ou="MB"} else if (ob>1024){ob=ob/1024; ou="KB"}
        printf "  %-8s  in=%7.1f %-2s  out=%7.1f %-2s\n", $1, ib, iu, ob, ou
    }
' | head -5
echo ""

# ── Per-process pipeline breakdown ────────────────────────────────────────────
echo "── Pipeline processes ───────────────────────────────────────────"
PIPELINE_PIDS=$(pgrep -f "train_ip_adapter|build_shards|filter_shards|precompute_qwen3|precompute_vae|precompute_siglip|clip_dedup|recaption|run_training_pipeline|run_shard_and_precompute" 2>/dev/null || true)

if [[ -n "$PIPELINE_PIDS" ]]; then
    printf "  %-8s %-28s %6s %6s %8s\n" "PID" "SCRIPT" "%CPU" "%MEM" "RSS"
    printf "  %-8s %-28s %6s %6s %8s\n" "---" "------" "----" "----" "---"
    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        # Extract the .py or .sh script name from full args; fall back to executable basename
        FULL_ARGS=$(ps -p "$pid" -o args= 2>/dev/null | head -1 || true)
        SCRIPT=$(echo "$FULL_ARGS" | grep -oE '[^ /]+\.(py|sh)' | head -1)
        [[ -z "$SCRIPT" ]] && SCRIPT=$(echo "$FULL_ARGS" | awk '{n=$1; sub(".*/","",n); print n}')
        RSS=$(ps -p "$pid" -o rss= 2>/dev/null | tr -d ' ' || echo 0)
        RSS_MB=$(( ${RSS:-0} / 1024 ))
        CPU=$(ps -p "$pid" -o pcpu= 2>/dev/null | tr -d ' ' || echo 0)
        MEM=$(ps -p "$pid" -o pmem= 2>/dev/null | tr -d ' ' || echo 0)
        printf "  %-8s %-28s %5s%% %5s%% %6d MB\n" "$pid" "$SCRIPT" "$CPU" "$MEM" "$RSS_MB"
    done <<< "$PIPELINE_PIDS"
else
    echo "  (no pipeline processes running)"
fi

echo ""
echo "================================================================="
