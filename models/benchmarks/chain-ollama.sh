#!/bin/bash
# Wait for the llama-bench sweep to exit, then run the ollama/ROCm sweep.
set -u
LOG=/home/alex/llm-serving/chain.log
: > "$LOG"
echo "$(date -Is) waiting for bench-ctx.sh to finish..." >> "$LOG"

# Wait for the main sweep to clear (guard against it never having started).
waited=0
while pgrep -f "bench-ctx\.sh" > /dev/null; do
  sleep 20
  waited=$((waited+20))
  if [ "$waited" -gt 9000 ]; then
    echo "$(date -Is) ABORT: bench-ctx.sh still running after 150min" >> "$LOG"
    exit 1
  fi
done

echo "$(date -Is) main sweep done after ${waited}s wait; settling 30s" >> "$LOG"
sleep 30
echo "$(date -Is) gtt before ollama sweep = $(awk '{printf "%.1fGiB",$1/1073741824}' /sys/class/drm/card1/device/mem_info_gtt_used)" >> "$LOG"
echo "$(date -Is) starting ollama sweep" >> "$LOG"
/home/alex/llm-serving/bench-ollama-ctx.sh >> "$LOG" 2>&1
echo "$(date -Is) OLLAMA SWEEP COMPLETE" >> "$LOG"
