#!/bin/bash
# Benchmark tokens/sec at 8k and 32k context depth on the Framework Desktop.
# W1 task for the LLM polarization study.
#
# llama-bench -d N prefills N tokens, then measures:
#   pp512 @ dN  = prefill throughput with N tokens already in KV
#   tg128 @ dN  = generation throughput with N tokens already in KV
#
# Model list comes from resolve_models.py (real paths, no hardcoded hashes).
# Results append to bench_results.jsonl so a hang loses only the current model.

set -u
cd /home/alex/llm-serving
LB=/home/alex/.local/llamacpp/llama-b10488/llama-bench
OUT=/home/alex/llm-serving/bench_results.jsonl
LOG=/home/alex/llm-serving/bench_run.log

python3 resolve_models.py > models.list 2>models.err
: > "$OUT"; : > "$LOG"

echo "=== bench start $(date -Is) ===" | tee -a "$LOG"
echo "host: $(hostname)  llama-bench: $(basename $(dirname $LB))" | tee -a "$LOG"
cat models.err | tee -a "$LOG"

N=$(wc -l < models.list); i=0
while IFS='|' read -r LABEL PATH_ TMO; do
  i=$((i+1))
  [ -r "$PATH_" ] || { echo "SKIP $LABEL (unreadable)" | tee -a "$LOG"; continue; }
  SZ=$(stat -c%s "$PATH_")
  echo "--- [$i/$N] $(date -Is)  $LABEL  ($(echo "scale=1;$SZ/1000000000"|bc) GB, timeout ${TMO}s)" | tee -a "$LOG"

  timeout "$TMO" "$LB" -m "$PATH_" \
      -p 512 -n 128 -d 0,8192,32768 \
      -fa on -r 2 -o jsonl 2>>"$LOG" \
    | python3 -c "
import json,sys
lbl=sys.argv[1]; sz=int(sys.argv[2])
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    try: d=json.loads(line)
    except Exception: continue
    d['study_label']=lbl; d['file_bytes']=sz
    print(json.dumps(d))
" "$LABEL" "$SZ" >> "$OUT"

  rc=${PIPESTATUS[0]}
  [ "$rc" = "124" ] && echo "    TIMEOUT after ${TMO}s (partial or no rows)" | tee -a "$LOG"
  echo "    rc=$rc rows_so_far=$(wc -l < $OUT) gtt=$(awk '{printf "%.1fGiB",$1/1073741824}' /sys/class/drm/card1/device/mem_info_gtt_used)" | tee -a "$LOG"
  sleep 5
done < models.list

echo "=== bench done $(date -Is) ===" | tee -a "$LOG"
echo "total rows: $(wc -l < $OUT)" | tee -a "$LOG"
