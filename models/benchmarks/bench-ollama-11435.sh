#!/bin/bash
# Second pass: models that live ONLY in the 11435 daemon's store
# (/home/alex/.ollama), which the first pass missed by hardcoding :11434.
set -u
OUT=/home/alex/llm-serving/bench_ollama.jsonl     # append to the same file
LOG=/home/alex/llm-serving/bench_ollama.log
PORT=11435

mkprompt () {
  python3 -c "
import sys
n=int(sys.argv[1])
s='The committee reviewed the proposal and recorded its objections in detail. '
print((s*((n//10)+40))[:int(n*4.2)])
" "$1"
}

run () {
  M=$1; D=$2
  P=$(mkprompt "$D")
  R=$(curl -s --max-time 900 http://127.0.0.1:$PORT/api/generate \
      -H 'Content-Type: application/json' \
      --data-binary @<(python3 -c "
import json,sys
p=sys.stdin.read()
print(json.dumps({'model':sys.argv[1],'prompt':p+'\n\nSummarize the above in one paragraph.',
 'stream':False,'keep_alive':0,
 'options':{'num_predict':128,'num_ctx':int(sys.argv[2])+2048,'temperature':0.7}}))
" "$M" "$D" <<< "$P"))
  echo "$R" | python3 -c "
import json,sys
m=sys.argv[1]; d=int(sys.argv[2])
try: x=json.load(sys.stdin)
except Exception:
    print(json.dumps({'model':m,'target_depth':d,'error':'bad json','backend':'ollama-rocm:11435'})); sys.exit()
if 'error' in x:
    print(json.dumps({'model':m,'target_depth':d,'error':str(x['error'])[:200],'backend':'ollama-rocm:11435'})); sys.exit()
pc,pd = x.get('prompt_eval_count'), x.get('prompt_eval_duration')
ec,ed = x.get('eval_count'), x.get('eval_duration')
print(json.dumps({
 'model':m,'target_depth':d,'actual_prompt_tokens':pc,
 'prefill_tps': round(pc/(pd/1e9),2) if pc and pd else None,
 'gen_tokens':ec,'gen_tps': round(ec/(ed/1e9),2) if ec and ed else None,
 'load_s': round((x.get('load_duration') or 0)/1e9,1),
 'backend':'ollama-rocm:11435'}))
" "$M" "$D" | tee -a "$OUT"
}

echo "=== SECOND PASS on :$PORT $(date -Is) ===" | tee -a "$LOG"
for M in "gpt-oss:20b" "gemma3:27b" "glm-4.7-flash:q4_K_M"; do
  echo "=== $M (:11435) $(date -Is) ===" | tee -a "$LOG"
  for D in 0 8192 32768; do
    echo "  depth ~$D" | tee -a "$LOG"
    run "$M" "$D" >> "$LOG" 2>&1 || echo "  run failed" | tee -a "$LOG"
    sleep 3
  done
done
echo "SECOND PASS COMPLETE $(date -Is)" | tee -a "$LOG"
