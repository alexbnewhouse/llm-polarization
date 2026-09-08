#!/bin/bash
# Corrective pass: hit TRUE 8192 / 32768 token depths.
#
# The first two passes used ~4.2 chars/token, which under-shot: a 32768 target
# produced 22040 actual tokens (ratio 0.673). Calibrated to 6.25 chars/token
# and the script now iterates until the measured depth is within 5% of target.

set -u
OUT=/home/alex/llm-serving/bench_ollama_calib.jsonl
LOG=/home/alex/llm-serving/bench_ollama_calib.log
: > "$OUT"; : > "$LOG"

# model:port pairs -- gemma4/gpt-oss:120b live on 11434
TARGETS=("gemma4:31b:11434" "gpt-oss:120b:11434" "qwen3.6:35b:11434")

mkprompt () {  # $1 = target tokens, $2 = chars-per-token
  python3 -c "
import sys
n=int(sys.argv[1]); cpt=float(sys.argv[2])
s='The committee reviewed the proposal and recorded its objections in detail. '
need=int(n*cpt)
print((s*(need//len(s)+2))[:need])
" "$1" "$2"
}

probe () {  # model, port, prompt -> json
  python3 -c "
import json,sys
p=sys.stdin.read()
print(json.dumps({'model':sys.argv[1],'prompt':p+'\n\nSummarize the above in one paragraph.',
 'stream':False,'keep_alive':0,
 'options':{'num_predict':128,'num_ctx':int(sys.argv[2]),'temperature':0.7}}))
" "$1" "$4" <<< "$3" | curl -s --max-time 900 "http://127.0.0.1:$2/api/generate" \
     -H 'Content-Type: application/json' --data-binary @-
}

for entry in "${TARGETS[@]}"; do
  M="${entry%:*}"; PORT="${entry##*:}"
  echo "=== $M on :$PORT $(date -Is) ===" | tee -a "$LOG"
  for D in 8192 32768; do
    CPT=6.25
    for attempt in 1 2 3; do
      P=$(mkprompt "$D" "$CPT")
      R=$(probe "$M" "$PORT" "$P" $((D+2048)))
      ACT=$(echo "$R" | python3 -c "import json,sys
try: print(json.load(sys.stdin).get('prompt_eval_count') or 0)
except Exception: print(0)")
      echo "  target=$D attempt=$attempt cpt=$CPT actual=$ACT" | tee -a "$LOG"
      [ "$ACT" = "0" ] && break
      # within 5%? accept
      python3 -c "import sys; sys.exit(0 if abs($ACT-$D)/$D<=0.05 else 1)" && break
      CPT=$(python3 -c "print(round($CPT*$D/$ACT,3))")
    done
    echo "$R" | python3 -c "
import json,sys
m=sys.argv[1]; d=int(sys.argv[2])
try: x=json.load(sys.stdin)
except Exception: print(json.dumps({'model':m,'target_depth':d,'error':'bad json'})); sys.exit()
if 'error' in x: print(json.dumps({'model':m,'target_depth':d,'error':str(x['error'])[:160]})); sys.exit()
pc,pd=x.get('prompt_eval_count'),x.get('prompt_eval_duration')
ec,ed=x.get('eval_count'),x.get('eval_duration')
print(json.dumps({'model':m,'target_depth':d,'actual_prompt_tokens':pc,
 'prefill_tps':round(pc/(pd/1e9),2) if pc and pd else None,
 'gen_tokens':ec,'gen_tps':round(ec/(ed/1e9),2) if ec and ed else None,
 'backend':'ollama-rocm','calibrated':True}))
" "$M" "$D" | tee -a "$OUT"
    sleep 3
  done
done
echo "CALIB PASS COMPLETE $(date -Is)" | tee -a "$LOG"
