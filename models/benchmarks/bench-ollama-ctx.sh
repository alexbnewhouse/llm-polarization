#!/bin/bash
# Supplementary benchmark via the ollama HTTP API (ROCm runner), covering:
#   1. gpt-oss MXFP4 models that the Vulkan llama.cpp build cannot load
#   2. a ROCm-vs-Vulkan cross-check on a model measured both ways
#
# Approximates context depth by sending a synthetic prompt of ~N tokens and
# reading prompt_eval_count / eval_count from the response, so the depth is
# measured rather than assumed.

set -u
OUT=/home/alex/llm-serving/bench_ollama.jsonl
LOG=/home/alex/llm-serving/bench_ollama.log
: > "$OUT"; : > "$LOG"
export OLLAMA_HOST=127.0.0.1:11434

# ~0.75 tokens per word for this filler; overshoot then let the model report actuals.
mkprompt () {  # $1 = target tokens
  python3 -c "
import sys
n=int(sys.argv[1])
s='The committee reviewed the proposal and recorded its objections in detail. '
print((s*((n//10)+40))[:int(n*4.2)])
" "$1"
}

run () {  # model, target_depth
  M=$1; D=$2
  P=$(mkprompt "$D")
  R=$(curl -s --max-time 900 http://127.0.0.1:11434/api/generate \
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
except Exception as e:
    print(json.dumps({'model':m,'target_depth':d,'error':'bad json'})); sys.exit()
if 'error' in x:
    print(json.dumps({'model':m,'target_depth':d,'error':str(x['error'])[:200]})); sys.exit()
pc,pd = x.get('prompt_eval_count'), x.get('prompt_eval_duration')
ec,ed = x.get('eval_count'), x.get('eval_duration')
print(json.dumps({
 'model':m,'target_depth':d,
 'actual_prompt_tokens':pc,
 'prefill_tps': round(pc/(pd/1e9),2) if pc and pd else None,
 'gen_tokens':ec,
 'gen_tps': round(ec/(ed/1e9),2) if ec and ed else None,
 'load_s': round((x.get('load_duration') or 0)/1e9,1),
 'backend':'ollama-rocm'}))
" "$M" "$D" | tee -a "$OUT"
}

# gpt-oss*: MXFP4, unloadable by the Vulkan llama.cpp build (fp4: 0)
# gemma3:27b: failed to load under llama-bench, cause TBD
# gemma4:31b: lives in root-owned /usr/share/ollama, unreadable by llama-bench
# qwen3.6:35b: measured under Vulkan too -> ROCm vs Vulkan cross-check
for M in "gpt-oss:20b" "gemma4:31b" "gemma3:27b" "glm-4.7-flash:q4_K_M" "qwen3.6:35b" "gpt-oss:120b"; do
  echo "=== $M $(date -Is) ===" | tee -a "$LOG"
  for D in 0 8192 32768; do
    echo "  depth ~$D" | tee -a "$LOG"
    run "$M" "$D" >> "$LOG" 2>&1 || echo "  run failed" | tee -a "$LOG"
    sleep 3
  done
done
echo "=== done $(date -Is) ===" | tee -a "$LOG"
