#!/bin/bash
# See serve-bulk.sh: -c is divided across slots, so it is computed from DEPTH,
# never hardcoded. Override from the environment, e.g. DEPTH=16384 ./serve-4b.sh
#
# This model's KV cache is *larger* than the 30B MoE's, which is the surprise
# worth writing down: Qwen3-4B has 8 KV heads over 36 layers against the MoE's 4
# over 48, so q8_0 costs ~76 KiB/token here versus ~51 there, about 20 GiB at the
# 270336 default. Arithmetic, not measured. Small weights do not mean small KV.
SLOTS=${SLOTS:-8}
DEPTH=${DEPTH:-32768}
HEADROOM=${HEADROOM:-1024}
CTX=$(( (DEPTH + HEADROOM) * SLOTS ))

cd ~/llm-serving/llama-b9592
while true; do
  LD_LIBRARY_PATH=. ./llama-server -m ../gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
    --alias qwen3-4b-q4 -ngl 99 -fa on -np "$SLOTS" -c "$CTX" -ctk q8_0 -ctv q8_0 \
    --cache-reuse 256 --host 0.0.0.0 --port 8091 >> ~/llm-serving/llama-4b.log 2>&1
  echo "$(date -Is) 4b server exited rc=$? -- restart in 10s" >> ~/llm-serving/llama-4b.log
  sleep 10
done
