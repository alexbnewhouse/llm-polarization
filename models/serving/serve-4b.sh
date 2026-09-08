#!/bin/bash
cd ~/llm-serving/llama-b9592
while true; do
  LD_LIBRARY_PATH=. ./llama-server -m ../gguf/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
    --alias qwen3-4b-q4 -ngl 99 -fa on -np 8 -c 49152 -ctk q8_0 -ctv q8_0 \
    --cache-reuse 256 --host 0.0.0.0 --port 8091 >> ~/llm-serving/llama-4b.log 2>&1
  echo "$(date -Is) 4b server exited rc=$? -- restart in 10s" >> ~/llm-serving/llama-4b.log
  sleep 10
done
