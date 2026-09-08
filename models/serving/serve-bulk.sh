#!/bin/bash
# watchdog: relaunch llama-server if it ever dies (driver wedge insurance)
cd ~/llm-serving/llama-b9592
while true; do
  LD_LIBRARY_PATH=. ./llama-server -m ../gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
    --alias bulk-moe-q4 -ngl 99 -fa on -np 8 -c 32768 -ctk q8_0 -ctv q8_0 \
    --cache-reuse 256 --host 0.0.0.0 --port 8090 >> ~/llm-serving/llama-server.log 2>&1
  echo "$(date -Is) llama-server exited rc=$? -- restarting in 10s" >> ~/llm-serving/llama-server.log
  sleep 10
done
