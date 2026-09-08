#!/bin/bash
# GLM-4.7-Flash Q4_K again on the HIP build, for backend comparison (Vulkan barely batches it).
cd ~/llm-serving
while pgrep -f "followup_chain4" >/dev/null; do sleep 20; done
HIP=~/.local/llamacpp/src/build-hip/bin/llama-server
G=~/llm-serving/gguf/GLM-4.7-Flash-Q4_K.gguf
python3 bench_parallel_cli.py --lcpp $HIP --model $G --label glm-4.7-flash-q4k-hip --plan 32768:1:q8_0,32768:4:q8_0,32768:8:q8_0 --ignore-eos --extra "--cache-ram 0" --out glm47flash_hip_parallel.jsonl --server-log glm47flash_hip_server.log > glm47flash_hip_parallel.log 2>&1
echo "$(date -Is) CHAIN5_DONE"
