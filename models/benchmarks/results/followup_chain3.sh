#!/bin/bash
# 2026-09-08 follow-up at the operating point. gpt-oss-20b MXFP4 on the HIP build first (Vulkan has no MXFP4
# path); then GLM-4.7-Flash Q4_K on the Vulkan build once its download lands. ggml-org GGUFs; the ollama blobs
# carry architecture names (glm4moelite, gptoss) that upstream llama.cpp does not know.
cd ~/llm-serving
G=~/llm-serving/gguf/GLM-4.7-Flash-Q4_K.gguf; O=~/llm-serving/gguf/gpt-oss-20b-MXFP4.gguf
VK=~/.local/llamacpp/llama-b10488/llama-server; HIP=~/.local/llamacpp/src/build-hip/bin/llama-server
PLAN=32768:1:q8_0,32768:4:q8_0,32768:8:q8_0
rm -f gptoss20b_parallel.jsonl gptoss20b_parallel.log gptoss20b_server.log glm47flash_parallel.jsonl glm47flash_parallel.log glm47flash_server.log
python3 bench_parallel_cli.py --lcpp $HIP --model $O --label gpt-oss-20b-mxfp4 --plan $PLAN --ignore-eos --extra "--cache-ram 0" --out gptoss20b_parallel.jsonl --server-log gptoss20b_server.log > gptoss20b_parallel.log 2>&1
echo "$(date -Is) gpt-oss done"
while pgrep -f "/hf download" >/dev/null; do sleep 15; done
if [ -f "$G" ]; then python3 bench_parallel_cli.py --lcpp $VK --model $G --label glm-4.7-flash-q4k --plan $PLAN --ignore-eos --extra "--cache-ram 0" --out glm47flash_parallel.jsonl --server-log glm47flash_server.log > glm47flash_parallel.log 2>&1; else echo "MISSING $G"; fi
echo "$(date -Is) FOLLOWUP_CHAIN_DONE"
