#!/bin/bash
# GLM-4.7-Flash Q4_K at the operating point on the Vulkan build, after the download lands. gpt-oss ran under chain3.
cd ~/llm-serving
G=~/llm-serving/gguf/GLM-4.7-Flash-Q4_K.gguf
VK=~/.local/llamacpp/llama-b10488/llama-server
PLAN=32768:1:q8_0,32768:4:q8_0,32768:8:q8_0
while pgrep -f "^python3 bench_parallel_cli" >/dev/null; do sleep 15; done
echo "$(date -Is) gpt-oss bench finished"
while pgrep -f "^python3 dl_glm.py" >/dev/null; do sleep 15; done
echo "$(date -Is) glm download finished: $(tail -1 dl_glm.log)"
if [ -f "$G" ]; then python3 bench_parallel_cli.py --lcpp $VK --model $G --label glm-4.7-flash-q4k --plan $PLAN --ignore-eos --extra "--cache-ram 0" --out glm47flash_parallel.jsonl --server-log glm47flash_server.log > glm47flash_parallel.log 2>&1; else echo "MISSING $G"; fi
echo "$(date -Is) FOLLOWUP_CHAIN_DONE"
