#!/bin/bash
set -u
L=/home/alex/llm-serving/chain.log
while pgrep -f "bench-ollama-ctx.sh" >/dev/null; do sleep 15; done
echo "$(date -Is) first ollama pass done; starting :11435 pass" >> $L
/home/alex/llm-serving/bench-ollama-11435.sh >> $L 2>&1
echo "$(date -Is) ALL SWEEPS COMPLETE" >> $L
