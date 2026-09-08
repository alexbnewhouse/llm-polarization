#!/bin/bash
set -u
L=/home/alex/llm-serving/chain.log
# wait for the 11435 second pass (driven by chain2.sh) to finish
while pgrep -f "chain2\.sh|bench-ollama-11435\.sh" >/dev/null; do sleep 15; done
echo "$(date -Is) starting calibrated pass" >> $L
/home/alex/llm-serving/bench-ollama-calib.sh >> $L 2>&1
echo "$(date -Is) EVERYTHING COMPLETE" >> $L
