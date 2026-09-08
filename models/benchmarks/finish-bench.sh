#!/bin/bash
# Remaining work, run sequentially in ONE process - no pgrep-based chaining.
set -u
L=/home/alex/llm-serving/chain.log
echo "$(date -Is) FINISH-BENCH START" >> $L
/home/alex/llm-serving/bench-ollama-11435.sh >> $L 2>&1
echo "$(date -Is) pass2 done" >> $L
/home/alex/llm-serving/bench-ollama-calib.sh >> $L 2>&1
echo "$(date -Is) EVERYTHING COMPLETE" >> $L
