#!/bin/bash
# Wait for bench-ollama-ctx.sh to exit, then run the :11435 second pass
# (bench-ollama-11435.sh). Appends to ~/llm-serving/chain.log. Provenance only.
set -u
L=/home/alex/llm-serving/chain.log
while pgrep -f "bench-ollama-ctx.sh" >/dev/null; do sleep 15; done
echo "$(date -Is) first ollama pass done; starting :11435 pass" >> $L
/home/alex/llm-serving/bench-ollama-11435.sh >> $L 2>&1
echo "$(date -Is) ALL SWEEPS COMPLETE" >> $L
