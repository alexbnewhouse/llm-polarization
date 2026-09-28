#!/bin/bash
# Wait for the :11435 pass to exit, then run the calibrated depth pass
# (bench-ollama-calib.sh). Appends to ~/llm-serving/chain.log. Provenance only.
set -u
SERVING_DIR=${SERVING_DIR:-/home/alex/llm-serving}   # scripts and logs; the box's layout by default
L="$SERVING_DIR/chain.log"
# wait for the 11435 second pass (driven by chain2.sh) to finish
while pgrep -f "chain2\.sh|bench-ollama-11435\.sh" >/dev/null; do sleep 15; done
echo "$(date -Is) starting calibrated pass" >> "$L"
"$SERVING_DIR/bench-ollama-calib.sh" >> "$L" 2>&1
echo "$(date -Is) EVERYTHING COMPLETE" >> $L
