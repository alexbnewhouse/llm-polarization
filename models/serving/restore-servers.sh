#!/bin/bash
# Restore the llama-server instances that were stopped on 2026-08-24 to free
# memory for the polarization experiments.
#
# RECONSTRUCTED from `ps` output captured before shutdown. Ports, model blobs,
# aliases and context length (65536, read from /props) are confirmed; any
# additional flags that were past the ps truncation point are NOT recovered.
# Check these against what you actually want before relying on them.
#
# Usage:  ./restore-servers.sh [fim|qwen36moe|qwen38|all]

set -u
LCPP=/home/alex/.local/llamacpp/llama-b10488/llama-server
BLOB=/home/alex/.ollama/models/blobs

start_fim () {   # CONFIRMED COMPLETE - this command line was not truncated
  echo "starting fim on :8097"
  nohup "$LCPP" --fim-qwen-1.5b-default --host 127.0.0.1 --port 8097 \
    --alias fim -ngl 999 -fa on >> /home/alex/llm-serving/llama-fim.log 2>&1 &
}

start_qwen36moe () {   # PARTIALLY RECONSTRUCTED - verify flags
  echo "starting qwen36moe (Qwen3.6-35B-A3B q4_K_M+MTP) on :8098"
  nohup "$LCPP" \
    -m "$BLOB/sha256-d372de8e934898a59e6ccfabc3368474711384d8f1fd4d22d87a3f0a45400cdc" \
    --alias qwen36moe --host 127.0.0.1 --port 8098 \
    -ngl 999 -fa on -c 65536 >> /home/alex/llm-serving/llama-qwen36moe.log 2>&1 &
}

start_qwen38 () {      # PARTIALLY RECONSTRUCTED - verify flags
  echo "starting qwen3.8 (Qwen3.8-27B q4_K_M) on :8099"
  nohup "$LCPP" \
    -m "$BLOB/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d" \
    --alias qwen3.8 --host 127.0.0.1 --port 8099 \
    -ngl 999 -fa on -c 65536 >> /home/alex/llm-serving/llama-qwen38.log 2>&1 &
}

case "${1:-all}" in
  fim)       start_fim ;;
  qwen36moe) start_qwen36moe ;;
  qwen38)    start_qwen38 ;;
  all)       start_fim; start_qwen36moe; start_qwen38 ;;
  *) echo "usage: $0 [fim|qwen36moe|qwen38|all]"; exit 1 ;;
esac

echo
echo "The two watchdog wrappers are separate and were also stopped:"
echo "  nohup ~/llm-serving/serve-4b.sh   >/dev/null 2>&1 &   # :8091 Qwen3-4B"
echo "  nohup ~/llm-serving/serve-bulk.sh >/dev/null 2>&1 &   # :8090 Qwen3-30B-A3B"
echo "  Both compute -c as (DEPTH + HEADROOM) * SLOTS, so each slot gets the full"
echo "  DEPTH (32768 by default). Shrink it with e.g. DEPTH=16384 if memory is tight."
