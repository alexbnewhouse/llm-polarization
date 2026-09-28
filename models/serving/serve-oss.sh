#!/bin/bash
# Watchdog for gpt-oss:120b (ollama blob) on :8092, 4 slots by default; logs to
# ~/llm-serving/llama-oss.log.
#
# Reference only: no working Vulkan path for MXFP4 on this box (see
# models/serving/README.md). Kept in sync with the other wrappers so it is not a
# footgun if anyone resurrects it.
#
# See serve-bulk.sh: -c is divided across slots, so it is computed from DEPTH.
# This one runs f16 KV (no -ctk/-ctv), so its per-token cost is roughly double
# what q8_0 would be at the same shape.
#
# Paths, bind address and port come from the environment, as in serve-bulk.sh. Binds 127.0.0.1
# by default (it used to bind 0.0.0.0); HOST=0.0.0.0 exposes it on the LAN deliberately.
SERVING_DIR=${SERVING_DIR:-$HOME/llm-serving}
LLAMA_BIN=${LLAMA_BIN:-$SERVING_DIR/llama-b9592}   # build directory holding llama-server
OLLAMA_BLOBS=${OLLAMA_BLOBS:-/usr/share/ollama/.ollama/models/blobs}
MODEL=${MODEL:-$OLLAMA_BLOBS/sha256-6be6d66a3f546d8c19b130dc41dc24b2fc159f84ffbc76a0ee0676205083cf5a}
LOG_DIR=${LOG_DIR:-$SERVING_DIR}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8092}
SLOTS=${SLOTS:-4}
DEPTH=${DEPTH:-32768}
HEADROOM=${HEADROOM:-1024}
CTX=$(( (DEPTH + HEADROOM) * SLOTS ))

cd "$LLAMA_BIN" || exit 1
export AMD_VULKAN_ICD=RADV
while true; do
  LD_LIBRARY_PATH=. ./llama-server -m "$MODEL" \
    --alias gpt-oss-120b -ngl 99 -fa on -np "$SLOTS" -c "$CTX" \
    --cache-reuse 256 --host "$HOST" --port "$PORT" >> "$LOG_DIR/llama-oss.log" 2>&1
  rc=$?   # capture first: the $(date) below resets $?, so the old line always logged rc=0
  echo "$(date -Is) oss server exited rc=$rc -- restart in 10s" >> "$LOG_DIR/llama-oss.log"
  sleep 10
done
