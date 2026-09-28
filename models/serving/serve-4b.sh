#!/bin/bash
# Watchdog: keep llama-server running Qwen3-4B-Instruct-2507 Q4_K_M on :8091, restarting
# it 10 s after any exit. Old llama-b9592 build; logs to ~/llm-serving/llama-4b.log.
#
# See serve-bulk.sh: -c is divided across slots, so it is computed from DEPTH,
# never hardcoded. Override from the environment, e.g. DEPTH=16384 ./serve-4b.sh
#
# This model's KV cache is *larger* than the 30B MoE's, which is the surprise
# worth writing down: Qwen3-4B has 8 KV heads over 36 layers against the MoE's 4
# over 48, so q8_0 costs ~76 KiB/token here versus ~51 there, about 20 GiB at the
# 270336 default. Arithmetic, not measured. Small weights do not mean small KV.
#
# Paths, bind address and port come from the environment, as in serve-bulk.sh. Binds 127.0.0.1
# by default (it used to bind 0.0.0.0); HOST=0.0.0.0 exposes it on the LAN deliberately.
SERVING_DIR=${SERVING_DIR:-$HOME/llm-serving}
LLAMA_BIN=${LLAMA_BIN:-$SERVING_DIR/llama-b9592}   # build directory holding llama-server
GGUF_DIR=${GGUF_DIR:-$SERVING_DIR/gguf}
MODEL=${MODEL:-$GGUF_DIR/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}
LOG_DIR=${LOG_DIR:-$SERVING_DIR}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8091}
SLOTS=${SLOTS:-8}
DEPTH=${DEPTH:-32768}
HEADROOM=${HEADROOM:-1024}
CTX=$(( (DEPTH + HEADROOM) * SLOTS ))

cd "$LLAMA_BIN" || exit 1
while true; do
  LD_LIBRARY_PATH=. ./llama-server -m "$MODEL" \
    --alias qwen3-4b-q4 -ngl 99 -fa on -np "$SLOTS" -c "$CTX" -ctk q8_0 -ctv q8_0 \
    --cache-reuse 256 --host "$HOST" --port "$PORT" >> "$LOG_DIR/llama-4b.log" 2>&1
  rc=$?   # capture first: the $(date) below resets $?, so the old line always logged rc=0
  echo "$(date -Is) 4b server exited rc=$rc -- restart in 10s" >> "$LOG_DIR/llama-4b.log"
  sleep 10
done
