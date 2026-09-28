#!/bin/bash
# watchdog: relaunch llama-server if it ever dies (driver wedge insurance)
# Serves Qwen3-30B-A3B-Instruct-2507 Q4_K_M on :8090 with the old llama-b9592 build;
# logs to ~/llm-serving/llama-server.log.
#
# llama-server divides -c across parallel slots: each slot gets -c / -np. So -c
# is computed here from the depth one dialogue needs, never hardcoded, because a
# hardcoded -c that looks like a per-slot depth is silently 1/SLOTS of one. DEPTH
# is the usable context per dialogue; HEADROOM covers the turn being generated
# plus the reinforced-persona reminder. Matches models/RUN_APPROACH.md. Override
# from the environment, e.g. DEPTH=16384 ./serve-bulk.sh
#
# KV cost is arithmetic, not measured: Qwen3-30B-A3B is 48 layers x 4 KV heads x
# 128 head_dim, so q8_0 costs ~51 KiB/token, ~13 GiB at the 270336 default, on
# top of ~17 GiB of weights. The resident systemd tiers hold ~50 GiB; budget
# against the ~75 GiB that is actually free (see models/serving/README.md).
#
# Paths, bind address and port come from the environment. The defaults are the box's layout, so a
# bare run behaves as before except that it binds 127.0.0.1 (it used to bind 0.0.0.0);
# HOST=0.0.0.0 exposes it on the LAN deliberately.
SERVING_DIR=${SERVING_DIR:-$HOME/llm-serving}
LLAMA_BIN=${LLAMA_BIN:-$SERVING_DIR/llama-b9592}   # build directory holding llama-server
GGUF_DIR=${GGUF_DIR:-$SERVING_DIR/gguf}
MODEL=${MODEL:-$GGUF_DIR/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}
LOG_DIR=${LOG_DIR:-$SERVING_DIR}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8090}
SLOTS=${SLOTS:-8}
DEPTH=${DEPTH:-32768}
HEADROOM=${HEADROOM:-1024}
CTX=$(( (DEPTH + HEADROOM) * SLOTS ))

cd "$LLAMA_BIN" || exit 1
while true; do
  LD_LIBRARY_PATH=. ./llama-server -m "$MODEL" \
    --alias bulk-moe-q4 -ngl 99 -fa on -np "$SLOTS" -c "$CTX" -ctk q8_0 -ctv q8_0 \
    --cache-reuse 256 --host "$HOST" --port "$PORT" >> "$LOG_DIR/llama-server.log" 2>&1
  rc=$?   # capture first: the $(date) below resets $?, so the old line always logged rc=0
  echo "$(date -Is) llama-server exited rc=$rc -- restarting in 10s" >> "$LOG_DIR/llama-server.log"
  sleep 10
done
