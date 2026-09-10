#!/bin/bash
# watchdog: relaunch llama-server if it ever dies (driver wedge insurance)
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
SLOTS=${SLOTS:-8}
DEPTH=${DEPTH:-32768}
HEADROOM=${HEADROOM:-1024}
CTX=$(( (DEPTH + HEADROOM) * SLOTS ))

cd ~/llm-serving/llama-b9592
while true; do
  LD_LIBRARY_PATH=. ./llama-server -m ../gguf/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
    --alias bulk-moe-q4 -ngl 99 -fa on -np "$SLOTS" -c "$CTX" -ctk q8_0 -ctv q8_0 \
    --cache-reuse 256 --host 0.0.0.0 --port 8090 >> ~/llm-serving/llama-server.log 2>&1
  echo "$(date -Is) llama-server exited rc=$? -- restarting in 10s" >> ~/llm-serving/llama-server.log
  sleep 10
done
