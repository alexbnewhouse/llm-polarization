#!/bin/bash
# Reference only: no working Vulkan path for MXFP4 on this box (see
# models/serving/README.md). Kept in sync with the other wrappers so it is not a
# footgun if anyone resurrects it.
#
# See serve-bulk.sh: -c is divided across slots, so it is computed from DEPTH.
# This one runs f16 KV (no -ctk/-ctv), so its per-token cost is roughly double
# what q8_0 would be at the same shape.
SLOTS=${SLOTS:-4}
DEPTH=${DEPTH:-32768}
HEADROOM=${HEADROOM:-1024}
CTX=$(( (DEPTH + HEADROOM) * SLOTS ))

cd ~/llm-serving/llama-b9592
export AMD_VULKAN_ICD=RADV
while true; do
  LD_LIBRARY_PATH=. ./llama-server -m /usr/share/ollama/.ollama/models/blobs/sha256-6be6d66a3f546d8c19b130dc41dc24b2fc159f84ffbc76a0ee0676205083cf5a \
    --alias gpt-oss-120b -ngl 99 -fa on -np "$SLOTS" -c "$CTX" \
    --cache-reuse 256 --host 0.0.0.0 --port 8092 >> ~/llm-serving/llama-oss.log 2>&1
  echo "$(date -Is) oss server exited rc=$? -- restart in 10s" >> ~/llm-serving/llama-oss.log
  sleep 10
done
