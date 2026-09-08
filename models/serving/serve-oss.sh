#!/bin/bash
cd ~/llm-serving/llama-b9592
export AMD_VULKAN_ICD=RADV
while true; do
  LD_LIBRARY_PATH=. ./llama-server -m /usr/share/ollama/.ollama/models/blobs/sha256-6be6d66a3f546d8c19b130dc41dc24b2fc159f84ffbc76a0ee0676205083cf5a     --alias gpt-oss-120b -ngl 99 -fa on -np 4 -c 32768     --cache-reuse 256 --host 0.0.0.0 --port 8092 >> ~/llm-serving/llama-oss.log 2>&1
  echo "$(date -Is) oss server exited rc=$? -- restart in 10s" >> ~/llm-serving/llama-oss.log
  sleep 10
done
