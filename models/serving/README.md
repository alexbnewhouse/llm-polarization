# Serving scripts (Framework Desktop)

Watchdog wrappers that keep a `llama-server` alive for a given model. These
are the box's operational scripts as of 2026-09-08, copied from
`~/llm-serving/` on the box; the box copies are the live ones.

| Script | Model | Port | Notes |
|---|---|---|---|
| `serve-bulk.sh` | Qwen3-30B-A3B-Instruct-2507 Q4_K_M | 8090 | `-np 8 -c 32768` gives each slot only **4,096 tokens**. Raise `-c` to `(depth + 1024) * 8` before using it for long dialogues. Uses the older llama-b9592 build. |
| `serve-4b.sh` | Qwen3-4B-Instruct-2507 Q4_K_M | 8091 | `-np 8 -c 49152`, 6k per slot. Older build. |
| `serve-oss.sh` | gpt-oss:120b (ollama blob) | 8092 | No working Vulkan path for MXFP4; kept for reference only. |
| `restore-servers.sh` | fim (Qwen 1.5B), qwen36moe, qwen3.8 | 8097 / 8098 / 8099 | Reconstructed from `ps` after the 2026-08-24 shutdown that freed memory for benchmarking. Flags marked "partially reconstructed" in the script are not verified. |

Current resident servers on the box (not managed by these scripts, started by
hand with llama-b10488): qwen3-4b on 8096, fim on 8097, qwen36moe on 8098,
qwen3.8 on 8099, plus `ollama serve` on 11434. Together they hold about 50 GiB
of GTT, leaving about 75 GiB for experiment servers.

For the study itself, launch one server per arm with the operating-point flags
in `models/RUN_APPROACH.md` rather than reusing these wrappers.
