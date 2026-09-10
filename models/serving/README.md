# Serving scripts (Framework Desktop)

Watchdog wrappers that keep a `llama-server` alive for a given model. These
are the box's operational scripts as of 2026-09-08, copied from
`~/llm-serving/` on the box; the box copies are the live ones.

| Script | Model | Port | Notes |
|---|---|---|---|
| `serve-bulk.sh` | Qwen3-30B-A3B-Instruct-2507 Q4_K_M | 8090 | `-c` is computed as `(DEPTH + HEADROOM) * SLOTS`, so each slot gets the full 32k. ~13 GiB of q8_0 KV at the default on top of ~17 GiB of weights. Uses the older llama-b9592 build. |
| `serve-4b.sh` | Qwen3-4B-Instruct-2507 Q4_K_M | 8091 | Same computed `-c`. Its KV is **larger** than the 30B MoE's (8 KV heads over 36 layers vs 4 over 48): ~76 KiB/token, ~20 GiB at the default. Older build. |
| `serve-oss.sh` | gpt-oss:120b (ollama blob) | 8092 | No working Vulkan path for MXFP4; kept for reference only. Same computed `-c`, but f16 KV (no `-ctk`/`-ctv`). |
| `restore-servers.sh` | fim (Qwen 1.5B), qwen36moe, qwen3.8 | 8097 / 8098 / 8099 | Reconstructed from `ps` after the 2026-08-24 shutdown that freed memory for benchmarking. Flags marked "partially reconstructed" in the script are not verified. |

## `-c` is divided across slots

`llama-server` treats `-c` as the total context shared by `-np` slots, so a
hardcoded `-c 32768 -np 8` gives each dialogue 4,096 tokens, not 32,768. That was
a live bug: a 4,570-token prompt failed against port 8090 during benchmarking on
2026-08-24, and in a dialogue run it would have truncated silently rather than
erroring. All three wrappers now compute it:

```bash
SLOTS=${SLOTS:-8}; DEPTH=${DEPTH:-32768}; HEADROOM=${HEADROOM:-1024}
CTX=$(( (DEPTH + HEADROOM) * SLOTS ))     # 270336 at the defaults
```

`DEPTH` is the usable context per dialogue and `HEADROOM` covers the turn being
generated plus the persona reminder, matching the operating point in
`models/RUN_APPROACH.md`. Override either from the environment
(`DEPTH=16384 ./serve-bulk.sh`) when memory is tight. The KV figures in the table
are arithmetic from each model's layer and KV-head counts, not measurements.

Current resident servers on the box are **systemd user units**, not these
scripts: `llm-cheap.service` (qwen3-4b, 8096), `llm-fim.service` (8097),
`llm-fast.service` (qwen36moe, 8098), `llm-deep.service` (qwen3.8, 8099),
`llm-share.service` (tailnet forwarder), plus `ollama serve` on 11434. They
run llama-b10488 and together hold about 50 GiB of GTT, leaving about 75 GiB
for experiment servers. They restart on their own if killed: on 2026-09-08 a
16-slot Olmo benchmark pushed total memory past 125 GB and the kernel OOM
killer took `llm-fast` and `llm-deep` down; both were back within 30 seconds.
Budget experiment servers against the 75 GiB that is actually free, or stop
the tiers first (`systemctl --user stop llm-fast llm-deep`).

For the study itself, launch one server per arm with the operating-point flags
in `models/RUN_APPROACH.md` rather than reusing these wrappers.
