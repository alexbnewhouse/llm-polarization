# Run approach: the serving operating point and what each arm gets from it

Measured on the Framework Desktop (Ryzen AI MAX+ 395, Radeon 8060S, 128 GB
unified memory), llama.cpp build b10488 (commit 9d77fa172), Vulkan (RADV).
Raw numbers: `parallel_scaling.csv` and `benchmarks/results/`. Method:
`benchmarks/bench_parallel.py` (N concurrent streams, each holding a 32k-token
KV cache, each generating one 200-token turn; the warm round is the study's
steady state).

## The operating point

One `llama-server` per arm:

```
llama-server -m <gguf> -ngl 999 -fa on \
  -np <slots> -c $(( (32768 + 1024) * slots )) \
  -ctk q8_0 -ctv q8_0 --cache-reuse 256 \
  --host 127.0.0.1 --port <port>
```

- **Depth 32k** per dialogue: about 80 exchanges at 200 tokens, enough for
  40-turn dialogues plus persona reminders.
- **q8_0 KV cache**: faster than f16 at 32k on this box (107 vs 92 tok/s at
  np=4 for qwen3.6) and half the memory.
- **One dialogue per slot** (`id_slot` in every request), so each turn
  prefills only the new tokens. Measured warm prefill per turn: 8 tokens
  (qwen3.6) and 5 tokens (Olmo). If `prompt_n` in the server timings ever
  reads in the thousands, the cache was lost and the run is a thousand times
  more expensive than budgeted.
- **Slots**: 8 for qwen3.6 (peak); 4 or 8 for Olmo (equal aggregate, see
  below). 16 is worse for qwen3.6 and does not fit for Olmo.

## Measured throughput at 32k depth, q8_0 KV

| Model | np | Aggregate tok/s | Per-stream tok/s | Cold prefill tok/s | GTT (with resident tiers) |
|---|---|---|---|---|---|
| qwen3.6:35b-a3b (MoE, GQA) | 1 | 50.9 (f16 KV) | 60.9 | 1030 | 22 GiB |
| qwen3.6:35b-a3b | 4 | 107.2 | 29.1 | 920 | 23 GiB |
| qwen3.6:35b-a3b | **8** | **116.4** | 16.1 | 905 | 25 GiB |
| qwen3.6:35b-a3b | 16 | 94.0 | 6.7 | 924 | 28 GiB |
| Olmo-3-7B-Instruct (dense, MHA) | 1 | 28.2 | 29.7 | 868 | 58 GiB |
| Olmo-3-7B-Instruct | **4** | **41.4** | 11.2 | 858 | 67 GiB |
| Olmo-3-7B-Instruct | 8 | 41.5 | 5.6 | 852 | 80 GiB |
| Olmo-3-7B-Instruct | 16 | crashed | | 803 | 100 GiB |

The qwen3.6 rows are from 2026-08-25 (resident tiers were stopped, so GTT is
the server alone). The Olmo rows are from 2026-09-08 with the four resident
tiers up (about 50 GiB), so subtract 50 GiB for the server's own footprint.

## What the Olmo re-measurement changed

The 2026-08-25 budget extrapolated Olmo from a single-stream `llama-bench`
figure (18.1 tok/s, f16 KV) times the 2.29x parallel factor fitted on
qwen3.6, giving about 41 tok/s. The measurement lands at **41.5 tok/s**, but
for different reasons:

- Single-stream is faster than the `llama-bench` number: 28.2 tok/s in the
  server with q8_0 KV versus 18.1 with f16 KV.
- The parallel factor is smaller: **1.47x** at np=4 or np=8, not 2.29x.
  Olmo-3-7B has 32 KV heads and no grouped-query attention, so attention over
  32k tokens dominates each decode step and adding slots mostly adds work,
  not utilization. Dense full-attention models batch *worse* than the MoE at
  this depth, not better as the old caveat guessed.
- np=4 and np=8 give the same aggregate. np=4 uses 12 GiB less memory and
  turns each dialogue over twice as fast, so prefer **np=4 for the Olmo arm**
  unless slot-count uniformity across arms matters more.

## Days per arm

2,700 dialogues x 20 turns x 200 tokens x both dyad roles = 21.6 M generated
tokens per arm, so days = 250 / (aggregate tok/s).

| Arm | tok/s at the operating point | Days, 20 turns | Days, 40 turns | Basis |
|---|---|---|---|---|
| qwen3.6:35b-a3b | 116.4 | 2.15 | 4.3 | measured 2026-08-25 |
| gpt-oss:20b | ~98 | ~2.6 | ~5.1 | extrapolated (42.9 x 2.29) |
| glm-4.7-flash | ~60 | ~4.5 | ~8.3 | extrapolated (26.4 x 2.29) |
| Olmo-3-7B-Instruct | 41.5 | **6.0** | **12.0** | measured 2026-09-08 |
| **Four arms** | | **~15.3** | **~30** | |

Twenty-one days are allocated to waves W7 to W9. Twenty turns fits with
slack; forty turns for all four arms does not fit in the wave window alone
and needs the W6 machine-only week or the descope list (drop glm-4.7-flash
first, never Olmo).

The two extrapolated arms are both MoE models with grouped-query attention,
so the 2.29x factor is more plausible for them than it was for Olmo, but they
are still unmeasured. Each takes about eight minutes with
`bench_parallel.py --plan 32768:1:q8_0,32768:4:q8_0,32768:8:q8_0`; measure
them before freezing N.

## Olmo-specific serving requirements (found the hard way)

1. **`--no-jinja --chat-template chatml`**. llama.cpp b10488 parses the GGUF's
   chat template at startup and rejects Olmo-3's (`tojson` filter on `tools`),
   so the server exits before loading. Every other arm runs `--jinja`, which
   makes `/apply-template` use the model's own template; Olmo is the one
   exception and runs an explicit ChatML template instead.

   The harness renders prompts itself from the GGUF's template, so for this arm
   the string it renders with and the string the server formats with are two
   different objects. `harness check` reports that as
   `warn server_chat_template` (expected) and **`template_parity` must still
   pass** against the Olmo server. That parity check is the pre-pilot gate for
   this arm: run it against the real Olmo server before any wave, because a
   parity failure there invalidates the arm's prompts wholesale and the pilot
   does not start. See `harness/README.md` and `docs/REPRODUCIBILITY.md`.
2. **`--cache-ram 0`**. Olmo-3 uses sliding-window attention on three of every
   four layers. llama.cpp keeps a 4k-window KV cache for those layers (that is
   why 8 x 33k slots fit in 30 GiB), but the server's host-side prompt cache
   cannot serialize that cache: the first slot switch aborts with
   `GGML_ASSERT(tensor->data != NULL && "tensor not allocated")` in
   `server_slot::prompt_save`. Trace in
   `benchmarks/results/olmo3_7b_np8_prompt_cache_crash.log`. Disabling the
   host prompt cache avoids the path entirely; exact-prefix caching inside the
   slot still works (5-token warm prefill).
3. **Pin each dialogue to a slot** with `id_slot`. With the host prompt cache
   off, a slot switch means a full re-prefill instead of a crash, which is
   silent and expensive.
4. **Do not run np=16 with the resident tiers up.** It needs about 100 GiB of
   GTT; on 2026-09-08 it pushed the box past 125 GB and the kernel OOM killer
   took down `llm-fast` and `llm-deep` (systemd restarted them) before killing
   the benchmark server. np=8 (80 GiB total) is the ceiling alongside the
   tiers.
5. **Verify the reinforced-persona reminder does not defeat the cache.** The
   reminder is appended after the history and removed next turn, so the
   cached sequence diverges a few hundred tokens before its end. That is
   inside the 4k sliding window, so llama.cpp should truncate rather than
   re-prefill, but check `prompt_n` in the pilot.

## Rerunning any of this

```bash
cd models/benchmarks
python3 bench_parallel.py --model <gguf> --label <label> \
  --plan 32768:1:q8_0,32768:4:q8_0,32768:8:q8_0 \
  --ignore-eos --extra "--cache-ram 0" \
  --out results/<label>_parallel.jsonl --server-log results/<label>_server.log
```

`--ignore-eos` forces the full 200-token turn so idle slots do not understate
the aggregate (Olmo stops early on the filler prompt; qwen3.6 did not). The
2026-09-08 Olmo runs without it are kept as `olmo3_7b_parallel_run1.jsonl`
and `run2.jsonl` for the record.
