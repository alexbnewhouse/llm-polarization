# Model benchmarks

Everything here ran on the Framework Desktop (AMD Ryzen AI MAX+ 395, Radeon
8060S, 128 GB unified memory, Fedora). llama.cpp build b10488 (commit
9d77fa172), Vulkan backend (RADV). Results are committed so the compute budget
in `models/RUN_APPROACH.md` can be re-derived from raw numbers.

## Scripts

| Script | What it does |
|---|---|
| `bench-ctx.sh` | W1 sweep: `llama-bench` prefill (pp512) and generation (tg128) at 0, 8k and 32k depth for every candidate model. Model list from `resolve_models.py`. |
| `resolve_models.py` | Resolves ollama tags and standalone GGUFs to real blob paths with a per-model timeout. |
| `bench_parallel.py` | Parallel-slot scaling: N concurrent dialogues each holding a depth-D KV cache. Reports cold prefill and warm per-turn generation. **This is what sets the operating point.** Takes `--model`, `--label`, `--plan`, `--out`. |
| `bench-ollama-ctx.sh`, `bench-ollama-11435.sh`, `bench-ollama-calib.sh` | The same depth sweep through ollama (ROCm) rather than raw llama.cpp, plus a calibration pass. |
| `chain-ollama.sh`, `chain2.sh`, `chain3.sh`, `finish-bench.sh` | Sequencing wrappers used to chain the overnight sweeps on 2026-08-24. Kept for provenance; not needed to rerun anything. |

## Results (`results/`)

| File | Run | Content |
|---|---|---|
| `bench_results.jsonl`, `bench_run.log`, `models.list`, `models.err` | 2026-08-24 | `llama-bench` rows at d0/d8k/d32k for 12 candidate models. Killed gemma4:31b (6.8 tok/s at 32k) and gpt-oss:120b (no working path). |
| `bench_ollama*.jsonl`, `bench_ollama*.log`, `chain.log`, `bench_nohup.log` | 2026-08-24 | ollama/ROCm passes of the same sweep. |
| `olmo_bench.jsonl` | 2026-08-25 | `llama-bench` for Olmo-3-7B-Instruct Q4_K_M, f16 KV: 48.8 tok/s at d0, 27.4 at d8k, **18.1 at d32k** single stream. This is the number the first compute budget extrapolated from. |
| `parallel_results.jsonl`, `parallel_bench.log` | 2026-08-25 | `bench_parallel.py` on qwen3.6:35b-a3b: depth x slots x KV type. Peak **116.4 tok/s aggregate at 32k, np=8, q8_0**. Also shows np=16 is slower than np=8. Flattened to `models/parallel_scaling.csv`. |
| `olmo3_7b_parallel.jsonl`, `olmo3_7b_parallel.log` | 2026-09-08 | **The re-measurement at the operating point.** `bench_parallel.py` on Olmo-3-7B-Instruct Q4_K_M at 32k depth, q8_0 KV, np = 1/4/8, `--cache-ram 0`, `--ignore-eos`, slots pinned. 28.2 / 41.4 / 41.5 tok/s aggregate. See `models/RUN_APPROACH.md`. |
| `olmo3_7b_parallel_run1.jsonl`, `_run1.log` | 2026-09-08 | First attempt, default server flags: np=1 and np=4 fine (23.4 / 34.8 tok/s, Olmo hit EOS early so idle slots understate the aggregate); np=8 and np=16 crashed in the warm round. |
| `olmo3_7b_np8_prompt_cache_crash.log` | 2026-09-08 | Server log excerpt of that crash: `GGML_ASSERT(tensor->data != NULL)` in `server_slot::prompt_save`, the host prompt cache trying to serialize the sliding-window KV cache. |
| `olmo3_7b_parallel_run2.jsonl`, `_run2.log` | 2026-09-08 | np=8 and np=16 with `--cache-ram 0`: np=8 works (28.8 tok/s with early EOS); np=16 needs ~100 GiB GTT and was OOM-killed by the kernel along with two resident tiers. |
| `gptoss20b_parallel.jsonl`, `.log` | 2026-09-08 | gpt-oss-20b MXFP4 (ggml-org GGUF) on the **HIP build**, np = 1/4/8: 52.2 / 82.5 / 83.4 tok/s. `llama_build` reads `bin` in these rows because the HIP binary lives in `src/build-hip/bin/`. |
| `glm47flash_parallel.jsonl`, `.log` | 2026-09-08 | GLM-4.7-Flash Q4_K (ggml-org GGUF) on Vulkan, np = 1/4/8: 18.1 / 19.2 / 25.9 tok/s; cold prefill only ~120 tok/s. |
| `glm47flash_hip_parallel.jsonl`, `.log` | 2026-09-08 | The same on the HIP build: 22.6 / 25.6 / 29.1 tok/s; prefill ~250 tok/s. |
| `followup_chain3.sh`, `followup_chain4.sh`, `followup_chain5.sh` | 2026-09-08 | The sequencing scripts that ran the two follow-ups behind the GGUF downloads. Provenance only. |

## Rerunning

```bash
# on the Framework Desktop
python3 bench_parallel.py \
  --model ~/llm-serving/gguf/<model>.gguf --label <label> \
  --plan 32768:1:q8_0,32768:4:q8_0,32768:8:q8_0,32768:16:q8_0 \
  --out results/<label>_parallel.jsonl
```

The script starts its own `llama-server` on port 8199 for each plan entry and
kills it afterwards. It can run while the box's other servers are up as long
as GTT has room (`/sys/class/drm/card1/device/mem_info_gtt_used`); each record
stores the GTT reading at the end of the warm round.

Gotchas found so far:

- The ollama blobs for gpt-oss:20b and glm-4.7-flash carry architecture names
  (`gptoss`, `glm4moelite`) that upstream llama.cpp b10488 does not know, so
  they fail to load on any backend. Use the ggml-org GGUFs in
  `~/llm-serving/gguf/` (`gpt-oss-20b-MXFP4.gguf`, `GLM-4.7-Flash-Q4_K.gguf`).
- gpt-oss (MXFP4) has no Vulkan path on this box; pass
  `--lcpp ~/.local/llamacpp/src/build-hip/bin/llama-server` to use the HIP
  build of the same commit. GLM runs on both and is faster on HIP.

- llama.cpp b10488 parses the chat template at startup and rejects Olmo-3's
  (`tojson` filter). `bench_parallel.py` passes `--no-jinja`; harness code
  that needs the chat template must supply one with `--chat-template`.
- Olmo-3 uses sliding-window attention on three of every four layers.
  llama.cpp keeps a 4k window for those layers and full depth only for the
  others, which is why the 7B fits 16 x 33k slots. The server reports
  `cache_reuse is not supported by this context` for it; exact-prefix prompt
  caching still works (see `warm_prefill_per_turn` in the results).
