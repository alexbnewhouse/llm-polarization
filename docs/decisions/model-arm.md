# Decision: which architectures form the model arm

Status: **decided** (Notion task closed 2026-09-08, originally 2026-08-28; reopened 2026-08-25 after the 8k/32k benchmark).
Source: Notion task "DECIDE: which architectures form the model arm (what 128GB can actually run)".

## What the benchmark changed

The W1 context-depth sweep (`models/benchmarks/bench-ctx.sh`, results in
`models/benchmarks/results/bench_results.jsonl`) invalidated two of the four
originally proposed arms.

| Arm | Verdict | Evidence |
|---|---|---|
| gemma4:31b | **Out** | 6.8 tok/s at 32k depth, roughly 38 days per arm |
| gpt-oss:120b | **Out** | No working path: Vulkan cannot load MXFP4 (`fp4: 0`); ollama times out on the 65 GB load |
| qwen3.6:35b-a3b | **Keep** | 59.7 tok/s single-stream at 32k, 77% retained from 0-depth. Best measured |
| Olmo-3.1-32B | **Downgrade** | Dense 32B is 24 to 38 days per arm. Use **Olmo-3-7B-Instruct** instead |

## The arm

| Arm | Lab | Days per arm (20 turns) | Why it is in |
|---|---|---|---|
| qwen3.6:35b-a3b | Alibaba | 2.15 (measured at the operating point) | Fastest, best context retention |
| gpt-oss:20b | OpenAI | ~2.6 (extrapolated) | Replaces the 120b, same alignment lineage. Needs ROCm, not Vulkan |
| glm-4.7-flash | Zhipu | 8.6 (measured 2026-09-08) | Replaced gemma4. **Cut 2026-09-10** when 40 turns was chosen: four arms at 40 turns is ~39.5 days against a 21-day window, three is ~22.3 |
| Olmo-3-7B-Instruct | Ai2 | see `models/RUN_APPROACH.md` (re-measured 2026-09-08) | **Only arm with public training data**. Carries the "varies by training dataset" claim |

Days per arm assume 2,700 dialogues x 20 turns x 200 tokens x both dyad roles,
i.e. 21.6M generated tokens per arm, and the parallel operating point described
in `models/RUN_APPROACH.md`.

**Do not cut Olmo under schedule pressure.** Architecture varies freely across
the other three; training data varies only through Olmo. Cutting it reduces
the paper to another architecture comparison. Drop glm-4.7-flash first.

**That cut has now happened** (2026-09-10, `docs/decisions/persona-stability.md`
section 2). Choosing 40-turn dialogues spends the compute that the fourth arm
would have used. The arm is three: qwen3.6:35b-a3b, gpt-oss:20b,
Olmo-3-7B-Instruct. This was decided ahead of the Oct 19 checkpoint rather than
at it, because the frozen factorial inherits the arm count.

**Bonus available in the Olmo family.** Base, SFT and Instruct are the same
weights at different training stages, which isolates what alignment does to
political stance. No other family supports that contrast.

## Closing this decision

Confirming the four arms above unblocked `FREEZE THE FACTORIAL`.
