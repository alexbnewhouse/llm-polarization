# llm-polarization

Code, benchmarks, design records and bibliography for **LLM-to-LLM dialogue as
a laboratory for measuring political polarization in chatbot responses**.

**Research question.** How does the ideological slant of a user affect the
political polarization of LLM responses after long dialogues?

Pairs of local models hold structured conversations in dyads. A **seeker**
plays a human with an assigned political persona and comes for guidance; a
**mentor** is a zero-shot LLM advisor whose movement is the outcome. Seeker
persona attributes (topic, social role, openness) are experimental conditions;
the mentor's pre/post survey batteries are the outcome. Everything runs
locally on one Framework Desktop (AMD Strix Halo, 128 GB unified memory) with
llama.cpp, so every model is hash-pinned and reproducible.

Full draft due 2026-11-30. Funded by IHS. Task tracking lives in Notion
("Polarization Tasks"); this repo is the artifact of record.

## Layout

```
docs/
  design/research-design.md      the design, terminology, runway, descope plan
  design/persona-stability.md    research pass on drift; harness requirements
  decisions/model-arm.md         which four models form the architecture arm
  decisions/compute-budget.md    dialogues x turns x tok/s = wall-clock days
  hardware/                      box tuning notes and verification scripts
models/
  RUN_APPROACH.md                the serving operating point and measured tok/s per arm
  parallel_scaling.csv           the parallel-slot scaling data behind it
  benchmarks/                    scripts + raw results (llama-bench sweeps, parallel scaling)
  serving/                       llama-server watchdog scripts used on the box
harness/                         dyad harness requirements (code not yet written)
prompts/                         seeker persona templates (to be written)
instruments/survey-batteries.md  ideological + affective batteries (from de Jong 2024)
analysis/                        analysis code (to be written)
paper/references.bib             running bibliography
data/                            experiment output, git-ignored
```

## Status (2026-09-08)

- Hardware online, tuned, benchmarked. Operating point: llama.cpp Vulkan,
  8 parallel slots, q8_0 KV, `-c = (depth + 1024) * 8`, KV cache reuse on.
- Model arm decided: qwen3.6:35b-a3b, gpt-oss:20b, glm-4.7-flash,
  Olmo-3-7B-Instruct. Olmo is the only arm with public training data and is
  not to be cut under schedule pressure.
- Compute budget measured for qwen3.6 and Olmo at the operating point; the
  other two arms are still extrapolated. See `models/RUN_APPROACH.md`.
- Pipeline: nothing written yet. Requirements in `harness/README.md`.
- Pilot 2026-09-18, baseline W5, waves W7 to W9, checkpoint 2026-10-19.

## The two machines

| Box | Role |
|---|---|
| Framework Desktop (Fedora, Strix Halo, 128 GB) | Runs every dialogue and every benchmark. `llama-server` from `~/.local/llamacpp/llama-b10488`, GGUFs in `~/llm-serving/gguf/`. |
| Desktop (WSL2, RTX 5080 16 GB) | Development, analysis, writing. Not a target for the model arm. |

## Reproducing a benchmark

```bash
cd models/benchmarks
python3 bench_parallel.py --model ~/llm-serving/gguf/Olmo-3-7B-Instruct-Q4_K_M.gguf \
  --label olmo3-7b-instruct-q4km --plan 32768:1:q8_0,32768:8:q8_0 \
  --out results/olmo3_7b_parallel.jsonl
```

See `models/benchmarks/README.md` for what each script and result file is.
