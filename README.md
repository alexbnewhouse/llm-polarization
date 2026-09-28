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

## Getting started

Everything that runs is in `harness/`. Start with `harness/README.md`: install, config, the one-dyad
pilot, and what each subcommand writes. `data/README.md` is the data dictionary for the output, and
`docs/REPRODUCIBILITY.md` is the standard every run is held to.

```bash
pip install -r harness/requirements.txt
python -m pytest harness/tests -q          # no server needed; the live test skips without HARNESS_LIVE_URL
```

## Layout

```
docs/
  REPRODUCIBILITY.md             the standard every run is held to, and how to reproduce one dialogue
  design/research-design.md      the design, terminology, runway, descope plan
  design/persona-stability.md    research pass on drift; harness requirements
  decisions/model-arm.md         which models form the architecture arm (three since 2026-09-10)
  decisions/compute-budget.md    dialogues x turns x tok/s = wall-clock days
  decisions/persona-stability.md delivery mode, turn count, adherence measurement, keep-under-ITT
  decisions/factorial.md         the frozen condition grid, the pilot, the Oct 19 cut list
  hardware/                      box tuning notes and verification scripts
  superpowers/specs/             the harness design spec (2026-09-08)
  superpowers/plans/             the generated plan the harness was first built from; historical,
                                 the code and harness/README.md have moved on
models/
  RUN_APPROACH.md                the serving operating point and measured tok/s per arm
  parallel_scaling.csv           the parallel-slot scaling data behind it
  benchmarks/                    scripts + raw results (llama-bench sweeps, parallel scaling)
  serving/                       llama-server watchdog scripts used on the box
harness/                         the dyad harness: randomizer, dialogues, surveys, scoring, flags
                                 (see harness/README.md)
prompts/
  grid.json                      the frozen factorial, machine-readable
  personas/                      persona catalogue format (catalogue.example.json); the real
                                 catalogue.json is still to be written
instruments/
  batteries.json                 the survey items the harness administers (placeholder wording)
  survey-batteries.md            ideological + affective batteries (from de Jong 2024)
analysis/                        analysis code (to be written)
paper/references.bib             running bibliography
data/                            experiment output, git-ignored except each run's manifest.json
                                 and judge-*.json (see data/README.md for every file and field)
LICENSE, CITATION.cff            MIT; cite the repository as in CITATION.cff
```

## Status (2026-09-16)

- Hardware online, tuned, benchmarked. Operating point: llama.cpp Vulkan,
  8 parallel slots, q8_0 KV, `-c = (depth + 1024) * 8`, KV cache reuse on.
- Model arm decided: qwen3.6:35b-a3b, gpt-oss:20b, Olmo-3-7B-Instruct.
  glm-4.7-flash was cut on 2026-09-10 to pay for 40-turn dialogues
  (`docs/decisions/persona-stability.md`). Olmo is the only arm with public
  training data and is not to be cut under schedule pressure.
- Compute budget measured for all four candidate arms at the operating point:
  qwen3.6 116.4 tok/s, gpt-oss:20b 83.4, Olmo-3-7B 41.5, glm-4.7-flash 29.1.
  Three arms at 40 turns is about 22.3 days; the frozen grid with its control
  cells is about 24.5 on the same basis, to be recomputed once the pilot has
  chosen the seeker model. See `models/RUN_APPROACH.md`.
- Factorial frozen 2026-09-11: 20 treated cells plus 2 control cells, 2,970
  dialogues per arm (`docs/decisions/factorial.md`, `prompts/grid.json`).
- Pipeline: `harness/` randomizes conditions, runs dialogues and surveys, scores
  adherence and stance with a judge model, flags low-adherence dialogues and
  reports cross-judge agreement (unit-tested; live-tested on the desktop 5080).
  The persona catalogue and the US wording of the batteries are not written
  yet. No pilot data yet.
- Reproducibility standard: `docs/REPRODUCIBILITY.md`. Same seed does not mean same tokens under
  prompt caching and continuous batching; what is and is not reproducible is written down there,
  and the appendix checklist lives at the end of it.
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
  --label olmo3-7b-instruct-q4km --plan 32768:1:q8_0,32768:4:q8_0,32768:8:q8_0 \
  --ignore-eos --extra "--cache-ram 0" \
  --out results/olmo3_7b_parallel_rerun.jsonl
```

`--extra "--cache-ram 0"` is required for Olmo: without it np=8 crashes in the warm round
(`models/RUN_APPROACH.md`, "Olmo-specific serving requirements"). `--out` appends, so write to a new
file rather than onto a committed result. See `models/benchmarks/README.md` for what each script and
result file is.
