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

Everything that runs on the box is in `harness/`; `analysis/` turns a run directory into the paper's
tables. Start with `harness/README.md`: install, config, the one-dyad pilot, and what each subcommand
writes. `data/README.md` is the data dictionary for the output, `docs/REPRODUCIBILITY.md` is the standard
every run is held to, and `docs/pap/pre-analysis-plan.md` is the analysis the runs feed.

```bash
pip install -e ".[dev]"                    # jinja2, numpy, pyyaml and pytest, from pyproject.toml
# or the pinned versions CI installs: pip install -r harness/requirements.lock pytest
python -m pytest harness/tests -q          # no server needed; the live tests skip without their variables
```

Both run from the repository root; `pip install -e .` only puts the dependencies in place, and the
harness and `analysis/` are always run as `python -m harness.run ...` and `python -m analysis.<module>`.

## Layout

```
docs/
  REPRODUCIBILITY.md             the standard every run is held to, and how to reproduce one dialogue
  pap/pre-analysis-plan.md       the pre-analysis plan: hypotheses, outcomes, models, exclusions,
                                 descope and judge rules (draft for registration, not yet registered)
  pap/README.md                  where and when it is registered, and the power / MDE tables
  audit/                         the 2026-09-28 gap audit, red-team and parallelism reviews, the
                                 remediation plan, and the closure table for every finding
  design/research-design.md      the design, terminology, runway, descope plan
  design/persona-stability.md    research pass on drift; harness requirements
  decisions/model-arm.md         which models form the architecture arm (three since 2026-09-10)
  decisions/compute-budget.md    dialogues x turns x tok/s = wall-clock days, and the 2026-09-28
                                 re-estimate with the seeker half and judge time
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
  serving/                       llama-server watchdog scripts used on the box; serve-study.sh
                                 starts the study's seeker, mentor and judge servers
harness/                         the dyad harness: randomizer, study lock, dialogues, surveys, the
                                 no-dialogue baseline, scoring, flags (see harness/README.md)
prompts/
  grid.json                      the frozen factorial, machine-readable
  personas/catalogue.json        the seeker persona catalogue 1.0.0: 15 backstories, 10 anchor sets
  personas/catalogue.example.json  the catalogue format with placeholder text; the tests use it
instruments/
  batteries.json                 the 15 survey items the harness administers, 1.0.0, US-adapted,
                                 with item directions and the index definitions
  survey-batteries.md            the items, their source (de Jong 2024) and the indices
analysis/                        loader, outcomes, estimators, appendix rates, judge calibration, the
                                 reproduce-one-dialogue check, power.py (see analysis/README.md)
paper/references.bib             running bibliography
data/                            experiment output, git-ignored except each run's manifest.json,
                                 judge-*.json, assignment.json, SHA256SUMS and archive.json
                                 (see data/README.md for every file and field)
pyproject.toml                   dependencies, pytest and ruff settings; Python 3.9 or later
.python-version                  3.11, the interpreter on the box
.github/workflows/tests.yml      CI: the unit tests on Python 3.9 and 3.12, ruff as advisory
LICENSE, CITATION.cff            MIT; cite the repository as in CITATION.cff
```

## Status (2026-09-28)

What exists:

- Hardware online, tuned, benchmarked. Operating point: llama.cpp Vulkan (HIP for gpt-oss),
  8 parallel slots, q8_0 KV, `-c = (depth + 1024) * 8`, KV cache reuse on (`models/RUN_APPROACH.md`).
- Model arm decided: qwen3.6:35b-a3b, gpt-oss:20b, Olmo-3-7B-Instruct.
  glm-4.7-flash was cut on 2026-09-10 to pay for 40-turn dialogues
  (`docs/decisions/persona-stability.md`). Olmo is the only arm with public
  training data and is not to be cut under schedule pressure.
- Factorial frozen 2026-09-11: 20 treated cells plus 2 control cells, 2,970
  dialogues per arm, 8,910 in all (`docs/decisions/factorial.md`, `prompts/grid.json`).
- Stimuli and instrument: the persona catalogue 1.0.0 (`prompts/personas/catalogue.json`) and the
  US-adapted batteries 1.0.0 with the two topic items, item directions and indices
  (`instruments/batteries.json`).
- Pipeline: `harness/` randomizes conditions, locks the settings the three arms share (`study`), runs
  dialogues and surveys, runs the no-dialogue baseline, scores adherence and stance with a judge on all
  its slots, flags low-adherence dialogues and reports cross-judge agreement. `analysis/` applies the
  exclusions and fits the pre-analysis plan's models. Unit-tested in CI; the live tests need a server
  and have no recorded run since the 2026-09-28 changes.
- Pre-analysis plan and power: `docs/pap/pre-analysis-plan.md`, `docs/pap/README.md`.
- Reproducibility standard: `docs/REPRODUCIBILITY.md`. Same seed does not mean same tokens under
  prompt caching and continuous batching; what is and is not reproducible is written down there,
  and the appendix checklist lives at the end of it.
- The 2026-09-28 audits and what closed each finding: `docs/audit/2026-09-28-closure.md`.

Still to do or decide before wave 1:

- **The pilot has not run.** It was planned for 2026-09-18; no pilot data exists and no manifest is
  committed. The command is `--n-per-cell 6`, 252 dialogues per candidate seeker (`harness/README.md`).
- **The seeker and the judge are not chosen.** Both come from the pilot; the example config's judge on
  `:8099` is a qwen model, which the plan's judge rule excludes.
- **The pre-analysis plan is written, not registered.** Target 2026-10-02, once its [pilot] values are
  filled in (`docs/pap/README.md`). The baseline question (pre-measure greedy or sampled, PAP section
  10) is the PI's.
- **The budget is an estimate.** At the 300-token cap, with the seeker half and judge time, the three
  arms need about 22 to 63 days of machine time depending on turn length and the seeker's speed, against
  21 allocated (`docs/decisions/compute-budget.md`, 2026-09-28). The pilot measures both.
- **`check` has not passed against any study server on record**, including the Olmo parity gate and the
  gpt-oss arm, whose raw replies may carry harmony channel markup that fails the dyad
  (`harness/README.md`). Olmo at np=4 or np=8 is open (`models/RUN_APPROACH.md`).

Runway: the pilot was due 2026-09-18 and the baseline in W5, and neither ran; waves W7 to W9 start
2026-10-05; checkpoint 2026-10-19.

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
