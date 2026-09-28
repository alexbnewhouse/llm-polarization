# Dyad harness

`harness/` runs seeker/mentor dialogues against two llama-server endpoints, administers the mentor's
pre/post survey batteries, and scores seeker adherence and mentor stance offline with a judge model.
`harness.randomize` builds the dyad manifest a run reads from the frozen grid and the persona catalogue.
Original design: `docs/superpowers/specs/2026-09-08-dyad-harness-design.md`; this README and
`data/README.md` track the code since then.
Reproducibility standard, and how to reproduce one dialogue from its rows: `docs/REPRODUCIBILITY.md`.

**On reproducibility.** The same seed does not guarantee the same tokens: every generation is sent with
`cache_prompt`, eight dialogues share a server through continuous batching, and llama.cpp does not promise
identical arithmetic when the cache state or the batch composition differs. The harness records enough to
verify exactly what every model was asked, and to detect a lost cache, but not to replay a run bit-exactly
— see `docs/REPRODUCIBILITY.md` section 5 before writing any methods text.

## Use

```bash
pip install -r harness/requirements.txt            # jinja2 + numpy; gguf-py comes from the llama.cpp checkout
cp harness/config.example.json config.json         # edit urls, gguf_py_path, run_seed
python -m harness.randomize --catalogue prompts/personas/catalogue.json --out pilot-dyads.jsonl \
    --seed 20260918 --n-per-cell 5 --modes reinforced,once --prefix p       # grid -> manifest + assignment log
python -m harness.run check  --config config.json --manifest pilot-dyads.jsonl
python -m harness.run run    --config config.json --manifest pilot-dyads.jsonl --run-id pilot-2026-09-18
python -m harness.run score  --config config.json --run-id pilot-2026-09-18 --scope pilot
python -m harness.run flags  --config config.json --run-id pilot-2026-09-18 --threshold 0.55   # after calibration
python -m harness.run score  --config config-judge2.json --run-id wave1 --scope stance --subsample 0.1
python -m harness.run agreement --config config.json --run-id wave1          # cross-judge, on the subsample
python -m harness.run survey --config config.json --run-id pilot-2026-09-18 --phase post   # re-administer
python -m pytest harness/tests -q                  # unit tests; HARNESS_LIVE_URL=... adds the live test
```

Survey replies are schema-constrained to `{"answer": <int>}`; `harness/parser.py` parses them and, when
the schema was not honoured (a truncated reply, a server without grammar support), salvages the number
from free text and labels the row's `answer_method` accordingly. Its 50 hand-written cases are
`harness/tests/parser_cases.jsonl`.

`run` exits 0 when every dyad completed, 2 when any failed, 130 when Ctrl-C stopped it (in-flight dyads
finish, queued ones never start; re-run with the same `--run-id` to resume), and 1 when it refused to start.
A `*.jsonl` whose last line a crash cut off stops `run`, `survey` and `score` with the file and line
named; add `--repair-torn-line` to back the file up and drop that one line, then carry on.
`survey` and `score` exit 2 when any item or judge call failed in that pass; re-run them to fill in what
failed, since neither repeats what is already done. Every subcommand exits 1 with one `error:` line on
stderr for a dead server, a changed model or template, a missing file or GGUF, or a malformed config or
manifest; `check` exits 1 when any row FAILs.

`run`, `survey` and `score` hold an exclusive lock on `data/<run_id>/.lock` while they work, so a second
one on the same `run_id` exits 1 naming the process that holds it. `check` and `run` also read each
server's `/slots` (or `/props` `total_slots` when `/slots` is off) and FAIL `slots` when `concurrency`
exceeds the slot count, when `concurrency` is null and no count is reported, or when one of the slots the
run needs is busy: another client (a second arm on a shared seeker server, a `survey` pass) is using it.
The cache probe is not sent to a busy slot.

| Module | What it holds |
|---|---|
| `run.py` | The CLI: `check`, `run`, `survey`, `score`, `flags`, `agreement`; config loading, pre-flight checks, the worker pool, provenance capture. |
| `randomize.py` | `python -m harness.randomize`: grid + persona catalogue -> dyad manifest and assignment log. |
| `grid.py` | Loads `prompts/grid.json` and checks each manifest row's `condition` against it. |
| `dialogue.py` | One dyad's turn loop, the per-message log row, and the KV-cache reuse audit. |
| `transcript.py` | The canonical transcript and each agent's egocentric view of it. |
| `survey.py` | Pre/post survey administration to the mentor, one branch per item. |
| `parser.py` | Survey reply -> integer on the item's scale, with the method that found it. |
| `scorer.py` | The judge: target selection per scope, the judge prompt, the flag rule, cross-judge agreement. |
| `templates.py` | Reads the chat template out of the GGUF, renders it with jinja2, checks parity with the server. |
| `client.py` | The llama-server HTTP client. No retries. |
| `log.py` | Run paths, JSONL writer, manifest write-once, resume index, seed derivation, hashing. |

### Run a one-dyad pilot end to end

```bash
head -1 harness/dyads.example.jsonl > one-dyad.jsonl        # then edit the persona and set n_turns small
python -m harness.run check --config config.json --manifest one-dyad.jsonl   # must be clean
python -m harness.run run   --config config.json --manifest one-dyad.jsonl --run-id smoke-$(date +%F)
python -m harness.run score --config config.json --run-id smoke-$(date +%F) --scope pilot
ls data/smoke-*/            # manifest.json judge-*.json dyads/status/turns/surveys/scores .jsonl
```

## The dyad manifest you pass to `--manifest`

One JSON object per line, one line per dialogue, written by `harness.randomize` from `prompts/grid.json`
and the persona catalogue (`prompts/README.md`). This is the **input**; the harness copies each row into
`data/<run_id>/dyads.jsonl` as it starts that dyad, so the output directory has its own file of the same
shape. **They are different files**: name the input something like `pilot-dyads.jsonl` so the two are never
confused. A worked example with two rows: `harness/dyads.example.jsonl`.

```json
{"dyad_id": "p01-immig-rural-open-a",
 "condition": {"topic": "immigration_enforcement", "ideology": "lean_right", "openness": "open", "role": "rural_rancher"},
 "persona_text": "You are Dana, a 54-year-old rancher ... Open by asking for guidance about ...",
 "persona_reminder": "Note to self: I am Dana, a rancher; worried but open-minded.",
 "persona_mode": "reinforced", "seed": 4242, "n_turns": 40}
```

| Field | What it does |
|---|---|
| `dyad_id` | Unique in the file: it is the resume key. A duplicate is refused before anything is written. |
| `condition` | The experimental cell: `topic`, `ideology`, `openness`, `role`, always all four (the control row is `ideology: "none"` with `openness` and `role` null). Levels are frozen in `prompts/grid.json` (`docs/decisions/factorial.md`); `harness/tests/test_grid.py` checks the example manifest against it. Copied verbatim into `dyads.jsonl` and read by the scorer for `topic`. |
| `persona_text` | The seeker's system prompt, in full. Copied into `dyads.jsonl`, so the archive is self-contained even if the persona templates change later. It carries the instruction to open the conversation. |
| `persona_reminder` | The compact reminder appended as a trailing system message on every seeker turn in `reinforced` mode. Required and non-empty when the mode is `reinforced`. |
| `persona_mode` | `once` (system prompt only) or `reinforced` (reminder every seeker turn). |
| `seed` | The per-dyad seed. It is a live component of every derived seed for this dyad, so two rows that differ only in `seed` are independent replicates. |
| `n_turns` | Exchanges, not messages: 40 turns is 80 rows in `turns.jsonl`. |

`check --manifest` uses the largest `n_turns` in the file to check that a dialogue fits in one slot's
context before the run starts.

### KV cache reuse is asserted, not only logged (2026-09-14)

A turn at 32k depth should prefill the partner's last line (a few hundred tokens), not the transcript.
Nothing errors when that breaks; the wave just runs thousands of times slower. Two guards:

- `check` sends two one-token probes to each role's slot, the second extending the first, and **FAILs
  `cache_reuse`** when the second prefills more than the new tokens plus a 64-token margin. Catches a
  server without prompt caching, or a template that rewrites the prefix between turns, before the run.
- During a run, a mid-dialogue turn whose `prompt_n` is both unexpected (`cache_warning`) and above
  `cache_reuse_limit` (config, default 1000; `null` disables) **fails the dyad** with `CacheReuseLost`.
  The row is logged first, so the evidence is in `turns.jsonl`; `run` exits 2, and the next `run` with the
  same `--run-id` retries the dyad as a new attempt. A systematic loss fails every dyad at its second turn,
  which is the point.

### Reasoning is kept out of the transcript (2026-09-28)

The harness calls raw `/completion`, so llama-server never separates a model's reasoning from its answer.
After each turn the harness does: `reasoning_content` from the server, every `<think>...</think>` block,
and the text before an unmatched `</think>` go to the row's `reasoning` field, and only what is left is
the line the partner sees and the judge scores. A reply with gpt-oss harmony channel markup
(`<|channel|>`, `<|start|>assistant`, `<|message|>`) or a `<think>` it never closes has no clean answer to
pass on: its row is logged with `error` and the dyad fails (`HarmonyMarkup`, `UnterminatedThink`).

## What the config fields mean

| Field | What it does |
|---|---|
| `run_seed` | The one number every generation seed is derived from, together with the per-dyad `seed`. Change it and you get a different run. Record it in the paper. |
| `now` | The date fed to templates that print the current date (gpt-oss does). Pinned so the prompt is the same tomorrow. Changing it changes every prompt for those models: freeze it for the life of the study, not per run. |
| `concurrency` | How many dialogues run at once, and how many judge slots `score` uses at once. `null` means "as many as the smaller server has slots" (for `score`, the judge's). More than a server's slots is refused: llama.cpp wraps an out-of-range slot id, so two dialogues would share a slot. |
| `cache_reuse_limit` | Tokens. A mid-dialogue turn that prefills more than this when the cache should have held fails the dyad. Default 1000; must exceed `2 * n_predict` plus the reminder. `null` disables. Operational: not compared on resume. |
| `gguf_py_path` | Path to llama.cpp's `gguf-py` directory; the harness reads the chat template out of the GGUF with it. On the Framework Desktop: `/home/alex/.local/llamacpp/src/gguf-py` (this is what `config.example.json` ships with). On the development desktop: `/home/alex/llm-serving/llama.cpp/gguf-py`. |
| `batteries` | The survey items file. Its sha256 and item ids go into `manifest.json`, and the sha256 onto every survey row. |
| `grid` | The frozen factorial (`prompts/grid.json` by default). `check --manifest` and `run` refuse a row whose condition is not a cell of it. `null` disables the gate, for smoke tests only. |
| `generation` | `temperature`, `top_p`, `n_predict`, `timeout`, `enable_thinking` for dialogue turns. Surveys and the judge use their own fixed settings (temperature 0; `n_predict` 32 and 160), which are written onto the rows and into `judge-*.json`. |
| `data_dir` | Where `data/<run_id>/` is created. |
| `seeker`, `mentor` | `{url, gguf_path?, family?}`. Must be two different servers: one server would make the two agents evict each other's KV cache every turn, and `run` refuses it. Two URLs count as one server when they resolve to the same address, port and path (every loopback name is one address; a trailing slash is ignored), or when the servers report the same model file, model hash, build, slot count and per-slot context. |
| `judge` | Only needed by `score`. Must be a third model: `score` refuses if the judge hash equals the seeker's or the mentor's, or if the judge is from the mentor's model family. |
| `family` (in `seeker`, `mentor`, `judge`) | The model family slug (`qwen`, `gpt-oss`, `olmo`, `glm`, `llama`, `gemma`, `mistral`, `deepseek`, `phi`), when the GGUF name, its directory and the server alias do not show it: an ollama blob served without `--alias` does not. `score` refuses when the judge's or the mentor's family is unknown, so set it for those; `check` warns. The mentor's may also be set in the `score` config when `manifest.json` has none. |

`concurrency`, `data_dir`, `gguf_py_path`, `cache_reuse_limit` and `grid` are operational: changing them
and resuming the same `run_id` is allowed. What a `run` resume compares with `manifest.json` —
`RESUME_COMPARES` in `harness/run.py`, recorded in the manifest as `resume_compares` — is:

- the run-affecting config, `RUN_AFFECTING_CONFIG` in `harness/log.py`: `seeker`, `mentor`, `judge`,
  `generation` (the whole block, so `generation.timeout` is compared too, even though it changes no
  prompt), `run_seed`, `batteries` (as a path) and `now`;
- the `batteries` file's sha256, so an instrument edited in place is refused;
- every input dyad row, per `dyad_id`, against the copy of the input manifest the run started with
  (`data/<run_id>/input-dyads.jsonl`): `condition`, `persona_text`, `persona_reminder`, `persona_mode`,
  `seed`, `n_turns`. A dyad added to the input manifest is refused; a dyad dropped from it is not (a
  descope is a subset);
- each served model's sha256, its template's sha256, and its llama.cpp `build_info`;
- the harness commit, and whether the tree was dirty: a dirty tree resumes only with the same
  uncommitted diff under `harness/` and `instruments/` (`harness_diff_sha256`), and a git state that
  cannot be read refuses.

Any difference refuses with one `error:` line naming everything that changed; use a new `run_id`.

## Retries, attempts and which rows count

A dyad that fails is restarted by the next `run` with the same `--run-id` as a **new attempt**, and
`attempt` is part of the derived seed — so attempt 2 is a fresh draw, not a re-run of attempt 1. A dyad
left `started` (the process died mid-dyad) is restarted the same way. The earlier attempt's rows are kept, never deleted.
**Analysis uses the highest attempt whose status is `complete`** (`status.jsonl`), and must filter the
rest out. Because that rule conditions on failure and failures are not random, report the number of dyads
with `attempt > 1`. See `docs/REPRODUCIBILITY.md` section 5.

## Servers

Servers are started outside the harness with the flags in `models/RUN_APPROACH.md`. **Every server used
with the harness runs `--jinja`**, so `/apply-template` uses the model's real chat template and
`check`'s parity test compares like with like.

The one exception is the **Olmo arm**: llama.cpp b10488 rejects Olmo-3's template at startup, so that
server runs `--no-jinja --chat-template chatml` (plus `--cache-ram 0`). `check` then reports
`warn server_chat_template` — the served template is not the GGUF's — and **`template_parity` must still
pass** against it. That parity check is the pre-pilot gate for the Olmo arm: if it fails, the arm's
prompts are not what the harness thinks they are and the pilot does not start.

## Output

Output lands in `data/<run_id>/` as `manifest.json`, `judge-<sha12>.json`, `dyads.jsonl`, `status.jsonl`,
`turns.jsonl`, `surveys.jsonl`, `scores.jsonl` and `flags.jsonl`. Every file and every field:
`data/README.md`.

## Terms (fixed 2026-09-02)

- **Seeker**: the persona-prompted LLM standing in for a human who comes to the
  conversation for guidance.
- **Mentor**: the zero-shot LLM advisor whose responses are the outcome.
- **Dyad**: one seeker paired with one mentor for one dialogue.
- Chat-template roles (system, user, assistant) are a separate axis: each agent
  sees its own lines as assistant-role messages and the other's as user-role
  messages.

## Four things the harness must do

1. **Egocentric context projection** (`luo2026spasm`). Keep one canonical
   transcript. Before each generation, build that agent's message list with
   its own lines as assistant-role messages and the other agent's lines as
   user-role messages. Never pass a shared transcript with speaker labels in
   the text. This is what stops echoing and role confusion in LLM-to-LLM
   dialogue.
2. **A persona-mode switch**, once or reinforced, logged per dialogue. In
   reinforced mode, append the compact reminder from the seeker template as the
   final message before the seeker generates. Put it after the history, not
   before, so the cached prefix (system prompt plus history) stays valid.
3. **Nothing on the mentor side** beyond what the chat template requires. Log
   the exact template string and the model hash. Any system text on the mentor
   is treatment.
4. **Per-turn log row**: dialogue id, turn index, agent (seeker or mentor),
   model hash, persona mode, prompt token count, generation, and an
   `adherence` column that stays null: rows are never rewritten, so the
   scorer writes its scores to `scores.jsonl` instead.

## Serving contract the harness talks to

llama.cpp `llama-server` on the Framework Desktop, one process per model,
`-np 8` slots, q8_0 KV cache, `-c = (depth + 1024) * 8`. Each dialogue keeps
its own slot so every turn only prefills the new tokens. See
`models/RUN_APPROACH.md` for the operating point and the throughput each arm
achieves there. The single largest lever is KV cache reuse: a turn at 32k
depth should prefill tens of tokens, not 32,768. Log `prompt_n` from the
server's `timings` on every call so a lost cache is visible immediately.

## Persona-stability requirements (decided 2026-09-10)

- **Reinforced** delivery: the compact reminder is appended after the history on
  every seeker turn. Never reinforce the mentor.
- **40 turns.** Neither agent is told the turn budget, so turn 20 of a 40-turn
  dialogue is a valid 20-turn observation and no separate 20-turn arm is needed.
- The W4 pilot still runs both delivery modes, 40 turns, five dyads each, but as
  a **measurement rather than a gate**: five dyads cannot support the
  non-inferiority claim that would license dropping the reminder. The pilot
  supplies the drift curve and calibrates the threshold instead.
- **The adherence threshold is calibrated on pilot hand labels, not set at
  0.8.** That number is a rate in `li2024instability` and does not transfer to a
  continuous per-turn score.
- Flag dialogues where seeker adherence falls under threshold for three
  consecutive **scored** seeker turns (`harness.scorer.flag_dialogues`;
  `harness.run flags`, which writes `flags.jsonl` and prints the flagged rate
  by ideology level and by delivery mode). Scored turns, not dialogue turns:
  `main` scope scores the seeker every fourth turn, so the run is over turns
  4, 8, 12. A null score is an unscored turn and neither extends nor breaks
  the run. `--threshold` has no default; it is the calibrated number.
- **Flagged dialogues are kept, not excluded.** ITT over all completed dialogues
  is the primary estimand, adherence enters as a continuous moderator, and
  per-protocol is a labelled sensitivity analysis. The only pre-registered
  exclusion is technical incompleteness: error rows, truncation, judge failure.
- The judge is never the seeker or the mentor of that dialogue, and never the
  mentor's model family. `score` refuses both (`harness.scorer.model_family`
  matches the GGUF name, its directory or the alias), and refuses an unknown
  judge or mentor family unless the config states it as `family`.
- Two judges on the stance metric: `score --scope stance --subsample F` with a
  second judge config scores the mentor's `alignment` on the same
  deterministic subsample of dyads (rows are done per judge), and
  `agreement` reports n, mean absolute difference, Pearson r and the share
  within 0.1 for every judge pair.
- Fix one seeker model across every arm; choose it by measured adherence, not
  size; use a different model family from the mentor. Its throughput sets the
  seeker half of every arm's wave budget (`docs/decisions/factorial.md`).

Decision record: `docs/decisions/persona-stability.md`. Full reasoning and
citations: `docs/design/persona-stability.md`.
