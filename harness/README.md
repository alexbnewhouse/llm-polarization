# Dyad harness

`harness/` runs seeker/mentor dialogues against two llama-server endpoints, administers the mentor's
pre/post survey batteries, and scores seeker adherence offline. Design: `docs/superpowers/specs/2026-09-08-dyad-harness-design.md`.
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
python -m harness.run check  --config config.json --manifest pilot-dyads.jsonl
python -m harness.run run    --config config.json --manifest pilot-dyads.jsonl --run-id pilot-2026-09-18
python -m harness.run score  --config config.json --run-id pilot-2026-09-18 --scope pilot
python -m harness.run survey --config config.json --run-id pilot-2026-09-18 --phase post   # re-administer
python -m pytest harness/tests -q                  # unit tests; HARNESS_LIVE_URL=... adds the live test
```

`run` exits 0 when every dyad completed, 2 when any failed, 130 when Ctrl-C stopped it (in-flight dyads
finish, queued ones never start; re-run with the same `--run-id` to resume), and 1 when it refused to start.

### Run a one-dyad pilot end to end

```bash
head -1 harness/dyads.example.jsonl > one-dyad.jsonl        # then edit the persona and set n_turns small
python -m harness.run check --config config.json --manifest one-dyad.jsonl   # must be clean
python -m harness.run run   --config config.json --manifest one-dyad.jsonl --run-id smoke-$(date +%F)
python -m harness.run score --config config.json --run-id smoke-$(date +%F) --scope pilot
ls data/smoke-*/            # manifest.json judge-*.json dyads/status/turns/surveys/scores .jsonl
```

## The dyad manifest you pass to `--manifest`

One JSON object per line, one line per dialogue. This is the **input**; the harness copies each row into
`data/<run_id>/dyads.jsonl` as it starts that dyad, so the output directory has its own file of the same
shape. **They are different files**: name the input something like `pilot-dyads.jsonl` so the two are never
confused. A worked example with two rows: `harness/dyads.example.jsonl`.

```json
{"dyad_id": "p01-immig-rural-open-a",
 "condition": {"topic": "immigration_enforcement", "role": "rural_rancher", "openness": "open"},
 "persona_text": "You are Dana, a 54-year-old rancher ... Open by asking for guidance about ...",
 "persona_reminder": "Note to self: I am Dana, a rancher; worried but open-minded.",
 "persona_mode": "reinforced", "seed": 4242, "n_turns": 40}
```

| Field | What it does |
|---|---|
| `dyad_id` | Unique in the file: it is the resume key. A duplicate is refused before anything is written. |
| `condition` | The experimental cell. Copied verbatim into `dyads.jsonl` and read by the scorer for `topic`. |
| `persona_text` | The seeker's system prompt, in full. Copied into `dyads.jsonl`, so the archive is self-contained even if the persona templates change later. It carries the instruction to open the conversation. |
| `persona_reminder` | The compact reminder appended as a trailing system message on every seeker turn in `reinforced` mode. Required and non-empty when the mode is `reinforced`. |
| `persona_mode` | `once` (system prompt only) or `reinforced` (reminder every seeker turn). |
| `seed` | The per-dyad seed. It is a live component of every derived seed for this dyad, so two rows that differ only in `seed` are independent replicates. |
| `n_turns` | Exchanges, not messages: 40 turns is 80 rows in `turns.jsonl`. |

`check --manifest` uses the largest `n_turns` in the file to check that a dialogue fits in one slot's
context before the run starts.

## What the config fields mean

| Field | What it does |
|---|---|
| `run_seed` | The one number every generation seed is derived from, together with the per-dyad `seed`. Change it and you get a different run. Record it in the paper. |
| `now` | The date fed to templates that print the current date (gpt-oss does). Pinned so the prompt is the same tomorrow. Changing it changes every prompt for those models: freeze it for the life of the study, not per run. |
| `concurrency` | How many dialogues run at once. `null` means "as many as the smaller server has slots". |
| `gguf_py_path` | Path to llama.cpp's `gguf-py` directory; the harness reads the chat template out of the GGUF with it. On the Framework Desktop: `/home/alex/.local/llamacpp/src/gguf-py` (this is what `config.example.json` ships with). On the development desktop: `/home/alex/llm-serving/llama.cpp/gguf-py`. |
| `batteries` | The survey items file. Its sha256 and item ids go into `manifest.json`, and the sha256 onto every survey row. |
| `generation` | `temperature`, `top_p`, `n_predict`, `timeout`, `enable_thinking` for dialogue turns. Surveys and the judge use their own fixed settings (temperature 0; `n_predict` 32 and 160), which are written onto the rows and into `judge-*.json`. |
| `data_dir` | Where `data/<run_id>/` is created. |
| `seeker`, `mentor` | `{url, gguf_path?}`. Must be two different servers: one server would make the two agents evict each other's KV cache every turn, and `run` refuses it. |
| `judge` | Only needed by `score`. Must be a third model: `score` refuses if the judge hash equals the seeker's or the mentor's. |

`concurrency`, `data_dir` and `gguf_py_path` are operational: changing them and resuming the same
`run_id` is allowed. What a resume actually compares — `RUN_AFFECTING_CONFIG` in `harness/log.py` — is
exactly `seeker`, `mentor`, `judge`, `generation` (the whole block, so `generation.timeout` is compared
too, even though it changes no prompt), `run_seed`, `batteries` and `now`; changing any of those means a
new `run_id`.

## Retries, attempts and which rows count

A dyad that fails is restarted as a **new attempt**, and `attempt` is part of the derived seed — so
attempt 2 is a fresh draw, not a re-run of attempt 1. The earlier attempt's rows are kept, never deleted.
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

Output lands in `data/<run_id>/` as `manifest.json`, `judge-<sha>.json`, `dyads.jsonl`, `status.jsonl`,
`turns.jsonl`, `surveys.jsonl`, `scores.jsonl`. Every file and every field: `data/README.md`.

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
   model hash, persona mode, prompt token count, generation, and an empty
   adherence column the scorer fills in later.

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
  consecutive turns. **Not yet built** — this rule is prose, not code, and the
  pilot needs it.
- **Flagged dialogues are kept, not excluded.** ITT over all completed dialogues
  is the primary estimand, adherence enters as a continuous moderator, and
  per-protocol is a labelled sensitivity analysis. The only pre-registered
  exclusion is technical incompleteness: error rows, truncation, judge failure.
- The judge is never the seeker or the mentor of that dialogue, and never the
  mentor's model family.
- Fix one seeker model across every arm; choose it by measured adherence, not
  size; use a different model family from the mentor.

Decision record: `docs/decisions/persona-stability.md`. Full reasoning and
citations: `docs/design/persona-stability.md`.
