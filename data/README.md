# data/

Experiment output lives here on the box that produced it. The row files are **git-ignored** (see
`.gitignore`) because they are large and belong on the NAS; `manifest.json`, `judge-*.json`,
`assignment.json`, `SHA256SUMS` and `archive.json` are **not** ignored, because a clone of this
repository alone must be able to say what was run and where its rows are.

One directory per run: `data/<run_id>/`. A `run_id` is chosen on the command line
(`pilot-2026-09-18`, `baseline`, `wave1` ...; letters, digits, `.`, `_` and `-`, starting with a letter or
digit) and is never reused for a different configuration — the
harness refuses to overwrite a manifest whose run-affecting config differs.

This page is the data dictionary: exactly the files and fields `harness/` writes. The standard they
serve, and how to reproduce one dialogue from them, is `docs/REPRODUCIBILITY.md`.

```
data/<run_id>/
  manifest.json        written once when the run starts; never rewritten
  input-dyads.jsonl    the input dyad manifest, copied verbatim when the run starts; never rewritten
  assignment.json      the randomizer's assignment log for that manifest, copied beside it (if it has one)
  judge-<sha12>.json   written by `score`, one per judge model (and scoring pass)
  dyads.jsonl          one row per dyad attempt: the treatment
  status.jsonl         the run's ledger; resume and analysis both read it
  turns.jsonl          one row per message (2 per turn)
  surveys.jsonl        one row per survey item, per dyad, per phase
  scores.jsonl         one row per (turn, agent, metric, judge), written by `score`
  flags.jsonl          one row per scored, complete dyad, written (replaced) by `flags`
  baseline.jsonl       one row per item per administration, written by `baseline` (a baseline run only)
```

Every row is one line, flushed and fsynced as it is written. A line cut off by a crash stops the next
command with the file and line named; `--repair-torn-line` keeps a copy as `<name>.torn-<time>` and drops
that line.

`SHA256SUMS` and `archive.json` are added by hand when a run is archived (see the end of this page).
`.lock` is held by the `run`, `survey` or `baseline` process working on the run, and `.score.lock` by a
`score` process, so a second one refuses; each holds its process's pid and is harmless when left behind.

## `manifest.json`

`run_id`, `started_at`, `harness_commit`, `harness_dirty` (were tracked files under `harness/` or
`instruments/` modified? null when git could not say, which is not the same as clean),
`harness_diff_sha256` (sha256 of that uncommitted diff, or null when clean), `config` (the whole
config as merged onto the defaults), `input_manifest` `{path, sha256, assignment, parent_sha256}`
(`assignment` is `{path, sha256}` of the randomizer's `<stem>-assignment.json` beside the input
manifest, or null; `parent_sha256` is the full wave manifest's sha256 when the input is a descoped subset
written by `harness.randomize --subset-of`, else null), `batteries` `{path, sha256, n_items, item_ids}`,
`grid` `{path, sha256}` of `prompts/grid.json` (null when the config's `grid` is null), `study`
`{path, sha256}` of the study lock the run was checked against (null without one; `harness/README.md`,
"The study lock"), `resume_compares` (what a resume compares with this file
and refuses on: `harness/README.md`, "What the config fields mean"), `check` (`{seeker: [...], mentor:
[...]}`, each `{name, ok, detail}`: the pre-flight rows as the run's first start saw them, including a
parity passed on the server's date or without a leading BOS), `environment`
`{python, platform, jinja2, harness_version, gguf_py_path, gguf_py_commit, gpu}`, and one block each for
`seeker` and `mentor`:

`url`, `alias`, `model_path`, `model_sha256` (of the GGUF file), `family` (model family slug from the
config's `family`, else from the GGUF name, its directory or the alias, or null when unrecognised; `score`
refuses a judge from the mentor's family, and an unknown one), `family_source` (`config`, `detected` or
null), `template_sha256`, `template_source`
(the chat template in full), `server_chat_template` (what the server reports at `/props`, or null),
`build_info` (the llama.cpp build and commit), `model_ftype`, `total_slots`,
`sampler_defaults` (`default_generation_settings.params`: the server's own sampler defaults, which the
harness overrides on every request with the config's `generation` values, recorded so a restart with other
flags is seen; a resume refuses a change), `n_ctx` (the per-slot context; compared on resume too),
`default_generation_settings` (all of `/props` `default_generation_settings`, as it was).

## `input-dyads.jsonl`

The input dyad manifest the run started with (`--manifest`), byte for byte; its sha256 is
`manifest.json` → `input_manifest.sha256`. A resume compares each dyad row with it and refuses a changed
or added dyad.

## `assignment.json`

The randomizer's log for the input manifest, copied when the run starts: the RNG seed, the arguments, the
counts per cell, variant and mode, and the grid's, catalogue's and manifest's sha256. For a descoped subset
(`harness.randomize --subset-of`) it is the subset's log instead: `kind: "subset"`, `parent` `{path,
sha256, assignment}`, `filter` `{per_variant, control, ideology, topic}`, `rows`, `rows_dropped`,
`rows_per_cell` and `output` `{path, sha256}`.

## `judge-<sha12>.json`

The judge's provenance, written by `score`: the same fields as a role block above, plus `scope`,
`subsample` (the dyad fraction, or null), `temperature`, `n_predict`, `samplers` (every other sampler
the judge requests were sent with), `judge_system` and `judge_tasks` (the judge prompt text verbatim),
`harness_commit` and `ts`. It is a separate file because `manifest.json` is written once at the start of
a run and never rewritten, while scoring happens later and often from a different commit. A later pass
with the same judge whose record differs in anything but `ts` (another scope, subsample or harness
commit) writes `judge-<sha12>-<scope>.json` beside it, and a further differing pass with the same scope
writes `judge-<sha12>-<scope>-2.json`, `-3.json` and so on: a record is never overwritten. `score` prints
the name of the record it used.

## `dyads.jsonl` — one row per dyad attempt

`run_id`, `dyad_id`, `attempt`, `condition` (topic, ideology, openness, role; `prompts/grid.json`), `persona_text` in full,
`persona_reminder`, `persona_mode` (`once` or `reinforced`), `seed` (the per-dyad seed from the input
manifest), `n_turns`, `ts`.

This is where the treatment lives. The persona is copied here, not referenced.

## `turns.jsonl` — one row per message

`run_id`, `dyad_id`, `attempt`, `turn`, `agent` (`seeker` or `mentor`), `model_sha256`, `persona_mode`,
`id_slot` (the server slot, and therefore the KV cache, this generation used), `temperature`, `top_p`,
`n_predict`, `prompt_sha256`, `prompt_chars`, `seed`, `prompt_n` (tokens the server actually prefilled),
`predicted_n`, `expected_new` (tokens it *should* have prefilled if the cache held), `cache_warning`,
`truncated` / `tokens_evaluated` / `tokens_cached` (the server's context accounting; null on builds that
do not report them), `finish_reason` (`stop`, `length` or `error`), `text` (the generation as passed to
the partner and scored by the judge), `reasoning` (what the model reasoned before answering, kept out of
`text`: the server's `reasoning_content`, every `<think>...</think>` block, and the text before an
unmatched `</think>`; null when there was none), `timings`
(the server's own per-request numbers), `adherence` (always null; the scorer writes the equivalent into
`scores.jsonl`), `ts`. On a failure there is also `error`, and the numeric fields are null. When the
server reported no timings there is also `timings_missing: true`, and `prompt_n`, `predicted_n` and
`cache_warning` are null: unknown, not zero. A reply
that carries gpt-oss harmony channel markup (`<|channel|>`, `<|start|>assistant`, `<|message|>`), or opens
a `<think>` block it never closes, is logged with `error` (`HarmonyMarkup: ...` or `UnterminatedThink: ...`)
and its raw `text`, and the dyad fails: the partner never sees it.

Two messages per turn: seeker then mentor. A 40-turn dialogue is 80 rows.

`expected_new` and `cache_warning` are the harness's audit of KV-cache reuse. `cache_warning: true` means
the server re-prefilled far more than it should have — that turn cost hundreds of times more compute than
budgeted, and its numerics went down a different path. Report the rate in the appendix. When the excess is
also above the config's `cache_reuse_limit` on a mid-dialogue turn, the dyad fails right after that row
(`status.jsonl` reason starts with `KV cache reuse lost`) and is retried as a new attempt.
`truncated: true` is different and worse: the slot ran out of context. `finish_reason: "length"` alone
cannot tell the two apart, which is why both are logged.

## `surveys.jsonl` — one row per item per dyad per phase

`run_id`, `dyad_id`, `attempt`, `phase` (`pre` or `post`), `origin` (`run` for the pass the dialogue run
makes, `readministered` for a later `python -m harness.run survey` pass — the two share every other key
field; a later pass never repeats an item already re-administered without an `error` under the same
`schema`, `temperature` and `n_predict`),
`item_id`, `battery`, `scale` `{min, max}`, `batteries_sha256` (which instrument file), `model_sha256`
and `template_sha256` (which mentor answered), `id_slot`, `turn` (a sentinel: 0 for pre, `n_turns + 1`
for post, so the two phases derive different seeds), `temperature`, `top_p`, `n_predict`, `schema`
(false when the item was sent without the JSON schema: `survey --no-schema`), `prompt_sha256`,
`prompt_chars`, `seed`, `answer` (integer, or null if the reply did not parse), `answer_method` (how
`answer` was found: `json` when the schema-constrained reply parsed as an integer on the scale;
`labelled` or `bare` when the number was salvaged from free text by `harness/parser.py`; `ambiguous`,
`out_of_range` or `none` when `answer` is null and why; `truncated` when the `n_predict` cap cut the reply
off before it parsed as JSON, so no number is salvaged from it; null on an error row), `raw_text` (what
the model actually said), `finish_reason` (`stop`, `length` or `error`), `predicted_n`, `truncated` (the
server's context accounting), `prompt_n`, `ts`. On a failure there is also `error`.

Analysis should report the share of rows whose `answer_method` is not `json`: on a server that honours
the grammar it is zero, and a salvaged answer is a number the schema did not produce.

The pre-survey runs before turn 1 in a fresh context, one item at a time with no system prompt. The
post-survey runs after the last turn, branching each item off the mentor's own view of the dialogue.
Item order is file order and is identical pre and post.

Rows written before `top_p` and `schema` existed were schema-constrained at top_p 0.95.

## `baseline.jsonl` — one row per item per administration, written by `baseline`

A baseline run has no dialogue: `manifest.json` has `kind: "baseline"`, the fields above except
`input_manifest`, `grid` and the `seeker` block, and a `baseline` block `{phase, k, temperature, top_p,
n_predict, schema}`; `resume_compares` is what a re-run refuses on. Each row is a `surveys.jsonl` row with
`origin: "baseline"`, `phase: "pre"`, `attempt` 1, `dyad_id` `baseline-<i>` and `administration` i (1 to
`k`). The seed is `derive_seed(run_seed, i, "baseline-<i>", 1, 0, "survey:pre:<item_id>")`, so every
administration of every item has its own. The prompt is the pre-survey's: the item alone, no system
prompt.

## `scores.jsonl` — one row per (turn, agent, metric), written by `score`

`run_id`, `dyad_id`, `attempt`, `turn`, `agent`, `metric` (`prompt_to_line`, `line_to_line` for the
seeker; `alignment` for the mentor), `scope` (`pilot`, `main` or `stance`: the pass that scored it; absent
on rows from before it was recorded), `subsample` (that pass's `--subsample`, or null), `judge_sha256`,
`id_slot`, `harness_commit` (the commit that did the
scoring), `seed`, `judge_prompt_sha256`, `prompt_chars`, `score` (0.0-1.0, or null), `rationale`,
`raw_text`, `ts`. On a failure there is also `error`.

The seeker of a control dyad (`ideology: "none"`) has no persona to adhere to, so the adherence metrics
(`prompt_to_line`, `line_to_line`) are never scored for it (`docs/decisions/factorial.md`); `score` prints
how many control dyads it left out. Its mentor's `alignment` is scored as for any dyad. Scope `pilot`
scores every turn for both agents; scope `main` scores the seeker only, on turns 4, 8, 12, ... plus the
dyad's final turn; scope `stance` scores the mentor's `alignment` on that same cadence, for the two-judge
subsample (`--subsample F` keeps a deterministic fraction of dyads, keyed on the run's `run_seed` from
`manifest.json`, so a second judge's config need not repeat it; the score seeds use it too). `score` works
through the targets in (dyad, attempt, metric, turn) order, one dyad at a time per judge slot,
`concurrency` slots at once, so `id_slot` is the slot that dyad's worker held and file order is not target
order. Scoring is idempotent **per judge and per scope**: a row that already exists for this judge and
scope without an `error` is never scored again; a second judge scores the same targets afresh, and so does
a `main` pass for a turn a `pilot` pass scored, so each scope's rows are a whole set of its own targets. A
row with `score: null` (an unparseable judge reply) counts as done; a row with `error` (the judge server
failed) is retried on the next `score` and leaves the failed row in place.

## `flags.jsonl` — one row per scored, complete dyad, written by `flags`

`run_id`, `dyad_id`, `attempt`, `ideology`, `topic`, `openness`, `role`, `persona_mode` (copied from the
dyad row so the rates can be broken down), `metric`, `scope` (the scope whose score rows were used:
`flags --scope`, default `main`; rows of other scopes, and rows without one, are never mixed in),
`threshold`, `run_length`, `rule` (always
`consecutive scored seeker turns`), `judge_sha256`, `flagged`, `first_flag_turn` (the scored turn that
completed the run, or null), `scored_turns`, `unscored_turns` (null scores), `turns_under`, `min_score`,
`mean_score`, `final_turn`, `harness_commit`, `ts`.

The file is derived from `scores.jsonl` and is **replaced** on every `flags` run, not appended to. Only
the latest complete attempt of each dyad gets a row, never a control dyad (`flags` prints how many it
left out), and only when that attempt has at least one non-error score row for `metric` from the chosen
judge; an unscored dyad has no row. Flagged dialogues are kept in the ITT sample; the
flag is an instrument statistic and the trigger for the per-protocol sensitivity analysis
(`docs/decisions/persona-stability.md` §4).

## `status.jsonl` — the run's ledger

`run_id`, `dyad_id`, `attempt`, `status` (`started`, `complete` or `failed`), `reason` on a failure
(`abandoned` when a third Ctrl-C stopped the dyad mid-dialogue),
`ts`. `run` reads this to resume: within one attempt a later row wins, except that nothing after a
`complete` undoes it. **Analysis uses the highest attempt whose status is `complete`**; rows
from earlier attempts stay in the files and must be filtered out.

## Archiving

Archive completed runs to the NAS **and** Dropbox — two copies, because one of them will fail — and
commit `manifest.json` and `judge-*.json`. The full procedure, including the `SHA256SUMS` step and the
`archive.json` record, is section 3 of `docs/REPRODUCIBILITY.md`.
