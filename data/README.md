# data/

Experiment output lives here on the box that produced it. The row files are **git-ignored** (see
`.gitignore`) because they are large and belong on the NAS; `manifest.json` and `judge-*.json` are
**not** ignored, because a clone of this repository alone must be able to say what was run.

One directory per run: `data/<run_id>/`. A `run_id` is chosen on the command line
(`pilot-2026-09-18`, `baseline`, `wave1` ...) and is never reused for a different configuration — the
harness refuses to overwrite a manifest whose run-affecting config differs.

This page is the data dictionary: exactly the files and fields `harness/` writes. The standard they
serve, and how to reproduce one dialogue from them, is `docs/REPRODUCIBILITY.md`.

```
data/<run_id>/
  manifest.json        written once when the run starts; never rewritten
  judge-<sha12>.json   written by `score`, one per judge model (and scoring pass)
  dyads.jsonl          one row per dyad attempt: the treatment
  status.jsonl         the run's ledger; resume and analysis both read it
  turns.jsonl          one row per message (2 per turn)
  surveys.jsonl        one row per survey item, per dyad, per phase
  scores.jsonl         one row per (turn, agent, metric), written by `score`
```

## `manifest.json`

`run_id`, `started_at`, `harness_commit`, `harness_dirty` (was the tree clean?), `config` (the whole
config as merged onto the defaults), `input_manifest` `{path, sha256}`, `batteries`
`{path, sha256, n_items, item_ids}`, `environment`
`{python, platform, jinja2, harness_version, gguf_py_path, gguf_py_commit, gpu}`, and one block each for
`seeker` and `mentor`:

`url`, `alias`, `model_path`, `model_sha256` (of the GGUF file), `template_sha256`, `template_source`
(the chat template in full), `server_chat_template` (what the server reports at `/props`, or null),
`build_info` (the llama.cpp build and commit), `model_ftype`, `total_slots`,
`default_generation_settings` (the server's own sampler defaults — `top_k`, `min_p` and the penalties
that the harness never sets).

## `judge-<sha12>.json`

The judge's provenance, written by `score`: the same fields as a role block above, plus `scope`,
`temperature`, `n_predict`, `judge_system` and `judge_tasks` (the judge prompt text verbatim),
`harness_commit` and `ts`. It is a separate file because `manifest.json` is written once at the start of
a run and never rewritten, while scoring happens later and often from a different commit. A second
scoring pass with a different scope writes `judge-<sha12>-<scope>.json` beside it.

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
do not report them), `finish_reason` (`stop`, `length` or `error`), `text` (the generation), `timings`
(the server's own per-request numbers), `adherence` (always null; the scorer writes the equivalent into
`scores.jsonl`), `ts`. On a failure there is also `error`, and the numeric fields are null.

Two messages per turn: seeker then mentor. A 40-turn dialogue is 80 rows.

`expected_new` and `cache_warning` are the harness's audit of KV-cache reuse. `cache_warning: true` means
the server re-prefilled far more than it should have — that turn cost hundreds of times more compute than
budgeted, and its numerics went down a different path. Report the rate in the appendix.
`truncated: true` is different and worse: the slot ran out of context. `finish_reason: "length"` alone
cannot tell the two apart, which is why both are logged.

## `surveys.jsonl` — one row per item per dyad per phase

`run_id`, `dyad_id`, `attempt`, `phase` (`pre` or `post`), `origin` (`run` for the pass the dialogue run
makes, `readministered` for a later `harness survey` pass — the two share every other key field),
`item_id`, `battery`, `scale` `{min, max}`, `batteries_sha256` (which instrument file), `model_sha256`
and `template_sha256` (which mentor answered), `id_slot`, `turn` (a sentinel: 0 for pre, `n_turns + 1`
for post, so the two phases derive different seeds), `temperature`, `n_predict`, `prompt_sha256`,
`prompt_chars`, `seed`, `answer` (integer, or null if the reply did not parse), `raw_text` (what the
model actually said), `prompt_n`, `ts`. On a failure there is also `error`.

The pre-survey runs before turn 1 in a fresh context, one item at a time with no system prompt. The
post-survey runs after the last turn, branching each item off the mentor's own view of the dialogue.
Item order is file order and is identical pre and post.

## `scores.jsonl` — one row per (turn, agent, metric), written by `score`

`run_id`, `dyad_id`, `attempt`, `turn`, `agent`, `metric` (`prompt_to_line`, `line_to_line` for the
seeker; `alignment` for the mentor), `judge_sha256`, `id_slot`, `harness_commit` (the commit that did the
scoring), `seed`, `judge_prompt_sha256`, `prompt_chars`, `score` (0.0-1.0, or null), `rationale`,
`raw_text`, `ts`. On a failure there is also `error`.

Scope `pilot` scores every turn for both agents; scope `main` scores the seeker only, on turns
4, 8, 12, ... plus the dyad's final turn. Scoring is idempotent: a row that already exists without an
`error` is never scored again. A row with `score: null` (an unparseable judge reply) counts as done; a
row with `error` (the judge server failed) is retried on the next `score` and leaves the failed row in
place.

## `status.jsonl` — the run's ledger

`run_id`, `dyad_id`, `attempt`, `status` (`started`, `complete` or `failed`), `reason` on a failure,
`ts`. `run` reads this to resume. **Analysis uses the highest attempt whose status is `complete`**; rows
from earlier attempts stay in the files and must be filtered out.

## Archiving

Archive completed runs to the NAS **and** Dropbox — two copies, because one of them will fail — and
commit `manifest.json` and `judge-*.json`. The full procedure, including the `SHA256SUMS` step and the
`archive.json` record, is section 3 of `docs/REPRODUCIBILITY.md`.
