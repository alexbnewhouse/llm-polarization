# Reproducibility

Standard of record, 2026-09-08. Describes the harness as of the fix wave on `feat/dyad-harness`.

This is the standard every run of this study is held to, what each run records and where, how to
reproduce one dialogue from its logged rows, what is genuinely not reproducible and why, and the
checklist to work through before the paper goes out.

Nothing below claims a capability the code does not have. The few things the harness still does not do
are marked **(open)** and are listed with their reasons in "Deferred" at the end.

---

## 1. The standard

A run is reproducible-to-journal-standard when all of the following are true. Check a run against this
list before you archive it. If an item is false, the run is still usable — but the appendix has to say so.

| # | The standard | Where it is satisfied today |
|---|---|---|
| 1 | Every model used is pinned by the SHA-256 of its GGUF file, not by a name or a tag. | `manifest.json` → `seeker.model_sha256`, `mentor.model_sha256`; `judge-<sha>.json` → `model_sha256`; and on every row of `turns.jsonl`, `surveys.jsonl` and `scores.jsonl`. |
| 2 | Every model's chat template is pinned by hash, and the harness's rendering was proved identical to the server's before the run started. | `manifest.json` → `template_sha256`; `harness check` runs the parity test twice (with and without a leading system message) and `run` runs `check` first. |
| 3 | The template *source string* is archived, not only its hash. | `manifest.json` → `seeker.template_source`, `mentor.template_source`; `judge-<sha>.json` → `template_source`. The template the server itself reports is stored beside it as `server_chat_template`. |
| 4 | Every random choice is seeded from one number, and every seed is written down. | `run_seed` in `manifest.json` → `config`, the per-dyad `seed` in `dyads.jsonl`, and the derived per-generation seed on every row of `turns.jsonl`, `surveys.jsonl` and `scores.jsonl`. |
| 5 | The exact prompt sent for every generation is recoverable, or its hash is logged. | `prompt_sha256` on every row of all three files, and recoverable end to end from the archive alone (section 4a). |
| 6 | The sampling parameters for every generation are recorded. | Dialogue turns: `temperature`, `top_p`, `n_predict` on the row itself, and `manifest.json` → `config.generation`. Surveys: `temperature`, `n_predict` on the row. Judge: `judge-<sha>.json`. Server-side defaults the harness never sets (`top_k`, `min_p`, penalties): `manifest.json` → `default_generation_settings`. |
| 7 | The software environment is recorded: Python, jinja2, gguf-py, OS. | `manifest.json` → `environment` (`python`, `platform`, `jinja2`, `harness_version`, `gguf_py_path`, `gguf_py_commit`). `harness/requirements.lock` pins the development environment. |
| 8 | The llama.cpp build and commit are recorded. | `manifest.json` → `build_info`, from the server's `/props`. |
| 9 | The server's flags, backend and GPU are recorded. | `build_info`, `total_slots`, `model_ftype` and `default_generation_settings` from `/props`, and `environment.gpu` (nvidia-smi / rocm-smi / `/sys/class/drm`). The exact `llama-server` command line is **(open)**: it is not in the manifest, and the operating point in `models/RUN_APPROACH.md` is the intended configuration. Say in the appendix that you took the flags from there. |
| 10 | The harness's own git commit is recorded, and the tree was clean. | `manifest.json` → `harness_commit` (read in the harness's own directory) and `harness_dirty`. A run refuses to start when the commit cannot be read, and prints a WARN when the tree is dirty. `scores.jsonl` rows carry the commit that scored them, which is often a later one. |
| 11 | The run manifest is sufficient to re-run any single dialogue. | Yes. See section 4. |
| 12 | The survey items and their version are recorded with the run. | `manifest.json` → `batteries` `{path, sha256, n_items, item_ids}`; `batteries_sha256` on every `surveys.jsonl` row; `version` and `adapted` inside `instruments/batteries.json`, which is committed. The wording is recoverable by checking out the commit whose file hashes to `batteries.sha256`. |
| 13 | The judge model and the judge prompt are recorded. | `judge-<sha>.json`: model path and hash, template hash and source, build info, scope, `temperature`, `n_predict`, and the judge prompt text (`judge_system`, `judge_tasks`) verbatim. Plus `judge_sha256`, `judge_prompt_sha256` and `harness_commit` on every `scores.jsonl` row. |
| 14 | There is a clear statement of what is archived where. | Section 3 below. The archive record itself is written by hand (section 3, step 4). |
| 15 | A stranger can follow a written procedure to reproduce one dialogue from its rows. | Section 4 below. |
| 16 | The known sources of nondeterminism are named and explained. | Section 5 below. |
| 17 | The repository has a licence and a citation file. | `LICENSE` (MIT) and `CITATION.cff` at the repository root. |

---

## 2. What every run records, and where

Output goes to `data/<run_id>/`. `run_id` is chosen on the command line and never reused for a different
configuration — the harness refuses to overwrite a manifest whose run-affecting config differs
(`seeker`, `mentor`, `judge`, `generation`, `run_seed`, `batteries`, `now`). `concurrency` and `data_dir`
are operational and may change on a resume.

The field-by-field data dictionary is `data/README.md`, derived from the code. In outline:

| File | One row per | Carries |
|---|---|---|
| `manifest.json` | (one object, written once) | run identity, harness commit and dirtiness, the whole config, input-manifest hash, instrument hash and item ids, environment, and per role: model path/hash, template hash **and source**, the served template, build info, slots, server defaults |
| `judge-<sha12>.json` | scoring pass | the judge's model, template and prompt text, its sampling settings, the scope, and the commit that scored |
| `dyads.jsonl` | dyad attempt | the treatment: condition, persona text in full, reminder, mode, per-dyad seed, `n_turns` |
| `turns.jsonl` | message | model hash, slot, sampling, prompt hash, seed, the generation, cache accounting, context accounting, timings |
| `surveys.jsonl` | item × dyad × phase | phase, `origin`, item id/battery/scale, instrument hash, mentor model and template hash, slot, seed, answer and raw text |
| `scores.jsonl` | (turn, agent, metric) | judge hash, scoring commit, seed, judge prompt hash, score, rationale, raw text |
| `status.jsonl` | status transition | `started` / `complete` / `failed` + reason. Resume and analysis both read it |

`surveys.jsonl`'s `origin` is `run` for the pass the dialogue run itself makes and `readministered` for a
later `harness survey` pass. A re-administration refuses (`error:`, exit 1) if the live `batteries` file's
sha256 no longer matches `manifest.json` → `batteries.sha256`: a changed instrument is a different
measurement, and mixing its rows under the same battery/item ids as the original pass would be silently
wrong. Use a new `run_id` against the new instrument instead.

If you re-run the same `run_id` with a changed run-affecting config, or against a model or template whose
hash differs from the manifest's, the harness stops with a `ManifestMismatch` rather than writing over or
into the record. Use a new `run_id`.

---

## 3. The archival rule

Three tiers. Every completed run passes through all three.

**Tier 1 — in git, always.** The code (`harness/`), the design and decision records (`docs/`), the survey
instrument (`instruments/`), the persona templates (`prompts/`), the benchmark scripts and their raw
results (`models/benchmarks/`), the bibliography, `LICENSE` and `CITATION.cff`. These are small and they
are the artefact of record. `.gitignore` keeps model weights (`*.gguf`, `*.safetensors`, `*.bin`) and the
bulk experiment output out.

**Tier 2 — in git, and now automatically.** `data/<run_id>/manifest.json` and
`data/<run_id>/judge-*.json` are un-ignored: they are a few kilobytes each and they are the entire
provenance record for a wave. Commit them when the run finishes. A reader who has only the repository
can then say exactly what was run.

**Tier 3 — NAS or Dropbox, never git.** The row files themselves — `dyads.jsonl`, `turns.jsonl`,
`surveys.jsonl`, `scores.jsonl`, `status.jsonl` — and the GGUF files for every model in the arm.

The archival step, in order:

1. Verify the run is finished: every dyad in the input manifest has a `complete` row in `status.jsonl`,
   or a documented reason it does not.
2. `sha256sum data/<run_id>/*.jsonl data/<run_id>/*.json > data/<run_id>/SHA256SUMS`.
3. Copy the whole `data/<run_id>/` directory to the NAS, and to Dropbox as the second copy. Two
   independent copies, because one of them will fail.
4. Write `data/<run_id>/archive.json` by hand recording the two destination paths, the date, and the
   `SHA256SUMS` digest. Nothing automates this step; it is a written procedure, not code.
   ```bash
   cat > data/<run_id>/archive.json <<JSON
   {"run_id": "<run_id>", "archived_at": "$(date -Is)",
    "destinations": ["nas:/volume1/polarization/<run_id>", "dropbox:/polarization/<run_id>"],
    "sha256sums_sha256": "$(sha256sum data/<run_id>/SHA256SUMS | cut -d' ' -f1)"}
   JSON
   ```
5. `git add -f data/<run_id>/manifest.json data/<run_id>/judge-*.json` and commit. (They are un-ignored,
   so plain `git add` works; `-f` is harmless insurance.)
6. Archive the GGUF for every model named in the manifest, once, alongside the first run that used it,
   filed under its SHA-256. A run is not reproducible without the weights, and the weights are the one
   thing that will quietly disappear from the internet.

For the journal deposit (AJPS Dataverse, ICWSM/ACM artefact), tiers 1 and 2 go in whole; tier 3 goes in
as the row files plus `SHA256SUMS`, with the GGUF files referenced by SHA-256 and by their upstream
source rather than uploaded.

---

## 4. Reproducing one dialogue from its rows

This is the procedure a stranger follows. It assumes they have the archive from section 3.

### 4a. Verify the record without running anything

This needs no GPU, no server, and — since the template source is archived — not even the GGUF. It is the
check a reviewer will actually run.

1. Pick a dyad: `grep '"dyad_id": "p01-immig-rural-open-a"' data/<run_id>/status.jsonl` and note the
   highest `attempt` with `"status": "complete"`.
2. Pull its rows:
   ```bash
   jq -c 'select(.dyad_id=="p01-immig-rural-open-a" and .attempt==1)' data/<run_id>/dyads.jsonl
   jq -c 'select(.dyad_id=="p01-immig-rural-open-a" and .attempt==1)' data/<run_id>/turns.jsonl
   ```
   You now have the persona, the condition, the mode, and all 80 messages in order.
3. Confirm the model: every turn row's `model_sha256` must equal the matching role's `model_sha256` in
   `manifest.json`. If you hold the GGUF, `sha256sum <model>.gguf` must equal it too.
4. Confirm the code: `git checkout <manifest.harness_commit>` in this repository. Check
   `manifest.harness_dirty` is `false`; if it is `true`, the commit does not fully describe the code
   that ran and the appendix has to say so. `harness_dirty` covers tracked files under `harness/` and
   `instruments/` only (`git status --porcelain --untracked-files=no -- harness instruments`) — it is not
   affected by run output, including the run's own un-ignored `data/<run_id>/manifest.json`.
5. **Rebuild any prompt and check its hash.** This is the real test. For turn `t`, agent `a`:
   - Take the messages before that generation from `turns.jsonl`, in order (sort by `turn`, seeker
     before mentor within a turn — `harness.transcript.message_order`).
   - Build the egocentric view exactly as `harness/transcript.py:view_for` does: the agent's own lines as
     `assistant`, the partner's as `user`; for the seeker, `persona_text` as a leading `system` message,
     and in `reinforced` mode `persona_reminder` as a trailing `system` message; for the mentor, no
     system message at all.
   - Render it with the archived template and the run's `now` / `enable_thinking`.
   - `sha256` of the result must equal the row's `prompt_sha256`.

   In practice, from the manifest alone:
   ```python
   import json
   from harness.templates import ChatTemplate, render
   from harness.transcript import Transcript, message_order
   from harness.log import sha256_text, derive_seed

   man = json.load(open("data/<run_id>/manifest.json"))
   cfg = man["config"]
   tpl = ChatTemplate.from_source(man["seeker"]["template_source"])
   assert tpl.sha256 == man["seeker"]["template_sha256"]

   t = Transcript(dyad["dyad_id"], dyad["persona_text"],
                  dyad["persona_reminder"] or None, dyad["persona_mode"])
   for r in sorted(rows_before_the_one_you_want, key=message_order):
       t.append(r["turn"], r["agent"], r["text"])
   prompt = render(tpl, t.view_for("seeker"), now=cfg["now"],
                   enable_thinking=cfg["generation"]["enable_thinking"])
   assert sha256_text(prompt) == row["prompt_sha256"]
   assert row["seed"] == derive_seed(cfg["run_seed"], dyad["seed"], dyad["dyad_id"],
                                     row["attempt"], row["turn"], row["agent"])
   ```
   If those assertions hold for every row of a dialogue, the transcript, the persona, the template, the
   projection rules, the seeds and the harness version are all confirmed against each other. That is the
   strongest statement this study can make about a single dialogue, and it can be made from the archive
   alone.

### 4b. Re-generate the dialogue

1. Start the two `llama-server` processes with the flags in `models/RUN_APPROACH.md`, matching
   `manifest.build_info`, `total_slots` and `default_generation_settings`. The exact command line is not
   in the manifest **(open)**; take it from `RUN_APPROACH.md` and say in the appendix that you did.
2. Write a one-line dyad manifest containing the `dyads.jsonl` row's fields:
   `dyad_id, condition, persona_text, persona_reminder, persona_mode, seed, n_turns`.
3. Copy `manifest.config` back into a `config.json` — in particular `run_seed` and `now`, both of which
   change every prompt if they differ.
4. `python -m harness.run check --config config.json --manifest one-dyad.jsonl`. It must pass. If
   template parity fails, the server is not using the template the original run used, and nothing
   downstream is comparable.
5. `python -m harness.run run --config config.json --manifest one-dyad.jsonl --run-id repro-<date>`.
6. Compare. The seeds will match exactly — they are derived from `run_seed`, the dyad's `seed`,
   `dyad_id`, `attempt`, `turn` and `agent`. The prompt hash for turn 1 will match exactly. The
   *generated text* very probably will not match token for token, for the reasons in section 5. Compare
   distributions of survey answers and adherence scores, not strings.

---

## 5. Known sources of nondeterminism

Be direct about this in the paper. Every LLM study run on batched GPU inference has these properties; the
ones that get into trouble are the ones that claimed otherwise. **The same seed does not guarantee the
same tokens.** The harness records enough to verify exactly what each model was asked and to detect a
lost cache; it does not, and cannot, replay a run bit-exactly.

**1. Prompt caching and KV-cache reuse.** Every generation is sent with `cache_prompt: true`, against
servers running `--cache-reuse 256`. The point is compute — a turn at 32k depth prefills tens of tokens
instead of 32,768 — and without it this study does not fit in the budget. The cost is that the arithmetic
for a given token depends on what was already in the slot's cache. llama.cpp does not promise identical
output for an identical seed when the cache state differs. *How it is handled:* the harness computes what
the prefill should have been and logs `expected_new`, `prompt_n` and `cache_warning` on every turn, so a
lost or unexpected cache is visible as data. Report the `cache_warning` rate.

**2. Continuous batching across slots.** Eight dialogues share one `llama-server`. Which requests land in
the same batch depends on timing, and that changes the order of floating-point reductions, which can
change a token. *How it is handled:* it is not eliminated, and it cannot be without giving up eight-way
concurrency and the compute budget with it. It is declared.

**3. Which slot a dialogue gets.** Slots are handed out from a pool as workers free up, so a given dyad
does not get the same slot on a second run, and the slot determines which cache it reuses. *How it is
handled:* `id_slot` is on every `turns.jsonl`, `surveys.jsonl` and `scores.jsonl` row, so this is an
auditable source of variation rather than an invisible one.

**4. GPU kernel nondeterminism.** Vulkan/RADV on the Framework Desktop and CUDA on the desktop 5080 both
use reduction strategies that are not bitwise reproducible across different batch shapes. *How it is
handled:* declared, and `environment.gpu` records which device ran the wave; all production runs happen
on one machine (`models/RUN_APPROACH.md`), so the variation is within-machine only.

**5. A retry is a fresh draw, not a re-run.** A dyad that fails is restarted as a new `attempt`, and the
seed derivation includes `attempt` — so attempt 2 uses different seeds throughout and is a genuinely new
dialogue. Analysis uses the highest `complete` attempt. *How it is handled:* the rule is stated here and
belongs in the pre-analysis plan. Because it conditions on failure and failures are not random, report
the number of dyads with `attempt > 1` and check that it is not correlated with condition.

**6. Thread scheduling inside the harness.** Which Python thread runs when does *not* affect what is
written: each dyad is independent, the JSONL writer is locked and appends whole lines, and every field on
a row is computed from that dyad's own state. The only thing thread timing changes is the *order of lines
within a file* and, indirectly, item 2 above. Sort by `(dyad_id, attempt, turn, agent)` in analysis and
file order never matters.

**7. Ctrl-C, and what an interrupted wave means.** An interrupt lets in-flight dyads finish and never
starts a queued one; `run` then exits 130. The stopped dyads have no `status` row at all and are simply
run by the next `run` with the same `run_id`, as attempt 1. Nothing is half-written, because a dyad
writes its `complete` row only after its post-survey.

**8. Scoring is idempotent, with one asymmetry to declare.** `score` skips any (dyad, attempt, turn,
agent, metric) that already has a row without an `error`. A row whose `score` is `null` — the judge
replied but the reply did not parse — counts as done and is never retried. A row with an `error` — the
judge server failed — is retried on the next `score`, and the failed row stays in the file. Report the
number of null scores; both behaviours are deliberate, and neither rewrites a row.

**9. What is fully deterministic, and should be said so.** The seeds
(`sha256(run_seed|dyad_seed|dyad_id|attempt|turn|agent)`), the prompt strings and therefore
`prompt_sha256`, the survey item order, the choice of which turns get scored, the resume logic, and the
assignment of conditions to dyads (which comes from the input manifest, not from the harness). None of
these depend on timing, hardware or the model. A re-run reproduces all of them exactly, and section 4a
checks it without a GPU.

---

## 6. Before you submit

Work through this when the draft is assembled — W14 in the runway, but start it at the Oct 19 checkpoint
so anything still open is closed before the last wave rather than after.

**The repository**

- [ ] `LICENSE` exists at the root.
- [ ] `CITATION.cff` exists at the root, with the ORCID filled in.
- [ ] `README.md` says which commit produced the results in the paper.
- [ ] Every run's `manifest.json` and `judge-*.json` are committed.
- [ ] `data/README.md` still describes the files the harness actually writes.
- [ ] The final `instruments/batteries.json` carries a bumped `version` and `adapted: true`, and every
      run records the sha256 of the file it used.
- [ ] `python -m pytest harness/tests -q` passes at the submission commit, and the live test passes
      against a real server.

**Every run in the paper**

- [ ] `manifest.harness_commit` is a real commit and `manifest.harness_dirty` is `false` (it covers
      tracked files under `harness/` and `instruments/` only, and is not affected by run output).
- [ ] `manifest.build_info` matches the llama.cpp build named in the methods section.
- [ ] Model SHA-256s in the manifest match the GGUF files in the archive.
- [ ] The `run_seed` is stated in the paper.
- [ ] The `now` value is the same across every run in the study — it is in every prompt for the
      gpt-oss-family arms. Nothing enforces it across runs; check it by hand.
- [ ] `manifest.environment` is present and its `jinja2` version is the one in the methods section.
- [ ] `manifest.environment.gpu` and `build_info` are reported, and the appendix says the server flags
      were the operating point in `models/RUN_APPROACH.md`.
- [ ] `template_source` is present for every role, and `server_chat_template` was checked (for the Olmo
      arm it differs by design; the parity check is what must have passed).
- [ ] `archive.json` records both destinations and `SHA256SUMS` verifies.

**The scoring pass**

- [ ] `judge-*.json` is in the archive for every scoring pass, with the judge prompt text.
- [ ] The judge is confirmed to be a third model, distinct from both seeker and mentor. (The harness
      refuses otherwise, so this is already true if `score` ran.)
- [ ] The judge passed `check` before scoring (`score` refuses otherwise).

**The appendix text**

- [ ] Section 5 of this document, in prose, appears in the methods or the appendix. Do not claim bitwise
      reproducibility.
- [ ] The rate of `cache_warning = true` is reported.
- [ ] The rate of `truncated = true` is reported (a slot that ran out of context, not a capped turn).
- [ ] The number of dyads with `attempt > 1` is reported, with a check against condition.
- [ ] The number of survey items that failed to parse (`answer: null`) is reported.
- [ ] The number of turns with `finish_reason: "length"` is reported — those turns hit the 300-token cap
      and their text is truncated by design.
- [ ] Section 4 of this document — the reproduce-one-dialogue procedure — is included verbatim or
      referenced by URL. It is the part a reviewer will try.

---

## Deferred

Known and deliberately not done in this wave. Each is a judgement about cost, not an oversight.

- **The exact `llama-server` command line is not in the manifest** (review gap B5, in part). `/props`
  gives the build, the slots, the file type and the server's sampler defaults, and `environment.gpu`
  gives the device; the flags that do not surface at `/props` (`-ngl`, `-fa`, `-ctk/-ctv`,
  `--cache-reuse`, `--cache-ram`, `--jinja`) come from `models/RUN_APPROACH.md`. A config field the
  operator pastes the invocation into was considered and not adopted: a field nobody updates is worse
  than a document that is the operating point of record.
- **Full server `timings` are kept only on dialogue rows** (N1). Survey rows keep `prompt_n`; score rows
  keep none. The compute-budget claims in `models/RUN_APPROACH.md` are checkable from the dialogue rows,
  which are 99% of the compute.
- **Timestamps are local time with a UTC offset, not UTC** (N4). Unambiguous as long as the box's
  timezone does not change mid-study; changing the format now would make old and new runs inconsistent,
  which is worse.
- **No `--verify-hashes` flag** (S6, superseded). The GGUF hash cache is keyed by
  `path|size|mtime_ns`, which closes the stale-hash case that motivated the flag. To force a full
  re-hash anyway, delete `~/.cache/llm-polarization/gguf-hashes.json`.
- **`archive.json` is written by hand** (S12). The destination is not known when a run starts and
  `manifest.json` is deliberately never rewritten, so the archival step is the written procedure in
  section 3 rather than code.
- **`write_manifest` is not atomic across processes** and `JsonlWriter` reopens the file per write. Both
  assume one `run` process per `run_id`, which is how the study is operated; the per-write open and flush
  is the right durability trade for a research log.
- **`cmd_run` builds its agents twice**, once inside `check` and once for the run: two extra `/props`
  round-trips per run, in exchange for the manifest recording `/props` as it stands at the moment the run
  actually starts.
- **Test-level minors** left as they are: `test_cache_warning_true` asserts only the second seeker row,
  the error-row test does not assert `prompt_n is None`, and one docstring says "if None" where the code
  means "if falsy".
- **The `Message` docstring says "turn message"** (`harness/transcript.py`) for what is actually one
  agent's utterance within a turn — a turn is the seeker's line and the mentor's reply together. Left as
  worded: it does not affect what the class holds or how it is used.
- **`import os, sys` on one line** in `harness/templates.py`. A style nit; both names are used nearby and
  splitting the import onto two lines changes nothing about the code.
- **`_gguf_module` mutates `sys.path`** for the life of the process (`harness/templates.py`), once per
  unique `gguf_py_path`. Acceptable here: the harness is a single-purpose CLI invocation, not a library
  loaded into a longer-lived process where that mutation could collide with something else.
- **Task 4's RED-phase evidence was prose, not pasted output.** The fix-wave record describes what the
  failing test showed rather than including its console output verbatim. Left as is: the passing
  GREEN-phase run is what a reviewer checks today, and it is pasted in full.
- **The redundant `pass` in `ServerError`** (`harness/client.py`). Its docstring already makes the class
  body non-empty; the `pass` is a no-op left over from before the docstring was added.
- **`run.py` is broad** (about 600 lines) but cohesive as planned: it is the one place that owns the
  check/run/survey/score subcommands and the plumbing (`RunContext`, `_agents`, `_verify_identity`,
  provenance capture) they share. Splitting it was considered and rejected — the shared machinery would
  then cross a module boundary for no isolation gained.
