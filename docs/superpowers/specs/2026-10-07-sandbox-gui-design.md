# Sandbox GUI design

Date: 2026-10-07. Status: approved in conversation (the request), spec of record for `sandbox/`.
Request: "a GUI sandbox for scientists to test different axes (such as political polarization) on
long-duration conversations with many turns; follow the structure of this repo; ideally specify the exact
study this repo is implementing from the GUI."

## 1. Purpose

A local web GUI over `harness/`. A scientist uses it to:

1. **Design a study** as a set of crossed *axes* (factors), nested persona variants, lookup tables of stance
   text, persona and reminder templates, an optional no-persona control, a survey instrument, models and
   sampling settings. Any axes, not only ideology.
2. **Load the repo's own study exactly**: the preset is built live from `prompts/grid.json`, the persona
   catalogue, `instruments/batteries.json` and the config, and exporting it produces the manifest that
   `python -m harness.randomize` produces, by calling that code. Nothing about the study is re-implemented
   for the preset path.
3. **Run long conversations** through the real CLI (`python -m harness.run check|run|score|...`), launched
   as subprocesses, so a sandbox run is a harness run with all of its guards and provenance.
4. **Watch and inspect** runs live: transcript per dyad, cache accounting per turn, pre/post survey answers,
   judge scores over turns, and the outcome (mentor survey shift post minus pre) broken down by any axis.

Non-goals (v1): editing the judge prompt or metrics (they live in `harness/scorer.py` and are recorded in
`judge-*.json`); anything on the mentor side (the harness never sends the mentor a system prompt, and the
GUI must not offer one: "any system text on the mentor is treatment"); authentication (the server binds
`127.0.0.1`; reach it from another machine with an SSH tunnel).

## 2. Architecture

A Python package `sandbox/` beside `harness/`. Same constraints as the harness: Python 3.11+, the standard
library plus `jinja2` (via the harness); no web framework, no build step, no CDN. The frontend is static
HTML, CSS and ES-module JavaScript served by the same process, so it works offline on the Framework Desktop.

| Module | Responsibility | Depends on |
|---|---|---|
| `study.py` | The study spec (section 3): validate, enumerate cells, render a cell's persona, compile a manifest. Pure functions. | harness.transcript, harness.log |
| `repo_study.py` | Build the spec from the repo's files; map a repo-shaped spec back to `grid.json` + catalogue. | study, harness.randomize |
| `export.py` | Write the manifest, assignment log, config, batteries (and grid + catalogue when repo-shaped) into an export directory; build the CLI argv for each command. | study, repo_study, harness.randomize |
| `jobs.py` | Run `harness.run` / `harness.randomize` as subprocesses; log to file; stop with SIGINT. | stdlib |
| `runs.py` | Read `data/<run_id>/`: list runs, summarise one, a dyad's transcript, tail new rows by byte offset. | harness.log, harness.scorer |
| `analysis.py` | Survey shift (post minus pre) and judge-score curves by axis level. Pure functions, stdlib statistics. | harness.scorer |
| `mock_server.py` | A fake llama-server (and a minimal GGUF writer) so the sandbox and the end-to-end test run without a GPU. Never for data. | harness.templates |
| `server.py`, `__main__.py` | `ThreadingHTTPServer`, JSON API (section 5), static files. `python -m sandbox`. | all |
| `static/` | The single-page app (section 6). | the API |

Data does not move: runs are written by the harness to the config's `data_dir` (`data/` by default) and the
GUI reads them there, so runs started from a terminal show up in the GUI too.

Files on disk:

```
studies/*.study.json                 saved study specs (tracked: a study design is a design of record)
workspace/                           git-ignored scratch
  exports/<study>-<stamp>/           <prefix>-dyads.jsonl, <prefix>-assignment.json, config.json,
                                     batteries.json | grid.json | catalogue.json (only when changed),
                                     study.json (the spec as exported)
  configs/<run_id>*.json             configs derived from a run's manifest (for score / flags / survey)
  jobs/<job_id>.log                  subprocess output
  mock/                              mock GGUFs
```

## 3. The study spec (`sandbox-study/1`)

One JSON object. Every key below is required unless marked optional.

```jsonc
{
  "schema": "sandbox-study/1",
  "name": "LLM political polarization (repo study)",
  "description": "",                                   // optional
  "factors": [                                         // crossed, in crossing order (outermost first)
    {"key": "topic", "label": "Topic",                 // label optional
     "levels": [{"id": "immigration_enforcement", "code": "immig",   // code optional: used in dyad_id
                 "slots": {"topic_phrase": "immigration enforcement"}}]},
    {"key": "ideology", "levels": [{"id": "strong_left", "slots": {}}]},
    {"key": "openness", "levels": [{"id": "open", "slots": {"openness_text": "...", "openness_reminder": "..."}}]}
  ],
  "nested": {                                          // optional (null): persona variants nested in one factor
    "key": "role", "within": "ideology", "max_per_level": 3,
    "variants": {"strong_left": [{"id": "union_organizer", "slots": {"name": "...", "backstory": "...", "reminder_self": "..."}}]}
  },
  "tables": [                                          // optional ([]): text looked up by a combination of levels
    {"name": "anchors", "by": ["ideology", "topic"], "join": " ", "item_slot": "anchor_{i}",
     "values": {"strong_left": {"immigration_enforcement": ["...", "..."]}}}
  ],
  "derived": {"opening": "You came ... about {topic_phrase}. ..."},   // optional ({}): slots built from slots
  "templates": {"persona": "You are {name}. {backstory} {anchors} {openness_text} {opening}",
                "reminder": "Note to self: I am {name}, ..."},
  "control": {                                         // optional (null): a no-persona cell per `by` combination
    "factor": "ideology", "level": "none", "by": ["topic"],
    "persona": "{opening}", "reminder": "Note to self: ...", "persona_mode": "reinforced",
    "n_per_cell": null                                 // null: same as randomization.n_per_cell
  },
  "randomization": {"seed": 20261005, "n_per_cell": 135, "modes": ["reinforced"], "n_turns": 40,
                    "prefix": "w", "cells": null},     // cells: null (all) or a list of cell keys (a subset)
  "instrument": {"version": "...", "items": [{"id": "...", "battery": "...", "scale": {"min": 1, "max": 5}, "text": "..."}]},
  "run": {"data_dir": "data", "gguf_py_path": null, "run_seed": 20260908, "now": "2026-09-08",
          "concurrency": null, "cache_reuse_limit": 1000,
          "generation": {"temperature": 0.7, "top_p": 0.95, "n_predict": 300, "timeout": 600, "enable_thinking": false},
          "seeker": {"url": "...", "gguf_path": null}, "mentor": {...}, "judge": {...}},
  "repo": null                                         // optional; set only by repo_study (section 4)
}
```

`instrument` has the shape of `instruments/batteries.json` (extra keys such as `adapted`, `_note` are kept).
`run` is a harness config without `batteries` and `grid`; export fills those two in.

### 3.1 Slots

A template is filled with Python `str.format_map`; an unfilled slot is an error naming the slot, never left
in a prompt; the result is `.strip()`ped (exactly `harness.randomize._fill`). The slots of a **treated**
cell, for one level per factor, one nested variant and one mode:

| Slot | Value |
|---|---|
| `{<factor key>}` | the level id (`{topic}`, `{ideology}`, `{openness}`) |
| each key of each chosen level's `slots` | its value (`{topic_phrase}`, `{openness_text}` ...) |
| `{<nested key>}` | the variant id (`{role}`) |
| each key of the chosen variant's `slots` | its value (`{name}`, `{backstory}`, `{reminder_self}`) |
| `{<table name>}` | the looked-up list joined with `join` (`{anchors}`) |
| `item_slot` with `{i}` = 1..n | each list item (`{anchor_1}`, `{anchor_2}` ...) |
| each `derived` key, in order | its template filled from the slots above plus earlier derived slots |

A **control** cell has only `{<by factor key>}`, the `by` levels' slots, and the derived slots that can be
filled from those (a derived slot that cannot is simply absent from the control). Slot names must not
collide across sources (a level slot may repeat across levels of the same factor, which is the point).

### 3.2 Cells, rows and ids (identical to `harness.randomize.build_manifest`)

- Treated cells: the cartesian product of the factors' levels in factor order. Cell key: level ids joined
  with `/` (`immigration_enforcement/strong_left/open`).
- For each treated cell, for each mode in `modes`, for each variant of the cell's `within` level (in
  catalogue order; one implicit variant when `nested` is null), for `k` in `1..n_per_cell / n_variants`
  (refused unless it divides): one row with
  `dyad_id = "-".join([prefix, *(level.code or level.id for each factor), variant.id (if nested), mode, f"{k:03d}"])`
  and `condition = {factor.key: level.id ..., nested.key: variant.id}`.
- Then, if `control`: for each combination of the `by` factors' levels (factor order), for `k` in
  `1..n_control`: `dyad_id = "-".join([prefix, *(codes of by levels), "control", control.persona_mode, f"{k:03d}"])`,
  `condition` = every factor key in factor order with the `by` level, `control.level` for `control.factor`,
  null otherwise, then `nested.key: null`. Cell key: `by` level ids plus `control.level`, joined with `/`.
- Row: `{"dyad_id", "condition", "persona_text", "persona_reminder", "persona_mode", "seed", "n_turns"}`.
- Seeds: `rng = random.Random(seed)`; for each row in the order above `s = rng.getrandbits(31)`, redrawn while
  `s` was already drawn or is 0. Then `rng.shuffle(rows)`. Duplicate `dyad_id`s are an error.
- `cells` (subset): applied after the shuffle, so a subset's rows are byte-identical to the full design's
  rows for the same cells. This is how a sandbox pilot of two cells at 100 turns is made.

The assignment summary has the keys `harness.randomize` writes (`generated, harness_version, rng_seed,
n_per_cell, n_control, modes, n_turns, prefix, rows, rows_per_cell, rows_per_variant, rows_per_mode,
variants_per_level, catalogue, grid`) plus `engine` (`"sandbox.study"`), `study` `{name, sha256}` (of the
canonical JSON of the spec) and `cells_filter`.

### 3.3 Validation

`validate_study(spec)` returns a list of issues `{"level": "error" | "warning", "path": "factors[1].levels[0].id",
"message": "..."}`. Errors block export; warnings are shown. It checks shape and types; key and id syntax
(`^[a-z][a-z0-9_]*$` for factor/nested keys and slot names, `^[A-Za-z0-9_.]+$` for level ids, codes, variant
ids and prefix); uniqueness; `nested.within` is a factor; every `within` level has 1..`max_per_level`
variants and none is registered under an unknown level; variant ids unique across the study; every table
combination present and non-empty; no slot collisions; every treated and control cell renders; a rendered
reminder is non-empty wherever the mode is `reinforced`; `control.level` is not a level of `control.factor`;
`control.by` excludes `control.factor`; `n_per_cell` divides by each level's variant count; `modes` a non-empty
subset of `harness.transcript.PERSONA_MODES`; `n_turns >= 1`; `cells` entries are cell keys; the instrument
passes the rules of `harness.survey.load_batteries` and `scale.min <= scale.max`. Warnings: missing
seeker/mentor URL; seeker URL equal to mentor URL (`run` refuses it); condition keys other than
`topic, ideology, openness, role` (the harness grid gate will be off: `config.grid = null`, the sandbox has
validated the conditions); no factor named `topic` (the judge prompt reads `condition.topic`).

## 4. The repo study

`repo_study.load_repo_study(root)` reads `prompts/grid.json`, `prompts/personas/catalogue.json` (or
`catalogue.example.json` until the real catalogue is written), `instruments/batteries.json`, and `config.json`
at the root (or `harness/config.example.json`). Mapping:

| Spec | From |
|---|---|
| factors `topic` / `ideology` / `openness` | `grid.factors`; topic `code` = `harness.randomize._topic_code`; topic slot `topic_phrase` from `shared.topic_phrase`; openness slots `openness_text`, `openness_reminder` from `openness.<level>.text/reminder` |
| `nested` | `role` within `ideology`, `max_per_level = grid.variants_per_level`, variants from `roles` (`slug` -> `id`, every other field -> `slots`) |
| `tables` | one: `anchors` by `[ideology, topic]`, join `" "`, item slot `anchor_{i}` |
| `derived` | `{"opening": shared.opening}` |
| `templates`, `control` | `template`; control `factor: ideology`, `level: grid.control.ideology`, `by: [topic]`, `persona_mode: reinforced`, persona and reminder from `control`, `n_per_cell` null when `grid.control.n_per_cell == grid.n_per_cell` |
| `randomization` | seed 20261005, `n_per_cell = grid.n_per_cell`, modes `[grid.fixed.persona_mode]`, `n_turns = grid.fixed.n_turns`, prefix `w` (wave 1 in `prompts/README.md`); presets `pilot` (seed 20260918, 5 per cell, reinforced + once, prefix p) and `wave1` |
| `instrument`, `run` | the batteries file; the config minus `batteries` and `grid` |
| `repo` | `{"grid": <grid.json verbatim>, "catalogue": <the catalogue's other keys: version, _note, ...>, "sources": {grid, catalogue, batteries, config: root-relative paths}}` |

`to_repo_files(spec)` inverts this into `(grid, catalogue)` (the stored grid with `factors`,
`variants_per_level` and `control.ideology` replaced from the spec) and raises `NotRepoShaped(reason)` for a
spec that cannot be expressed as the repo's files. `to_repo_files(load_repo_study(root))` equals the files
on disk. Export of a repo-shaped spec writes the manifest with `harness.randomize.build_manifest` and
`write_manifest` (engine `harness.randomize`), reusing a source file's path whenever the spec's content
equals it, so the assignment log and `manifest.json` carry the same hashes as a CLI-made manifest. A test
holds `study.compile_manifest` and `harness.randomize.build_manifest` to the same rows for the preset, so
the GUI's preview of the repo study is the study.

## 5. HTTP API

JSON in and out. Errors are `{"error": "...", "issues": [...]?}` with status 400 (bad input), 404, 409
(e.g. saving over the `repo` preset) or 500. Every non-GET request must carry `X-Sandbox: 1` and
`Content-Type: application/json` (a cross-site page cannot send that header without a preflight the server
never approves), and every request's `Host` must be the bound host, `localhost` or `127.0.0.1`
(DNS rebinding). Run ids and dyad ids match `^[A-Za-z0-9_.-]+$`; every path is resolved and checked to lie
inside its directory.

| Method, path | Body / query | Returns |
|---|---|---|
| `GET /` and `/static/...` | | the app |
| `GET /api/meta` | | `root, harness_version, python, workspace, studies_dir, data_dirs, randomization_presets, gguf_importable, repo_sources` |
| `GET /api/studies` | | `{"studies": [{"name", "title", "kind": "preset" \| "saved", "path"}]}`; `repo` first |
| `GET /api/studies/<name>` | | `{"name", "kind", "path", "spec"}` (`repo` is rebuilt from the files on every call) |
| `PUT /api/studies/<name>` | `{"spec"}` | `{"path", "issues"}`; name `^[a-z0-9][a-z0-9_-]*$`; `repo` -> 409; drafts with errors may be saved |
| `POST /api/study/validate` | `{"spec"}` | `{"issues", "summary", "slots", "repo_shaped", "repo_shape_reason"}` |
| `POST /api/study/cells` | `{"spec"}` | `{"cells": [{"key", "kind", "condition", "variants", "n_rows"}]}` |
| `POST /api/study/render` | `{"spec", "kind", "condition", "variant"}` | `{"persona_text", "persona_reminder", "slots"}` |
| `POST /api/study/manifest` | `{"spec", "limit"}` | `{"rows" (first limit), "total", "assignment"}` |
| `POST /api/study/export` | `{"spec"}` | export result (below) |
| `POST /api/study/repo-files` | `{"spec"}` | `{"grid", "catalogue"}`, or 400 with the reason |
| `GET /api/runs` | `?data_dir=` | `{"data_dir", "runs": [...]}` |
| `GET /api/runs/<run_id>` | `?data_dir=` | run summary |
| `GET /api/runs/<run_id>/dyads/<dyad_id>` | `?data_dir=&attempt=` | dyad detail |
| `GET /api/runs/<run_id>/tail` | `?data_dir=&turns=&status=&surveys=&scores=` (byte offsets) | `{"rows": {file: [...]}, "offsets": {...}}` |
| `GET /api/runs/<run_id>/analysis` | `?data_dir=&factor=&metric=&judge=` | analysis (section 7) |
| `POST /api/runs/<run_id>/config` | `{"data_dir", "judge"?}` | `{"path"}`: `workspace/configs/<run_id>[-judge-<sha8>].json` from `manifest.json -> config` |
| `GET /api/jobs` | | `{"jobs": [...]}` |
| `POST /api/jobs` | `{"kind", ...}` (below) | the job |
| `GET /api/jobs/<id>/log` | `?offset=` | `{"job", "text", "offset"}` |
| `POST /api/jobs/<id>/stop` | | the job (SIGINT: in-flight dyads finish, queued ones never start) |
| `GET /api/mock`, `POST /api/mock/start`, `POST /api/mock/stop` | `{"slots", "delay"}` | `{"running", "seeker", "mentor", "judge"}` each `{url, gguf_path}` |

**Export result:** `{"dir", "manifest", "assignment", "config", "batteries", "grid", "catalogue", "study",
"engine": "harness.randomize" | "sandbox.study", "unchanged_from_repo": {"grid", "catalogue", "batteries"},
"rows", "cells_filter", "commands": {"randomize"?, "check", "run", "score"}}` where each command is an argv
list (run from the repo root) that the GUI prints as a copyable shell line.

**Jobs:** the server builds every argv itself from structured fields, never from a client string: `check
{config, manifest?}`, `run {config, manifest, run_id}`, `score {config, run_id, scope, subsample?}`,
`flags {config, run_id, threshold, metric?, run_length?, judge?}`, `agreement {config, run_id, metric?}`,
`survey {config, run_id, phase}`. Argv is `[sys.executable, "-m", "harness.run", kind, ...]`, cwd the repo
root, `PYTHONUNBUFFERED=1`. A job is `{"id", "kind", "label", "argv", "run_id", "status": "running" |
"finished" | "failed" | "stopped", "returncode", "meaning", "started_at", "ended_at", "log_path"}`; `meaning`
follows the harness's exit codes (0 ok, 1 refused, 2 some dyads failed, 130 stopped).

### 5.1 Payload shapes

Fixed here so the readers and the app are built against one contract. A level that is JSON null is the
string `"(none)"` wherever it is a key or a label; with no factor chosen there is one level, `"all"`.

```jsonc
// GET /api/meta
{"root", "harness_version", "sandbox_version", "python", "workspace", "studies_dir",
 "data_dirs": ["data"], "randomization_presets": {"pilot": {...randomization}, "wave1": {...}},
 "gguf_importable": true, "repo_sources": {"grid", "catalogue", "batteries", "config"}}

// POST /api/study/validate
{"issues": [{"level", "path", "message"}], "repo_shaped": true, "repo_shape_reason": null,
 "summary": {"factors": [{"key", "label", "n_levels"}], "nested": {"key", "within", "n_variants": {"<level>": 3}} | null,
             "cells_treated", "cells_control", "rows", "rows_per_mode": {"reinforced": 2700}, "n_turns",
             "messages", "condition_keys": ["topic", "ideology", "openness", "role"]},
 "slots": {"treated": [{"name": "topic_phrase", "source": "level:topic"}], "control": [...]}}
//   slot sources: "factor:<key>", "level:<key>", "nested:<key>", "variant:<key>", "table:<name>", "derived"

// POST /api/study/cells
{"cells": [{"key": "immigration_enforcement/strong_left/open", "kind": "treated",
            "condition": {"topic": "...", "ideology": "...", "openness": "..."}, "variants": ["slug", ...],
            "n_rows": 135, "selected": true}]}     // control: condition holds only the `by` keys

// GET /api/runs
{"data_dir", "runs": [{"run_id", "started_at", "mtime", "has_manifest", "has_scores", "has_flags", "n_dyads",
  "planned",                                           // rows in the input manifest, or null if unreadable
  "status_counts": {"complete", "failed", "running"},  // per dyad, latest attempt; "started" counts as running
  "seeker": {"alias", "model_path", "model_sha256"} | null, "mentor": {...} | null}]}

// GET /api/runs/<run_id>
{"run_id", "path", "planned", "status_counts", "has_scores", "has_flags",
 "manifest": {"started_at", "harness_commit", "harness_dirty", "input_manifest", "batteries", "environment", "config"} | null,
 "roles": {"seeker": RoleInfo, "mentor": RoleInfo},   // RoleInfo: url, alias, model_path, model_sha256, family,
                                                       //   template_sha256, build_info, total_slots
 "judges": [{"file", "alias", "model_path", "model_sha256", "scope", "subsample", "ts"}],
 "dyads": [{"dyad_id", "attempt", "attempts", "condition", "persona_mode", "n_turns",
            "status": "complete" | "failed" | "running", "turns_done", "reason", "last_ts"}],
 "factors": {"ideology": ["strong_left", "..."]},     // condition keys -> levels, in level order
 "metrics": ["alignment"], "judge_shas": ["..."]}

// GET /api/runs/<run_id>/dyads/<dyad_id>
{"run_id", "dyad_id", "attempt", "attempts": [1, 2], "dyad": {...dyads.jsonl row},
 "status": [...status rows], "turns": [...turns.jsonl rows in message_order],
 "surveys": {"pre": [...], "post": [...]},
 "survey_pairs": [{"item_id", "battery", "scale", "pre", "post", "delta"}],
 "scores": [...scores.jsonl rows], "flag": {...} | null}

// GET /api/runs/<run_id>/tail
{"rows": {"turns": [], "status": [], "surveys": [], "scores": []}, "offsets": {"turns": 0, "status": 0, "surveys": 0, "scores": 0}}

// GET /api/runs/<run_id>/analysis
{"factor", "metric", "judge", "factors": {...}, "status_counts", "metrics_available", "judges",
 "survey": {"factor", "levels", "n_dyads": {"<level>": n}, "items": [{"item_id", "battery", "scale"}], "batteries",
            "by_item": {"<level>": {"<item_id>": {"n", "mean_pre", "mean_post", "mean_delta", "se_delta"}}},
            "by_battery": {"<level>": {"<battery>": {"n", "mean_delta_norm", "se_delta_norm"}}}},
 "scores": {"metric", "judge", "judges", "factor", "levels", "error",
            "series": {"<level>": [{"turn", "n", "mean", "se"}]}}}

// mock
{"running": true, "seeker": {"url", "gguf_path"}, "mentor": {...}, "judge": {...}}
```

## 6. The app

One page, a top bar (study picker, New, Save, Save as, validity pill, an "exact repo study" badge when the
export engine would be `harness.randomize`) and these views:

| View | What it holds |
|---|---|
| Study | name, description; summary cards (axes, cells, dialogues, messages, a compute estimate from a tok/s input); the cell table with rows per cell; every validation issue with its path |
| Axes | factors (key, label, levels with id, code and slot columns; add, remove, reorder); nested variants per parent level; lookup tables (one textarea per combination, one item per line) |
| Personas | persona and reminder templates with slot chips that insert at the cursor; derived slots; the control; a live preview of any cell and variant, re-rendered as the templates change |
| Instrument | survey items (id, battery, scale, text), in administration order |
| Models | seeker, mentor, judge (`url`, `gguf_path`); sampling; `run_seed`, `now`, `gguf_py_path`, `data_dir`, concurrency, cache limit; a button that starts the mock backend and fills in its URLs, marked as not for data |
| Run | randomization (seed, n per cell, control n, modes, turns, prefix; repo presets); a cell subset picker; the manifest preview; Export; pre-flight `check`; `run` with a run id; the CLI line for each step; the job's console |
| Runs | run list; a run's models, hashes, settings, status counts and per-dyad progress, refreshed by tailing; a dyad's transcript (seeker left, mentor right, turn numbers, `prompt_n`, cache warnings, finish reason), pre/post survey with deltas, judge scores per turn; analysis by axis; actions: score, flags, agreement, re-administer |
| Jobs | every job, its live log, stop |

## 7. Analysis

Over the latest complete attempt of each dyad (`harness.scorer.latest_complete_attempts`), `origin == "run"`
survey rows. Per dyad and item: `delta = post - pre` when both answers are non-null. Grouped by a chosen
condition key (null level shown as `"(none)"`), ordered by the grid or study level order when the run's
config names a grid or its input manifest has a `study.json` beside it, else by first appearance:
per item `{n, mean_pre, mean_post, mean_delta, se_delta}`; per battery the mean of each dyad's
scale-normalised delta (`delta / (max - min)`) so a 0-10 thermometer and a 1-5 item are comparable. Judge
curves: per level and turn, `{n, mean, se}` of `score` for one metric and one judge (required when two have
scored). Standard error is the sample SD over `sqrt(n)`, null below n = 2. These are descriptive views for a
sandbox; the paper's estimands live in `analysis/`.

## 8. Testing

`python -m pytest sandbox/tests -q`, beside the harness suite. Unit tests per module; the API in process on
an ephemeral port; the randomizer equivalence and repo round-trip tests; an end-to-end test (skipped when
`gguf` cannot be imported) that starts the mock backend, exports a two-cell subset of the repo study at a
few turns, runs `check`, `run` and `score` through the job manager (the real CLI), and reads the result
back through `runs` and `analysis`; a browser smoke test (skipped without Playwright).
