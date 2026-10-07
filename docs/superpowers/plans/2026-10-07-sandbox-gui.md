# Sandbox GUI Implementation Plan

> **For agentic workers:** implemented with subagent-driven development: one subagent per task group,
> groups A, B, C and E in parallel, D after A, B and C. Agents do not commit; the controller commits after
> each phase. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a local web GUI (`python -m sandbox`) in which a scientist designs a study over arbitrary axes,
loads this repo's own study exactly, runs long seeker/mentor dialogues through the real harness CLI,
watches them live and reads the outcome by axis.

**Architecture:** package `sandbox/` beside `harness/`: pure modules for the study spec (`study`,
`repo_study`, `export`), for reading runs (`runs`, `analysis`), a subprocess job manager (`jobs`), a mock
llama-server for GPU-free use (`mock_server`), and a stdlib `ThreadingHTTPServer` (`server`) serving a JSON
API and a static ES-module frontend (`static/`).

**Tech stack:** Python 3.11+ stdlib, `jinja2` (through the harness), `pytest`. `gguf` (gguf-py) only for
the end-to-end test. Playwright only for the browser smoke test. No web framework, no npm, no CDN.

**Spec:** `docs/superpowers/specs/2026-10-07-sandbox-gui-design.md` (the schema, the API contract and the
views are defined there and not repeated here).

## Global constraints

- Do not modify anything under `harness/`, `prompts/`, `instruments/`: the sandbox is a layer over the
  instrument, not a change to it. Import from the harness; never re-implement what it already does
  (`harness.randomize._fill`, `build_manifest`, `write_manifest`, `harness.survey.load_batteries`,
  `harness.scorer.latest_complete_attempts`, `harness.log.read_jsonl`, `harness.transcript.message_order`).
- Code style follows `harness/`: `from __future__ import annotations`, a module docstring saying what the
  module is for, a one-line-plus docstring on every public function saying *why* where the code cannot,
  dataclasses, no classes where a function will do, line length about 120.
- Errors the operator can cause are `ValueError` subclasses with a message that names the field; the API
  turns them into 400s. Nothing the user types reaches a shell.
- Tests: `python -m pytest sandbox/tests -q` from the repo root; `harness/tests` must stay green.

## File structure

```
sandbox/
  __init__.py            package marker, __version__
  __main__.py            python -m sandbox -> server.main()
  study.py               the study spec: validate, summarize, slots, cells, render, compile
  repo_study.py          repo files <-> spec; randomization presets
  export.py              write an export directory; CLI commands
  jobs.py                build_argv, JobManager
  runs.py                list runs, run summary, dyad detail, tail, level order
  analysis.py            survey shifts, score curves, analyze_run
  mock_server.py         write_gguf, MockLlamaServer, MockBackend, main()
  server.py              App (routing, testable without sockets), make_server, main
  README.md
  static/index.html  static/css/app.css  static/js/*.js
  tests/
    __init__.py  conftest.py  fakerun.py (make_fake_run: real-shaped run data via harness.run.run_dyad + FakeClient)
    test_study.py test_repo_study.py test_export.py test_jobs.py test_mock_server.py
    test_runs.py test_analysis.py test_server.py test_e2e.py test_ui.py
studies/README.md, studies/example-institutional-trust.study.json
```

---

### Task group A: the study engine (`study.py`, `repo_study.py`, example study)

**Interfaces produced:**

```python
# sandbox/study.py
SCHEMA = "sandbox-study/1"
class StudyError(ValueError):              # .issues: list[dict]
def validate_study(spec) -> list[dict]     # never raises, whatever the input
def has_errors(issues) -> bool
def summarize(spec) -> dict                # best effort on invalid specs: factors [{key, label, n_levels}], nested,
                                           # cells_treated, cells_control, rows, rows_per_mode, n_turns, messages,
                                           # condition_keys
def condition_keys(spec) -> list[str]
def slot_catalog(spec) -> dict             # {"treated": [{"name", "source"}], "control": [...]}
def enumerate_cells(spec) -> list[dict]    # [{"key", "kind", "condition", "variants", "n_rows"}]
def render_cell(spec, kind, condition, variant=None) -> dict   # {"persona_text", "persona_reminder", "slots"}
def resolved_n_control(spec) -> int | None
def compile_manifest(spec) -> tuple[list[dict], dict]          # raises StudyError when validation has errors
def canonical_sha256(spec) -> str          # sha256 of json.dumps(spec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
def blank_study() -> dict                  # a small valid generic study for "New"

# sandbox/repo_study.py
RANDOMIZATION_PRESETS: dict                # {"pilot": {...}, "wave1": {...}} (prompts/README.md)
class NotRepoShaped(ValueError)
def repo_sources(root) -> dict[str, Path]  # grid, catalogue, batteries, config (absolute; catalogue.json else .example)
def from_repo_files(grid, catalogue, batteries, config, sources=None) -> dict
def load_repo_study(root) -> dict
def repo_shape_reason(spec) -> str | None  # None when to_repo_files would succeed
def to_repo_files(spec) -> tuple[dict, dict]
def build_manifest_kwargs(spec) -> dict    # seed, n_per_cell, n_control, modes, n_turns, prefix for build_manifest
```

- [ ] Tests first (`test_study.py`, `test_repo_study.py`), then code. Required tests include:
  - `compile_manifest(load_repo_study(REPO))` equals `harness.randomize.build_manifest(grid, catalogue, ...)`
    row for row (dyad ids, conditions, persona text, reminder, modes, seeds, order) for the grid defaults and
    for the pilot preset; assignment `rows_per_cell`, `rows_per_variant`, `rows_per_mode` equal too.
  - `to_repo_files(load_repo_study(REPO)) == (grid, catalogue)` as on disk.
  - Every validation rule in spec 3.3 has a test that triggers it and checks the `path`.
  - `validate_study` returns issues (does not raise) for `None`, `[]`, missing keys, wrong types.
  - A subset (`cells`) gives rows identical to the same cells' rows in the full manifest.
  - A generic study (the example in `studies/`) validates clean and compiles; a study without `nested`,
    without `tables`, without `control` compiles.
  - The repo study with a changed persona template is still repo-shaped; with a fourth factor it is not, and
    the reason says why.

### Task group B: mock backend and jobs (`mock_server.py`, `jobs.py`)

```python
# sandbox/mock_server.py
CHATML: str
def write_gguf(path, metadata: dict) -> Path       # minimal GGUF v3, zero tensors; str, int (uint32), list[str]
def write_mock_gguf(path, role: str, template: str = CHATML) -> Path
class MockLlamaServer:  # (role, gguf_path, *, host="127.0.0.1", port=0, slots=8, n_ctx=131072, delay=0.0, template=CHATML)
    url: str; start() -> str; stop() -> None
class MockBackend:      # (directory, *, slots=8, delay=0.0, base_port=0)  seeker, mentor, judge
    start() -> dict; stop(); running: bool; info() -> dict   # {"running", "seeker": {"url", "gguf_path"}, ...}
def main(argv=None) -> int

# sandbox/jobs.py
JOB_KINDS = ("check", "run", "survey", "score", "flags", "agreement")
EXIT_MEANINGS: dict[int, str]
def build_argv(kind: str, fields: dict, python: str = sys.executable) -> list[str]   # ValueError on bad fields
class JobManager:       # (root, log_dir, python=sys.executable)
    start(kind, fields, label="") -> dict
    start_argv(argv, *, kind, label="", run_id=None) -> dict
    list() -> list[dict]; get(job_id) -> dict; read_log(job_id, offset=0, max_bytes=65536) -> tuple[str, int]
    stop(job_id) -> dict; wait(job_id, timeout=None) -> dict
```

- [ ] The mock implements `/health`, `/props` (`model_path`, `model_alias`, `total_slots`, `chat_template`,
  `build_info: "sandbox-mock"`, `default_generation_settings.n_ctx`), `/apply-template` (rendered with
  `harness.templates.render`), `/tokenize` (whitespace tokens), `/completion` with a per-slot prompt cache
  (`prompt_n` = tokens past the common prefix with that slot's previous prompt), `stop_type`, `timings`,
  `truncated`, `tokens_evaluated`, `tokens_cached`; JSON-schema requests get `{"answer": int}` within the
  schema's bounds or `{"score": float, "rationale": str}`; free text is deterministic in (seed, prompt) and
  capped at `n_predict` words. Tests: `harness.templates.read_template_from_gguf` reads a written GGUF back
  (skip without `gguf`); `harness.run.check_agent` passes against a running mock (parity, cache reuse,
  trailing system) using a `LlamaClient`; concurrent completions on different slots.
- [ ] Jobs tests: argv for every kind; refused fields (unknown kind, a missing `run_id`, a non-numeric
  threshold, an option-looking value such as `--x`); a job that exits 0, 1 and 2 gets the right status and
  meaning; `stop` on a long-running Python child ends it as `stopped`; `read_log` offsets.

### Task group C: reading runs (`runs.py`, `analysis.py`, `tests/fakerun.py`)

```python
# sandbox/runs.py
RUN_ID_RE; class RunNotFound(LookupError)
def resolve_data_dir(root, data_dir=None) -> Path
def run_dir(data_dir, run_id) -> Path
def read_from(path, offset) -> tuple[list[dict], int]       # whole lines only; offset in bytes
def list_runs(data_dir) -> list[dict]
def run_summary(data_dir, run_id) -> dict
def dyad_detail(data_dir, run_id, dyad_id, attempt=None) -> dict
def tail(data_dir, run_id, offsets: dict) -> dict
def level_order(data_dir, run_id, root=None) -> dict[str, list]

# sandbox/analysis.py
def survey_shifts(dyads, status, surveys, factor=None, level_order=None, origin="run") -> dict
def score_curves(dyads, status, scores, factor=None, metric="alignment", judge=None, level_order=None) -> dict
def analyze_run(data_dir, run_id, factor=None, metric="alignment", judge=None, level_order=None) -> dict

# sandbox/tests/fakerun.py
def make_fake_run(data_dir, run_id="fake-run", *, n_turns=3, conditions=None, fail=(), scores=True) -> Path
```

- [ ] `make_fake_run` writes a real-shaped run with the harness's own code (`harness.run.run_dyad`,
  `RunContext`, `FakeClient`, `JsonlWriter`, `write_manifest`, `Scorer`), so the readers are tested on what
  the harness writes, not on hand-made rows.
- [ ] Tests: summaries count complete/failed/in-flight; a retried dyad's latest complete attempt is used;
  transcripts are in `message_order`; `tail` returns only whole lines and resumes from the offset; bad run
  ids and `..` are refused; survey shift arithmetic on hand-checked numbers including a null answer;
  battery normalisation across 1-5 and 0-10; score curves refuse two judges without `judge`; `level_order`
  follows the grid.

### Task group D (after A, B, C): export and server (`export.py`, `server.py`, `__main__.py`)

```python
# sandbox/export.py
def default_export_dir(workspace, spec) -> Path
def export_study(spec, *, root, workspace, out_dir=None, python=sys.executable) -> dict   # spec section 5
def config_from_manifest(data_dir, run_id, workspace, judge=None) -> Path

# sandbox/server.py
class App:      # (root, *, workspace=None, studies_dir=None, host="127.0.0.1", port=8765)
    def handle(self, method, path, query: dict, body: bytes, headers: dict) -> tuple[int, dict, bytes]
def make_server(root, *, host="127.0.0.1", port=8765, workspace=None, studies_dir=None) -> ThreadingHTTPServer
def main(argv=None) -> int      # --host --port --root --workspace --studies
```

- [ ] Export tests: the repo study exports through `harness.randomize` with the source paths reused and the
  assignment log byte-identical (bar `generated`) to running the `randomize` CLI with the printed command;
  a changed instrument is written into the export directory and the config points at it; a generic study
  exports through `sandbox.study` with `config.grid = null` and a `study.json`; a subset is noted;
  validation errors refuse the export.
- [ ] Server tests: every route in spec section 5 (through `App.handle`, and one real socket round trip);
  the `X-Sandbox` and `Host` checks; path traversal; saving over `repo` is 409.

### Task group E (parallel with A-D, against the API contract): the app (`static/`)

- [ ] Every view of spec section 6, vanilla ES modules, no external assets, light and dark via
  `prefers-color-scheme`, usable at 1280 px and at 900 px. Mutating requests send `X-Sandbox: 1`.
  Editing is local state with debounced `/api/study/validate`; nothing is saved without Save.
- [ ] `test_ui.py` (skipped without Playwright): load the page, the repo study validates, a persona preview
  renders, the run list renders.

### Task group F (controller): end to end, docs, review

- [ ] `test_e2e.py`: mock backend -> export a two-cell subset of the repo study at 3 turns -> `check`,
  `run`, `score` via `JobManager` -> `runs.run_summary` all complete -> `analysis` has shifts and curves.
- [ ] `sandbox/README.md`, `studies/README.md`, the root README's layout and status, `.gitignore`
  (`workspace/`).
- [ ] Review pass against this plan and the spec; both test suites green; commit; push.
