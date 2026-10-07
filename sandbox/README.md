# Dyad sandbox

A local web GUI over `harness/`: design a study over any axes, load this repository's own study exactly, run
long seeker/mentor dialogues through the harness CLI, watch them live, and read the mentor's movement by
axis. Design: `docs/superpowers/specs/2026-10-07-sandbox-gui-design.md`.

```bash
pip install -r harness/requirements.txt         # jinja2 + numpy; nothing else for the sandbox
python -m sandbox                               # http://127.0.0.1:8765
python -m sandbox --port 8800 --open            # another port, and open a browser
```

On the Framework Desktop the server stays on `127.0.0.1`; reach it from the development desktop with a
tunnel: `ssh -L 8765:127.0.0.1:8765 <box>` and open `http://127.0.0.1:8765`. There is no authentication,
and the GUI can start runs, so do not bind it to a public interface.

## What it does

| View | Use it to |
|---|---|
| Study | Name the study, see its cells, dialogue count, message count and a compute estimate, and every validation issue. |
| Axes | Define the crossed factors and their levels (any axes: ideology, trust, certainty ...), persona variants nested in one factor, and lookup tables such as stance anchors per (ideology, topic). |
| Personas | Write the seeker's persona and per-turn reminder as templates over the axes' slots, the derived slots (the opening request), and the no-persona control; preview any cell. |
| Instrument | Edit the mentor's survey items, administered pre and post in this order. |
| Models | Point at the seeker, mentor and judge `llama-server`s; set sampling, `run_seed` and `now`. Or start the mock backend. |
| Run | Randomize (seed, N per cell, delivery modes, turns, prefix; the repo's pilot and wave-1 presets), pick a subset of cells for a sandbox pilot, preview the manifest, export it, run `check` and `run`. Every step prints the CLI line it runs. |
| Runs | Every run in the data directory: models and hashes, progress per dyad, live transcripts with cache accounting, pre/post survey shifts, judge scores over turns, the shift by any axis level; `score`, `flags`, `agreement` and survey re-administration. |
| Jobs | Every CLI process the GUI started, its log, and a stop button (SIGINT: in-flight dyads finish, queued ones never start; resume with the same run id). |

## The repo study, exactly

The study picker's first entry, **repo study**, is rebuilt from the files on every load:
`prompts/grid.json`, `prompts/personas/catalogue.json` (or `catalogue.example.json` until the real
catalogue is written), `instruments/batteries.json`, and `config.json` (or `harness/config.example.json`).
While the study keeps the repo's shape (the three crossed factors, `role` nested in `ideology`, the anchor
table, one derived `opening`, the per-topic control), **Export goes through `harness.randomize` itself**
and reuses the source files' paths whenever their content is unchanged. The manifest and assignment log it
writes are the ones `python -m harness.randomize` writes with the printed arguments, so a run started from
the GUI carries the same hashes as one started from a terminal. The "exact repo study" badge in the top bar
says when that holds. Edit the persona text, the anchors or the instrument and it still holds; the changed
files are written into the export directory and the config points at them.

A study that is not repo-shaped (other axes) exports through `sandbox/study.py`, which follows the same
rules (spec section 3.2, held to `harness.randomize` row for row by `sandbox/tests/test_repo_study.py`).
Its config sets `grid: null`, because the harness's grid gate only knows the repo's four condition keys;
the sandbox validates the conditions itself, and the export directory keeps the `study.json` that defines
them.

## Where things go

| Path | What |
|---|---|
| `studies/*.study.json` | Saved study designs (tracked). Schema: spec section 3. |
| `workspace/exports/<study>-<stamp>/` | An export: `<prefix>-dyads.jsonl`, `<prefix>-assignment.json`, `config.json`, `study.json`, and `grid.json` / `catalogue.json` / `batteries.json` only when they differ from the repo's. |
| `workspace/configs/` | Configs derived from a run's `manifest.json`, for `score`, `flags`, `survey` (optionally with another judge). |
| `workspace/jobs/` | One log and one record per CLI process. |
| `workspace/mock/` | The mock backend's GGUFs. |
| `data/<run_id>/` | Runs, written by the harness as always (`data/README.md`). The GUI only reads them. |

`workspace/` is git-ignored.

## The mock backend

`Models -> Start demo mock servers` (or `python -m sandbox.mock_server --port 18201`) starts three fake
`llama-server`s for the seeker, mentor and judge, each with a tiny GGUF that carries only a ChatML chat
template. They answer every endpoint the harness uses, simulate a per-slot KV cache so `check` passes, and
return deterministic synthetic text, survey answers and judge scores. Use them to learn the GUI and to test
the pipeline on a laptop. **Never for data**: the text is not a model's, and the run's `manifest.json`
records the mock's model path and `build_info: sandbox-mock`. The harness still needs gguf-py to read the
template (`pip install gguf`, or `gguf_py_path` in Models).

## Tests

```bash
python -m pytest sandbox/tests -q       # unit + API; the end-to-end test needs `gguf`, the UI test Playwright
python -m pytest harness/tests -q
```

`test_e2e.py` starts the mock backend, exports a two-cell subset of the repo study at three turns, runs
`check`, `run` and `score` as subprocesses through the job manager, and reads the run back as the GUI does.
