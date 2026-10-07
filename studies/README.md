# studies/

Saved study designs for the sandbox GUI. Each `<name>.study.json` is one study spec, schema
`sandbox-study/1`, defined in section 3 of `docs/superpowers/specs/2026-10-07-sandbox-gui-design.md`:
crossed factors (axes) with their levels and slot text, persona variants nested in one factor, lookup
tables, derived slots, the persona and reminder templates, an optional no-persona control, the
randomization, the survey instrument and the run settings. `sandbox/study.py` validates a spec and compiles
it into the same manifest rows `python -m harness.randomize` writes. A study design is a design of record,
so these files are tracked.

`example-institutional-trust.study.json` shows an axis other than political ideology: trust in institutions
(low/high) crossed with topic and certainty. It has two persona variants nested in each trust level, a
`positions` table looked up by trust and topic, a bare control per topic, and 60-turn dialogues.

The repo's own study is **not** saved here. `sandbox/repo_study.py` rebuilds it from `prompts/grid.json`,
the persona catalogue, `instruments/batteries.json` and the config every time it is loaded, so it can never
drift from the files the harness reads. To change it, edit those files. To vary it, open it in the GUI and
"Save as" a new name: the copy is then an ordinary study here.

The GUI (`python -m sandbox`) lists the `repo` preset first and then every `*.study.json` in this directory.
Save writes `studies/<name>.study.json` (names match `^[a-z0-9][a-z0-9_-]*$`). A draft with validation
errors can be saved but not exported. The files are plain JSON, so they can also be written by hand and
checked with `sandbox.study.validate_study`.
