# prompts/

The frozen factorial (`grid.json`) and the seeker persona catalogue
(`personas/`). The catalogue **format** is fixed and tested
(`personas/catalogue.example.json`, placeholder text); the real
`personas/catalogue.json` is not written yet (Notion task: "Write the seeker
persona prompt template"). Requirements from `docs/design/persona-stability.md`:

- Rich narrative persona: backstory, values, two or three explicit stance
  anchors in the persona's own words, not a demographic list.
- The persona opens by asking for guidance, which fixes the mentor in the
  advisor frame. Hold that constant across every cell.
- A two-to-three sentence reminder (name, ideology, stance anchors, openness)
  phrased as a note to self, appended after the history in reinforced mode.
- Nothing is ever injected on the mentor side.

Version every template: bump the catalogue's `version`. The randomizer's
assignment log records that version and the catalogue's sha256, and every
`dyads.jsonl` row carries the rendered persona and reminder in full.

`grid.json` is the frozen factorial (`docs/decisions/factorial.md`,
2026-09-11): the factor levels, N per cell, the control cell and extension E1.
Persona templates fill against its levels; the randomizer reads it; the role
slugs for each ideology level's three variants are registered alongside the
personas, not in the grid. `harness/tests/test_grid.py` pins it.

## The persona catalogue and the randomizer (2026-09-14)

`personas/catalogue.json` is what the randomizer reads (not yet written: the
Notion task "Write the persona catalogue" fills it; `personas/catalogue.example.json`
fixes the **format** with placeholder text and is what the tests run against).
One JSON file:

| Key | What it holds |
|---|---|
| `version` | Bump when any shared string changes. Goes into the assignment log. |
| `template.persona`, `template.reminder` | The slotted seeker system prompt and the two-to-three sentence note-to-self. Slots: `{name}` `{backstory}` `{anchors}` `{anchor_1}`.. `{openness_text}` `{openness_reminder}` `{reminder_self}` `{opening}` `{topic}` `{topic_phrase}` `{ideology}` `{role}`. An unfilled slot is an error, never left in a prompt. |
| `shared.opening`, `shared.topic_phrase` | The setting line and opening request: one string, identical in every cell, control included. |
| `openness.open`, `openness.closed` | `text` (the swapped persona line) and `reminder` (its clause in the note-to-self). |
| `roles.<ideology>[]` | The role variants for that level: `slug` (becomes `condition.role`), `name`, `backstory`, `reminder_self`. At most `variants_per_level` (3) per level; slugs unique; a level not in the grid is refused. One per level is the pilot minimum. |
| `anchors.<ideology>.<topic>[]` | Two or three stance anchors in the persona's own words, for every (level, topic). Moderate anchors are specific centrist positions. |
| `control.persona`, `control.reminder` | The bare control: normally just `{opening}`, and the framing-only reminder. |

```bash
python -m harness.randomize --catalogue prompts/personas/catalogue.json --out pilot-dyads.jsonl \
    --seed 20260918 --n-per-cell 5 --modes reinforced,once --prefix p        # 210 rows: the pilot grid
python -m harness.randomize --catalogue prompts/personas/catalogue.json --out wave1-dyads.jsonl \
    --seed 20261005 --prefix w1                                            # 2,970 rows: grid defaults
```

Each treated cell's `n_per_cell` rows are split evenly across the level's role
variants (refused if they do not divide); the control cells are always
`reinforced` and are not multiplied by `--modes`; every row gets a distinct
per-dyad `seed` from one RNG seeded with `--seed`; the rows are shuffled so
cells interleave across the server's slots. `<stem>-assignment.json` beside the
manifest records the grid and catalogue hashes, the RNG seed, rows per cell,
per variant and per mode, and the output hash: that file is the randomization
paragraph of the design section. `<stem>` is the output name without its
extension and without a trailing `-dyads`, so `pilot-dyads.jsonl` gets
`pilot-assignment.json`. `--check` validates a catalogue against the grid and
writes nothing. `--grid` defaults to `prompts/grid.json`, relative to the
working directory, so run from the repository root. `check --manifest` and
`run` refuse any row whose condition is not a cell of the grid.

When `--n-per-cell` is given, the control cells follow it unless
`--n-control` is also given; with neither, both come from the grid (135).
