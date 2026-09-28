# prompts/

The frozen factorial (`grid.json`) and the seeker persona catalogue
(`personas/`). The catalogue **format** is fixed and tested
(`personas/catalogue.example.json`, placeholder text); the real
`personas/catalogue.json` is written (version 1.0.0, 2026-09-28; see
"Catalogue 1.0.0" below). Requirements from `docs/design/persona-stability.md`:

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

`personas/catalogue.json` is what the randomizer reads (version 1.0.0, three
role variants per level; `personas/catalogue.example.json` fixes the
**format** with placeholder text and is what the tests run against).
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
    --seed 20260918 --n-per-cell 6 --modes reinforced,once --prefix p        # 252 rows: the pilot grid
python -m harness.randomize --catalogue prompts/personas/catalogue.json --out wave1-dyads.jsonl \
    --seed 20261005 --prefix w1                                            # 2,970 rows: grid defaults
```

Each treated cell's `n_per_cell` rows are split evenly across the level's role
variants (refused if they do not divide, which is why the pilot runs 6 per cell
rather than the 5 the design first named: `docs/decisions/factorial.md`); the
control cells are always `reinforced` and are not multiplied by `--modes`; every row gets a distinct
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

## Catalogue 1.0.0

`personas/catalogue.json`, 2026-09-28 (remediation W1, gap audit F1). 15
backstories (5 levels x 3 variants), 10 anchor sets of three, two openness
texts, the bare control. Topics are framed as "the militarization of
immigration enforcement" and "the decarbonization of the economy", US, 2026.
`strong_right` is pro-militarization and decarbonization-skeptical,
`strong_left` the reverse, the `lean_*` levels softer versions of the same
direction; `moderate` anchors are specific centrist positions (Guard support
at the border but no troops in cities; gas as backup while building out
wind, solar and nuclear).

How the pieces fit:

- Backstories are second person ("You ...") and end with one self-placement
  sentence shared by the level's three variants: "firmly on the left", "a
  little left of center", "squarely in the center", "a little right of
  center", "firmly on the right". No party names anywhere.
- Anchors are first person, quoted in the system prompt, shared by the
  level's three variants, and written with the same three-part structure at
  every level (core position; money or instruments; cities, communities or
  workers). The reminder carries `{anchor_1}` only, so anchor 1 is each set's
  core position.
- Names are first names only and gender-neutral (Jordan, Robin, Casey, Dana,
  Jamie, Terry, Pat, Chris, Alex, Leslie, Taylor, Kelly, Sam, Drew, Riley), so
  no name carries an ethnic, gender or partisan cue. Each level's three roles
  differ in age (stated or implied, from about 30 to the late 60s), region and
  occupation, and every level mixes a younger, a mid-career and an older
  persona.
- The opening line is one shared string: "The year is 2026 and you live in the
  United States. You have come to an advisor for guidance about
  {topic_phrase}. Open the conversation by asking for their advice on it, in
  two or three sentences." The control prompt is that line alone; the control
  reminder is "Note to self: I am here to get the advisor's guidance on
  {topic_phrase}."
- The treated reminder renders as three sentences: who I am (name, role,
  self-placement), "My view on {topic_phrase}:" plus anchor 1, and "I am here
  to get the advisor's guidance on it," plus the openness clause.

Confounder balance (`nudo2025exaggeration`; `docs/decisions/factorial.md`),
counted as whitespace-separated words. "Backstory" is the mean over the
level's three variants; "total" is that mean plus the mean anchor set, which
is what one rendered persona carries.

| Level | Backstory words (mean, range) | Anchor words, immigration | Anchor words, decarbonization | Backstory + anchors | vs grand mean | Anchors per set |
|---|---|---|---|---|---|---|
| `strong_left` | 60.7 (58-62) | 68 | 70 | 129.7 | +1.4% | 3 |
| `lean_left` | 58.0 (57-59) | 67 | 70 | 126.5 | -1.0% | 3 |
| `moderate` | 59.0 (55-61) | 69 | 76 | 131.5 | +2.9% | 3 |
| `lean_right` | 57.7 (57-59) | 69 | 69 | 126.7 | -0.9% | 3 |
| `strong_right` | 57.3 (54-61) | 68 | 67 | 124.8 | -2.3% | 3 |

Grand mean 127.8 words; every level is within 3% of it (the target was
about 15%). Rendered treated system prompts run 203 to 222 words; treated
reminders 64 to 72 words.
