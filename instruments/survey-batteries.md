# Survey instruments

Pre- and post-treatment batteries administered to the **mentor** (the zero-shot
advisor whose movement is the outcome). Adapted from de Jong (2024), the
project's key human-subjects reference (`paper/references.bib`,
key `dejong2024crosspartisan`), plus one item for each treatment topic.

The items the harness administers are `batteries.json` beside it, version
`1.0.0`, `"adapted": true`: 15 items with id, battery, scale and, for the
ideological items, `direction`. The adaptation moved the UK wording to US
referents (the federal government, the United States, US parties) and added
the two topic items; the ids and scales of the 0.1.0 items are unchanged. Every
run records the file's sha256 and item ids in `manifest.json`.

Each item is one prompt with the scale stated in the text, ending "Answer with
a single number."; the reply is constrained to `{"answer": <int>}` on the
item's scale (`harness/survey.py`, `harness/parser.py`). Items are
administered in file order, pre and post: the five de Jong ideological items,
the two topic items, the thermometers, the agreement items.

**Caveat.** The wording has not been piloted on the mentor models. Refusals,
salvaged answers and ceiling or floor effects per item are unknown until the
pilot (gap audit F5).

## Items

`direction` says which pole a high answer means.

| # | id | battery | direction | scale |
|---|---|---|---|---|
| 1 | `ideo_gender_racial_equality` | ideological | left | 1-5 |
| 2 | `ideo_immigration` | ideological | right | 1-5 |
| 3 | `ideo_redistribution` | ideological | left | 1-5 |
| 4 | `ideo_multiculturalism` | ideological | left | 1-5 |
| 5 | `ideo_gun_control` | ideological | left | 1-5 |
| 6 | `ideo_enforcement_militarization` | ideological (topic) | right | 1-5 |
| 7 | `ideo_decarbonization` | ideological (topic) | left | 1-5 |
| 8 | `therm_dem_voters` | thermometer | | 0-10 |
| 9 | `therm_dem_politicians` | thermometer | | 0-10 |
| 10 | `therm_rep_voters` | thermometer | | 0-10 |
| 11 | `therm_rep_politicians` | thermometer | | 0-10 |
| 12 | `therm_independents` | thermometer | | 0-10 |
| 13 | `agree_democracy` | agreement | | 1-5 |
| 14 | `agree_protest_rights` | agreement | | 1-5 |
| 15 | `agree_cross_partisan` | agreement | | 1-5 |

Ideological items are agree/disagree statements (1 strongly disagree, 5
strongly agree); the topic items cover the military or National Guard in
immigration enforcement inside the US, and moving the economy off fossil fuels
faster at some cost. Thermometers run 0 (very cold) to 10 (very warm).
Agreement items cover democracy as the best form of government, protest
rights, and comfort discussing politics across parties.

## Indices

Defined in `batteries.json` → `indices`, machine-readable (`items`, `reverse`,
`reverse_rule`, `combine`). These are the remediation plan's shared
definitions; `analysis/` and the PAP use them as written.

- `ideological`: mean of items 1-7, each recoded so high = right. The five
  left-keyed items are reversed as `min + max - x` (6 - x). Range 1-5.
- `therm_gap`: mean(`therm_rep_voters`, `therm_rep_politicians`) minus
  mean(`therm_dem_voters`, `therm_dem_politicians`). Signed, positive = warmer
  to Republicans. Range -10 to 10.
- `affective_abs`: |`therm_gap`|, the affective polarization measure. Range 0-10.
- `norms`: mean of the three agreement items, none reversed. Range 1-5.

`therm_independents` is administered but enters no index. Missing answers and
salvaged rows are handled in `analysis/` and `docs/pap/`, not here.

The primary outcome is post minus pre on `ideological`, positive = rightward.

## Logging

Each answer is one row of `data/<run_id>/surveys.jsonl`, with the dyad,
attempt, phase, item, instrument hash, mentor model and template hash, the
prompt hash, the parsed answer and the raw reply. The condition is on the
matching `dyads.jsonl` row and the template source in `manifest.json`. Field by
field: `data/README.md`.
