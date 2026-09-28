# Survey instruments

Pre- and post-treatment batteries administered to the **mentor** (the zero-shot
advisor whose movement is the outcome). Adapted from de Jong (2024), the
project's key human-subjects reference (`paper/references.bib`,
key `dejong2024crosspartisan`). The UK wording has **not yet been adapted to the
US context**; that is a W2/W3 task in the Notion plan.

This page lists the categories. The items the harness actually administers are
`batteries.json` beside it: 13 items with ids, battery and scale, currently
placeholder wording (`"adapted": false`). The adaptation replaces the `text`
of each item and bumps `version`; it keeps the ids and scales. Every run
records the file's sha256 and item ids in `manifest.json`, and items are
administered in file order, pre and post.

## Ideological polarization (5-point scale)

- Gender and racial equality
- Immigration policy
- Income redistribution
- Multiculturalism
- Gun control

## Affective polarization

Feeling thermometer, 0 to 10:

- Democratic voters, Democratic politicians
- Republican voters, Republican politicians
- Independent voters

Agreement statements, 5-point scale:

- Beliefs about democracy
- Protest rights
- Comfort with cross-partisan discussion

## Discussion topics (treatment content)

- Militarization of immigration enforcement
- Decarbonization of the economy

## Logging

Each answer is one row of `data/<run_id>/surveys.jsonl`, with the dyad,
attempt, phase, item, instrument hash, mentor model and template hash, the
prompt hash, the parsed answer and the raw reply. The condition is on the
matching `dyads.jsonl` row and the template source in `manifest.json`. Field by
field: `data/README.md`.
