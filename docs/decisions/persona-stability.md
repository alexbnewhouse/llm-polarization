# Decision: persona stability and drift over long dialogues

Status: **decided** 2026-09-10 (Notion task "DECIDE: how to handle persona
stability / drift over long dialogues"). Research pass behind it:
`docs/design/persona-stability.md`. Citation keys resolve in
`paper/references.bib`.

Four things were open. All four are closed. Three keep the 2026-09-02 default
and change only how the decision is written down, because as originally worded
two of them would not have survived review.

## 1. Delivery mode: reinforced

The seeker's system prompt carries the full narrative persona, and a compact
two-to-three sentence reminder is appended last, before every seeker
generation. The mentor gets nothing. About 100 tokens a turn, 4k over a
40-turn dialogue inside a 32k window; the reminder goes after the history
(`harness/transcript.py`), so the cached prefix stays valid.

**The W4 pilot A/B is a measurement, not a gate.** The earlier rule was "if
the stated-once arm holds mean adherence at or above 0.8 through turn 40,
switch to stated-once." That is a non-inferiority claim and five dyads per arm
cannot support one: at a plausible per-dyad SD of 0.15 at turn 40
(`tosato2025persistent` reports SD above 0.3 on 5-point scales;
`choi2024identity` has drift rate varying by model) the standard error is about
0.067, so an observed 0.82 carries a CI of roughly 0.69 to 0.95. A true 0.82
and a true 0.70 are indistinguishable at that n.

So the pilot can confirm reinforced — a stated-once arm landing at 0.65 settles
it — but cannot license dropping the reminder. Run it anyway, for the drift
curve under both modes (the manipulation-check figure, and the basis for the
threshold in section 3) and as an artifact check on the reminder itself:
leakage, stiltedness, anchor parroting.

Naturalism was the case for stated-once and it is weaker than it looked. The
reminder is invisible to the mentor, so no external-validity claim rides on the
seeker's internal prompt structure. What *can* cost naturalism is the seeker's
output going stilted, and that is measurable.

**Reinforced versus once is a dose question, not a naturalism one.**
`nudo2025exaggeration` and `khetan2026politicsbench` both have richer persona
conditioning raising consistency and polarization together. That makes keeping
stated-once cells in the main grid a dose-response contrast rather than a
curiosity about persona decay, and dose-response is the most direct answer
available to the insufficient-dose null. If anything survives the Oct 19
descope, this is a better use of the compute than a fourth model arm.

## 2. Turn count: 40, which also fixes the arm count at three

`khetan2026politicsbench` has stance commitment still rising across stages, so
20 turns may not have saturated; `li2024instability` means an unreinforced
second half is a faded dose. Forty turns plus reinforced delivery is the
coherent pair.

Forty turns and four arms do not both fit. From `models/RUN_APPROACH.md`:

| Configuration | Days | Fits the 21-day W7–W9 window? |
|---|---|---|
| Four arms, 20 turns | ~19.8 | Yes, barely |
| Four arms, 40 turns | ~39.5 | No |
| Three arms (no glm), 40 turns | ~22.3 | One day over; yes in practice |

So choosing 40 turns is also choosing three arms, and `glm-4.7-flash` is the
cut — which the pre-committed descope in `docs/decisions/model-arm.md` already
names as the first to go. This is decided now rather than at the Oct 19
checkpoint, because the frozen factorial inherits it.

**Twenty turns needs no separate condition.** Neither agent is told the turn
budget: `n_turns` lives only in the driver loop (`harness/dialogue.py`) and
appears in no template or prompt. Nothing about a dialogue depends on its
planned length, so turn 20 of a 40-turn dialogue is a valid 20-turn
observation. Forty strictly dominates 20 for information, and the "drop to 20
if the mentor is flat from turn 20 on" fallback can be evaluated from the
40-turn data without rerunning anything.

## 3. Adherence measurement

Structure as proposed — third-model judge, per-turn scoring in the pilot,
threshold plus a three-consecutive-turns flag — with three corrections.

**Score at the cadence the harness implements, and pre-register that.**
`harness/scorer.py` scores `prompt_to_line` and `line_to_line` on the seeker
and `alignment` on the mentor; `pilot` scope scores every turn, `main` scope
scores seeker turns on a turn-4 cadence plus the dyad's final turn and does not
score the mentor. That sampling is defensible on cost grounds, but the earlier
wording ("every turn, both sides") did not match it, and pre-registering one
plan then running another is what reviewers catch. The pre-analysis plan states
the sampled version.

**Q&A consistency moves to the pilot probe protocol.** As a per-turn score it
is a counterfactual — would this seeker, asked the anchor questions now, give
the anchor answers — and most turns never touch the anchors. Forced into a 0-1
score, an unaddressed anchor either goes missing or scores low, and a low score
there fabricates drift that is not there. Either give it an explicit "not
addressed" category distinct from a score, or run it only as probes in the
pilot (`li2024instability`), which perturb the dialogue and so cannot run in
main waves.

**0.8 is not carried over; it is calibrated.** `li2024instability`'s 95-to-75
is a *rate*, the share of responses judged adherent across turns.
`abdulhai2025consistently`'s metrics are a *continuous per-turn score*.
Different scales, so the number does not transfer. Hand-label 100 to 150 pilot
turns stratified by turn index and ideology arm, report judge agreement as
Krippendorff's alpha or weighted kappa, and set the threshold at the judge score
that best separates the hand labels.

Two additions: the judge exclusion extends from "not the seeker or mentor of
that dialogue" to "not the same model family as the mentor," since the mentor's
stance score is the turn-level DV and a same-family judge is both self-favouring
and likely to share political priors; and two judges score a subsample of that
stance metric so cross-judge agreement is reportable.

**Still to build:** the three-consecutive-turns flag rule exists only in prose
here and in `harness/README.md`. It needs to be code before the pilot, because
its output is what section 4 reports.

## 4. Low-adherence dialogues: keep them

The options were exclusion or a compliance-weighted estimate, offered as
equally defensible. They are not.

**Adherence is a post-treatment variable.** It is measured during and after
treatment and is plausibly caused by it — a mentor that pushes back hard may
itself induce seeker drift. Excluding on it conditions on a post-treatment
outcome and breaks randomisation. That bites hardest where the evidence says
drift will be uneven: `bernardelle2025mapping` and `bernardelle2025shifts`
predict asymmetric dose between the two ideology arms, so a threshold rule
differentially prunes one arm and leaves the arms non-comparable. Exclusion
converts an instrument asymmetry into a treatment-effect artifact.

**Exclusion also discards the observations that answer the paper's main risk.**
The named null risk is insufficient dose. Low-adherence dialogues *are* the
low-dose observations. Keeping them and modelling adherence-by-turn against
mentor stance movement is what lets the results distinguish "the dose faded and
the effect faded with it" from "the null is real."

For the pre-analysis plan:

| Estimand | Sample | Status |
|---|---|---|
| ITT | every dialogue that ran to completion, any adherence | Primary |
| Dose-response | adherence as a continuous time-varying covariate, turn level | Secondary |
| Per-protocol | non-flagged dialogues only | Sensitivity, labelled as potentially biased |

The three-consecutive-turns flag is still computed and reported — by ideology
arm and by delivery mode — and triggers the sensitivity analysis. It is an
instrument statistic, not a delete key. A large flagged rate is a finding about
the instrument.

**The one pre-registerable exclusion is separate: technical incompleteness.**
Generation errors, truncated dialogues, judge failures. The data model already
separates these from low scores (`harness/scorer.py` filters
`finish_reason == "error"`; a failed judge call writes a null score with an
error field). Completeness is "reached the planned final turn with no error
rows," handled as missing data. Refusal cascades are ambiguous — arguably
treatment-caused too — so report refusal rate by arm and treat a differential
rate as a finding rather than dropping silently.

## What this decision changes elsewhere

- `docs/design/research-design.md` — persona stability is no longer an open
  question; three arms, not four.
- `docs/decisions/model-arm.md` — glm-4.7-flash is cut, not merely "first cut
  if time is short."
- `docs/decisions/compute-budget.md` — 40 turns is the chosen row.
- `harness/README.md` — the persona-stability section states these rules.
- The pre-analysis plan (W5) — section 4's table, and the sampled scoring
  cadence from section 3.
