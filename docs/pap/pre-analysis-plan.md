# Pre-analysis plan

Status: **draft for registration**, written 2026-09-28 (W6). It closes gap-audit F4, and it writes down the
rules for F9 (descope), F10 (stance transform, control) and F11 (baseline). The runway put registration
in W5, which has passed. The honest window left is before wave 1 starts on 2026-10-05, before any wave
data exists. Registration mechanics and the power tables are in `docs/pap/README.md`.

Every number marked **[pilot]** comes from the pilot record and is filled in before registration. A
[pilot] value changed after registration is a deviation (section 14). Citation keys resolve in
`paper/references.bib`. This plan inherits `docs/decisions/factorial.md`, `persona-stability.md`,
`model-arm.md` and `compute-budget.md`, and the outcome definitions in
`docs/audit/2026-09-28-remediation-plan.md` ("Shared definitions"). Section 16 lists the places where
this plan had to settle a disagreement between those records.

## 1. Hypotheses

**H1, ideology dose-response (confirmatory).** The mentor's post-dialogue ideological position moves in
the direction of the seeker's persona, monotonically in the persona's intensity. Test: the slope of the
index change on ideology dose (section 3.5) is non-zero, per topic, pooled over arms. Predicted sign
positive (rightward dose, rightward movement). The test is two-sided.

**H2, the accommodation shape test (confirmatory; `factorial.md`, consequence 3).**
`tornberg2026audits` finds that models accommodate the interlocutor they infer, with about an 8x
asymmetry: conservative-coded askers move responses far more than progressive-coded ones. Accommodation
therefore predicts large movement at the two right levels, small movement at the two left levels, and
near zero in the bare control. Genuine dialogue-driven drift predicts movement that is roughly symmetric
in ideology and monotone in intensity. Test: the asymmetry contrast A (section 7.3) is non-zero, per
topic. Either result is a finding, so the plan classifies the outcome rather than only rejecting a null.

**H3, adherence dose-response (secondary; `persona-stability.md` §4).** Within treated dyads, the mentor
agrees more with the persona at turns where the seeker holds the persona more closely, and a dyad's
index change per unit of ideology dose is larger when its mean adherence is higher. Adherence is
post-treatment, so H3 is an association within randomized cells, not a randomized contrast (section 4).

Other secondary hypotheses, all two-sided and in the secondary family (section 8):

| | Hypothesis |
|---|---|
| S1 | The dose slope differs by mentor arm (arm x dose). Olmo carries the training-data claim (`model-arm.md`) |
| S2 | Affective polarization (`affective_abs`) rises with intensity \|dose\| relative to the control |
| S3 | The signed thermometer gap (`therm_gap`) follows dose |
| S4 | The norms index moves with \|dose\|. No direction is predicted |
| S5 | Openness moderates the dose slope (openness x dose) |
| S6 | The dialogue's own topic item moves with dose (section 3.1; the index dilutes it) |

## 2. Design

The frozen grid (`docs/decisions/factorial.md`, `prompts/grid.json`) runs identically in every mentor
arm.

| Element | Value |
|---|---|
| Treated cells | 2 topics (`immigration_enforcement`, `decarbonization`) x 5 ideology levels x 2 openness (`open`, `closed`) = 20 |
| Role | 3 congruent persona variants per ideology level, nested; a random effect. 15 per topic, the same slugs in both topics |
| Control | 1 bare cell per topic: `ideology: none`, `openness: null`, `role: null`. It carries the advisor framing only |
| N | 135 per treated cell (45 per variant) and 135 per control cell: 2,970 per arm |
| Arms | qwen3.6:35b-a3b, gpt-oss:20b, Olmo-3-7B-Instruct. Model is a run-level factor, one run per arm: 8,910 dialogues |
| Turns | 40. Neither agent is told the budget, so turn 20 is a valid 20-turn observation (exploratory only) |
| Delivery | reinforced: the persona in the seeker's system prompt, plus a trailing reminder on every seeker turn. The control gets a framing-only reminder with the same mechanics. Nothing on the mentor side |
| Seeker | one model for every arm, from a family outside all three arms, chosen by pilot adherence and throughput **[pilot]** |
| Sampling | dialogue: temperature 0.7, top_p 0.95, n_predict 300. Survey and judge: temperature 0. `run_seed` and `now` fixed for the study |
| Assignment | one wave manifest from `harness.randomize` (grid x catalogue x seed), shared by all three arms (section 11) |
| Survey | `instruments/batteries.json` at its registered, adapted version (1.0.0, 15 items, 2026-09-28): pre with an empty context, post appended as a user turn after the 40-turn dialogue |

Arm is not randomized per dyad. Every arm runs the same manifest, with the same seeds, persona texts and
seeker, on the same machine. Arm contrasts are therefore comparisons of models under an identical
protocol, not randomized treatment effects, and waves differ in calendar date.

**The post-survey measures in-context accommodation.** The post items are asked inside the dialogue's
context, so they measure how the mentor answers the interlocutor in front of it, not a
context-independent position. That is the sycophancy confound the design names. The bare control is the
only thing that separates dialogue-driven movement from mirroring. The results section says so wherever
it reports a post-survey number (gap-audit F11).

## 3. Outcomes

As the remediation plan's Shared definitions. W2 (`instruments/`) and W3 (`analysis/`) implement the
same definitions.

**3.1 Ideological index.** The mean of the five de Jong items and the two topic items,
`ideo_enforcement_militarization` (direction `right`) and `ideo_decarbonization` (direction `left`). Each
item is recoded so that high = right, using its `direction` field in the registered `batteries.json`
(`x' = min + max - x` for a `left` item), on the original 1-5 scale. S6 uses the dialogue's own topic
item alone, recoded the same way.

**3.2 Affective.** `therm_gap = mean(therm_rep_voters, therm_rep_politicians) - mean(therm_dem_voters,
therm_dem_politicians)`, signed on the 0-10 scale; positive means warmer toward Republicans.
`affective_abs = |therm_gap|` is affective polarization. `therm_independents` enters no index and is
reported descriptively.

**3.3 Norms.** The mean of `agree_democracy`, `agree_protest_rights` and `agree_cross_partisan` (1-5).

**3.4 Change scores.** Each outcome is post minus pre for the dyad, from the latest complete attempt.
The primary outcome is the change in the ideological index; positive means rightward. `affective_abs`
changes as |post gap| - |pre gap|. Pre is the harness's pre-survey as administered (section 10).

**3.5 Dose.** `strong_left` -2, `lean_left` -1, `moderate` 0, `lean_right` +1, `strong_right` +2. The
control is coded dose 0 with the treated indicator T = 0. `moderate` has dose 0 and T = 1, so
`moderate` is not the control (`factorial.md`, consequence 1).

**3.6 Signed stance.** The judge's `alignment` scores the mentor's line on 0-1 against the persona's
position; 0.5 is neutral or balanced. The Shared definition is `alignment` times the sign of the
persona's dose. It is applied to the centred score, because 0.5 is the zero point:

    stance = sign(dose) x (2 x alignment - 1),  in [-1, 1],  positive = rightward

Without centring, a mentor that fully opposes a left persona and one that fully opposes a right persona
would both score 0. `stance` is defined for the four directional levels. It is **undefined for the
control**, which states no position (F10). Any control rows that are scored are reported separately as
raw `alignment` and never enter a model. It is also undefined for `moderate`, since sign(0) = 0:
moderate's `2 x alignment - 1` is agreement with the centrist anchors, which have no left-right sign. It
enters H3 as agreement and never the signed trajectory. The moderate-anchor rule (`factorial.md`,
consequence 2) is what makes that agreement scoreable.

**3.7 Adherence.** The seeker's `prompt_to_line` (primary) and `line_to_line` (secondary), 0-1. Treated
dyads only: adherence does not apply to the control (`factorial.md`; gap-audit F10). Control scores, if
the harness writes them, are dropped by the analysis and counted.

## 4. Estimands

From `persona-stability.md` §4. "Dose-response" there means adherence dose-response. The ideology
dose-response of H1 is a randomized, ITT estimand.

| Estimand | Sample | Contrast | Status |
|---|---|---|---|
| ITT, per level | every dyad whose latest attempt is complete, at any adherence | each ideology level vs the control, per topic, pooled over arms and averaged over openness | Primary |
| ITT, ideology dose slope (H1) and asymmetry A (H2) | as above | section 7 | Primary, confirmatory |
| Adherence dose-response (H3) | treated dyads, scored turns | adherence as a continuous, time-varying covariate at the turn level; mean adherence x dose at the dyad level | Secondary. Adherence is post-treatment, so this is an association |
| Per-protocol | non-flagged dyads only (section 5) | the ITT models re-fit | Sensitivity, labelled as potentially biased |

## 5. Scoring cadence

As `harness/scorer.py` implements it. The pre-registration states the sampled cadence, not "every turn,
both sides" (`persona-stability.md` §3).

| Scope | Agent, metric | Turns | Calls per 40-turn dyad | Run on |
|---|---|---|---|---|
| `main` | seeker: `prompt_to_line`, `line_to_line` | t mod 4 = 0, plus the dyad's final turn (4, 8, ..., 40) | 20 | every treated dyad, primary judge |
| `stance` | mentor: `alignment` | the same cadence | 10 | every treated dyad, primary judge; a subsample of f **[pilot]** by the second judge (section 12) |
| `pilot` | both agents, all metrics | every turn | about 120 | pilot and calibration only; never mixed into wave analyses |

The final turn is the highest non-error turn of that agent in that attempt, which is 40 for a complete
dyad. Rows with `finish_reason == "error"` are never scored.

**Flag rule** (`harness.scorer.flag_dialogues`). A treated dyad is flagged when `prompt_to_line` is
strictly under the threshold tau **[pilot]** on 3 consecutive *scored* `main`-scope seeker turns, which
spans at least 8 dialogue turns. A null score is an unscored turn: it neither extends nor breaks a run.
tau is the judge score that best separates 100 to 150 hand-labelled pilot turns, stratified by turn and
ideology, with judge-human agreement reported as Krippendorff's alpha or weighted kappa **[pilot]**. The
rule is applied to `main`-scope targets only, never mixed with `pilot`-scope rows (red-team M9). It is
reported by ideology level (and by delivery mode if E1 runs) and triggers the per-protocol sensitivity.
It is an instrument statistic, not a delete key.

If judge compute (gap-audit F7) forces a cut, `stance` is scored on a deterministic subsample of treated
dyads (`subsample_dyads`, the manifest's `run_seed`) at a fraction fixed before scoring **[pilot]**.
`main` is never subsampled.

## 6. Exclusions and missing data

**Unit.** The latest attempt per `dyad_id`, if it is `complete` (`status.jsonl`;
`latest_complete_attempts`). The number of dyads with `attempt > 1` is reported by condition and tested
against it.

**The only exclusion is technical incompleteness.** That is a dyad with no complete attempt, any turn
row with `finish_reason == "error"`, a truncated dialogue (`truncated = true`, a slot out of context), or
fewer than `n_turns` turns. It is handled as missing data. Counts are reported by arm x cell, with a test
of incompleteness against ideology level per arm. If incompleteness differs by level (p < 0.05), Lee
bounds on the level contrasts are reported beside the ITT estimates.

**Not exclusions.** The adherence flag; `finish_reason == "length"` (the 300-token cap, by design);
`cache_warning`; `attempt > 1`; refusals.

**Refusals.** Mentor refusals to answer a survey item, and refusal cascades in the dialogue, are
detected by the W3 refusal detector, validated on pilot transcripts **[pilot]**. The refusal rate is
reported by arm x topic x ideology level. A differential rate across levels is a finding, reported in
the main text, and is never dropped silently. Sensitivity: refused items imputed at the scale midpoint,
then at each endpoint (bounds).

**Survey items.** A null answer is a missing item. An index is missing for a dyad when any of its items
is missing (primary). Sensitivity: the mean of the available items when at least 5 of 7 ideological
items (2 of 3 norms items) are present. The counts of null answers and salvaged answers
(`answer_method` other than `json`) are reported.

**Judge rows.** An error row is retried on the next `score` run. A persistent error or a null score is
an unscored turn: missing in the turn-level models, and counted.

## 7. Models

The primary estimator is a linear mixed model fit by REML, with Satterthwaite degrees of freedom
(Kenward-Roger where available). There is one row per dyad for the survey outcomes. The role random
intercept u_role is nested in ideology; slugs are unique within a level, so the nesting is implicit. It
applies to treated dyads only, because the control prompt is fixed, not sampled. Openness is coded +1/2
for `open` and -1/2 for `closed` among treated dyads, and 0 for the control, so every level contrast
averages over openness.

**7.1 Per topic (confirmatory), pooled over arms.**

    level model:  y_i = alpha_arm + sum_l tau_l 1[level_i = l] + omega open_i + u_role + e_i
    slope model:  y_i = alpha_arm + gamma T_i + beta dose_i + omega open_i + u_role + e_i

tau_l is the ITT of level l against the control. beta (H1) is the ideology dose slope. gamma is
`moderate` minus the control, the effect of persona presence without direction. Arm enters as a fixed
factor. S1 adds arm x dose and arm x T (arm x level in the level model) and reports per-arm slopes and
contrasts.

**7.2 Pooled.** Both topics and all arms, with topic and arm fixed effects, topic x (T, dose), and a role
random intercept plus role x topic. It reports the average slope and level contrasts as the summary
estimate. It is not a separate confirmatory test.

**7.3 Shape test (H2).** From the level model per topic: R = (tau_+1 + tau_+2)/2 and
L = (tau_-1 + tau_-2)/2.

- Asymmetry A = R + L. Symmetric drift gives A = 0. Right-heavy accommodation gives A > 0.
- The ratio rho = R / (-L), with a Fieller 95% CI from the model's covariance.
- Classification, stated in the abstract whichever way it falls:
  - **Accommodation-consistent**: A > 0 (Holm-adjusted) and the CI for rho excludes 1.
  - **Symmetric-drift-consistent**: A not significant, the CI for rho includes 1 and excludes 8 (the
    `tornberg2026audits` asymmetry), and both intensity contrasts are non-negative:
    tau_+2 - tau_+1 >= 0 and tau_-1 - tau_-2 >= 0, with point estimates and 95% CIs.
  - **Inconclusive**: anything else.
- The control's own change (post minus the no-dialogue pre) is reported per arm with its CI, as the
  "dialogue per se" contrast.

**7.4 Adherence dose-response (H3).** At the turn level, over treated dyads and `stance`-scored turns:

    agree_it = 2 x alignment_it - 1
             = theta adh_it + turn_t + cell_c + alpha_arm + u_role + v_dyad + e_it

adh_it is the seeker's `prompt_to_line` on the same scored turn; the seeker speaks first in a turn. turn
is a factor, and cell is topic x ideology x openness. v_dyad is the dyad random intercept, which does
the dyad clustering. Robustness uses cumulative mean adherence to turn t. At the dyad level, over
treated dyads: the slope model of 7.1, plus mean adherence and dose x mean adherence. The signed-stance
trajectory (`stance` by dose x turn, directional levels only) is exploratory. It includes turn 20
against turn 40.

**7.5 Robustness, pre-specified.**

- (a) An item-level stacked model: item fixed effects, a dyad random intercept, and the same fixed
  terms.
- (b) A two-stage role-means analysis: the change is averaged to role x arm x topic, then level
  contrasts are fit on the role means. This is exact for the nested design.
- (c) A residual variance that differs by arm.
- (d) ANCOVA, with post on pre, if section 10 option B is adopted.

OLS with CR1 standard errors clustered by role is **not** used for inference. With 15 role clusters,
and 3 behind any one level, its size at a nominal 0.05 was 0.08 to 0.15 in the power simulation
(`analysis/power.py`).

**7.6 Extension E1** (only if triggered at the Oct 19 checkpoint). This is qwen only: delivery mode as a
within-arm factor, and mode x dose on the slope model. It is exploratory, and its manifest is a separate
randomization of new rows (section 11).

## 8. Multiple comparisons

| Family | Tests | Control |
|---|---|---|
| Confirmatory H1 | beta per topic (2) | Holm, FWER 0.05 |
| Confirmatory H2 | A per topic (2) | Holm, FWER 0.05 |
| Primary ITT per level | tau_l, 5 per topic | Holm within topic; CIs reported unadjusted |
| Secondary | H3 (theta; dose x adherence), S1-S6, per topic | Benjamini-Hochberg, q = 0.05, across the family |
| Exploratory | per-arm contrasts, trajectories, turn 20, E1, `therm_independents` | unadjusted, labelled |

All tests are two-sided. Effects are reported with 95% CIs in index points and as a standardized effect
(divided by the post-survey index SD in the control).

## 9. Power

`analysis/power.py` simulates the grid and fits the level and slope contrasts, using the nested-ANOVA
form of the model in 7.1. The model reduces to that form on balanced data, and its test size is
nominal. Inputs: the within-cell post-survey index SD **[pilot]**, and the role share of variance
icc_role **[pilot]**. The tables are in `docs/pap/README.md`.

In short, at SD 0.6 and icc_role 0.02, per topic pooled over arms, the MDE at 80% power is:

- the slope: 0.052 index points per dose step at N = 135 (0.053 at N = 90);
- strong_right vs control: 0.18 points (0.19 at N = 90).

Role variance, not N, binds the slope. Three variants per level are the replicates that a
between-persona difference has to beat. The index is a mean of seven items, so a movement confined to
the dialogue's own topic item shows up in the index at one seventh of its size. That is why S6 is
registered.

## 10. The baseline question (F11, a decision)

As built, the pre-survey is `[user: item]` in an empty context at temperature 0 (`harness/survey.py`).
Every dyad in an arm gets the same greedy answer, up to batching noise. The post is also greedy, but
conditioned on the dialogue. research-design.md's "1,000-administration baseline" has no procedure, and
at T = 0 it would be 1,000 copies.

**Option A: keep the pre greedy and context-free.** The pre is one constant per arm and item.
- The change score is post minus an arm constant, so every between-cell contrast is identical to a
  post-only contrast. "Pre/post" describes the sign convention and the arm's starting point, not a
  within-dyad baseline.
- It costs nothing and changes no code. Pre and post are measured the same way: both are the greedy
  answer, one without context and one with it.
- The control's change is interpretable as the effect of dialogue per se, against a single-point
  reference.
- What it gives up: no within-dyad adjustment. That adjustment would be worthless anyway, because the
  mentor has no dyad-specific state before the dialogue. It has the same weights and the same empty
  context every time.
- It gives no distribution of the mentor's baseline answers either.

**Option B: sample the pre at the dialogue temperature (0.7) with the dyad's seed.**
- The pre becomes a per-dyad draw from the mentor's zero-shot answer distribution.
- It is drawn in a different context, with a different seed, from the post. So pre and post are close
  to independent within a cell (rho near 0), and a change score adds the pre's variance to the outcome.
  In `analysis/power.py --pre-rho 0` at SD 0.6, the strong_right vs control MDE at N = 135 rises from
  0.104 to 0.147 points at icc_role 0, and from 0.18 to 0.21 at 0.02. ANCOVA recovers the loss only if
  rho > 0, and here it is not.
- Pre and post are measured differently (a sample against a mode). Their difference therefore carries a
  sampling-versus-greedy artifact. The artifact is constant per arm and cancels between cells, but not
  in the control's absolute change. Removing it means sampling the post as well, and that changes the
  instrument after the pilot.
- It is a harness change (`survey.py`, owned by W0/W6), and a resume would have to refuse the mix.

**Option C, the no-dialogue baseline run.** This is separate from both. Per arm, a run with no dialogue
that administers the pre battery K times at the dialogue sampling settings (temperature 0.7, top_p
0.95), one derived seed per administration. K = 1,000 per arm is proposed; research-design.md counts
1,000 in total. It gives:
- the mentor's zero-shot answer distribution per item and index;
- the scale for "how large is a movement" (baseline SD);
- a check that the greedy pre is the mode of that distribution.

The cost is 15 short completions per administration (the 1.0.0 instrument), about 45,000 calls for three
arms. That is minutes to an hour of compute, and nothing against 8,910 dialogues. The code path is
`python -m harness.run baseline --k 1000` (`harness/README.md`). It is a new `run_id` per arm with the
same `run_seed`, `now`, instrument and mentor build as that arm's wave; the study lock refuses another
`run_seed`, `now` or instrument.

**Recommendation: A plus C.**
- Keep the in-run pre greedy and context-free, which leaves the instrument as piloted.
- State in the paper that the primary contrasts are between cells, and that the change score equals
  post minus an arm constant.
- Run the baseline as C before wave 1 (or beside it on the same build), and report the share of dyads
  whose pre differs from the arm's modal pre as the batching-noise check.

B buys a distribution that C gives more cleanly, and pays for it in power, in measurement comparability
and in an instrument change after the pilot. Until the PI decides, the analysis code treats pre as "as
administered" (Shared definitions), which is correct under either option.

## 11. The descope rule (F9)

The Oct 19 cut list (`factorial.md`) is:

1. N per cell 135 -> 90;
2. collapse ideology to three levels;
3. drop a topic.

Olmo and the control are never cut. A cut **subsets the existing wave manifest by `dyad_id`. It never
re-randomizes.** Re-running `harness.randomize` with `--n-per-cell 90` gives dyad_ids that are a subset
of the 135 manifest, but with different seeds (30 of 1,980 matched in the audit's probe). A "descoped"
arm would then not be the same dyads as an arm already run.

- **One wave manifest for all three arms**, written once at N = 135, committed with its
  `-assignment.json`, and its sha256 registered. The `--prefix` is immaterial; the arms differ by
  `run_id`, not by manifest.
- **Cut 1:** keep a treated row when its within-variant index k, the last three digits of the
  `dyad_id`, is at most 30, and a control row when k is at most 90. k is assigned in grid order before
  the shuffle and is independent of seeds and outcomes, so this is a fixed, pre-registered subset.
- **Cut 2:** keep `ideology` in {`strong_left`, `moderate`, `strong_right`, `none`}.
- **Cut 3:** keep one topic's rows. The topic is chosen by the pilot-transcript criterion in
  `factorial.md`, never by wave outcomes.
- The cut is written as a derived manifest: the parent's lines copied byte for byte, in order, with the
  parent's sha256 and the filter recorded beside it. Arms that have already run keep their full data.
  The confirmatory models use the subset common to all arms; an arm's full data is a sensitivity.
- **All arms share** `run_seed`, `now`, the parent manifest sha256, the instrument sha256, the seeker
  model sha256, the generation settings and the judge (model sha256 and judge prompt sha256).
  - The study lock (`study.json`; `harness/README.md`, "The study lock") makes `check`, `run`,
    `baseline` and `score` refuse another `run_seed`, `now`, manifest (or a subset not cut from it),
    instrument, grid or judge model.
  - The seeker model sha256, the generation settings and the judge prompt sha256 are not in the lock.
    They are checked by hand across the arms' `manifest.json` and `judge-*.json` files and recorded in
    the appendix (`REPRODUCIBILITY.md` §6).
- E1 is an extension that adds new rows, not a cut, so its own `harness.randomize` manifest is allowed.

## 12. The judge rule

- **One primary judge, fixed study-wide.** It comes from a model family outside all three arms (not
  qwen, gpt-oss or olmo). It is not the seeker model, and it is outside the seeker's family where the
  candidates allow **[pilot]**. It is pinned by model sha256, the judge prompt (`JUDGE_SYSTEM`,
  `JUDGE_TASKS` at the registered commit), temperature 0 and n_predict 160. The harness refuses a judge
  from the family of the run's own mentor. The study rule is stricter, because a judge refused for one
  arm and accepted for the others is a judge-by-arm confound (F9).
  `harness/config.example.json` points the judge at `:8099`, the resident qwen tier, which this rule
  excludes.
- **A second judge** comes from another family outside the three arms, different from the primary's. It
  scores `stance --subsample f` **[pilot; 0.10 proposed]** on treated dyads. The subsample uses the
  manifest's `run_seed`, so both judges score the same dyads (red-team M8a).
- **Agreement.** The headline is Krippendorff's alpha (interval) on the targets both judges scored.
  Beside it goes the harness `agreement` output: n, mean absolute difference, Pearson r and the share
  within 0.1. If alpha < 0.667, the turn-level stance results (H3's turn model and the trajectories) are
  reported as exploratory. The survey outcomes do not depend on the judge. H3 is re-fit on the
  subsample with each judge.
- **The judge is not blind to condition.** `alignment` needs the persona, which is a declared
  limitation (F10). It scores the mentor's line with the reasoning taken out: `<think>` blocks go to
  the row's `reasoning` field, and a reply with harmony channel markup fails the dyad instead of reaching
  the judge (red-team H1, W0).

## 13. What the appendix reports

- The `docs/REPRODUCIBILITY.md` §6 checklist, item by item. That includes the `cache_warning` rate, the
  `truncated` rate, `attempt > 1` by condition, null and salvaged survey answers,
  `finish_reason == "length"` turns, judge null scores, the `run_seed`, `now`, the build and the GPU,
  and the reproduce-one-dialogue procedure (§4).
- Also reported:
  - incompleteness by arm x cell, with the test against condition;
  - refusal rates by arm x topic x level;
  - the flag rate by ideology level;
  - judge-human and judge-judge agreement;
  - the baseline distribution and the modal-pre check (section 10);
  - the per-protocol results;
  - the cross-arm identity check (section 11), any descope with its derived-manifest record, and the
    deviations log.

## 14. Deviations

- **Recording.** Any change after registration goes into `docs/pap/deviations.md`, which is created at
  the first deviation. Each entry records the date, the section, the change, why, and whether any wave
  outcome data (surveys or scores) had been seen. Where it is feasible, both the registered and the
  deviated analysis are reported.
- **What counts as a deviation.**
  - Changing a [pilot] value after registration is one.
  - A harness fix that changes what a model is sent, or how a row is scored, is one, and it needs a new
    `run_id`.
  - A fix that changes neither is recorded through `harness_commit` and is not a deviation.
- **A cut** under section 11 follows a registered rule and is recorded, not deviated.

## 15. Frozen at registration

This file, `prompts/grid.json`, `prompts/personas/catalogue.json`, `instruments/batteries.json`, the wave
manifest and its assignment log, the harness commit, the seeker and both judges, tau, and the
`analysis/power.py` inputs and output. The [pilot] values to fill are:
- the seeker;
- the judges;
- tau and its hand-label agreement;
- the post-survey index SD and icc_role;
- the second-judge fraction f, and a `stance` subsample fraction if compute forces one;
- the refusal-detector validation;
- the recomputed wave budget.

## 16. Places this plan settles a disagreement

1. **Stance transform.** The Shared definition, `alignment` x sign(dose), is not centred at 0.5, and it
   is zero for `moderate`. This plan centres it and leaves `moderate` unsigned (3.6).
   `factorial.md`'s "alignment on all five levels" is kept as agreement, not as signed stance.
2. **"Dose-response" names two estimands.** In `factorial.md` it is the randomized ideology slope (H1).
   In `persona-stability.md` §4 it is the adherence covariate (H3). This plan renames the second
   "adherence dose-response".
3. **Stance coverage.** `persona-stability.md` §3 says `main` does not score the mentor, and
   `scorer.py` and `harness/README.md` describe `stance` as a two-judge subsample. H3 still needs
   turn-level mentor stance on every treated dyad. So the primary judge scores `stance` on all treated
   dyads, and the second judge scores a subsample (5, 12).
4. **Manifest per wave or per study.** The `randomize.py --prefix` help ("w1 for wave 1") reads as one
   manifest per wave. F9 needs one manifest for all arms, and this plan takes that (11).
5. **Judge family.** The per-run harness check and the example config's qwen judge are weaker than the
   study rule. The study rule governs (12).
6. **The 1,000-administration baseline** (`research-design.md`) is undefined and degenerate at T = 0.
   It is defined as run C, per arm (10).
7. **Control size under cut 1.** The control is "sized at one treated cell per topic", so it goes to 90
   with the treated cells (11).
