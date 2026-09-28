# analysis/

From a run directory to the registered estimates: exclusions, outcomes, models, appendix rates, judge
calibration and the reproduce-one-dialogue check. The plan it implements is
`docs/pap/pre-analysis-plan.md`; the data it reads is described field by field in `data/README.md`.
Reporting vocabulary for stance movement follows `hao2026flips`: spontaneous
instability, conformity, persuasion.

Pure python plus numpy. Every module is a CLI run from the repository root, and each writes to
`data/<run_id>/analysis/` (git-ignored with the rows) unless `--out` says otherwise. One run directory is
one mentor arm; `estimate` and `rates` take `--run-dir` once per arm.

```bash
python -m analysis.load     --run-dir data/wave1-qwen                  # exclusion log, tidy dyad table
python -m analysis.outcomes --run-dir data/wave1-qwen                  # indices and change scores
python -m analysis.outcomes --run-dir data/wave1-qwen --baseline data/baseline-qwen   # + baseline mean, SD
python -m analysis.estimate --run-dir data/wave1-qwen --run-dir data/wave1-oss --run-dir data/wave1-olmo
python -m analysis.rates    --run-dir data/wave1-qwen --run-dir data/wave1-oss --by topic_ideology
python -m analysis.calibrate sample --run-dir data/pilot-2026-09-18 --n 120
python -m analysis.calibrate score  --run-dir data/pilot-2026-09-18 --labels labels.csv
python -m analysis.verify_dialogue --run-dir data/wave1-qwen --dyad-id <id>     # or --all
python -m analysis.synth --out /tmp/data --run-id synthetic                     # a dry-run dataset
```

Flags come from `flags.jsonl` (written by `python -m harness.run flags`; rows of another `--scope` are
ignored), or from `--threshold T` on `load`, `estimate` and `rates`, which applies
`harness.scorer.flag_dialogues` to the main-scope score rows.

Which rows are analysed:

- **Survey rows.** The pass named by `--origin` (default `run`) at the instrument's settings: the JSON
  schema, temperature 0, 32 tokens. `--survey-temperature` and `--survey-n-predict` pick a pass made with
  `survey --temperature/--n-predict` instead. Rows of that origin at other settings are never mixed in:
  they are set aside and logged as `survey_settings`. The unconstrained check (`survey --no-schema
  --sample N`, rows with `schema: false`) never enters the outcome tables; `rates` compares it with the
  constrained answers, and `--include-unconstrained` analyses it in place of the constrained pass (only
  the sampled dyads then have outcomes). A row with `truncated: true` or answer_method `truncated` is a
  missing answer (`missing_survey`).
- **Score rows.** Each row's `scope` says which `score` pass wrote it. Adherence, `judge_failure` and the
  flag rule read `main` rows; the mentor stance analyses (H3, trajectory) read `stance` rows; the
  judge-agreement statistic pairs the primary stance judge with the second judge's rows marked
  `subsample`. `pilot` rows (every turn, for calibration) feed none of these; the control's raw
  alignment, which only `pilot` scores now, is reported apart. A row without a scope (written before it
  was recorded) is `main` for a seeker row and `stance` for a mentor row on the main cadence (turns 4, 8,
  ... and the final turn), `pilot` otherwise.
- **Baseline rows.** `data/<run_id>/baseline.jsonl` from `harness.run baseline` (PAP §10 option C), one
  run directory per arm: `load.load_baseline` and `load.baseline_distributions` give the answer
  distribution per arm and item, `outcomes.baseline_reference` the per-arm mean and SD of each index.

## Modules

| Module | What it does |
|---|---|
| `load.py` | Reads a run directory. Keeps the latest complete attempt per dyad (`harness.scorer.latest_complete_attempts`, the harness's own rule) and drops every other attempt's rows. Selects the survey pass and the score scopes (above). Applies the technical exclusions, each with a scope: `not_run`, `incomplete`, `error_rows`, `short_dialogue`, `truncated` (dyad: out of the ITT sample); `missing_survey` (one index); `survey_settings` (rows set aside, counted); `judge_failure` (one turn, counted); `unscored` (out of the adherence sample). Attaches mean seeker adherence and the flag. Writes `exclusions.md/.json/.jsonl` and `dyads.csv`, with counts by reason x ideology and by cell and a chi-square test of incompleteness against ideology. Accessors: `unconstrained_surveys`, `constrained_answers`, `load_baseline`, `baseline_distributions` |
| `outcomes.py` | Reads the instrument the run recorded (`manifest.json` `batteries.path`), checks its sha256, and falls back to the matching version in git history. Index definitions come from its `indices` block, else from the remediation plan's Shared definitions. Recodes by `direction`. Builds `ideological`, `therm_gap`, `affective_abs` and `norms`, plus `topic_item` (the dialogue's own topic item, PAP S6). Change = post - pre. `--json-only`, `--available-items` and `--impute-refusals mid/low/high` are the PAP §6 sensitivities. `baseline_reference` (`--baseline`, once per arm) gives each index's mean and SD over the no-dialogue baseline's administrations, written to `baseline_reference.json` and `outcomes.md` |
| `estimate.py` | The PAP §7 models, pooled over arms with arm fixed effects: level model (ITT per level against the bare control), slope model (H1), shape test and its classification (H2), H3 at the turn level (stance-scope alignment on main-scope adherence) and the dyad level, judge agreement on the second judge's subsample, secondary tests S1-S6, per-protocol, and the §7.5 robustness checks. Holm and Benjamini-Hochberg as §8. Writes `estimates.md` and `estimates.json` |
| `rates.py` | Every rate the REPRODUCIBILITY §6 appendix checklist asks for, by arm and by `--by` (ideology, topic, openness, topic_ideology, cell): `cache_warning`, `truncated`, `finish_reason=length`, `attempt > 1` (with a chi-square against condition), null and salvaged answers by `answer_method`, the unconstrained check (answers parsed, and agreement with the constrained answer to the same item), judge nulls and unresolved errors, the flag rate, and refusals. Writes `rates.md`, `rates.json`, `refusals.jsonl` |
| `calibrate.py` | `sample` writes a label sheet of 100-150 seeker turns stratified by ideology x turn bin, with the persona and the line and an empty `label` column (never the judge's score). `score` reads it back: Krippendorff's alpha (interval) with a bootstrap CI, quadratic-weighted kappa, and the threshold that maximises balanced accuracy for `score < threshold` |
| `verify_dialogue.py` | REPRODUCIBILITY §4a as code. Rebuilds every prompt of a dyad from `dyads.jsonl`, `turns.jsonl` and the template source in `manifest.json`, and compares its sha256, the model hash and the derived seed with each row; with the instrument, the pre and post survey prompts too. Reports the first mismatch |
| `stats.py` | OLS (HC1, CR1), a linear mixed model with one or two nested random intercepts by REML with Satterthwaite df, Wald F, Holm, BH, Fieller, Welch, Lee bounds, and the t/F/chi-square tails |
| `synth.py` | A synthetic run in the harness's formats (survey rows with `schema`, `temperature`, `top_p`, `n_predict`, `finish_reason`, `predicted_n`, `truncated`; score rows with `scope` and `subsample`), with planted effects and planted anomalies (`truth.json`), optionally an unconstrained check and a second stance judge on a subsample; `make_baseline` writes a no-dialogue baseline run. The tests are built on it |
| `power.py` | The PAP's power and MDE simulation (workstream W4; `docs/pap/README.md`) |

## Estimands

From `docs/decisions/persona-stability.md` §4 (the PAP §4 table refines it):

| Estimand | Sample | Status |
|---|---|---|
| ITT | every dialogue that ran to completion, any adherence | Primary |
| Dose-response | adherence as a continuous time-varying covariate, turn level | Secondary |
| Per-protocol | non-flagged dialogues only | Sensitivity, labelled as potentially biased |

"Dose-response" here is the adherence dose-response (PAP H3). The ideology dose slope (H1) is a
randomized ITT contrast.

## Statistical choices, and what stands in for statsmodels

statsmodels and scipy are not installed and not pinned, so nothing depends on them (scipy's special
functions are used for the tails when present).

- **Mixed model.** `stats.mixed` fits y = Xb + u_outer + v_inner + e by REML from per-group sums (V is
  block diagonal and each block has a closed-form inverse), with Satterthwaite df per contrast from a
  numerical information matrix. Role is the outer group (ideology/role); the pooled model adds role x
  topic, the turn-level H3 model adds dyad. On a balanced grid this is the nested ANOVA the power
  calculation uses: the tests check that a level contrast's SE and df (10) match the ANOVA exactly.
  Within-role demeaning is not used: role is nested in ideology, so role fixed effects would absorb
  the ideology effect.
- **When the role variance is estimated at zero**, Satterthwaite's df becomes the residual df (the fit
  is singular and the role term drops out). The two-stage role-means check keeps the roles - levels df
  either way, so read the two together.
- **Clustering.** Survey outcomes have one row per dyad, so dyad clustering is the model itself.
  Turn-level models carry a dyad random intercept; the adherence drift curve uses CR1 by dyad.
  CR1 by role is not used for inference (15 clusters; PAP §7.5).
- **p-values** are two-sided t with the contrast's df. CIs are unadjusted.

## Limits

- The refusal detector (`rates.REFUSAL_PATTERNS`) is a regex screen: English only, phrase-based, blind
  to soft refusals, and it counts a partial refusal as a refusal. Validate it on pilot transcripts by
  hand-checking `refusals.jsonl` and a sample of non-hits (PAP §6, [pilot]).
- `verify_dialogue` renders with empty BOS/EOS strings, because the manifest archives the template
  source but not the tokens. It warns when a template prints `bos_token` or `eos_token`; pass
  `--bos/--eos` then.
- A score row written before `scope` was recorded gets its scope from the turn cadence, so an old pilot
  pass's rows on turns 4, 8, ... count as main (or stance). `load` notes how many rows had no scope.

- The residual-variance-by-arm check (§7.5c) is a two-step reweighting, not a heteroscedastic REML fit.
  The ANCOVA check (§7.5d) is not built: it applies only if the PI adopts PAP §10 option B.
