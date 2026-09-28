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
python -m analysis.estimate --run-dir data/wave1-qwen --run-dir data/wave1-oss --run-dir data/wave1-olmo
python -m analysis.rates    --run-dir data/wave1-qwen --run-dir data/wave1-oss --by topic_ideology
python -m analysis.calibrate sample --run-dir data/pilot-2026-09-18 --n 120
python -m analysis.calibrate score  --run-dir data/pilot-2026-09-18 --labels labels.csv
python -m analysis.verify_dialogue --run-dir data/wave1-qwen --dyad-id <id>     # or --all
python -m analysis.synth --out /tmp/data --run-id synthetic                     # a dry-run dataset
```

Flags come from `flags.jsonl` (written by `python -m harness.run flags`), or from `--threshold T` on
`load`, `estimate` and `rates`, which applies `harness.scorer.flag_dialogues` to the main-cadence scores.

## Modules

| Module | What it does |
|---|---|
| `load.py` | Reads a run directory. Keeps the latest complete attempt per dyad (`harness.scorer.latest_complete_attempts`, the harness's own rule) and drops every other attempt's rows. Applies the technical exclusions, each with a scope: `not_run`, `incomplete`, `error_rows`, `short_dialogue`, `truncated` (dyad: out of the ITT sample); `missing_survey` (one index); `judge_failure` (one turn, counted); `unscored` (out of the adherence sample). Attaches mean seeker adherence and the flag. Writes `exclusions.md/.json/.jsonl` and `dyads.csv`, with counts by reason x ideology and by cell and a chi-square test of incompleteness against ideology |
| `outcomes.py` | Reads the instrument the run recorded (`manifest.json` `batteries.path`), checks its sha256, and falls back to the matching version in git history. Index definitions come from its `indices` block, else from the remediation plan's Shared definitions. Recodes by `direction`. Builds `ideological`, `therm_gap`, `affective_abs` and `norms`, plus `topic_item` (the dialogue's own topic item, PAP S6). Change = post - pre. `--json-only`, `--available-items` and `--impute-refusals mid/low/high` are the PAP §6 sensitivities |
| `estimate.py` | The PAP §7 models, pooled over arms with arm fixed effects: level model (ITT per level against the bare control), slope model (H1), shape test and its classification (H2), H3 at the turn level and the dyad level, secondary tests S1-S6, per-protocol, and the §7.5 robustness checks. Holm and Benjamini-Hochberg as §8. Writes `estimates.md` and `estimates.json` |
| `rates.py` | Every rate the REPRODUCIBILITY §6 appendix checklist asks for, by arm and by `--by` (ideology, topic, openness, topic_ideology, cell): `cache_warning`, `truncated`, `finish_reason=length`, `attempt > 1` (with a chi-square against condition), null and salvaged answers by `answer_method`, judge nulls and unresolved errors, the flag rate, and refusals. Writes `rates.md`, `rates.json`, `refusals.jsonl` |
| `calibrate.py` | `sample` writes a label sheet of 100-150 seeker turns stratified by ideology x turn bin, with the persona and the line and an empty `label` column (never the judge's score). `score` reads it back: Krippendorff's alpha (interval) with a bootstrap CI, quadratic-weighted kappa, and the threshold that maximises balanced accuracy for `score < threshold` |
| `verify_dialogue.py` | REPRODUCIBILITY §4a as code. Rebuilds every prompt of a dyad from `dyads.jsonl`, `turns.jsonl` and the template source in `manifest.json`, and compares its sha256, the model hash and the derived seed with each row; with the instrument, the pre and post survey prompts too. Reports the first mismatch |
| `stats.py` | OLS (HC1, CR1), a linear mixed model with one or two nested random intercepts by REML with Satterthwaite df, Wald F, Holm, BH, Fieller, Welch, Lee bounds, and the t/F/chi-square tails |
| `synth.py` | A synthetic run in the harness's formats, with planted effects and planted anomalies (`truth.json`); the tests are built on it |
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
- Score rows do not record their scope, so `--threshold` flags use the main cadence (turns 4, 8, ... and
  the final turn), which keeps pilot-scope rows from the same judge out of the registered rule.
- The residual-variance-by-arm check (§7.5c) is a two-step reweighting, not a heteroscedastic REML fit.
  The ANCOVA check (§7.5d) is not built: it applies only if the PI adopts PAP §10 option B.
