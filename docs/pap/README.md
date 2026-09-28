# docs/pap/

The pre-analysis plan (`pre-analysis-plan.md`) and the power numbers behind it (`analysis/power.py`).

## Registration

- **Where:** OSF Registries, using the OSF Preregistration template with `pre-analysis-plan.md`
  attached. It is chosen over AsPredicted because it takes the full document, timestamps it, and can
  embargo it until the paper goes out. An AsPredicted entry that points to the OSF record is optional.
- **When:** 2026-10-02 is the target. That is after the pilot values marked [pilot] are filled in and
  before wave 1 starts on 2026-10-05, so no wave data exists at registration. The runway put
  registration in W5 (Sep 21-27), which has passed. The registration says so, and it says which pilot
  data had been seen.
- **What is frozen:** the plan at a tagged commit (`pap-v1`), with its sha256 in the registration.
  Frozen with it are:
  - `prompts/grid.json` and `prompts/personas/catalogue.json`;
  - the adapted `instruments/batteries.json`;
  - the wave manifest and its `-assignment.json` (sha256 of each);
  - the harness commit;
  - the seeker model and both judge models (sha256), and the judge prompt;
  - the adherence threshold tau;
  - the `analysis/power.py` command line and its output.
- **After registration:** any change is a deviation (plan section 14).

## Power and MDE

`python3 analysis/power.py --sd 0.4,0.6,0.8,1.0 --icc-role 0,0.02,0.05 --mde-only --sims 4000` (seed
20260928). Assumptions are in the script's docstring.

- **Effect:** strong_right minus control on the change in the ideological index, in index points on the
  1-5 scale. The truth is linear in dose, so the slope is effect/2 per dose step.
- **Test:** two-sided, alpha 0.05, 80% power.
- **Model:** the plan's mixed model in its balanced (nested-ANOVA) form, whose test size is nominal.
- **Inputs:** SD is the within-cell post-survey index SD [pilot]. icc_role is the persona variants'
  share of that variance [pilot].
- **Scaling:** MDEs scale linearly with SD, so the d columns hold for any SD.

**Per-topic model pooled over the three arms** (the confirmatory unit), icc_role = 0.02, MDE in index
points:

| SD | Slope per step, N=135, 5 levels | Slope, N=90, 5 levels | Slope, N=90, 3 levels | Strong vs control, N=135, 5 levels | Strong vs control, N=90, 5 levels | Strong vs control, N=90, 3 levels |
|---|---|---|---|---|---|---|
| 0.4 | 0.034 | 0.036 | 0.044 | 0.120 | 0.128 | 0.135 |
| 0.6 | 0.052 | 0.053 | 0.066 | 0.180 | 0.193 | 0.202 |
| 0.8 | 0.069 | 0.071 | 0.088 | 0.240 | 0.257 | 0.269 |
| 1.0 | 0.086 | 0.089 | 0.110 | 0.300 | 0.321 | 0.337 |

**The same model in Cohen's d (MDE / SD), by icc_role**, where "3 lv" means three ideology levels:

| Pooled over arms | Slope, 135 | Slope, 90 | Slope, 90 and 3 lv | Strong vs control, 135 | Strong vs control, 90 | Strong vs control, 90 and 3 lv |
|---|---|---|---|---|---|---|
| icc_role 0 | 0.03 | 0.04 | 0.05 | 0.17 | 0.21 | 0.21 |
| icc_role 0.02 | 0.09 | 0.09 | 0.11 | 0.30 | 0.32 | 0.34 |
| icc_role 0.05 | 0.13 | 0.13 | 0.16 | 0.43 | 0.44 | 0.47 |

**One arm alone** (`--arms 1`), in the same d units. This is what an arm-specific claim, such as the
Olmo one, has to clear:

| One arm | Slope, 135 | Slope, 90 | Slope, 90 and 3 lv | Strong vs control, 135 | Strong vs control, 90 | Strong vs control, 90 and 3 lv |
|---|---|---|---|---|---|---|
| icc_role 0 | 0.06 | 0.07 | 0.09 | 0.30 | 0.36 | 0.37 |
| icc_role 0.02 | 0.10 | 0.11 | 0.13 | 0.38 | 0.44 | 0.45 |
| icc_role 0.05 | 0.14 | 0.15 | 0.18 | 0.49 | 0.53 | 0.55 |

**What "135 to 90 loses power" costs** (`factorial.md`, cut 1). These power figures do not depend on SD.
Take an effect that N = 135 detects with 80% power:

| icc_role | Slope power, N=90 | Slope power, N=90 and 3 levels | Strong-vs-control power, N=90 | Strong-vs-control power, N=90 and 3 levels |
|---|---|---|---|---|
| 0 | 0.63 | 0.47 | 0.64 | 0.63 |
| 0.02 | 0.77 | 0.59 | 0.75 | 0.71 |
| 0.05 | 0.79 | 0.61 | 0.77 | 0.72 |

In the MDE, cut 1 costs about 22% with no role variance (0.17 to 0.21 d for strong vs control) and
about 2 to 7% once the persona variants differ at all.

- **Role variance binds, not N.** Three variants per level are the replicates that a between-persona
  difference has to beat. So cut 1 is cheap if icc_role is above zero.
- **Cut 2 costs the slope more than it costs any single level contrast.** It removes the lean levels,
  which carry dose information.

The pilot's icc_role estimate is therefore as important as its SD. With five dyads per cell it will be
imprecise, so the registration quotes the table across the range above.

The index averages seven items, and only one of them is on the dialogue's topic. A movement confined
to that item reaches the index at one seventh of its size: 0.18 index points is about 1.3 points on
the own-topic item. That is why the plan registers the own-topic item as secondary outcome S6.

A pre sampled at the dialogue temperature (plan section 10, option B; `--pre-rho 0`) raises the
strong-vs-control MDE at N = 135 from 0.17 to 0.24 d with icc_role 0, and from 0.30 to 0.34 d with
0.02.
