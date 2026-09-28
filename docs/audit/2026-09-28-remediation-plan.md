# Remediation plan for the 2026-09-28 audits

Covers every finding in `2026-09-28-gap-audit.md` (F1-F20), `2026-09-28-red-team.md`
(H1-H4, M1-M9, L1-L10) and `2026-09-28-parallelism.md` (H1-H2, M1-M4, L1-L5). Each
workstream owns a disjoint set of paths so they can run in parallel. "Decision" marks an
item that needs the PI, not code; the plan produces the material the decision needs.

| Workstream | Owns | Closes |
|---|---|---|
| W0 harness fixes | `harness/`, `.gitignore`, `data/README.md` | RT H1-H4, M1-M9, L2-L3, L6-L9; PAR H1-H2, M1, M3-M4, L1; F6, F8, F10, F14, F16 |
| W1 persona catalogue | `prompts/personas/catalogue.json`, `prompts/README.md` | F1 |
| W2 instrument | `instruments/` | F2 |
| W3 analysis package | `analysis/`, `harness/tests/test_analysis_*` | F3, F12 (calibration, refusal), F19 (4a verifier) |
| W4 pre-analysis plan and power | `docs/pap/`, `analysis/power.py` | F4, F9 (descope rule), F10 (stance transform), F11 (baseline) |
| W5 hygiene | `pyproject.toml`, `.github/`, `.python-version`, `harness/requirements*`, `models/serving/`, `CITATION.cff` | F18, F19 (ORCID slot), PAR L3 |
| W6 harness follow-ups | `harness/` after W0 | F9 (study lock, subset descope), F12 (unconstrained survey check), F17 tests, RT L1, L4-L5, L10 note, PAR M2 note |
| W7 documentation reconciliation | `docs/`, `README.md`, `models/*.md` after W0 and W6 | F7, F15, F20; PAR L2, L5; RT H4 operating note |

## Sequencing

1. W0 runs alone in the main tree (it rewrites `harness/run.py`, `log.py`, `scorer.py`).
2. W1-W5 run in parallel in worktrees; none touches `harness/*.py`. W3 and W4 share
   the outcome definition below so the analysis code and the plan agree.
3. W6 and W7 start after W0 merges; W7 last, so the docs describe the code that exists.

## Shared definitions (W2, W3, W4 all use these)

Item direction: every ideological item gets `direction: "right"` or `"left"`, stating
which pole a high answer means. The ideological index is the mean of the five de Jong
items plus the two topic items, each recoded so high = right, on the original 1-5 scale.
Topic items: `ideo_enforcement_militarization` (right) and `ideo_decarbonization` (left).
Affective: `therm_gap = mean(therm_rep_*) - mean(therm_dem_*)`, signed, and its absolute
value as affective polarization. Norms index: mean of the three agreement items.

Primary outcome: post minus pre on the ideological index, signed so positive is rightward.
Ideology dose is coded -2 (strong_left) to +2 (strong_right), control at 0 without a
persona. The stance transform for the judge's `alignment` metric: `alignment` times the
sign of the persona's dose, undefined for the control and reported separately. Pre is
the harness's pre-survey as administered (F11 stays a decision: see `docs/pap/`).

## Items that remain decisions after this plan

- F5: run the pilot on the box, pick the seeker and the judge, commit the manifests.
- F11: whether the pre-measure stays greedy and context-free or is sampled at the
  dialogue temperature. The PAP lays out both with their cost.
- F13: real chat templates can only be tested against the GGUFs on the box; W6 adds
  the test scaffold and marks it live-only.
- F15 (Olmo np=4 vs 8) and the two-servers-one-GPU benchmark need the hardware.
