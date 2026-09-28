"""The registered estimates (docs/pap/pre-analysis-plan.md §4, §7, §8): the ITT of each ideology level
against the bare control, the ideology dose slope (H1), the accommodation shape test (H2), the adherence
dose-response (H3), the secondary tests, the per-protocol sensitivity and the §7.5 robustness checks.

    python -m analysis.estimate --run-dir data/<arm_a> [--run-dir data/<arm_b> ...] [--threshold T]
        [--stance-judge PREFIX] [--out DIR]

One run directory is one mentor arm. Arms are pooled with arm fixed effects: the plan's confirmatory
unit is per topic, pooled over arms.

Models (PAP §7). y is a change score, one row per dyad. Openness is +1/2 open, -1/2 closed, 0 for the
control. The role random intercept (group ideology/role) applies to treated dyads only; the control has
no role. Every model is fitted by REML with Satterthwaite df per contrast (analysis.stats.mixed), which on
a balanced grid is the nested ANOVA the power calculation uses.

- level model: y = arm + sum_l tau_l [level = l] + omega open + u_role. tau_l is level l's ITT against
  the control, averaged over openness.
- slope model: y = arm + gamma T + beta dose + omega open + u_role. beta is H1; gamma is moderate minus
  control. The unsigned outcomes (affective_abs, norms) use intensity |dose| in place of dose (S2, S4).
- pooled over topics (summary, not confirmatory): topic (+-1/2) and its interaction with every level (or
  with T and dose), and a role x topic random intercept nested in role.
- H2 from the level model per topic: R = (tau_+1 + tau_+2)/2, L = (tau_-1 + tau_-2)/2, A = R + L,
  rho = R / (-L) with a Fieller 95% CI, and the intensity contrasts tau_+2 - tau_+1, tau_-1 - tau_-2.
  Classified as accommodation-consistent (Holm-adjusted A > 0 and rho's CI excludes 1),
  symmetric-drift-consistent (A not significant, rho's CI includes 1 and excludes 8, both intensity
  contrasts >= 0) or inconclusive.
- H3, turn level: agree = 2 x alignment - 1 (stance-scope rows) on the seeker's prompt_to_line (main-scope
  rows) at the same scored turn, with turn and cell (topic x ideology x openness) fixed effects, arm, and
  random intercepts for role and dyad (nested). Robustness: cumulative mean adherence to turn t. Dyad
  level: the slope model on treated dyads plus centred mean adherence and dose x adherence. Judge
  agreement pairs the primary stance judge with the second judge's `subsample`-marked rows.
- secondary: S1 arm x dose (a Wald test; per-arm slopes), S2 affective_abs on intensity, S3 therm_gap on
  dose, S4 norms on intensity, S5 openness x dose, S6 the topic item on dose, and H3; BH across the family.
- robustness (primary outcome): (a) item-level stacked model with item fixed effects and dyad (and role)
  random intercepts; (b) two-stage role means; (c) residual variance by arm (two-step reweighting, when
  there is more than one arm); Lee bounds when incompleteness differs by level; the §6 survey
  sensitivities (json-only, available items, refusals imputed at the midpoint and at either end).
- per-protocol: the level and slope models on non-flagged treated dyads and every control.

Multiplicity (§8): Holm across the two topics for H1 and for H2, Holm within topic across the five
level contrasts, Benjamini-Hochberg across the secondary family. CIs are unadjusted. Standardised effects
divide by the post-survey SD of the outcome among control dyads of the topic."""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
import numpy as np
from analysis import stats
from analysis._util import fmt, md_table, write_json
from analysis.calibrate import krippendorff_interval
from analysis.load import LEVELS, DOSE, Run, add_common_args, default_out, exclusion_summary, load_from_args
from analysis.load import pick_judge
from analysis.outcomes import SIGNED, add_instrument_args, compute_outcomes, item_changes, load_instrument
from harness.transcript import MENTOR, SEEKER

PRIMARY = "ideological"
SECONDARY = {"affective_abs": "S2", "therm_gap": "S3", "norms": "S4", "topic_item": "S6"}
DIRECTIONAL = ("strong_left", "lean_left", "lean_right", "strong_right")
SYMMETRY_BOUND = 8.0                  # the tornberg2026audits asymmetry the symmetric reading must exclude


# ---- design ---------------------------------------------------------------------------------------

class Design:
    """Named columns for a design matrix."""

    def __init__(self, n: int):
        self.n, self.names, self.cols = n, [], []

    def add(self, name: str, values) -> None:
        self.names.append(name)
        self.cols.append([float(v) for v in values])

    def matrix(self) -> np.ndarray:
        return np.array(self.cols).T


def _open(r) -> float:
    if r["control"]:
        return 0.0
    return 0.5 if r["openness"] == "open" else -0.5


def _topic(r, topics) -> float:
    return 0.5 if r["topic"] == topics[-1] else -0.5


def _role(r):
    return None if r["control"] or r.get("role") is None else f"{r['ideology']}/{r['role']}"


def _role_topic(r):
    return None if _role(r) is None else f"{_role(r)}/{r['topic']}"


def _x(r, signed: bool) -> float:
    return float(r["dose"]) if signed else float(abs(r["dose"]))


def _sample(rows, outcome, topic=None, keep=None, treated_only=False):
    out = [r for r in rows if r["in_itt"] and r.get(f"{outcome}_change") is not None
           and (topic is None or r["topic"] == topic) and (not treated_only or not r["control"])]
    return [r for r in out if keep(r)] if keep else out


def _arms(D: Design, s, arms) -> list[str]:
    names = []
    for a in arms[1:]:
        if any(r["arm"] == a for r in s):
            D.add(f"arm[{a}]", [r["arm"] == a for r in s])
            names.append(f"arm[{a}]")
    return names


def _variance(fit: stats.Fit) -> dict:
    return {k: fit.extra.get(k) for k in ("sigma2_group", "sigma2_inner", "sigma2_resid", "icc", "n_groups")}


def _fit_mixed(D, y, s, pooled: bool, weights=None):
    X, yv = D.matrix(), np.asarray(y, float)
    if weights is not None:
        X, yv = X * weights[:, None], yv * weights
    inner = [_role_topic(r) for r in s] if pooled else None
    return stats.mixed(X, yv, D.names, [_role(r) for r in s], inner)


def _control_sd(rows, outcome, topic) -> float | None:
    v = [r[f"{outcome}_post"] for r in rows if r["control"] and r["in_itt"]
         and r.get(f"{outcome}_post") is not None and (topic is None or r["topic"] == topic)]
    return float(np.std(v, ddof=1)) if len(v) > 1 and np.std(v) > 0 else None


def _std(c: dict | None, sd: float | None) -> dict | None:
    if c is None:
        return None
    return {**c, "d": c["estimate"] / sd if sd else None}


# ---- the survey models ----------------------------------------------------------------------------

def level_model(rows, outcome, topic, arms, topics, keep=None, weights_by_arm=None) -> tuple[dict, object]:
    """The level model (see the module docstring). Returns (result, fit); fit is None when skipped."""
    s = _sample(rows, outcome, topic, keep)
    levels = [lv for lv in LEVELS if any(r["ideology"] == lv for r in s)]
    n_ctrl = sum(r["control"] for r in s)
    if not n_ctrl or not levels:
        return {"skipped": "no control dyads" if not n_ctrl else "no treated dyads", "n": len(s)}, None
    pooled = topic is None and len({r["topic"] for r in s}) > 1
    D = Design(len(s))
    D.add("control", [1] * len(s))
    arm_names = _arms(D, s, arms)
    for lv in levels:
        D.add(f"tau[{lv}]", [r["ideology"] == lv for r in s])
    if len({r["openness"] for r in s if not r["control"]}) > 1:
        D.add("open", [_open(r) for r in s])
    if pooled:
        D.add("topic", [_topic(r, topics) for r in s])
        for lv in levels:
            D.add(f"topic x tau[{lv}]", [_topic(r, topics) * (r["ideology"] == lv) for r in s])
    w = np.array([weights_by_arm.get(r["arm"], 1.0) for r in s]) if weights_by_arm else None
    try:
        fit = _fit_mixed(D, [r[f"{outcome}_change"] for r in s], s, pooled, w)
    except (ValueError, np.linalg.LinAlgError) as e:
        return {"skipped": str(e), "n": len(s)}, None
    sd = _control_sd(rows, outcome, topic)
    arm_avg = {a: 1 / (len(arm_names) + 1) for a in arm_names}
    res = {"n": len(s), "n_control": n_ctrl,
           "n_by_level": {lv: sum(r["ideology"] == lv for r in s) for lv in levels},
           "method": fit.method, "variance": _variance(fit), "control_post_sd": sd,
           "tau": {lv: _std(fit.coef(f"tau[{lv}]"), sd) for lv in levels},
           "control_mean": fit.lincom({"control": 1, **arm_avg}),
           "terms": {n: fit.coef(n) for n in D.names}}
    if outcome in SIGNED:
        res["shape"] = shape_test(fit, sd)
    return res, fit


def shape_test(fit: stats.Fit, sd: float | None = None) -> dict:
    """R, L, A, rho (Fieller) and the intensity contrasts from a level-model fit (PAP §7.3)."""
    t = lambda lv: f"tau[{lv}]"  # noqa: E731
    wr = {t("lean_right"): 0.5, t("strong_right"): 0.5}
    wl = {t("lean_left"): 0.5, t("strong_left"): 0.5}
    out = {"R": _std(fit.lincom(wr), sd), "L": _std(fit.lincom(wl), sd),
           "A": _std(fit.lincom({**wr, **wl}), sd),
           "intensity_right": fit.lincom({t("strong_right"): 1, t("lean_right"): -1}),
           "intensity_left": fit.lincom({t("lean_left"): 1, t("strong_left"): -1}),
           "moderate": fit.lincom({t("moderate"): 1})}
    cr, cl = fit.vec(wr), fit.vec(wl)
    if cr is not None and cl is not None:
        # rho = R / (-L): a = R, b = -L, so var(b) = var(L) and cov(a, b) = -cov(R, L)
        df = min(fit.contrast_df(cr), fit.contrast_df(cl))
        out["rho"] = stats.fieller(float(cr @ fit.beta), -float(cl @ fit.beta), float(cr @ fit.vcov @ cr),
                                   float(cl @ fit.vcov @ cl), -float(cr @ fit.vcov @ cl), stats.t_crit(df))
    return out


def classify(shape: dict, p_holm: float | None) -> str:
    """The PAP §7.3 reading of one topic's shape test."""
    A, rho = shape.get("A"), shape.get("rho")
    if not A or not rho or p_holm is None:
        return "not computed"
    has = lambda x: stats.fieller_contains(rho, x)  # noqa: E731
    if p_holm < 0.05 and A["estimate"] > 0 and not has(1.0):
        return "accommodation-consistent"
    ir, il = shape.get("intensity_right"), shape.get("intensity_left")
    if (p_holm >= 0.05 and has(1.0) and not has(SYMMETRY_BOUND)
            and ir and il and ir["estimate"] >= 0 and il["estimate"] >= 0):
        return "symmetric-drift-consistent"
    return "inconclusive"


def slope_model(rows, outcome, topic, arms, topics, keep=None, arm_x=False, open_x=False, adherence=False,
                weights_by_arm=None) -> tuple[dict, object]:
    """The slope model, and its S1 (arm_x), S5 (open_x) and dyad-level H3 (adherence) extensions."""
    signed = outcome in SIGNED
    term = "dose" if signed else "intensity"
    s = _sample(rows, outcome, topic, keep, treated_only=adherence)
    if adherence:
        s = [r for r in s if r.get("adherence_mean") is not None]
    if len(s) < 6 or not any(not r["control"] for r in s):
        return {"skipped": f"{len(s)} dyads", "n": len(s)}, None
    pooled = topic is None and len({r["topic"] for r in s}) > 1
    D = Design(len(s))
    D.add("intercept", [1] * len(s))
    arm_names = _arms(D, s, arms)
    has_T = any(r["control"] for r in s)
    if has_T:
        D.add("T", [not r["control"] for r in s])
    D.add(term, [_x(r, signed) for r in s])
    if len({r["openness"] for r in s if not r["control"]}) > 1:
        D.add("open", [_open(r) for r in s])
    if pooled:
        D.add("topic", [_topic(r, topics) for r in s])
        if has_T:
            D.add("topic x T", [_topic(r, topics) * (not r["control"]) for r in s])
        D.add(f"topic x {term}", [_topic(r, topics) * _x(r, signed) for r in s])
    if arm_x:
        for a in arm_names:
            arm = a[4:-1]
            if has_T:
                D.add(f"{a} x T", [(r["arm"] == arm) * (not r["control"]) for r in s])
            D.add(f"{a} x {term}", [(r["arm"] == arm) * _x(r, signed) for r in s])
    if open_x and "open" in D.names:
        D.add(f"open x {term}", [_open(r) * _x(r, signed) for r in s])
    if adherence:
        mean_a = float(np.mean([r["adherence_mean"] for r in s]))
        D.add("adherence", [r["adherence_mean"] - mean_a for r in s])
        D.add(f"{term} x adherence", [_x(r, signed) * (r["adherence_mean"] - mean_a) for r in s])
    w = np.array([weights_by_arm.get(r["arm"], 1.0) for r in s]) if weights_by_arm else None
    try:
        fit = _fit_mixed(D, [r[f"{outcome}_change"] for r in s], s, pooled, w)
    except (ValueError, np.linalg.LinAlgError) as e:
        return {"skipped": str(e), "n": len(s)}, None
    sd = _control_sd(rows, outcome, topic)
    res = {"n": len(s), "term": term, "method": fit.method, "variance": _variance(fit),
           "control_post_sd": sd, "slope": _std(fit.coef(term), sd), "T": fit.coef("T") if has_T else None,
           "terms": {n: fit.coef(n) for n in D.names}}
    if arm_x:
        inter = [n for n in D.names if n.endswith(f" x {term}") and n.startswith("arm[")]
        res["arm_x_wald"] = fit.wald([{n: 1.0} for n in inter]) if inter else None
        res["per_arm_slope"] = {arms[0]: fit.coef(term)}
        for n in inter:
            res["per_arm_slope"][n[4:n.index("]")]] = fit.lincom({term: 1.0, n: 1.0})
    if open_x:
        res["open_x"] = fit.coef(f"open x {term}")
    if adherence:
        res["dose_x_adherence"] = fit.coef(f"{term} x adherence")
        res["adherence"] = fit.coef("adherence")
    return res, fit


def control_change(rows, outcome, topic) -> dict:
    """The control's own change per arm (the 'dialogue per se' contrast), mean with a t interval."""
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        v = np.array([r[f"{outcome}_change"] for r in rows
                      if r["control"] and r["in_itt"] and r["arm"] == arm and r["topic"] == topic
                      and r.get(f"{outcome}_change") is not None])
        if len(v) < 2:
            continue
        se = float(v.std(ddof=1) / math.sqrt(len(v)))
        q = stats.t_crit(len(v) - 1)
        out[arm] = {"n": int(len(v)), "estimate": float(v.mean()), "se": se, "df": len(v) - 1,
                    "ci": [v.mean() - q * se, v.mean() + q * se],
                    "p": stats.t_pvalue(v.mean() / se, len(v) - 1) if se > 0 else None}
    return out


# ---- robustness -----------------------------------------------------------------------------------

def two_stage(rows, outcome, topic, arms) -> dict:
    """PAP §7.5(b): change averaged to role x arm, level contrasts and the slope fitted on the role means;
    the control enters as its per-arm means (it has no roles), combined by Welch-Satterthwaite."""
    s = _sample(rows, outcome, topic)
    units: dict = {}
    for r in s:
        if not r["control"] and r.get("role") is not None:
            units.setdefault((r["arm"], r["ideology"], r["role"]), []).append(r[f"{outcome}_change"])
    levels = [lv for lv in LEVELS if any(k[1] == lv for k in units)]
    u_arms = [a for a in arms if any(k[0] == a for k in units)]
    if not units or len(units) <= len(levels) + len(u_arms) - 1:
        return {"skipped": f"{len(units)} role means"}
    keys = sorted(units)
    y = np.array([np.mean(units[k]) for k in keys])
    D = Design(len(keys))
    for lv in levels:
        D.add(f"tau[{lv}]", [k[1] == lv for k in keys])
    for a in u_arms[1:]:
        D.add(f"arm[{a}]", [k[0] == a for k in keys])
    try:
        f2 = stats.ols_classical(D.matrix(), y, D.names)
    except (ValueError, np.linalg.LinAlgError) as e:
        return {"skipped": str(e)}
    ctrl = []
    for a in u_arms:
        v = np.array([r[f"{outcome}_change"] for r in s if r["control"] and r["arm"] == a])
        if len(v) < 2:
            return {"skipped": f"fewer than 2 control dyads in arm {a}"}
        ctrl.append((float(v.mean()), float(v.var(ddof=1) / len(v)), len(v) - 1))
    c_mean = sum(c[0] for c in ctrl) / len(ctrl)

    def combo(wl: dict[str, float]) -> dict | None:
        """sum_l w_l tau_l at the arm average, minus sum_l w_l times the arm-averaged control mean."""
        if any(f"tau[{lv}]" not in D.names for lv in wl):
            return None
        tot = sum(wl.values())
        w = {f"tau[{lv}]": x for lv, x in wl.items()}
        w.update({f"arm[{a}]": tot / len(u_arms) for a in u_arms[1:]})
        part = f2.lincom(w)
        parts = [(1.0, part["se"] ** 2, part["df"])] + [(tot / len(ctrl), v, d) for _, v, d in ctrl]
        var, df = stats.welch(parts)
        est = part["estimate"] - tot * c_mean
        se = math.sqrt(var)
        q = stats.t_crit(df)
        return {"estimate": est, "se": se, "df": df, "p": stats.t_pvalue(est / se, df) if se else None,
                "ci": [est - q * se, est + q * se]}
    signed = outcome in SIGNED
    Ds = Design(len(keys))
    Ds.add("intercept", [1] * len(keys))
    Ds.add("slope", [DOSE[k[1]] if signed else abs(DOSE[k[1]]) for k in keys])
    for a in u_arms[1:]:
        Ds.add(f"arm[{a}]", [k[0] == a for k in keys])
    try:
        slope = stats.ols_classical(Ds.matrix(), y, Ds.names).coef("slope")
    except (ValueError, np.linalg.LinAlgError):
        slope = None
    return {"n_units": len(keys), "tau": {lv: combo({lv: 1.0}) for lv in levels}, "slope": slope,
            "A": combo({lv: 0.5 for lv in DIRECTIONAL}) if all(lv in levels for lv in DIRECTIONAL) else None}


def stacked(item_rows, topic, arms) -> dict:
    """PAP §7.5(a): per-item change scores of the primary index stacked, item fixed effects, the level and
    slope terms, random intercepts for role and for dyad nested in it (control dyads: dyad only)."""
    s = [r for r in item_rows if r["topic"] == topic]
    if not s:
        return {"skipped": "no item rows"}
    items = sorted({r["item"] for r in s})
    out = {}
    for kind in ("level", "slope"):
        D = Design(len(s))
        for it in items:
            D.add(f"item[{it}]", [r["item"] == it for r in s])
        _arms(D, s, arms)
        if kind == "level":
            for lv in LEVELS:
                if any(r["ideology"] == lv for r in s):
                    D.add(f"tau[{lv}]", [r["ideology"] == lv for r in s])
        else:
            D.add("T", [not r["control"] for r in s])
            D.add("dose", [float(r["dose"]) for r in s])
        if len({r["openness"] for r in s if not r["control"]}) > 1:
            D.add("open", [_open(r) for r in s])
        try:
            fit = stats.mixed(D.matrix(), [r[f"{PRIMARY}_change"] for r in s], D.names,
                              [_role(r) for r in s], [f"{r['run_id']}/{r['dyad_id']}" for r in s])
        except (ValueError, np.linalg.LinAlgError) as e:
            out[kind] = {"skipped": str(e)}
            continue
        out[kind] = {"n": len(s), "method": fit.method, "variance": _variance(fit)}
        if kind == "level":
            out[kind]["shape"] = shape_test(fit)
        else:
            out[kind]["slope"] = fit.coef("dose")
    return out


def arm_weights(rows, outcome, topic, arms, topics) -> dict | None:
    """PAP §7.5(c), two-step: residual variance by arm from the slope model, returned as row weights
    sqrt(mean variance / arm variance) for a reweighted refit."""
    _, fit = slope_model(rows, outcome, topic, arms, topics)
    s = _sample(rows, outcome, topic)
    if fit is None or len({r["arm"] for r in s}) < 2:
        return None
    signed = outcome in SIGNED
    resid: dict = {}
    for r in s:
        x = _row_terms(r, fit.names, signed, arms, topics)
        resid.setdefault(r["arm"], []).append(r[f"{outcome}_change"] - float(np.dot(fit.beta, x)))
    var = {a: float(np.var(v, ddof=1)) for a, v in resid.items() if len(v) > 1}
    mean = float(np.mean(list(var.values())))
    return {a: math.sqrt(mean / v) if v > 0 else 1.0 for a, v in var.items()}


def _row_terms(r, names, signed, arms, topics) -> list[float]:
    """One row's slope-model covariates in `names` order (to form residuals for the arm weights)."""
    term = "dose" if signed else "intensity"
    vals = {"intercept": 1.0, "T": float(not r["control"]), term: _x(r, signed), "open": _open(r),
            "topic": _topic(r, topics), "topic x T": _topic(r, topics) * (not r["control"]),
            f"topic x {term}": _topic(r, topics) * _x(r, signed)}
    for a in arms[1:]:
        vals[f"arm[{a}]"] = float(r["arm"] == a)
    return [vals.get(n, 0.0) for n in names]


def lee(rows, outcome, topic) -> dict:
    """Lee bounds on each level's contrast with the control, pooled over arms: assigned is every planned
    dyad of the cell, observed is the ITT dyads with the outcome."""
    out = {}
    ctrl_all = [r for r in rows if r["control"] and r["topic"] == topic]
    seen = lambda rs: [r[f"{outcome}_change"] for r in rs  # noqa: E731
                       if r["in_itt"] and r.get(f"{outcome}_change") is not None]
    y0 = seen(ctrl_all)
    for lv in LEVELS:
        all_l = [r for r in rows if r["ideology"] == lv and r["topic"] == topic]
        y1 = seen(all_l)
        if all_l and ctrl_all:
            out[lv] = stats.lee_bounds(y1, len(all_l), y0, len(ctrl_all))
    return out


# ---- turn level -----------------------------------------------------------------------------------

def _stance_scores(run: Run) -> list[dict]:
    """The run's mentor `alignment` rows from stance-scope passes (`score --scope stance`), the turn-level
    DV. Pilot-scope rows (every turn, for calibration) never enter the stance analyses."""
    return [s for s in run.scores_in("stance") if s.get("metric") == "alignment" and s.get("agent") == MENTOR]


def stance_judge(runs: list[Run], prefix: str | None) -> tuple[str | None, list[str]]:
    """The primary stance judge: the one matching `prefix`, else the adherence judge when it scored
    alignment, else the judge with a full (not `subsample`-marked) stance pass that scored the most dyads
    (the second judge scores a subsample). Stance-scope rows only."""
    scores = [s for run in runs for s in _stance_scores(run) if not s.get("error")]
    judges = sorted({s["judge_sha256"] for s in scores})
    if not judges:
        return None, []
    if prefix:
        return pick_judge(scores, "alignment", prefix), judges
    for j in judges:
        if j in {run.judge for run in runs}:
            return j, judges
    full = {s["judge_sha256"] for s in scores if s.get("subsample") is None}
    cover = {j: len({(s.get("run_id"), s["dyad_id"]) for s in scores if s["judge_sha256"] == j})
             for j in judges}
    return max(judges, key=lambda j: (j in full, cover[j])), judges


def _adherence_scores(run: Run) -> list[dict]:
    """The run's usable seeker adherence scores: main-scope rows of the chosen judge and metric, not null,
    not an error."""
    return [s for s in run.scores_in("main") if s.get("judge_sha256") == run.judge
            and s.get("metric") == run.metric and s.get("agent") == SEEKER and s.get("score") is not None
            and not s.get("error")]


def turn_rows(runs: list[Run], judge: str, dyads: set | None = None) -> list[dict]:
    """One row per stance-scored mentor turn of a treated dyad in the adherence sample, with the seeker's
    adherence at the same turn and its cumulative mean to that turn."""
    out = []
    for run in runs:
        dy = {d["dyad_id"]: d for d in run.dyads if d.get("in_adherence")}
        seeker: dict = {}
        for s in _adherence_scores(run):
            if s["dyad_id"] in dy:
                seeker.setdefault(s["dyad_id"], {})[s["turn"]] = s["score"]
        for s in _stance_scores(run):
            d = dy.get(s["dyad_id"])
            if (d is None or s.get("judge_sha256") != judge or s.get("score") is None or s.get("error")):
                continue
            if dyads is not None and (run.run_id, d["dyad_id"]) not in dyads:
                continue
            sk = seeker.get(d["dyad_id"], {})
            if s["turn"] not in sk:
                continue
            upto = [v for t, v in sk.items() if t <= s["turn"]]
            out.append({**d, "run_id": run.run_id, "turn": s["turn"], "agree": 2 * s["score"] - 1,
                        "alignment": s["score"], "adh": sk[s["turn"]], "adh_cum": sum(upto) / len(upto)})
    return out


def h3_turn(rows, topic, arms, covariate: str = "adh") -> dict:
    """PAP §7.4 turn-level model (see the module docstring)."""
    s = [r for r in rows if topic is None or r["topic"] == topic]
    if len(s) < 20:
        return {"skipped": f"{len(s)} scored turns"}
    D = Design(len(s))
    D.add("intercept", [1] * len(s))
    D.add("adherence", [r[covariate] for r in s])
    turns = sorted({r["turn"] for r in s})
    for t in turns[1:]:
        D.add(f"turn[{t}]", [r["turn"] == t for r in s])
    cells = sorted({r["cell"] for r in s})
    for c in cells[1:]:
        D.add(f"cell[{c}]", [r["cell"] == c for r in s])
    _arms(D, s, arms)
    try:
        fit = stats.mixed(D.matrix(), [r["agree"] for r in s], D.names, [_role(r) for r in s],
                          [f"{r['run_id']}/{r['dyad_id']}" for r in s])
    except (ValueError, np.linalg.LinAlgError) as e:
        return {"skipped": str(e)}
    return {"n_rows": len(s), "n_dyads": len({(r["run_id"], r["dyad_id"]) for r in s}),
            "method": fit.method, "variance": _variance(fit), "theta": fit.coef("adherence")}


def judge_agreement(runs: list[Run], judges: list[str]) -> dict:
    """Krippendorff's alpha (interval) on stance-scope `alignment` between the primary judge (judges[0])
    and the second judge, on shared targets. The second judge is the first other judge with rows marked
    `subsample` (`score --scope stance --subsample F`), whose marked rows are its targets; failing that,
    the first other judge and all its stance rows."""
    if len(judges) < 2:
        return {"skipped": "one stance judge"}
    usable = [(run.run_id, s) for run in runs for s in _stance_scores(run)
              if s.get("score") is not None and not s.get("error")]
    marked = {s["judge_sha256"] for _, s in usable if s.get("subsample") is not None}
    a = judges[0]
    b = next((j for j in judges[1:] if j in marked), judges[1])
    by: dict = {a: {}, b: {}}
    fractions = set()
    for run_id, s in usable:
        j = s.get("judge_sha256")
        if j == a:
            by[a][(run_id, s["dyad_id"], s["turn"])] = s["score"]
        elif j == b and (b not in marked or s.get("subsample") is not None):
            by[b][(run_id, s["dyad_id"], s["turn"])] = s["score"]
            fractions.add(s.get("subsample"))
    shared = sorted(set(by[a]) & set(by[b]))
    alpha = krippendorff_interval([[by[a][k], by[b][k]] for k in shared]) if shared else None
    return {"judges": [a, b], "n": len(shared), "alpha": alpha, "dyads": sorted({k[:2] for k in shared}),
            "subsample": sorted(f for f in fractions if f is not None) or None,
            "exploratory": alpha is None or alpha < 0.667}


def trajectory(rows) -> dict:
    """Exploratory: mean signed stance, sign(dose) x (2 alignment - 1), by level and turn; moderate as
    unsigned agreement (2 alignment - 1)."""
    out: dict = {}
    for r in rows:
        v = r["agree"] * (1 if r["dose"] > 0 else -1) if r["dose"] else r["agree"]
        out.setdefault(r["ideology"], {}).setdefault(r["turn"], []).append(v)
    return {lv: {t: {"n": len(v), "mean": float(np.mean(v))} for t, v in sorted(ts.items())}
            for lv, ts in out.items()}


def signed_stance(alignment: float, dose: int | None) -> float | None:
    """PAP §3.6: sign(dose) x (2 alignment - 1), undefined (None) for the control and for moderate."""
    if not dose:
        return None
    return (1 if dose > 0 else -1) * (2 * alignment - 1)


def control_alignment(runs: list[Run], judge: str) -> dict:
    """Control dyads' raw `alignment`, reported apart and never modelled (stance is undefined there). main
    and stance scope leave it unscored (harness.scorer.CONTROL_ALIGNMENT_UNSCORED), so these are pilot
    rows, or rows written before that default; `scopes` says which."""
    v, scopes = [], set()
    for run in runs:
        ctrl = {d["dyad_id"] for d in run.dyads if d["control"]}
        for s in run.scores:
            if (s.get("metric") == "alignment" and s.get("score") is not None and not s.get("error")
                    and s.get("judge_sha256") == judge and s["dyad_id"] in ctrl):
                v.append(s["score"])
                scopes.add(run.scope_of(s))
    return {"n": len(v), "mean": float(np.mean(v)) if v else None, "scopes": sorted(scopes)}


def adherence_drift(runs: list[Run]) -> dict:
    """Seeker adherence over turns by ideology level (the manipulation-check curve), OLS with SEs clustered
    by dyad, and the flag rate by level."""
    rows, dy = [], {}
    for run in runs:
        for d in run.dyads:
            if d.get("in_adherence"):
                dy[(run.run_id, d["dyad_id"])] = d
        rows += [(run.run_id, s) for s in _adherence_scores(run) if (run.run_id, s["dyad_id"]) in dy]
    out = {}
    for lv in LEVELS:
        rs = [(rid, s) for rid, s in rows if dy[(rid, s["dyad_id"])]["ideology"] == lv]
        ds = [d for d in dy.values() if d["ideology"] == lv]
        known = [d for d in ds if d.get("flagged") is not None]
        entry = {"n_dyads": len(ds), "flagged": sum(bool(d["flagged"]) for d in known),
                 "flag_known": len(known)}
        if rs:
            entry["mean"] = float(np.mean([s["score"] for _, s in rs]))
        if len({(rid, s["dyad_id"]) for rid, s in rs}) >= 2 and len({s["turn"] for _, s in rs}) >= 2:
            X = np.array([[1.0, s["turn"]] for _, s in rs])
            f = stats.ols(X, np.array([s["score"] for _, s in rs]), ["intercept", "turn"],
                          clusters=[f"{rid}/{s['dyad_id']}" for rid, s in rs])
            entry["slope_per_turn"] = f.coef("turn")
        out[lv] = entry
    return out


# ---- the whole analysis ---------------------------------------------------------------------------

def per_protocol_keep(r) -> bool:
    """Per-protocol sample: every control, and treated dyads whose flag is known and False."""
    return r["control"] or r.get("flagged") is False


def estimate(runs: list[Run], rows: list[dict], item_rows: list[dict], outcomes: list[str],
             stance_prefix: str | None = None, sensitivity: dict | None = None) -> dict:
    """Every registered estimate over the pooled rows of `runs` (one per arm)."""
    arms = [run.arm for run in runs]
    topics = sorted({r["topic"] for r in rows if r["topic"]})
    cols = topics + (["pooled"] if len(topics) > 1 else [])
    modes = sorted({str(r["persona_mode"]) for r in rows if r["in_itt"]})
    res = {"arms": arms, "topics": topics, "persona_modes": modes, "outcomes": {}, "notes": [],
           "runs": [exclusion_summary(run) for run in runs],
           "methods": {"model": "REML; random intercept for role (and role x topic when pooled); "
                                "Satterthwaite df", "tails": stats.SPECIAL_BACKEND}}
    for run in runs:
        res["notes"] += [f"{run.arm}: {n}" for n in run.notes]
    if len(modes) > 1:
        res["notes"].append(f"persona modes {modes} are pooled; pass --persona-mode")
    has_flags = any(r.get("flagged") is not None for r in rows)
    has_adh = any(run.judge for run in runs)
    for name in outcomes:
        o = {"level": {}, "slope": {}, "control_change": {}, "per_protocol": {"level": {}, "slope": {}}}
        for c in cols:
            topic = None if c == "pooled" else c
            o["level"][c] = level_model(rows, name, topic, arms, topics)[0]
            o["slope"][c] = slope_model(rows, name, topic, arms, topics)[0]
            if topic:
                o["control_change"][c] = control_change(rows, name, topic)
            if has_flags:
                pp = o["per_protocol"]
                pp["level"][c] = level_model(rows, name, topic, arms, topics, keep=per_protocol_keep)[0]
                pp["slope"][c] = slope_model(rows, name, topic, arms, topics, keep=per_protocol_keep)[0]
        if not has_flags:
            o["per_protocol"] = {"skipped": "no flags: run `harness.run flags` or pass --threshold"}
        if has_adh:
            o["adherence"] = {c: slope_model(rows, name, None if c == "pooled" else c, arms, topics,
                                             adherence=True)[0] for c in cols}
        if len(arms) > 1:
            o["arm_x"] = {t: slope_model(rows, name, t, arms, topics, arm_x=True)[0] for t in topics}
        o["open_x"] = {t: slope_model(rows, name, t, arms, topics, open_x=True)[0] for t in topics}
        res["outcomes"][name] = o

    if PRIMARY in res["outcomes"]:
        o = res["outcomes"][PRIMARY]
        h1 = [(t, o["slope"][t].get("slope")) for t in topics]
        adj = stats.holm([c["p"] if c else None for _, c in h1])
        res["H1"] = [{"topic": t, **(c or {}), "p_holm": p} for (t, c), p in zip(h1, adj)]
        h2 = [(t, o["level"][t].get("shape") or {}) for t in topics]
        adj = stats.holm([sh["A"]["p"] if sh.get("A") else None for _, sh in h2])
        res["H2"] = [{"topic": t, **sh, "p_holm": p, "classification": classify(sh, p)}
                     for (t, sh), p in zip(h2, adj)]
        for t in topics:
            taus = o["level"][t].get("tau") or {}
            adj = stats.holm([v["p"] if v else None for v in taus.values()])
            for v, p in zip(taus.values(), adj):
                if v:
                    v["p_holm_within_topic"] = p

    judge, judges = stance_judge(runs, stance_prefix)
    h3 = {"judge": judge, "judges": judges}
    if judge and has_adh:
        tr = turn_rows(runs, judge)
        h3["turn"] = {c: h3_turn(tr, None if c == "pooled" else c, arms) for c in cols}
        h3["turn_cumulative"] = {c: h3_turn(tr, None if c == "pooled" else c, arms, "adh_cum") for c in cols}
        h3["agreement"] = judge_agreement(runs, [judge] + [j for j in judges if j != judge])
        if h3["agreement"].get("n"):
            sub = {tuple(k) for k in h3["agreement"]["dyads"]}
            h3["subsample_refit"] = {j[:12]: h3_turn(turn_rows(runs, j, sub), None, arms)
                                     for j in h3["agreement"]["judges"]}
        h3["trajectory"] = trajectory(tr)
        h3["control_alignment"] = control_alignment(runs, judge)
    else:
        h3["skipped"] = ("no mentor alignment scores (score --scope stance)" if not judge
                         else "no adherence scores")
    h3["adherence_drift"] = adherence_drift(runs) if has_adh else {}
    res["H3"] = h3

    fam = []
    prim = res["outcomes"].get(PRIMARY)
    for t in topics:
        if h3.get("turn"):
            fam.append(("H3 theta (turn level)", t, (h3["turn"].get(t) or {}).get("theta")))
        if prim and "adherence" in prim:
            fam.append(("H3 dose x adherence (dyad level)", t,
                        (prim["adherence"].get(t) or {}).get("dose_x_adherence")))
        if prim and "arm_x" in prim:
            w = (prim["arm_x"].get(t) or {}).get("arm_x_wald")
            fam.append(("S1 arm x dose (Wald F)", t,
                        w and {"estimate": w["F"], "p": w["p"], "df": w["df2"]}))
        for name, label in SECONDARY.items():
            if name in res["outcomes"]:
                fam.append((f"{label} {name} on {'dose' if name in SIGNED else 'intensity'}", t,
                            res["outcomes"][name]["slope"][t].get("slope")))
        if prim:
            fam.append(("S5 open x dose", t, (prim["open_x"].get(t) or {}).get("open_x")))
    q = stats.bh([c["p"] if c else None for _, _, c in fam])
    res["secondary"] = [{"test": lab, "topic": t, **(c or {}), "q_bh": qq}
                        for (lab, t, c), qq in zip(fam, q)]

    if prim:
        rob = {"two_stage": {t: two_stage(rows, PRIMARY, t, arms) for t in topics},
               "stacked": {t: stacked(item_rows, t, arms) for t in topics} if item_rows else {}}
        if len(arms) > 1:
            rob["by_arm_variance"] = {}
            for t in topics:
                w = arm_weights(rows, PRIMARY, t, arms, topics)
                sl = slope_model(rows, PRIMARY, t, arms, topics, weights_by_arm=w)[0]
                lv = level_model(rows, PRIMARY, t, arms, topics, weights_by_arm=w)[0]
                rob["by_arm_variance"][t] = {"weights": w, "slope": sl.get("slope"),
                                             "A": (lv.get("shape") or {}).get("A")}
        tests = [s["incompleteness_test"]["p"] for s in res["runs"]]
        if any(p is not None and p < 0.05 for p in tests):
            rob["lee_bounds"] = {t: lee(rows, PRIMARY, t) for t in topics}
        else:
            rob["lee_bounds"] = {"skipped": "incompleteness does not differ by level (p >= 0.05) in any arm"}
        rob["sensitivity"] = {}
        for rule, rr in (sensitivity or {}).items():
            rob["sensitivity"][rule] = {}
            for t in topics:
                sl = slope_model(rr, PRIMARY, t, arms, topics)[0]
                lv = level_model(rr, PRIMARY, t, arms, topics)[0]
                rob["sensitivity"][rule][t] = {"slope": sl.get("slope"),
                                               "A": (lv.get("shape") or {}).get("A")}
        res["robustness"] = rob
    return res


# ---- report ---------------------------------------------------------------------------------------

def _p(p) -> str:
    return "n/a" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def _c(c, d=False) -> str:
    """estimate (SE) [CI] p df, and d when asked."""
    if not c or "se" not in c:
        return ""
    out = f"{fmt(c['estimate'])} ({fmt(c['se'])}) [{fmt(c['ci'][0])}, {fmt(c['ci'][1])}] p {_p(c.get('p'))}"
    if c.get("df") is not None and math.isfinite(c["df"]):
        out += f" df {c['df']:.1f}"
    if d and c.get("d") is not None:
        out += f" d {c['d']:.2f}"
    return out


def _rho(rho: dict) -> str:
    if not rho:
        return ""
    lo, hi = rho["ci"]
    ci = {"interval": f"[{fmt(lo)}, {fmt(hi)}]", "complement": f"outside ({fmt(lo)}, {fmt(hi)})",
          "all": "the whole line"}[rho["kind"]]
    return f"{fmt(rho.get('ratio'))}, set {ci}"


def _confirmatory_md(res: dict) -> list[str]:
    h1 = [[h["topic"], _c(h, True), _p(h["p_holm"])] for h in res["H1"]]
    out = ["## Confirmatory", "", "H1, ideology dose slope on the ideological index (Holm across topics):",
           "", md_table(["topic", "beta", "p Holm"], h1), "",
           "H2, shape test (Holm across topics on A):", ""]
    rows = []
    for h in res["H2"]:
        rho = h.get("rho") or {}
        rows.append([h["topic"], _c(h.get("R")), _c(h.get("L")), _c(h.get("A"), True), _p(h["p_holm"]),
                     _rho(rho),
                     _c(h.get("intensity_right")), _c(h.get("intensity_left")), h["classification"]])
    return out + [md_table(["topic", "R", "L", "A", "p Holm", "rho = R/(-L), Fieller", "strong - lean right",
                            "lean - strong left", "reading"], rows), ""]


def _outcome_md(name: str, o: dict, cols: list[str], topics: list[str]) -> list[str]:
    tag = " (primary)" if name == PRIMARY else f" ({SECONDARY.get(name, 'reported')})"
    out = [f"## {name}{tag}", "", "Level model, ITT against the bare control:", ""]
    levels = [lv for lv in LEVELS if any(lv in (o["level"][c].get("tau") or {}) for c in cols)]

    def tau_cell(c, lv):
        v = (o["level"][c].get("tau") or {}).get(lv)
        extra = f"; Holm {_p(v.get('p_holm_within_topic'))}" if v and "p_holm_within_topic" in v else ""
        return _c(v, True) + extra
    rows = [["control mean change"] + [_c(o["level"][c].get("control_mean")) for c in cols]]
    rows += [[lv] + [tau_cell(c, lv) for c in cols] for lv in levels]
    def n_cell(c):
        lv = o["level"][c]
        if lv.get("skipped"):
            return f"skipped: {lv['skipped']}"
        return f"{lv.get('n')} ({lv.get('n_control', '')})"
    rows.append(["n (control)"] + [n_cell(c) for c in cols])
    rows.append(["role variance / ICC"] + [f"{fmt(o['level'][c]['variance']['sigma2_group'])} / "
                                           f"{fmt(o['level'][c]['variance']['icc'])}"
                                           if o["level"][c].get("variance") else "" for c in cols])
    out += [md_table(["term"] + cols, rows), ""]
    term = next((o["slope"][c]["term"] for c in cols if "term" in o["slope"][c]), "slope")
    rows = [[c, _c(o["slope"][c].get("slope"), True), _c(o["slope"][c].get("T")), o["slope"][c].get("n")]
            for c in cols]
    out += ["Slope model:", "", md_table(["sample", term, "T (moderate - control)", "n"], rows), ""]
    cc = [[t, a, _c(v)] for t in topics for a, v in o["control_change"].get(t, {}).items()]
    if cc:
        out += ["Control's own change (dialogue per se):", "",
                md_table(["topic", "arm", "mean change"], cc), ""]
    if "arm_x" in o:
        rows = [[t, a, _c(v)] for t in topics for a, v in (o["arm_x"][t].get("per_arm_slope") or {}).items()]
        out += ["Per-arm slopes (S1 model):", "", md_table(["topic", "arm", "slope"], rows), ""]
    if o.get("adherence"):
        rows = [[c, _c(o["adherence"][c].get("dose_x_adherence")), _c(o["adherence"][c].get("adherence")),
                 o["adherence"][c].get("n")] for c in cols]
        out += ["Dyad-level adherence (H3; treated dyads, adherence is post-treatment):", "",
                md_table(["sample", f"{term} x adherence", "adherence", "n"], rows), ""]
    pp = o["per_protocol"]
    if pp.get("skipped"):
        out += [f"Per-protocol: {pp['skipped']}.", ""]
    else:
        rows = [[c, _c(pp["slope"][c].get("slope")), _c((pp["level"][c].get("shape") or {}).get("A")),
                 pp["level"][c].get("n")] for c in cols]
        out += ["Per-protocol (non-flagged; sensitivity, potentially biased):", "",
                md_table(["sample", term, "A", "n"], rows), ""]
    return out


def _h3_md(h3: dict, cols: list[str]) -> list[str]:
    out = ["## H3 at the turn level", ""]
    if h3.get("skipped"):
        out += [f"skipped: {h3['skipped']}", ""]
    else:
        rows = [[c, _c(h3["turn"][c].get("theta")), _c(h3["turn_cumulative"][c].get("theta")),
                 h3["turn"][c].get("n_rows", h3["turn"][c].get("skipped"))] for c in cols]
        out += [f"Stance judge {str(h3['judge'])[:12]}; agree = 2 x alignment - 1 on seeker adherence.", "",
                md_table(["sample", "theta (same turn)", "theta (cumulative mean)", "turns"], rows), ""]
        ag = h3.get("agreement", {})
        if ag.get("n"):
            low = "; below 0.667: the turn-level stance results are exploratory" if ag["exploratory"] else ""
            sub = (f" (second judge's subsample {', '.join(map(str, ag['subsample']))})"
                   if ag.get("subsample") else "")
            out.append(f"Two judges: Krippendorff alpha {fmt(ag['alpha'])} on {ag['n']} shared "
                       f"targets{sub}{low}")

            out += [f"- subsample refit with {j}: theta {_c(v.get('theta'))}"
                    for j, v in (h3.get("subsample_refit") or {}).items()]
            out.append("")
        ca = h3.get("control_alignment", {})
        out += [f"Control dyads' raw alignment (not modelled): n {ca.get('n')}, "
                f"mean {fmt(ca.get('mean'))}.", ""]
    if h3.get("adherence_drift"):
        rows = [[lv, v["n_dyads"], fmt(v.get("mean")), _c(v.get("slope_per_turn")),
                 f"{v['flagged']}/{v['flag_known']}"] for lv, v in h3["adherence_drift"].items()]
        out += ["Seeker adherence by level (manipulation check; flag rate by level):", "",
                md_table(["ideology", "dyads", "mean", "slope per turn (CR1 by dyad)", "flagged"], rows), ""]
    return out


def _robustness_md(rob: dict, topics: list[str]) -> list[str]:
    rows = []
    for t in topics:
        ts = rob["two_stage"].get(t, {})
        rows.append([t, "two-stage role means", _c(ts.get("slope")), _c(ts.get("A"))])
        st = (rob.get("stacked") or {}).get(t)
        if st:
            rows.append([t, "item-level stacked", _c((st.get("slope") or {}).get("slope")),
                         _c(((st.get("level") or {}).get("shape") or {}).get("A"))])
        if rob.get("by_arm_variance"):
            b = rob["by_arm_variance"][t]
            rows.append([t, "residual variance by arm", _c(b.get("slope")), _c(b.get("A"))])
        for rule, v in rob.get("sensitivity", {}).items():
            rows.append([t, rule, _c(v[t].get("slope")), _c(v[t].get("A"))])
    out = ["## Robustness (primary outcome)", "", md_table(["topic", "check", "slope", "A"], rows), ""]
    lb = rob.get("lee_bounds", {})
    if lb.get("skipped"):
        return out + [f"Lee bounds: {lb['skipped']}.", ""]
    rows = [[t, lv, f"[{fmt(v['lower'])}, {fmt(v['upper'])}]", v.get("trimmed")]
            for t in topics for lv, v in lb.get(t, {}).items()]
    return out + ["Lee bounds on each level against the control:", "",
                  md_table(["topic", "level", "bounds", "trimmed"], rows), ""]


def markdown(res: dict) -> str:
    topics = res["topics"]
    cols = topics + (["pooled"] if len(topics) > 1 else [])
    out = ["# Estimates", "",
           f"Arms: {', '.join(res['arms'])}. Topics: {', '.join(topics)}. Cells: estimate (SE) [95% CI] p "
           "df (Satterthwaite); d = estimate / control post SD. Change = post - pre, positive = rightward "
           "on the signed indices.", ""]
    for s in res["runs"]:
        out.append(f"- {s['arm']} ({s['run_id']}): planned {s['planned']}, ITT {s['itt']}, exclusions "
                   f"{s['by_reason']}; incompleteness against ideology p "
                   f"{_p(s['incompleteness_test']['p'])}")
    out += [f"- {n}" for n in res["notes"]] + [""]
    if "H1" in res:
        out += _confirmatory_md(res)
    for name, o in res["outcomes"].items():
        out += _outcome_md(name, o, cols, topics)
    if res.get("secondary"):
        rows = [[s["test"], s["topic"], _c(s) if "se" in s else
                 (f"{fmt(s.get('estimate'))} p {_p(s.get('p'))}" if s.get("estimate") is not None else ""),
                 _p(s["q_bh"])] for s in res["secondary"]]
        out += ["## Secondary family (Benjamini-Hochberg)", "",
                md_table(["test", "topic", "estimate", "q"], rows), ""]
    out += _h3_md(res.get("H3", {}), cols)
    if res.get("robustness"):
        out += _robustness_md(res["robustness"], topics)
    return "\n".join(out) + "\n"


SENSITIVITY = (("json-only", {"json_only": True}), ("available items", {"available": True}),
               ("refusals at midpoint", {"impute_refusals": "mid"}),
               ("refusals at left end", {"impute_refusals": "low"}),
               ("refusals at right end", {"impute_refusals": "high"}))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m analysis.estimate", description=__doc__.split("\n\n")[0])
    add_common_args(p, multi=True)
    add_instrument_args(p)
    p.add_argument("--persona-mode", default=None, choices=("once", "reinforced"),
                   help="keep one delivery mode (E1 stated-once cells are a separate factor)")
    p.add_argument("--outcome", action="append", default=None, help="outcome index; repeat (default all)")
    p.add_argument("--stance-judge", default=None, help="judge_sha256 prefix of the primary stance judge")
    p.add_argument("--no-sensitivity", action="store_true", help="skip the §6 survey sensitivity refits")
    a = p.parse_args(argv)
    runs, rows, items, sens, names = [], [], [], {}, None
    try:
        for rd in a.run_dir:
            run = load_from_args(a, rd)
            inst = load_instrument(run, a.batteries, a.allow_instrument_mismatch)
            r, counts = compute_outcomes(run, inst, a.json_only, a.available_items, a.impute_refusals)
            names = names or counts["indices"]
            runs.append(run)
            rows += r
            if "ideological" in inst.indices:
                items += item_changes(run, inst)
            if not a.no_sensitivity:
                for rule, kw in SENSITIVITY:
                    sens.setdefault(rule, []).extend(compute_outcomes(run, inst, log=False, **kw)[0])
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if len({run.arm for run in runs}) < len(runs):
        print("error: two run directories have the same mentor arm; pass one run directory per arm",
              file=sys.stderr)
        return 1
    if a.persona_mode:
        keep = lambda r: r["persona_mode"] == a.persona_mode  # noqa: E731
        rows, items = [r for r in rows if keep(r)], [r for r in items if keep(r)]
        sens = {k: [r for r in v if keep(r)] for k, v in sens.items()}
    res = estimate(runs, rows, items, a.outcome or names, a.stance_judge, sens)
    out = Path(a.out) if a.out else default_out(a.run_dir[0])
    md = markdown(res)
    out.mkdir(parents=True, exist_ok=True)
    (out / "estimates.md").write_text(md, encoding="utf-8")
    write_json(out / "estimates.json", res)
    print(md)
    print(f"wrote {out}/estimates.md, estimates.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
