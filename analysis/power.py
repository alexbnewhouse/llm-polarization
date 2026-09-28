"""Simulation-based power and minimum detectable effect (MDE) for the frozen grid.

    python3 analysis/power.py --sd 0.6 --icc-role 0.02
    python3 analysis/power.py --sd 0.4,0.6,0.8,1.0 --icc-role 0,0.02,0.05 --mde-only

The plan of record is docs/pap/pre-analysis-plan.md; the tables quoted from this script are in
docs/pap/README.md. Every input marked [pilot] there is a flag here, so the table is re-run, not re-derived,
once the pilot has measured it.

Assumptions (each one a modelling choice the pilot can overturn):
- Unit and outcome: one dyad; the change in the ideological index (post minus pre, 1-5 points, positive =
  rightward). Under the greedy, context-free pre-survey the harness administers (PAP section 10, option A)
  the pre is one constant per arm, so the outcome SD is the post-survey index SD within a cell (--sd,
  [pilot]). --pre-rho r models a pre sampled per dyad (option B) with the post's SD and within-cell
  pre/post correlation r: the residual variance becomes 2(1-r)sd^2 - icc*sd^2. r is about 0 when pre and
  post are separate draws from different contexts. Cohen's d is always effect / --sd, so A and B compare.
- Truth: linear in dose through the control. The control and `moderate` have mean 0; level d in -2..+2 has
  mean (effect/2)*d. So `effect` is strong_right minus control, and the slope is effect/2 per dose step.
  Arm, topic and openness effects are zero; the design is balanced, so they would cancel from every contrast.
- Role: a normal random intercept per persona variant (3 per ideology level), variance icc*SD^2 (--icc-role,
  [pilot]), shared by both topics, both openness levels and every arm. The same persona texts run
  everywhere, so a shared effect is the conservative case: it gives the fewest independent replicates.
  Control dyads have no role effect: the bare prompt is the fixed definition of the zero point, not a sample.
- Residual: independent normal, variance (1-icc)*SD^2 under option A.
- Estimator: the PAP's mixed model (role random intercept nested in ideology) in the form it takes on
  balanced data, the nested ANOVA. Level contrasts are tested against the between-role-within-level mean
  square MS_R, with df = roles - levels (10 for five levels, 6 for three).
    slope:              sum(d * level mean) / sum(d^2), SE^2 = MS_R / (3 m sum(d^2)), t with df_R
    strong vs control:  level(+2) mean - control mean, SE^2 = MS_R / (3 m) + MS_E / n_control, and a
                        Satterthwaite df. m = dyads per role; MS_E = within-cell mean square.
  OLS with CR1 standard errors clustered by role was tried first and rejected. With 15 role clusters, and
  3 behind any one level, its size at a nominal 0.05 was 0.08 to 0.15 in this simulation.
- Power is the share of --sims draws with |t| > t_crit. The truth moves only the level means, so the mean
  squares do not depend on the effect. One set of draws therefore serves every effect size, and the MDE at
  --target power (0.80) is found by bisection on those same draws.
"""
from __future__ import annotations
import argparse
import math
import sys
import numpy as np

LEVELS = {5: (-2, -1, 0, 1, 2), 3: (-2, 0, 2)}
ROLES_PER_LEVEL = 3


def t_crit(alpha: float, df: float) -> float:
    """Two-sided critical value of Student's t with `df` degrees of freedom (numerical CDF; no scipy)."""
    if df > 1e5:
        df = 1e5
    x = np.linspace(0.0, 50.0, 250001)
    logc = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    pdf = np.exp(logc - (df + 1) / 2 * np.log1p(x * x / df))
    cdf = 0.5 + np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(x))])
    return float(np.interp(1 - alpha / 2, cdf, x))


def design(n_per_cell: int, n_levels: int = 5, arms: int = 3, topics: int = 1) -> dict:
    """Cell sizes for `arms` x `topics` copies of the grid: n_levels ideology x 3 roles x 2 openness, each
    role-cell of n_per_cell/3 dyads, plus one control cell of n_per_cell per arm and topic."""
    if n_per_cell % ROLES_PER_LEVEL:
        raise ValueError(f"n_per_cell {n_per_cell} does not split across {ROLES_PER_LEVEL} role variants")
    n_roles = n_levels * ROLES_PER_LEVEL
    per_role_cell = n_per_cell // ROLES_PER_LEVEL
    cells_per_role = 2 * arms * topics            # openness x arm x topic, each holding the same role
    n_control = n_per_cell * arms * topics
    n = n_roles * cells_per_role * per_role_cell + n_control
    return {"levels": np.array(LEVELS[n_levels], float), "n_roles": n_roles, "cells_per_role": cells_per_role,
            "per_role_cell": per_role_cell, "m": cells_per_role * per_role_cell, "n_control": n_control,
            "control_cells": arms * topics, "n": n, "df_r": n_roles - n_levels,
            "df_e": n - n_roles * cells_per_role - arms * topics}


def null_draws(dz: dict, se_role: float, se_e: float, sims: int, rng: np.random.Generator,
               batch: int = 250) -> dict[str, tuple[np.ndarray, ...]]:
    """Simulate `sims` data sets with zero effect. Returns {contrast: (estimates, SEs, dfs)}.

    Draws the sufficient statistics directly: each role-cell mean is its role's intercept plus mean noise,
    and the within-cell sum of squares is sigma_e^2 times a chi-square on df_e."""
    lv, k, c, pc = dz["levels"], dz["n_roles"], dz["cells_per_role"], dz["per_role_cell"]
    ssd = float((lv ** 2).sum())
    parts: dict[str, list] = {"slope": [[], [], []], "strong_vs_control": [[], [], []]}
    done = 0
    while done < sims:
        s = min(batch, sims - done)
        u = rng.normal(0.0, se_role, (k, 1, s))
        cell_means = u + rng.normal(0.0, se_e / math.sqrt(pc), (k, c, s))
        role_means = cell_means.mean(axis=1)                                   # (roles, s)
        level_means = role_means.reshape(len(lv), ROLES_PER_LEVEL, s).mean(axis=1)
        control_mean = rng.normal(0.0, se_e / math.sqrt(dz["n_control"]), s)
        dev = role_means.reshape(len(lv), ROLES_PER_LEVEL, s) - level_means[:, None, :]
        ms_r = dz["m"] * (dev ** 2).sum(axis=(0, 1)) / dz["df_r"]
        ms_e = se_e ** 2 * rng.chisquare(dz["df_e"], s) / dz["df_e"]
        slope = (lv[:, None] * level_means).sum(axis=0) / ssd
        a = ms_r / (ROLES_PER_LEVEL * dz["m"])
        b = ms_e / dz["n_control"]
        for name, est, se2, df in (
                ("slope", slope, a / ssd, np.full(s, float(dz["df_r"]))),
                ("strong_vs_control", level_means[-1] - control_mean, a + b,
                 (a + b) ** 2 / (a ** 2 / dz["df_r"] + b ** 2 / dz["df_e"]))):
            for store, val in zip(parts[name], (est, np.sqrt(se2), df)):
                store.append(val)
        done += s
    return {k2: tuple(np.concatenate(v) for v in vals) for k2, vals in parts.items()}


def crit_for(dfs: np.ndarray, alpha: float) -> np.ndarray:
    """Critical values per draw, interpolated in 1/df from a grid of exact ones."""
    grid = np.unique(np.r_[np.geomspace(max(1.0, dfs.min()), max(2.0, dfs.max()), 25), dfs.min(), dfs.max()])
    vals = np.array([t_crit(alpha, g) for g in grid])
    return np.interp(1 / dfs, (1 / grid)[::-1], vals[::-1])


def truth(model: str, effect: float) -> float:
    """The true value of a tested contrast for a strong_right-minus-control effect."""
    return effect / 2 if model == "slope" else effect


def power(draws: tuple[np.ndarray, np.ndarray, np.ndarray], theta: float) -> float:
    """Share of draws that reject when the tested contrast's true value is theta; draws carry crit."""
    est, se, crit = draws
    return float(np.mean(np.abs(est + theta) / se > crit))


def mde(draws, model: str, target: float = 0.8, hi: float = 20.0) -> float:
    """The smallest effect (strong_right minus control, points) with power >= target, by bisection."""
    lo = 0.0
    if power(draws, truth(model, hi)) < target:
        return float("nan")
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if power(draws, truth(model, mid)) < target else (lo, mid)
    return hi


def run(n_per_cell: int, sd: float, icc: float, n_levels: int = 5, arms: int = 3, topics: int = 1,
        alpha: float = 0.05, sims: int = 2000, seed: int = 20260928, pre_rho: float | None = None) -> dict:
    """Simulate one configuration: {contrast: (estimates, SEs, critical values)}, plus n and df_R."""
    total = sd ** 2 if pre_rho is None else 2 * (1 - pre_rho) * sd ** 2
    if total <= icc * sd ** 2:
        raise ValueError("pre_rho leaves no residual variance; lower it")
    dz = design(n_per_cell, n_levels, arms, topics)
    raw = null_draws(dz, math.sqrt(icc) * sd, math.sqrt(total - icc * sd ** 2), sims,
                     np.random.default_rng(seed))
    draws = {k: (est, se, crit_for(df, alpha)) for k, (est, se, df) in raw.items()}
    return {"draws": draws, "df_r": dz["df_r"], "n": dz["n"]}


def floats(s: str) -> list[float]:
    """A comma-separated list of floats."""
    return [float(v) for v in s.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    """CLI: the power curve for each N (five levels), then the MDE table per SD, ICC, N and level count."""
    ap = argparse.ArgumentParser(prog="analysis/power.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-per-cell", default="135,90", help="dyads per treated cell, comma list")
    ap.add_argument("--sd", default="0.6", help="post-survey index SD within a cell, points [pilot]; comma list")
    ap.add_argument("--icc-role", default="0.02", help="role share of the outcome variance [pilot]; comma list")
    ap.add_argument("--effect", default=None, help="strong_right minus control, index points; comma list")
    ap.add_argument("--effect-d", default="0,0.05,0.1,0.15,0.2,0.3,0.5",
                    help="the same effect as Cohen's d (points / --sd); used without --effect")
    ap.add_argument("--alpha", type=float, default=0.05, help="two-sided test size")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--arms", type=int, default=3, help="3 = the per-topic model pooled over arms; 1 = one arm")
    ap.add_argument("--topics", type=int, default=1, help="1 = the per-topic model; 2 = the pooled model")
    ap.add_argument("--pre-rho", type=float, default=None, help="model a sampled pre (option B) with this r")
    ap.add_argument("--target", type=float, default=0.8, help="power at which the MDE is read")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--mde-only", action="store_true", help="skip the power curves")
    a = ap.parse_args(argv)
    ns = [int(v) for v in a.n_per_cell.split(",")]
    pre = "greedy constant" if a.pre_rho is None else f"sampled, rho={a.pre_rho}"
    print(f"# arms={a.arms} topics={a.topics} alpha={a.alpha} sims={a.sims} target={a.target} "
          f"pre={pre} seed={a.seed}")
    print("# effect = strong_right minus control (index points); slope = effect/2 per dose step; d = points / --sd")
    rows = []
    for sd in floats(a.sd):
        for icc in floats(a.icc_role):
            for n in ns:
                for lv in (5, 3):
                    r = run(n, sd, icc, lv, a.arms, a.topics, a.alpha, a.sims, a.seed, a.pre_rho)
                    d = r["draws"]
                    if not a.mde_only and lv == 5:
                        effects = floats(a.effect) if a.effect else [x * sd for x in floats(a.effect_d)]
                        print(f"\n## power: N={n}/cell, five levels, SD={sd}, icc_role={icc}, "
                              f"n={r['n']}, df_R={r['df_r']}")
                        print(f"{'effect':>8} {'d':>6} {'slope':>7} {'strong-vs-control':>18}")
                        for e in effects:
                            print(f"{e:8.3f} {e / sd:6.3f} {power(d['slope'], truth('slope', e)):7.3f} "
                                  f"{power(d['strong_vs_control'], e):18.3f}")
                    rows.append((sd, icc, n, lv, mde(d["slope"], "slope", a.target),
                                 mde(d["strong_vs_control"], "strong_vs_control", a.target)))
    print(f"\n## MDE at {a.target:.0%} power (index points; d in brackets)")
    print(f"{'SD':>5} {'icc':>5} {'N':>4} {'levels':>6} {'slope/step':>15} {'strong-vs-control':>19}")
    for sd, icc, n, lv, ms, mc in rows:
        print(f"{sd:5.2f} {icc:5.2f} {n:4d} {lv:6d} {ms / 2:7.3f} [{ms / 2 / sd:4.2f}] "
              f"{mc:10.3f} [{mc / sd:4.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
