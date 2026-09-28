"""analysis.stats: the tail functions against reference values, OLS and cluster SEs, and the REML random
intercept recovering a planted variance; plus the calibration statistics."""
import numpy as np
import pytest
from analysis import stats
from analysis.calibrate import allocate, best_threshold, krippendorff_interval, turn_bin, weighted_kappa


def test_tails_match_reference_values():
    # reference values from scipy.stats: t.sf(2.228, 10), chi2.sf(3.841, 1), chi2.sf(20, 10), norm.sf(1.96)
    assert stats.t_sf(2.228, 10) == pytest.approx(0.025006, abs=1e-5)
    assert stats.chi2_sf(3.841, 1) == pytest.approx(0.050014, abs=1e-5)
    assert stats.chi2_sf(20, 10) == pytest.approx(0.029253, abs=1e-5)
    assert stats.normal_sf(1.96) == pytest.approx(0.024998, abs=1e-5)
    assert stats.t_crit(10) == pytest.approx(2.228139, abs=1e-5)
    assert stats.t_crit(float("inf")) == pytest.approx(1.959964, abs=1e-5)
    assert stats.betainc(2, 3, 0.4) == pytest.approx(0.5248, abs=1e-9)
    assert stats.t_pvalue(0.0, 5) == pytest.approx(1.0)


def test_chi2_independence_drops_empty_columns_and_detects_dependence():
    flat = stats.chi2_independence([[5, 5, 5, 0], [45, 45, 45, 0]])
    assert flat["df"] == 2 and flat["chi2"] == pytest.approx(0.0) and flat["p"] == pytest.approx(1.0)
    skew = stats.chi2_independence([[20, 2, 2], [30, 48, 48]])
    assert skew["p"] < 1e-6
    assert stats.chi2_independence([[0, 0], [5, 5]])["p"] is None


def test_ols_hc1_and_cluster_robust():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = 1 + 2 * x + rng.normal(size=400)
    X = np.column_stack([np.ones(400), x])
    f = stats.ols(X, y, ["c", "x"])
    assert f.coef("x")["estimate"] == pytest.approx(2, abs=0.15) and f.df == 398
    g = stats.ols(X, y, ["c", "x"], clusters=np.repeat(np.arange(40), 10))
    assert g.df == 39 and g.extra["n_clusters"] == 40
    assert np.allclose(f.beta, g.beta)
    assert f.lincom({"c": 1, "x": 1})["estimate"] == pytest.approx(f.beta.sum())
    assert f.lincom({"missing": 1}) is None
    with pytest.raises(ValueError):
        stats.ols(np.column_stack([np.ones(10), np.ones(10)]), np.arange(10.0), ["a", "b"])


def test_random_intercept_recovers_the_group_variance_and_leaves_ungrouped_rows_alone():
    rng = np.random.default_rng(1)
    G, n = 60, 8
    g = np.repeat(np.arange(G), n)
    x = rng.normal(size=G * n)
    y = 1 + 2 * x + rng.normal(0, 0.7, G)[g] + rng.normal(0, 1, G * n)
    X = np.column_stack([np.ones_like(x), x])
    f = stats.random_intercept(X, y, ["c", "x"], list(g))
    assert f.extra["sigma2_group"] == pytest.approx(0.49, abs=0.2)
    assert f.extra["sigma2_resid"] == pytest.approx(1.0, abs=0.15)
    assert abs(f.coef("x")["estimate"] - 2) < 3 * f.coef("x")["se"]
    # no group structure: the variance goes to zero and the fit is OLS
    y0 = 1 + 2 * x + rng.normal(0, 1, G * n)
    f0 = stats.random_intercept(X, y0, ["c", "x"], list(g))
    assert f0.extra["sigma2_group"] < 0.05
    # rows with group None (the bare control) carry no random effect
    groups = [None if i < 80 else int(v) for i, v in enumerate(g)]
    f1 = stats.random_intercept(X, y, ["c", "x"], groups, df=7)
    assert f1.df == 7 and f1.extra["n_groups"] == len(set(g[80:]))


def test_krippendorff_alpha_interval():
    assert krippendorff_interval([[1, 1], [2, 2], [3, 3]]) == pytest.approx(1.0)
    # a hand-computed case: units (1,2), (3,3): n=4 values 1,2,3,3. Do = (2*1)/4 = 0.5.
    # De = sum over ordered pairs / (4*3)
    # pairs: (1-2)^2=1, (1-3)^2=4 x2, (2-3)^2=1 x2, (3-3)=0 -> 2*(1+8+2)=22 -> 22/12. alpha = 1 - 0.5/(22/12)
    assert krippendorff_interval([[1, 2], [3, 3]]) == pytest.approx(1 - 0.5 / (22 / 12))
    assert krippendorff_interval([[1, None], [2]]) is None
    rng = np.random.default_rng(2)
    noise = [[a, b] for a, b in zip(rng.normal(size=500), rng.normal(size=500))]
    assert abs(krippendorff_interval(noise)) < 0.1


def test_weighted_kappa_and_threshold():
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert weighted_kappa([1, 2, 3], [3, 2, 1], [1, 2, 3]) < 0
    t = best_threshold([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [False, False, False, True, True, True])
    assert 0.3 < t["threshold"] < 0.7 and t["balanced_accuracy"] == pytest.approx(1.0)
    assert best_threshold([0.1, 0.2], [True, True])["threshold"] is None


def test_turn_bins_and_allocation():
    assert [turn_bin(t, 40, 4) for t in (1, 10, 11, 40)] == [1, 1, 2, 4]
    assert allocate({"a": 100, "b": 100, "c": 100}, 120) == {"a": 40, "b": 40, "c": 40}
    # a stratum too small to fill its share passes the remainder on
    assert allocate({"a": 5, "b": 100, "c": 100}, 120) == {"a": 5, "b": 58, "c": 57}
    assert sum(allocate({"a": 3, "b": 2}, 100).values()) == 5


def _nested_grid(seed=3, m=20, role_sd=0.3):
    """5 levels x 3 roles x m dyads plus 40 control dyads (no role)."""
    rng = np.random.default_rng(seed)
    rows = []
    for lv in range(5):
        for r in range(3):
            u = rng.normal(0, role_sd)
            rows += [(lv, f"{lv}/{r}", 0.1 * lv + u + rng.normal(0, 0.6)) for _ in range(m)]
    rows += [(-1, None, rng.normal(0, 0.6)) for _ in range(40)]
    y = np.array([r[2] for r in rows])
    X = np.array([[1.0] + [1.0 if r[0] == lv else 0.0 for lv in range(5)] for r in rows])
    return rows, X, y


def test_reml_with_satterthwaite_is_the_nested_anova_on_a_balanced_grid():
    """The PAP's power calculation tests level contrasts against the between-role mean square with
    df = roles - levels; the mixed model must give the same SE and df on balanced data."""
    m = 20
    rows, X, y = _nested_grid(m=m)
    f = stats.mixed(X, y, ["c"] + [f"l{i}" for i in range(5)], [r[1] for r in rows])
    means: dict = {}
    for lv, g, v in rows:
        if g:
            means.setdefault(g, []).append(v)
    by_level: dict = {}
    for g, v in means.items():
        by_level.setdefault(int(g.split("/")[0]), []).append(np.mean(v))
    ms_r = m * sum(sum((x - np.mean(v)) ** 2 for x in v) for v in by_level.values()) / (15 - 5)
    c = f.lincom({"l4": 1, "l0": -1})
    assert c["estimate"] == pytest.approx(np.mean(by_level[4]) - np.mean(by_level[0]))
    assert c["se"] == pytest.approx(np.sqrt(2 * ms_r / (3 * m)), rel=1e-4)
    assert c["df"] == pytest.approx(10, abs=0.05)
    assert "Satterthwaite" in f.method
    # a level against the control mixes the two variances: Welch-like df between 10 and n - k
    assert 10 < f.lincom({"l4": 1})["df"] < len(y) - 6


def test_two_nested_random_intercepts():
    rng = np.random.default_rng(4)
    rows, outer, inner = [], [], []
    for g in range(15):
        u = rng.normal(0, 0.4)
        for d in range(30):
            v = rng.normal(0, 0.5)
            for _ in range(10):
                x = rng.normal()
                rows.append((x, 0.4 * x + u + v + rng.normal(0, 0.7)))
                outer.append(g); inner.append(f"{g}/{d}")
    X = np.array([[1.0, r[0]] for r in rows]); y = np.array([r[1] for r in rows])
    f = stats.mixed(X, y, ["c", "x"], outer, inner)
    assert f.extra["sigma2_group"] == pytest.approx(0.16, abs=0.12)
    assert f.extra["sigma2_inner"] == pytest.approx(0.25, abs=0.08)
    assert f.extra["sigma2_resid"] == pytest.approx(0.49, abs=0.05)
    assert f.coef("x")["estimate"] == pytest.approx(0.4, abs=0.03)
    assert f.coef("c")["df"] < 20                   # the intercept rests on 15 groups
    # an inner group with no outer one (a control dyad) takes the dyad variance only
    outer2 = [None if o < 3 else o for o in outer]
    f2 = stats.mixed(X, y, ["c", "x"], outer2, inner)
    assert f2.extra["n_groups"] == 12 and f2.extra["n_inner"] == 450


def test_wald_holm_bh_fieller_welch_and_lee():
    rows, X, y = _nested_grid()
    f = stats.mixed(X, y, ["c"] + [f"l{i}" for i in range(5)], [r[1] for r in rows])
    w = f.wald([{"l4": 1, "l0": -1}, {"l3": 1, "l1": -1}])
    assert w["df1"] == 2 and 0 <= w["p"] <= 1 and w["df2"] == pytest.approx(10, abs=0.05)
    assert stats.f_sf(4.10, 2, 10) == pytest.approx(0.05, abs=0.002)
    assert stats.holm([0.01, 0.04, None, 0.03]) == [0.03, 0.06, None, 0.06]
    assert stats.bh([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    q = stats.t_crit(1e9)
    iv = stats.fieller(2.0, 1.0, 0.01, 0.01, 0.0, q)
    assert iv["kind"] == "interval" and iv["ci"][0] < 2 < iv["ci"][1] and not stats.fieller_contains(iv, 1.0)
    comp = stats.fieller(1.0, 0.05, 0.01, 0.04, 0.0, q)         # b indistinguishable from 0, a is not
    assert comp["kind"] == "complement" and not stats.fieller_contains(comp, 1.0)
    assert stats.fieller_contains(comp, 1000.0)
    assert stats.fieller(0.01, 0.01, 0.04, 0.04, 0.0, q)["kind"] == "all"
    var, df = stats.welch([(1, 0.01, 10), (1, 0.01, 10)])
    assert var == pytest.approx(0.02) and df == pytest.approx(20)
    # 10 of 10 level dyads observed, 8 of 10 control: trim 2 of the level's 10
    lb = stats.lee_bounds(list(range(10)), 10, [0.0] * 8, 10)
    assert lb["trimmed"] == "level" and lb["trimmed_n"] == 2
    assert lb["lower"] == pytest.approx(np.mean(range(8)))
    assert lb["upper"] == pytest.approx(np.mean(range(2, 10)))
