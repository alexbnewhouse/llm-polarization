"""The statistics the analysis needs, in numpy: OLS with HC1 or cluster-robust (CR1) standard errors, a
linear mixed model with one random intercept or two nested ones, fitted by REML with Satterthwaite degrees
of freedom, linear contrasts and Wald tests, Holm and Benjamini-Hochberg adjustment, Fieller intervals,
Lee bounds, and the t, F, chi-square and normal tails behind the p-values.

Why not statsmodels: it is not in the environment the harness pins (harness/requirements.lock), and a
second numerical stack is one more thing to pin for a result. When scipy is installed its special
functions are used for the tails; otherwise the continued fractions below (Numerical Recipes, 6.2 and
6.4), which agree with scipy to about 1e-10 on the tests' reference values.

The random intercepts are fitted exactly, not approximated by demeaning: role is nested in ideology, so
within-role demeaning (role fixed effects) would absorb the ideology effect itself. `mixed` works from
per-group sums. V is block diagonal by outer group; each block is I + l2 (inner blocks of ones) + l1 (a
block of ones), whose inverse and determinant have closed forms (Sherman-Morrison twice), so a turn-level
model with a dyad intercept inside a role intercept stays cheap. On a balanced grid the REML fit with
Satterthwaite df is the nested ANOVA the pre-analysis plan's power calculation uses (docs/pap, sections 7
and 9); harness/tests/test_analysis_stats.py checks that."""
from __future__ import annotations
import functools
import math
from dataclasses import dataclass, field
import numpy as np

try:                                                    # optional: exact tails when scipy is present
    from scipy import special as _sp                    # type: ignore
except ImportError:                                     # the fallback below is what runs in this repo
    _sp = None

SPECIAL_BACKEND = "scipy" if _sp is not None else "numpy continued fractions"


# ---- tail functions -------------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if _sp is not None:
        return float(_sp.betainc(a, b, x))
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x)."""
    if _sp is not None:
        return float(_sp.gammaincc(a, x))
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:                                       # series for P, then 1 - P
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    tiny = 1e-300                                       # continued fraction for Q
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b; d = 1 / (d if abs(d) > tiny else tiny)
        c = b + an / c; c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def normal_sf(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with `df` degrees of freedom; the normal when df is None or infinite."""
    if df is None or not math.isfinite(df):
        return normal_sf(t)
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2, 0.5, x)
    return tail if t > 0 else 1 - tail


def t_pvalue(t: float, df: float) -> float:
    """Two-sided p-value."""
    if t != t:
        return float("nan")
    return min(1.0, 2 * t_sf(abs(t), df))


def t_crit(df: float, level: float = 0.95) -> float:
    """The two-sided critical value, by bisection on t_sf (cached on df to 4 decimals)."""
    return _t_crit(round(float(df), 4) if df is not None and math.isfinite(df) else float("inf"), level)


@functools.lru_cache(maxsize=4096)
def _t_crit(df: float, level: float) -> float:
    target = (1 - level) / 2
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def chi2_sf(x: float, df: float) -> float:
    """P(X > x) for a chi-square with `df` degrees of freedom."""
    return gammaincc(df / 2, x / 2)


def chi2_independence(table: list[list[int]]) -> dict:
    """Pearson chi-square test of independence on a contingency table (rows x columns). Empty rows and
    columns are dropped. Returns {chi2, df, p, min_expected}; p is None when df is 0."""
    t = np.asarray(table, float)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0] if t.size else t
    if t.ndim != 2 or min(t.shape) < 2:
        return {"chi2": None, "df": 0, "p": None, "min_expected": None}
    exp = np.outer(t.sum(axis=1), t.sum(axis=0)) / t.sum()
    chi2 = float(((t - exp) ** 2 / exp).sum())
    df = (t.shape[0] - 1) * (t.shape[1] - 1)
    return {"chi2": chi2, "df": df, "p": chi2_sf(chi2, df), "min_expected": float(exp.min())}


# ---- models ---------------------------------------------------------------------------------------

def f_sf(f: float, d1: float, d2: float) -> float:
    """P(F > f) for an F distribution with (d1, d2) degrees of freedom."""
    if f <= 0:
        return 1.0
    if d2 is None or not math.isfinite(d2):
        return chi2_sf(f * d1, d1)
    return betainc(d2 / 2, d1 / 2, d2 / (d2 + d1 * f))


@dataclass
class Fit:
    """A fitted linear model: coefficients by name, their covariance, how the SEs were made, and the
    degrees of freedom of a contrast (a fixed `df`, or `df_fn(c)` for Satterthwaite)."""
    names: list[str]
    beta: np.ndarray
    vcov: np.ndarray
    n: int
    df: float
    method: str
    extra: dict = field(default_factory=dict)
    df_fn: object = None

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.vcov), 0, None))

    def vec(self, weights: dict[str, float]) -> np.ndarray | None:
        """The contrast vector for {name: weight}; None when a name is not in the model."""
        if any(k not in self.names for k in weights):
            return None
        c = np.zeros(len(self.names))
        for k, w in weights.items():
            c[self.names.index(k)] = w
        return c

    def contrast_df(self, c: np.ndarray) -> float:
        return self.df_fn(c) if self.df_fn is not None else self.df

    def coef(self, name: str) -> dict | None:
        """Estimate, SE, t, df, p and 95% CI of one coefficient."""
        return self.lincom({name: 1.0})

    def lincom(self, weights: dict[str, float], level: float = 0.95) -> dict | None:
        """A linear combination sum(w * beta) of named coefficients, with its SE, t, df, p and CI. Names not
        in the model make the result None (a level missing from the data, say)."""
        c = self.vec(weights)
        if c is None:
            return None
        est = float(c @ self.beta)
        se = math.sqrt(max(0.0, float(c @ self.vcov @ c)))
        df = self.contrast_df(c)
        t = est / se if se > 0 else float("nan")
        q = t_crit(df, level)
        return {"estimate": est, "se": se, "t": t, "df": df, "p": t_pvalue(t, df),
                "ci": [est - q * se, est + q * se]}

    def wald(self, rows: list[dict[str, float]]) -> dict | None:
        """Joint F test that every linear combination in `rows` is zero; the denominator df is the smallest
        contrast df among the rows (conservative)."""
        C = [self.vec(r) for r in rows]
        if not C or any(c is None for c in C):
            return None
        C = np.array(C)
        b = C @ self.beta
        f = float(b @ np.linalg.pinv(C @ self.vcov @ C.T) @ b) / len(rows)
        d2 = min(self.contrast_df(c) for c in C)
        return {"F": f, "df1": len(rows), "df2": d2, "p": f_sf(f, len(rows), d2)}


def _design_ok(X: np.ndarray) -> None:
    if X.shape[0] <= X.shape[1]:
        raise ValueError(f"{X.shape[0]} observations for {X.shape[1]} parameters")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("design matrix is rank deficient (a level with no observations, or collinear "
                         "terms)")


def ols(X, y, names: list[str], clusters=None) -> Fit:
    """OLS with HC1 standard errors, or CR1 cluster-robust ones when `clusters` (one label per row) is
    given. Degrees of freedom: n - k for HC1, G - 1 for CR1."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    _design_ok(X)
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    e = y - X @ beta
    if clusters is None:
        u = X * e[:, None]
        vcov = n / (n - k) * bread @ (u.T @ u) @ bread
        return Fit(list(names), beta, vcov, n, n - k, "OLS, HC1", {"resid": e})
    labels = np.asarray([str(c) for c in clusters])
    groups = np.unique(labels)
    g = len(groups)
    meat = np.zeros((k, k))
    for lab in groups:
        idx = labels == lab
        s = X[idx].T @ e[idx]
        meat += np.outer(s, s)
    adj = g / (g - 1) * (n - 1) / (n - k) if g > 1 else 1.0
    return Fit(list(names), beta, adj * bread @ meat @ bread, n, max(g - 1, 1),
               f"OLS, CR1 by cluster (G={g})", {"resid": e, "n_clusters": g})


def ols_classical(X, y, names: list[str]) -> Fit:
    """OLS with the classical covariance and df n - k: the second stage of the two-stage analysis, where
    every unit is a mean over the same design."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    _design_ok(X)
    n, k = X.shape
    inv = np.linalg.inv(X.T @ X)
    beta = inv @ X.T @ y
    s2 = float(((y - X @ beta) ** 2).sum()) / (n - k)
    return Fit(list(names), beta, s2 * inv, n, n - k, "OLS, classical", {"sigma2": s2})


class _Suff:
    """Per-subgroup sums for the nested random-intercept likelihood. A subgroup is an inner group (or an
    outer group when there is no inner level); rows with no outer group are in the totals only."""

    def __init__(self, X, y, outer, inner):
        self.n, self.p = X.shape
        self.XtX, self.Xty, self.yty = X.T @ X, X.T @ y, float(y @ y)
        keys: dict = {}
        outer_ids: dict = {}
        g_of_s = []
        idx = np.full(self.n, -1)
        for i, g in enumerate(outer):
            if g is None and (inner is None or inner[i] is None):
                continue
            key = (g, inner[i]) if inner is not None else (g,)
            if key not in keys:
                keys[key] = len(keys)
                # an inner group with no outer one (a control dyad in a turn-level model) gets l2 only
                g_of_s.append(-1 if g is None else outer_ids.setdefault(g, len(outer_ids)))
            idx[i] = keys[key]
        m = idx >= 0
        self.S, self.G = len(keys), len(outer_ids)
        self.N = np.bincount(idx[m], minlength=self.S).astype(float)
        self.SX = np.zeros((self.S, self.p)); np.add.at(self.SX, idx[m], X[m])
        self.SY = np.zeros(self.S); np.add.at(self.SY, idx[m], y[m])
        self.g_of_s = np.array(g_of_s, int)

    def core(self, l1: float, l2: float):
        """(X'WX, X'Wy, y'Wy, log|Vt|) for Vt = V / s2_e = I + l2 (inner ones) + l1 (outer ones) and
        W = Vt^-1."""
        d = 1.0 + self.N * l2
        c = l2 / d
        M = self.XtX - (self.SX * c[:, None]).T @ self.SX
        v = self.Xty - self.SX.T @ (c * self.SY)
        yy = self.yty - float((c * self.SY ** 2).sum())
        logdet = float(np.log(d).sum())
        if l1 > 0 and self.G:
            o = self.g_of_s >= 0
            gs = self.g_of_s[o]
            W = np.zeros((self.G, self.p)); np.add.at(W, gs, self.SX[o] / d[o, None])
            wy = np.zeros(self.G); np.add.at(wy, gs, self.SY[o] / d[o])
            q = np.zeros(self.G); np.add.at(q, gs, self.N[o] / d[o])
            k = l1 / (1.0 + l1 * q)
            M = M - (W * k[:, None]).T @ W
            v = v - W.T @ (k * wy)
            yy -= float((k * wy ** 2).sum())
            logdet += float(np.log1p(l1 * q).sum())
        return M, v, yy, logdet


def _neg_reml_profiled(suff: _Suff, l1: float, l2: float) -> float:
    M, v, yy, logdet = suff.core(l1, l2)
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return float("inf")
    rss = max(yy - float(v @ np.linalg.solve(M, v)), 1e-300)
    dfr = suff.n - suff.p
    return 0.5 * (dfr * math.log(rss / dfr) + logdet + 2 * float(np.log(np.diag(L)).sum()) + dfr)


def _golden(f, lo: float, hi: float, iters: int = 60) -> tuple[float, float]:
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = f(d)
    x = (a + b) / 2
    return x, f(x)


def _nelder_mead(f, x0, step: float = 1.0, iters: int = 300) -> tuple[np.ndarray, float]:
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + step * e for e in np.eye(len(x0))]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < 1e-10:
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = cen + (cen - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = cen + 2 * (cen - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen); fcn = f(xc)
            if fcn < vals[-1]:
                pts[-1], vals[-1] = xc, fcn
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [f(p) for p in pts]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


LOG_LO, LOG_HI = -12.0, 6.0          # search range for log(variance ratio); below LOG_LO counts as zero


def _optimise(suff: _Suff, has_inner: bool) -> tuple[float, float, float]:
    """REML estimates of the variance ratios (l1 outer, l2 inner) and the minimised criterion. Every
    boundary (either ratio zero) is a candidate beside the interior optimum."""
    f = lambda l1, l2: _neg_reml_profiled(suff, l1, l2)  # noqa: E731
    e = lambda u: math.exp(min(u, 30.0))                 # noqa: E731
    grid = np.linspace(LOG_LO, LOG_HI, 19)
    cands = [(0.0, 0.0, f(0.0, 0.0))]

    def line(fn):
        g0 = min(grid, key=fn)
        u, v = _golden(fn, max(LOG_LO, g0 - 1), min(LOG_HI, g0 + 1))
        return e(u), v
    if suff.G:
        l1, v = line(lambda u: f(e(u), 0.0))
        cands.append((l1, 0.0, v))
    if has_inner:
        l2, v = line(lambda u: f(0.0, e(u)))
        cands.append((0.0, l2, v))
        if suff.G:
            g2 = [((a, b), f(e(a), e(b))) for a in grid[::2] for b in grid[::2]]
            x, v = _nelder_mead(lambda x: f(e(x[0]), e(x[1])), min(g2, key=lambda t: t[1])[0])
            cands.append((e(x[0]), e(x[1]), v))
    l1, l2, v = min(cands, key=lambda t: t[2])
    lo = math.exp(LOG_LO)
    return (0.0 if l1 < lo else l1), (0.0 if l2 < lo else l2), v


def mixed(X, y, names: list[str], outer, inner=None, df: float | None = None) -> Fit:
    """Linear model with a random intercept per `outer` group and, when `inner` is given, one per inner
    group nested in it, fitted by REML: y = X b + u_outer + v_inner + e. A row whose outer label is None
    carries no outer effect (the bare control has no role), and no effect at all unless it has an inner
    label (a control dyad's turns still share a dyad intercept). Degrees of freedom are Satterthwaite's, per
    contrast, from a numerical REML information matrix; a variance estimated at zero drops out of it, and
    with none left the df is n - k. Pass `df` to fix it instead."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    _design_ok(X)
    n, k = X.shape
    outer = list(outer)
    inner = list(inner) if inner is not None else None
    suff = _Suff(X, y, outer, inner)
    l1, l2, crit = _optimise(suff, inner is not None)
    M, v, yy, _ = suff.core(l1, l2)
    Minv = np.linalg.inv(M)
    beta = Minv @ v
    s2 = (yy - float(v @ beta)) / (n - k)
    theta = {"s2e": s2, "s2u1": l1 * s2, "s2u2": l2 * s2}
    total = s2 + theta["s2u1"] + theta["s2u2"]
    extra = {"sigma2_resid": s2, "sigma2_group": theta["s2u1"],
             "sigma2_inner": theta["s2u2"] if inner else None,
             "n_groups": suff.G, "n_inner": suff.S if inner else None,
             "icc": theta["s2u1"] / total if total > 0 else None, "neg_reml": crit}
    label = f"REML random intercept ({suff.G} groups" + (f", {suff.S} nested)" if inner else ")")
    fit = Fit(list(names), beta, s2 * Minv, n, n - k, label, extra)
    if df is not None:
        fit.df = df
        return fit
    active = ["s2e"] + [a for a in ("s2u1", "s2u2") if theta[a] > 1e-8 * max(s2, 1e-300)]
    fit.df_fn = _satterthwaite(suff, theta, active, n, k)
    fit.method += ", Satterthwaite df"
    return fit


def _satterthwaite(suff: _Suff, theta: dict, active: list[str], n: int, k: int):
    """c -> Satterthwaite df for c'b: 2 Var(c'b)^2 / (g' A g), with g the gradient of Var(c'b) in the active
    variance parameters and A the inverse observed information of the unprofiled REML log-likelihood, both
    by central differences."""
    if len(active) == 1:
        return lambda c: float(n - k)

    def full(x):
        t = dict(theta)
        t.update(dict(zip(active, x)))
        return t

    def core(t):
        return suff.core(t["s2u1"] / t["s2e"], t["s2u2"] / t["s2e"])

    def ll(x):
        t = full(x)
        M, v, yy, logdet = core(t)
        rss = yy - float(v @ np.linalg.solve(M, v))
        logdet_m = float(np.linalg.slogdet(M)[1])
        return -0.5 * ((n - k) * math.log(t["s2e"]) + logdet + logdet_m + rss / t["s2e"])

    x0 = np.array([theta[a] for a in active])
    h = np.maximum(np.abs(x0) * 1e-3, 1e-10)
    m = len(x0)
    E = np.eye(m)
    H = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            a, b = E[i] * h[i], E[j] * h[j]
            d2 = ll(x0 + a + b) - ll(x0 + a - b) - ll(x0 - a + b) + ll(x0 - a - b)
            H[i, j] = H[j, i] = d2 / (4 * h[i] * h[j])
    try:
        A = np.linalg.inv(-H)
        if not np.all(np.isfinite(A)) or np.any(np.linalg.eigvalsh((A + A.T) / 2) <= 0):
            raise np.linalg.LinAlgError("information not positive definite")
    except np.linalg.LinAlgError:
        return lambda c: float(n - k)
    V = [[full(x0 + s * E[i] * h[i])["s2e"] * np.linalg.inv(core(full(x0 + s * E[i] * h[i]))[0])
          for s in (1, -1)] for i in range(m)]
    V0 = theta["s2e"] * np.linalg.inv(core(theta)[0])

    def df(c):
        var = float(c @ V0 @ c)
        g = np.array([(float(c @ V[i][0] @ c) - float(c @ V[i][1] @ c)) / (2 * h[i]) for i in range(m)])
        den = float(g @ A @ g)
        if den <= 0 or var <= 0:
            return float(n - k)
        return float(min(max(2 * var * var / den, 1.0), n - k))
    return df


def random_intercept(X, y, names: list[str], groups, df: float | None = None) -> Fit:
    """One random intercept per group (None = no random effect): `mixed` without an inner level."""
    return mixed(X, y, names, groups, None, df)


# ---- multiplicity, intervals, bounds --------------------------------------------------------------

def holm(ps: list) -> list:
    """Holm-adjusted p-values (family-wise), None passed through."""
    order = sorted((i for i, p in enumerate(ps) if p is not None), key=lambda i: ps[i])
    out: list = [None] * len(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * ps[i]))
        out[i] = run
    return out


def bh(ps: list) -> list:
    """Benjamini-Hochberg adjusted p-values (false discovery rate), None passed through."""
    order = sorted((i for i, p in enumerate(ps) if p is not None), key=lambda i: ps[i], reverse=True)
    m = len(order)
    out: list = [None] * len(ps)
    run = 1.0
    for r, i in enumerate(order):
        run = min(run, ps[i] * m / (m - r))
        out[i] = run
    return out


def fieller(a: float, b: float, vaa: float, vbb: float, vab: float, q: float) -> dict:
    """Fieller confidence set for a / b at critical value q: {rho: (a - rho b)^2 <= q^2 Var(a - rho b)}.
    `kind` is `interval` ([lo, hi]) when b is distinguishable from zero; otherwise `complement` (every
    value outside (lo, hi), when a is distinguishable from zero) or `all` (the whole line). `contains`
    reads any of the three."""
    A = b * b - q * q * vbb
    B = a * b - q * q * vab
    C = a * a - q * q * vaa
    disc = B * B - A * C
    ratio = a / b if b else None
    if A > 0 and disc >= 0:
        r = math.sqrt(disc)
        return {"ratio": ratio, "kind": "interval", "ci": [(B - r) / A, (B + r) / A]}
    if A < 0 and disc > 0:
        r = math.sqrt(disc)
        return {"ratio": ratio, "kind": "complement", "ci": sorted([(B - r) / A, (B + r) / A])}
    return {"ratio": ratio, "kind": "all", "ci": [None, None]}


def fieller_contains(f: dict, x: float) -> bool:
    """Whether the Fieller set `f` contains x."""
    lo, hi = f["ci"]
    if f["kind"] == "interval":
        return lo <= x <= hi
    if f["kind"] == "complement":
        return x <= lo or x >= hi
    return True


def welch(parts: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Variance and Welch-Satterthwaite df of a weighted sum of independent estimates, parts given as
    (weight, variance, df)."""
    var = sum(w * w * v for w, v, _ in parts)
    den = sum((w * w * v) ** 2 / d for w, v, d in parts if d and d > 0 and v > 0)
    return var, (var * var / den if den > 0 else float("inf"))


def lee_bounds(y1, n1: int, y0, n0: int) -> dict:
    """Lee (2009) bounds on mean(y1) - mean(y0) when the groups are observed at different rates (len(y)
    of n assigned): the better-observed group is trimmed by the difference, from the top for the lower
    bound and from the bottom for the upper."""
    y1, y0 = np.sort(np.asarray(y1, float)), np.sort(np.asarray(y0, float))
    if not len(y1) or not len(y0):
        return {"lower": None, "upper": None, "trimmed": None}
    q1, q0 = len(y1) / n1, len(y0) / n0
    if q1 >= q0:
        k = int(round(len(y1) * (q1 - q0) / q1))
        lo, hi, who = y1[:len(y1) - k].mean() - y0.mean(), y1[k:].mean() - y0.mean(), "level"
    else:
        k = int(round(len(y0) * (q0 - q1) / q0))
        lo, hi, who = y1.mean() - y0[k:].mean(), y1.mean() - y0[:len(y0) - k].mean(), "control"
    return {"lower": float(lo), "upper": float(hi), "trimmed": who, "trimmed_n": k,
            "observed_share": [q1, q0]}
