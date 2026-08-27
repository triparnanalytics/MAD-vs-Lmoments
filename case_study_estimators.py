#!/usr/bin/env python3
"""MAD / MLE / quartile / L-moment estimators for the paper case studies.

Used by ``case_studies.ipynb`` for:
  Case 1 — Bitcoin POT (Pareto II / Lomax) and monthly block maxima (GEV/Gumbel)
  Case 2 — Wind speed (Weibull) and monthly gust maxima (Gumbel/GEV)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import beta as beta_fn, betainc, exp1, gamma, gammainc

EULER = float(np.euler_gamma)
P_Q13 = np.array([0.25, 0.75], dtype=float)
MINIMIZE_KW = {"method": "bounded", "options": {"xatol": 1e-4}}


# ---------------------------------------------------------------------------
# MAD auxiliary functions A(p; shape) from main_mdpi.tex (unit scale)
# ---------------------------------------------------------------------------
def nu_lower(s: float, x) -> np.ndarray:
    return gamma(s) * gammainc(s, np.asarray(x, dtype=float))


def A_pareto_II(p, alpha):
    p = np.asarray(p, dtype=float)
    alpha = float(alpha)
    u = np.power(1.0 - p, 1.0 / alpha)
    b02 = beta_fn(alpha - 1.0, 2.0)
    return (
        2.0 * alpha * betainc(alpha - 1.0, 2.0, u) * b02
        - alpha * b02
        + (2.0 * p - 1.0) * (np.power(1.0 - p, -1.0 / alpha) - 1.0)
    )


def A_frechet(p, alpha):
    p = np.asarray(p, dtype=float)
    alpha = float(alpha)
    u = -np.log(p)
    g0 = gamma(1.0 - 1.0 / alpha)
    return (
        2.0 * nu_lower(1.0 - 1.0 / alpha, u)
        - g0
        + (2.0 * p - 1.0) * np.power(u, -1.0 / alpha)
    )


def A_weibull(p, alpha):
    p = np.asarray(p, dtype=float)
    alpha = float(alpha)
    u = -np.log(1.0 - p)
    g1 = gamma(1.0 + 1.0 / alpha)
    return (
        g1
        - 2.0 * nu_lower(1.0 + 1.0 / alpha, u)
        + (2.0 * p - 1.0) * np.power(u, 1.0 / alpha)
    )


def Q_gumbel(p):
    p = np.asarray(p, dtype=float)
    return -np.log(-np.log(p))


def A_gumbel_unit(p):
    p = np.asarray(p, dtype=float)
    return EULER - Q_gumbel(p) + 2.0 * exp1(-np.log(p))


def sample_H(x, p=P_Q13):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    O = np.quantile(x, p)
    H = np.abs(x[:, None] - O).mean(axis=0)
    return O, H


def _grid_refine(objective, bounds, n_grid=21):
    lo, hi = bounds
    grid = np.linspace(lo, hi, n_grid)
    vals = np.array([objective(float(g)) for g in grid])
    i = int(np.argmin(vals))
    lo2 = float(grid[max(0, i - 1)])
    hi2 = float(grid[min(n_grid - 1, i + 1)])
    return (lo2, hi2) if hi2 > lo2 else bounds


# ---------------------------------------------------------------------------
# Generic shape+scale MAD matches
# ---------------------------------------------------------------------------
def mad_q_shape_scale(
    x,
    A_fn,
    shape_bounds=(1.05, 12.0),
    free_scale=True,
    shape_fallback=None,
):
    """MAD-Q: match H at Q1 and Q3. Returns (shape, scale).

    Profiles the scale-free H-ratio when it lies in the attainable range of
    A(3/4)/A(1/4); otherwise uses ``shape_fallback`` (e.g. quartile match)
    and sets scale from the MAD equations.
    """
    _, H = sample_H(x)
    lo, hi = map(float, shape_bounds)
    if not np.all(np.isfinite(H)) or np.any(H <= 0):
        return np.nan, np.nan

    target = float(H[1] / H[0])
    grid = np.linspace(lo, hi, 80)
    ratios = []
    for g in grid:
        try:
            A = np.asarray(A_fn(P_Q13, float(g)), dtype=float)
            ratios.append(float(A[1] / A[0]) if A[0] > 0 and np.all(np.isfinite(A)) else np.nan)
        except Exception:
            ratios.append(np.nan)
    ratios = np.asarray(ratios, dtype=float)
    finite = ratios[np.isfinite(ratios)]
    in_range = len(finite) > 0 and finite.min() - 1e-3 <= target <= finite.max() + 1e-3

    if in_range:

        def ratio_err(sh):
            try:
                A = np.asarray(A_fn(P_Q13, float(sh)), dtype=float)
            except Exception:
                return 1e6
            if not np.all(np.isfinite(A)) or A[0] <= 0:
                return 1e6
            return abs(float(A[1] / A[0]) - target)

        sh0 = float(grid[int(np.nanargmin(np.abs(ratios - target)))])
        try:
            res = minimize_scalar(
                ratio_err, bounds=_grid_refine(ratio_err, (lo, hi)), **MINIMIZE_KW
            )
            if np.isfinite(res.fun):
                sh0 = float(res.x)
        except Exception:
            pass
    elif shape_fallback is not None and np.isfinite(shape_fallback):
        sh0 = float(np.clip(shape_fallback, lo, hi))
    else:
        sh0 = float(grid[int(np.nanargmin(np.abs(ratios - target)))])

    A0 = np.asarray(A_fn(P_Q13, sh0), dtype=float)
    sc0 = float(np.mean(H / np.maximum(A0, 1e-12)))
    if not free_scale:
        return sh0, 1.0

    def resid(th):
        sh, sc = float(th[0]), float(th[1])
        if sc <= 0:
            return np.array([1e6, 1e6])
        try:
            A = np.asarray(A_fn(P_Q13, sh), dtype=float)
        except Exception:
            return np.array([1e6, 1e6])
        if not np.all(np.isfinite(A)):
            return np.array([1e6, 1e6])
        return sc * A - H

    # If H-ratio is out of range, freeze shape at fallback and fit scale only.
    if not in_range:
        return sh0, sc0

    sol = least_squares(
        resid, [sh0, max(sc0, 1e-8)], bounds=([lo, 1e-10], [hi, np.inf])
    )
    if sol.success and np.all(np.isfinite(sol.x)):
        return float(sol.x[0]), float(sol.x[1])
    return sh0, sc0


def mad_l1_shape_scale(x, A_fn, shape_bounds=(1.05, 12.0), shape_fallback=None):
    sh, sc = mad_q_shape_scale(
        x, A_fn, shape_bounds=shape_bounds, free_scale=True, shape_fallback=shape_fallback
    )
    if not np.isfinite(sh):
        return np.nan, np.nan
    _, H = sample_H(x)
    A = np.asarray(A_fn(P_Q13, float(sh)), dtype=float)
    sc = float(np.median(H / np.maximum(A, 1e-12)))
    return float(sh), sc


def mad_l2_shape_scale(x, A_fn, shape_bounds=(1.05, 12.0), shape_fallback=None):
    return mad_q_shape_scale(
        x, A_fn, shape_bounds=shape_bounds, free_scale=True, shape_fallback=shape_fallback
    )


def quartile_match_lomax(x):
    """Match sample Q1, Q3 to Lomax Q(p)=s*((1-p)^(-1/a)-1)."""
    q1, q3 = np.quantile(x, [0.25, 0.75])
    if q1 <= 0 or q3 <= q1:
        return np.nan, np.nan

    def f(alpha):
        a = float(alpha)
        r = ((1 - 0.75) ** (-1 / a) - 1) / ((1 - 0.25) ** (-1 / a) - 1)
        return r - q3 / q1

    try:
        from scipy.optimize import brentq

        alpha = float(brentq(f, 1.05, 80.0))
    except ValueError:
        # nearest grid fallback when ratio is extreme (near-exponential tail)
        grid = np.linspace(1.05, 80.0, 200)
        alpha = float(grid[int(np.argmin(np.abs([f(g) for g in grid])))])
    s = q1 / ((1.0 - 0.25) ** (-1.0 / alpha) - 1.0)
    return alpha, float(s)


def quartile_match_weibull(x):
    """Match Q1,Q3 to Weibull Q(p)=c*(-log(1-p))^(1/k)."""
    q1, q3 = np.quantile(x, [0.25, 0.75])
    if q1 <= 0 or q3 <= q1:
        return np.nan, np.nan

    def f(k):
        k = float(k)
        return ((-np.log(0.25)) ** (1 / k)) / ((-np.log(0.75)) ** (1 / k)) - q3 / q1

    try:
        from scipy.optimize import brentq

        # Note: Q3/Q1 = [(-log(1-0.75))/(-log(1-0.25))]^(1/k)
        def g(k):
            k = float(k)
            num = (-np.log(1 - 0.75)) ** (1 / k)
            den = (-np.log(1 - 0.25)) ** (1 / k)
            return num / den - q3 / q1

        k = float(brentq(g, 0.4, 8.0))
    except ValueError:
        return np.nan, np.nan
    c = q1 / ((-np.log(1 - 0.25)) ** (1 / k))
    return k, float(c)


def lmoments_sample(x, nmom=3):
    """Unbiased sample L-moments (Hosking)."""
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    n = len(x)
    if n < nmom + 2:
        return np.full(nmom, np.nan)
    b = np.zeros(nmom)
    for r in range(nmom):
        # PWM b_r = n^{-1} sum_{j=r+1}^n C(j-1,r)/C(n-1,r) x_j
        coeffs = np.ones(n)
        if r > 0:
            j = np.arange(1, n + 1)
            # C(j-1,r)/C(n-1,r) = 0 for j<=r
            coeffs = np.zeros(n)
            for jj in range(r + 1, n + 1):
                coeffs[jj - 1] = np.prod([(jj - 1 - k) / (n - 1 - k) for k in range(r)])
        b[r] = np.mean(coeffs * x)
    l1 = b[0]
    l2 = 2 * b[1] - b[0]
    l3 = 6 * b[2] - 6 * b[1] + b[0] if nmom >= 3 else np.nan
    return np.array([l1, l2, l3])


def lomax_from_lmoments(x):
    """Pareto II / Lomax via L-CV τ = λ2/λ1."""
    l1, l2, _ = lmoments_sample(x, 3)
    if not np.isfinite(l1) or l1 <= 0 or not np.isfinite(l2) or l2 <= 0:
        return np.nan, np.nan
    tau = l2 / l1
    # For Lomax: λ1 = s/(α-1), λ2 = s/((α-1)(α)), τ=λ2/λ1 = 1/α ⇒ α = 1/τ
    # Actually λ2 = s α / ((α-1)^2 (α+?)) — use Hosking Pareto II formulas.
    # Standard: for GPA/GPD with ξ=1/α, β=s/α... Use numerical match on τ2.
    if tau <= 0 or tau >= 1:
        return np.nan, np.nan
    # Lomax: τ = λ2/λ1 = 1/(2(α-1)+2)? Closed form:
    # λ1 = s/(α-1), λ2 = s/(α(α-1)), so τ=λ2/λ1 = 1/α ⇒ α=1/τ
    alpha = 1.0 / tau
    if alpha <= 1.05:
        return np.nan, np.nan
    s = l1 * (alpha - 1.0)
    return float(alpha), float(s)


def weibull_from_lmoments(x):
    l1, l2, _ = lmoments_sample(x, 3)
    if not np.isfinite(l1) or not np.isfinite(l2) or l2 <= 0:
        return np.nan, np.nan
    tau = l2 / l1

    def f(k):
        k = float(k)
        # λ1 = c Γ(1+1/k), λ2 = c(1-2^{-1/k})Γ(1+1/k)
        return (1.0 - 2.0 ** (-1.0 / k)) - tau

    try:
        from scipy.optimize import brentq

        k = float(brentq(f, 0.4, 8.0))
    except ValueError:
        return np.nan, np.nan
    c = l1 / gamma(1.0 + 1.0 / k)
    return k, float(c)


def gumbel_from_lmoments(x):
    l1, l2, _ = lmoments_sample(x, 3)
    if not np.isfinite(l2) or l2 <= 0:
        return np.nan, np.nan
    s = l2 / np.log(2.0)
    m = l1 - EULER * s
    return m, float(s)


def gumbel_mad_q(x):
    """MAD-Q for Gumbel: estimate scale from H at Q1,Q3; location from median."""
    O, H = sample_H(x, np.array([0.25, 0.50, 0.75]))
    h = A_gumbel_unit(np.array([0.25, 0.75]))
    s = float(np.mean(H[[0, 2]] / h))
    # Q(0.5) = m - s log(log 2)
    m = float(O[1] + s * np.log(np.log(2.0)))
    return m, s


# ---------------------------------------------------------------------------
# MLE wrappers
# ---------------------------------------------------------------------------
def mle_lomax(x):
    """Fit Pareto II / Lomax via genpareto (ξ=1/α, β=s/α)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    # genpareto: c=ξ, scale=β, loc=0
    c, loc, scale = stats.genpareto.fit(x, floc=0)
    if c <= 1e-8:
        return np.nan, np.nan
    alpha = 1.0 / c
    s = scale / c  # Lomax scale
    return float(alpha), float(s)


def mle_weibull(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    # weibull_min: c=k shape, scale=c
    k, loc, c = stats.weibull_min.fit(x, floc=0)
    return float(k), float(c)


def mle_gumbel(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    # genextreme with c=0 is Gumbel; fit gumbel_r
    m, s = stats.gumbel_r.fit(x)
    return float(m), float(s)


def mle_gev(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    # scipy genextreme: c = -ξ
    c, loc, scale = stats.genextreme.fit(x)
    xi = -c
    return float(xi), float(loc), float(scale)


# ---------------------------------------------------------------------------
# Risk measures (POT / GPD-Lomax)
# ---------------------------------------------------------------------------
def pot_var_es(u, sigma, alpha, n, n_u, p=0.99):
    """VaR_p and ES_p for losses with Lomax/GPD exceedances.

    Lomax scale σ, shape α ↔ GPD ξ=1/α, β=σ/α.
    """
    if not np.isfinite(alpha) or not np.isfinite(sigma):
        return np.nan, np.nan
    if alpha <= 1.0 or sigma <= 0 or n_u <= 0:
        return np.nan, np.nan
    xi = 1.0 / alpha
    beta = sigma / alpha
    var = u + (beta / xi) * ((n / n_u * (1.0 - p)) ** (-xi) - 1.0)
    # unconditional ES for GPD excesses (McNeil et al.)
    es = (var + beta - xi * u) / (1.0 - xi)
    return float(var), float(es)


def gev_return_level(mu, sigma, xi, N):
    """N-block return level z_N with G(z_N)=1-1/N."""
    p = 1.0 - 1.0 / N
    if abs(xi) < 1e-8:
        return float(mu - sigma * np.log(-np.log(p)))
    return float(mu + sigma / xi * ((-np.log(p)) ** (-xi) - 1.0))


# ---------------------------------------------------------------------------
# Bundle fits for a sample
# ---------------------------------------------------------------------------
@dataclass
class FitRow:
    method: str
    shape: float
    scale: float
    loc: float = np.nan
    runtime_ms: float = np.nan
    var99: float = np.nan
    es99: float = np.nan
    extra: float = np.nan  # mean speed / z50 etc.


def fit_lomax_all(excesses, u, n_total, p_var=0.99) -> list[FitRow]:
    x = np.asarray(excesses, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    n_u = len(x)
    rows = []
    specs = [
        ("MLE", lambda: mle_lomax(x)),
        ("Quantile", lambda: quartile_match_lomax(x)),
        ("L1", lambda: mad_l1_shape_scale(x, A_pareto_II, (1.05, 12.0))),
        ("L2", lambda: mad_l2_shape_scale(x, A_pareto_II, (1.05, 12.0))),
        ("L-moments", lambda: lomax_from_lmoments(x)),
        ("MAD-Q", lambda: mad_q_shape_scale(x, A_pareto_II, (1.05, 12.0))),
    ]
    for name, fn in specs:
        t0 = time.perf_counter()
        alpha, sigma = fn()
        dt = 1000.0 * (time.perf_counter() - t0)
        var, es = pot_var_es(u, sigma, alpha, n_total, n_u, p=p_var)
        rows.append(
            FitRow(name, alpha, sigma, runtime_ms=dt, var99=var, es99=es)
        )
    return rows


def fit_weibull_all(speeds) -> list[FitRow]:
    x = np.asarray(speeds, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    q_k, _ = quartile_match_weibull(x)
    rows = []
    specs = [
        ("MLE", lambda: mle_weibull(x)),
        ("Quantile", lambda: quartile_match_weibull(x)),
        ("L1", lambda: mad_l1_shape_scale(x, A_weibull, (0.5, 8.0), shape_fallback=q_k)),
        ("L2", lambda: mad_l2_shape_scale(x, A_weibull, (0.5, 8.0), shape_fallback=q_k)),
        ("L-moments", lambda: weibull_from_lmoments(x)),
        ("MAD-Q", lambda: mad_q_shape_scale(x, A_weibull, (0.5, 8.0), shape_fallback=q_k)),
    ]
    for name, fn in specs:
        t0 = time.perf_counter()
        k, c = fn()
        dt = 1000.0 * (time.perf_counter() - t0)
        mean_spd = c * gamma(1.0 + 1.0 / k) if np.isfinite(k) and np.isfinite(c) else np.nan
        rows.append(FitRow(name, k, c, runtime_ms=dt, extra=mean_spd))
    return rows


def fit_gumbel_all(block_max) -> list[FitRow]:
    x = np.asarray(block_max, dtype=float)
    x = x[np.isfinite(x)]
    rows = []

    def q_gumbel():
        q1, q3 = np.quantile(x, [0.25, 0.75])
        denom = Q_gumbel(0.75) - Q_gumbel(0.25)
        s = (q3 - q1) / denom
        m = np.median(x) + s * np.log(np.log(2.0))
        return m, s

    def l1_g():
        # estimate scale by L1 MAD at Q1/Q3; location from median
        O, H = sample_H(x)
        h = A_gumbel_unit(P_Q13)
        s = float(np.median(H / h))
        m = float(np.median(x) + s * np.log(np.log(2.0)))
        return m, s

    def l2_g():
        O, H = sample_H(x)
        h = A_gumbel_unit(P_Q13)
        s = float(np.mean(H / h))
        m = float(np.median(x) + s * np.log(np.log(2.0)))
        return m, s

    specs = [
        ("MLE", lambda: mle_gumbel(x)),
        ("Quantile", q_gumbel),
        ("L1", l1_g),
        ("L2", l2_g),
        ("L-moments", lambda: gumbel_from_lmoments(x)),
        ("MAD-Q", lambda: gumbel_mad_q(x)),
    ]
    for name, fn in specs:
        t0 = time.perf_counter()
        m, s = fn()
        dt = 1000.0 * (time.perf_counter() - t0)
        z50 = gev_return_level(m, s, 0.0, 50)
        rows.append(FitRow(name, np.nan, s, loc=m, runtime_ms=dt, extra=z50))
    return rows


def mad_skew_kurt_sample(x):
    """Empirical MAD skew G=(μ-M)/H and kurtosis K=(HL+HR)/H."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    M = np.median(x)
    mu = np.mean(x)
    H = np.mean(np.abs(x - M))
    q1, q3 = np.quantile(x, [0.25, 0.75])
    left = x[x <= M]
    right = x[x >= M]
    HL = np.mean(np.abs(left - q1)) if len(left) else np.nan
    HR = np.mean(np.abs(right - q3)) if len(right) else np.nan
    G = (mu - M) / H if H > 0 else np.nan
    K = (HL + HR) / H if H > 0 else np.nan
    return float(G), float(K)


def lmoment_skew_kurt_sample(x):
    l1, l2, l3 = lmoments_sample(x, 3)
    # need λ4 for τ4 — extend
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8 or not np.isfinite(l2) or l2 == 0:
        return np.nan, np.nan
    b = np.zeros(4)
    for r in range(4):
        coeffs = np.zeros(n)
        for jj in range(r + 1, n + 1):
            coeffs[jj - 1] = np.prod([(jj - 1 - k) / (n - 1 - k) for k in range(r)]) if r else 1.0
        if r == 0:
            coeffs = np.ones(n)
        b[r] = np.mean(coeffs * x)
    l1 = b[0]
    l2 = 2 * b[1] - b[0]
    l3 = 6 * b[2] - 6 * b[1] + b[0]
    l4 = 20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]
    t3 = l3 / l2
    t4 = l4 / l2
    return float(t3), float(t4)
