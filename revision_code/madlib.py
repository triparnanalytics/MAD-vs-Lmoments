"""Vectorised closed forms (unit scale, m = 0) used by the extended Monte Carlo
study and the revised case studies.

h(p; theta): MAD about Q(p) for unit scale (Table 7 of the revised paper);
R(p; theta): standardised quantile function; mu0(theta): mean for unit scale.
"""
import numpy as np
from scipy.special import beta as Bfun, betainc, gamma as Gfun, gammainc, gammaincc, exp1

EULER = float(np.euler_gamma)

def Binc(x, a, b):
    return betainc(a, b, x) * Bfun(a, b)

def nu(s, x):
    return gammainc(s, x) * Gfun(s)

# ---------------------------------------------------------------- quantiles
def R(fam, p, a, k=None):
    p = np.asarray(p, float)
    if fam == "ParetoI":
        return (1 - p) ** (-1 / a)
    if fam == "Lomax":
        return (1 - p) ** (-1 / a) - 1
    if fam == "LogLogistic":
        return (p / (1 - p)) ** (1 / a)
    if fam == "ParetoIV":
        return ((1 - p) ** (-1 / a) - 1) ** k
    if fam == "Frechet":
        return (-np.log(p)) ** (-1 / a)
    if fam == "Weibull":
        return (-np.log(1 - p)) ** (1 / a)
    if fam == "Gumbel":
        return -np.log(-np.log(p))
    raise ValueError(fam)

def mu0(fam, a, k=None):
    if fam == "ParetoI":
        return a / (a - 1)
    if fam == "Lomax":
        return 1 / (a - 1)
    if fam == "LogLogistic":
        return (np.pi / a) / np.sin(np.pi / a)
    if fam == "ParetoIV":
        return a * Bfun(a - k, k + 1)
    if fam == "Frechet":
        return Gfun(1 - 1 / a)
    if fam == "Weibull":
        return Gfun(1 + 1 / a)
    if fam == "Gumbel":
        return EULER

def h(fam, p, a, k=None):
    """MAD about Q(p), unit scale."""
    p = np.asarray(p, float)
    if fam in ("ParetoI", "Lomax"):
        u = (1 - p) ** (1 / a)
        return 2 * a * Binc(u, a - 1, 2) - a * Bfun(a - 1, 2) + (2 * p - 1) * ((1 - p) ** (-1 / a) - 1)
    if fam == "LogLogistic":
        A, Bb = 1 - 1 / a, 1 + 1 / a
        return 2 * Binc(1 - p, A, Bb) - Bfun(A, Bb) + (2 * p - 1) * (p / (1 - p)) ** (1 / a)
    if fam == "ParetoIV":
        u = (1 - p) ** (1 / a)
        return (2 * a * Binc(u, a - k, k + 1) - a * Bfun(a - k, k + 1)
                + (2 * p - 1) * ((1 - p) ** (-1 / a) - 1) ** k)
    if fam == "Frechet":
        c = 1 - 1 / a
        return 2 * nu(c, -np.log(p)) - Gfun(c) + (2 * p - 1) * (-np.log(p)) ** (-1 / a)
    if fam == "Weibull":
        c = 1 + 1 / a
        return Gfun(c) - 2 * nu(c, -np.log(1 - p)) + (2 * p - 1) * (-np.log(1 - p)) ** (1 / a)
    if fam == "Gumbel":
        # scipy: Ei(-x) = -E1(x) for x>0
        return EULER + np.log(-np.log(p)) + 2 * exp1(-np.log(p))
    raise ValueError(fam)

def lmom_ratio(fam, a, k=None):
    """L-CV tau = lambda2/lambda1 for m = 0 (depends on shape only)."""
    if fam == "ParetoI":
        return 1 / (2 * a - 1)
    if fam == "Lomax":
        return a / (2 * a - 1)
    if fam == "LogLogistic":
        return 1 / a
    if fam == "Frechet":
        return 2 ** (1 / a) - 1
    if fam == "Weibull":
        return 1 - 2 ** (-1 / a)
    if fam == "ParetoIV":
        f = lambda j: a * Bfun(j * a - k, k + 1)
        return (f(1) - 2 * f(2)) / f(1)
    raise ValueError(fam)

def lmom_tau3(fam, a, k):
    f = lambda j: a * Bfun(j * a - k, k + 1)
    l2 = f(1) - 2 * f(2)
    l3 = 6 * f(3) - 6 * f(2) + f(1)
    return l3 / l2

def theo_GK(fam, a=None, k=None):
    """Theoretical MAD skewness G and kurtosis K = (H_L+H_R)/(2H) and tau3, tau4
    by numerical quadrature of the quantile function (unit scale)."""
    from scipy.integrate import quad
    Rf = lambda p: float(R(fam, p, a, k))
    I = lambda p: quad(Rf, 0, p, limit=400)[0]
    mu = quad(Rf, 0, 0.5, limit=400)[0] + quad(Rf, 0.5, 1, limit=400)[0]
    H = mu - 2 * I(0.5)
    M = Rf(0.5)
    HL = 2 * (I(0.5) - 2 * I(0.25))
    HR = 2 * (mu + I(0.5) - 2 * I(0.75))
    lam = lambda w: quad(lambda p: Rf(p) * w(p), 0, 0.5, limit=400)[0] + quad(lambda p: Rf(p) * w(p), 0.5, 1, limit=400)[0]
    l2 = lam(lambda p: 2 * p - 1)
    l3 = lam(lambda p: 6 * p * p - 6 * p + 1)
    l4 = lam(lambda p: 20 * p ** 3 - 30 * p * p + 12 * p - 1)
    return dict(G=(mu - M) / H, K=(HL + HR) / (2 * H), t3=l3 / l2, t4=l4 / l2)

# ------------------------------------------------------------ sample stats
def sample_stats(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    q1, M, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    Hs = np.array([np.mean(np.abs(x - q)) for q in (q1, M, q3)])
    return dict(q=np.array([q1, M, q3]), H=Hs, mean=x.mean(), n=n, x=x)

def sample_GK(x):
    x = np.asarray(x, float)
    M = np.median(x)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    H = np.mean(np.abs(x - M))
    left, right = x[x <= M], x[x >= M]
    HL = np.mean(np.abs(left - q1))
    HR = np.mean(np.abs(right - q3))
    return (x.mean() - M) / H, (HL + HR) / (2 * H)

def sample_lmoments(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    j = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((j - 1) / (n - 1) * x) / n
    b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * (n - 2)) * x) / n
    b3 = np.sum((j - 1) * (j - 2) * (j - 3) / ((n - 1) * (n - 2) * (n - 3)) * x) / n
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l1, l2, l3, l4
