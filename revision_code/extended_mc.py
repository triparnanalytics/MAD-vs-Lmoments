"""Extended Monte Carlo study (revision, Section 11.x / Appendix D).

Design: location m = 0 known; scale s AND shape(s) unknown (the realistic
setting of the case studies).  Shape alpha in {1.2, 1.5, 2.5, 4.0},
n in {100, 250, 500, 1000, 2000}, R = 500 replications per cell.
Pareto IV: (alpha, k) in {(3, 1.5), (2, 0.5)}, all three of (s, alpha, k) unknown.

Estimators
  MLE       scipy.stats .fit with floc=0 (closed form for Pareto I)
  Lmom      L-CV tau2 = l2/l1 matched to its closed form (plus tau3 for Pareto IV)
  Quantile  exact match of sample Q1, Q3 (plus the median for Pareto IV)
  MAD       exact match of the scale-free MAD ratio H(3/4)/H(1/4)
            (plus H(1/2)/H(1/4) for Pareto IV); when the ratio equation has two
            roots, the root on the branch containing the quartile estimate is used
  MAD-Q     quartile-consistency blend of MAD and Quantile (Section 8.1, c = 1)
Error metric: median absolute relative error (MAE%) of the shape estimate.
"""
import sys, time, json
import numpy as np
from multiprocessing import Pool
from scipy import stats
from scipy.optimize import brentq, least_squares
import madlib as L

NODES = np.array([0.25, 0.75])
BOUNDS = {"ParetoI": (1.01, 60.0), "Lomax": (1.01, 60.0), "LogLogistic": (1.01, 60.0),
          "Frechet": (1.01, 60.0), "Weibull": (0.2, 60.0)}
GRID = {f: np.geomspace(*BOUNDS[f], 600) for f in BOUNDS}
RHO = {f: L.h(f, 0.75, GRID[f]) / L.h(f, 0.25, GRID[f]) for f in BOUNDS}

def draw(rng, fam, n, a, k=None):
    return L.R(fam, rng.random(n), a, k)

# ------------------------------------------------------------- one-shape fits
def fit_mle(fam, x):
    try:
        if fam == "ParetoI":
            s = x.min(); a = len(x) / np.sum(np.log(x / s)); return a, s
        if fam == "Lomax":
            a, _, s = stats.lomax.fit(x, floc=0); return a, s
        if fam == "LogLogistic":
            a, _, s = stats.fisk.fit(x, floc=0); return a, s
        if fam == "Frechet":
            a, _, s = stats.invweibull.fit(x, floc=0); return a, s
        if fam == "Weibull":
            a, _, s = stats.weibull_min.fit(x, floc=0); return a, s
    except Exception:
        return np.nan, np.nan

def fit_lmom(fam, x):
    l1, l2, _, _ = L.sample_lmoments(x)
    tau = l2 / l1
    lo, hi = BOUNDS[fam]
    f = lambda a: L.lmom_ratio(fam, a) - tau
    try:
        a = brentq(f, lo, hi)
    except ValueError:
        a = lo if abs(f(lo)) < abs(f(hi)) else hi
    return a, l1 / L.mu0(fam, a)

def qratio(fam, a):
    return L.R(fam, 0.75, a) / L.R(fam, 0.25, a)

def fit_quant(fam, q1, q3):
    lo, hi = BOUNDS[fam]
    target = q3 / q1
    f = lambda a: qratio(fam, a) - target
    try:
        a = brentq(f, lo, hi)
    except ValueError:
        a = lo if abs(f(lo)) < abs(f(hi)) else hi
    return a, q1 / L.R(fam, 0.25, a)

def fit_mad(fam, H, a_ref):
    """Solve h(3/4;a)/h(1/4;a) = H3/H1. Returns (a, s, in_range)."""
    target = H[1] / H[0]
    g, r = GRID[fam], RHO[fam] - target
    roots = []
    idx = np.where(np.sign(r[:-1]) * np.sign(r[1:]) < 0)[0]
    f = lambda a: L.h(fam, 0.75, a) / L.h(fam, 0.25, a) - target
    for i in idx:
        try:
            roots.append(brentq(f, g[i], g[i + 1]))
        except ValueError:
            pass
    if roots:
        a = min(roots, key=lambda z: abs(np.log(z) - np.log(a_ref)))
        ok = True
    else:
        a = g[np.argmin(np.abs(r))]
        ok = False
    s = np.mean(H / L.h(fam, NODES, a))
    return a, s, ok

def blend(fam, st, aM, sM, aQ, sQ, c=1.0):
    q1, q3 = st["q"][0], st["q"][2]
    Qm = sM * L.R(fam, NODES, aM)
    delta = (abs(Qm[0] - q1) + abs(Qm[1] - q3)) / st["H"][1]
    r = delta / (c / np.sqrt(st["n"]))
    w = r / (1 + r)
    return (1 - w) * aM + w * aQ, w

def one_cell(args):
    fam, a0, n, R, seed = args
    rng = np.random.default_rng(seed)
    out = {m: [] for m in ("MLE", "Lmom", "Quantile", "MAD", "MAD-Q")}
    oor = 0
    wsum = 0.0
    for _ in range(R):
        x = draw(rng, fam, n, a0)
        st = L.sample_stats(x)
        H = st["H"][[0, 2]]
        aQ, sQ = fit_quant(fam, st["q"][0], st["q"][2])
        aM, sM, ok = fit_mad(fam, H, aQ)
        oor += (not ok)
        aB, w = blend(fam, st, aM, sM, aQ, sQ)
        wsum += w
        out["MLE"].append(fit_mle(fam, x)[0])
        out["Lmom"].append(fit_lmom(fam, x)[0])
        out["Quantile"].append(aQ)
        out["MAD"].append(aM)
        out["MAD-Q"].append(aB)
    res = {m: 100 * float(np.nanmedian(np.abs(np.array(v) - a0) / a0)) for m, v in out.items()}
    res["oor_pct"] = 100 * oor / R
    res["mean_w"] = wsum / R
    return fam, a0, n, res

# ------------------------------------------------------------- Pareto IV
def piv_fit_ratios(model_fn, target, x0):
    """Solve two scale-free ratio equations for (alpha, k)."""
    def resid(t):
        a, k = np.exp(t)
        if a <= k + 0.02:
            return np.array([1e3, 1e3])
        v = model_fn(a, k)
        return v - target
    try:
        sol = least_squares(resid, np.log(x0), method="lm", xtol=1e-12, ftol=1e-12, max_nfev=400)
        a, k = np.exp(sol.x)
        return a, k
    except Exception:
        return np.nan, np.nan

P3 = np.array([0.25, 0.5, 0.75])

def one_cell_piv(args):
    a0, k0, n, R, seed = args
    rng = np.random.default_rng(seed)
    out = {m: ([], []) for m in ("MLE", "Lmom", "Quantile", "MAD", "MAD-Q")}
    for _ in range(R):
        x = draw(rng, "ParetoIV", n, a0, k0)
        st = L.sample_stats(x)
        q, H = st["q"], st["H"]
        qt = np.array([q[1] / q[0], q[2] / q[0]])
        aQ, kQ = piv_fit_ratios(lambda a, k: L.R("ParetoIV", P3[1:], a, k) / L.R("ParetoIV", 0.25, a, k), qt, (a0, k0))
        ht = np.array([H[1] / H[0], H[2] / H[0]])
        aM, kM = piv_fit_ratios(lambda a, k: L.h("ParetoIV", P3[1:], a, k) / L.h("ParetoIV", 0.25, a, k), ht,
                                (aQ, kQ) if np.isfinite(aQ) else (a0, k0))
        # blend (componentwise), residual over Q1, M, Q3
        if np.isfinite(aM) and np.isfinite(aQ):
            sM = np.mean(H / L.h("ParetoIV", P3, aM, kM))
            Qm = sM * L.R("ParetoIV", P3, aM, kM)
            delta = np.sum(np.abs(Qm - q)) / H[1]
            r = delta * np.sqrt(n)
            w = r / (1 + r)
            aB, kB = (1 - w) * aM + w * aQ, (1 - w) * kM + w * kQ
        else:
            aB, kB = aQ, kQ
        l1, l2, l3, _ = L.sample_lmoments(x)
        aL, kL = piv_fit_ratios(lambda a, k: np.array([L.lmom_ratio("ParetoIV", a, k), L.lmom_tau3("ParetoIV", a, k)]),
                                np.array([l2 / l1, l3 / l2]), (a0, k0))
        try:
            c, d, _, s = stats.burr12.fit(x, 1 / k0, a0, floc=0)
            aML, kML = d, 1 / c
        except Exception:
            aML, kML = np.nan, np.nan
        for m, (a, k) in zip(("MLE", "Lmom", "Quantile", "MAD", "MAD-Q"),
                             ((aML, kML), (aL, kL), (aQ, kQ), (aM, kM), (aB, kB))):
            out[m][0].append(a); out[m][1].append(k)
    res = {}
    for m, (av, kv) in out.items():
        res[m] = (100 * float(np.nanmedian(np.abs(np.array(av) - a0) / a0)),
                  100 * float(np.nanmedian(np.abs(np.array(kv) - k0) / k0)))
    return ("ParetoIV", (a0, k0), n, res)

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    fams = ["ParetoI", "Lomax", "LogLogistic", "Frechet", "Weibull"]
    alphas = [1.2, 1.5, 2.5, 4.0]
    ns = [100, 250, 500, 1000, 2000]
    tasks = [(f, a, n, R, 1000 * i + j) for i, f in enumerate(fams) for j, (a, n) in
             enumerate([(a, n) for a in alphas for n in ns])]
    ptasks = [(a, k, n, R, 99000 + 10 * j + i) for j, (a, k) in enumerate([(3.0, 1.5), (2.0, 0.5)])
              for i, n in enumerate(ns)]
    t0 = time.time()
    with Pool(2) as pool:
        res = pool.map(one_cell, tasks, chunksize=1)
        pres = pool.map(one_cell_piv, ptasks, chunksize=1)
    json.dump({"one_shape": res, "pareto_iv": pres, "R": R}, open(f"extended_mc_R{R}.json", "w"), indent=1)
    print("done in %.1f min" % ((time.time() - t0) / 60))
