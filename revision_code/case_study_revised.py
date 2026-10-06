"""Revised case-study analysis (Sections 13-14 of the revised manuscript).

Changes relative to case_study_estimators.py used for the submission:
  1. Lomax L-moment estimator: the submitted code used tau = l2/l1 = 1/alpha,
     which is incorrect.  For Lomax/GPD excesses tau = alpha/(2 alpha - 1),
     hence alpha = tau/(2 tau - 1) (valid for 1/2 < tau < 1), s = l1 (alpha - 1).
  2. MAD kurtosis reported as K = (H_L + H_R)/(2H) (Eq. (53)); the submitted
     tables reported (H_L + H_R)/H = 2K.
  3. The MAD shape equation H(3/4)/H(1/4) = rho(alpha) is solved for all roots;
     the root on the branch of the quartile estimate is retained and an
     out-of-range flag is recorded (rho is not monotone, Section 10).
  4. MAD-Q = quartile-consistency blend (Section 8.1, c = 1) of the MAD and
     quartile solutions; the unblended MAD solution is reported as "MAD".
  5. POT threshold sensitivity (90, 92.5, 95, 97.5 %).
  6. GEV-MLE vs Gumbel likelihood-ratio test for every block-maxima series.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.special import gamma as Gfun
import madlib as L

DATA = "./"  # folder with data/bitcoin_daily_losses.csv, the wind_*.csv files of the GitHub repository, and SPY.csv

# ------------------------------------------------------------------ helpers
def mad_roots(fam, H, lo, hi):
    target = H[1] / H[0]
    g = np.geomspace(lo, hi, 2000)
    r = L.h(fam, 0.75, g) / L.h(fam, 0.25, g) - target
    idx = np.where(np.sign(r[:-1]) * np.sign(r[1:]) < 0)[0]
    f = lambda a: L.h(fam, 0.75, a) / L.h(fam, 0.25, a) - target
    roots = [brentq(f, g[i], g[i + 1]) for i in idx]
    return roots, g[np.argmin(np.abs(r))], target, float((r + target).max())

def quant_fit(fam, q1, q3, lo, hi):
    f = lambda a: L.R(fam, 0.75, a) / L.R(fam, 0.25, a) - q3 / q1
    try:
        a = brentq(f, lo, hi)
    except ValueError:
        a = lo if abs(f(lo)) < abs(f(hi)) else hi
    return a, q1 / L.R(fam, 0.25, a)

def fit_one_shape(fam, x, lo, hi):
    x = np.asarray(x, float)
    st = L.sample_stats(x)
    q1, q3 = st["q"][0], st["q"][2]
    H = st["H"][[0, 2]]
    aQ, sQ = quant_fit(fam, q1, q3, lo, hi)
    roots, amin, rho_hat, rho_max = mad_roots(fam, H, lo, hi)
    if roots:
        aM = min(roots, key=lambda z: abs(np.log(z) - np.log(aQ)))
        in_range = True
    else:
        aM, in_range = amin, False
    sM = float(np.mean(H / L.h(fam, np.array([0.25, 0.75]), aM)))
    Qm = sM * L.R(fam, np.array([0.25, 0.75]), aM)
    delta = (abs(Qm[0] - q1) + abs(Qm[1] - q3)) / st["H"][1]
    r = delta * np.sqrt(len(x))
    w = r / (1 + r)
    aB = (1 - w) * aM + w * aQ
    sB = float(np.mean(H / L.h(fam, np.array([0.25, 0.75]), aB)))
    # L-moments
    l1, l2, _, _ = L.sample_lmoments(x)
    tau = l2 / l1
    try:
        aL = brentq(lambda a: L.lmom_ratio(fam, a) - tau, lo, hi)
    except ValueError:
        aL = np.nan
    sL = l1 / L.mu0(fam, aL) if np.isfinite(aL) else np.nan
    # MLE
    if fam == "Lomax":
        c, _, beta = stats.genpareto.fit(x, floc=0)
        aML, sML = 1 / c, beta / c
    elif fam == "Weibull":
        aML, _, sML = stats.weibull_min.fit(x, floc=0)
    return dict(MLE=(aML, sML), Quantile=(aQ, sQ), MAD=(aM, sM), Lmom=(aL, sL), MADQ=(aB, sB),
                diag=dict(roots=roots, rho_hat=rho_hat, rho_max=rho_max, in_range=in_range,
                          r=r, w=w, n=len(x)))

def pot_var_es(u, s, a, n, nu, p=0.99):
    if not (np.isfinite(a) and a > 1):
        return np.nan, np.nan
    var = u + s * ((n / nu * (1 - p)) ** (-1 / a) - 1)
    es = var / (1 - 1 / a) + (s - u) / (a - 1)
    return var, es

# Gumbel (location-scale, no shape)
def gumbel_fits(x):
    x = np.asarray(x, float)
    st = L.sample_stats(x)
    q1, M, q3 = st["q"]
    hq = L.h("Gumbel", np.array([0.25, 0.75]), 1.0)
    out = {}
    m, s = stats.gumbel_r.fit(x); out["MLE"] = (m, s)
    s = (q3 - q1) / (L.R("Gumbel", 0.75, 1.0) - L.R("Gumbel", 0.25, 1.0)); out["Quantile"] = (M + s * np.log(np.log(2)), s)
    s = float(np.mean(st["H"][[0, 2]] / hq)); out["MAD-Q"] = (M + s * np.log(np.log(2)), s)
    l1, l2, _, _ = L.sample_lmoments(x); s = l2 / np.log(2); out["Lmom"] = (l1 - L.EULER * s, s)
    return out

def z_N(m, s, xi, N=50):
    p = 1 - 1 / N
    if abs(xi) < 1e-9:
        return m - s * np.log(-np.log(p))
    return m + s / xi * ((-np.log(p)) ** (-xi) - 1)

def gev_vs_gumbel(x):
    c, loc, sc = stats.genextreme.fit(x)
    ll_gev = np.sum(stats.genextreme.logpdf(x, c, loc, sc))
    m, s = stats.gumbel_r.fit(x)
    ll_g = np.sum(stats.gumbel_r.logpdf(x, m, s))
    D = 2 * (ll_gev - ll_g)
    return dict(xi=-c, loc=loc, scale=sc, z50=z_N(loc, sc, -c), LR=D, p=float(stats.chi2.sf(max(D, 0), 1)))

def sample_shape(x):
    G, K = L.sample_GK(x)
    l1, l2, l3, l4 = L.sample_lmoments(x)
    return dict(G=G, K=K, t3=l3 / l2, t4=l4 / l2, skew=float(pd.Series(x).skew()), kurt=float(pd.Series(x).kurtosis()))

# ----------------------------------------------------------------- analysis
if __name__ == "__main__":
    out = {}
    btc = pd.read_csv(DATA + "data/bitcoin_daily_losses.csv", parse_dates=["date"])
    ell = btc["loss"].to_numpy(float)
    n = len(ell)
    out["btc_n"] = n
    out["btc_dates"] = (str(btc.date.min().date()), str(btc.date.max().date()))
    pot = {}
    for thr in (0.90, 0.925, 0.95, 0.975):
        u = float(np.quantile(ell, thr))
        exc = ell[ell > u] - u
        f = fit_one_shape("Lomax", exc, 1.01, 80.0)
        row = {}
        for meth in ("MLE", "Quantile", "MAD", "Lmom", "MADQ"):
            a, s = f[meth]
            v, e = pot_var_es(u, s, a, n, len(exc))
            row[meth] = dict(alpha=a, s=s, VaR=v, ES=e)
        d = f["diag"]
        pot[str(thr)] = dict(u=u, nu=len(exc), fits=row,
                             diag=dict(roots=d["roots"], rho_hat=d["rho_hat"], rho_max=d["rho_max"],
                                       in_range=d["in_range"], r=d["r"], w=d["w"]),
                             shape=sample_shape(exc))
    out["btc_pot"] = pot
    # block maxima
    btc["ym"] = btc["date"].dt.to_period("M")
    bm = btc.groupby("ym")["loss"].max().to_numpy(float)
    gf = gumbel_fits(bm)
    out["btc_bm"] = dict(n=len(bm), fits={k: dict(m=v[0], s=v[1], z50=z_N(v[0], v[1], 0)) for k, v in gf.items()},
                         gev=gev_vs_gumbel(bm), shape=sample_shape(bm))

    # SPY (S&P 500 ETF) -- full re-analysis
    spy = pd.read_csv(DATA + "SPY.csv", skiprows=[1], parse_dates=["Date"])
    spy = spy[spy.Date >= "2000-01-01"].reset_index(drop=True)
    lr = np.log(spy.Close.astype(float)).diff()
    spy = spy.assign(loss=-lr).dropna(subset=["loss"])
    ell_s = spy["loss"].to_numpy(float); n_s = len(ell_s)
    out["spy_n"] = n_s; out["spy_dates"] = (str(spy.Date.min().date()), str(spy.Date.max().date()))
    spot = {}
    for thr in (0.90, 0.925, 0.95, 0.975):
        u = float(np.quantile(ell_s, thr)); exc = ell_s[ell_s > u] - u
        f = fit_one_shape("Lomax", exc, 1.01, 80.0)
        row = {}
        for meth in ("MLE", "Quantile", "MAD", "Lmom", "MADQ"):
            a, s_ = f[meth]; v, e = pot_var_es(u, s_, a, n_s, len(exc))
            row[meth] = dict(alpha=a, s=s_, VaR=v, ES=e)
        d = f["diag"]
        spot[str(thr)] = dict(u=u, nu=len(exc), fits=row,
                              diag=dict(roots=d["roots"], rho_hat=d["rho_hat"], rho_max=d["rho_max"],
                                        in_range=d["in_range"], r=d["r"], w=d["w"]), shape=sample_shape(exc))
    out["spy_pot"] = spot
    spy["ym"] = spy["Date"].dt.to_period("M")
    bms = spy.groupby("ym")["loss"].max().to_numpy(float)
    gf = gumbel_fits(bms)
    out["spy_bm"] = dict(n=len(bms), fits={k: dict(m=v[0], s=v[1], z50=z_N(v[0], v[1], 0)) for k, v in gf.items()},
                         gev=gev_vs_gumbel(bms), shape=sample_shape(bms))

    # S&P 500 POT: exact correction of the L-moment row from the stored (lambda1, lambda2)
    js = json.load(open(DATA + "case_study_results.json"))
    meta = js["meta"]
    for row in js["spx_pot"]:
        if row["method"] == "L-moments":
            a_old, s_old = row["alpha"], row["sigma"]
    tau = 1 / a_old
    l1 = s_old / (a_old - 1)
    a_new = tau / (2 * tau - 1)
    s_new = l1 * (a_new - 1)
    v, e = pot_var_es(meta["spx_u95"], s_new, a_new, meta["spx_n"], meta["spx_nu"])
    out["spx_lmom_corrected"] = dict(alpha=a_new, s=s_new, VaR=v, ES=e, tau=tau, l1=l1)
    # the same correction applied to the stored Bitcoin row, as a cross-check
    for row in js["bitcoin_pot"]:
        if row["method"] == "L-moments":
            a_old, s_old = row["alpha"], row["sigma"]
    tau = 1 / a_old; l1 = s_old / (a_old - 1); a_new = tau / (2 * tau - 1)
    out["btc_lmom_crosscheck"] = dict(alpha=a_new, s=l1 * (a_new - 1))

    # wind
    wind = {}
    for key, label in (("gusty_plains", "Amarillo"), ("steady_coastal", "Honolulu")):
        hourly = pd.read_csv(DATA + f"wind_{key}_hourly_raw.csv", parse_dates=["time"])
        daily = hourly.set_index("time")["windspeed_10m"].resample("D").mean().dropna().to_numpy(float)
        daily = daily[daily > 0]
        f = fit_one_shape("Weibull", daily, 0.6, 12.0)
        wrow = {}
        for meth in ("MLE", "Quantile", "MAD", "Lmom", "MADQ"):
            k, c = f[meth]
            wrow[meth] = dict(k=k, c=c, mean=c * Gfun(1 + 1 / k))
        d = f["diag"]
        gm = pd.read_csv(DATA + f"wind_{key}_monthly_gust_max.csv")["monthly_max_gust_ms"].to_numpy(float)
        gf = gumbel_fits(gm)
        wind[label] = dict(n_daily=len(daily), mean_daily=float(daily.mean()), weibull=wrow,
                           diag=dict(roots=d["roots"], rho_hat=d["rho_hat"], in_range=d["in_range"], r=d["r"], w=d["w"]),
                           shape_daily=sample_shape(daily),
                           gust=dict(n=len(gm), fits={k2: dict(m=v[0], s=v[1], z50=z_N(v[0], v[1], 0)) for k2, v in gf.items()},
                                     gev=gev_vs_gumbel(gm), shape=sample_shape(gm)))
    out["wind"] = wind
    json.dump(out, open("case_study_revised.json", "w"), indent=1, default=float)

    # ------------------------------------------------------------ print
    print("Bitcoin n =", n, out["btc_dates"])
    for thr, P in pot.items():
        print(f"\nPOT thr={thr} u={P['u']:.4f} n_u={P['nu']} diag: roots={np.round(P['diag']['roots'],3)} "
              f"rho_hat={P['diag']['rho_hat']:.5f} rho_max={P['diag']['rho_max']:.5f} r={P['diag']['r']:.1f} w={P['diag']['w']:.3f}")
        for meth, v in P["fits"].items():
            print(f"   {meth:9s} a={v['alpha']:.3f} s={v['s']:.4f} VaR={v['VaR']:.4f} ES={v['ES']:.4f}")
        print("   sample shape", {k: round(v, 3) for k, v in P["shape"].items()})
    print("\nBTC BM", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out["btc_bm"]["fits"].items()})
    print("   GEV", {k: round(v, 4) for k, v in out["btc_bm"]["gev"].items()})
    print("   shape", {k: round(v, 3) for k, v in out["btc_bm"]["shape"].items()})
    print("\nSPY n =", n_s, out["spy_dates"])
    for thr, P in spot.items():
        print(f"POT thr={thr} u={P['u']:.4f} n_u={P['nu']} roots={np.round(P['diag']['roots'],3)} rho_hat={P['diag']['rho_hat']:.5f} r={P['diag']['r']:.1f} w={P['diag']['w']:.3f}")
        for meth, v in P["fits"].items():
            print(f"   {meth:9s} a={v['alpha']:.3f} s={v['s']:.4f} VaR={v['VaR']:.4f} ES={v['ES']:.4f}")
        print("   sample shape", {k: round(v, 3) for k, v in P["shape"].items()})
    print("SPY BM", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out["spy_bm"]["fits"].items()})
    print("   GEV", {k: round(v, 4) for k, v in out["spy_bm"]["gev"].items()})
    print("   shape", {k: round(v, 3) for k, v in out["spy_bm"]["shape"].items()})
    print("\nSPX L-moment corrected", out["spx_lmom_corrected"])
    print("BTC L-moment cross-check (from stored row)", out["btc_lmom_crosscheck"])
    for site, W in wind.items():
        print(f"\n{site}: n={W['n_daily']} mean={W['mean_daily']:.3f} diag={W['diag']}")
        for meth, v in W["weibull"].items():
            print(f"   {meth:9s} k={v['k']:.3f} c={v['c']:.3f} mean={v['mean']:.3f}")
        print("   daily shape", {k: round(v, 3) for k, v in W["shape_daily"].items()})
        print("   gust", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in W["gust"]["fits"].items()})
        print("   GEV", {k: round(v, 4) for k, v in W["gust"]["gev"].items()})
        print("   gust shape", {k: round(v, 3) for k, v in W["gust"]["shape"].items()})
