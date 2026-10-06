"""Revised Table 16 (clean-sample MAD and L-moment skewness/kurtosis) and
revised Figures 2, 4 and 5.

* One Monte Carlo design for all four measures: n = 1000, R = 200.
* K = (H_L + H_R)/(2H) as in Eq. (53).
* Pareto IV uses a genuine two-shape configuration (alpha, k) = (3, 1.5)
  (tail index alpha/k = 2, finite mean) instead of k = 1, which coincides with Lomax.
* Pareto I and Lomax differ only by a location shift, so every location-free
  shape functional coincides; they are drawn as one marker.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import madlib as L

rng = np.random.default_rng(20260928)
FAMS = [("Pareto I / Lomax", "Lomax", 1.5, None),
        ("Log-logistic", "LogLogistic", 1.5, None),
        ("Pareto IV ($k=1.5$)", "ParetoIV", 3.0, 1.5),
        ("Fréchet", "Frechet", 1.5, None),
        ("Gumbel", "Gumbel", None, None),
        ("Weibull", "Weibull", 1.5, None)]
N, R = 1000, 200
rows = []
clouds = {}
for label, fam, a, k in FAMS:
    th = L.theo_GK(fam, a if a else 1.0, k)
    vals = []
    for _ in range(R):
        x = L.R(fam, rng.random(N), a if a else 1.0, k)
        G, K = L.sample_GK(x)
        l1, l2, l3, l4 = L.sample_lmoments(x)
        vals.append((G, K, l3 / l2, l4 / l2))
    v = np.array(vals)
    clouds[label] = v
    med = np.median(v, axis=0)
    rows.append(dict(label=label, G_th=th["G"], K_th=th["K"], t3_th=th["t3"], t4_th=th["t4"],
                     G=med[0], K=med[1], t3=med[2], t4=med[3]))
df = pd.DataFrame(rows)
df.to_csv("table16_revised.csv", index=False)
print(df.round(3).to_string())

# ------------------------------------------------------------ Figures 4 and 5
style = {"Pareto I / Lomax": ("#0072B2", "o"), "Log-logistic": ("#009E73", "^"),
         "Pareto IV ($k=1.5$)": ("#D55E00", "D"), "Fréchet": ("#CC79A7", "X"),
         "Gumbel": ("#8C6D1F", "h"), "Weibull": ("#555555", "*")}
KN, T4N = 0.5931, 0.1226
plt.rcParams.update({"font.family": "serif", "font.size": 11})

def panel(ax, xs, ys, xlab, ylab, base, title, cloud=None):
    for (lab, (col, mk)), x, y in zip(style.items(), xs, ys):
        if cloud is not None:
            ax.scatter(cloud[lab][:, 0], cloud[lab][:, 1], s=6, color=col, alpha=0.18, lw=0)
        ax.scatter([x], [y], s=110, color=col, marker=mk, edgecolor="k", lw=0.7, label=lab, zorder=3)
    ax.axhline(base, ls=":", color="k", lw=1.2, label="Normal baseline")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(alpha=0.3)

def fig_pair(fname, emp):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
    if emp:
        cl = {k: v[:, [2, 3]] for k, v in clouds.items()}
        panel(axs[0], df.t3, df.t4, r"L-skewness $\tau_3$", r"L-kurtosis $\tau_4$", T4N, "L-moments (Monte Carlo medians)", cl)
        cl = {k: v[:, [0, 1]] for k, v in clouds.items()}
        panel(axs[1], df.G, df.K, r"MAD skewness $G=(\mu-M)/H$", r"MAD kurtosis $K=(H_L+H_R)/(2H)$", KN, "MAD (Monte Carlo medians)", cl)
        axs[1].scatter(df.G_th, df.K_th, s=220, facecolor="none", edgecolor="k", lw=0.8, label="theoretical value")
    else:
        panel(axs[0], df.t3_th, df.t4_th, r"L-skewness $\tau_3$", r"L-kurtosis $\tau_4$", T4N, "L-moments (theoretical)")
        panel(axs[1], df.G_th, df.K_th, r"MAD skewness $G=(\mu-M)/H$", r"MAD kurtosis $K=(H_L+H_R)/(2H)$", KN, "MAD (theoretical)")
    axs[0].set_xlim(0, 0.8); axs[0].set_ylim(0, 0.65)
    axs[1].set_xlim(0, 1.0); axs[1].set_ylim(0.5, 0.9)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.savefig(fname, dpi=250)

fig_pair("figures_llm/skewness_vs_kurtosis_rev.png", True)
fig_pair("figures_llm/theoretical_skew_kurtosis_rev.png", False)

# ------------------------------------------------------------ Figure 2
csv = pd.read_csv("paper_tables_all_n.csv")
tab1000 = {"Pareto I": [2.12, 5.41, 2.74, 4.18, 4.19, 2.70], "Lomax": [2.21, 8.00, 2.67, 4.00, 4.06, 2.65],
           "Log-Logistic": [1.82, 3.89, 4.05, 3.57, 3.72, 3.07], "Pareto IV": [2.23, 7.84, 2.72, 3.80, 3.90, 2.71],
           "Frechet": [1.64, 4.57, 3.47, 3.70, 3.83, 3.00], "Gumbel": [1.67, 2.14, 3.18, 2.19, 2.10, 2.06],
           "Weibull": [1.50, 1.88, 2.98, 2.38, 2.31, 2.35]}
meths = ["MLE", "L-moment", "Quantile", "L1", "L2", "MAD-Q13"]
labels = {"MLE": "MLE", "L-moment": "L-moments", "Quantile": "Quantile", "L1": "$L_1$", "L2": "$L_2$", "MAD-Q13": "MAD-Q"}
cols = {"MLE": "#0072B2", "L-moment": "#C9A227", "Quantile": "#009E73", "L1": "#56B4E9", "L2": "#CC79A7", "MAD-Q13": "#D55E00"}
fams = ["Pareto I", "Lomax", "Log-Logistic", "Frechet", "Gumbel", "Weibull"]
fig, axs = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True)
for ax, f in zip(axs.flat, fams):
    for j, m in enumerate(meths):
        sub = csv[(csv.distribution == f) & (csv.method == m)].sort_values("n")
        ns = list(sub.n) + [1000]; ys = list(sub.clean_mae_pct) + [tab1000[f][j]]
        o = np.argsort(ns); ns = np.array(ns)[o]; ys = np.array(ys)[o]
        lw, ms = (2.6, 7) if m == "MAD-Q13" else (1.2, 4)
        ax.plot(ns, ys, marker="o", ms=ms, lw=lw, color=cols[m], label=labels[m], zorder=3 if m == "MAD-Q13" else 2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks([250, 500, 1000, 1500]); ax.set_xticklabels(["250", "500", "1000", "1500"])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.set_title({"Frechet": "Fréchet", "Lomax": "Lomax (= Pareto IV, $k=1$)"}.get(f, f), fontsize=11)
    ax.grid(alpha=0.3, which="both")
for ax in axs[:, 0]:
    ax.set_ylabel("MAE% of shape (log scale)")
for ax in axs[1]:
    ax.set_xlabel("sample size $n$ (log scale)")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("figures_llm/fig_method_error_by_n_rev.png", dpi=250)
print("figures written")
