#!/usr/bin/env python3
"""Theoretical skew–kurtosis points for L-moments and MAD (companion to Figure 3).

For each heavy-tail family in main_mdpi.tex Table tab:mad_skew_kurt, plot the
theoretical (skew, kurtosis) at the planted α = 1.5 (s = 1); Gumbel is the
single point at s = 1. No shape loci — one marker per family.

Outputs
-------
    figures/theoretical_lmoment_skew_kurtosis.png
    figures/theoretical_mad_skew_kurtosis.png
    figures_llm/theoretical_lmoment_skew_kurtosis.png
    figures_llm/theoretical_mad_skew_kurtosis.png
    figures_llm/theoretical_skew_kurtosis.png   (side-by-side)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Families matching Table tab:mad_skew_kurt / Figure fig:skew_kurt
# ---------------------------------------------------------------------------
PLANTED_ALPHA = 1.5
PARETO_IV_K = 1.0
N_GRID = 4000  # uniform probability grid for numerical integrals

FAMILIES = [
    # label, display, color, marker
    ("Pareto I", "Pareto I", "#1f77b4", "o"),
    ("Lomax", "Lomax", "#ff7f0e", "s"),
    ("Log-Logistic", "Log-Logistic", "#2ca02c", "^"),
    ("Pareto IV", "Pareto IV", "#d62728", "D"),
    ("Frechet", "Fréchet", "#e377c2", "X"),
    ("Gumbel", "Gumbel", "#bcbd22", "h"),
    ("Weibull", "Weibull", "#7f7f7f", "*"),
]


def quantile(label: str, p, alpha: float, k: float = PARETO_IV_K):
    """Standardized quantile Q(p) at m = 0, s = 1."""
    p = np.asarray(p, dtype=float)
    a = float(alpha)
    eps = 1e-10
    p = np.clip(p, eps, 1.0 - eps)
    if label == "Pareto I":
        return (1.0 - p) ** (-1.0 / a)
    if label == "Lomax":
        return (1.0 - p) ** (-1.0 / a) - 1.0
    if label == "Log-Logistic":
        return (p / (1.0 - p)) ** (1.0 / a)
    if label == "Pareto IV":
        return ((1.0 - p) ** (-1.0 / a) - 1.0) ** float(k)
    if label == "Frechet":
        return (-np.log(p)) ** (-1.0 / a)
    if label == "Weibull":
        return (-np.log(1.0 - p)) ** (1.0 / a)
    if label == "Gumbel":
        return -np.log(-np.log(p))
    raise ValueError(label)



def _measures(label: str, alpha: float, p: np.ndarray | None = None):
    """Return (G, K, τ3, τ4) from one quantile grid."""
    if p is None:
        # open interval avoids Q(0)/Q(1) poles
        p = np.linspace(1.0 / (N_GRID + 1), N_GRID / (N_GRID + 1), N_GRID)
    dp = p[1] - p[0]
    Q = quantile(label, p, alpha)
    if not np.all(np.isfinite(Q)):
        return (np.nan,) * 4

    # Trapezoidal cumulative integral I(p) = ∫_0^p Q
    # Approximate on the grid via cumsum; endpoints padded with 0 and μ
    I_grid = np.concatenate([[0.0], np.cumsum(0.5 * (Q[:-1] + Q[1:]) * dp)])
    # I_grid[i] ≈ ∫_{p0}^{p[i]} ; shift so I≈∫_0^{p}
    # With p0≈0 this is fine for our purposes.
    mu = float(I_grid[-1])  # ≈ ∫_0^1 Q

    def I_at(prob: float) -> float:
        return float(np.interp(prob, p, I_grid))

    def Q_at(prob: float) -> float:
        return float(np.interp(prob, p, Q))

    M = Q_at(0.5)
    I25, I50, I75 = I_at(0.25), I_at(0.5), I_at(0.75)
    H = mu - 2.0 * I50 + (2.0 * 0.5 - 1.0) * M
    if not np.isfinite(H) or abs(H) < 1e-15:
        return (np.nan,) * 4
    G = (mu - M) / H
    H_L = I50 - 2.0 * I25
    H_R = mu + I50 - 2.0 * I75
    K = (H_L + H_R) / H

    # L-moments via shifted Legendre on the same grid
    P1 = 2.0 * p - 1.0
    P2 = 6.0 * p**2 - 6.0 * p + 1.0
    P3 = 20.0 * p**3 - 30.0 * p**2 + 12.0 * p - 1.0
    trap = getattr(np, "trapezoid", np.trapz)
    lam2 = float(trap(Q * P1, p))
    lam3 = float(trap(Q * P2, p))
    lam4 = float(trap(Q * P3, p))
    if not np.isfinite(lam2) or abs(lam2) < 1e-15:
        return G, K, np.nan, np.nan
    return float(G), float(K), float(lam3 / lam2), float(lam4 / lam2)


def _point(label: str) -> dict:
    """Theoretical (G, K, τ3, τ4) at the planted parameter only."""
    alpha = 1.0 if label == "Gumbel" else PLANTED_ALPHA
    G, K, t3, t4 = _measures(label, alpha)
    return dict(G_p=G, K_p=K, t3_p=t3, t4_p=t4)


def _style():
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,
            "figure.titlesize": 17,
        }
    )


def _normal_baselines():
    tau4 = 30.0 / np.pi * np.arctan(np.sqrt(2.0)) - 9.0
    q1 = stats.norm.ppf(0.25)
    K = -1.0 + 2.0 * np.sqrt(2.0 * np.pi) * stats.norm.pdf(q1)
    return float(tau4), float(K)


def plot_panel(ax, points: dict, kind: str, with_legend: bool = True):
    tau4_n, K_n = _normal_baselines()
    for label, display, color, marker in FAMILIES:
        c = points[label]
        if kind == "lm":
            sp, kp = c["t3_p"], c["t4_p"]
        else:
            sp, kp = c["G_p"], c["K_p"]
        ax.scatter(
            [sp],
            [kp],
            s=140,
            color=color,
            marker=marker,
            edgecolors="k",
            linewidths=0.7,
            label=display,
            zorder=3,
        )

    if kind == "lm":
        ax.set_xlabel(r"L-skewness $\tau_3=\lambda_3/\lambda_2$")
        ax.set_ylabel(r"L-kurtosis $\tau_4=\lambda_4/\lambda_2$")
        ax.set_title("L-moments (theoretical)")
        ax.axhline(tau4_n, color="k", ls=":", lw=1.8, zorder=1, label="Normal")
    else:
        ax.set_xlabel(r"MAD skewness $G=(\mu-M)/H$")
        ax.set_ylabel(r"MAD kurtosis $K=(H_L+H_R)/H$")
        ax.set_title("MAD (theoretical)")
        ax.axhline(K_n, color="k", ls=":", lw=1.8, zorder=1, label="Normal")

    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    if with_legend:
        ax.legend(frameon=False, loc="best", title="Distribution")


def main():
    _style()
    out_dirs = [Path("figures"), Path("figures_llm")]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("Computing theoretical points at planted α = 1.5…")
    points = {label: _point(label) for label, *_ in FAMILIES}

    print("\nPlanted α = 1.5 (Gumbel: s = 1) theoretical points:")
    print(f"  {'Dist':<14} {'τ3':>8} {'τ4':>8} {'G':>8} {'K':>8}")
    for label, display, *_ in FAMILIES:
        c = points[label]
        print(
            f"  {display:<14} {c['t3_p']:8.4f} {c['t4_p']:8.4f} "
            f"{c['G_p']:8.4f} {c['K_p']:8.4f}"
        )

    fig_l, ax_l = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    plot_panel(ax_l, points, "lm", with_legend=True)
    for d in out_dirs:
        p = d / "theoretical_lmoment_skew_kurtosis.png"
        fig_l.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig_l)

    fig_m, ax_m = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    plot_panel(ax_m, points, "mad", with_legend=True)
    for d in out_dirs:
        p = d / "theoretical_mad_skew_kurtosis.png"
        fig_m.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig_m)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.0), constrained_layout=True)
    plot_panel(ax1, points, "lm", with_legend=False)
    plot_panel(ax2, points, "mad", with_legend=False)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title="Distribution",
    )
    fig.suptitle(
        r"Theoretical skewness vs kurtosis "
        r"($\alpha=1.5$, $s=1$; Gumbel: $s=1$)"
    )
    for d in out_dirs:
        p = d / "theoretical_skew_kurtosis.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
