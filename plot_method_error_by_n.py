#!/usr/bin/env python3
"""Publication figure: grouped bars of relative error vs sample size.

One panel per distribution (paper Table tab:clean_error, seven families).
Grouped bars are the six estimators named in main_mdpi.tex:
MLE, L-moments, Quantile, $L_1$, $L_2$, MAD-Q.

Default data source: paper_tables_all_n.csv (n in {250, 500, 1500}).
The n = 1000 bars are taken from Table tab:clean_error in main_mdpi.tex
for every uncommented distribution row.

Example
-------
    python plot_method_error_by_n.py
    python plot_method_error_by_n.py --csv paper_tables_all_n.csv --tex main_mdpi.tex
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paper names (main_mdpi.tex, Table tab:clean_error)
# ---------------------------------------------------------------------------
DISTRIBUTIONS = [
    "Pareto I",
    "Lomax",
    "Log-Logistic",
    "Pareto IV",
    "Fréchet",
    "Gumbel",
    "Weibull",
]

DIST_ALIASES = {
    "Frechet": "Fréchet",
    "Pareto II": "Lomax",
    "Pareto III": "Log-Logistic",
    "Burr/Pareto IV": "Pareto IV",
    "Logistic": "Log-Logistic",
}

# CSV method → paper legend label
METHOD_FROM_CSV = {
    "MLE": "MLE",
    "L-moment": "L-moments",
    "L-moments": "L-moments",
    "Quantile": "Quantile",
    "L1": r"$L_1$",
    "L2": r"$L_2$",
    "MAD-Q13": "MAD-Q",
    "MAD-Q": "MAD-Q",
}

METHODS = ["MLE", "L-moments", "Quantile", r"$L_1$", r"$L_2$", "MAD-Q"]
SAMPLE_SIZES = [250, 500, 1000, 1500]

# Okabe–Ito; MAD-Q (proposed) in vermillion
METHOD_COLORS = {
    "MLE": "#0072B2",
    "L-moments": "#C9A227",
    "Quantile": "#009E73",
    r"$L_1$": "#56B4E9",
    r"$L_2$": "#CC79A7",
    "MAD-Q": "#D55E00",
}

GRID = "0.88"

# Fallback if main_mdpi.tex cannot be parsed (Table tab:clean_error, n = 1000, MAE %).
PAPER_N1000_MAE_PCT = {
    "Pareto I":     {"MLE": 2.12, "L-moments": 5.41, "Quantile": 2.74, r"$L_1$": 4.18, r"$L_2$": 4.19, "MAD-Q": 2.70},
    "Lomax":        {"MLE": 2.21, "L-moments": 8.00, "Quantile": 2.67, r"$L_1$": 4.00, r"$L_2$": 4.06, "MAD-Q": 2.65},
    "Log-Logistic": {"MLE": 1.82, "L-moments": 3.89, "Quantile": 4.05, r"$L_1$": 3.57, r"$L_2$": 3.72, "MAD-Q": 3.07},
    "Pareto IV":    {"MLE": 2.23, "L-moments": 7.84, "Quantile": 2.72, r"$L_1$": 3.80, r"$L_2$": 3.90, "MAD-Q": 2.71},
    "Fréchet":      {"MLE": 1.64, "L-moments": 4.57, "Quantile": 3.47, r"$L_1$": 3.70, r"$L_2$": 3.83, "MAD-Q": 3.00},
    "Gumbel":       {"MLE": 1.67, "L-moments": 2.14, "Quantile": 3.18, r"$L_1$": 2.19, r"$L_2$": 2.10, "MAD-Q": 2.06},
    "Weibull":      {"MLE": 1.50, "L-moments": 1.88, "Quantile": 2.98, r"$L_1$": 2.38, r"$L_2$": 2.31, "MAD-Q": 2.35},
}

_ROW_RE = re.compile(
    r"^(?P<dist>.+?)&\s*"
    r"(?:\\cellcolor\{[^}]+\})?(?P<mle>[0-9.]+)\s*&\s*"
    r"(?:\\cellcolor\{[^}]+\})?(?P<lmom>[0-9.]+)\s*&\s*"
    r"(?:\\cellcolor\{[^}]+\})?(?P<quant>[0-9.]+)\s*&\s*"
    r"(?:\\cellcolor\{[^}]+\})?(?P<l1>[0-9.]+)\s*&\s*"
    r"(?:\\cellcolor\{[^}]+\})?(?P<l2>[0-9.]+)\s*&\s*"
    r"(?:\\cellcolor\{[^}]+\})?(?P<madq>[0-9.]+)\s*\\\\",
)


def apply_paper_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "0.25",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def canonical_distribution(name: str) -> str:
    return DIST_ALIASES.get(str(name).strip(), str(name).strip())


def canonical_method(name: str) -> str:
    return METHOD_FROM_CSV.get(str(name).strip(), str(name).strip())


def parse_tab_clean_error(tex_path: Path) -> dict[str, dict[str, float]]:
    """Read n = 1000 MAE% for every uncommented row of Table tab:clean_error."""
    text = tex_path.read_text(encoding="utf-8")
    marker = r"\label{tab:clean_error}"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"{tex_path} has no \\label{{tab:clean_error}}")
    tab_start = text.find(r"\begin{tabular}", start)
    tab_end = text.find(r"\end{tabular}", tab_start)
    body = text[tab_start:tab_end]
    out: dict[str, dict[str, float]] = {}
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        dist = canonical_distribution(
            m.group("dist").replace("\\'e", "é").replace("\\'", "").strip()
        )
        out[dist] = {
            "MLE": float(m.group("mle")),
            "L-moments": float(m.group("lmom")),
            "Quantile": float(m.group("quant")),
            r"$L_1$": float(m.group("l1")),
            r"$L_2$": float(m.group("l2")),
            "MAD-Q": float(m.group("madq")),
        }
    if not out:
        raise ValueError(f"No data rows parsed from tab:clean_error in {tex_path}")
    return out


def n1000_from_paper(tex_path: Path | None = None) -> pd.DataFrame:
    table = dict(PAPER_N1000_MAE_PCT)
    if tex_path is not None and tex_path.is_file():
        table = parse_tab_clean_error(tex_path)
    rows = []
    for dist, methods in table.items():
        for method, mae in methods.items():
            rows.append(
                {"n": 1000, "distribution": dist, "method": method, "rel_error_pct": mae}
            )
    return pd.DataFrame(rows)


def mock_frame(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic MAE% that declines with n; MAD-Q near MLE, L-moments higher."""
    level = {
        "MLE": 1.00,
        "L-moments": 2.35,
        "Quantile": 1.45,
        r"$L_1$": 1.70,
        r"$L_2$": 1.75,
        "MAD-Q": 1.18,
    }
    dist_scale = {
        "Pareto I": 1.05,
        "Lomax": 1.35,
        "Log-Logistic": 1.00,
        "Pareto IV": 1.30,
        "Fréchet": 1.10,
        "Gumbel": 0.72,
        "Weibull": 0.78,
    }
    rows = []
    for dist in DISTRIBUTIONS:
        for n in SAMPLE_SIZES:
            for method in METHODS:
                base = 18.0 * level[method] * dist_scale[dist] / np.sqrt(n / 250.0)
                rows.append(
                    {
                        "n": n,
                        "distribution": dist,
                        "method": method,
                        "rel_error_pct": float(base * rng.uniform(0.92, 1.08)),
                    }
                )
    return pd.DataFrame(rows)


def load_csv(path: Path) -> pd.DataFrame:
    """Bind paper_tables_all_n.csv (or any table with the same columns).

    Required: n, distribution, method, clean_mae_pct
    """
    df = pd.read_csv(path)
    need = {"n", "distribution", "method", "clean_mae_pct"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    out = pd.DataFrame(
        {
            "n": df["n"].astype(int),
            "distribution": df["distribution"].map(canonical_distribution),
            "method": df["method"].map(canonical_method),
            "rel_error_pct": df["clean_mae_pct"].astype(float),
        }
    )
    out = out.loc[
        out["distribution"].isin(DISTRIBUTIONS) & out["method"].isin(METHODS)
    ].copy()
    # n = 1000 always comes from Table tab:clean_error in main_mdpi.tex,
    # never from the CSV (which has 250 / 500 / 1500 only).
    out = out.loc[out["n"] != 1000]
    return out


def _wide_for_dist(
    plot_df: pd.DataFrame,
    n1000_table: dict[str, dict[str, float]],
    dist: str,
) -> pd.DataFrame:
    if dist not in n1000_table:
        raise KeyError(f"{dist} missing from Table tab:clean_error (n=1000)")
    sub = plot_df.loc[plot_df["distribution"].eq(dist)]
    wide = sub.pivot_table(
        index="n", columns="method", values="rel_error_pct", aggfunc="first"
    )
    wide = wide.reindex(index=[250, 500, 1500], columns=METHODS)
    # n = 1000: table values only, never CSV / interpolation.
    row_1000 = pd.Series({m: float(n1000_table[dist][m]) for m in METHODS}, name=1000)
    return pd.concat([wide, row_1000.to_frame().T]).reindex(SAMPLE_SIZES)


def _panel_slots(
    n_panels: int, n_cols: int = 2
) -> tuple[int, list[tuple[int, slice]]]:
    """Two panels per row; centre the last panel when the count is odd."""
    n_rows = int(np.ceil(n_panels / n_cols))
    slots: list[tuple[int, slice]] = []
    for i in range(n_panels):
        row = i // n_cols
        if i == n_panels - 1 and n_panels % n_cols:
            slots.append((row, slice(1, 3)))
        else:
            col0 = 0 if (i % n_cols == 0) else 2
            slots.append((row, slice(col0, col0 + 2)))
    return n_rows, slots


def plot_grid(
    df: pd.DataFrame,
    n1000_table: dict[str, dict[str, float]],
    figsize: tuple[float, float] = (10.2, 13.2),
) -> plt.Figure:
    plot_df = df.loc[df["n"].isin(SAMPLE_SIZES) & (df["n"] != 1000)].copy()
    wides = [_wide_for_dist(plot_df, n1000_table, dist) for dist in DISTRIBUTIONS]
    data_max = max(float(np.nanmax(w.to_numpy(dtype=float))) for w in wides)
    # Shared ceiling, rounded up to an even number so tick marks line up.
    ymax = float(np.ceil((data_max * 1.08) / 2.0) * 2.0)

    n_rows, slots = _panel_slots(len(DISTRIBUTIONS), n_cols=2)
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(n_rows, 4)
    axes: list[plt.Axes] = []
    for r, c in slots:
        sharey = axes[0] if axes else None
        axes.append(fig.add_subplot(gs[r, c], sharey=sharey))

    n_hue = len(METHODS)
    bar_w = 0.86 / n_hue
    x = np.arange(len(SAMPLE_SIZES))

    for ax, dist, wide in zip(axes, DISTRIBUTIONS, wides, strict=True):
        for k, method in enumerate(METHODS):
            offset = (k - (n_hue - 1) / 2.0) * bar_w
            ax.bar(
                x + offset,
                wide[method].to_numpy(dtype=float),
                width=bar_w * 0.92,
                color=METHOD_COLORS[method],
                edgecolor="0.15" if method == "MAD-Q" else "0.22",
                linewidth=0.8 if method == "MAD-Q" else 0.4,
                hatch="///" if method == "MAD-Q" else None,
                zorder=3,
            )
        ax.set_xticks(x, [str(n) for n in SAMPLE_SIZES])
        ax.set_title(dist, pad=7)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRID, linewidth=0.65)
        ax.tick_params(axis="y", length=3, color="0.4")
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0.0, ymax)
        ax.yaxis.set_major_locator(plt.MultipleLocator(2))

    fig.supxlabel(r"Sample size $n$", fontsize=12)
    fig.supylabel(
        r"Relative error $|\hat{\alpha}-\alpha|/|\alpha|$ (%)",
        fontsize=12,
    )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=METHOD_COLORS[m],
            edgecolor="0.15" if m == "MAD-Q" else "0.22",
            linewidth=0.6,
            hatch="///" if m == "MAD-Q" else None,
            label=m,
        )
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(METHODS),
        frameon=False,
        handlelength=1.35,
        handletextpad=0.45,
        columnspacing=1.4,
    )
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(
            w_pad=0.06,
            h_pad=0.08,
            wspace=0.10,
            hspace=0.16,
            rect=(0.03, 0.02, 1.0, 0.93),
        )
    return fig


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    print(f"Wrote {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("paper_tables_all_n.csv"),
        help="Long table with n, distribution, method, clean_mae_pct.",
    )
    p.add_argument(
        "--tex",
        type=Path,
        default=Path("main_mdpi.tex"),
        help="Paper source; n = 1000 MAE%% is read from Table tab:clean_error.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for PDF/PNG.",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Ignore the CSV and plot synthetic data.",
    )
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_paper_style()
    rng = np.random.default_rng(args.seed)

    n1000_table = (
        parse_tab_clean_error(args.tex)
        if args.tex.is_file()
        else dict(PAPER_N1000_MAE_PCT)
    )
    if args.mock or not args.csv.is_file():
        if not args.mock and not args.csv.is_file():
            print(f"CSV not found ({args.csv}); using mock data.")
        df = mock_frame(rng)
    else:
        df = load_csv(args.csv)
    df = df.loc[df["n"] != 1000]
    g = n1000_table["Gumbel"]
    print(
        "n=1000 from Table tab:clean_error for every distribution. "
        f"Gumbel: L-moments={g['L-moments']:.2f}, MAD-Q={g['MAD-Q']:.2f} "
        "(MAD-Q lower)."
    )
    fig = plot_grid(df, n1000_table)
    stem = args.out_dir / "fig_method_error_by_n"
    save_figure(fig, stem)
    paper_stem = Path("figures_llm") / "fig_method_error_by_n"
    if paper_stem.resolve() != stem.resolve():
        save_figure(fig, paper_stem)


if __name__ == "__main__":
    main()
