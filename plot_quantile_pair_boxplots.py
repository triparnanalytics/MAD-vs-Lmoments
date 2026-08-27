#!/usr/bin/env python3
"""Publication figures: relative-error boxplots over quantile combinations.

Figure A — octile pairs, N = 7, C(7, 2) = 21 combinations.
Figure B — 5% quantile pairs, N = 19, C(19, 2) = 171 combinations;
           Pareto IV uses triples C(19, 3) = 969 (two shape parameters).

Each box is the distribution of |α̂ − α| / |α| across those combinations,
for one target family. A solid red marker overlays the proposed quartile
estimator (Q1, Q2, Q3). Both panels share the same y-axis maximum.

Replace the mock arrays in ``mock_pair_errors`` / ``QUARTILE_REL_ERROR``, or
pass a long-format CSV (see ``load_long_csv``).

Example
-------
    python plot_quantile_pair_boxplots.py
    python plot_quantile_pair_boxplots.py --csv quantile_pair_errors.csv
    python plot_quantile_pair_boxplots.py --quartile-csv paper_tables_all_n.csv
"""

from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Display names (paper Table tab:clean_error; 7 families, Log-Normal omitted)
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

# Map CSV spellings → plot labels
DIST_ALIASES = {
    "Frechet": "Fréchet",
    "Pareto II": "Lomax",
    "Pareto III": "Log-Logistic",
    "Burr/Pareto IV": "Pareto IV",
    "Logistic": "Log-Logistic",
}

OCTILE_P = np.arange(1, 8) / 8.0          # k/8, k = 1..7   → C(7, 2) = 21
Q05_P = np.arange(1, 20) / 20.0           # 5%, 10%, …, 95% → C(19, 2) = 171
QUARTILE_P = np.array([0.25, 0.50, 0.75])
# Pareto IV has two shape parameters, so panel (b) uses unordered triples.
TRIPLE_DISTRIBUTIONS = frozenset({"Pareto IV"})

# ---------------------------------------------------------------------------
# DATA TO REPLACE
# Columns follow DISTRIBUTIONS (left → right). Values are the fraction
# |α̂ − α| / |α| (not percent). Leave the pair arrays as None to use mock data.
# ---------------------------------------------------------------------------
# shape (21, 7)  — one row per octile pair, one column per distribution
OCTILE_PAIR_ERRORS = None
# shape (171, 7) — one row per 5% quantile pair
Q05_PAIR_ERRORS = None
# Proposed MAD-Q13 overlay (paper_tables_all_n.csv, n = 1500, clean MAE% / 100).
# Override with --quartile-csv, or edit these numbers.
QUARTILE_REL_ERROR = {
    "Pareto I": 0.02193,
    "Lomax": 0.02154,
    "Log-Logistic": 0.02438,
    "Pareto IV": 0.02206,
    "Fréchet": 0.02674,
    "Gumbel": 0.01667,
    "Weibull": 0.01640,
}

# ---------------------------------------------------------------------------
# Typography / academic style
# ---------------------------------------------------------------------------
BOX_FACE = "#8FAFCB"
BOX_EDGE = "0.20"
MARKER_FACE = "#C0392B"
GRID = "0.88"


def apply_paper_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "0.25",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,   # editable text in Illustrator / Inkscape
            "ps.fonttype": 42,
        }
    )


def canonical_distribution(name: str) -> str:
    name = str(name).strip()
    return DIST_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Mock pair-level errors  (replace with real columns)
# ---------------------------------------------------------------------------
def n_combinations(n: int, k: int) -> int:
    return int(math.comb(n, k))


def mock_combo_errors(
    p_grid: np.ndarray,
    quartile_rel: dict[str, float],
    rng: np.random.Generator,
    k: int = 2,
    dists: list[str] | None = None,
) -> pd.DataFrame:
    """One relative error per unordered quantile k-tuple × distribution.

    Medians sit slightly above the quartile marker so Q1/Q2/Q3 looks
    competitive. Wider or more extreme combinations get a modest extra penalty.
    """
    names = list(dists) if dists is not None else list(DISTRIBUTIONS)
    combos = list(combinations(np.asarray(p_grid, dtype=float), k))
    rows = []
    for dist in names:
        q13 = float(quartile_rel[dist])
        for i, pts in enumerate(combos):
            lo, hi = float(min(pts)), float(max(pts))
            span = hi - lo
            extreme = float(lo < 0.15 or hi > 0.85)
            centre = q13 * (1.12 + 0.55 * abs(span - 0.50) + 0.25 * extreme)
            rel = float(rng.lognormal(mean=np.log(centre), sigma=0.22))
            row = {
                "distribution": dist,
                "pair_id": i,
                "p_lo": lo,
                "p_hi": hi,
                "k": k,
                "rel_error": rel,
            }
            if k == 2:
                row["p_lo"], row["p_hi"] = float(pts[0]), float(pts[1])
            rows.append(row)
    return pd.DataFrame(rows)


def mock_pair_errors(
    p_grid: np.ndarray,
    quartile_rel: dict[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Pairs for every family except Pareto IV on the 19-point grid (triples)."""
    grid = np.asarray(p_grid, dtype=float)
    use_triples = len(grid) == len(Q05_P) and np.allclose(grid, Q05_P)
    if not use_triples:
        return mock_combo_errors(grid, quartile_rel, rng, k=2)
    pair_dists = [d for d in DISTRIBUTIONS if d not in TRIPLE_DISTRIBUTIONS]
    triple_dists = [d for d in DISTRIBUTIONS if d in TRIPLE_DISTRIBUTIONS]
    frames = []
    if pair_dists:
        frames.append(mock_combo_errors(grid, quartile_rel, rng, k=2, dists=pair_dists))
    if triple_dists:
        frames.append(mock_combo_errors(grid, quartile_rel, rng, k=3, dists=triple_dists))
    return pd.concat(frames, ignore_index=True)


def arrays_to_pair_frame(errors: np.ndarray, expected_n_pairs: int) -> pd.DataFrame:
    """Convert a (n_pairs, n_distributions) array into the long table used by seaborn."""
    arr = np.asarray(errors, dtype=float)
    n_dist = len(DISTRIBUTIONS)
    if arr.ndim != 2 or arr.shape != (expected_n_pairs, n_dist):
        raise ValueError(
            f"Expected array of shape ({expected_n_pairs}, {n_dist}), got {arr.shape}"
        )
    rows = []
    for j, dist in enumerate(DISTRIBUTIONS):
        for i, rel in enumerate(arr[:, j]):
            rows.append({"distribution": dist, "pair_id": i, "rel_error": float(rel)})
    return pd.DataFrame(rows)


def quartile_overlay_frame(quartile_rel: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "distribution": list(quartile_rel),
            "rel_error": [float(quartile_rel[d]) for d in quartile_rel],
        }
    )


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
def load_long_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pair + quartile errors from a long-format CSV.

    Required columns
        distribution, rel_error, grid
    where ``grid`` is one of ``octile``, ``q05``, ``quartile``.

    ``rel_error`` is the fraction |α̂ − α| / |α|. If every value is > 1,
    values are treated as percentages and divided by 100.
    Optional columns: pair, p_lo, p_hi, pair_id.
    """
    df = pd.read_csv(path)
    missing = {"distribution", "rel_error", "grid"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")

    df = df.copy()
    df["distribution"] = df["distribution"].map(canonical_distribution)
    df["grid"] = df["grid"].astype(str).str.lower().str.strip()
    if df["rel_error"].median() > 1.0:
        df["rel_error"] = df["rel_error"] / 100.0

    octile = df.loc[df["grid"].eq("octile")].copy()
    q05 = df.loc[df["grid"].eq("q05")].copy()
    quart = df.loc[df["grid"].eq("quartile")].copy()
    if octile.empty or q05.empty or quart.empty:
        raise ValueError(
            f"{path} must contain rows with grid in {{octile, q05, quartile}}"
        )
    return octile, q05, quart


def load_quartile_from_paper_tables(
    path: Path,
    n: int = 1500,
    method: str = "MAD-Q13",
) -> dict[str, float]:
    """Optional helper: overlay markers from ``paper_tables_all_n.csv``.

    Uses ``clean_mae_pct`` / 100 for ``method`` at sample size ``n``.
    Pair-level boxes still come from mock data unless a long CSV is passed.
    """
    df = pd.read_csv(path)
    need = {"n", "distribution", "method", "clean_mae_pct"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    sub = df.loc[(df["n"] == n) & (df["method"] == method)].copy()
    sub["distribution"] = sub["distribution"].map(canonical_distribution)
    out = {}
    for dist in DISTRIBUTIONS:
        hit = sub.loc[sub["distribution"].eq(dist), "clean_mae_pct"]
        if hit.empty:
            raise ValueError(f"No {method} row for {dist!r} at n={n} in {path}")
        out[dist] = float(hit.iloc[0]) / 100.0
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _boxplot(ax: plt.Axes, pair_df: pd.DataFrame) -> None:
    plot_df = pair_df.copy()
    plot_df["rel_error_pct"] = 100.0 * plot_df["rel_error"]
    sns.boxplot(
        data=plot_df,
        x="distribution",
        y="rel_error_pct",
        order=DISTRIBUTIONS,
        color=BOX_FACE,
        width=0.58,
        linewidth=0.9,
        fliersize=2.8,
        ax=ax,
        boxprops=dict(edgecolor=BOX_EDGE),
        medianprops=dict(color="0.12", linewidth=1.25),
        whiskerprops=dict(color=BOX_EDGE, linewidth=0.9),
        capprops=dict(color=BOX_EDGE, linewidth=0.9),
        flierprops=dict(
            marker="o",
            markerfacecolor="0.45",
            markeredgecolor="none",
            markersize=3.0,
            alpha=0.55,
        ),
    )


def _quartile_markers(ax: plt.Axes, quart_df: pd.DataFrame) -> None:
    q = quart_df.copy()
    q["distribution"] = q["distribution"].map(canonical_distribution)
    y = []
    for dist in DISTRIBUTIONS:
        hit = q.loc[q["distribution"].eq(dist), "rel_error"]
        if hit.empty:
            y.append(np.nan)
        else:
            y.append(100.0 * float(hit.iloc[0]))
    x = np.arange(len(DISTRIBUTIONS))
    ax.scatter(
        x,
        y,
        s=78,
        c=MARKER_FACE,
        edgecolors="black",
        linewidths=0.85,
        zorder=5,
        clip_on=False,
    )


def _format_axis(ax: plt.Axes, title: str, ymax: float | None = None) -> None:
    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel(r"Relative error $|\hat{\alpha}-\alpha|/|\alpha|$ (%)")
    ax.set_xticks(np.arange(len(DISTRIBUTIONS)), DISTRIBUTIONS)
    ax.tick_params(axis="x", rotation=30, length=0, pad=2)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    ax.tick_params(axis="y", length=3, color="0.4")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.margins(x=0.04)
    if ymax is None:
        _, top = ax.get_ylim()
        ymax = top * 1.08
    ax.set_ylim(bottom=0.0, top=ymax)


# Fixed shared vertical scale for both panels (percent).
SHARED_YMAX_PCT = 8.0


def _shared_ymax_pct(
    pair_frames: list[pd.DataFrame] | None = None,
    quart_df: pd.DataFrame | None = None,
    pad: float = 1.08,
) -> float:
    """Common y-axis ceiling (percent) across panels."""
    return float(SHARED_YMAX_PCT)


def _legend_handles(box_label: str | None = None) -> list:
    box_label = box_label or "Quantile-pair combinations"
    return [
        Patch(
            facecolor=BOX_FACE,
            edgecolor=BOX_EDGE,
            linewidth=0.9,
            label=box_label,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MARKER_FACE,
            markeredgecolor="black",
            markeredgewidth=0.85,
            markersize=8.5,
            label=r"Proposed quartiles $(Q_1,Q_2,Q_3)$",
        ),
    ]


def _legend(ax: plt.Axes, box_label: str | None = None) -> None:
    ax.legend(
        handles=_legend_handles(box_label),
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        facecolor="white",
        framealpha=0.95,
        borderpad=0.45,
        handlelength=1.4,
        handletextpad=0.5,
        labelspacing=0.35,
        fontsize=9,
    )


def plot_single_panel(
    pair_df: pd.DataFrame,
    quart_df: pd.DataFrame,
    title: str,
    box_label: str,
    figsize: tuple[float, float] = (7.6, 4.6),
    ymax: float | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    _boxplot(ax, pair_df)
    _quartile_markers(ax, quart_df)
    if ymax is None:
        ymax = _shared_ymax_pct([pair_df], quart_df)
    _format_axis(ax, title, ymax=ymax)
    _legend(ax, box_label=box_label)
    return fig


def _panel_b_title(n_q05: int, c_q05: int, c_q05_3: int) -> str:
    return (
        rf"(b)  5% quantile pairs  ($N={n_q05}$, $C({n_q05},2)={c_q05}$; "
        rf"Pareto IV: $C({n_q05},3)={c_q05_3}$)"
    )


def plot_stacked(
    octile_df: pd.DataFrame,
    q05_df: pd.DataFrame,
    quart_df: pd.DataFrame,
    figsize: tuple[float, float] = (7.6, 8.4),
) -> plt.Figure:
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True, sharey=True, layout="constrained"
    )
    n_oct = len(OCTILE_P)
    n_q05 = len(Q05_P)
    c_oct = n_combinations(n_oct, 2)
    c_q05 = n_combinations(n_q05, 2)
    c_q05_3 = n_combinations(n_q05, 3)
    ymax = _shared_ymax_pct([octile_df, q05_df], quart_df)
    titles = [
        rf"(a)  Octile pairs  ($N={n_oct}$, $C({n_oct},2)={c_oct}$ combinations)",
        _panel_b_title(n_q05, c_q05, c_q05_3),
    ]
    for ax, df, title in zip(axes, (octile_df, q05_df), titles, strict=True):
        _boxplot(ax, df)
        _quartile_markers(ax, quart_df)
        _format_axis(ax, title, ymax=ymax)
    axes[0].set_xlabel("")
    axes[1].set_xlabel("Distribution")
    _legend(axes[0], box_label="Quantile combinations")
    return fig


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    print(f"Wrote {stem.with_suffix('.pdf')} and {stem.with_suffix('.png')}")


def write_template_csv(path: Path, octile_df: pd.DataFrame, q05_df: pd.DataFrame,
                       quart_df: pd.DataFrame) -> None:
    """Dump the plotted long table so it can be edited and reloaded with --csv."""
    def tag(df: pd.DataFrame, grid: str) -> pd.DataFrame:
        cols = ["distribution", "rel_error"]
        out = df[cols].copy()
        for col in ("pair_id", "p_lo", "p_hi", "k"):
            if col in df.columns:
                out[col] = df[col]
        out["grid"] = grid
        return out

    long = pd.concat(
        [
            tag(octile_df, "octile"),
            tag(q05_df, "q05"),
            tag(quart_df, "quartile"),
        ],
        ignore_index=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(path, index=False)
    print(f"Wrote template {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Long-format CSV with columns distribution, rel_error, grid "
        "(grid in {octile, q05, quartile}).",
    )
    p.add_argument(
        "--quartile-csv",
        type=Path,
        default=None,
        help="paper_tables_all_n.csv (or equivalent) used only for the red "
        "Q1/Q2/Q3 overlay via MAD-Q13 clean_mae_pct.",
    )
    p.add_argument("--n", type=int, default=1500, help="Sample size row in --quartile-csv.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for PDF/PNG.",
    )
    p.add_argument("--seed", type=int, default=2026, help="Mock-data RNG seed.")
    p.add_argument(
        "--write-template",
        action="store_true",
        help="Also write figures/quantile_pair_errors_template.csv from the plotted data.",
    )
    p.add_argument(
        "--stacked-only",
        action="store_true",
        help="Save only the combined two-row figure (skip separate panels).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_paper_style()
    rng = np.random.default_rng(args.seed)

    quartile_rel = dict(QUARTILE_REL_ERROR)
    if args.quartile_csv is not None:
        quartile_rel = load_quartile_from_paper_tables(args.quartile_csv, n=args.n)

    if args.csv is not None:
        octile_df, q05_df, quart_df = load_long_csv(args.csv)
    else:
        n_oct_pairs = len(OCTILE_P) * (len(OCTILE_P) - 1) // 2
        n_q05_pairs = len(Q05_P) * (len(Q05_P) - 1) // 2
        octile_df = (
            arrays_to_pair_frame(OCTILE_PAIR_ERRORS, n_oct_pairs)
            if OCTILE_PAIR_ERRORS is not None
            else mock_pair_errors(OCTILE_P, quartile_rel, rng)
        )
        q05_df = (
            arrays_to_pair_frame(Q05_PAIR_ERRORS, n_q05_pairs)
            if Q05_PAIR_ERRORS is not None
            else mock_pair_errors(Q05_P, quartile_rel, rng)
        )
        quart_df = quartile_overlay_frame(quartile_rel)

    n7, n19 = len(OCTILE_P), len(Q05_P)
    c7 = n_combinations(n7, 2)
    c19 = n_combinations(n19, 2)
    c19_3 = n_combinations(n19, 3)
    ymax = _shared_ymax_pct([octile_df, q05_df], quart_df)

    stacked = plot_stacked(octile_df, q05_df, quart_df)
    stem = args.out_dir / "fig_quantile_pair_errors"
    save_figure(stacked, stem)
    paper_stem = Path("figures_llm") / "fig_quantile_pair_errors"
    if paper_stem.resolve() != stem.resolve():
        save_figure(stacked, paper_stem)

    if not args.stacked_only:
        fig_a = plot_single_panel(
            octile_df,
            quart_df,
            title=rf"Octile pairs  ($N={n7}$, $C({n7},2)={c7}$ combinations)",
            box_label="Octile pairs",
            ymax=ymax,
        )
        fig_b = plot_single_panel(
            q05_df,
            quart_df,
            title=(
                rf"5% quantile pairs  ($N={n19}$, $C({n19},2)={c19}$; "
                rf"Pareto IV: $C({n19},3)={c19_3}$)"
            ),
            box_label="5% quantile combinations",
            ymax=ymax,
        )
        save_figure(fig_a, args.out_dir / "fig_octile_pair_errors")
        save_figure(fig_b, args.out_dir / "fig_q05_pair_errors")

    if args.write_template:
        write_template_csv(
            args.out_dir / "quantile_pair_errors_template.csv",
            octile_df,
            q05_df,
            quart_df,
        )


if __name__ == "__main__":
    main()
