"""Static matplotlib charts for the HTML reports (saved as PNG next to each report)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Validated categorical order (fixed, never cycled) + recessive inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
# Diverging blue <-> gray <-> red: blue = row leads, red = row lags.
LAG_CMAP = LinearSegmentedColormap.from_list("lag", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_MUTED,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})


def _shade_seasons(ax, seasons: pd.Series, covid: list[str]) -> None:
    for season in covid:
        weeks = seasons.index[seasons == season]
        if len(weeks):
            ax.axvspan(weeks.min(), weeks.max(), color=SHADE, zorder=0, lw=0)
            ax.text(weeks.min(), ax.get_ylim()[1] * 0.97, f" {season}\n (COVID)",
                    va="top", fontsize=8, color=INK_MUTED)


def plot_series(wide: pd.DataFrame, locations: list[str], seasons: pd.Series,
                covid: list[str], labels: dict[str, str], out: Path) -> None:
    """Weekly ILI % for the region's states (thin coloured lines) and national (black)."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for color, loc in zip(SERIES, locations):
        ax.plot(wide.index, wide[loc], color=color, lw=1.2, label=labels[loc])
    ax.plot(wide.index, wide["nat"], color=INK, lw=2, label="National")
    _shade_seasons(ax, seasons, covid)
    ax.set_ylabel("ILI % of outpatient visits")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(ncol=min(len(locations) + 1, 5), fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_lag_heatmap(lags: pd.DataFrame, rs: pd.DataFrame, labels: dict[str, str], out: Path,
                     max_lag: int) -> None:
    """Cell [row, col] = weeks by which row leads col at peak correlation (annotated with r)."""
    n = len(lags)
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * n, 0.8 + 0.7 * n))
    ax.imshow(lags.values.astype(float), cmap=LAG_CMAP, vmin=-max_lag, vmax=max_lag)
    names = [labels[c] for c in lags.columns]
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lag, r = int(lags.iat[i, j]), rs.iat[i, j]
            ax.text(j, i, f"{lag:+d}\nr={r:.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(np.arange(-.5, n), minor=True)
    ax.set_yticks(np.arange(-.5, n), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Peak lag (weeks): + = row leads column", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_vs_national(vs: pd.DataFrame, labels: dict[str, str], out: Path, max_lag: int) -> None:
    """Horizontal bars: each state's peak lag vs national (+ = state leads national)."""
    d = vs.sort_values(["peak_lag", "peak_r"])
    fig, ax = plt.subplots(figsize=(6.5, 0.9 + 0.35 * len(d)))
    colors = [SERIES[0] if v > 0 else SERIES[7] if v < 0 else INK_MUTED for v in d["peak_lag"]]
    ax.barh([labels[x] for x in d["location"]], d["peak_lag"], color=colors, height=0.6)
    for y, (lag, r) in enumerate(zip(d["peak_lag"], d["peak_r"])):
        ax.text(lag + (0.1 if lag >= 0 else -0.1), y, f"r={r:.2f}", va="center",
                ha="left" if lag >= 0 else "right", fontsize=8, color=INK_MUTED)
    ax.axvline(0, color=INK_MUTED, lw=0.8)
    ax.set_xlim(-max_lag - 1, max_lag + 1)
    ax.set_xlabel("Peak lag vs national (weeks; + = state moves first)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
