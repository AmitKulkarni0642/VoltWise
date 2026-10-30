"""All project figures, drawn in one consistent, quiet style."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import EOL_SOH, ROOT, load_cell

FIG_DIR = ROOT / "figures"

CELL_COLORS = {"B0005": "#2a78d6", "B0006": "#eb6834", "B0007": "#1baf7a", "B0018": "#eda100"}
ACTUAL, PREDICTED, REFERENCE = "#2a78d6", "#eb6834", "#1baf7a"
AGE_SHADES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive grid and axes, ink-colored text, left-aligned title."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def _save(fig, out: Path) -> Path:
    """Save a figure tightly and close it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_soh_curves(table: pd.DataFrame, out: Path = FIG_DIR / "soh_vs_cycle.png") -> Path:
    """SOH vs discharge cycle for every cell, with the 70% end-of-life line."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for cell, g in table.groupby("cell"):
        ax.plot(g["cycle"], g["soh"] * 100, color=CELL_COLORS[cell], linewidth=2, label=cell)
        last = g.iloc[-1]
        ax.annotate(cell, (last["cycle"], last["soh"] * 100), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
    ax.axhline(EOL_SOH * 100, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(2, EOL_SOH * 100 - 1.6, "End of life (70% SOH)", color=INK_2, fontsize=9)
    _style(ax, "State of health vs discharge cycle", "Discharge cycle", "SOH (%)")
    ax.set_xlim(0, table["cycle"].max() + 14)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    return _save(fig, out)


def plot_discharge_curves(cell: str = "B0005", out: Path = FIG_DIR / "discharge_curves.png") -> Path:
    """Voltage vs time for evenly spaced cycles of one cell, light (new) to dark (aged)."""
    cycles = load_cell(cell)
    step = (len(cycles) - 1) / (len(AGE_SHADES) - 1)
    picks = [cycles[round(i * step)] for i in range(len(AGE_SHADES))]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for c, shade in zip(picks, AGE_SHADES):
        t_min = (c["time_s"] - c["time_s"][0]) / 60
        ax.plot(t_min, c["voltage_v"], color=shade, linewidth=2,
                label=f"Cycle {c['cycle']} ({c['capacity_ah']:.2f} Ah)")
    _style(ax, f"Discharge voltage curves, {cell}: curves shorten as the cell ages",
           "Time since discharge start (min)", "Terminal voltage (V)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    return _save(fig, out)


def plot_predictions(preds: pd.DataFrame, metrics: pd.DataFrame, title: str, out: Path) -> Path:
    """Small multiples: actual vs predicted SOH for each held-out cell."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=150, sharey=True)
    for ax, (cell, g) in zip(axes.flat, preds.groupby("cell")):
        rmse = metrics.loc[metrics.test_cell == cell, "rmse_pp"].iloc[0]
        ax.plot(g["cycle"], g["soh"] * 100, color=ACTUAL, linewidth=2, label="Actual")
        ax.plot(g["cycle"], g["soh_pred"] * 100, color=PREDICTED, linewidth=2, label="Predicted")
        ax.axhline(EOL_SOH * 100, color=MUTED, linestyle="--", linewidth=1)
        _style(ax, f"Held-out {cell}  (RMSE {rmse:.2f} pp)", "Discharge cycle", "SOH (%)")
        ax.title.set_fontsize(10)
    axes.flat[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out)


def plot_importance(importance: pd.Series, title: str, out: Path) -> Path:
    """Horizontal bars of normalised impurity importance, largest at the top."""
    imp = importance.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(imp) + 1.2), dpi=150)
    ax.barh(imp.index, imp.values * 100, color=ACTUAL, height=0.6)
    for y, val in enumerate(imp.values * 100):
        ax.text(val + 0.5, y, f"{val:.1f}%", va="center", fontsize=8, color=INK_2)
    _style(ax, title, "Share of impurity importance, mean over 4 folds (%)", "")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, imp.max() * 100 * 1.15)
    return _save(fig, out)


def plot_rul(rul: pd.DataFrame, source: str, out: Path) -> Path:
    """Predicted vs true RUL for each cell that reaches EOL, with a measured-SOH reference."""
    cells = sorted(rul.loc[rul.eol_cycle.notna(), "cell"].unique())
    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 4), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cell in zip(axes, cells):
        g = rul[(rul.cell == cell) & (rul.rul_true > 0)]
        model, ref = g[g.source == source], g[g.source == "measured_soh"]
        ax.plot(model["cycle"], model["rul_true"], color=INK_2, linewidth=1.5, linestyle="--",
                label="True RUL")
        ax.plot(ref["cycle"], ref["rul_pred"], color=REFERENCE, linewidth=2,
                label="From measured SOH")
        ax.plot(model["cycle"], model["rul_pred"], color=PREDICTED, linewidth=2,
                label="From predicted SOH")
        _style(ax, f"{cell} (EOL at cycle {int(g.eol_cycle.iloc[0])})", "Discharge cycle",
               "Remaining cycles to 70% SOH")
        ax.title.set_fontsize(10)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    source_label = source.replace("_", " ")
    fig.suptitle(f"RUL estimate, linear SOH trend extrapolated to 70% ({source_label})",
                 color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out)
