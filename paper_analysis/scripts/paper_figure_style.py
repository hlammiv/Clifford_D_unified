"""Shared Matplotlib style for the paper's figures.

Follows the group style of QC/circuit_knitting/circuit_knitting/scripts/paper_figure_style.py:
Okabe--Ito colors, the series order blue/circle, vermillion/square, green/triangle, black
for exact or reference curves, LaTeX text in Latin Modern, full axes box with a light grid,
markers with a thin white edge, and frameless (transparent) legends.

Semantic assignment for this paper (fixed across every figure):
  C+D            BLUE, circle      (series 1)
  C+R            ORANGE, square    (series 2)
  two qubits     GREEN, triangle   (series 3)
  bounds/limits  BLACK, dotted
  gadget model   unitary: solid line / solid bar;  measurement: dashed line / hatched bar
  gate class     T-type SKY, level-4 AMBER, R PURPLE (remaining Okabe--Ito colors;
                 used only where gates are split by class)
"""

from __future__ import annotations

import matplotlib as mpl

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
BLACK = "#000000"
YELLOW = "#F0E442"
SKY = "#56B4E9"
AMBER = "#E69F00"
PURPLE = "#CC79A7"
SERIES_MARKERS = ("o", "s", "^")

CD = dict(color=BLUE, marker="o")
CR = dict(color=ORANGE, marker="s")
QUBIT = dict(color=GREEN, marker="^")
MODEL_LS = {"unitary": "-", "meas": "--"}
MODEL_HATCH = {"unitary": None, "meas": "////"}
CLASS_COLORS = {"T": SKY, "L": AMBER, "R": PURPLE}

COL_W = 3.4          # inches, one PRA column
PAGE_W = 7.0         # inches, two columns
MARKER_KW = dict(markeredgecolor="white", markeredgewidth=0.5)


def apply_paper_style() -> None:
    """Text sized for figures placed at their final width (8--9 pt, matching the body)."""
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{bm}",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "errorbar.capsize": 2.0,
        "hatch.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,      # no box, transparent background
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
    })


def finish_axes(ax, *, xlabel: str, ylabel: str, title: str | None = None, grid_axis: str = "both") -> None:
    """Common labels and the light grid of the group style."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.15, axis=grid_axis)
