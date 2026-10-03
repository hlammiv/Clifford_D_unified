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
  gadget model   unitary: solid line, filled marker, plain bar
                 measurement: dashed line, open marker, bar with white hatching
  gate class     shade of the gate-set color: T-type lightest, level-4 medium, R full color
                 (only where a bar is split by class)
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
CLASS_SHADE = {"T": 0.35, "L": 0.65, "R": 1.0}      # fraction of the gate-set color (rest white)
BAR_HATCH = {"unitary": None, "meas": "////"}       # hatch lines drawn in white over the fill


def shade(color: str, frac: float) -> tuple:
    """Blend a color with white: frac = 1 gives the color, 0 gives white."""
    r, g, b = mpl.colors.to_rgb(color)
    return (1 - frac + frac * r, 1 - frac + frac * g, 1 - frac + frac * b)

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
