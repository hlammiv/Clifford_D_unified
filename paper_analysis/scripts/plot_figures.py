#!/usr/bin/env python3
"""plot_figures.py — every data figure in the paper, from the canonical CSVs (read-only).

  fig/headline.pdf     (a) per-phase count, (b) T gates per rotation, C+D vs C+R
  fig/composition.pdf  where the gates go at eps = 1e-10, by gate class
  fig/qubit.pdf        magic states per arbitrary single-qutrit gate vs two-qubit emulation
  fig/angles.pdf       error at the special angles vs error inherited by a random angle

Colors, markers, line styles, and typography come from paper_figure_style.py (the group style;
see its docstring for the semantic assignment used in every figure).

Usage: python3 scripts/plot_figures.py
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_numbers as mn  # noqa: E402
import paper_figure_style as st  # noqa: E402

FIG = mn.ROOT / "fig"
FIG.mkdir(exist_ok=True)
L10_3 = math.log(10, 3)
CDL = r"$(\mathbf{C}+\mathbf{D})_3$"
CRL = r"$(\mathbf{C}+\mathbf{R})_3$"
TQ = r"$\mathbf{T}$"
MODEL_LABEL = {"unitary": "unitary", "meas": "measurement"}


def numbers():
    txt = (mn.ROOT / "numbers.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", txt, re.M))


def per_level_means(pts):
    """(log3(1/eps), y) pairs -> per-denominator-level means (log10 x, y, standard error)."""
    by = collections.defaultdict(list)
    for x, y in pts:
        by[round(x / 3)].append((x, y))      # levels sit ~3 units of log3 apart
    out = []
    for _, v in sorted(by.items()):
        ys = np.array([p[1] for p in v])
        out.append((np.median([p[0] for p in v]) / L10_3, ys.mean(), ys.std(ddof=1) / np.sqrt(len(ys))))
    return np.array(out)


def headline(rows, pts):
    delta, _ = mn.eps_penalty()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(st.PAGE_W, 2.7))
    xx = np.linspace(0, 11, 50)
    x3 = xx * L10_3

    # (a) per-phase count
    xs3 = [mn.L3(r["epsilon"]) for r in rows]
    ys = [r["N_D"] for r in rows]
    f = mn.jackfit(xs3, ys)
    m = per_level_means(zip(xs3, ys))
    # every computed approximant, faint (group style: raw samples at alpha ~0.15)
    a1.scatter(np.array(xs3) / L10_3, ys, s=4, color=st.BLUE, alpha=0.15, lw=0, rasterized=True, zorder=2)
    a1.errorbar(m[:, 0], m[:, 1], yerr=m[:, 2], fmt=st.CD["marker"], color=st.BLUE, zorder=4,
                label=CDL + r", special $\theta$", **st.MARKER_KW)
    a1.plot(xx, f["a"] + f["b"] * x3, ":", color=st.BLUE, lw=1.0)
    a1.plot(xx, f["a"] + f["b"] * (x3 + delta), "-", color=st.BLUE, label=CDL + r", prescribed $\theta$")
    a1.plot(xx, mn.CR["Householder"][0] + mn.CR["Householder"][1] * xx, "-", color=st.ORANGE,
            label=CRL + ", Householder")
    a1.plot(xx, mn.CR["Exhaustive"][0] + mn.CR["Exhaustive"][1] * xx, "-.", color=st.ORANGE,
            label=CRL + ", exhaustive")
    a1.plot(xx, -2.16 + 4.90 * x3, ":", color=st.BLACK, lw=1.0, label="covering bound")
    st.finish_axes(a1, xlabel=r"$\log_{10}(1/\varepsilon)$", ylabel=r"per-phase count $N_\varphi$")
    a1.set_xlim(0, 11); a1.set_ylim(0, 140)
    a1.legend(loc="upper left")
    a1.text(0.97, 0.04, "(a)", transform=a1.transAxes, ha="right")

    # (b) T gates per rotation: C+D markers are special-angle means, lines are at a prescribed angle
    for model in ("unitary", "meas"):
        p = np.array(pts[(model, "best")])
        g = mn.jackfit(p[:, 0], p[:, 1])
        mm = per_level_means(map(tuple, p))
        if model == "unitary":
            a2.scatter(p[:, 0] / L10_3, p[:, 1], s=4, color=st.BLUE, alpha=0.25, lw=0, rasterized=True, zorder=2)
        else:
            a2.scatter(p[:, 0] / L10_3, p[:, 1], s=5, facecolors="none", edgecolors=st.BLUE, alpha=0.4,
                       linewidths=0.4, rasterized=True, zorder=2)
        a2.errorbar(mm[:, 0], mm[:, 1], yerr=mm[:, 2], fmt=st.CD["marker"], color=st.BLUE, zorder=4,
                    mfc=st.BLUE if model == "unitary" else "white",
                    markeredgecolor="white" if model == "unitary" else st.BLUE, markeredgewidth=0.8)
        a2.plot(xx, g["a"] + g["b"] * (x3 + delta), st.MODEL_LS[model], color=st.BLUE,
                label=f"{CDL}, {MODEL_LABEL[model]}")
    for model in ("unitary", "meas"):
        a, b = mn.CR["Householder"]
        a2.plot(xx, mn.R_COST[model] * (a + b * xx), st.MODEL_LS[model], color=st.ORANGE,
                label=f"{CRL}, {MODEL_LABEL[model]}")
    st.finish_axes(a2, xlabel=r"$\log_{10}(1/\varepsilon)$", ylabel=TQ + r" gates per $R_z(\theta)$")
    a2.set_xlim(0, 11); a2.set_ylim(0, 800)
    a2.legend(loc="upper left")
    a2.text(0.97, 0.04, "(b)", transform=a2.transAxes, ha="right")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "headline.pdf", dpi=400)


def composition():
    """Class content of one rotation at a prescribed angle and eps = 1e-10."""
    c = json.loads((mn.ROOT / "tables" / "composition.json").read_text())
    k = c["classes"]["unitary"]            # the same copy is cheapest in both models (w4 = wR)
    nT, n4, nR = k["n_T3"], k["n_L4"], k["n_R"]
    NR = c["CR_NR"]["Householder"]
    col = st.CLASS_COLORS
    fig, axes = plt.subplots(2, 1, figsize=(st.COL_W, 2.7), gridspec_kw=dict(height_ratios=[2, 4], hspace=0.85))

    def bars(ax, rows, xmax, xlabel):
        for y, (_, segs, hatch) in enumerate(rows):
            left = 0.0
            for val, color in segs:
                ax.barh(y, val, left=left, color=color, height=0.62, edgecolor="white" if not hatch else "white",
                        linewidth=0.8, hatch=hatch)
                left += val
            ax.text(left + xmax * 0.012, y, f"{left:.0f}", va="center", fontsize=6.5)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        st.finish_axes(ax, xlabel=xlabel, ylabel="", grid_axis="x")

    bars(axes[0], [(CDL, [(2 * nT, col["T"]), (n4, col["L"]), (nR, col["R"])], None),
                   (CRL, [(NR, col["R"])], None)], 140, r"per-phase count $N_\varphi$")
    rows = []
    for model in ("unitary", "meas"):
        w = c["R_COST"][model]
        h = st.MODEL_HATCH[model]
        rows.append((f"{CDL}, {MODEL_LABEL[model]}", [(nT, col["T"]), (w * n4, col["L"]), (w * nR, col["R"])], h))
        rows.append((f"{CRL}, {MODEL_LABEL[model]}", [(w * NR, col["R"])], h))
    bars(axes[1], rows, 880, TQ + " gates")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col[x]) for x in "TLR"]
    axes[0].legend(handles, [r"$T$-type", "level-4", r"$\mathbf{R}$"], loc="lower right", ncol=3,
                   bbox_to_anchor=(1.0, 1.02), handlelength=1.0, columnspacing=0.8)
    fig.savefig(FIG / "composition.pdf")


def qubit():
    """Magic states per arbitrary single-qutrit gate (6 rotations) vs two-qubit emulation (10 R_z)."""
    n = numbers()
    num = lambda key: float(n[key].replace("{,}", ""))  # noqa: E731
    rows = [(f"qutrit {CDL}, unitary", num("Qutritunitary"), st.BLUE, st.MODEL_HATCH["unitary"]),
            (f"qutrit {CDL}, measurement", num("Qutritmeas"), st.BLUE, st.MODEL_HATCH["meas"]),
            ("two qubits, deterministic", num("QubitDet"), st.GREEN, None),
            ("two qubits, RUS", num("QubitRUS"), st.GREEN, "////")]
    fig, ax = plt.subplots(figsize=(st.COL_W, 1.8))
    for y, (lab, v, color, hatch) in enumerate(rows):
        if hatch:
            ax.barh(y, v, height=0.62, color="white", edgecolor=color, hatch=hatch, linewidth=0.8)
        else:
            ax.barh(y, v, height=0.62, color=color, edgecolor=color, linewidth=0.8)
        ax.text(v + 20, y, f"{v:,.0f}", va="center", fontsize=6.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1600)
    st.finish_axes(ax, xlabel=r"magic states per arbitrary gate at $\varepsilon=10^{-10}$", ylabel="", grid_axis="x")
    fig.savefig(FIG / "qubit.pdf")


def angles():
    """Error at the special angles vs the error a random target inherits from its nearest one."""
    d = mn.ROOT / "analysis" / "special_angles"
    rng = np.random.default_rng(0)
    fs, e_spec, e_near, cnt = [], [], [], []
    for f in (4, 6, 8, 10, 12, 14, 16):
        z = np.load(d / f"ang_f{f}.npz")
        th, eps = z["th"], z["eps"]
        keep = np.concatenate([[True], np.diff(th) > 1e-12])
        th, eps = th[keep], eps[keep]
        T = rng.uniform(0, np.pi, 20000)
        idx = np.searchsorted(th, T)
        best = np.full(T.shape, np.inf)
        for k in (idx - 1, idx):
            k = np.clip(k, 0, len(th) - 1)
            best = np.minimum(best, eps[k] + 2 * np.sqrt(2) * np.abs(np.sin(np.abs(T - th[k]) / 4)))
        fs.append(f); e_spec.append(np.median(eps)); e_near.append(np.median(best)); cnt.append(len(th))
    fig, ax = plt.subplots(figsize=(st.COL_W, 2.4))
    ax.semilogy(fs, e_spec, "-", marker="o", color=st.BLUE, label="at the special angles", **st.MARKER_KW)
    ax.semilogy(fs, e_near, "--", marker="o", color=st.BLUE, mfc="white", markeredgecolor=st.BLUE,
                markeredgewidth=0.8, label=r"random $\theta$, nearest special angle")
    for f, y, k in zip(fs, e_near, cnt):
        ax.annotate(f"{k:,}", (f, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=5.5)
    st.finish_axes(ax, xlabel=r"denominator exponent $f$", ylabel=r"median $\varepsilon$")
    ax.legend(loc="lower left")
    fig.savefig(FIG / "angles.pdf")


def main():
    st.apply_paper_style()
    rows = mn.load_nick()
    pts, _, _ = mn.load_stack()
    headline(rows, pts)
    composition()
    qubit()
    angles()
    print("wrote", *sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
