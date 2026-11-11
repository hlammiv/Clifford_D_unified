#!/usr/bin/env python3
"""plot_nd_vs_eps_v2.py — N_D vs ε plots, two panels (target ε + achieved frob).

Two panels:
  Left  — N_D vs TARGET ε  (the spec we asked the backend to meet)
  Right — N_D vs ACHIEVED Frobenius distance (what the backend actually found)

Colored by METHOD (not angle). One scatter per (method, ε) cell.

Takes any number of CSVs as positional args. Recognizes:
  - HRSA sweep CSVs (columns: theta, epsilon, method, N_D, achieved_frob, all_checks_pass)
  - zeta9 batched sweep CSVs (columns: max_f, theta, eps_target, achieved_frob, N_D, method, success)
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """Return list of dicts with normalized keys: theta, eps_target, method, N_D, achieved_frob, success.

    For zeta9 CSVs, the recorded method is "zeta9-householder" but the *max-f*
    knob varies per cell — we splice that into the label as "zeta9(max-f=N)"
    so the legend distinguishes different zeta9 capability levels.
    """
    path = Path(path)
    out = []
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return out
    # Detect schema
    if "epsilon" in rows[0]:  # HRSA
        for r in rows:
            try:
                if r.get("all_checks_pass") != "true":
                    continue
                out.append({
                    "theta": float(r["theta"]),
                    "eps_target": float(r["epsilon"]),
                    "method": r["method"],
                    "N_D": int(r["N_D"]),
                    "achieved_frob": float(r["achieved_frob"]),
                    "source": path.name,
                })
            except (ValueError, TypeError, KeyError):
                continue
    elif "eps_target" in rows[0]:  # zeta9 batched sweep
        for r in rows:
            try:
                if r.get("success") not in ("True", "true", "1"):
                    continue
                if not r.get("N_D"):
                    continue
                # Relabel "zeta9-householder" → "zeta9(max-f=N)" so the legend
                # distinguishes V-denom levels (max-f=2 today, max-f=3 later).
                base = r["method"]
                mf = r.get("max_f", "")
                if "zeta9" in base and mf:
                    method = f"zeta9(max-f={mf})"
                else:
                    method = base
                out.append({
                    "theta": float(r["theta"]),
                    "eps_target": float(r["eps_target"]),
                    "method": method,
                    "N_D": int(r["N_D"]),
                    "achieved_frob": float(r["achieved_frob"]),
                    "source": path.name,
                })
            except (ValueError, TypeError, KeyError):
                continue
    else:
        sys.stderr.write(f"warning: unknown schema in {path}\n")
    return out


def main():
    if len(sys.argv) < 3:
        sys.stderr.write(
            "usage: plot_nd_vs_eps_v2.py OUT.png CSV [CSV ...]\n"
        )
        sys.exit(1)
    argv = sys.argv[1:]
    no_combined_fit = "--no-combined-fit" in argv
    argv = [a for a in argv if a != "--no-combined-fit"]
    out_png = Path(argv[0])
    csv_paths = [Path(p) for p in argv[1:]]

    pts = []
    for p in csv_paths:
        pts.extend(load_csv(p))
    print(f"loaded {len(pts)} successful cells across {len(csv_paths)} CSVs")
    if not pts:
        sys.exit("no successful cells found")

    # Method cascade order: HRSA_tester tries Direct → SignExtClifford →
    # bidir(k+k) → HRSA(f=k), then zeta9 picks up the long tail at tight ε.
    # Legend reads top-to-bottom in this order; colors progress along the
    # cascade so close-cascade methods have similar hues (viridis).
    method_order = [
        "Direct",
        "SignExtClifford",
        "bidir(0+0)", "bidir(1+1)", "bidir(2+2)", "bidir(3+3)", "bidir(4+4)", "bidir(5+5)",
        "HRSA(f=0)", "HRSA(f=1)", "HRSA(f=2)", "HRSA(f=3)", "HRSA(f=4)",
        "zeta9(max-f=0)", "zeta9(max-f=1)", "zeta9(max-f=2)",
        "zeta9(max-f=3)", "zeta9(max-f=4)",
        "zeta9-householder",  # fallback if max_f wasn't in CSV
        "nick(f=4)", "nick(f=6)", "nick(f=8)",
        "nick(f=10)", "nick(f=12)", "nick(f=14)", "nick(f=16)",  # exact-ring Z-rot fits
    ]
    # Highlight series: exact-ring direct fits (Nick's Z-rotations), plotted
    # with a distinct star marker + warm colors instead of the viridis cascade.
    HIGHLIGHT = {
        "nick(f=4)":  "#ffbb33",
        "nick(f=6)":  "#ff7f0e",
        "nick(f=8)":  "#e6550d",
        "nick(f=10)": "#d62728",
        "nick(f=12)": "#8c2d04",
        "nick(f=14)": "#9467bd",
        "nick(f=16)": "#4a1486",
    }
    methods_in_data = sorted(set(p["method"] for p in pts),
                              key=lambda m: (method_order.index(m) if m in method_order else 99, m))
    # viridis traverses dark→light along the cascade. Use ALL known methods
    # for the colormap index space (not just the ones in data) so colors are
    # consistent across replots as more methods appear.
    cmap = plt.get_cmap("viridis", len(method_order))
    method_color = {m: cmap(method_order.index(m)) if m in method_order else (0.5, 0.5, 0.5)
                    for m in methods_in_data}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Group points by method for plotting
    by_method = defaultdict(list)
    for p in pts:
        by_method[p["method"]].append(p)

    for m in methods_in_data:
        ps = by_method[m]
        eps_t = np.array([p["eps_target"] for p in ps])
        frob = np.array([p["achieved_frob"] for p in ps])
        nd = np.array([p["N_D"] for p in ps])
        if m in HIGHLIGHT:
            kw = dict(c=HIGHLIGHT[m], s=90, alpha=0.9, marker="*",
                      edgecolors="black", linewidths=0.4, zorder=5)
        else:
            kw = dict(c=[method_color[m]], s=25, alpha=0.6, edgecolors="none")
        ax1.scatter(eps_t, nd, label=f"{m} (n={len(ps)})", **kw)
        ax2.scatter(frob, nd, label=f"{m} (n={len(ps)})", **kw)

    # Joint x-range across both panels so a cell at (ε_target, achieved_frob)
    # lines up vertically when its target ε and achieved frob are equal.
    all_eps = np.array([p["eps_target"] for p in pts if p["eps_target"] > 0])
    all_frob = np.array([p["achieved_frob"] for p in pts if p["achieved_frob"] > 0])
    xmin = min(all_eps.min(), all_frob.min()) * 0.7
    xmax = max(all_eps.max(), all_frob.max()) * 1.4

    # --- Achieved-side trend: average N_D per ε-bin + fit, and the C+R line. ---
    # Slopes are quoted per log_10(1/ε) (per DECADE), matching Gustafson's figure
    # convention. (per-log_3 = per-decade × log_10(3) = per-decade × 0.4771.)
    def _log10inv(e):
        return -np.log10(e)
    afr = np.array([p["achieved_frob"] for p in pts if p["achieved_frob"] > 0])
    afr_nd = np.array([p["N_D"] for p in pts if p["achieved_frob"] > 0],
                      dtype=float)
    logs = np.log10(afr)
    edges = np.arange(math.floor(logs.min() * 2) / 2,
                      math.ceil(logs.max() * 2) / 2 + 0.5, 0.5)
    bidx = np.digitize(logs, edges)
    bx, by = [], []
    for b in range(len(edges) + 2):
        sel = bidx == b
        if sel.sum() >= 3:
            bx.append(10 ** logs[sel].mean())
            by.append(afr_nd[sel].mean())
    bx = np.array(bx); by = np.array(by)
    egrid = np.logspace(np.log10(xmin), np.log10(xmax), 200)
    # Combined-data avg-N_D-per-ε-bin fit (spans ALL sources in the CSVs).
    # Suppress with --no-combined-fit (e.g. for a nick-only figure where the
    # nick-only fit below is the wanted C+D summary).
    if not no_combined_fit:
        Xb = _log10inv(bx)
        slope, interc = np.linalg.lstsq(
            np.vstack([Xb, np.ones_like(Xb)]).T, by, rcond=None)[0]
        pred = slope * Xb + interc
        r2 = 1.0 - np.sum((by - pred) ** 2) / np.sum((by - by.mean()) ** 2)
        ax2.scatter(bx, by, marker="D", s=46, facecolors="none",
                    edgecolors="black", linewidths=1.3, zorder=6,
                    label="avg $N_D$ per ε-bin")
        ax2.plot(egrid, slope * _log10inv(egrid) + interc, "k--", lw=1.8,
                 zorder=4,
                 label=(rf"C+D fit: ${slope:.2f}\,\log_{{10}}(1/\varepsilon)"
                        rf"{interc:+.1f}$  (R²={r2:.2f})"))
    # C+R (Gustafson 2503.20203). Use the EMPIRICAL algorithm fits to the SAME
    # diagonal R^Z target (eq:numexh paper_main.tex:626, eq:numhos :631) — NOT
    # the SU(3) information-theoretic lower bound (eq:ftgate_bound :150-169),
    # which an earlier version of this plot wrongly used. Slopes are per-DECADE
    # (per log_10); 10.77 log_10 = 5.139 log_3, 8.621 log_10 = 4.113 log_3.
    l10inv = -np.log10(egrid)
    ax2.plot(egrid, 3.20 + 10.77 * l10inv, color="crimson", ls="-.", lw=1.8,
             zorder=4,
             label=r"C+R Householder fit (Gustafson): $10.77\,\log_{10}(1/\varepsilon)+3.2$")
    ax2.plot(egrid, 2.193 + 8.621 * l10inv, color="darkorange", ls="-.", lw=1.8,
             zorder=4,
             label=r"C+R exhaustive fit (Gustafson): $8.62\,\log_{10}(1/\varepsilon)+2.2$")
    # SU(3) info-theoretic lower bound (covering floor; gate-set agnostic).
    ax2.plot(egrid, 10.27 * l10inv - 2.16, color="gray", ls=":", lw=1.3,
             zorder=3,
             label=r"SU(3) info floor: $10.27\,\log_{10}(1/\varepsilon)-2.16$")

    # Nick-only linear fit (the exact-ring C+D points across all f).
    nick_pts = [p for p in pts if p["method"].startswith("nick")
                and p["achieved_frob"] > 0]
    if nick_pts:
        nx = _log10inv(np.array([p["achieved_frob"] for p in nick_pts]))
        ny = np.array([p["N_D"] for p in nick_pts], dtype=float)
        ns, ni = np.polyfit(nx, ny, 1)
        ax2.plot(egrid, ns * _log10inv(egrid) + ni, color="mediumblue",
                 ls="-", lw=1.8, zorder=5,
                 label=(rf"nick-only fit: ${ns:.2f}\,\log_{{10}}(1/\varepsilon)"
                        rf"{ni:+.1f}$"))

    for ax, xlabel, title in [
        (ax1, r"$\varepsilon$ (target Frobenius)", "N_D vs target ε"),
        (ax2, r"achieved Frobenius distance", "N_D vs achieved frob"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$N_D$ (D-gate count)")
        ax.set_title(title)
        ax.set_xlim(xmax, xmin)  # inverted (descending)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.92)

    fig.suptitle(f"N_D vs ε — colored by method, all sources stacked", y=1.00)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    sys.exit(main())
