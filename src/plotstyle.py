"""Shared matplotlib style: recessive axes, thin lines, one ordinal blue ramp for qubit count."""

import matplotlib as mpl

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
REFERENCE = "#52514e"
ACCENT = "#eb6834"

# Ordinal blue ramp, light to dark (more qubits = darker).
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


def qubit_colors(ns):
    step = max(1, (len(BLUES) - 1) // max(1, len(ns) - 1)) if len(ns) < len(BLUES) else 1
    return {n: BLUES[min(i * step, len(BLUES) - 1)] for i, n in enumerate(ns)}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })
