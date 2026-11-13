"""Stage 1: at what depth does a random circuit's output look Porter-Thomas?

For each qubit count n and depth d, simulate many random brickwork circuits
exactly (statevector) and compare their output distributions with Haar-random
states of the same dimension.

A circuit "passes" at depth d if its KS distance to Exp(1) is below the 95th
percentile of KS distances for genuine Haar states. Once circuits are
indistinguishable from Haar states, about 95% should pass.

Outputs:
    results/porter_thomas_circuits.csv   one row per circuit
    results/porter_thomas_summary.csv    one row per (n, depth)
    results/porter_thomas.json           Haar thresholds and convergence depths
    figures/pt_histograms.png
    figures/pt_convergence.png
    figures/pt_depth_scaling.png

Usage: python run_porter_thomas.py [--circuits 100] [--max-depth 40]
"""

import argparse
import csv
import json
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from qiskit.quantum_info import Statevector

from src.circuits import brickwork_circuit
from src.stats import (
    HOP_PORTER_THOMAS,
    collision_ratio,
    haar_collision_mean,
    haar_null,
    heavy_output_probability,
    ks_to_exponential,
)
from src import plotstyle as ps

QUBITS = [4, 6, 8, 10, 12, 14]
PASS_LEVEL = 0.9  # KS convergence: first depth where at least 90% of circuits pass
COLLISION_TOL = 0.05  # collision convergence: mean within 5% of the Haar value
HIST_N = 12
HIST_DEPTHS = [0, 2, 4, 8, 16]


def simulate(n_circuits, max_depth, seed):
    rng = np.random.default_rng(seed)
    rows, hist_probs = [], {d: [] for d in HIST_DEPTHS}
    for n in QUBITS:
        t0 = time.time()
        for depth in range(max_depth + 1):
            for c in range(n_circuits):
                probs = Statevector(brickwork_circuit(n, depth, rng)).probabilities()
                rows.append({
                    "n_qubits": n, "depth": depth, "circuit": c,
                    "ks": ks_to_exponential(probs),
                    "collision_ratio": collision_ratio(probs),
                    "heavy_output_prob": heavy_output_probability(probs),
                })
                if n == HIST_N and depth in hist_probs and c < 20:  # kept for the histogram figure
                    hist_probs[depth].append(2**n * probs)
        print(f"n={n}: {time.time() - t0:.0f} s")
    return rows, {d: np.concatenate(v) for d, v in hist_probs.items() if v}


def summarise(rows, thresholds, max_depth):
    summary, convergence, collision_conv = [], {}, {}
    for n in QUBITS:
        for depth in range(max_depth + 1):
            sel = [r for r in rows if r["n_qubits"] == n and r["depth"] == depth]
            ks = np.array([r["ks"] for r in sel])
            coll = np.array([r["collision_ratio"] for r in sel])
            hop = np.array([r["heavy_output_prob"] for r in sel])
            pass_rate = float(np.mean(ks <= thresholds[n]))
            summary.append({
                "n_qubits": n, "depth": depth, "circuits": len(sel),
                "pass_rate": pass_rate,
                "collision_mean": float(coll.mean()),
                "collision_se": float(coll.std(ddof=1) / np.sqrt(len(coll))),
                "hop_mean": float(hop.mean()),
                "hop_se": float(hop.std(ddof=1) / np.sqrt(len(hop))),
            })
            if n not in convergence and pass_rate >= PASS_LEVEL:
                convergence[n] = depth
            if n not in collision_conv and coll.mean() <= (1 + COLLISION_TOL) * haar_collision_mean(2**n):
                collision_conv[n] = depth
    return summary, convergence, collision_conv


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_histograms(hist_probs, path):
    depths = sorted(hist_probs)
    fig, axes = plt.subplots(1, len(depths), figsize=(13, 3.1), sharey=True, squeeze=False)
    axes = axes[0]
    bins = np.linspace(0, 10, 41)
    x = np.linspace(0, 10, 200)
    for ax, depth in zip(axes, depths):
        dens, edges = np.histogram(hist_probs[depth], bins=bins, density=True)
        ax.bar(edges[:-1], dens, width=np.diff(edges) * 0.85, align="edge",
               color=ps.BLUES[2], linewidth=0)
        ax.plot(x, np.exp(-x), color=ps.REFERENCE, lw=1.5, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 10)
        ax.set_xlim(0, 10)
        ax.set_title(f"depth {depth}", fontweight="normal", color=ps.INK_2)
        ax.set_xlabel("D · p(x)")
    axes[0].set_ylabel("density")
    axes[0].text(4.2, np.exp(-4.2) * 3.5, "Porter-Thomas\nExp(1)", color=ps.INK_2, fontsize=9)
    fig.suptitle(f"Output probabilities of random circuits, {HIST_N} qubits (20 circuits per depth)",
                 x=0.01, ha="left", fontweight="bold", color=ps.INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(summary, convergence, path):
    colors = ps.qubit_colors(QUBITS)
    max_d = max(s["depth"] for s in summary)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = [
        ("pass_rate", None, "Circuits indistinguishable from Haar", "fraction passing KS test"),
        ("collision_mean", "collision_se", "Collision ratio  D·Σp²", "mean over circuits (log scale)"),
        ("hop_mean", "hop_se", "Heavy-output probability", "mean over circuits"),
    ]
    for ax, (key, se_key, title, ylabel) in zip(axes, panels):
        for n in QUBITS:
            sel = [s for s in summary if s["n_qubits"] == n]
            d = np.array([s["depth"] for s in sel])
            y = np.array([s[key] for s in sel])
            ax.plot(d, y, color=colors[n])
            if se_key:
                se = np.array([s[se_key] for s in sel])
                ax.fill_between(d, y - 2 * se, y + 2 * se, color=colors[n], alpha=0.18, lw=0)
            if key == "pass_rate":
                ax.lines[-1].set_label(f"{n} qubits")
        ax.set_title(title)
        ax.set_xlabel("circuit depth (CZ layers)")
        ax.set_ylabel(ylabel)
    axes[0].axhline(0.95, color=ps.REFERENCE, ls="--", lw=1.2)
    axes[0].text(max_d * 0.78, 0.06, "dashed: 95%\nexpected for\nHaar states", color=ps.INK_2, fontsize=8.5)
    axes[0].set_ylim(-0.02, 1.05)
    axes[1].set_yscale("log")
    axes[1].set_yticks([2, 3, 5, 10, 20, 50])
    axes[1].yaxis.set_major_formatter(mticker.ScalarFormatter())
    axes[1].yaxis.set_minor_formatter(mticker.NullFormatter())
    axes[1].axhline(2.0, color=ps.REFERENCE, ls="--", lw=1.2)
    axes[1].text(max_d * 0.62, 2.35, "dashed: Haar value 2", color=ps.INK_2, fontsize=8.5)
    axes[2].axhline(HOP_PORTER_THOMAS, color=ps.REFERENCE, ls="--", lw=1.2)
    axes[2].set_ylim(0.80, 1.0)
    axes[2].text(max_d * 0.3, 0.808, "Porter-Thomas: (1 + ln 2)/2 = 0.847 (dashed)",
                 color=ps.INK_2, fontsize=8.5)
    for ax in axes:
        ax.set_xlim(0, max_d)
    fig.legend(loc="upper center", ncol=len(QUBITS), bbox_to_anchor=(0.5, 1.06),
               fontsize=9, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_depth_scaling(ks_conv, coll_conv, max_depth, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    series = [
        (ks_conv, ps.BLUES[3], "Full distribution: 90% of circuits pass KS test"),
        (coll_conv, ps.ACCENT, "Collision ratio within 5% of Haar"),
    ]
    for conv, color, label in series:
        ns = sorted(conv)
        ax.plot(ns, [conv[n] for n in ns], color=color, marker="o", ms=7,
                mec=ps.SURFACE, mew=1.5, label=label)
    missing = [n for n in QUBITS if n not in ks_conv]
    for n in missing:
        ax.annotate(f"> {max_depth}", (n, max_depth), ha="center", va="bottom",
                    fontsize=8.5, color=ps.INK_2, xytext=(0, 4), textcoords="offset points")
        ax.plot([n], [max_depth], marker="^", ms=7, color=ps.BLUES[3], mec=ps.SURFACE, mew=1.5)
    ax.set_xticks(QUBITS)
    ax.set_xlabel("number of qubits n")
    ax.set_ylabel("circuit depth needed (CZ layers)")
    ax.set_title("How deep must a random circuit be to look random?")
    ax.set_ylim(0, max_depth + 6)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def replot():
    with open("results/porter_thomas_summary.csv") as f:
        summary = [{k: (float(v) if "." in v else int(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    with open("results/porter_thomas.json") as f:
        saved = json.load(f)
    ks_conv = {int(k): v for k, v in saved["convergence_depth_ks"].items()}
    coll_conv = {int(k): v for k, v in saved["convergence_depth_collision"].items()}
    plot_convergence(summary, ks_conv, "figures/pt_convergence.png")
    plot_depth_scaling(ks_conv, coll_conv, saved["settings"]["max_depth"], "figures/pt_depth_scaling.png")
    try:
        hist = np.load("results/pt_histogram_samples.npz")
        plot_histograms({int(k): hist[k] for k in hist.files}, "figures/pt_histograms.png")
    except FileNotFoundError:
        print("no histogram samples saved; skipping pt_histograms.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=40)
    ap.add_argument("--haar-states", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--plot-only", action="store_true", help="redraw figures from saved results")
    args = ap.parse_args()
    ps.apply()

    if args.plot_only:
        replot()
        return

    thresholds, null_stats = {}, {}
    for n in QUBITS:
        null = haar_null(n, args.haar_states, seed=args.seed + n)
        thresholds[n] = float(np.quantile(null[:, 0], 0.95))
        null_stats[n] = {
            "ks_95th_percentile": thresholds[n],
            "collision_mean": float(null[:, 1].mean()),
            "collision_theory": haar_collision_mean(2**n),
            "hop_mean": float(null[:, 2].mean()),
        }

    rows, hist_probs = simulate(args.circuits, args.max_depth, args.seed)
    summary, convergence, collision_conv = summarise(rows, thresholds, args.max_depth)

    np.savez_compressed("results/pt_histogram_samples.npz", **{str(d): v for d, v in hist_probs.items()})
    write_csv("results/porter_thomas_circuits.csv", rows)
    write_csv("results/porter_thomas_summary.csv", summary)
    slope, intercept = np.polyfit(list(convergence), list(convergence.values()), 1)
    with open("results/porter_thomas.json", "w") as f:
        json.dump({
            "settings": vars(args) | {"qubits": QUBITS, "pass_level": PASS_LEVEL},
            "haar_null": null_stats,
            "convergence_depth_ks": convergence,
            "convergence_fit_ks": {"slope_per_qubit": slope, "intercept": intercept},
            "convergence_depth_collision": collision_conv,
        }, f, indent=2)

    plot_histograms(hist_probs, "figures/pt_histograms.png")
    plot_convergence(summary, convergence, "figures/pt_convergence.png")
    plot_depth_scaling(convergence, collision_conv, args.max_depth, "figures/pt_depth_scaling.png")

    print("\nConvergence depth (first depth where >= 90% of circuits pass):")
    for n, d in convergence.items():
        print(f"  n={n:2d}: depth {d}")
    print(f"  linear fit: depth = {slope:.2f} * n + {intercept:.2f}")
    print("Depth where mean collision ratio is within 5% of Haar:")
    for n, d in collision_conv.items():
        print(f"  n={n:2d}: depth {d}")


if __name__ == "__main__":
    main()
