"""
Benchmark Results Visualizer
=============================
Reads results/baseline_results.csv and produces 4 plots:
  Plot 1 — Frobenius Norm vs. Noise Level
  Plot 2 — Frobenius Gap Bar Chart (requires gt_frobenius column)
  Plot 4 — Accuracy vs. Noise Level
  Plot 5 — Accuracy vs. Frobenius Gap (requires gt_frobenius column)

Usage:
    python tests/visualize.py
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Ensure paths resolve from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

CSV_PATH = os.path.join("results", "baseline_results.csv")
if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found. Run `python tests/benchmark.py` first.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["dataset", "algorithm"])

has_gt = "gt_frobenius" in df.columns and df["gt_frobenius"].notna().any()
# In short: it collapses multiple trial runs into one mean data point
df_mean = df.groupby(["dataset", "noise", "algorithm"]).mean(numeric_only=True).reset_index()
df_mean_mu = df.groupby(["dataset", "noise", "algorithm", "mu"], dropna=False).mean(numeric_only=True).reset_index()

datasets = datasets = sorted(df["dataset"].unique())
    # ["euroroad", "netscience"]
    # sorted(df["dataset"].unique())
algorithms = (sorted(df["algorithm"].unique()))
    # ["Fugal", "Fugal_init", "QAP_init"]
    # (sorted(df["algorithm"].unique()))

# Consistent color palette
PALETTE = {
    "Fugal":      "#1f77b4",  # blue
    "Fugal_init": "#ff7f0e",  # orange
    "QAP":        "#d62728",  # red
    "QAP_init":   "#2ca02c",  # green
}
# Fallback colors for any algorithm not in the palette
_fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
def alg_color(alg):
    return PALETTE.get(alg, _fallback[algorithms.index(alg) % len(_fallback)])

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Plot 1 — Frobenius Norm vs. Noise Level
# ---------------------------------------------------------------------------

ncols = len(datasets)
fig1, axes1 = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
fig1.suptitle("Frobenius Norm vs. Noise Level", fontsize=14)

for col, ds in enumerate(datasets):
    ax = axes1[0][col]
    sub = df_mean[df_mean["dataset"] == ds]
    sub_mu = df_mean_mu[df_mean_mu["dataset"] == ds]

    for alg in algorithms:
        if alg == "Fugal_init":
            mu_vals = sorted(sub_mu.loc[sub_mu["algorithm"] == alg, "mu"].dropna().unique())
            cmap = plt.cm.Oranges
            colors = [cmap(0.4 + 0.5 * i / max(len(mu_vals) - 1, 1)) for i in range(len(mu_vals))]
            for mu_color, mu_val in zip(colors, mu_vals):
                alg_sub = sub_mu[(sub_mu["algorithm"] == alg) & (sub_mu["mu"] == mu_val)].sort_values("noise")
                ax.plot(alg_sub["noise"], alg_sub["frobenius"],
                        marker="o", label=f"Fugal_init (\u03bc={int(mu_val)})", color=mu_color)
        else:
            alg_sub = sub[sub["algorithm"] == alg].sort_values("noise")
            ax.plot(alg_sub["noise"], alg_sub["frobenius"],
                    marker="o", label=alg, color=alg_color(alg))

    if has_gt:
        gt_sub = sub.groupby("noise")["gt_frobenius"].mean().reset_index().sort_values("noise")
        ax.plot(gt_sub["noise"], gt_sub["gt_frobenius"],
                linestyle="--", color="black", marker="s", label="GT (lower bound)")

    ax.set_title(ds)
    ax.set_xlabel("Noise (%)")
    ax.set_ylabel("Frobenius Norm")
    ax.legend(fontsize=8)

fig1.tight_layout()
out1 = os.path.join("results", "plot1_frob_vs_noise.png")
fig1.savefig(out1, dpi=150)
print(f"Saved: {out1}")

# ---------------------------------------------------------------------------
# Plot 2 — Frobenius Gap Bar Chart
# ---------------------------------------------------------------------------

# if not has_gt:
#     print("WARNING: 'gt_frobenius' column missing or empty — skipping Plot 2 (Frobenius Gap).")
# else:
#     df2 = df.copy()
#     df2["frob_gap"] = df2["frobenius"] - df2["gt_frobenius"]
#     df2_agg = df2.groupby(["dataset", "noise", "algorithm"])["frob_gap"].mean().reset_index()
#
#     noise_levels = sorted(df2_agg["noise"].unique())
#     x = np.arange(len(noise_levels))
#     bar_width = 0.8 / len(algorithms)
#
#     fig2, axes2 = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
#     fig2.suptitle("Frobenius Gap Above GT (lower=better)", fontsize=14)
#
#     for col, ds in enumerate(datasets):
#         ax = axes2[0][col]
#         sub = df2_agg[df2_agg["dataset"] == ds]
#
#         for i, alg in enumerate(algorithms):
#             alg_sub = sub[sub["algorithm"] == alg].sort_values("noise")
#             offsets = x[:len(alg_sub)] + (i - len(algorithms) / 2 + 0.5) * bar_width
#             ax.bar(offsets, alg_sub["frob_gap"].values,
#                    width=bar_width, label=alg, color=alg_color(alg))
#
#         ax.set_xticks(x)
#         ax.set_xticklabels([f"{n*100:.0f}%" for n in noise_levels])
#         ax.set_ylim(bottom=0)
#         ax.set_title(ds)
#         ax.set_xlabel("Noise Level")
#         ax.set_ylabel("Mean Frobenius Gap")
#         ax.legend(fontsize=8)
#
#     fig2.tight_layout()
#     out2 = os.path.join("results", "plot2_frob_gap.png")
#     fig2.savefig(out2, dpi=150)
#     print(f"Saved: {out2}")

# ---------------------------------------------------------------------------
# Plot 4 — Accuracy vs. Noise Level
# ---------------------------------------------------------------------------

fig4, axes4 = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
fig4.suptitle("Node Alignment Accuracy vs. Noise Level", fontsize=14)

for col, ds in enumerate(datasets):
    ax = axes4[0][col]
    sub = df_mean[df_mean["dataset"] == ds]

    for alg in algorithms:
        alg_sub = sub[sub["algorithm"] == alg].sort_values("noise")
        ax.plot(alg_sub["noise"], alg_sub["accuracy"],
                marker="o", label=alg, color=alg_color(alg))

    ax.set_title(ds)
    ax.set_xlabel("Noise (%)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

fig4.tight_layout()
out4 = os.path.join("results", "plot4_acc_vs_noise.png")
fig4.savefig(out4, dpi=150)
print(f"Saved: {out4}")

# ---------------------------------------------------------------------------
# Plot 5 — Accuracy vs. Frobenius Gap (scatter, per trial)
# ---------------------------------------------------------------------------

# if not has_gt:
#     print("WARNING: 'gt_frobenius' column missing or empty — skipping Plot 5 (Accuracy vs. Frob Gap).")
# else:
#     df5 = df.copy()
#     df5["frob_gap"] = df5["frobenius"] - df5["gt_frobenius"]
#
#     fig5, ax5 = plt.subplots(figsize=(8, 6))
#     fig5.suptitle("Accuracy vs. Frobenius Gap (per trial)", fontsize=14)
#
#     for alg in algorithms:
#         alg_sub = df5[df5["algorithm"] == alg]
#         ax5.scatter(alg_sub["frob_gap"], alg_sub["accuracy"],
#                     label=alg, color=alg_color(alg), alpha=0.7, s=40)
#
#     ax5.set_xlabel("Frobenius Gap (frobenius − gt_frobenius)")
#     ax5.set_ylabel("Accuracy")
#     ax5.legend(fontsize=9)
#
#     fig5.tight_layout()
#     out5 = os.path.join("results", "plot5_acc_vs_frob_gap.png")
#     fig5.savefig(out5, dpi=150)
#     print(f"Saved: {out5}")

plt.show()
