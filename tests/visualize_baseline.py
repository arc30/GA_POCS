"""
Benchmark Results Visualizer
=============================
Reads results/baseline_results.csv and produces 4 plots:
  Plot 1 — Frobenius Norm vs. Noise Level
  Plot 2 — Frobenius Gap Bar Chart (requires gt_frobenius column)
  Plot 4 — Accuracy vs. Noise Level
  Plot 5 — Accuracy vs. Frobenius Gap (requires gt_frobenius column)

Usage:
    python tests/visualize_baseline.py
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
                        marker="o", label=f"Fugal_init (\u03bc={(mu_val)})", color=mu_color)
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
# Plot 1b — Frobenius Norm vs. Noise Level (Best Fugal_init μ)
# ---------------------------------------------------------------------------

# One best mu per dataset: the mu with the lowest mean frobenius across all noise levels
fugal_init_mu = df_mean_mu[df_mean_mu["algorithm"] == "Fugal_init"]
best_mu_per_ds = (
    fugal_init_mu.groupby(["dataset", "mu"])["frobenius"].mean()
    .reset_index()
    .sort_values("frobenius")
    .drop_duplicates("dataset")
    .set_index("dataset")["mu"]
)

fig1b, axes1b = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
fig1b.suptitle("Frobenius Norm vs. Noise Level (Best Fugal_init μ)", fontsize=14)

for col, ds in enumerate(datasets):
    ax = axes1b[0][col]
    sub = df_mean[df_mean["dataset"] == ds]

    # Fugal line
    fugal_sub = sub[sub["algorithm"] == "Fugal"].sort_values("noise")
    ax.plot(fugal_sub["noise"], fugal_sub["frobenius"],
            marker="o", label="Fugal", color=alg_color("Fugal"))

    # Fugal_init — single best mu for this dataset
    best_mu = best_mu_per_ds.get(ds)
    env_sub = fugal_init_mu[
        (fugal_init_mu["dataset"] == ds) & (fugal_init_mu["mu"] == best_mu)
    ].sort_values("noise")
    ax.plot(env_sub["noise"], env_sub["frobenius"],
            marker="o", label=f"Fugal_init (μ={best_mu})", color=alg_color("Fugal_init"))

    # QAP_init line
    qap_init_sub = sub[sub["algorithm"] == "QAP_init"].sort_values("noise")
    if not qap_init_sub.empty:
        ax.plot(qap_init_sub["noise"], qap_init_sub["frobenius"],
                marker="o", label="QAP_init", color=alg_color("QAP_init"))

    # GT lower bound
    if has_gt:
        gt_sub = sub.groupby("noise")["gt_frobenius"].mean().reset_index().sort_values("noise")
        ax.plot(gt_sub["noise"], gt_sub["gt_frobenius"],
                linestyle="--", color="black", marker="s", label="GT (lower bound)")

    # Tight y-axis zoom
    all_vals = pd.concat([
        fugal_sub["frobenius"],
        env_sub["frobenius"],
        qap_init_sub["frobenius"],
        *([] if not has_gt else [gt_sub["gt_frobenius"]])
    ]).dropna()
    margin = (all_vals.max() - all_vals.min()) * 0.1 or 0.5
    ax.set_ylim(all_vals.min() - margin, all_vals.max() + margin)

    ax.set_title(ds)
    ax.set_xlabel("Noise (%)")
    ax.set_ylabel("Frobenius Norm")
    ax.legend(fontsize=8)

fig1b.tight_layout()
out1b = os.path.join("results", "plot1b_frob_best_mu.png")
fig1b.savefig(out1b, dpi=150)
print(f"Saved: {out1b}")



# ---------------------------------------------------------------------------
# Plot 4 — Accuracy vs. Noise Level
# ---------------------------------------------------------------------------
#
# fig4, axes4 = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
# fig4.suptitle("Node Alignment Accuracy vs. Noise Level", fontsize=14)
#
# for col, ds in enumerate(datasets):
#     ax = axes4[0][col]
#     sub = df_mean[df_mean["dataset"] == ds]
#
#     for alg in algorithms:
#         alg_sub = sub[sub["algorithm"] == alg].sort_values("noise")
#         ax.plot(alg_sub["noise"], alg_sub["accuracy"],
#                 marker="o", label=alg, color=alg_color(alg))
#
#     ax.set_title(ds)
#     ax.set_xlabel("Noise (%)")
#     ax.set_ylabel("Accuracy")
#     ax.set_ylim(0, 1.05)
#     ax.legend(fontsize=8)
#
# fig4.tight_layout()
# out4 = os.path.join("results", "plot4_acc_vs_noise.png")
# fig4.savefig(out4, dpi=150)
# print(f"Saved: {out4}")


plt.show()
