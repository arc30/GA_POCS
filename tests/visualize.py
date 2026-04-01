import pandas as pd
import matplotlib.pyplot as plt
import os

import seaborn as sns

df_mu_lamstep_all = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/csvs/res_cache_8_mar.csv" )
df_gradual_lam_step = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/csvs/res_gradual_lamstep.csv")

# df_lam0_30iters_euro = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/csvs/euroroad_lam0_30iters.csv")
# df_lam0_100 = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/csvs/euroroad_lam0_100iters.csv")

df = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/csvs/new_alpha.csv")


df = df.drop_duplicates()  # handle duplicate rows


# Compute frobenius delta as excess
df["excess"] = df["frobenius"] - df["gt_frobenius"]

# Aggregate qap_init across trials
qap = (
      df[df["algorithm"] == "qap_init"]
      .groupby(["dataset", "noise"])["excess"]
      .mean()
      .reset_index()
      .rename(columns={"excess": "qap_excess"})
  )

# Step 4a: mean across 3 trials for each (dataset, noise, mu, lam_step)
fugal_by_params = (
    df[df["algorithm"] == "fugal_init"]
    .groupby(["dataset", "noise", "mu", "lam_step"])["excess"]
    .mean()
    .reset_index()
)

# Step 4b: for each (dataset, noise), get best/worst/mean across all (mu,lam_step) combos
fugal_agg = fugal_by_params.groupby(["dataset", "noise"])["excess"].agg(
      best="min",
      worst="max",
      mean_all="mean"
  ).reset_index()


#  Plotting
merged = qap.merge(fugal_agg, on=["dataset", "noise"])
datasets = sorted(merged["dataset"].unique())
noise_levels = sorted(merged["noise"].unique())


def plot_frob_gap():
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    fig.suptitle("Mean Excess Frobenius (frob - gt_frob) vs Noise Level", fontsize=13)

    for col, ds in enumerate(datasets):
          ax = axes[0][col]
          sub = merged[merged["dataset"] == ds].sort_values("noise")

          ax.plot(sub["noise"], sub["qap_excess"], marker="o", label="qap_init", color="green")
          ax.plot(sub["noise"], sub["best"], marker="o", label="fugal_init (best config)",
      color="orange")
          ax.fill_between(sub["noise"], sub["best"], sub["worst"],
                          alpha=0.15, color="orange", label="fugal_init range")

          ax.set_title(ds)
          ax.set_xticks(noise_levels)
          ax.set_xticklabels(noise_levels)
          ax.set_xlabel("Noise (%)")
          ax.set_ylabel("Mean Excess Frobenius")
          # ax.axhline(0, linestyle="--", color="black", linewidth=0.8, label="GT baseline (0)")
          ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/optionA_excess_frob.png", dpi=150)
    plt.show()

    print("done frob diff plot")

def plot_heatmap_mu_lam():
    fugal_heatmap_per_noise = (fugal_by_params.copy())
    qap_ref_per_noise = (
        df[df["algorithm"] == "qap_init"]
        .groupby(["dataset", "noise"])["excess"]
        .mean()
    )

    noise_levels = sorted(fugal_heatmap_per_noise["noise"].unique())

    datasets = sorted(fugal_agg["dataset"].unique())
    for ds in datasets:
        fig, axes = plt.subplots(1, len(noise_levels),
                                 figsize=(4 * len(noise_levels), 5),
                                 squeeze=False)
        fig.suptitle(f"{ds} — Fugal_init Excess Frobenius per Noise Level",
                     fontsize=13)

        ds_data = fugal_heatmap_per_noise[fugal_heatmap_per_noise["dataset"] == ds]

        # Shared color scale across all noise levels for this dataset
        vmin = ds_data["excess"].min()
        vmax = ds_data["excess"].max()

        for col, noise in enumerate(noise_levels):
            ax = axes[0][col]
            sub = ds_data[ds_data["noise"] == noise]
            pivot = sub.pivot(index="mu", columns="lam_step", values="excess")

            qap_val = qap_ref_per_noise.get((ds, noise), None)

            sns.heatmap(
                pivot,
                ax=ax,
                annot=True, fmt=".1f",
                cmap="RdYlGn_r",
                vmin=vmin, vmax=vmax,  # shared scale, across noise comparison
            linewidths = 0.5,
            cbar = (col == len(noise_levels) - 1),  # only one colorbar


            cbar_kws = {"label": "mean excess frob"}
            )

            title = f"noise={noise}%"
            if qap_val is not None:
                title += f"\nqap_init={qap_val:.1f}"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("lam_step")
            ax.set_ylabel("mu" if col == 0 else "")

        fig.tight_layout()
        fig.savefig(f"results/optionB_heatmap_{ds}_lam0_qap1.png", dpi=150)
        plt.show()


def plot_fugal_vs_init_heatmap():
    df_new = pd.read_csv("/csvs/new_alpha.csv")

    df_new["excess"] = df_new["frobenius"] - df_new["gt_frobenius"]

    # qap_init: mean over trials per (dataset, noise) — no mu/lam_step dependence
    qap_agg = (
        df_new[df_new["algorithm"] == "qap_init"]
        .groupby(["dataset", "noise"])[["accuracy", "excess"]]
        .mean()
    )

    # fugal / fugal_init: mean over trials per (dataset, algorithm, noise, mu, lam_step)
    agg = (
        df_new[df_new["algorithm"] != "qap_init"]
        .groupby(["dataset", "algorithm", "noise", "mu", "lam_step"])[["accuracy", "excess"]]
        .mean()
        .reset_index()
    )

    datasets = sorted(agg["dataset"].unique())
    noise_levels = sorted(agg["noise"].unique())
    algorithms = ["fugal", "fugal_init"]
    metrics = [("accuracy", "Mean Accuracy", "YlGn"),
               ("excess",   "Mean Excess Frobenius", "RdYlGn_r")]

    os.makedirs("results", exist_ok=True)

    for ds in datasets:
        ds_data = agg[agg["dataset"] == ds]

        for noise in noise_levels:
            noise_data = ds_data[ds_data["noise"] == noise]

            # qap_init reference values for this (dataset, noise)
            qap_acc = qap_agg.loc[(ds, noise), "accuracy"] if (ds, noise) in qap_agg.index else None
            qap_exc = qap_agg.loc[(ds, noise), "excess"]   if (ds, noise) in qap_agg.index else None

            fig, axes = plt.subplots(
                len(metrics), len(algorithms),
                figsize=(6 * len(algorithms), 5 * len(metrics)),
                squeeze=False,
            )
            qap_str = ""
            if qap_acc is not None:
                qap_str = f"  |  qap_init: acc={qap_acc:.3f}, excess frob={qap_exc:.1f}"
            fig.suptitle(f"{ds} — noise={noise}%{qap_str}", fontsize=12)

            for row, (metric, metric_label, cmap) in enumerate(metrics):
                qap_val = qap_acc if metric == "accuracy" else qap_exc
                vmin = min(noise_data[metric].min(), qap_val) if qap_val is not None else noise_data[metric].min()
                vmax = max(noise_data[metric].max(), qap_val) if qap_val is not None else noise_data[metric].max()

                for col, algo in enumerate(algorithms):
                    ax = axes[row][col]
                    sub = noise_data[noise_data["algorithm"] == algo]
                    pivot = sub.pivot(index="mu", columns="lam_step", values=metric)

                    sns.heatmap(
                        pivot,
                        ax=ax,
                        annot=True,
                        fmt=".3f",
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        linewidths=0.5,
                        cbar=(col == len(algorithms) - 1),
                        cbar_kws={"label": metric_label},
                    )
                    qap_label = f"{algo}\n(qap_init ref: {qap_val:.3f})" if qap_val is not None else algo
                    ax.set_title(qap_label, fontsize=10)
                    ax.set_xlabel("lam_step")
                    ax.set_ylabel("mu" if col == 0 else "")

            fig.tight_layout()
            fig.savefig(f"results/fugal_vs_init_heatmap_{ds}_noise{noise}.png", dpi=150)
            plt.show()
            print(f"Saved: results/fugal_vs_init_heatmap_{ds}_noise{noise}.png")



def final_plot():
    summary1 = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/csvs/summary_baseline.csv" )
    summary2 = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/csvs/summary_grid_search.csv")
    summary3 = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/csvs/summary_new_alpha.csv")
    # Build a unified dataframe with a "series" label per line
    def extract_series(summary, algo_filter, label):
        sub = summary[summary["algorithm"].str.lower() == algo_filter][["dataset", "noise", "frobenius_gap"]].copy()
        sub["series"] = label
        return sub

    # qap_init: take from summary1 (same values expected across all 3)
    series_list = [
        extract_series(summary1, "qap_init",   "qap_init"),
        extract_series(summary1, "fugal_init", "fugal_init_baseline"),
        extract_series(summary1, "fugal",      "fugal_baseline"),
        extract_series(summary2, "fugal_init", "fugal_init_grid_search"),
        extract_series(summary3, "fugal_init", "fugal_init_new_alpha"),
        extract_series(summary3, "fugal", "fugal_new_alpha"),

    ]
    plot_df = pd.concat(series_list, ignore_index=True)

    series_order = [
        "qap_init",
        "fugal_baseline",
        "fugal_init_baseline",
        "fugal_init_grid_search",
        "fugal_init_new_alpha",
        "fugal_new_alpha",
    ]
    colors = {
        "qap_init":                "tab:gray",
        "fugal_baseline":          "tab:blue",
        "fugal_init_baseline":     "tab:orange",
        "fugal_init_grid_search":  "tab:green",
        "fugal_init_new_alpha":    "tab:red",
        "fugal_new_alpha":         "tab:purple",
    }

    noise_levels = sorted(plot_df["noise"].unique())
    datasets = sorted(plot_df["dataset"].unique())
    os.makedirs("results", exist_ok=True)

    for ds in datasets:
        fig, ax = plt.subplots(figsize=(8, 5))
        ds_data = plot_df[plot_df["dataset"] == ds]

        for series in series_order:
            sub = ds_data[ds_data["series"] == series].sort_values("noise")
            ax.plot(
                sub["noise"], sub["frobenius_gap"],
                marker="o", label=series, color=colors[series],
            )

        ax.set_title(f"{ds} — Frobenius Gap vs Noise")
        ax.set_xlabel("Noise (%)")
        ax.set_ylabel("Frobenius Gap (frob - gt_frob)")
        ax.set_xticks(noise_levels)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"results/frob_gap_{ds}.png", dpi=150)
        plt.show()
        print(f"Saved: results/frob_gap_{ds}.png")

def paired_plot():
    """Paired accuracy + frobenius gap line plots per dataset (NeurIPS style)."""
    summary1 = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/csvs/summary_baseline.csv")
    summary2 = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/csvs/summary_grid_search.csv")
    summary3 = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/csvs/summary_new_alpha.csv")

    def extract_series(summary, algo_filter, label):
        sub = summary[summary["algorithm"].str.lower() == algo_filter][
            ["dataset", "noise", "accuracy", "frobenius_gap"]
        ].copy()
        sub["series"] = label
        return sub

    series_list = [
        extract_series(summary1, "qap_init",   "qap_init"),
        extract_series(summary1, "fugal_init", "fugal_init_baseline"),
        extract_series(summary1, "fugal",      "fugal_baseline"),
        extract_series(summary2, "fugal_init", "fugal_init_grid_search"),
        extract_series(summary3, "fugal_init", "fugal_init_new_alpha"),
        extract_series(summary3, "fugal",      "fugal_new_alpha"),
    ]
    plot_df = pd.concat(series_list, ignore_index=True)

    series_order = [
        "qap_init",
        "fugal_baseline",
        "fugal_init_baseline",
        "fugal_init_grid_search",
        "fugal_init_new_alpha",
        "fugal_new_alpha",
    ]
    display_names = {
        "qap_init":                "FAQ_init",
        "fugal_baseline":          "FUGAL",
        "fugal_init_baseline":     "FUGAL_init",
        "fugal_init_grid_search":  "FUGAL_init (tuned)",
        "fugal_init_new_alpha":    "FUGAL*_init",
        "fugal_new_alpha":         "FUGAL*",
    }
    colors = {
        "qap_init":                "tab:gray",
        "fugal_baseline":          "tab:blue",
        "fugal_init_baseline":     "tab:orange",
        "fugal_init_grid_search":  "tab:green",
        "fugal_init_new_alpha":    "tab:red",
        "fugal_new_alpha":         "tab:purple",
    }
    markers = {
        "qap_init":                "s",
        "fugal_baseline":          "o",
        "fugal_init_baseline":     "^",
        "fugal_init_grid_search":  "D",
        "fugal_init_new_alpha":    "v",
        "fugal_new_alpha":         "P",
    }

    noise_levels = sorted(plot_df["noise"].unique())
    datasets = sorted(plot_df["dataset"].unique())
    os.makedirs("results", exist_ok=True)

    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    for ds in datasets:
        fig, (ax_acc, ax_gap) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        ds_data = plot_df[plot_df["dataset"] == ds]

        for series in series_order:
            sub = ds_data[ds_data["series"] == series].sort_values("noise")
            if sub.empty:
                continue
            label = display_names[series]
            ax_acc.plot(sub["noise"], sub["accuracy"],
                        marker=markers[series], label=label, color=colors[series],
                        linewidth=1.5, markersize=6)
            ax_gap.plot(sub["noise"], sub["frobenius_gap"],
                        marker=markers[series], label=label, color=colors[series],
                        linewidth=1.5, markersize=6)

        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title(ds, fontsize=13)
        ax_acc.set_ylim(bottom=0)
        ax_acc.grid(True, alpha=0.3)

        ax_gap.set_ylabel("Excess Frobenius Norm")
        ax_gap.set_xlabel("Noise (%)")
        ax_gap.set_xticks(noise_levels)
        ax_gap.set_xticklabels([str(int(n)) for n in noise_levels])
        ax_gap.set_ylim(bottom=0)
        ax_gap.grid(True, alpha=0.3)

        # Single shared legend between the two axes
        handles, labels = ax_acc.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9,
                   bbox_to_anchor=(0.5, 1.02), frameon=False)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(f"results/paired_{ds}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: results/paired_{ds}.png")


# Call Plotting fns

# plot_frob_gap()
# plot_heatmap_mu_lam()
# plot_fugal_vs_init_heatmap()
# final_plot()

# paired_plot()

"""
Runtime comparison plot.
data source: summary_new_alpha.csv
algos: fugal_init, fugal, qap_init

fugal's runtime is present in time_seconds column.
fugal_init's and qap_init's runtime is the sum of time_seconds column plus init_time column. If time_seconds in absent, let's ignore that entry in the comparison.

fugal_init and fugal use new alpha, so they should be renamed to "FUGAL*_init", "FUGAL*" in the plots. qap_init should be renamed to "FAQ_init".

1. We need to present the runtimes in a table. Per dataset, per noise level, we need to show the mean runtime of all algorithms.
2. We need a bar plot of the runtimes, fugal* vs FAQ_init. where in faq_init, we need to see the split of init_time and time_seconds within the bar plot.
"""


def runtime_comparison():
    summary = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/csvs/summary_new_alpha.csv")

    df_rt = summary.copy()

    # ── Compute total runtime per row, drop rows where time_seconds is absent ──
    # fugal: runtime = time_seconds only (no init_time column)
    # fugal_init / qap_init: runtime = time_seconds + init_time; skip if time_seconds NaN
    df_rt = df_rt[df_rt["time_seconds"].notna()].copy()

    df_rt["algo_time"] = df_rt["time_seconds"].fillna(0)
    df_rt["init_part"] = df_rt["init_time"].fillna(0)
    df_rt["total_time"] = df_rt["algo_time"] + df_rt["init_part"]

    display_names = {"fugal": "FUGAL*", "fugal_init": "FUGAL*_init", "qap_init": "FAQ_init"}
    df_rt["label"] = df_rt["algorithm"].map(display_names)

    # ── 1. Runtime table: per (dataset, noise, algorithm) ──
    table = (
        df_rt.pivot_table(index=["dataset", "noise"], columns="label",
                          values="total_time", aggfunc="first")
        .reindex(columns=["FUGAL*", "FUGAL*_init", "FAQ_init"])
        .round(1)
    )
    print("\n=== Runtime (seconds) by Dataset and Noise Level ===")
    print(table.to_string())
    print()

    # ── 2. Bar plot: one subplot per dataset, x-axis = noise levels ──
    datasets = sorted(df_rt["dataset"].unique())
    n_ds = len(datasets)

    algo_order = ["FUGAL*", "FAQ_init"]
    colors = {
        "FUGAL*":   {"algo": "tab:purple", "init": None},
        "FAQ_init": {"algo": "tab:green",  "init": "lightgreen"},
    }

    bar_width = 0.3
    offsets = {"FUGAL*": -bar_width / 2, "FAQ_init": bar_width / 2}

    legend_handles = {}

    ds_display = {"multimanga": "multimagna"}

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 5), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df_rt[df_rt["dataset"] == ds]

        # Only keep noise levels where both FUGAL* and FAQ_init are present
        valid_noises = []
        for noise in sorted(sub["noise"].unique()):
            noise_sub = sub[sub["noise"] == noise]
            has_fugal = not noise_sub[noise_sub["label"] == "FUGAL*"].empty
            has_faq   = not noise_sub[noise_sub["label"] == "FAQ_init"].empty
            if has_fugal and has_faq:
                valid_noises.append(noise)

        for ni, noise in enumerate(valid_noises):
            noise_sub = sub[sub["noise"] == noise]

            for algo in algo_order:
                row = noise_sub[noise_sub["label"] == algo]
                if row.empty:
                    continue
                algo_t = row["algo_time"].values[0]
                init_t = row["init_part"].values[0]
                x = ni + offsets[algo]

                if algo == "FAQ_init":
                    plot_algo_t = max(algo_t, 1)  # show at least 1s so the bar is visible
                    b1 = ax.bar(x, init_t, width=bar_width, color=colors[algo]["init"])
                    b2 = ax.bar(x, plot_algo_t, width=bar_width, bottom=init_t, color=colors[algo]["algo"])
                    if "FAQ_init (init_time)" not in legend_handles:
                        legend_handles["FAQ_init (init_time)"] = b1
                        legend_handles["FAQ_init (algo_time)"] = b2
                else:
                    b = ax.bar(x, algo_t, width=bar_width, color=colors[algo]["algo"])
                    if algo not in legend_handles:
                        legend_handles[algo] = b

        ax.set_title(ds_display.get(ds, ds), fontsize=14)
        ax.set_xticks(range(len(valid_noises)))
        ax.set_xticklabels([f"noise={n}%" for n in valid_noises], rotation=20, ha="right", fontsize=12)
        ax.set_ylabel("Runtime (s)", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.3)

    fig.legend(legend_handles.values(), legend_handles.keys(),
               loc="upper center", ncol=3, fontsize=20,
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle("Runtime Comparison per Noise Level", fontsize=14, y=1.08)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/runtime_comparison2.png", dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved: results/runtime_comparison.png")


runtime_comparison()