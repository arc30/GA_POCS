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

# Call Plotting fns

# plot_frob_gap()
# plot_heatmap_mu_lam()
# plot_fugal_vs_init_heatmap()


final_plot()