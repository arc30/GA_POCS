import pandas as pd
import matplotlib.pyplot as plt
import os

import seaborn as sns

df_mu_lamstep_all = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/res_cache_8_mar.csv" )
df = pd.read_csv("/Users/archana/Desktop/repos/gaProj/GA_POCS/res_gradual_lamstep.csv")
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
        fig.savefig(f"results/optionB_heatmap_{ds}.png", dpi=150)
        plt.show()


# Call Plotting fns

plot_frob_gap()
plot_heatmap_mu_lam()