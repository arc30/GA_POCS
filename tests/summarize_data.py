"""
1. Establish baseline results.
csvs/baseline_results.csv :
Give a summary table with:
Algo, noise, frobenius - gt_frobenius, comments
for each dataset. (frobenius - gt_frobenius is the frobenius gap) should be the average acoss
all 3 trials. for fugal_init, use the best mu (the one which minimizes frobenius gap). add the best mu  in the comments.


2. Next, we do grid search on the hyperparameters, and do lam_step gradual reduction.
Look at res_cache_8_mar.csv and res_gradual_lamstep.csv and get the best lam_step and mu combination (the one which minimizes frobenius gap).

Summarize the results in a table.
 In the format of Algo, noise, frobenius - gt_frobenius, comments - for each dataset. add best lam_step and mu in comments.

3. Next we change the alpha schedule and see how the results change.
Look at new_alpha.csv and get the best lam_step and mu combination (the one which minimizes frobenius gap). Summarize the results in a table for each dataset. In the format of Algo, noise, frobenius - gt_frobenius, comments - for each dataset. add best lam_step and mu in comments.

# Plotting
Now we have 3 summary tables. Let's plot.

4. For each dataset,
We need 1 plot showing frobenius gap across different noise levels. If values are missing, that should be also clear.

qap_init (this value should be same across all the 3 summary csvs)
fugal_init_baseline  (summary_baseline.csv)
fugal_baseline (summary_baseline.csv)
fugal_init_best_grid_search (summary_grid_search.csv)
fugal_new_alpha (summary_new_alpha.csv)
fugal_init_new_alpha (summary_new_alpha.csv)


"""

import pandas as pd
import os

# Resolve paths relative to project root regardless of where script is invoked from
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)


def best_per_dataset_noise(df, has_lam_step=True):
    """
    For each (dataset, algorithm, noise):
      - Average excess frobenius over trials for each (mu, lam_step) combo
      - Pick the combo with minimum mean excess frobenius
      - Return a summary table: dataset, algorithm, noise, excess_frob, comments
    """
    df = df.copy()
    df["excess"] = df["frobenius"] - df["gt_frobenius"]

    param_cols = ["mu", "lam_step"] if has_lam_step else ["mu"]
    # keep only param_cols that actually exist
    param_cols = [c for c in param_cols if c in df.columns]

    # Fill NaN params with sentinel so qap/qap_init rows aren't silently dropped by groupby
    for col in param_cols:
        df[col] = df[col].fillna("N/A")

    group_cols = ["dataset", "algorithm", "noise"] + param_cols
    mean_over_trials = (
        df.groupby(group_cols)[["accuracy", "excess"]]
        .mean()
        .reset_index()
    )

    # For each (dataset, algorithm, noise), find the row with minimum excess
    idx = mean_over_trials.groupby(["dataset", "algorithm", "noise"])["excess"].idxmin()
    best = mean_over_trials.loc[idx].reset_index(drop=True)

    # Build comments: "best mu=X" or "best mu=X, lam_step=Y"
    def make_comment(row):
        parts = [f"mu={row['mu']}"]
        if "lam_step" in row and pd.notna(row.get("lam_step")):
            parts.append(f"lam_step={row['lam_step']}")
        return ", ".join(parts)

    best["comments"] = best.apply(make_comment, axis=1)

    out = best[["dataset", "algorithm", "noise", "accuracy", "excess", "comments"]].copy()
    out = out.rename(columns={"excess": "frobenius_gap"})
    out = out.sort_values(["dataset", "noise", "algorithm"]).reset_index(drop=True)
    return out


def print_and_save(summary, title, out_path):
    print(f"\n{'#'*70}")
    print(f"# {title}")
    print(f"{'#'*70}")
    for ds in sorted(summary["dataset"].unique()):
        print(f"\n  Dataset: {ds}")
        print(
            summary[summary["dataset"] == ds]
            .drop(columns="dataset")
            .to_string(index=False)
        )
    summary.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\nSaved: {out_path}")


# ---------------------------------------------------------------------------
# Task 1: Baseline (baseline_results.csv — no lam_step column)
# ---------------------------------------------------------------------------
df1 = pd.read_csv("csvs/baseline_results.csv")
df1.columns = df1.columns.str.lower()  # normalise column names

summary1 = best_per_dataset_noise(df1, has_lam_step=False)
print_and_save(summary1, "Task 1: Baseline Results", "csvs/summary_baseline.csv")

# ---------------------------------------------------------------------------
# Task 2: Grid search — res_cache_8_mar.csv + res_gradual_lamstep.csv
# ---------------------------------------------------------------------------
df2 = pd.concat([
    pd.read_csv("csvs/res_cache_8_mar.csv"),
    pd.read_csv("csvs/res_gradual_lamstep.csv"),
], ignore_index=True).drop_duplicates()

summary2 = best_per_dataset_noise(df2, has_lam_step=True)
print_and_save(summary2, "Task 2: Grid Search (mu x lam_step)", "csvs/summary_grid_search.csv")

# ---------------------------------------------------------------------------
# Task 3: New alpha schedule (new_alpha.csv)
# ---------------------------------------------------------------------------
df3 = pd.read_csv("csvs/new_alpha.csv")

summary3 = best_per_dataset_noise(df3, has_lam_step=True)
print_and_save(summary3, "Task 3: New Alpha Schedule", "csvs/summary_new_alpha.csv")

# ---------------------------------------------------------------------------
# Task 4: Plot — frobenius gap vs noise, one figure per dataset
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
