"""
Baseline Benchmark Test Harness
================================
Reproducible benchmark for all 4 graph alignment variants:
Fugal, Fugal_init, QAP, QAP_init.

Usage:
    python benchmark.py                 # Full pipeline (generate + run)
    python benchmark.py --generate-only # Phase 1: generate & save graph pairs
    python benchmark.py --run-only      # Phase 2: run algorithms on saved graphs
"""

import os
import sys
import time
import argparse
import csv

import numpy as np
import networkx as nx
import scipy.optimize
import torch

# Ensure relative paths (datasets/) resolve correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from GA_Archana import Fugal, Fugal_init, QAP, QAP_init
from noise import generate_graphs, edges_to_adj, eval_align, read_real_graph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS = {
    "netscience": {"n": 379, "path": "datasets/netscience.txt"},
    # "highschool": {"n": 327, "path": "datasets/highschool.txt"},
}

NOISE_LEVELS = [0.0, 0.10, 0.20]
TRIAL_SEEDS = [42, 123, 7]

ALGORITHMS = {
    "Fugal":      lambda Src, Tar: Fugal(Src, Tar, iter=10, simple=True, mu=1, EFN=5),
    "Fugal_init": lambda Src, Tar: Fugal_init(Src, Tar, iter=10, simple=True, mu=1, EFN=5),
    "QAP":        lambda Src, Tar: QAP(Src, Tar),
    "QAP_init":   lambda Src, Tar: QAP_init(Src, Tar),
}

DATA_DIR = "baseline_data"
RESULTS_CSV = "baseline_results.csv"
RESULTS_SUMMARY = "baseline_results_summary.txt"

# ---------------------------------------------------------------------------
# Phase 1: Generate & save graph pairs
# ---------------------------------------------------------------------------

def extract_edges(adj):
    """Extract upper-triangular edge list from adjacency matrix."""
    return np.array(np.transpose(np.nonzero(np.triu(adj, 1))))


def generate_and_save():
    os.makedirs(DATA_DIR, exist_ok=True)

    for ds_name, ds_info in DATASETS.items():
        G = read_real_graph(n=ds_info["n"], name_=ds_info["path"])
        A = nx.to_numpy_array(G, dtype=int)
        edges = extract_edges(A)
        n = ds_info["n"]

        for noise in NOISE_LEVELS:
            noise_pct = int(noise * 100)
            for trial_idx, seed in enumerate(TRIAL_SEEDS):
                fname = f"{ds_name}_noise{noise_pct}_trial{trial_idx}.npz"
                fpath = os.path.join(DATA_DIR, fname)

                if os.path.exists(fpath):
                    print(f"  [skip] {fname} already exists")
                    continue

                np.random.seed(seed)
                Src_e, Tar_e, GT = generate_graphs(edges.copy(), 0, noise)
                Src_adj = edges_to_adj(Src_e, n)
                Tar_adj = edges_to_adj(Tar_e, n)

                np.savez(fpath, Src_adj=Src_adj, Tar_adj=Tar_adj,
                         GT0=GT[0], GT1=GT[1],
                         seed=seed, noise=noise)
                print(f"  [saved] {fname}")

    saved = [f for f in os.listdir(DATA_DIR) if f.endswith(".npz")]
    print(f"\nGeneration complete: {len(saved)} files in {DATA_DIR}/")


# ---------------------------------------------------------------------------
# Phase 2: Run experiments
# ---------------------------------------------------------------------------

def evaluate(P, GT):
    """Run Hungarian + 4-orientation eval, return best accuracy."""
    if isinstance(P, torch.Tensor):
        P = P.detach().numpy()
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(P, maximize=True)
    GT0, GT1 = GT
    return max(
        eval_align(row_ind, col_ind, GT0),
        eval_align(row_ind, col_ind, GT1),
        eval_align(col_ind, row_ind, GT0),
        eval_align(col_ind, row_ind, GT1),
    )


def run_experiments():
    results = []

    for ds_name in DATASETS:
        for noise in NOISE_LEVELS:
            noise_pct = int(noise * 100)
            for trial_idx in range(len(TRIAL_SEEDS)):
                fname = f"{ds_name}_noise{noise_pct}_trial{trial_idx}.npz"
                fpath = os.path.join(DATA_DIR, fname)

                if not os.path.exists(fpath):
                    print(f"  [missing] {fname} — run --generate-only first")
                    continue

                data = np.load(fpath)
                Src_adj = data["Src_adj"]
                Tar_adj = data["Tar_adj"]
                GT = (data["GT0"], data["GT1"])

                for algo_name, algo_fn in ALGORITHMS.items():
                    # Copy arrays — algorithms mutate inputs (add self-loops)
                    S = Src_adj.copy()
                    T = Tar_adj.copy()

                    print(f"  Running {algo_name} on {ds_name} "
                          f"noise={noise_pct}% trial={trial_idx} ...",
                          end="", flush=True)

                    t0 = time.time()
                    P = algo_fn(S, T)
                    elapsed = time.time() - t0

                    acc = evaluate(P, GT)
                    print(f"  acc={acc:.4f}  time={elapsed:.1f}s")

                    results.append({
                        "dataset": ds_name,
                        "noise": noise_pct,
                        "trial": trial_idx,
                        "algorithm": algo_name,
                        "accuracy": round(acc, 6),
                        "time_sec": round(elapsed, 2),
                    })

    return results


# ---------------------------------------------------------------------------
# Phase 3: Output
# ---------------------------------------------------------------------------

def save_csv(results):
    fieldnames = ["dataset", "noise", "trial", "algorithm", "accuracy", "time_sec"]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-trial results saved to {RESULTS_CSV} ({len(results)} rows)")


def save_summary(results):
    # Group by (dataset, noise, algorithm)
    groups = {}
    for r in results:
        key = (r["dataset"], r["noise"], r["algorithm"])
        groups.setdefault(key, []).append(r)

    lines = []
    header = f"{'Dataset':<14} {'Noise':>5} {'Algorithm':<14} {'Acc Mean':>9} {'Acc Std':>9} {'Time Mean':>10}"
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)

    for key in sorted(groups.keys()):
        ds, noise, algo = key
        accs = [r["accuracy"] for r in groups[key]]
        times = [r["time_sec"] for r in groups[key]]
        lines.append(
            f"{ds:<14} {noise:>4}% {algo:<14} {np.mean(accs):>9.4f} {np.std(accs):>9.4f} {np.mean(times):>9.1f}s"
        )

    summary = "\n".join(lines)
    print(f"\n{summary}")

    with open(RESULTS_SUMMARY, "w") as f:
        f.write(summary + "\n")
    print(f"\nSummary saved to {RESULTS_SUMMARY}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Baseline benchmark harness")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate and save graph pairs")
    parser.add_argument("--run-only", action="store_true",
                        help="Only run algorithms on saved graph pairs")
    args = parser.parse_args()

    if args.generate_only and args.run_only:
        print("Error: --generate-only and --run-only are mutually exclusive")
        sys.exit(1)

    do_generate = not args.run_only
    do_run = not args.generate_only

    if do_generate:
        print("=" * 60)
        print("Phase 1: Generating graph pairs")
        print("=" * 60)
        generate_and_save()

    if do_run:
        print("\n" + "=" * 60)
        print("Phase 2: Running experiments")
        print("=" * 60)
        results = run_experiments()
        if results:
            save_csv(results)
            save_summary(results)
        else:
            print("No results collected. Run --generate-only first.")


if __name__ == "__main__":
    main()
