import argparse
import csv
import os
import sys
from datetime import datetime
from typing import Dict, Any

import networkx as nx
import numpy as np
import scipy
import torch

dry_run = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Ensure relative paths (datasets/) resolve correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.insert(0, project_root)

from src.GA_Archana import relaxed_normAPPB_FW_seeds, QAP, Fugal, QAP_init, Fugal_init
from src.noise import read_real_graph, generate_graphs, edges_to_adj, eval_align

print(f"Current working directory: {project_root}")


DATASETS = {
    "netscience": {"n": 379, "path": "datasets/netscience.txt"},
    "highschool": {"n": 327, "path": "datasets/highschool.txt"},
    "euroroad": {"n": 1175, "path": "datasets/inf-euroroad.txt"},
    "multimanga": {"n": 1004, "path": "datasets/multimanga.txt"},
    "voles": {"n": 712, "path": "datasets/voles.txt"},
}

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_DIR = "baseline_data"
P0_DIR = "baseline_fw_seeds"

ts = datetime.now().strftime("%m%d_%H%M")
results_csv = os.path.join(RESULTS_DIR, f"exp_{ts}.csv")
summary_txt = os.path.join(RESULTS_DIR, f"summary_{ts}.txt")

NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20]
TRIAL_SEEDS = [42, 123, 7]

def _print(str):
    # appends to summary.txt and stdout
    logline = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {str}"
    print(logline)
    with open(summary_txt, "a") as f:
        f.write(logline + "\n")

def _write_csv(results_batch: Dict[str, Any]):
    fieldnames = ["dataset", "noise", "trial", "algorithm",
                  "mu", "lam_step",
                  "accuracy", "frobenius", "gt_frobenius"]

    # if results_csv doesn't exist, create it
    if not os.path.exists(results_csv):
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # append results to results_csv
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(results_batch)


# ---------------------------------------------------------------------------
# Phase 1: Generate & save graph pairs
# ---------------------------------------------------------------------------

def extract_edges(adj):
    """Extract upper-triangular edge list from adjacency matrix."""
    return np.array(np.transpose(np.nonzero(np.triu(adj, 1))))


def save_perm_matrix(ds_name, noise_pct, trial_idx, src_adj, tar_adj):
    # Cache / load FW seed (shared by Fugal_init and QAP_init)
    fw_fname = f"{ds_name}_noise{noise_pct}_trial{trial_idx}_fw.npz"
    fw_fpath = os.path.join(P0_DIR, fw_fname)

    if not os.path.exists(fw_fpath):
        print(f"  Computing initial permutaion matrix for {ds_name} "
              f"noise={noise_pct}% trial={trial_idx} ...")
        P0 = relaxed_normAPPB_FW_seeds(src_adj, tar_adj)
        np.savez(fw_fpath, P0=P0)
    else:
        _print(f"  [cached] Initial permutaion matrix for {ds_name} ")


def generate_and_save_graphs():
    _print("=" * 60)
    _print("Phase 1: Generating graph pairs")
    _print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    for ds_name, ds_info in DATASETS.items():
        G = read_real_graph(n=ds_info["n"], name_=ds_info["path"])
        A = nx.to_numpy_array(G, dtype=int)
        edges = extract_edges(A)
        n = int(edges.max()) + 1

        for noise in NOISE_LEVELS:
            noise_pct = int(noise * 100)
            for trial_idx, seed in enumerate(TRIAL_SEEDS):
                fname = f"{ds_name}_noise{noise_pct}_trial{trial_idx}.npz"
                fpath = os.path.join(DATA_DIR, fname)

                if os.path.exists(fpath):
                    _print(f"  [skip] {fname} already exists")
                    continue

                np.random.seed(seed)
                Src_e, Tar_e, GT = generate_graphs(edges, 0, noise)
                Src_adj = edges_to_adj(Src_e, n)
                Tar_adj = edges_to_adj(Tar_e, n)

                np.savez(fpath, Src_adj=Src_adj, Tar_adj=Tar_adj,
                         GT0=GT[0], GT1=GT[1],
                         seed=seed, noise=noise)

                save_perm_matrix(ds_name, noise_pct, trial_idx, Src_adj, Tar_adj)
                print(f"  [saved] {fname}")

    saved_graphs = [f for f in os.listdir(DATA_DIR) if f.endswith(".npz")]
    saved_perm_matrices = [f for f in os.listdir(P0_DIR) if f.endswith(".npz")]

    _print(f"\nGeneration complete: {len(saved_graphs)} files in {DATA_DIR}/, and {len(saved_perm_matrices)} files in {P0_DIR}/")





# ---------------------------------------------------------------------------
# Phase 2: Run experiments
# ---------------------------------------------------------------------------

def get_saved_graphs(ds_name, noise, trial_idx):
    noise_pct = int(noise * 100)
    fname = f"{ds_name}_noise{noise_pct}_trial{trial_idx}.npz"
    fpath = os.path.join(DATA_DIR, fname)

    # It'll crash if the file doesn't exist

    data = np.load(fpath)
    src_adj = data["Src_adj"]
    tar_adj = data["Tar_adj"]
    GT0 = data["GT0"]
    GT1 = data["GT1"]

    return src_adj, tar_adj, GT0, GT1

def get_saved_perm_matrix(ds_name, noise, trial_idx):
    noise_pct = int(noise * 100)
    fw_fname = f"{ds_name}_noise{noise_pct}_trial{trial_idx}_fw.npz"
    fw_fpath = os.path.join(P0_DIR, fw_fname)

    P0 = np.load(fw_fpath)["P0"]
    return P0


def f1(row_ind, col_ind, A, B):
    n = len(row_ind)
    Perm = np.zeros((n, n))
    Perm[row_ind, col_ind] = 1.0
    frob_norm = np.linalg.norm(A - Perm @ B @ Perm.T, 'fro') ** 2
    return frob_norm

def gtFrobNorm(GT0, src_adj, tar_adj):
    # GT Frobenius: best possible ||Src - P_GT @ Tar @ P_GT^T||_F^2
    # GT0 is the forward mapping (src node i -> tar node GT0[i]),
    # consistent with how Tar was constructed: Tar_e = GT0[Src_e]
    n_gt = len(GT0)
    P_GT = np.zeros((n_gt, n_gt))
    P_GT[np.arange(n_gt), GT0] = 1.0
    gt_frob = round(float(np.linalg.norm(src_adj - P_GT @ tar_adj @ P_GT.T, 'fro') ** 2), 4)
    return gt_frob

def evaluate(P, GT0, src_adj, tar_adj):
    """Run Hungarian, return best accuracy and Frobenius norm."""
    if isinstance(P, torch.Tensor):
        P = P.detach().numpy()

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(P, maximize=True)
    acc = eval_align(row_ind, col_ind, GT0)
    frob_norm = f1(row_ind, col_ind, src_adj, tar_adj)

    return acc, frob_norm

def run_experiments(muVals, lamSteps, algos):
    _print("\n" + "=" * 60)
    _print(f"Phase 2: Running experiment")
    _print("=" * 60)
    results = []

    for ds_name in DATASETS:    #5
        for noise in NOISE_LEVELS:  #5
            for trial_idx in range(len(TRIAL_SEEDS)): # 3
                src_adj, tar_adj, GT0, GT1 = get_saved_graphs(ds_name, noise, trial_idx)

                for algo in algos:  # 4 + 5*5

                    # multiple parameters only for fugal_init
                    params = [(None, None)]
                    if algo == "fugal_init":
                        params = [(mu, ls) for mu in muVals for ls in lamSteps]

                    if dry_run:
                        continue

                    for mu, lam_step in params:
                        if algo == "fugal_init":
                            p0 = get_saved_perm_matrix(ds_name, noise, trial_idx)
                            permMatrix = Fugal_init(src_adj, tar_adj, iter=10, mu=mu, lam_step=lam_step, P0=p0)

                        elif algo == "qap_init":
                            # qap_init algo
                            p0 = get_saved_perm_matrix(ds_name, noise, trial_idx)
                            permMatrix = QAP_init(src_adj, tar_adj, p0)

                        elif algo == "fugal":
                            # fugal algo
                            permMatrix = Fugal(src_adj, tar_adj, iter=10)

                        elif algo == "qap":
                            permMatrix = QAP(src_adj, tar_adj)

                        acc, frob = evaluate(permMatrix, GT0, src_adj, tar_adj)
                        gt_frob = gtFrobNorm(GT0, src_adj, tar_adj)

                        res = {
                            "dataset": ds_name,
                            "noise": int(noise * 100),
                            "trial": trial_idx,
                            "algorithm": algo,
                            "mu": mu,
                            "lam_step": lam_step,
                            "accuracy": round(acc, 6),
                            "frobenius": round(frob, 4),
                            "gt_frobenius": gt_frob,
                        }

                        results.append(res)
                        _write_csv(res)



    return results





# ---------------------------------------------------------------------------
# Phase 3: Main
# ---------------------------------------------------------------------------

def save_summary(results):
    pass


def main():
    parser = argparse.ArgumentParser(description="Baseline benchmark harness")

    # Interface shall support a list of mu values
    parser.add_argument("--mu", nargs='+', type=float, default=[1])
    # Interface shall support a list of lam_step values
    parser.add_argument("--lam-step", nargs='+', type=float, default=[1.0])
    # Interface should take list of algos
    parser.add_argument("--algos", nargs='+',  type=str, default=['fugal_init', 'qap_init'], choices=['fugal_init', 'fugal', 'qap', 'qap_init'])

    args = parser.parse_args()

    # summary should contain the args passed to the script in the first row

    generate_and_save_graphs()

    results = run_experiments(args.mu, args.lam_step, args.algos)

    if results:
        save_summary(results)
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()