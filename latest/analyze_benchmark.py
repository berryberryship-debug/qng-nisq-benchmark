import os
import json
import glob
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats

def load_initial_energies(history_dir):
    initial_energies = {}
    for file in glob.glob(os.path.join(history_dir, "history_seed_*.json")):
        seed = int(os.path.basename(file).replace("history_seed_", "").replace(".json", ""))
        with open(file, "r") as f:
            hist = json.load(f)
            if "L-BFGS-B" in hist and len(hist["L-BFGS-B"]) > 0:
                initial_energies[seed] = hist["L-BFGS-B"][0]
            elif "COBYLA" in hist and len(hist["COBYLA"]) > 0:
                initial_energies[seed] = hist["COBYLA"][0]
    return initial_energies

def summarize_distributions(df, initial_energies, epsilon_threshold=0.1):
    rows = []
    for opt, group in df.groupby("Optimizer"):
        de = group["Delta_E"].to_numpy()
        ev = group["Evals"].to_numpy()
        seeds = group["Seed"].to_numpy()
        
        q25_de, med_de, q75_de = np.percentile(de, [25, 50, 75])
        iqr_de = q75_de - q25_de
        success_rate = np.mean(de <= epsilon_threshold) * 100.0

        etas = []
        for s, energy_final, n_evals in zip(seeds, group["Energy"].to_numpy(), ev):
            if s in initial_energies and n_evals > 0:
                gain = initial_energies[s] - energy_final
                etas.append(gain / n_evals)
            else:
                etas.append(0.0)
        
        med_eta = np.median(etas) if etas else 0.0

        rows.append({
            "Optimizer": opt,
            "DeltaE_Med": med_de,
            "DeltaE_IQR": iqr_de,
            f"Succes_Rate_({epsilon_threshold})_%": success_rate,
            "Efficacite_Eta_Med": med_eta,
            "Evals_Med": np.median(ev)
        })
    
    return pd.DataFrame(rows).sort_values(by=[f"Succes_Rate_({epsilon_threshold})_%", "DeltaE_Med"], ascending=[False, True])

def run_nonparametric_tests(df):
    pivot_de = df.pivot(index="Seed", columns="Optimizer", values="Delta_E").dropna()
    optimizers = list(pivot_de.columns)
    print(f"Graines complètes appariées analysées : {len(pivot_de)}\n")

    samples = [pivot_de[opt].to_numpy() for opt in optimizers]
    friedman_stat, friedman_p = stats.friedmanchisquare(*samples)
    print("=== Test global de Friedman (Delta_E) ===")
    print(f"Statistique Q = {friedman_stat:.4f} | p-value = {friedman_p:.4e}\n")

    pairwise_rows = []
    raw_pvalues = []

    for opt_a, opt_b in combinations(optimizers, 2):
        a = pivot_de[opt_a].to_numpy()
        b = pivot_de[opt_b].to_numpy()
        stat_w, p_val = stats.wilcoxon(a, b, alternative="two-sided")
        win_rate_a = np.mean(a < b) * 100.0
        win_rate_b = np.mean(b < a) * 100.0

        raw_pvalues.append(p_val)
        pairwise_rows.append({
            "Comparaison (A vs B)": f"{opt_a} vs {opt_b}",
            "Victoires_A (%)": win_rate_a,
            "Victoires_B (%)": win_rate_b,
            "p_brut": p_val,
        })

    m = len(raw_pvalues)
    sorted_indices = np.argsort(raw_pvalues)
    adjusted_p = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        adj = raw_pvalues[idx] * (m - rank)
        running_max = max(running_max, adj)
        adjusted_p[idx] = min(1.0, running_max)

    for i, row in enumerate(pairwise_rows):
        row["p_Holm"] = adjusted_p[i]
        row["Significatif (alpha=0.05)"] = "Oui" if adjusted_p[i] < 0.05 else "Non"

    return pd.DataFrame(pairwise_rows)

def analyze(csv_path="benchmark_results.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    out_dir = os.path.dirname(csv_path) or "."
    history_dir = os.path.join(out_dir, "history")
    df = pd.read_csv(csv_path)
    
    initial_energies = load_initial_energies(history_dir)

    print("\n=== 1. Métriques de Performance (Seuil epsilon=0.1) ===")
    dist_df = summarize_distributions(df, initial_energies, epsilon_threshold=0.1)
    print(dist_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print()

    pairwise_df = run_nonparametric_tests(df)
    print("=== 2. Comparaisons appariées (Wilcoxon + correction de Holm) ===")
    print(pairwise_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    dist_path = os.path.join(out_dir, "benchmark_distributions.csv")
    pairwise_path = os.path.join(out_dir, "benchmark_pairwise_tests.csv")
    dist_df.to_csv(dist_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"\nTableaux exportés : {dist_path} et {pairwise_path}")
