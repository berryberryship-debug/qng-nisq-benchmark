import os
import time
import argparse
import numpy as np

from benchmark_nisq import build_heisenberg_xxz, run_benchmark
from analyze_benchmark import analyze
from plot_benchmark import plot_benchmark_results
from plot_convergence import plot_convergence

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline complet de benchmark variationnel XXZ bruité.")
    parser.add_argument("--seeds", type=int, default=77, help="Nombre de graines aléatoires (défaut : 77)")
    parser.add_argument("--budget", type=int, default=80, help="Budget maximal d'évaluations par optimiseur (défaut : 80)")
    parser.add_argument("--out-dir", type=str, default="results_xxz", help="Dossier de sortie (défaut : results_xxz)")
    parser.add_argument("--skip-sim", action="store_true", help="Ignorer la simulation et exécuter uniquement l'analyse et les graphiques")
    return parser.parse_args()

def main():
    args = parse_args()
    t_start = time.perf_counter()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "benchmark_results.csv")
    history_dir = os.path.join(args.out_dir, "history")
    dist_plot_path = os.path.join(args.out_dir, "benchmark_distributions.png")
    conv_plot_path = os.path.join(args.out_dir, "convergence_curves.png")

    hamiltonian = build_heisenberg_xxz(n_qubits=6, J=1.0, delta=1.0)
    exact_energy = float(np.linalg.eigvalsh(hamiltonian.to_matrix())[0])

    print("============================================================")
    print(f"PIPELINE BENCHMARK XXZ | Seeds: {args.seeds} | Budget: {args.budget}")
    print(f"Énergie fondamentale exacte : {exact_energy:.6f} J")
    print(f"Dossier de sortie : {args.out_dir}/")
    print("============================================================\n")

    if not args.skip_sim:
        print("[1/4] Exécution du benchmark quantique...")
        run_benchmark(n_seeds=args.seeds, budget_evals=args.budget, csv_path=csv_path, history_dir=history_dir)
    else:
        print("[1/4] Simulation ignorée (--skip-sim activé).")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Aucun résultat trouvé à {csv_path}.")

    print("\n[2/4] Analyse statistique non paramétrique...")
    analyze(csv_path=csv_path)

    print("\n[3/4] Génération du graphique des distributions...")
    plot_benchmark_results(csv_path=csv_path, output_path=dist_plot_path)

    print("\n[4/4] Génération des courbes de convergence...")
    plot_convergence(history_dir=history_dir, output_path=conv_plot_path, exact_energy=exact_energy, best_so_far=True)

    elapsed = time.perf_counter() - t_start
    print("\n============================================================")
    print(f"Pipeline terminé avec succès en {elapsed:.1f} secondes.")
    print("============================================================")

if __name__ == "__main__":
    main()
