import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(history_dir="history", output_path="convergence_curves.png", exact_energy=None, best_so_far=True):
    history_files = sorted(glob.glob(os.path.join(history_dir, "history_seed_*.json")))
    if not history_files:
        raise FileNotFoundError(f"Aucun fichier d'historique trouvé dans {history_dir}/")

    print(f"Chargement de {len(history_files)} historiques...")
    data = {"L-BFGS-B": [], "COBYLA": [], "GD (CG)": [], "SPSA": []}

    for file in history_files:
        with open(file, "r") as f:
            hist = json.load(f)
            for opt, vals in hist.items():
                if opt in data and vals:
                    traj = np.minimum.accumulate(vals).tolist() if best_so_far else vals
                    data[opt].append(traj)

    all_lengths = [len(run) for runs in data.values() for run in runs]
    global_max_len = max(all_lengths)

    plt.figure(figsize=(10, 6), dpi=150)
    colors = {"L-BFGS-B": "#1f77b4", "COBYLA": "#2ca02c", "GD (CG)": "#d95f02", "SPSA": "#e377c2"}

    for opt, runs in data.items():
        if not runs:
            continue
        padded_runs = np.array([run + [run[-1]] * (global_max_len - len(run)) for run in runs])
        median_curve = np.median(padded_runs, axis=0)
        q25 = np.percentile(padded_runs, 25, axis=0)
        q75 = np.percentile(padded_runs, 75, axis=0)
        x_axis = np.arange(1, global_max_len + 1)

        plt.plot(x_axis, median_curve, label=f"{opt} (Médiane)", color=colors[opt], linewidth=2)
        plt.fill_between(x_axis, q25, q75, color=colors[opt], alpha=0.15)

    if exact_energy is not None:
        plt.axhline(exact_energy, color="#444444", linestyle="--", linewidth=1.2, label=f"Énergie exacte ({exact_energy:.4f})")

    plt.title(f"Vitesse de descente cumulée (n = {len(history_files)} graines)")
    plt.xlabel("Nombre d'appels à l'Estimator (Évaluations)")
    plt.ylabel("Meilleure énergie atteinte" if best_so_far else "Énergie évaluée")
    plt.legend(frameon=False)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Graphique de convergence sauvegardé dans {output_path}")
