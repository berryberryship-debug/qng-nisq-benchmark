import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_benchmark_results(csv_path="benchmark_results.csv", output_path="benchmark_distributions.png"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    df = pd.read_csv(csv_path)
    metric_col = "Delta_E" if "Delta_E" in df.columns else "Energy"
    metric_label = r"Écart à l'énergie exacte \(\Delta E\)" if metric_col == "Delta_E" else "Énergie"

    order = df.groupby("Optimizer")[metric_col].median().sort_values().index.tolist()
    palette = ["#1f77b4", "#2ca02c", "#d95f02", "#e377c2"]
    colors = {opt: palette[i % len(palette)] for i, opt in enumerate(order)}

    pivot_metric = df.pivot(index="Seed", columns="Optimizer", values=metric_col).dropna()[order]
    n_seeds = len(pivot_metric)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    rng = np.random.default_rng(42)

    ax1 = axes[0, 0]
    data_metric = [pivot_metric[opt].to_numpy() for opt in order]
    bp1 = ax1.boxplot(data_metric, tick_labels=order, patch_artist=True, widths=0.45, showfliers=False, medianprops={"color": "#111111", "linewidth": 2.0})
    for patch, opt in zip(bp1["boxes"], order):
        patch.set_facecolor(colors[opt]); patch.set_alpha(0.35)
    for idx, opt in enumerate(order, start=1):
        y_vals = pivot_metric[opt].to_numpy()
        ax1.scatter(np.full_like(y_vals, idx, dtype=float) + rng.normal(0, 0.06, size=len(y_vals)), y_vals, color=colors[opt], alpha=0.75, s=18, zorder=3)
    ax1.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0, label="Exact (\(\Delta E = 0\))")
    ax1.set_title(f"Distribution de l'erreur (n = {n_seeds})")
    ax1.set_ylabel(metric_label)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.legend(frameon=False, loc="upper right")

    ax2 = axes[0, 1]
    x_positions = np.arange(1, len(order) + 1)
    for _, row in pivot_metric.iterrows():
        ax2.plot(x_positions, row.to_numpy(), color="#666666", alpha=0.15, linewidth=0.8, zorder=1)
    medians = [pivot_metric[opt].median() for opt in order]
    ax2.plot(x_positions, medians, color="#b2182b", marker="o", linewidth=2.4, markersize=6, label="Médiane", zorder=4)
    for idx, opt in enumerate(order, start=1):
        ax2.scatter(np.full_like(pivot_metric[opt].to_numpy(), idx, dtype=float), pivot_metric[opt].to_numpy(), color=colors[opt], alpha=0.65, s=16, zorder=3)
    ax2.set_xticks(x_positions); ax2.set_xticklabels(order)
    ax2.set_title("Évolution par graine (même point de départ)")
    ax2.set_ylabel(metric_label)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    ax3 = axes[1, 0]
    pivot_evals = df.pivot(index="Seed", columns="Optimizer", values="Evals").dropna()[order]
    data_evals = [pivot_evals[opt].to_numpy() for opt in order]
    bp3 = ax3.boxplot(data_evals, tick_labels=order, patch_artist=True, widths=0.45, showfliers=False, medianprops={"color": "#111111", "linewidth": 2.0})
    for patch, opt in zip(bp3["boxes"], order):
        patch.set_facecolor(colors[opt]); patch.set_alpha(0.35)
    for idx, opt in enumerate(order, start=1):
        y_vals = pivot_evals[opt].to_numpy()
        ax3.scatter(np.full_like(y_vals, idx, dtype=float) + rng.normal(0, 0.06, size=len(y_vals)), y_vals, color=colors[opt], alpha=0.75, s=18, zorder=3)
    ax3.set_title("Budget d'évaluations consommé")
    ax3.set_ylabel("Appels à l'Estimator")
    ax3.grid(axis="y", linestyle=":", alpha=0.5)

    ax4 = axes[1, 1]
    success_rates = [(pivot_metric[opt] <= 0.1).mean() * 100 for opt in order]
    bars = ax4.bar(order, success_rates, color=[colors[opt] for opt in order], alpha=0.7)
    ax4.set_ylabel("Taux de succès (%)")
    ax4.set_title("Fiabilité (\(\Delta E \le 0.1\))")
    ax4.set_ylim(0, 110)
    ax4.grid(axis="y", linestyle=":", alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.0f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Graphique sauvegardé dans {output_path}")
