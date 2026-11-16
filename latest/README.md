# NISQ VQE Benchmark : Optimiseurs sous bruit réaliste

Ce projet est un banc d'essai (benchmark) comparatif d'optimiseurs pour l'algorithme VQE (*Variational Quantum Eigensolver*) appliqué à une chaîne de spins de Heisenberg XXZ à 6 qubits.

## 🔬 Objectif
Évaluer la performance de 4 optimiseurs face au bruit réaliste d'un ordinateur quantique NISQ (dépolarisation, relaxation thermique T1/T2, et bruit d'échantillonnage à 512 shots).

## ⚙️ Algorithmes comparés
- **L-BFGS-B** (Quasi-Newton, gradient numérique)
- **CG** (Gradient Conjugué)
- **COBYLA** (Sans gradient, méthode des simplexes)
- **SPSA** (Simultaneous Perturbation Stochastic Approximation, standard de l'industrie pour le NISQ)

## 📊 Résultats principaux (77 graines aléatoires)

L'étude démontre statistiquement (test de Wilcoxon corrigé de Holm, p < 0.05) que **COBYLA et SPSA écrasent les méthodes à gradient** (L-BFGS-B et CG).

- **SPSA et COBYLA** atteignent les meilleures énergies (erreur médiane la plus faible) tout en respectant strictement le budget de 80 évaluations.
- **L-BFGS-B et CG** échouent à converger. L'estimation numérique du gradient sous un fort bruit quantique consomme tout le budget computationnel (jusqu'à 299 évaluations pour CG) et génère des directions de descente erratiques.

## 🚀 Utilisation

1. Installer les dépendances : `pip install -r requirements.txt`
2. Lancer le test rapide (3 graines) : `python run_all.py --seeds 3`
3. Lancer l'étude complète (77 graines) : `python run_all.py --seeds 77 --out-dir results_77_seeds`
