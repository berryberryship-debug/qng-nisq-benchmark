import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from qiskit import transpile
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

try:
    from qiskit_algorithms.optimizers import SPSA
except ModuleNotFoundError:
    from qiskit.algorithms.optimizers import SPSA

warnings.filterwarnings('ignore')

def build_heisenberg_xxz(n_qubits=6, J=1.0, delta=1.0):
    pauli_list = []
    for i in range(n_qubits - 1):
        op_x = "".join(['X' if j in (i, i + 1) else 'I' for j in range(n_qubits)])
        op_y = "".join(['Y' if j in (i, i + 1) else 'I' for j in range(n_qubits)])
        op_z = "".join(['Z' if j in (i, i + 1) else 'I' for j in range(n_qubits)])
        pauli_list.append((op_x, J))
        pauli_list.append((op_y, J))
        pauli_list.append((op_z, J * delta))
    return SparsePauliOp.from_list(pauli_list)

def get_realistic_noise_model(p1=0.001, p2=0.01, t1=50e-6, t2=70e-6):
    noise_model = NoiseModel()
    gate_time_1q = 1e-7
    gate_time_2q = 3e-7
    error_1q_total = depolarizing_error(p1, 1).compose(thermal_relaxation_error(t1, t2, gate_time_1q))
    noise_model.add_all_qubit_quantum_error(error_1q_total, ['u'])
    error_2q_total = depolarizing_error(p2, 2).compose(thermal_relaxation_error(t1, t2, gate_time_2q).expand(thermal_relaxation_error(t1, t2, gate_time_2q)))
    noise_model.add_all_qubit_quantum_error(error_2q_total, ['cx'])
    return noise_model

def evaluate_energy(params, ansatz, hamiltonian, estimator, history_list=None):
    job = estimator.run(circuits=[ansatz], observables=[hamiltonian], parameter_values=[params])
    val = float(job.result().values[0])
    if history_list is not None:
        history_list.append(val)
    return val

def load_checkpoint(csv_path, expected_optimizers=4):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return [], set()
    df_existing = pd.read_csv(csv_path)
    counts = df_existing.groupby('Seed')['Optimizer'].nunique()
    completed_seeds = set(counts[counts == expected_optimizers].index)
    df_clean = df_existing[df_existing['Seed'].isin(completed_seeds)]
    return df_clean.to_dict('records'), completed_seeds

def save_checkpoint(results, csv_path):
    tmp_path = f"{csv_path}.tmp"
    pd.DataFrame(results).to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)

def run_benchmark(n_seeds=77, budget_evals=80, csv_path="benchmark_results.csv", history_dir="history"):
    os.makedirs(history_dir, exist_ok=True)
    
    n_qubits = 6
    hamiltonian = build_heisenberg_xxz(n_qubits=n_qubits, J=1.0, delta=1.0)
    exact_energy = float(np.linalg.eigvalsh(hamiltonian.to_matrix())[0])
    print(f"Énergie fondamentale exacte (XXZ, {n_qubits} qubits) : {exact_energy:.6f}")

    raw_ansatz = TwoLocal(n_qubits, rotation_blocks='ry', entanglement_blocks='cx', entanglement='linear', reps=1)
    ansatz = transpile(raw_ansatz, basis_gates=['u', 'cx'], optimization_level=1)
    num_params = ansatz.num_parameters

    noise_model = get_realistic_noise_model()
    estimator = Estimator(backend_options={"method": "density_matrix", "noise_model": noise_model}, run_options={"shots": 512})

    grad_iters = max(2, budget_evals // (num_params + 1))
    results, completed_seeds = load_checkpoint(csv_path, expected_optimizers=4)
    
    if completed_seeds:
        print(f"Reprise détectée : {len(completed_seeds)}/{n_seeds} seeds déjà complétées.\n")

    for seed in range(n_seeds):
        if seed in completed_seeds:
            continue

        print(f"--- Seed {seed + 1}/{n_seeds} (en cours) ---")
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0, 2 * np.pi, size=num_params)
        seed_results = []
        seed_histories = {}

        hist_lbfgs = []
        res_lbfgs = minimize(evaluate_energy, x0, args=(ansatz, hamiltonian, estimator, hist_lbfgs),
                             method='L-BFGS-B', options={'maxiter': grad_iters, 'maxfun': budget_evals, 'eps': 0.1})
        seed_results.append({"Optimizer": "L-BFGS-B", "Seed": seed, "Energy": res_lbfgs.fun, "Delta_E": res_lbfgs.fun - exact_energy, "Evals": res_lbfgs.nfev})
        seed_histories["L-BFGS-B"] = hist_lbfgs

        hist_cobyla = []
        res_cobyla = minimize(evaluate_energy, x0, args=(ansatz, hamiltonian, estimator, hist_cobyla),
                              method='COBYLA', options={'maxiter': budget_evals, 'rhobeg': 0.5})
        seed_results.append({"Optimizer": "COBYLA", "Seed": seed, "Energy": res_cobyla.fun, "Delta_E": res_cobyla.fun - exact_energy, "Evals": res_cobyla.nfev})
        seed_histories["COBYLA"] = hist_cobyla

        hist_cg = []
        res_gd = minimize(evaluate_energy, x0, args=(ansatz, hamiltonian, estimator, hist_cg),
                          method='CG', options={'maxiter': grad_iters, 'eps': 0.1})
        seed_results.append({"Optimizer": "GD (CG)", "Seed": seed, "Energy": res_gd.fun, "Delta_E": res_gd.fun - exact_energy, "Evals": res_gd.nfev})
        seed_histories["GD (CG)"] = hist_cg

        max_spsa_iters = budget_evals // 2
        spsa_opt = SPSA(maxiter=max_spsa_iters)
        hist_spsa = []
        res_spsa = spsa_opt.minimize(fun=lambda p: evaluate_energy(p, ansatz, hamiltonian, estimator, hist_spsa), x0=x0)
        energy_spsa = float(res_spsa.fun)
        evals_spsa = int(res_spsa.nfev) if res_spsa.nfev is not None else len(hist_spsa)
        seed_results.append({"Optimizer": "SPSA", "Seed": seed, "Energy": energy_spsa, "Delta_E": energy_spsa - exact_energy, "Evals": evals_spsa})
        seed_histories["SPSA"] = hist_spsa

        results.extend(seed_results)
        save_checkpoint(results, csv_path)
        
        with open(os.path.join(history_dir, f"history_seed_{seed}.json"), "w") as f:
            json.dump(seed_histories, f)
            
        print(f"Seed {seed + 1}/{n_seeds} sauvegardée.")
