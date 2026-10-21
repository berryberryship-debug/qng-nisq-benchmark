import argparse
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from qng_benchmark import heisenberg_xxz

BUDGET = {"circuits": 0, "shots": 0}

PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

def pauli_to_matrix(s):
    m = np.array([[1]], dtype=np.complex128)
    for c in s:
        m = np.kron(m, PAULI[c])
    return m

def build_pool(n):
    pool = []
    for i in range(n):
        for P in "XYZ":
            s = ["I"] * n
            s[i] = P
            pool.append(pauli_to_matrix("".join(s)))
    for i in range(n):
        for j in range(i + 1, n):
            for P1 in "XYZ":
                for P2 in "XYZ":
                    s = ["I"] * n
                    s[i], s[j] = P1, P2
                    pool.append(pauli_to_matrix("".join(s)))
    return pool

def rot(psi, P, th):
    return np.cos(th / 2) * psi - 1j * np.sin(th / 2) * (P @ psi)

def forward(psi0, ops, th):
    states = [psi0]
    for P, t in zip(ops, th):
        states.append(rot(states[-1], P, t))
    return states

class Evaluator:
    def __init__(self, H, psi0, ops, shots):
        self.H, self.psi0, self.ops = H, psi0, ops
        self.shots = shots
        self.cost = 0

    def energy(self, th):
        self.cost += 1
        global BUDGET
        BUDGET["circuits"] += 1
        BUDGET["shots"] += self.shots
        psi = forward(self.psi0, self.ops, th)[-1]
        E_exact = float(np.real(np.vdot(psi, self.H @ psi)))
        noise = np.random.normal(0, 1.0 / np.sqrt(self.shots))
        return E_exact + noise

    def _deriv(self, states, th):
        p = len(th)
        D = np.empty((p, self.psi0.size), dtype=np.complex128)
        for j in range(p):
            d = -0.5j * (self.ops[j] @ states[j + 1])
            for k in range(j + 1, p):
                d = rot(d, self.ops[k], th[k])
            D[j] = d
        return D

    def value_grad(self, th):
        p = len(th)
        self.cost += 1 + 2 * p
        global BUDGET
        BUDGET["circuits"] += 1 + 2 * p
        BUDGET["shots"] += (1 + 2 * p) * self.shots
        
        states = forward(self.psi0, self.ops, th)
        psi = states[-1]
        Hpsi = self.H @ psi
        E_exact = float(np.real(np.vdot(psi, Hpsi)))
        D = self._deriv(states, th)
        g_exact = 2.0 * np.real(D @ np.conj(Hpsi))
        
        noise_scale = 1.0 / np.sqrt(self.shots)
        E_noisy = E_exact + np.random.normal(0, noise_scale)
        g_noisy = g_exact + np.random.normal(0, noise_scale, size=g_exact.shape)
        return E_noisy, g_noisy, states, D

    def qfim(self, states, D):
        p = D.shape[0]
        self.cost += p * (p + 1) // 2
        global BUDGET
        BUDGET["circuits"] += p * (p + 1) // 2
        BUDGET["shots"] += (p * (p + 1) // 2) * self.shots
        
        psi = states[-1]
        A = np.conj(D) @ D.T
        v = np.conj(D) @ psi
        Q = A - np.outer(v, np.conj(v))
        F = 4.0 * np.real(Q)
        return 0.5 * (F + F.T)

def opt_lbfgs(ev, th, K, state, lr):
    res = minimize(lambda x: ev.value_grad(x)[:2], th, jac=True, method="L-BFGS-B",
                   options={"maxiter": K, "ftol": 1e-12, "gtol": 1e-10})
    return res.x

def opt_descent(ev, th, K, state, lr, method_type):
    lam_min, lam_max = 1e-6, 1e3
    tau_rel = 1e-4 if method_type == "qng_trunc" else 0.0

    for _ in range(K):
        E, g, states, D = ev.value_grad(th)
        if np.linalg.norm(g) < 1e-9:
            break
        
        if method_type in ["qng_standard", "qng_trunc", "qng_shrunk"]:
            F = ev.qfim(states, D)
            lam = state.get("lam", 1e-6)
            
            if method_type == "qng_shrunk":
                n_shots = getattr(ev, 'shots', 512)
                sigma_2 = 1.0 / n_shots
                gamma = sigma_2 * np.trace(F) / len(th)
                F_reg = F + (gamma + lam) * np.eye(len(th))
                try:
                    d = np.linalg.solve(F_reg, g)
                except np.linalg.LinAlgError:
                    d = g
                eigvals = np.linalg.eigvalsh(F_reg)
                state["n_kept"] = len(eigvals)
                state["cond_eff"] = eigvals.max() / eigvals.min() if eigvals.min() > 0 else np.inf
            elif method_type == "qng_trunc":
                eigvals, eigvecs = np.linalg.eigh(F)
                eigvals = np.maximum(eigvals, 0.0)
                tau_eff = tau_rel * max(float(eigvals.max()), 1e-15)
                coeff = np.where(eigvals > tau_eff, 1.0 / (eigvals + lam), 0.0)
                d = eigvecs @ (coeff * (eigvecs.T @ g))
                positive = eigvals[eigvals > tau_eff]
                state["n_kept"] = len(positive)
                state["cond_eff"] = positive.max() / positive.min() if len(positive) > 0 else np.inf
            else: # qng_standard
                eigvals, eigvecs = np.linalg.eigh(F)
                eigvals = np.maximum(eigvals, 0.0)
                coeff = 1.0 / (eigvals + lam)
                d = eigvecs @ (coeff * (eigvecs.T @ g))
                state["n_kept"] = len(eigvals)
                state["cond_eff"] = eigvals.max() / eigvals.min() if eigvals.min() > 0 else np.inf
        else:
            d = g
        
        pred = float(g @ d)
        gamma = lr
        ok = False
        Et = E
        for _ in range(12):
            cand = th - gamma * d
            Et = ev.energy(cand)
            if E - Et > 0:
                ok = True
                break
            gamma *= 0.5
            
        if not ok:
            if method_type in ["qng_standard", "qng_trunc", "qng_shrunk"]:
                state["lam"] = min(lam * 2.0, lam_max)
                continue
            break
            
        if method_type in ["qng_standard", "qng_trunc", "qng_shrunk"]:
            rho = (E - Et) / max(gamma * pred, 1e-15)
            if rho >= 0.75:
                state["lam"] = max(lam * 0.5, lam_min)
            elif rho < 0.25:
                state["lam"] = min(lam * 2.0, lam_max)
                
        th = cand
    return th

OPTIMIZERS = {
    "lbfgs": opt_lbfgs,
    "qng_standard": lambda ev, th, K, st, lr: opt_descent(ev, th, K, st, lr, "qng_standard"),
    "qng_trunc": lambda ev, th, K, st, lr: opt_descent(ev, th, K, st, lr, "qng_trunc"),
    "qng_shrunk": lambda ev, th, K, st, lr: opt_descent(ev, th, K, st, lr, "qng_shrunk"),
    "gd": lambda ev, th, K, st, lr: opt_descent(ev, th, K, st, lr, "gd"),
}

def run_one(seed, method, H, pool_arr, pool_list, E0, max_ops, K, lr, shots):
    rng = np.random.default_rng(seed)
    psi0 = rng.normal(size=H.shape[0]) + 1j * rng.normal(size=H.shape[0])
    psi0 /= np.linalg.norm(psi0)
    ops, th = [], np.zeros(0)
    ev = Evaluator(H, psi0, ops, shots)
    state = {"lam": 1e-3, "cond_eff": 0, "n_kept": 0}
    psi = psi0
    rows = []
    t0 = time.perf_counter()
    
    for step in range(1, max_ops + 1):
        Hpsi = H @ psi
        Ppsi = pool_arr @ psi
        x = np.sum(np.conj(Ppsi) * Hpsi, axis=1)
        grads = np.real(0.5j * (x - np.conj(x)))
        idx = int(np.argmax(np.abs(grads)))
        gmax = float(abs(grads[idx]))
        if gmax < 1e-6:
            break
        
        ops.append(pool_list[idx])
        th = np.append(th, 0.0)
        th = OPTIMIZERS[method](ev, th, K, state, lr)
        psi = forward(psi0, ops, th)[-1]
        E = float(np.real(np.vdot(psi, H @ psi)))
        
        rows.append({
            "seed": seed, "optimizer": method, "step": step,
            "energy": E, "gap": E - E0, "cost_proxy": ev.cost,
            "cond_eff": state["cond_eff"], "n_kept": state["n_kept"],
            "time_s": time.perf_counter() - t0,
            "total_shots": BUDGET["shots"], "total_circuits": BUDGET["circuits"]
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--max-ops", type=int, default=30)
    ap.add_argument("--inner-iters", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--n-qubits", type=int, default=6)
    ap.add_argument("--out", default="compare_nisq_rigorous.csv")
    args = ap.parse_args()

    methods = ["lbfgs", "qng_standard", "qng_trunc", "qng_shrunk", "gd"]
    n = args.n_qubits
    H = np.asarray(heisenberg_xxz(n), dtype=np.complex128)
    E0 = float(np.linalg.eigvalsh(H)[0])
    pool_list = build_pool(n)
    pool_arr = np.stack(pool_list)
    
    print(f"E_exact = {E0:.8f} | graines = {args.n_seeds} | Shots = {args.shots}")
    rows = []
    
    for seed in range(args.n_seeds):
        for m in methods:
            global BUDGET
            BUDGET = {"circuits": 0, "shots": 0}
            t = time.perf_counter()
            rows += run_one(seed, m, H, pool_arr, pool_list, E0, args.max_ops, args.inner_iters, args.lr, args.shots)
            print(f"graine {seed:3d} | {m:12s} | {time.perf_counter()-t:6.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n=== Résultat de l'analyse comparative sous bruit (NISQ) ===")
    for m in methods:
        sub = df[df.optimizer == m]
        if len(sub) == 0:
            continue
        last = sub.sort_values(["seed", "step"]).groupby("seed").tail(1)
        final_gaps = last["gap"]
        print(f"{m:12s} | Gap final médian = {final_gaps.median():.4g} | n = {len(final_gaps)}")

if __name__ == '__main__':
    main()
