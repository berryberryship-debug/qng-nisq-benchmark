# Quantum Natural Gradient vs Classical Optimizers under NISQ Noise

## Overview
This repository contains a rigorous comparative benchmark of optimization algorithms for Variational Quantum Algorithms (VQA) applied to a 6-qubit Heisenberg XXZ spin chain. The study focuses on the impact of finite-shot noise (NISQ regime) on optimization performance and resource consumption.

## Algorithms Compared
- **L-BFGS-B** (Quasi-Newton)
- **GD** (Standard Gradient Descent)
- **QNG Standard** (Quantum Natural Gradient with Levenberg-Marquardt)
- **QNG Truncated** (QNG with binary spectral thresholding)
- **QNG Shrunk** (QNG with continuous Ledoit-Wolf shrinkage)

## Key Results (N=77 seeds, 512 shots)

| Optimizer | Median Gap (Ha) | Total Shots (M) | Statistical Significance vs L-BFGS |
| :--- | :--- | :--- | :--- |
| **QNG Standard** | **1.688** | ~10.89 | p = 0.0002 (Wins) |
| **QNG Truncated** | 1.697 | ~10.90 | - |
| **GD** | 1.721 | ~6.47 | p = 0.0002 (Wins) |
| **QNG Shrunk** | 1.774 | ~6.51 | - |
| **L-BFGS** | 1.889 | ~6.00 | (Baseline) |

### Key Findings
1. **Precision**: QNG Standard achieves the lowest median gap (1.688), followed by QNG Trunc (1.697), GD (1.721), QNG Shrunk (1.774), and L-BFGS (1.889).
2. **Statistical Significance**: QNG Standard and GD significantly outperform L-BFGS (p = 0.0002, Wilcoxon signed-rank test). However, no statistical difference is found between QNG Standard and GD (p = 0.209).
3. **Resource Tradeoff**: QNG Standard requires ~68% more shots (~10.89M) than GD (~6.47M) to achieve a marginal precision gain of 0.033 Ha.
4. **Conclusion**: Under severe shot noise (512 shots), the geometric information of the Fisher metric does not justify its computational overhead. Simple Gradient Descent (GD) offers the best cost/precision tradeoff.

## Files
- `compare_nisq_rigorous.py`: Main benchmark script.
- `qng_benchmark.py`: Hamiltonian generation.
- `effect_sizes.py`: Statistical analysis.
- `budget_audit.py`: Shot and circuit resource audit.

## Usage
```bash
python compare_nisq_rigorous.py --n-seeds 77 --max-ops 30 --inner-iters 20 --shots 512
python effect_sizes.py
python budget_audit.py
