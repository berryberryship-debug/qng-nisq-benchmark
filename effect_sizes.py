import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

CSV = "compare_nisq_rigorous.csv"
N_BOOT = 10000
SEED = 42

def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0: return 0.0
    ranks = pd.Series(np.abs(d)).rank()
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    return float((w_plus - w_minus) / (w_plus + w_minus))

def main():
    df = pd.read_csv(CSV)
    last = (df.sort_values(["seed", "optimizer", "step"]).groupby(["seed", "optimizer"]).tail(1))
    p = last.pivot(index="seed", columns="optimizer", values="gap")

    pairs = [("gd", "qng_shrunk"), ("gd", "qng_standard"), ("gd", "qng_trunc"), ("gd", "lbfgs"), ("qng_shrunk", "qng_standard"), ("qng_shrunk", "qng_trunc")]
    rng = np.random.default_rng(SEED)
    
    print("%22s | %8s | %8s | %10s | %22s" % ("paire", "p-value", "r", "victoires", "IC95 diff mediane"))
    print("-" * 85)

    for a, b in pairs:
        if a not in p or b not in p: continue
        pair_df = p[[a, b]].dropna()
        diff = (pair_df[a] - pair_df[b]).values
        stat, pval = wilcoxon(pair_df[a], pair_df[b])
        r = rank_biserial(diff)
        wins = int((diff < 0).sum())
        
        boots = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, len(diff), len(diff))
            boots[i] = np.median(diff[idx])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        
        print("%22s | %8.4f | %8.3f | %10s | [%+.4f, %+.4f]" % (f"{a} vs {b}", pval, r, f"{wins}/{len(diff)}", lo, hi))

if __name__ == "__main__":
    main()
