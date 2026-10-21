import pandas as pd
import numpy as np

CSV = "compare_nisq_rigorous.csv"
BUDGET_COLS = ["total_shots", "total_circuits", "shots", "n_shots", "circuits", "n_circuits"]

def main():
    df = pd.read_csv(CSV)
    print("Colonnes disponibles :", list(df.columns))
    
    found = [c for c in BUDGET_COLS if c in df.columns]
    if not found:
        print("AUCUNE colonne de budget detectee.")
        return

    last = (df.sort_values(["seed", "optimizer", "step"]).groupby(["seed", "optimizer"]).tail(1))
    budget_col = found[0]
    print(f"\nColonne de budget utilisee : {budget_col}")
    
    summary = last.groupby("optimizer").agg(
        gap_mediane=("gap", "median"),
        budget_mediane=(budget_col, "median"),
        budget_total=(budget_col, "sum"),
    )
    summary["gap_par_budget"] = (summary["gap_mediane"] / summary["budget_mediane"])
    
    print("\n", summary.sort_values("gap_par_budget"))
    print("\nLecture : a budget egal, le meilleur optimiseur est celui avec le plus petit 'gap_par_budget'.")

if __name__ == "__main__":
    main()
