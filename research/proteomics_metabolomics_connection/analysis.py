import pandas as pd
import numpy as np
import os
from body_system_loader.load_feature_df import BODY_SYSTEMS


def run_analysis():
    # Paths
    met_path = os.path.join(BODY_SYSTEMS, "metabolites_annotated.csv")
    prot_path = os.path.join(BODY_SYSTEMS, "proteomics.csv")
    age_path = os.path.join(BODY_SYSTEMS, "Age_Gender_BMI.csv")

    print("Loading data...")
    df_met = pd.read_csv(met_path, index_col=[0, 1])
    df_prot = pd.read_csv(prot_path, index_col=[0, 1])
    df_age = pd.read_csv(age_path, index_col=[0, 1])

    # Age filter 40-70
    age_mask = (df_age["age"] >= 40) & (df_age["age"] <= 70)
    valid_ids = df_age.index[age_mask].get_level_values(1).unique()

    # Protein targets
    targets = ["SIRT1", "SIRT2", "TNF", "IL6"]
    # Ensure targets exist in df_prot
    targets = [t for t in targets if t in df_prot.columns]

    print(f"Merging data for targets: {targets}...")
    df = df_prot[targets].join(df_met, how="inner")
    df = df[df.index.get_level_values(1).isin(valid_ids)]

    print(f"Final sample size: {len(df)}")

    results = {}
    for target in targets:
        corrs = df.corr(method="spearman")[target].drop(labels=targets, errors="ignore")
        results[target] = corrs.sort_values(ascending=False)

    # Save results to file for inspection
    with open(
        "/home/adamgab/PycharmProjects/LabTools/research/proteomics_metabolomics_connection/correlation_results.txt",
        "w",
    ) as f:
        for target, series in results.items():
            f.write(f"\n=== {target} ===\n")
            f.write("TOP POSITIVE:\n")
            f.write(series.head(20).to_string())
            f.write("\n\nTOP NEGATIVE:\n")
            f.write(series.tail(20).to_string())
            f.write("\n" + "=" * 20 + "\n")

    print("Analysis complete. Results written to correlation_results.txt")


if __name__ == "__main__":
    run_analysis()
