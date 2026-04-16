import pandas as pd
import numpy as np
import os
from body_system_loader.load_feature_df import BODY_SYSTEMS

def run_deep_analysis():
    met_path = os.path.join(BODY_SYSTEMS, 'metabolites_annotated.csv')
    prot_path = os.path.join(BODY_SYSTEMS, 'proteomics.csv')
    age_path = os.path.join(BODY_SYSTEMS, 'Age_Gender_BMI.csv')

    df_met = pd.read_csv(met_path, index_col=[0, 1])
    df_prot = pd.read_csv(prot_path, index_col=[0, 1])
    df_age = pd.read_csv(age_path, index_col=[0, 1])

    age_mask = (df_age['age'] >= 40) & (df_age['age'] <= 70)
    valid_ids = df_age.index[age_mask].get_level_values(1).unique()

    targets = ['SIRT1', 'SIRT2', 'TNF', 'IL6']
    targets = [t for t in targets if t in df_prot.columns]
    
    df = df_prot[targets].join(df_met, how='inner')
    df = df[df.index.get_level_values(1).isin(valid_ids)]

    # Focus on the SIRT2-Dipeptide connection
    # Phenylalanylphenylalanine and Phenylalanylthreonine showed strong positive correlation with SIRT2
    dipeptides = ['Phenylalanylphenylalanine', 'Phenylalanylthreonine']
    
    # Calculate correlation specifically for these
    sirt2_dipep_corr = df[['SIRT2'] + dipeptides].corr(method='spearman')
    
    # Check if SIRT2 is negatively correlated with inflammation markers (TNF, IL6) in the same subjects
    inflammation_corr = df[['SIRT2', 'TNF', 'IL6']].corr(method='spearman')

    print("--- SIRT2 vs Dipeptides ---")
    print(sirt2_dipep_corr['SIRT2'])
    print("\n--- SIRT2 vs Inflammation ---")
    print(inflammation_corr['SIRT2'])

if __name__ == "__main__":
    run_deep_analysis()
EOF
python3 /home/adamgab/PycharmProjects/LabTools/research/proteomics_metabolomics_connection/findings_analysis.py

