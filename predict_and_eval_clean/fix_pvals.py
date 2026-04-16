import os
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests


def fix_pvals(dirs):
    """
    Collect all p-value columns from metrics.csv files across directories,
    apply FDR correction (Benjamini-Hochberg) separately for each
    (feature_system, pvalue_column) combination, and write corrected columns
    back to the same metrics.csv files.

    FDR is applied per:
    - feature_system: each feature/run system is a separate hypothesis family
    - pvalue_column: implicitly separates by gender (all, male, female via column name)

    Args:
        dirs: List of directory paths containing metrics.csv files.
              Expected path format: .../target_system/label_name/feature_name/metrics.csv
    """
    pvalue_data = []

    for dir_path in dirs:
        metrics_file = os.path.join(dir_path, 'metrics.csv')
        if not os.path.exists(metrics_file):
            continue

        feature_system = os.path.basename(dir_path)
        df = pd.read_csv(metrics_file)

        pvalue_cols = [col for col in df.columns if 'pvalue' in col.lower() and 'fdr' not in col.lower()]

        for col in pvalue_cols:
            for idx, row in df.iterrows():
                pvalue_data.append({
                    'dir': dir_path,
                    'row_idx': idx,
                    'feature_system': feature_system,
                    'pvalue_col': col,
                    'pvalue': row[col]
                })

    if not pvalue_data:
        return

    pvalue_df = pd.DataFrame(pvalue_data)
    dir_updates = {}

    for (feature_system, pvalue_col), group in pvalue_df.groupby(['feature_system', 'pvalue_col']):
        pvals = group['pvalue'].values
        valid_mask = ~np.isnan(pvals)

        if not valid_mask.any():
            continue

        corrected = np.full(len(pvals), np.nan)
        _, corrected[valid_mask], _, _ = multipletests(pvals[valid_mask], method='fdr_bh')

        fdr_col_name = pvalue_col.replace('pvalue', 'pvalue_fdr')

        for i, (_, item) in enumerate(group.iterrows()):
            dir_path = item['dir']
            row_idx = item['row_idx']

            if dir_path not in dir_updates:
                dir_updates[dir_path] = {}
            if fdr_col_name not in dir_updates[dir_path]:
                dir_updates[dir_path][fdr_col_name] = {}

            dir_updates[dir_path][fdr_col_name][row_idx] = corrected[i]

    for dir_path, fdr_columns in dir_updates.items():
        metrics_file = os.path.join(dir_path, 'metrics.csv')
        df = pd.read_csv(metrics_file)

        for fdr_col_name, row_values in fdr_columns.items():
            df[fdr_col_name] = [row_values.get(i, np.nan) for i in range(len(df))]

        df.to_csv(metrics_file, index=False)
