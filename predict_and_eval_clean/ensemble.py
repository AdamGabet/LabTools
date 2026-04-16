"""
Ensemble predictions from multiple feature systems.

For each (target_system, label) in save_dir, this module:
  1. Loads predictions.csv from each feature system sub-folder
  2. Averages seed columns across feature systems on common subjects
  3. Evaluates the averaged predictions and saves to an 'ensemble' sub-folder

Runnable standalone:
    python -m predict_and_eval_clean.ensemble --save_dir /path/to/results
    python -m predict_and_eval_clean.ensemble --save_dir /path/to/results --systems sys_a sys_b
    python -m predict_and_eval_clean.ensemble --save_dir /path/to/results --skip baseline

Or from BuildResults via config:
    config.ensemble_after_run = True
    config.ensemble_systems = ['system_a', 'system_b']   # None = all
    config.ensemble_skip_systems = ['baseline']           # None = none
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Optional

from .evaluation import evaluate_regression, evaluate_classification, evaluate_ordinal


# =============================================================================
# Internal helpers
# =============================================================================

def _infer_label_type_from_metrics(metrics_path: str) -> str:
    """Infer label type (regression/categorical/ordinal) from model_key in metrics.csv."""
    try:
        df = pd.read_csv(metrics_path)
        if 'model_key' not in df.columns or df.empty:
            return 'regression'
        model_key = str(df['model_key'].iloc[0]).lower()
        if 'ordinal' in model_key:
            return 'ordinal'
        if 'classifier' in model_key or 'logit' in model_key:
            return 'categorical'
    except Exception:
        pass
    return 'regression'


def _load_predictions(dir_path: str) -> Optional[pd.DataFrame]:
    """
    Load predictions.csv from dir_path.

    Returns None if the file is missing, empty (skipped run), or has no seed columns.
    Expected columns: RegistrationCode, research_stage, true_values, seed_0, seed_1, ...
    """
    pred_path = os.path.join(dir_path, 'predictions.csv')
    if not os.path.exists(pred_path):
        return None
    try:
        df = pd.read_csv(pred_path)
    except Exception:
        return None
    if df.empty or 'RegistrationCode' not in df.columns:
        return None
    if not any(c.startswith('seed_') for c in df.columns):
        return None
    return df


def _compute_metrics(true_values: np.ndarray, avg_preds: np.ndarray,
                     label_type: str) -> dict:
    """Compute evaluation metrics for one seed of ensemble predictions."""
    true_values = np.asarray(true_values)
    avg_preds = np.asarray(avg_preds)

    if label_type == 'regression':
        return evaluate_regression(true_values, avg_preds)
    elif label_type == 'categorical':
        # avg_preds are averaged probabilities; compute AUC from proba, accuracy from threshold
        y_pred = (avg_preds > 0.5).astype(int)
        return evaluate_classification(
            true_values.astype(int), y_pred=y_pred, y_pred_proba=avg_preds
        )
    elif label_type == 'ordinal':
        return evaluate_ordinal(true_values, avg_preds)
    return {}


# =============================================================================
# Core ensemble logic
# =============================================================================

def _ensemble_one_label(
    label_dir: str,
    feature_systems: Optional[List[str]],
    skip_systems: set,
    ensemble_name: str,
    target_system: str,
    label: str,
) -> bool:
    """
    Ensemble predictions for a single (target_system, label) directory.

    Returns True if ensemble was computed and saved, False if skipped.
    """
    # Collect valid feature system directories
    candidates = []
    for fs_name in sorted(os.listdir(label_dir)):
        if fs_name in skip_systems:
            continue
        if feature_systems is not None and fs_name not in feature_systems:
            continue
        fs_dir = os.path.join(label_dir, fs_name)
        if not os.path.isdir(fs_dir):
            continue
        df = _load_predictions(fs_dir)
        if df is not None:
            candidates.append((fs_name, fs_dir, df))

    if len(candidates) < 2:
        print(f"  Skipping {target_system}/{label}: "
              f"need ≥2 feature systems, found {len(candidates)}")
        return False

    # Intersection of (RegistrationCode, research_stage) across all systems
    index_sets = [
        set(zip(df['RegistrationCode'], df['research_stage']))
        for _, _, df in candidates
    ]
    common_index_set = index_sets[0].intersection(*index_sets[1:])
    if not common_index_set:
        print(f"  Skipping {target_system}/{label}: no common subjects")
        return False

    common_idx = pd.MultiIndex.from_tuples(
        sorted(common_index_set), names=['RegistrationCode', 'research_stage']
    )

    # Intersection of seed columns
    seed_col_sets = [
        {c for c in df.columns if c.startswith('seed_')}
        for _, _, df in candidates
    ]
    common_seeds = sorted(
        seed_col_sets[0].intersection(*seed_col_sets[1:]),
        key=lambda s: int(s.split('_')[1])
    )
    if not common_seeds:
        print(f"  Skipping {target_system}/{label}: no common seed columns")
        return False

    # Align DataFrames to common index
    aligned_arrays = []
    true_values = None

    for fs_name, _, df in candidates:
        df = df.set_index(['RegistrationCode', 'research_stage'])
        df = df.loc[common_idx]
        if true_values is None and 'true_values' in df.columns:
            true_values = df['true_values'].values
        aligned_arrays.append(df[common_seeds].values)

    if true_values is None:
        print(f"  Skipping {target_system}/{label}: no true_values column")
        return False

    # Average across feature systems: shape (n_systems, n_subjects, n_seeds)
    avg_preds = np.stack(aligned_arrays, axis=0).mean(axis=0)

    # Build output predictions DataFrame
    out_df = pd.DataFrame(avg_preds, index=common_idx, columns=common_seeds)
    out_df.insert(0, 'true_values', true_values)
    out_df = out_df.reset_index()

    # Infer label type from the first feature system's metrics.csv
    first_metrics_path = os.path.join(candidates[0][1], 'metrics.csv')
    label_type = _infer_label_type_from_metrics(first_metrics_path)

    # Compute per-seed metrics
    systems_str = ','.join(fs for fs, _, _ in candidates)
    model_key = f'ensemble_{len(candidates)}_systems'
    metrics_rows = []
    for i, seed_col in enumerate(common_seeds):
        seed_idx = int(seed_col.split('_')[1])
        m = _compute_metrics(true_values, avg_preds[:, i], label_type)
        m['seed'] = seed_idx
        m['model_key'] = model_key
        m['n_systems'] = len(candidates)
        m['systems'] = systems_str
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)

    # Save
    out_dir = os.path.join(label_dir, ensemble_name)
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
    metrics_df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)

    print(f"  Ensemble {target_system}/{label}: "
          f"{len(candidates)} systems, {len(common_index_set)} subjects, "
          f"{len(common_seeds)} seeds → {out_dir}")
    return True


def ensemble_predictions(
    save_dir: str,
    feature_systems: List[str] = None,
    skip_systems: List[str] = None,
    ensemble_name: str = 'ensemble',
) -> None:
    """
    Ensemble predictions from multiple feature systems.

    Walks save_dir and for each (target_system, label) sub-directory,
    averages seed predictions across feature system folders, then saves
    ensemble/predictions.csv and ensemble/metrics.csv.

    Args:
        save_dir: Root results directory (BuildResultsConfig.save_dir).
        feature_systems: Feature system folder names to include.
                         If None, all valid folders are used.
        skip_systems: Folder names to exclude (e.g. ['baseline']).
                      The ensemble output folder is always excluded automatically.
        ensemble_name: Name for the ensemble output sub-folder (default: 'ensemble').
    """
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"save_dir not found: {save_dir}")

    excluded = set(skip_systems or [])
    excluded.add(ensemble_name)

    n_computed = 0
    n_skipped = 0

    for target_system in sorted(os.listdir(save_dir)):
        target_dir = os.path.join(save_dir, target_system)
        if not os.path.isdir(target_dir):
            continue

        for label in sorted(os.listdir(target_dir)):
            label_dir = os.path.join(target_dir, label)
            if not os.path.isdir(label_dir):
                continue

            ok = _ensemble_one_label(
                label_dir=label_dir,
                feature_systems=feature_systems,
                skip_systems=excluded,
                ensemble_name=ensemble_name,
                target_system=target_system,
                label=label,
            )
            if ok:
                n_computed += 1
            else:
                n_skipped += 1

    print(f"\nEnsemble done: {n_computed} computed, {n_skipped} skipped.")


# =============================================================================
# Standalone CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ensemble predictions from multiple feature systems.'
    )
    parser.add_argument(
        '--save_dir', required=True,
        help='Root results directory (BuildResultsConfig.save_dir)'
    )
    parser.add_argument(
        '--systems', nargs='+', default=None,
        help='Feature system folder names to include (default: all)'
    )
    parser.add_argument(
        '--skip', nargs='+', default=None,
        help='Feature system folder names to exclude (e.g. baseline)'
    )
    parser.add_argument(
        '--name', default='ensemble',
        help='Output folder name inside each label directory (default: ensemble)'
    )
    args = parser.parse_args()

    ensemble_predictions(
        save_dir=args.save_dir,
        feature_systems=args.systems,
        skip_systems=args.skip,
        ensemble_name=args.name,
    )
