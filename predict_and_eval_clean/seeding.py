"""
Seeding function for running cross-validation across multiple seeds.

The seeding function runs cross-validation across multiple random seeds and collects
all predictions and metrics using the MetricsCollector class. This ensures consistent
evaluation across different random initializations.
"""

import os
import platform
import warnings
import logging
import signal
import functools
import pandas as pd
import numpy as np
import joblib
from .ids_folds import save_folds, id_fold_with_stratified_threshold
from .evaluation import MetricsCollector
from .Regressions import Regressions

_IS_UNIX = platform.system() != 'Windows'


def timeout(seconds):
    """Decorator that raises TimeoutError if the function runs longer than `seconds`.

    Uses SIGALRM on Unix. No-op on Windows (SIGALRM unavailable).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _IS_UNIX:
                return func(*args, **kwargs)

            def handler(signum, frame):
                raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds")

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
        return wrapper
    return decorator

# Suppress LightGBM warnings (parameter conflicts etc.)
logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')


def get_completed_seeds(save_dir: str) -> set:
    """Load existing metrics.csv and return set of completed seed indices."""
    metrics_path = os.path.join(save_dir, 'metrics.csv')
    if not os.path.exists(metrics_path):
        return set()

    metrics_df = pd.read_csv(metrics_path)
    # Check if was skipped (empty or has 'skipped' column)
    if metrics_df.empty or 'skipped' in metrics_df.columns:
        return set()
    if 'seed' not in metrics_df.columns:
        return set()

    return set(metrics_df['seed'].unique())


def load_existing_results(save_dir: str) -> dict:
    """Load existing predictions and metrics from save_dir."""
    predictions_path = os.path.join(save_dir, 'predictions.csv')
    metrics_path = os.path.join(save_dir, 'metrics.csv')

    if not os.path.exists(predictions_path) or not os.path.exists(metrics_path):
        return None

    predictions_df = pd.read_csv(predictions_path)
    metrics_df = pd.read_csv(metrics_path)

    # Check if was skipped
    if predictions_df.empty or 'skipped' in metrics_df.columns:
        return None

    return {'predictions': predictions_df, 'metrics': metrics_df}


@timeout(48 * 60 * 60)  # 48 hour timeout for the seeding function
def seeding(x: pd.DataFrame,
            y: pd.DataFrame,
            folds: list,
            model_key: str = 'all',
            average_by_subject_id: bool = True,
            gender_split_evaluation: bool = True,
            save_dir: str = None,
            params: dict = None,
            testing: bool = False,
            use_lgbm: bool = False,
            num_threads: int = None,
            resume_seeds: bool = False) -> dict:
    """
    Perform cross-validation across multiple seeds and collect metrics.

    Label type (regression/categorical/ordinal) is auto-detected from y column name and values.

    Args:
        x: Feature DataFrame with MultiIndex (RegistrationCode, research_stage)
        y: Target DataFrame
        folds: List of fold definitions [seed_index][fold_index] = (train_ids, test_ids)
        model_key: Model type to use ('all' to tune across all models, or specific model name)
        average_by_subject_id: Whether to average predictions by (RegistrationCode, research_stage)
        gender_split_evaluation: Whether to evaluate separately by gender
        params: Model hyperparameters (if None, will be tuned)
        testing: If True, use preset params from model_params.py (skips tuning for faster testing)
        use_lgbm: If True, include LGBM models; otherwise use linear models only
        num_threads: Number of threads for model training
        resume_seeds: If True, load existing results and skip completed seeds

    Returns:
        Dictionary with 'predictions' and 'metrics' DataFrames.
    """
    if gender_split_evaluation and 'gender' not in x.columns:
        gender_split_evaluation = False

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Check for completed seeds if resuming
    completed_seeds = set()
    existing_results = None
    if resume_seeds and save_dir is not None:
        completed_seeds = get_completed_seeds(save_dir)
        if completed_seeds:
            existing_results = load_existing_results(save_dir)
            print(f"Resuming: found {len(completed_seeds)} completed seeds: {sorted(completed_seeds)}")

    # If all seeds done, return existing results
    all_seeds = set(range(len(folds)))
    if completed_seeds == all_seeds and existing_results is not None:
        print(f"All {len(folds)} seeds already completed, returning existing results.")
        return existing_results

    regressions = Regressions(num_threads=num_threads)
    if not use_lgbm:
        regressions.reg_models = ["LR_ridge"]
        regressions.clas_models = ['Logit']
    metrics_collector = None
    example_model = None
    example_model_key = None
    example_seed = None

    for fold_index, fold in enumerate(folds):
        # Skip if already completed
        if fold_index in completed_seeds:
            print(f"Seed {fold_index} already completed, skipping.")
            continue

        # Run nested cross-validation (label_type auto-detected from y)
        payload = regressions.nested_cross_validate(x=x, y=y, cv_folds=fold, model_key=model_key, testing=testing)
        if example_model is None:
            example_model = payload['model']
            example_model_key = payload['model_key']
            example_seed = fold_index
        # Handle skipped targets (e.g., single class)
        if payload.get('skipped'):
            print(f"Skipping target: {payload.get('skip_reason', 'unknown reason')}")
            if save_dir is not None:
                pd.DataFrame().to_csv(f"{save_dir}/predictions.csv")
                pd.DataFrame({'skipped': [True], 'reason': [payload.get('skip_reason')]}).to_csv(f"{save_dir}/metrics.csv")
            return {'predictions': pd.DataFrame(), 'metrics': pd.DataFrame(), 'skipped': True}

        # Initialize collector on first successful fold (now we know model_key)
        if metrics_collector is None:
            metrics_collector = MetricsCollector(seeds=range(len(folds)), model_key=payload['model_key'])

        # Evaluate predictions (use label_type from payload to ensure consistency)
        evaluate_payload = regressions.evaluate_predictions(
            x=x,
            y=y,
            predictions=payload['predictions'],
            gender_split_evaluation=gender_split_evaluation,
            average_by_subject_id=average_by_subject_id,
            label_type=payload['label_type'])

        # Collect results
        metrics_collector.add_seed_results(
            seed=fold_index,
            predictions=evaluate_payload['predictions'],
            id_research_pairs=evaluate_payload['id_research_pairs'],
            metrics=evaluate_payload['metrics'],
            metrics_male=evaluate_payload.get('metrics_male', {}),
            metrics_female=evaluate_payload.get('metrics_female', {}),
            true_values=evaluate_payload.get('true_values'),
        )

    # Get new results (only if we ran any new seeds)
    if metrics_collector is not None:
        results = metrics_collector.get_results()
    else:
        # No new seeds run, return existing
        return existing_results if existing_results else {'predictions': pd.DataFrame(), 'metrics': pd.DataFrame()}

    # Merge with existing results if resuming
    if existing_results is not None and completed_seeds:
        results = _merge_seeding_results(existing_results, results)
        print(f"Merged results: now have {len(results['metrics'])} seeds total.")

    if save_dir is not None:
        results['predictions'].to_csv(f"{save_dir}/predictions.csv")
        results['metrics'].to_csv(f"{save_dir}/metrics.csv")
        save_folds(folds, save_dir)
        if example_model is not None:
            joblib.dump(example_model, os.path.join(save_dir, f'example_model_seed_{example_seed}.pkl'))

    return results


def _merge_seeding_results(existing: dict, new: dict) -> dict:
    """Merge existing and new seeding results."""
    existing_preds = existing['predictions']
    new_preds = new['predictions']

    existing_seed_cols = [c for c in existing_preds.columns if c.startswith('seed_')]
    new_seed_cols = [c for c in new_preds.columns if c.startswith('seed_')]

    # Start with existing, add new seed columns
    merged_preds = existing_preds.copy()
    for col in new_seed_cols:
        if col not in merged_preds.columns:
            merged_preds[col] = new_preds[col]

    # Merge metrics - concat and drop duplicates on seed
    merged_metrics = pd.concat([existing['metrics'], new['metrics']], ignore_index=True)
    merged_metrics = merged_metrics.drop_duplicates(subset=['seed'], keep='last')
    merged_metrics = merged_metrics.sort_values('seed').reset_index(drop=True)

    return {'predictions': merged_preds, 'metrics': merged_metrics}
