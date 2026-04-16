"""
Evaluation functions and MetricsCollector.

Evaluation functions compute metrics for a single set of predictions:
  - evaluate_regression / evaluate_classification / evaluate_ordinal
  - *_with_gender_split variants: compute metrics separately for male/female/combined
  - average_scores_by_subject_id_research_stage: average predictions per (subject, stage)
  - get_gender_for_index: extract gender values aligned with a MultiIndex

MetricsCollector accumulates results across multiple CV seeds and returns:
  - predictions DataFrame: one column per seed (seed_0, seed_1, ...) + RegistrationCode/research_stage
  - metrics DataFrame: one row per seed with all evaluation metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, balanced_accuracy_score
)


# =============================================================================
# REGRESSION
# =============================================================================

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Pearson r, Spearman r, R², MAE, MSE, RMSE."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) < 2:
        return {'pearson_r': 0.0, 'pearson_pvalue': 1.0, 'spearman_r': 0.0,
                'spearman_pvalue': 1.0, 'r2': 0.0, 'mae': 0.0, 'mse': 0.0, 'rmse': 0.0}
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    return {
        'pearson_r': float(pearson_r), 'pearson_pvalue': float(pearson_p),
        'spearman_r': float(spearman_r), 'spearman_pvalue': float(spearman_p),
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def evaluate_regression_with_gender_split(gender: np.ndarray, y_true: np.ndarray,
                                          y_pred: np.ndarray) -> Dict[str, Any]:
    """Regression metrics for all / male (gender==1) / female (gender!=1) subsets."""
    gender = np.asarray(gender)
    male = gender == 1
    return {
        **{f'male_{k}': v for k, v in evaluate_regression(y_true[male], y_pred[male]).items()},
        **{f'female_{k}': v for k, v in evaluate_regression(y_true[~male], y_pred[~male]).items()},
        **evaluate_regression(y_true, y_pred),
    }


# =============================================================================
# CLASSIFICATION
# =============================================================================

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray = None,
                            y_pred_proba: np.ndarray = None,
                            average: str = 'binary') -> Dict[str, Any]:
    """AUC, accuracy, balanced accuracy, F1, precision, recall."""
    if y_pred is None and y_pred_proba is None:
        raise ValueError("Either y_pred or y_pred_proba must be provided")
    if y_pred is None:
        y_pred = (y_pred_proba > 0.5).astype(int) if y_pred_proba.ndim == 1 else np.argmax(y_pred_proba, axis=1)

    if len(y_true) < 2:
        return {'auc': None, 'accuracy': 0.0, 'balanced_accuracy': 0.0,
                'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    n_classes = len(np.unique(y_true))
    if n_classes > 2 and average == 'binary':
        return {'auc': None, 'accuracy': 0.0, 'balanced_accuracy': 0.0,
                'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    auc_val = None
    if y_pred_proba is not None:
        try:
            proba = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
            auc_val = float(roc_auc_score(y_true, proba) if average == 'binary'
                            else roc_auc_score(y_true, y_pred_proba, average=average, multi_class='ovr'))
        except (ValueError, IndexError):
            pass

    return {
        'auc': auc_val,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }


def evaluate_classification_with_gender_split(gender: np.ndarray, y_true: np.ndarray,
                                              y_pred: np.ndarray = None,
                                              y_pred_proba: np.ndarray = None,
                                              average: str = 'binary') -> Dict[str, Any]:
    gender = np.asarray(gender)
    male = gender == 1
    return {
        **{f'male_{k}': v for k, v in evaluate_classification(
            y_true[male], y_pred[male] if y_pred is not None else None,
            y_pred_proba[male] if y_pred_proba is not None else None, average).items()},
        **{f'female_{k}': v for k, v in evaluate_classification(
            y_true[~male], y_pred[~male] if y_pred is not None else None,
            y_pred_proba[~male] if y_pred_proba is not None else None, average).items()},
        **evaluate_classification(y_true, y_pred, y_pred_proba, average),
    }


# =============================================================================
# ORDINAL
# =============================================================================

def evaluate_ordinal(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Spearman correlation for ordinal targets."""
    if len(y_true) < 2:
        return {'spearman_r': 0.0, 'spearman_pvalue': 1.0}
    r, p = spearmanr(y_true, y_pred)
    return {'spearman_r': float(r), 'spearman_pvalue': float(p)}


def evaluate_ordinal_with_gender_split(gender: np.ndarray, y_true: np.ndarray,
                                       y_pred: np.ndarray) -> Dict[str, Any]:
    gender = np.asarray(gender)
    male = gender == 1
    return {
        **{f'male_{k}': v for k, v in evaluate_ordinal(y_true[male], y_pred[male]).items()},
        **{f'female_{k}': v for k, v in evaluate_ordinal(y_true[~male], y_pred[~male]).items()},
        **evaluate_ordinal(y_true, y_pred),
    }


# =============================================================================
# UTILITIES
# =============================================================================

def average_scores_by_subject_id_research_stage(
        all_subject_ids: pd.MultiIndex, y: np.ndarray,
        predictions: np.ndarray) -> Tuple[pd.MultiIndex, np.ndarray, np.ndarray]:
    """Average y and predictions over duplicate (RegistrationCode, research_stage) pairs."""
    df = pd.DataFrame({'y': y.flatten()}, index=all_subject_ids)
    is_2d = predictions.ndim == 2
    if is_2d:
        for i in range(predictions.shape[1]):
            df[f'pred_{i}'] = predictions[:, i]
    else:
        df['pred'] = predictions

    grouped = df.groupby(level=[0, 1]).mean()
    y_avg = grouped['y'].values
    if is_2d:
        pred_cols = [c for c in grouped.columns if c.startswith('pred_')]
        predictions_avg = grouped[pred_cols].values
    else:
        predictions_avg = grouped['pred'].values

    return grouped.index, y_avg, predictions_avg


def get_gender_for_index(x: pd.DataFrame, index: pd.MultiIndex) -> np.ndarray:
    """Extract gender values from x aligned to the given MultiIndex."""
    return x.groupby(level=[0, 1])['gender'].first().loc[index].values


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """
    Accumulates predictions and metrics across multiple CV seeds.

    Usage:
        collector = MetricsCollector(seeds=range(10), model_key='LR_ridge')
        for seed, fold in enumerate(folds):
            # ... run cross-validation ...
            collector.add_seed_results(seed=seed, predictions=..., id_research_pairs=..., metrics=...)
        results = collector.get_results()
        # results['predictions']: DataFrame with seed_0, seed_1, ... columns
        # results['metrics']:     DataFrame with one row per seed
    """

    def __init__(self, seeds: range, model_key: str = None):
        self.seeds = list(seeds)
        self.model_key = model_key
        self.all_predictions = {"id_research_pairs": None}
        self.all_metrics = {}
        self._first_seed = True

    def add_seed_results(self, seed: int, predictions: np.ndarray,
                         id_research_pairs: pd.MultiIndex,
                         metrics: Dict[str, float],
                         metrics_male: Dict[str, float] = None,
                         metrics_female: Dict[str, float] = None,
                         true_values: np.ndarray = None) -> None:
        if metrics_male is None:
            metrics_male = {}
        if metrics_female is None:
            metrics_female = {}

        self.all_predictions[f"seed_{seed}"] = predictions

        if self.all_predictions["id_research_pairs"] is None:
            self.all_predictions["id_research_pairs"] = id_research_pairs
        elif not np.all(self.all_predictions["id_research_pairs"] == id_research_pairs):
            raise ValueError(f"Seed {seed}: id_research_pairs are not consistent across seeds")

        if true_values is not None and "true_values" not in self.all_predictions:
            self.all_predictions["true_values"] = true_values

        combined = {**metrics, **metrics_male, **metrics_female}
        if self._first_seed:
            self.all_metrics = {k: [v] for k, v in combined.items()}
            self._first_seed = False
        else:
            for key, value in combined.items():
                if key not in self.all_metrics:
                    raise KeyError(f"Seed {seed}: Metric '{key}' not in first seed results")
                self.all_metrics[key].append(value)

    def get_results(self) -> Dict[str, pd.DataFrame]:
        """Return {'predictions': DataFrame, 'metrics': DataFrame}."""
        if self._first_seed:
            raise ValueError("No results have been added yet")

        predictions_flat = {}
        id_research_pairs = None
        for key, val in self.all_predictions.items():
            if key == 'id_research_pairs':
                id_research_pairs = val
            elif isinstance(val, np.ndarray) and val.ndim == 2:
                predictions_flat[key] = val[:, 1] if val.shape[1] == 2 else val[:, 0]
            else:
                predictions_flat[key] = val

        preds_df = pd.DataFrame(predictions_flat)
        if id_research_pairs is not None and hasattr(id_research_pairs, 'get_level_values'):
            preds_df.insert(0, 'RegistrationCode', id_research_pairs.get_level_values(0))
            preds_df.insert(1, 'research_stage', id_research_pairs.get_level_values(1))

        metrics_df = pd.DataFrame({
            'seed': self.seeds,
            'model_key': [self.model_key] * len(self.seeds),
            **self.all_metrics
        })
        return {'predictions': preds_df, 'metrics': metrics_df}

    def reset(self) -> None:
        self.all_predictions = {"id_research_pairs": None}
        self.all_metrics = {}
        self._first_seed = True
