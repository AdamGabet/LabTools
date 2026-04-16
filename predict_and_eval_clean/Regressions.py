# Nested Cross-Validation with Proper Parallelization
# No data leakage: hyperparameters tuned inside each outer fold

import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict

from .models import ModelAndPipeline
from .categorical_utils import CategoricalUtils, get_ordinal_as_regression
from .evaluation import (
    evaluate_regression_with_gender_split,
    evaluate_classification_with_gender_split,
    evaluate_ordinal_with_gender_split,
    evaluate_regression,
    evaluate_classification,
    evaluate_ordinal,
    average_scores_by_subject_id_research_stage,
    get_gender_for_index
)
from .ids_folds import create_cv_folds

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# =============================================================================
# CONFIGURATION
# =============================================================================
N_JOBS_CV = 8        # Parallel folds for tree models
N_THREADS_MODEL = 1  # Per-model threads


class Regressions:
    """
    Nested cross-validation for regression/classification/ordinal prediction.
    
    Key features:
    - No data leakage: hyperparameters tuned inside each outer fold
    - Optimized parallelization for cluster jobs
    - Auto-detection of label type (regression/classification/ordinal)
    """
    
    def __init__(self, num_threads: int = None):
        # Model candidates per label type
        self.reg_models = ["LR_ridge", "LGBM_regression"]
        self.clas_models = ['Logit', 'LGBM_classifier']
        self.ordinal_models = ['Ordinal_logit']
        self.num_threads_cv = num_threads if num_threads is not None else N_JOBS_CV
    # =========================================================================
    # MAIN PUBLIC METHOD
    # =========================================================================
    
    def nested_cross_validate(self,
                              x: pd.DataFrame,
                              y: pd.DataFrame,
                              cv_folds: list,
                              model_key: str = 'all',
                              testing: bool = False) -> dict:
        """
        Nested cross-validation: runs all model types, picks best by primary metric.

        Args:
            x: Feature DataFrame with MultiIndex (RegistrationCode, research_stage)
            y: Target DataFrame
            cv_folds: List of (train_subject_ids, test_subject_ids) tuples
            model_key: 'all' to run all models, or specific model name
            testing: If True, run only first model (fast mode for testing)

        Returns:
            Dict with predictions, model_key, label_type, etc.
        """
        # Detect label type and column types
        y_col = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
        label_type = CategoricalUtils.get_label_type(y_col)
        categorical_cols, numeric_cols = CategoricalUtils.detect_categorical_and_numeric_columns(x)

        # Validate data
        skip_result = self._validate_data(x, y_col, label_type, cv_folds)
        if skip_result:
            return skip_result

        # Convert to sample-level folds
        subject_ids = x.index.get_level_values(0).values
        sample_folds = create_cv_folds(cv_folds, subject_ids)

        # Convert to numpy
        x_arr = x.values
        y_arr = y.values.flatten()

        # Get model candidates
        model_types = self._get_models_for_label_type(label_type, model_key)
        if testing:
            model_types = [model_types[0]]

        # Run nested CV for each model type independently
        all_predictions = {}
        all_params = {}
        all_models = {}
        for model_type in model_types:
            predictions, params, model = self._run_nested_cv_single_model(
                x_arr, y_arr, sample_folds, model_type, label_type,
                categorical_cols, numeric_cols
            )
            all_predictions[model_type] = predictions
            all_params[model_type] = params
            all_models[model_type] = model

        # Evaluate all models and pick the best
        all_metrics = {}
        for model_type, preds in all_predictions.items():
            eval_result = self.evaluate_predictions(
                x, y, preds, gender_split_evaluation=True,
                average_by_subject_id=True, label_type=label_type
            )
            all_metrics[model_type] = eval_result

        best_model_key, best_predictions = self._select_best_by_metrics(
            all_predictions, all_metrics, label_type
        )

        return {
            "all_predictions": all_predictions,
            "all_metrics": all_metrics,
            "model_key": best_model_key,
            "model": all_models.get(best_model_key),
            "predictions": best_predictions,
            "params": all_params,
            "cv_fold": cv_folds,
            "label_type": label_type,
        }
    
    # =========================================================================
    # NESTED CV CORE LOGIC
    # =========================================================================
    
    def _run_nested_cv_single_model(self, x: np.ndarray, y: np.ndarray,
                                     sample_folds: list, model_type: str,
                                     label_type: str, categorical_cols: list,
                                     numeric_cols: list) -> tuple:
        """
        Run nested CV for a single model type using cross_val_predict for parallelization.
        
        Returns:
            (predictions_array, params, fitted_model)
        """
        # Get static params (tuning disabled for speed)
        params = ModelAndPipeline.get_model_type_static_params(model_type, x)
        
        pipe = ModelAndPipeline.initialize_model_and_pipeline(
            model_type, n_jobs=N_THREADS_MODEL, params=params,
            categorical_cols=categorical_cols, numeric_cols=numeric_cols
        )
        
        # Linear models: no parallelization (fast enough, serialization overhead hurts)
        # Tree models: parallelize folds
        is_tree = 'LGBM' in model_type or 'XGB' in model_type
        n_jobs = self.num_threads_cv if is_tree else 1
        
        if label_type == 'categorical':
            predictions = cross_val_predict(
                pipe, x, y, cv=sample_folds, method='predict_proba', n_jobs=n_jobs
            )
        else:
            predictions = cross_val_predict(
                pipe, x, y, cv=sample_folds, n_jobs=n_jobs
            )
        
        # Fit final model on all data for saving
        pipe.fit(x, y)
        
        return predictions, params, pipe
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _validate_data(self, x: pd.DataFrame, y_col: pd.Series,
                       label_type: str, cv_folds: list) -> dict:
        """Validate data and return skip result if invalid, None otherwise."""
        if x.ndim == 2 and x.shape[1] == 0:
            return self._skip_result(cv_folds, label_type, "No input features available (shape[1] == 0)")

        n_unique = y_col.nunique()
        if label_type in ('categorical', 'ordinal') and n_unique < 2:
            return self._skip_result(cv_folds, label_type, f"Only {n_unique} class(es) in target, need at least 2")

        if label_type == 'categorical':
            unique_vals, counts = np.unique(y_col.dropna(), return_counts=True)
            if len(unique_vals) >= 2 and counts.min() < 8:
                return self._skip_result(cv_folds, label_type, f"Minority class n={counts.min()} < 8, too few samples")

        return None

    def _skip_result(self, cv_folds: list, label_type: str, reason: str) -> dict:
        """Return a skip result dict."""
        return {
            "model_key": None,
            "model": None,
            "params": None,
            "predictions": None,
            "cv_fold": cv_folds,
            "label_type": label_type,
            "skipped": True,
            "skip_reason": reason,
        }
    
    def _get_models_for_label_type(self, label_type: str, model_key: str) -> list:
        """Get list of model types to try based on label type."""
        if model_key != 'all':
            return [model_key]
        
        if label_type == 'categorical':
            return self.clas_models
        # we want to use regression models for ordinal
        elif label_type == 'ordinal':
            if get_ordinal_as_regression():
                return self.reg_models
            else:
                return self.ordinal_models
        else:
            return self.reg_models
    
    def _select_best_by_metrics(self, all_predictions: dict, all_metrics: dict, 
                                 label_type: str) -> tuple:
        """
        Select best model based on primary metric.
        
        Primary metrics: pearson_r (regression), roc_auc (classification), spearman_r (ordinal)
        """
        # Pick primary metric based on label type
        if label_type == 'categorical':
            metric_key = 'auc'  # Key from evaluate_classification
        elif label_type == 'ordinal':
            metric_key = 'spearman_r'
        else:
            metric_key = 'pearson_r'
        
        best_model = None
        best_score = -np.inf
        for model_type, eval_result in all_metrics.items():
            score = eval_result['metrics'].get(metric_key, -np.inf)
            # Handle None, NaN, or non-numeric scores
            try:
                score = float(score) if score is not None else -np.inf
                if np.isnan(score):
                    score = -np.inf
            except (TypeError, ValueError):
                score = -np.inf
            if score > best_score:
                best_score = score
                best_model = model_type
        
        # Fallback: if no best model found, use first available
        if best_model is None and all_predictions:
            best_model = list(all_predictions.keys())[0]
        
        return best_model, all_predictions.get(best_model)
    
    # =========================================================================
    # EVALUATION (unchanged from original)
    # =========================================================================
    
    def evaluate_predictions(self,
                             x: pd.DataFrame,
                             y: pd.DataFrame,
                             predictions: np.ndarray,
                             gender_split_evaluation: bool = True, 
                             average_by_subject_id: bool = True,
                             label_type: str = None) -> dict:
        """
        Evaluate predictions using appropriate metrics based on label type.
        
        Args:
            x: Feature DataFrame with index
            y: Target DataFrame
            predictions: Model predictions
            gender_split_evaluation: Whether to split evaluation by gender
            average_by_subject_id: Whether to average by subject
            label_type: Explicit label type. If None, auto-detected.
        """
        # Check if gender column exists when gender_split_evaluation is requested
        if gender_split_evaluation and 'gender' not in x.columns:
            gender_split_evaluation = False
            
        y_col = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
        
        if label_type is None:
            label_type = CategoricalUtils.get_label_type(y_col)

        id_research_pairs = x.index.copy()
        y = y.copy().values

        if average_by_subject_id:
            id_research_pairs, y, predictions = average_scores_by_subject_id_research_stage(
                id_research_pairs, y, predictions
            )

        flat_y = y.flatten()
        
        # Get gender for the (possibly averaged) index
        gender = get_gender_for_index(x, id_research_pairs) if gender_split_evaluation else None

        if label_type == 'categorical':
            if gender_split_evaluation:
                eval_metrics = evaluate_classification_with_gender_split(
                    gender, flat_y, y_pred=None, y_pred_proba=predictions
                )
            else:
                eval_metrics = evaluate_classification(flat_y, y_pred=None, y_pred_proba=predictions)
        elif label_type == 'ordinal' and not get_ordinal_as_regression():
            if gender_split_evaluation:
                eval_metrics = evaluate_ordinal_with_gender_split(gender, flat_y, predictions)
            else:
                eval_metrics = evaluate_ordinal(flat_y, predictions)
        else:
            # this will run for regression and ordinal if ORDINAL_AS_REGRESSION is True
            if gender_split_evaluation:
                eval_metrics = evaluate_regression_with_gender_split(gender, flat_y, predictions)
            else:
                eval_metrics = evaluate_regression(flat_y, predictions)
        
        # Add n_subjects (and n_positive for classification)
        n_subjects = len(np.unique(id_research_pairs.get_level_values(0)))
        eval_metrics['n_subjects'] = n_subjects
        if label_type == 'categorical':
            eval_metrics['n_positive'] = int(np.sum(flat_y == 1))
        
        return {
            "metrics": {k: v for k, v in eval_metrics.items() if "male" not in k},
            "metrics_male": {k: v for k, v in eval_metrics.items() if "male" in k and "female" not in k},
            "metrics_female": {k: v for k, v in eval_metrics.items() if "female" in k},
            "predictions": predictions,
            "id_research_pairs": id_research_pairs,
            "true_values": y
        }