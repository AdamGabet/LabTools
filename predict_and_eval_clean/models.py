"""
Model definitions, hyperparameter presets, and sklearn pipelines.

Two public classes:
  - FIXED_PARAM_PRESETS / TUNABLE_PARAM_RANGES: dicts with default/tunable hyperparameters per model type
  - ModelAndPipeline: factory for creating sklearn Pipelines (preprocessing + model)

Supported model types:
  Regression:     LR_ridge, LR_lasso, LR_elastic, LGBM_regression
  Classification: Logit, LGBM_classifier
  Ordinal:        Ordinal_logit
"""

import logging
import json
import os
from typing import Dict, Any, List

import numpy as np
import joblib
import lightgbm as lgb
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from statsmodels.miscmodels.ordinal_model import OrderedModel

logging.getLogger('lightgbm').setLevel(logging.ERROR)


# =============================================================================
# HYPERPARAMETER PRESETS
# =============================================================================

# Fixed presets — used by default (no tuning)
FIXED_PARAM_PRESETS = {
    # LGBM regressor — tuned for ~800K rows, ~1000 features
    "LGBM_regression": dict(
        objective="regression",
        min_child_weight=16000,  # ~0.02 × 800K - prevents overfitting on large data
        max_depth=3,
        n_estimators=200,
        num_leaves=300,
        feature_fraction=0.15,
        bagging_fraction=0.5,
        reg_alpha=0.5,
        reg_lambda=0.5,
        n_jobs=1,
        verbosity=-1,
    ),
    "LGBM_classifier": dict(
        objective="binary",
        min_child_weight=16000,
        n_estimators=200,
        max_depth=3,
        subsample=0.5,
        colsample_bytree=0.1,
        n_jobs=1,
        verbosity=-1,
    ),
    "LR_ridge": dict(alpha=1.0),
    "LR_lasso": dict(alpha=1.0),
    "LR_elastic": dict(alpha=11.0, l1_ratio=0.2, max_iter=10000),
    "Logit": dict(C=1.0, penalty='l2', max_iter=10000, solver='lbfgs'),
    "Ordinal_logit": dict(distr='logit'),
}

# Tunable ranges for RandomizedSearchCV (not used in default pipeline, available for custom use)
TUNABLE_PARAM_RANGES = {
    "LGBM_regression": dict(
        min_child_weight=[0.01*4000, 0.02*4000, 0.03*4000],
        max_depth=[3, 4],
        n_estimators=[100, 200],
        num_leaves=[100, 300],
        feature_fraction=[0.1, 0.15, 0.2],
        reg_alpha=[0.2, 0.5, 1.0],
        reg_lambda=[0.2, 0.5, 1.0],
    ),
    "LGBM_classifier": dict(
        min_child_weight=[0.01*4000, 0.02*4000, 0.03*4000],
        n_estimators=[100, 200],
        max_depth=[3, 4],
        subsample=[0.3, 0.5, 0.8],
        colsample_bytree=[0.1, 0.2],
    ),
    "LR_ridge": dict(alpha=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
    "LR_lasso": dict(alpha=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
    "LR_elastic": dict(alpha=[0.1, 1.0, 5.0, 10.0, 50.0], l1_ratio=[0.1, 0.5, 0.9], max_iter=[2000]),
    "Logit": dict(C=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], penalty=['l2'], solver=['lbfgs'],
                  max_iter=[1000], tol=[1e-3]),
}


# =============================================================================
# ORDINAL MODEL WRAPPER
# =============================================================================

class OrdinalModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for statsmodels OrderedModel.

    Fits an ordered logit (proportional odds) model and predicts the expected
    ordinal value as a weighted sum of class probabilities.
    """
    def __init__(self, distr: str = 'logit', maxiter: int = 1000):
        self.distr = distr
        self.maxiter = maxiter
        self.model_ = None
        self.result_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        self.model_ = OrderedModel(y, X, distr=self.distr, hasconst=False)
        self.result_ = self.model_.fit(method='lbfgs', disp=False, maxiter=self.maxiter)
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.sum(proba * self.classes_, axis=1)

    def predict_proba(self, X):
        return self.result_.predict(X)

    def get_params(self, deep=True):
        return {'distr': self.distr, 'maxiter': self.maxiter}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# =============================================================================
# MODEL + PIPELINE FACTORY
# =============================================================================

PARAMS_DIR = os.getenv('LINEAR_PARAM_DIR', './params/')
STATIC_PARAMS_FILE = os.path.join(PARAMS_DIR, 'static_params.json')
DYNAMIC_PARAMS_FILE = os.path.join(PARAMS_DIR, 'dynamic_params.json')


class ModelAndPipeline:
    """Factory for creating sklearn estimators and Pipelines."""

    @staticmethod
    def initialize_model(model_type: str, n_jobs: int = 8, params: Dict[str, Any] = None):
        model_map = {
            'LR_ridge': Ridge,
            'LR_lasso': Lasso,
            'LR_elastic': ElasticNet,
            'Logit': LogisticRegression,
        }
        if model_type in model_map:
            model = model_map[model_type]()
        elif model_type == 'LGBM_regression':
            model = lgb.LGBMRegressor(n_jobs=n_jobs, verbose=-1, objective="regression")
        elif model_type == 'LGBM_classifier':
            model = lgb.LGBMClassifier(n_jobs=n_jobs, verbose=-1, objective="binary")
        elif model_type == 'Ordinal_logit':
            model = OrdinalModelWrapper(distr='logit')
        else:
            raise ValueError(f"Model type '{model_type}' not found")

        if params is not None:
            model = model.set_params(**params)
        return model

    @staticmethod
    def initialize_model_and_pipeline(model_type: str, n_jobs: int = 8,
                                      params: Dict[str, Any] = None,
                                      categorical_cols: List[str] = None,
                                      numeric_cols: List[str] = None,
                                      model_path: str = None) -> Pipeline:
        """
        Create a sklearn Pipeline: [preprocessor → variance_threshold → model].

        Tree models (LGBM, XGB): OrdinalEncoder for categoricals, passthrough for numerics.
        Linear/Ordinal models: OneHotEncoder + imputation + scaling.
        """
        model = joblib.load(model_path) if model_path else ModelAndPipeline.initialize_model(model_type, n_jobs, params)
        is_tree = 'LGBM' in model_type

        if is_tree:
            if categorical_cols is None:
                preprocessor = 'passthrough'
            else:
                preprocessor = ColumnTransformer([
                    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
                    ('num', 'passthrough', numeric_cols)
                ], remainder='drop')
        else:
            if categorical_cols is None:
                preprocessor = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
            else:
                preprocessor = ColumnTransformer([
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
                    ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric_cols)
                ], remainder='drop')

        return Pipeline([
            ('preprocessor', preprocessor),
            ('drop_constant', VarianceThreshold(threshold=0.0)),
            ('model', model)
        ])

    @staticmethod
    def get_model_type_static_params(model_type: str, x) -> Dict[str, Any]:
        """Get fixed params, scaling min_child_weight to actual sample size."""
        params = ModelAndPipeline._load_params(STATIC_PARAMS_FILE, 'static')
        if model_type not in params:
            raise KeyError(f"Model type '{model_type}' not found in static params")
        model_params = params[model_type].copy()

        n_samples = len(x)
        if model_type in ('LGBM_regression', 'LGBM_classifier'):
            if 'min_child_weight' in model_params:
                model_params['min_child_weight'] = int(0.02 * n_samples)

        if n_samples > 15000 and model_type in ('LR_ridge', 'LR_lasso'):
            model_params['solver'] = 'lsqr'

        return model_params

    @staticmethod
    def get_model_type_dynamic_params(model_type: str, x) -> Dict[str, List[Any]]:
        """Get tunable param ranges, scaling min_child_weight to actual sample size."""
        params = ModelAndPipeline._load_params(DYNAMIC_PARAMS_FILE, 'dynamic')
        if model_type not in params:
            raise KeyError(f"Model type '{model_type}' not found in dynamic params")
        model_params = params[model_type].copy()

        n_samples = len(x)
        if model_type in ('LGBM_regression', 'LGBM_classifier'):
            if 'min_child_weight' in model_params:
                model_params['min_child_weight'] = [int(r * n_samples) for r in (0.01, 0.02, 0.03)]

        if n_samples > 50000 and model_type in ('LR_ridge', 'LR_lasso'):
            model_params['solver'] = ['lsqr']

        return model_params

    @staticmethod
    def _load_params(json_path: str, type_: str) -> Dict[str, Any]:
        """Load params from JSON file, falling back to hardcoded presets."""
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return FIXED_PARAM_PRESETS if type_ == 'static' else TUNABLE_PARAM_RANGES

    @staticmethod
    def create_model_params_json(params: Dict[str, Dict[str, Any]], path: str) -> str:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)
        return path
