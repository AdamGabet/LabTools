import pandas as pd
from typing import List, Tuple, Optional
from .load_feature_df import load_system_description_json

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

# Module-level cache for column descriptions JSON (read-only after first load, safe).
_COLUMN_TYPES_CACHE = None

# Module-level flag for ordinal-as-regression — set once at run start via set_ordinal_as_regression().
# get_label_type() accepts an explicit override; this global is the fallback.
_ORDINAL_AS_REGRESSION = False


def set_ordinal_as_regression(value: bool):
    """Set the global flag to treat ordinal targets as regression (called from BuildResults)."""
    global _ORDINAL_AS_REGRESSION
    _ORDINAL_AS_REGRESSION = value


def get_ordinal_as_regression() -> bool:
    return _ORDINAL_AS_REGRESSION


def _load_column_descriptions() -> dict:
    """Load column type descriptions from JSON (cached after first call)."""
    global _COLUMN_TYPES_CACHE
    if _COLUMN_TYPES_CACHE is not None:
        return _COLUMN_TYPES_CACHE

    systems_dict = load_system_description_json()
    column_types = {}
    for system, data in systems_dict.items():
        for column, info in data['columns'].items():
            column_types[column] = info['type']
    _COLUMN_TYPES_CACHE = column_types
    return column_types


class CategoricalUtils:
    """Utility functions for detecting and handling categorical columns."""

    @staticmethod
    def get_label_type(
        column: pd.Series,
        column_name: str = None,
        ordinal_as_regression: Optional[bool] = None,
    ) -> Literal["regression", "categorical", "ordinal"]:
        """
        Determine the prediction type for a target column.

        Lookup order:
        1. JSON config (explicit type for known HPP columns)
        2. Heuristic: is_categorical() + ordinal detection

        Args:
            column: The target column data.
            column_name: Column name for JSON lookup (defaults to column.name).
            ordinal_as_regression: If True, return "regression" for ordinal columns.
                                   If None, falls back to the global set by set_ordinal_as_regression().

        Returns:
            "regression", "categorical", or "ordinal".
        """
        use_ordinal_as_reg = _ORDINAL_AS_REGRESSION if ordinal_as_regression is None else ordinal_as_regression
        col_name = column_name or column.name

        col_types = _load_column_descriptions()
        if col_types and col_name in col_types:
            label_type = col_types[col_name]
            if label_type in ("regression", "categorical", "ordinal"):
                if label_type == "ordinal" and use_ordinal_as_reg:
                    return "regression"
                return label_type
            print(f"Warning: Invalid label_type '{label_type}' for column '{col_name}' in JSON, using heuristic")

        label_type = CategoricalUtils.get_type(column)
        if label_type == "ordinal" and use_ordinal_as_reg:
            return "regression"
        return label_type

    @staticmethod
    def is_categorical(column: pd.Series, unique_threshold: int = 10, ratio_threshold: float = 0.05) -> bool:
        """
        Determine if a column should be treated as categorical.

        Returns True if: object/category/bool dtype OR numeric with ≤unique_threshold
        unique values and ≤ratio_threshold uniqueness ratio.
        """
        if len(column) == 0:
            return False

        dtype = column.dtype

        if (pd.api.types.is_object_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(dtype)):
            return True

        if pd.api.types.is_numeric_dtype(dtype):
            non_null = column.dropna()
            if len(non_null) == 0:
                return False
            n_unique = non_null.nunique()
            return n_unique <= unique_threshold and (n_unique / len(non_null)) <= ratio_threshold

        return False

    @staticmethod
    def get_type(column: pd.Series, unique_threshold: int = 10, ratio_threshold: float = 0.05) -> Literal["regression", "categorical", "ordinal"]:
        """
        Infer column type via heuristic (prefer get_label_type() when JSON config is available).

        Ordinal: numeric integers forming a near-consecutive sequence (e.g. Likert scales 1–5).
        Categorical: non-numeric or sparse numeric with non-sequential values.
        Regression: everything else.
        """
        if not CategoricalUtils.is_categorical(column, unique_threshold, ratio_threshold):
            return "regression"

        non_null = column.dropna()
        if len(non_null) == 0:
            return "categorical"

        if pd.api.types.is_numeric_dtype(column.dtype):
            unique_vals = sorted(non_null.unique())
            if all(float(v).is_integer() for v in unique_vals) and len(unique_vals) > 2:
                int_vals = [int(v) for v in unique_vals]
                if max(int_vals) - min(int_vals) + 1 == len(int_vals):
                    return "ordinal"

        return "categorical"

    @staticmethod
    def get_categorical_columns(df: pd.DataFrame, unique_threshold: int = 10, ratio_threshold: float = 0.05) -> List[str]:
        """Return list of categorical column names in a DataFrame."""
        return [col for col in df.columns if CategoricalUtils.get_type(df[col], unique_threshold, ratio_threshold) == "categorical"]

    @staticmethod
    def detect_categorical_and_numeric_columns(df: pd.DataFrame, unique_threshold: int = 10, ratio_threshold: float = 0.05) -> Tuple[List[int], List[int]]:
        """
        Detect categorical and numeric column indices in a DataFrame.

        Returns:
            Tuple of (categorical_indices, numeric_indices) as integer position lists.
        """
        categorical_idx = []
        numeric_idx = []

        for i, col in enumerate(df.columns):
            if CategoricalUtils.get_type(df[col], unique_threshold, ratio_threshold) == "categorical" and df[col].nunique() > 2:
                categorical_idx.append(i)
            else:
                numeric_idx.append(i)

        return categorical_idx, numeric_idx
