"""
Build and run cross-validation results for multiple body systems.
Cleaned version of run_on_systems.py (~250 lines vs ~500 lines).
"""
import os
import warnings
import logging
import pandas as pd

# Suppress LightGBM warnings early (before any imports that might trigger them)
logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')

from .load_feature_df import (
    add_body_system_csv,
    create_body_system_from_other_systems_csv,
    clear_temp_systems,
    cleanup_temp_session,
    get_body_system_column_names,
    filter_existing_columns
)
from .preprocess_features import PreprocessFeatures
from .seeding import seeding
from .ids_folds import id_fold_with_stratified_threshold
from .categorical_utils import set_ordinal_as_regression
from .compare_results import compare_and_collect_results
from .ensemble import ensemble_predictions
from .config import *

try:
    from LabQueue.qp import qp
    from LabUtils.addloglevels import sethandlers
except ImportError:
    qp = None
    sethandlers = None


# ============================================================================
# Helper Functions
# ============================================================================

def get_name(item) -> str:
    """Extract name from config item (str or dict)."""
    return list(item.keys())[0] if isinstance(item, dict) else item


def get_columns(item, filter_missing: bool = True) -> list:
    """Extract column list from config item."""
    if isinstance(item, dict):
        value = list(item.values())[0]
        if isinstance(value, list):
            columns = value
        else:
            columns = get_body_system_column_names(list(item.keys())[0])
    else:
        columns = get_body_system_column_names(item)
    
    return filter_existing_columns(columns) if filter_missing else columns


def is_csv_file(path: str) -> bool:
    """Check if path is a valid CSV file."""
    return isinstance(path, str) and os.path.isfile(path) and path.endswith('.csv')


def has_predictions(directory: str) -> bool:
    """Check if directory contains any predictions.csv file."""
    if not os.path.exists(directory):
        return False
    return any(
        f.endswith('predictions.csv')
        for root, _, files in os.walk(directory)
        for f in files
    )


def save_skipped_result(save_dir: str, reason: str):
    """Save skip marker files to prevent re-running."""
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame().to_csv(os.path.join(save_dir, 'predictions.csv'))
    pd.DataFrame({'skipped': [True], 'reason': [reason]}).to_csv(
        os.path.join(save_dir, 'metrics.csv')
    )
    print(f"Skipped: {reason} -> {save_dir}")


def run_seeding_task(x, y, save_dir: str, config: BuildResultsConfig):
    """
    Run seeding with cross-validation. Works for both queue and local execution.

    Returns dict with 'predictions', 'metrics', and optionally 'skipped' keys.
    """
    n_subjects = x.index.get_level_values(0).nunique()
    if n_subjects < config.num_splits:
        reason = f"insufficient_subjects ({n_subjects} < {config.num_splits} folds)"
        save_skipped_result(save_dir, reason)
        return {'predictions': pd.DataFrame(), 'metrics': pd.DataFrame(), 'skipped': True}

    folds = id_fold_with_stratified_threshold(
        y, seeds=range(config.num_seeds), n_splits=config.num_splits,
        stratified_threshold=config.stratified_minority_threshold
    )
    os.makedirs(save_dir, exist_ok=True)
    return seeding(
        x, y, folds, model_key='all', average_by_subject_id=True,
        gender_split_evaluation=True, save_dir=save_dir, testing=config.testing,
        use_lgbm=True, num_threads=config.num_threads, resume_seeds=config.resume_seeds
    )


# ============================================================================
# Main Class
# ============================================================================

class BuildResults:
    """Build and run cross-validation results for multiple body systems."""
    
    def __init__(self, config: BuildResultsConfig = None):
        self.config = config or DEFAULT_CONFIG
    
    def run(self, session_id: str = None):
        """Main entry point. session_id isolates temp files for parallel runs."""
        # Set global flag for ordinal handling
        set_ordinal_as_regression(self.config.ordinal_as_regression)
        
        clear_temp_systems(session_id)  # Each run gets unique temp file
        self._prepare_data()
        print("Prepared data for all body systems.")
        
        if self.config.with_queue:
            if qp is None:
                raise ImportError("LabQueue is required to run with queue")
            self._run_with_queue()
        else:
            self._run_local()
        
        # Collect and compare results
        baseline_name = get_name(self.config.baseline) if self.config.baseline else None
        compare_and_collect_results(self.config.save_dir, baseline_name)

        # Optional: ensemble predictions across feature systems
        if self.config.ensemble_after_run:
            print("Running ensemble...")
            ensemble_predictions(
                save_dir=self.config.save_dir,
                feature_systems=self.config.ensemble_systems,
                skip_systems=self.config.ensemble_skip_systems,
            )

        cleanup_temp_session()  # Clean up temp file
    
    # ------------------------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------------------------
    
    def _prepare_data(self):
        """Register all systems (run_list, target_systems, baseline) with temp config."""
        # Register feature systems
        for item in self.config.run_list:
            self._register_system(item, self.config.run_column_descriptions)
        
        # Register target systems
        for item in self.config.target_systems:
            self._register_system(item, self.config.target_column_descriptions)
        
        # Register baseline
        if self.config.baseline:
            self._register_system(self.config.baseline)
    
    def _register_system(self, item, column_types: dict = None):
        """Register a single system (handles str, csv path, or column list)."""
        if isinstance(item, str):
            print(f"Using existing system: {item}")
            return
        
        name = list(item.keys())[0]
        value = list(item.values())[0]
        
        if isinstance(value, str):
            # String value - should be a CSV file path
            if not value.endswith('.csv'):
                raise ValueError(f"System '{name}' has a string value that is not a CSV file: {value}\n"
                               f"Expected a path ending with '.csv' or a list of column names.")
            if not os.path.isfile(value):
                raise FileNotFoundError(f"CSV file does not exist for system '{name}':\n"
                                      f"  Path: {value}\n"
                                      f"  Please check the file path in the configuration.")
            add_body_system_csv(value, name, temp=True, column_types=column_types)
        elif isinstance(value, list):
            create_body_system_from_other_systems_csv(name, value)
        else:
            raise ValueError(f"Invalid value type for system '{name}': {type(value).__name__}\n"
                           f"Expected: CSV file path (str) or column list (list)\n"
                           f"Got: {value}")
        
        print(f"Prepared system: {name}")
    
    # ------------------------------------------------------------------------
    # Task Generation
    # ------------------------------------------------------------------------
    
    def _iter_tasks(self):
        """
        Yield preprocessed (x, y, save_dir) tuples for all combinations.

        For each target/feature pair, yields:
          1. Full model task
          2. Baseline model task (filtered to same subjects)
        """
        baseline_name = get_name(self.config.baseline) if self.config.baseline else None
        
        for target_system in self.config.target_systems:
            target_name = get_name(target_system)
            
            for feature_run in self.config.run_list:
                feature_name = get_name(feature_run)
                
                # Create preprocessors
                preprocessor = PreprocessFeatures(
                    feature_systems=feature_name,
                    target_systems=target_name,
                    confounders=self.config.confounders if target_name != 'age_gender_bmi_VAT' else ['gender'],
                    merge_closest_research_stage=self.config.merge_closest_research_stage,
                )
                baseline_preprocessor = PreprocessFeatures(
                    feature_systems=baseline_name,
                    target_systems=target_name,
                    confounders=[],
                    merge_closest_research_stage=self.config.merge_closest_research_stage,
                ) if baseline_name else None
                
                # Yield tasks for each label
                for label_name in preprocessor.targets:
                    # Filter by only_labels if specified
                    if self.config.only_labels and label_name not in self.config.only_labels:
                        continue
                    
                    full_dir = os.path.join(
                        self.config.save_dir, target_name, label_name, feature_name
                    )
                    baseline_dir = os.path.join(
                        self.config.save_dir, target_name, label_name, baseline_name
                    ) if baseline_name else None
                    
                    # Skip if already processed, UNLESS label is in only_labels (force redo those)
                    force_redo = self.config.only_labels and label_name in self.config.only_labels
                    if has_predictions(full_dir) and not force_redo:
                        print(f"Skipping {label_name}/{feature_name}, already finished.")
                        continue
                    
                    # Preprocess full model
                    x, y = preprocessor.preprocess(label_name)
                    yield x, y, full_dir

                    # Preprocess baseline (filtered to same subjects)
                    if baseline_preprocessor:
                        x_base, y_base = baseline_preprocessor.preprocess(
                            label_name, filter_index=x.index
                        )
                        yield x_base, y_base, baseline_dir
    
    # ------------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------------
    
    def _run_local(self):
        """Run all tasks locally (no queue)."""
        for x, y, save_dir in self._iter_tasks():
            run_seeding_task(x, y, save_dir, self.config)

    def _run_with_queue(self):
        """Run all tasks with LabQueue."""
        sethandlers()
        queue_dir = "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/queue_logs/"
        os.makedirs(queue_dir, exist_ok=True)
        os.chdir(queue_dir)

        with qp(jobname="adam", max_u=100, _trds_def=self.config.num_threads, _mem_def='4G') as q:
            q.startpermanentrun()
            methods = []

            for x, y, save_dir in self._iter_tasks():
                method = q.method(run_seeding_task, [x, y, save_dir, self.config])
                methods.append(method)
                print(f"Queued: {save_dir}")

            q.wait(methods, False)


if __name__ == "__main__":
    import argparse
    config_map = {
        'default': DEFAULT_CONFIG,
        'medical': MEDICAL_CONDITIONS_CONFIG,
        'mental': MENTAL_CONFIG,
        'more_seeds': MORE_SEEDS_CONFIG,
        'gait_only': GAIT_ONLY_CONFIG,
        'movement_data': MOVEMENT_DATA_CONFIG,
        'adding_metabolomics': ADDING_METABOLOMICS_CONFIG,
        'adding_metabolomics_gait_only': ADDING_METABOLOMICS_GAIT_ONLY_CONFIG,
        'ablation_all': ABLATION_ALL_CONFIG,
        'shira_glyco': GLYCO_CONFIG,
        'high_level_diet': HIGH_LEVEL_DIET_CONFIG,
    }
    parser = argparse.ArgumentParser(description='Run cross-validation for body systems')
    parser.add_argument('--config', type=str, default=list(config_map.keys())[0],
                        choices=list(config_map.keys()),
                        help='Config to use: ' + ', '.join(list(config_map.keys())))
    parser.add_argument('--session', type=str, default=None,
                        help='Unique session ID for temp files (auto-generated if not set)')
    args = parser.parse_args()
    
    config = config_map[args.config]
    print(f"Running with config: {args.config}")
    print(f"  save_dir: {config.save_dir}")
    print(f"  num_seeds: {config.num_seeds}")
    print(f"  resume_seeds: {config.resume_seeds}")
    print(f"  with_queue: {config.with_queue}")
    
    builder = BuildResults(config=config)
    builder.run(session_id=args.session or args.config)  # Use config name as default session

