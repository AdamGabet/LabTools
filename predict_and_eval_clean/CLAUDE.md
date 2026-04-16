# predict_and_eval_clean

Nested cross-validation pipeline that predicts any HPP body system from any other,
including external CSV files. Produces per-seed predictions, evaluation metrics,
FDR-corrected p-values, and a summary comparison across all runs.

---

## File Map

| File | Purpose |
|------|---------|
| `run_on_systems_clean.py` | **Entry point.** `BuildResults(config).run()` loops over all target×feature combinations, calls `seeding()`, collects results. |
| `config.py` | All `BuildResultsConfig` dataclass instances (DEFAULT, MEDICAL, MENTAL, etc.). Start here to set up a new experiment. |
| `seeding.py` | Outer loop over seeds. Calls `Regressions.nested_cross_validate()` for each seed, accumulates via `MetricsCollector`, saves CSVs. |
| `Regressions.py` | Nested CV core. Runs all model types, picks best by primary metric (pearson_r / AUC / spearman_r). |
| `models.py` | All model definitions + sklearn Pipeline factory. Also contains `FIXED_PARAM_PRESETS` and `TUNABLE_PARAM_RANGES`. |
| `evaluation.py` | Metric functions (regression / classification / ordinal) + gender-split variants + `MetricsCollector` class. |
| `preprocess_features.py` | `PreprocessFeatures`: loads & aligns feature/target DataFrames, drops NaN rows, encodes categoricals. |
| `load_feature_df.py` | Low-level data loading from HPP body system CSVs. Manages temp session JSON for parallel runs. |
| `ids_folds.py` | Subject-level CV split generation. **Always split on `RegistrationCode`, never rows.** |
| `categorical_utils.py` | `CategoricalUtils`: auto-detects label type (regression/categorical/ordinal) from column name + values. |
| `compare_results.py` | Reads output directories, computes deltas vs baseline, runs Wilcoxon tests, saves summaries. |
| `fix_pvals.py` | FDR correction: Benjamini-Hochberg only. |
| `ensemble.py` | Average predictions across feature systems. Standalone CLI or via `ensemble_after_run` config flag. |

---

## Data Flow

```
config.py (BuildResultsConfig)
    └─ run_on_systems_clean.py (BuildResults.run)
           ├─ load_feature_df.py   ← loads HPP CSVs or external CSVs
           ├─ preprocess_features.py  ← aligns X/Y, drops NaN
           ├─ ids_folds.py         ← stratified subject-level CV splits
           └─ seeding.py           ← loop over seeds
                  └─ Regressions.py  ← nested CV per seed
                         ├─ models.py      ← Pipeline factory
                         └─ evaluation.py  ← metric computation
                  └─ evaluation.py (MetricsCollector)
    └─ compare_results.py  ← delta vs baseline, Wilcoxon, FDR
           └─ fix_pvals.py
    └─ ensemble.py  ← (optional) average predictions across feature systems
```

**Output structure:**
```
{save_dir}/
  {target_system}/
    {label}/
      {feature_system}/
        predictions.csv   ← one column per seed
        metrics.csv       ← one row per seed
        folds.json
        example_model_seed_0.pkl
      ensemble/
        predictions.csv   ← averaged predictions across feature systems
        metrics.csv
      comparison_summary.csv
    system_summary.csv
  all_comparisons.csv   ← main result table
```

---

## Key Design Decisions

### Subject-level CV splits (no data leakage)
Folds are always split on `RegistrationCode`, never on rows. A subject's multiple
research stages always go into the same fold. See `ids_folds.py`.

### Auto label type detection
`CategoricalUtils.get_label_type()` inspects the column name suffix and value
distribution to decide regression / categorical / ordinal. Rules are in
`categorical_utils.py`. Ordinal defaults to regression models (`ORDINAL_AS_REGRESSION=True`).

### Nested CV model selection
Each seed runs all model candidates independently, evaluates each on the test fold,
picks the winner by primary metric. No hyperparameter tuning by default — fixed presets
from `FIXED_PARAM_PRESETS` in `models.py` (tuned for ~800K rows, ~1000 features).

### Temp session files
`load_feature_df.py` writes a per-session JSON (`/tmp/pac_session_{id}.json`) to store
column → body system mappings for the duration of a run. This lets parallel queue jobs
share the same data without race conditions. Cleaned up via `cleanup_temp_session()`.

### use_lgbm flag
`config.use_lgbm=False` → only linear models (LR_ridge / Logit). Faster, useful for
quick checks. `use_lgbm=True` → adds LGBM_regression / LGBM_classifier.

### resume_seeds
`config.resume_seeds=True` reads existing `metrics.csv`, skips completed seeds, merges
new results with existing. Safe to interrupt and re-run.

### only_labels
`config.only_labels=['label1', 'label2']` forces only those labels to run (and forces
redo even if output already exists). Useful for incremental runs.

---

## Known Issues / Things to Improve

- **PerformanceWarning in `load_feature_df.py` ~line 314**: fragmented DataFrame from
  repeated column inserts in a loop. Could be fixed with `pd.concat()` instead.
- `run_on_systems_clean.py` hardcodes `queue_dir` to a lab network path — should
  be moved to config if the queue feature is used by others.
- `MetricsCollector` requires `id_research_pairs` to be identical across seeds.
  If subjects differ across seeds (e.g. after filtering), this raises `ValueError`.
  Consider relaxing to union/intersection alignment.
