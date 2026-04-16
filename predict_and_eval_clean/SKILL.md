# predict-and-eval — Cross-Validation Pipeline for HPP Body Systems

## What this does

Runs nested cross-validation to predict any HPP body system (or external CSV) from
any other. Produces per-seed predictions, evaluation metrics, and a comparison table.

**Output**: `{save_dir}/all_comparisons.csv` — one row per (label × feature_system),
columns: `score`, `delta` (vs baseline), `is_significant` (Wilcoxon + FDR).

**Package location**: `/home/adamgab/PycharmProjects/LabTools/predict_and_eval_clean/`

---

## Quick Start

```python
from predict_and_eval_clean.run_on_systems_clean import BuildResults
from predict_and_eval_clean.config import BuildResultsConfig

config = BuildResultsConfig(
    run_list=['blood_lipids'],          # feature systems to test
    target_systems=['body_composition'], # what to predict
    save_dir='/tmp/my_results/',
    num_seeds=10,
    num_splits=5,
    testing=False,                      # True = fast mode, preset params only
    use_lgbm=True,                      # False = linear models only (faster)
)
BuildResults(config=config).run()
```

Results in `/tmp/my_results/all_comparisons.csv`.

---

## Config Options (BuildResultsConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `run_list` | list | — | Feature systems: HPP system names, `{'name': 'path.csv'}`, or `{'name': [col_list]}` |
| `target_systems` | list | — | What to predict (same format as run_list) |
| `baseline` | str/dict | None | Baseline to compare against (e.g. `'age_gender_bmi'`). If None, outputs absolute scores. |
| `save_dir` | str | — | Root output directory |
| `num_seeds` | int | 10 | Number of random seeds |
| `num_splits` | int | 5 | Number of CV folds per seed |
| `testing` | bool | False | Fast mode: use preset params, skip tuning |
| `use_lgbm` | bool | True | Include LGBM models; False = linear only |
| `num_threads` | int | 8 | Parallel threads for tree models |
| `confounders` | list | `['age','gender','bmi']` | Columns added to features as confounders |
| `only_labels` | list | None | Run only these labels (forces redo even if done) |
| `resume_seeds` | bool | False | Load existing results, skip completed seeds |
| `merge_closest_research_stage` | bool | False | When subject has multiple stages, merge to closest |
| `stratified_minority_threshold` | int | 100 | Min minority class size for stratified splits |
| `with_queue` | bool | False | Submit tasks to LabQueue (cluster) |
| `ordinal_as_regression` | bool | True | Treat ordinal labels as regression |
| `ensemble_after_run` | bool | False | Run ensemble after all CV is done |
| `ensemble_systems` | list | None | Feature system names to include in ensemble (None = all) |
| `ensemble_skip_systems` | list | None | Feature system names to exclude from ensemble (e.g. `['baseline']`) |

---

## Using External CSVs

```python
config = BuildResultsConfig(
    run_list=[
        {'my_embeddings': '/path/to/embeddings.csv'},   # external CSV
        'blood_lipids',                                  # HPP system
    ],
    target_systems=['body_composition'],
    baseline='age_gender_bmi',
    save_dir='/tmp/results/',
)
```

CSV requirements:
- Must have `RegistrationCode` column (subject ID, format: `10K_XXXXXXXXXX`)
- One row per subject (or per subject+research_stage if longitudinal)
- All other columns become features

---

## Using a Custom Column List

```python
config = BuildResultsConfig(
    run_list=[
        {'my_subset': ['col1', 'col2', 'col3']},  # subset of HPP columns
    ],
    ...
)
```

---

## Running compare_results Manually

`compare_and_collect_results()` runs automatically at the end of `BuildResults.run()`.
If you need to re-run it (e.g. you added more seeds, changed the baseline, or the run
was interrupted after the CV finished but before the summary was written):

```python
from predict_and_eval_clean.compare_results import compare_and_collect_results

compare_and_collect_results(
    main_dir='/tmp/my_results/',
    baseline_features_system_name='age_gender_bmi',  # or None for absolute scores
)
# Writes: all_comparisons.csv, per-system system_summary.csv, per-label comparison_summary.csv
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `main_dir` | — | Root save_dir used when running seeding |
| `baseline_features_system_name` | None | Name of the baseline feature system folder. If None, no delta/wilcox columns. |
| `main_reg_score` | `'r2'` | Primary regression metric to report |
| `main_clas_score` | `'auc'` | Primary classification metric |
| `main_ordinal_score` | `'spearman_r'` | Primary ordinal metric |
| `significance_threshold` | `0.05` | FDR threshold for `is_significant` |
| `skip_systems` | None | List of target system folder names to skip |
| `wilcox_exclude_by` | `'pvalue'` | Exclude labels from Wilcoxon by `'pvalue'` (FDR-corrected score p-value) or `'score'` (raw score threshold) |
| `wilcox_exclude_threshold` | `0.05` | Threshold for `wilcox_exclude_by` |

**What it produces:**
- `{main_dir}/all_comparisons.csv` — full summary across all systems and labels
- `{main_dir}/{target_system}/system_summary.csv` — one file per target system
- `{main_dir}/{target_system}/{label}/comparison_summary.csv` — one file per label
- Also re-applies FDR corrections to all `metrics.csv` files it finds

**Baseline name must match the folder name exactly** — it's the feature system directory
name under `{main_dir}/{target_system}/{label}/`. If unsure:
```python
import os
label_path = '/tmp/my_results/body_composition/total_scan_vat_area/'
print(os.listdir(label_path))  # shows available feature system folder names
```

---

## Reading Results

```python
import pandas as pd
df = pd.read_csv('/tmp/my_results/all_comparisons.csv')

# Top significant results
sig = df[df['is_significant'] == True].sort_values('delta', ascending=False)

# Key columns:
# system, label, score_type (r2/auc/spearman), score, delta, is_significant
# wilcox_pvalue_fdr, score_pvalue_fdr, n_subjects, n_seeds
# gender (all/male/female)
```

---

## Available HPP Body Systems

```
Age_Gender_BMI, blood_lipids, body_composition, bone_density,
cardiovascular_system, diet, family_history, family_medical_conditions,
frailty, gait, glycemic_status, hematopoietic, immune_system, lifestyle,
liver, medical_conditions, medications, mental, metabolites, microbiome,
nightingale, proteomics, renal_function, rna, sleep, study_type
```

---

## Existing Configs

Pre-built configs are in `predict_and_eval_clean/config.py`:

| Config name | Description |
|-------------|-------------|
| `DEFAULT_CONFIG` | Main run: gait embeddings → all body systems |
| `MEDICAL_CONDITIONS_CONFIG` | Medical condition targets |
| `MENTAL_CONFIG` | Mental health targets |
| `MORE_SEEDS_CONFIG` | 20 seeds (more stable estimates) |
| `GAIT_ONLY_CONFIG` | Gait features only |
| `MOVEMENT_DATA_CONFIG` | Movement sensor data |
| `ADDING_METABOLOMICS_CONFIG` | Adding metabolomics to gait |
| `GLYCO_CONFIG` | Glycemia targets |
| `HIGH_LEVEL_DIET_CONFIG` | Dietary targets |

Run a specific config from command line:
```bash
cd /home/adamgab/PycharmProjects/LabTools
python -m predict_and_eval_clean.run_on_systems_clean --config medical
```

---

## Label Type Auto-Detection

Labels are auto-classified based on column name suffix and value distribution:
- **regression**: numeric with >10 unique values
- **categorical**: column name ends in `_bin`, `_flag`, `_status`, or has ≤10 unique values and name suggests category
- **ordinal**: column name ends in `_score`, `_level`, `_grade` with integer values

Score reported in `all_comparisons.csv`:
- Regression → `r2` (primary: `pearson_r`, p-value: `pearson_pvalue`)
- Classification → `auc`
- Ordinal → `spearman_r` (with `spearman_pvalue`)

---

## P-value Corrections

Benjamini-Hochberg FDR is applied automatically to all `metrics.csv` files via `fix_pvals.py`.
For each `(feature_system, pvalue_column)` group, corrected columns are written back as
`*_fdr` variants (e.g. `pearson_pvalue_fdr`).

Wilcoxon (feature vs baseline) also gets BH correction across all labels per model.

---

## Ensemble

Averages seed predictions across multiple feature systems for each `(target_system, label)`.
Saved as an `ensemble/` sub-folder alongside the feature system folders — shows up automatically
in `compare_results` output.

**Via config** (runs automatically after the main CV loop):
```python
config = BuildResultsConfig(
    ...
    ensemble_after_run=True,
    ensemble_skip_systems=['baseline'],  # exclude baseline from ensemble
)
```

**Standalone** (re-run on an existing results directory):
```python
from predict_and_eval_clean.ensemble import ensemble_predictions

ensemble_predictions(
    save_dir='/tmp/my_results/',
    feature_systems=None,        # None = all valid folders
    skip_systems=['baseline'],
    ensemble_name='ensemble',    # output sub-folder name
)
```

Or from the command line:
```bash
python -m predict_and_eval_clean.ensemble \
    --save_dir /tmp/my_results/ \
    --skip baseline \
    --name ensemble
```

**Requirements**: `predictions.csv` must contain a `true_values` column (written automatically
by `seeding.py`). Needs ≥2 feature systems with overlapping subjects and seed columns.

---

## Workflow Tips

1. **Always test first**: set `testing=True, num_seeds=2, num_splits=3` to verify the pipeline runs
2. **Use `only_labels`** to run a specific label fast: `only_labels=['my_label']`
3. **Resume interrupted runs**: set `resume_seeds=True` to skip completed seeds
4. **Linear-only for speed**: `use_lgbm=False` runs ~10× faster (Ridge/Logit only)
5. **No baseline**: omit `baseline` to get absolute scores without delta/wilcox columns
6. **Confounders**: default includes age/gender/bmi. Set `confounders=[]` for raw features

---

## Common Issues

**`insufficient_subjects` skip**: fewer subjects than `num_splits`. Reduce `num_splits` or check your data.

**`Only 1 class in target` skip**: label has no variance. Usually a filtering issue.

**Import errors**: make sure to use the lab venv:
```bash
source /home/adamgab/PycharmProjects/LabTools/.venv/bin/activate
```
