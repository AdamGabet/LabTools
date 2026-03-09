# Prevalence by Age: Non-Stratified and Gender-Stratified

Methods for computing prevalence by age (with 95% Wilson CI), overall and by gender. Data are restricted to ages 40–70; conditions are all medical-condition columns in the merged dataset (optionally filtered by minimum case count). Results are saved to CSV.

---

## 1. Data extraction

```python
import numpy as np
import pandas as pd

# Project path
sys.path.insert(0, '/path/to/LabTools')
from body_system_loader.load_feature_df import load_body_system_df

AGE_MIN, AGE_MAX = 40, 70
AGE_BINS = [40, 45, 50, 55, 60, 65, 70]

# Load and merge: age, gender, medical conditions
age_df = load_body_system_df('Age_Gender_BMI', specific_columns=['age', 'gender'])
mc_df = load_body_system_df('medical_conditions')
df = pd.merge(age_df, mc_df, left_index=True, right_index=True, how='inner')
df = df[(df['age'] >= AGE_MIN) & (df['age'] <= AGE_MAX)]

# All condition columns (no specific list)
condition_cols = [c for c in df.columns if c not in ['age', 'gender']]

# Optional: keep only conditions with at least min_n positives
MIN_POSITIVES = 50
n_pos = df[condition_cols].sum()
condition_cols = [c for c in condition_cols if n_pos.get(c, 0) >= MIN_POSITIVES]
```

---

## 2. Wilson 95% CI for proportion

```python
def wilson_ci(n_pos, n_total, z=1.96):
    """95% Wilson score interval for proportion in [0,1]."""
    p = n_pos / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return np.clip(center - margin, 0, 1), np.clip(center + margin, 0, 1)
```

---

## 3. Non-stratified prevalence by age

One row per condition × age bin: prevalence (%), n_positive, n_total, ci_lower, ci_upper.

```python
def prevalence_by_age(df, condition):
    d = df.copy()
    d['age_bin'] = pd.cut(d['age'], bins=AGE_BINS, right=False)
    g = d.groupby('age_bin', observed=True).agg({condition: ['sum', 'count']})
    g.columns = ['n_positive', 'n_total']
    g['prevalence'] = g['n_positive'] / g['n_total']
    ci = g.apply(lambda r: wilson_ci(r['n_positive'], r['n_total']), axis=1)
    g['ci_lower'] = [x[0] for x in ci]
    g['ci_upper'] = [x[1] for x in ci]
    g['age_mid'] = [(i.left + i.right) / 2 for i in g.index]
    g['condition'] = condition
    return g.reset_index()[['condition', 'age_mid', 'n_positive', 'n_total', 'prevalence', 'ci_lower', 'ci_upper']]

# Run for all conditions and save
rows = []
for condition in condition_cols:
    rows.append(prevalence_by_age(df, condition))
prev_df = pd.concat(rows, ignore_index=True)
prev_df['prevalence_pct'] = prev_df['prevalence'] * 100
prev_df[['ci_lower', 'ci_upper']] = prev_df[['ci_lower', 'ci_upper']] * 100
prev_df.to_csv('prevalence_by_age.csv', index=False)
```

---

## 4. Gender-stratified prevalence by age

One row per condition × gender × age bin. Gender: 0 = Female, 1 = Male.

```python
def prevalence_by_age_gender(df, condition, gender_val):
    d = df[df['gender'] == gender_val].copy()
    d['age_bin'] = pd.cut(d['age'], bins=AGE_BINS, right=False)
    g = d.groupby('age_bin', observed=True).agg({condition: ['sum', 'count']})
    g.columns = ['n_positive', 'n_total']
    g['prevalence'] = g['n_positive'] / g['n_total']
    ci = g.apply(lambda r: wilson_ci(r['n_positive'], r['n_total']), axis=1)
    g['ci_lower'] = [x[0] for x in ci]
    g['ci_upper'] = [x[1] for x in ci]
    g['age_mid'] = [(i.left + i.right) / 2 for i in g.index]
    g['condition'] = condition
    g['gender'] = 'Male' if gender_val == 1 else 'Female'
    return g.reset_index()[['condition', 'gender', 'age_mid', 'n_positive', 'n_total', 'prevalence', 'ci_lower', 'ci_upper']]

# Run for all conditions and both genders, save
rows = []
for condition in condition_cols:
    for gender_val, gender_label in [(0, 'Female'), (1, 'Male')]:
        part = prevalence_by_age_gender(df, condition, gender_val)
        if part['n_positive'].sum() >= 50:  # optional: skip if too few cases
            rows.append(part)
strat_df = pd.concat(rows, ignore_index=True)
strat_df['prevalence_pct'] = strat_df['prevalence'] * 100
strat_df[['ci_lower', 'ci_upper']] = strat_df[['ci_lower', 'ci_upper']] * 100
strat_df.to_csv('prevalence_by_age_gender.csv', index=False)
```

---

## Output CSVs

| File | Contents |
|------|----------|
| `prevalence_by_age.csv` | condition, age_mid, n_positive, n_total, prevalence, ci_lower, ci_upper, prevalence_pct |
| `prevalence_by_age_gender.csv` | condition, gender, age_mid, n_positive, n_total, prevalence, ci_lower, ci_upper, prevalence_pct |

Prevalence and CI bounds are in [0,1] plus a percentage column; age_mid is the midpoint of each 5-year bin (42.5, 47.5, …, 67.5).
