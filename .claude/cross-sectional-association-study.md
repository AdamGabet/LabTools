# Cross-Sectional Association Study Pattern

Use this skill when investigating associations between exposures and outcomes in the HPP 10K dataset.

## When to Use

- Investigating whether factor X is associated with outcome Y
- Exploring relationships between body systems (e.g., sleep → glycemic, liver → cardiovascular)
- Hypothesis-driven research requiring confound control

## Study Design Checklist

### 1. Literature Validation (BEFORE analysis)
```
- Search PubMed/Google Scholar for existing evidence
- Identify established confounders from literature
- Note expected direction and effect size
- Document if replicating or extending prior work
```

### 2. Variable Selection

**Exposure variables**: The factor you hypothesize affects the outcome
**Outcome variables**: The health measure you're studying
**Confounders (ALWAYS include)**:
- Age (continuous)
- Gender (binary)
- BMI (continuous)

**Additional confounders to consider**:
- Diabetes status (from `medical_conditions`)
- Medication use (from `medications`)
- Socioeconomic factors (from `lifestyle`)

### 3. Data Loading Pattern

```python
from body_system_loader.load_feature_df import load_columns_as_df

# Define columns by category
exposure_cols = ['ahi']  # Your exposure
outcome_cols = ['iglu_cv', 'iglu_mage']  # Your outcomes
confounder_cols = ['age', 'gender', 'bmi']
stratification_cols = ['Diabetes', 'Prediabetes']

# Load all columns
df = load_columns_as_df(exposure_cols + outcome_cols + confounder_cols + stratification_cols)

# Handle multi-index (take first research stage)
df = df.groupby(level=0).first()

# CRITICAL: Filter to ages 40-70 (94% of data, reliable estimates)
df = df[(df['age'] >= 40) & (df['age'] <= 70)].copy()

# Complete case analysis
df_analysis = df.dropna()
print(f"Analysis sample: N = {len(df_analysis)}")
```

### 4. Analysis Steps

#### Step 1: Descriptive Statistics
```python
# Create exposure categories if continuous
def categorize_exposure(value, thresholds, labels):
    for i, thresh in enumerate(thresholds):
        if value < thresh:
            return labels[i]
    return labels[-1]

# Check sample sizes per category (need n > 100 per group)
print(df_analysis.groupby('exposure_category').size())
```

#### Step 2: Unadjusted Analysis
```python
from scipy import stats

# Simple correlation
r, p = stats.pearsonr(df_analysis['exposure'], df_analysis['outcome'])

# ANOVA across categories
groups = [df[df['cat'] == c]['outcome'] for c in categories]
f_stat, p_val = stats.f_oneway(*groups)
```

#### Step 3: Adjusted Analysis (ALWAYS DO THIS)
```python
import statsmodels.api as sm

# Partial correlation (controls for confounders)
def partial_corr(df, x, y, covars):
    X_cov = sm.add_constant(df[covars])
    resid_x = sm.OLS(df[x], X_cov).fit().resid
    resid_y = sm.OLS(df[y], X_cov).fit().resid
    return stats.pearsonr(resid_x, resid_y)

r_adj, p_adj = partial_corr(df, 'exposure', 'outcome', ['age', 'gender', 'bmi'])

# Multiple regression
X = sm.add_constant(df_analysis[['exposure', 'age', 'gender', 'bmi']])
y = df_analysis['outcome']
model = sm.OLS(y, X).fit()
print(model.summary())
```

#### Step 4: Stratified Analysis
```python
# Check if effect differs by subgroup (effect modification)
for subgroup in ['Diabetes', 'gender']:
    for level in df_analysis[subgroup].unique():
        subset = df_analysis[df_analysis[subgroup] == level]
        # Run regression on subset
```

#### Step 5: Robustness Checks
```python
# 1. Non-linearity check (compare AIC)
# 2. Sensitivity analysis (exclude medication users)
# 3. Check for collinearity (r > 0.7 is problematic)
# 4. Gender-stratified analysis
```

### 5. Reporting Template

```
RESEARCH QUESTION: Is [exposure] associated with [outcome]?

SAMPLE: N = X subjects, age 40-70, X% male

MAIN FINDING:
[Exposure] is [significantly/not] associated with [outcome]
after adjusting for age, gender, and BMI.
- Partial correlation: r = X.XX (p = X.XX)
- Regression coefficient: β = X.XX (p = X.XX)

EFFECT SIZE:
Going from [low exposure] to [high exposure]:
- Expected [outcome] change: X units (X% relative change)

STRATIFIED FINDINGS:
- Subgroup A: β = X.XX (p = X.XX)
- Subgroup B: β = X.XX (p = X.XX)

LIMITATIONS:
1. Cross-sectional design - cannot infer causality
2. [List specific limitations]

STRENGTHS:
1. [List specific strengths]
```

## Common Pitfalls

1. **NOT controlling for confounders** - Always adjust for age, gender, BMI at minimum
2. **Ignoring age restriction** - Filter to 40-70 unless studying extremes
3. **Reporting only unadjusted results** - Crude associations are often confounded
4. **Not checking effect modification** - Effects may differ by diabetes status, gender
5. **Over-interpreting small effects** - R² < 0.05 is common in epidemiology

## Example Studies

### Sleep Apnea and Glycemic Variability (completed 2026-01)
- Exposure: AHI (sleep system)
- Outcome: MAGE, glucose CV, SD (glycemic_status system)
- Finding: AHI associated with MAGE (partial r=0.068, p<0.001)
- Effect 3x stronger in prediabetics
- Full study: `research/osa_glycemic_variability/`

## Related Skills

- `data-quality-check.md` - Check distributions before analysis
- `disease-groups.md` - Standard disease groupings
- `create-report.md` - Generate PDF reports
