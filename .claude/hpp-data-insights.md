# HPP 10K Dataset Insights

Empirical observations from analyses. Update as new findings emerge.

## Sample Characteristics (ages 40-70)

| Metric | Value |
|--------|-------|
| Total subjects with CGM + sleep data | ~7,700 |
| Age (median) | 52 years |
| Male % | ~49% |
| BMI (mean ± SD) | 26.1 ± 4.2 |

## OSA Distribution

| Category | AHI Range | Prevalence |
|----------|-----------|------------|
| Normal | <5 | 37.8% |
| Mild | 5-15 | 40.3% |
| Moderate | 15-30 | 17.3% |
| Severe | ≥30 | 4.7% |

**Note**: High prevalence of mild-moderate OSA in this population.

## Glycemic Status Distribution

| Status | N | % |
|--------|---|---|
| Normoglycemic | ~7,100 | 92% |
| Prediabetes | ~600 | 8% |
| Diabetes | ~40 | <1% |

**Note**: Very few diabetics in CGM-analyzed subset (selection bias - diabetics may have different CGM patterns).

## Unexpected Findings (Potential Future Research)

### 1. BMI-Glucose Variability Paradox
**Observation**: Higher BMI is associated with LOWER glucose variability (MAGE, CV, SD)
- r = -0.056 in normoglycemics (p < 0.001)
- Persists after excluding medication users

**Possible explanations**:
1. More regular eating patterns in higher BMI individuals
2. CGM placement/calibration affected by adipose tissue
3. Selection bias (healthy enough to participate)
4. Statistical artifact (regression to mean)

**Status**: Unexplained - worth investigating

### 2. Gender Differences in OSA-Glycemic Association
**Observation**: OSA-MAGE effect is stronger in females
- Female: β = 0.120 (p < 0.001)
- Male: β = 0.077 (p < 0.001)

**Possible explanations**:
1. Hormonal differences in glucose regulation
2. Different OSA phenotypes by sex
3. Women may be more vulnerable to metabolic effects of hypoxia

**Status**: Consistent with some literature, worth deeper investigation

### 3. Prediabetes Vulnerability
**Observation**: OSA-MAGE effect is 3x stronger in prediabetics
- Normoglycemic: AHI +25 → MAGE +1.8 mg/dL
- Prediabetes: AHI +25 → MAGE +5.2 mg/dL

**Clinical implication**: OSA screening may be particularly important in prediabetes

### 4. Grip Strength-Cardiovascular Paradox (NEW)
**Observation**: Higher grip strength is associated with HIGHER (worse) CV markers
- Grip → PWV: partial r = +0.081 (p < 0.001)
- Grip → Systolic BP: partial r = +0.111 (p < 0.001)
- Effect persists in both genders and all age groups (40-70)

**OPPOSITE to literature** which suggests low grip strength → worse CV health

**Possible explanations**:
1. Grip strength is a marker of physical activity/muscle mass → higher cardiac output → higher BP
2. The frailty-mortality association operates through different mechanisms than CV markers
3. Healthy population - "low grip" individuals are not truly sarcopenic
4. Cross-sectional markers ≠ longitudinal mortality outcomes

**Status**: Novel finding - documented in `research/grip_strength_cardiovascular/`

## Data Quality Notes

### High Missingness Variables
- CGM metrics (iglu_*): ~55% missing
- HbA1c (bt__hba1c): ~69% missing
- Sleep metrics: ~2-6% missing

### Reliable Variables (low missingness)
- Age, gender: 0% missing
- BMI: <1% missing
- AHI: ~2% missing
- Medical conditions: 0% missing

## Validated Associations

| Exposure | Outcome | Partial r | p-value | Confounders |
|----------|---------|-----------|---------|-------------|
| AHI | MAGE | 0.068 | <0.001 | Age, gender, BMI |
| AHI | Glucose CV | 0.043 | <0.001 | Age, gender, BMI |
| AHI | Glucose SD | 0.058 | <0.001 | Age, gender, BMI |
| AHI | Time in Range | -0.001 | 0.94 | Age, gender, BMI |
| Grip Strength | PWV | +0.081 | <0.001 | Age, gender, BMI |
| Grip Strength | Systolic BP | +0.111 | <0.001 | Age, gender, BMI |
| Grip Strength | Diastolic BP | +0.061 | <0.001 | Age, gender, BMI |
| Grip Strength | IMT | +0.030 | 0.002 | Age, gender, BMI |

## Column Name Conventions

### Glycemic Status (iglu_* prefix)
- `iglu_cv`: Coefficient of variation
- `iglu_mage`: Mean Amplitude of Glycemic Excursions
- `iglu_sd`: Standard deviation
- `iglu_in_range_70_180`: Time in range (%)

### Sleep (ahi_* prefix)
- `ahi`: Overall Apnea-Hypopnea Index
- `ahi_obstructive`: Obstructive events only
- `ahi_central`: Central events only
- `ahi_during_rem/nrem`: Stage-specific
- `odi_*`: Oxygen Desaturation Index

### Blood Tests (bt__* prefix)
- `bt__hba1c`: HbA1c
- `bt__alt_gpt`: ALT (liver)
- `bt__ast_got`: AST (liver)
