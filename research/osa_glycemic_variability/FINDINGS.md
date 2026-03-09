# OSA and Glycemic Variability - Detailed Findings

## Executive Summary

Obstructive sleep apnea (OSA) severity, measured by the Apnea-Hypopnea Index (AHI), is significantly associated with increased glycemic variability in the HPP 10K cohort, independent of age, gender, and BMI. The effect is notably stronger in individuals with prediabetes (3x stronger), suggesting OSA screening may be particularly important in this population.

## Research Question

**Is obstructive sleep apnea associated with glycemic variability independent of age, gender, and BMI?**

## Study Population

| Characteristic | Value |
|----------------|-------|
| Total N | 7,722 |
| Age range | 40-70 years |
| Age (median) | 52 years |
| Male | 49.4% |
| BMI (mean ± SD) | 26.1 ± 4.2 |
| Normoglycemic | 91.8% |
| Prediabetes | 7.7% |
| Diabetes | 0.5% |

### OSA Distribution

| Severity | AHI Range | N | % |
|----------|-----------|---|---|
| Normal | <5 | 2,919 | 37.8% |
| Mild | 5-15 | 3,110 | 40.3% |
| Moderate | 15-30 | 1,333 | 17.3% |
| Severe | ≥30 | 360 | 4.7% |

**Note:** 62% of the cohort has at least mild OSA (AHI ≥5), indicating high prevalence in this population.

## Key Results

### Finding 1: AHI is Associated with Glucose Variability

After adjusting for age, gender, and BMI:

| Outcome | Partial r | p-value | Interpretation |
|---------|-----------|---------|----------------|
| MAGE | 0.068 | <0.001 | Higher AHI → Higher variability |
| Glucose CV | 0.043 | <0.001 | Higher AHI → Higher variability |
| Glucose SD | 0.058 | <0.001 | Higher AHI → Higher variability |
| Time in Range | -0.001 | 0.94 | No association after adjustment |

**Interpretation:** The association is small but highly significant. Time in Range showed no independent association, suggesting OSA affects glucose dynamics (variability) rather than overall control.

### Finding 2: Effect Size Quantification

Multiple regression model (MAGE ~ AHI + Age + Gender + BMI):

| Predictor | β | p-value | Interpretation |
|-----------|---|---------|----------------|
| AHI | 0.097 | <0.001 | +0.097 mg/dL MAGE per 1 AHI increase |
| Age | 0.240 | <0.001 | +0.24 mg/dL MAGE per year of age |
| Gender | 0.258 | 0.35 | No significant gender difference |
| BMI | -0.228 | <0.001 | **Paradox: Higher BMI → Lower variability** |

**Clinical translation:**
- Going from AHI=5 (mild) to AHI=30 (severe): Expected MAGE increase of **2.4 mg/dL** (~6.5% relative increase)

### Finding 3: Effect Modification by Glycemic Status

The OSA-MAGE association is **3x stronger in prediabetics**:

| Subgroup | N | β (AHI) | p-value | Effect (AHI +25) |
|----------|---|---------|---------|------------------|
| Normoglycemic | 7,088 | 0.073 | <0.001 | +1.8 mg/dL |
| Prediabetes | 596 | 0.210 | <0.001 | +5.2 mg/dL |

**Clinical implication:** Prediabetics are more vulnerable to the metabolic effects of OSA. OSA screening and treatment may be particularly beneficial in this population.

### Finding 4: Unexpected BMI Paradox

Higher BMI was associated with **LOWER** glucose variability (β = -0.23, p < 0.001).

**Possible explanations:**
1. More regular eating patterns in higher BMI individuals (less fasting = less variability)
2. Selection bias (healthier obese individuals enrolled in study)
3. CGM calibration/placement differences by adiposity
4. Statistical phenomenon (regression to mean)

**Status:** Unexplained - warrants dedicated investigation.

## Robustness Checks

### 1. Non-linearity
- Linear model had lowest AIC (60007) compared to quadratic (60009) and log (60020)
- **Conclusion:** Linear relationship is appropriate

### 2. Gender Stratification
| Gender | β (AHI) | p-value |
|--------|---------|---------|
| Female | 0.120 | <0.001 |
| Male | 0.077 | <0.001 |

Effect is significant in both genders but **stronger in females**.

### 3. Medication Sensitivity
Excluding glucose-lowering medication users (N=146):
- AHI coefficient: β = 0.093, p < 0.001
- **Conclusion:** Results robust to medication exclusion

### 4. Collinearity
AHI-BMI correlation: r = 0.44 (moderate)
- VIF acceptable for regression
- Both remain significant when included together

## Limitations

1. **Cross-sectional design**: Cannot establish causality. Reverse causation possible (poor glucose control → worse sleep).

2. **Residual confounding**: Did not control for:
   - Diet quality/timing
   - Physical activity
   - Alcohol consumption
   - Sleep duration (beyond apnea)

3. **Selection bias**:
   - Only 45% of cohort has CGM data
   - Very few diabetics (0.5%) - may be excluded or not enrolled

4. **BMI paradox unexplained**: Counterintuitive finding requires further investigation.

5. **Effect sizes are small**: R² = 3.3% means 97% of variability unexplained by this model.

## Strengths

1. **Large sample size** (N = 7,722) provides excellent statistical power
2. **Objective measurements**: Both AHI (polysomnography) and glycemic metrics (CGM) are objective
3. **Multiple glycemic outcomes**: Consistent findings across CV, MAGE, and SD
4. **Confounder adjustment**: Controlled for major confounders (age, gender, BMI)
5. **Stratified analysis**: Identified effect modification by glycemic status
6. **Reproducible**: Full analysis code provided

## Comparison to Literature

Our findings align with published research:

| Study | Finding | Our Result |
|-------|---------|------------|
| Diabetes Therapy 2025 | OSA affects GV irrespective of glycemic status | Confirmed |
| Cardiovasc Diabetol 2020 | AHI correlates positively with MAGE/SD | Confirmed (r=0.07) |
| Scientific Reports 2020 | Nocturnal glucose patterns differ by OSA | Not directly tested |

**Novel contribution:** First demonstration that the OSA-GV effect is 3x stronger in prediabetics in a large cohort.

## Clinical Implications

1. **Screening**: Consider OSA screening in patients with high glucose variability, especially prediabetics

2. **Treatment**: CPAP/OSA treatment may improve glycemic stability (requires interventional study)

3. **Risk stratification**: Prediabetics with OSA may be at higher risk for progression to diabetes

## Future Directions

1. **Longitudinal analysis**: Does treating OSA reduce glucose variability?
2. **Mechanism investigation**: Is intermittent hypoxia the mediator?
3. **BMI paradox**: Dedicated study to explain negative BMI-GV association
4. **Subgroup analysis**: Effect in diabetics (requires larger diabetic sample)

## Files

| File | Description |
|------|-------------|
| `analysis.py` | Complete reproducible analysis code |
| `report.pdf` | Visual PDF report (6 pages) |
| `figures/` | Individual PNG figures |
| `FINDINGS.md` | This document |

## References

1. Diabetes Therapy (2025). Continuous Glucose Monitoring Among People with and without Diabetes Mellitus and Sleep Apnoea.
2. Cardiovasc Diabetol (2020). A study of glycemic variability in patients with type 2 diabetes mellitus with obstructive sleep apnea syndrome using a continuous glucose monitoring system.
3. Scientific Reports (2020). Dynamic changes in nocturnal blood glucose levels are associated with sleep-related features in patients with obstructive sleep apnea.

---
*Analysis conducted January 2026 using HPP 10K dataset*
