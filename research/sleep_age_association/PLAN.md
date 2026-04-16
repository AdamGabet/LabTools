# Study: Association of sleep features with age in male and female participants

## Goal
Recreate Fig. 2 from "Phenome-wide associations of sleep characteristics in the Human Phenotype Project", showing how various sleep features associate with age, stratified by sex.

## Plan
1. Load sleep data and age/gender demographics.
2. Filter for the core age range (40-70 years) to ensure robust estimates.
3. Identify key sleep features (e.g., sleep duration, quality, apnea markers).
4. Perform correlation analysis (Pearson/Spearman) between each sleep feature and age, separately for males and females.
5. Generate a forest plot or similar visualization showing the effect size (correlation coefficient) and confidence intervals/p-values.
6. Produce `report.pdf` with the final figure.

## Deliverables
- `analysis.py`
- `figures/sleep_age_sex_association.png`
- `FINDINGS.md`
- `report.pdf`
