# Research Plan: Metabolites and Gait Connection

## Research Questions
- Primary: Which metabolites are most strongly associated with gait performance metrics, and can they be used to predict gait performance?
- Secondary: Do these metabolic associations persist after adjusting for age, gender, BMI, and common medical conditions?

## Hypotheses
- H1: Specific metabolic profiles (e.g., lipid metabolites, amino acids) will show significant positive or negative correlations with gait speed and stability.
- H2: A combination of metabolites will provide better predictive power for gait performance than any single metabolite.

## Data Requirements
- Body systems needed: `metabolites`, `gait`, `Age_Gender_BMI`, `medical_conditions`
- Target variable: Gait metrics (specifically prioritising speed, stability/variability, and cadence)
- Covariates / confounders: Age, gender, BMI, and key medical conditions (e.g., Diabetes, CVD, Neurological disorders)

## Inclusion / Exclusion
- Age range: 40–70 (to ensure statistical reliability per bin)
- Cohort: Main 10K cohort (`study_type == 10`) to maintain general population representativeness.
- Minimum n per stratum: n ≥ 100 for any subgroup analysis.

## Analysis Steps
1. **Data quality check**: 
    - Implement checks from `.claude/data-quality-check.md` for age, gender, and BMI.
    - Verify distributions of target gait metrics and metabolite levels.
    - Filter to age 40-70 and `study_type == 10`.
2. **Correlation Analysis**:
    - Calculate Spearman correlations (to account for non-linear metabolic distributions) between all metabolites and priority gait metrics.
    - Perform **Adjusted Analysis**: Calculate partial correlations controlling for age, gender, and BMI using the pattern in `.claude/cross-sectional-association-study.md`.
    - Multiple testing correction: Apply Benjamini-Hochberg (FDR) correction across all metabolite-gait pairs.
3. **Prediction Modeling**:
    - Use `predict_and_eval/regression_seeding/Regressions` to build models predicting gait metrics.
    - Compare `LR_ridge` (linear baseline) vs `LGBM` (non-linear) to determine the best predictor.
    - Feature Selection: Identify top features via coefficient magnitude (Ridge) or feature importance (LGBM).
4. **Robustness check**: 
    - Age-stratified analysis (e.g., 40-55 vs 55-70).
    - Gender-stratified analysis to check for sex-specific metabolic drivers of gait.
    - Sensitivity check: Exclude subjects with severe neurological medical conditions.

## Risks and Mitigations
| Risk | Mitigation |
|------|-----------|
| Data leakage | Subject-level split using `ids_folds` during CV in prediction steps |
| Age confounding | Age as covariate; strictly limit range to 40–70 |
| Collinearity | Check for high correlation between metabolites before Ridge regression |
| Overfitting | Use cross-validation and Ridge regularization for linear models |

## Definition of Done
- [ ] `analysis.py` runs end-to-end, producing correlations and prediction metrics.
- [ ] `FINDINGS.md` written, documenting the most significant metabolites and model performance.
- [ ] `figures/` contains:
    - `correlation_heatmap.png` (Top metabolites vs Gait metrics)
    - `feature_importance.png` (Predictive power of metabolites)
    - `actual_vs_predicted.png` (Prediction performance)
- [ ] `report.pdf` produced following `.claude/create-report.md` structure.
