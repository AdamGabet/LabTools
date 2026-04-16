# Findings: Association of Sleep Features with Age

## Overview
This study recreated Figure 2 from "Phenome-wide associations of sleep characteristics in the Human Phenotype Project" (Nature Medicine, 2025), which explores how core sleep metrics evolve with age, stratified by sex.

## Analysis Details
- **Cohort**: HPP Participants aged 40-70 years.
- **Features analyzed**:
    - **pAHI (Apnea-Hypopnea Index)**: Used a logarithmic scale to visualize the distribution of respiratory events.
    - **Mean Nadir SpO2**: The lowest average oxygen saturation during sleep desaturations.
    - **Light Sleep Percentage**: Proportion of total sleep in N1/N2 stages.
    - **Deep Sleep Percentage**: Proportion of total sleep in N3 stage.
- **Statistical Approach**:
    - **Linear Regression**: Trends for males and females were fitted using linear regression.
    - **Percentiles**: Population-level 3rd, 10th, 50th, 90th, and 97th percentiles were estimated across age using LOWESS smoothing of binned quantiles.
- **Visualization**:
    - Scatter plots split by sex (Blue for Male, Red for Female).
    - pAHI panel includes severity zones: Normal-Mild (<15), Moderate (15-30), and Severe (>30).

## Key Observations (to be verified via figure)
- **pAHI**: Generally increases with age, often more steeply in males.
- **Nadir SpO2**: Tends to decrease slightly with age, reflecting increased respiratory burden.
- **Sleep Architecture**: Light sleep typically increases while deep sleep (N3) significantly decreases with age in both sexes.

## Limitations
- The recreation uses a single visit (or most recent) rather than the multi-night aggregated averages used in the original paper's high-fidelity analysis.
- Robust regression was approximated with standard OLS for the initial figure recreation.
