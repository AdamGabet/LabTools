# Requirements for Figure 2: Association of sleep features with age in male and female participants

Based on the paper "Phenome-wide associations of sleep characteristics in the Human Phenotype Project" (Nature Medicine, 2025), the requirements to recreate Figure 2 are as follows:

## 1. Sleep Features Plotted
The figure consists of four panels (a-d), each plotting a specific sleep feature against age:
- **Panel a**: Peripheral Apnea–Hypopnea Index (**pAHI**) — plotted on a **logarithmic scale**.
- **Panel b**: Mean nadir oxygen saturation during sleep desaturations (**mean nadir SpO2**).
- **Panel c**: Percentage of time spent in **light sleep**.
- **Panel d**: Percentage of time spent in **deep sleep**.

## 2. Statistical Methods
- **Association**: **Robust linear regression** was used to determine the trend of each feature with age. The regression equations are displayed in the top left of each graph.
- **Reference Lines**: **LOWESS (Locally Weighted Scatterplot Smoothing)** regression was used to calculate the 3rd, 10th, 50th, 90th, and 97th percentiles, shown as dotted black lines.
- **Aggregation**: Multi-night monitoring data were used (aggregated values across multiple nights to enhance reliability).

## 3. Visualization Details
- **Plot Type**: Scatter plots with overlaid regression lines and percentile curves.
- **Grouping**: Data is split by sex:
    - **Female**: Orange
    - **Male**: Blue
- **Axes/Labels**:
    - **X-axis**: Age.
    - **Y-axis**: The respective sleep feature (note: pAHI is log-scaled).
- **Additional Visual Elements**:
    - **Marginal Histograms**: Histograms of the $x$ (age) and $y$ (feature) axis values are displayed on the top and right of each graph.
    - **OSA Severity Markers (Panel a only)**: Color-coded regions for pAHI:
        - Green: Normal-mild (pAHI < 15)
        - Yellow: Moderate (15 < pAHI < 30)
        - Red: Severe (pAHI > 30)

## 4. Cohort Filtering and Adjustments
- **Population**: Adults from the Human Phenotype Project (HPP) cohort.
- **Filtering**: 
    - Exclusion of invalid recordings.
    - Participants who did not meet project inclusion criteria.
    - Total included: 6,366 participants (3,043 male, 3,323 female).
- **Time Window**: Body system characteristics were measured within a period of ±6 months from the visit.
- **Monitoring**: Sleep monitoring was performed over three nights within a 2-week time period.

---
**Source**: Kohn et al. (2025). "Phenome-wide associations of sleep characteristics in the Human Phenotype Project". Nature Medicine. https://doi.org/10.1038/s41591-024-03481-x
