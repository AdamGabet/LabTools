# Research Memory: Proteomics-Metabolomics Integration

## Project: The SIRT1-Lipid Axis in Metabolic Aging
**Goal**: Identify novel functional modules between longevity proteins (SIRT1/2) and the metabolome.

## 📚 Literature Context (2024-2025)
- Trend: Moving from single-molecule biomarkers to "functional modules" (Protein-Metabolite pairs).
- Focus: Inflammation and metabolic dysfunction integration (e.g., HFpEF, cancer).

## 🛠️ Data Environment
- **Proteomics**: Available in `proteomics.csv`. Key targets: `SIRT1`, `SIRT2`, `TNF`, `IL-6`, `CCL2`.
- **Metabolomics**: Available in `metabolites_annotated.csv`.
- **Constraint**: Age distribution is skewed; analyses should focus on 40-70 range for stability.

## 🔍 Initial Findings & Observations
- Confirmed availability of `SIRT1` and `SIRT2` in the proteomics dataset.
- Validated the existence of `metabolites_annotated.csv` in the `BODY_SYSTEMS` directory.
- Identified that simple system loading (`load_body_system_df('metabolites')`) fails due to naming discrepancies in the config; direct file access or updated system names are required.

## 📈 Hypothesis
SIRT1/2 protein levels are strongly coupled with specific lipidomic signatures, forming a "metabolic-proteomic fingerprint" of biological aging.
