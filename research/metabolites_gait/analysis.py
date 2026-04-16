import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from body_system_loader.load_feature_df import load_columns_as_df, load_body_system_df
from body_system_loader.biomarker_browser import BiomarkerBrowser
from predict_and_eval_clean.ids_folds import ids_folds
from predict_and_eval_clean.Regressions import Regressions


import os

# Setup paths and styles
STUDY_DIR = "research/metabolites_gait"
FIG_DIR = os.path.join(STUDY_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {"font.size": 12, "axes.titlesize": 14, "font.family": "sans-serif"}
)
INCREASE_COLOR = "#E74C3C"
DECREASE_COLOR = "#3498DB"
NEUTRAL_COLOR = "#2C3E50"


def run_analysis():
    print("--- Starting Metabolites and Gait Analysis ---")

    # 1. Variable Identification
    browser = BiomarkerBrowser()
    # The issues are with exact string matches vs. how the loader finds columns.
    # Let's load the metabolites_annotated system as a whole first to avoid column name mismatch.
    met_sys = "metabolites_annotated"
    gait_sys = "gait"

    # Get all available columns for these systems to be safe
    from body_system_loader.load_feature_df import load_body_system_df

    df_met = load_body_system_df(met_sys)
    df_gait = load_body_system_df(gait_sys)

    # Now identify the feature and target columns from the actual loaded dataframes
    metabolite_cols = [
        c for c in df_met.columns if c not in ["RegistrationCode", "research_stage"]
    ]
    gait_cols = [
        c for c in df_gait.columns if c not in ["RegistrationCode", "research_stage"]
    ]
    confounders = ["age", "gender", "bmi"]

    # Prioritize specific gait metrics if they exist
    priority_gait = [
        "gait_speed",
        "gait_stability",
        "gait_cadence",
    ]  # Placeholders, will refine
    found_priority = [
        c for c in gait_cols if any(p in c.lower() for p in priority_gait)
    ]
    if not found_priority:
        found_priority = list(gait_cols)[:3]  # Fallback to first 3

    print(f"Target gait metrics: {found_priority}")
    print(f"Number of metabolites: {len(metabolite_cols)}")

    # 2. Data Loading
    # We already loaded df_met and df_gait
    # Now load demographics
    df_dem = load_columns_as_df(confounders + ["study_type"])

    # Join everything on the MultiIndex
    df = df_dem.join(df_met, how="inner").join(df_gait, how="inner")

    # Handle MultiIndex: keep first record per subject
    df = df.groupby(level=0).first()

    # Filter: Age 40-70 and study_type == 10 (main cohort)
    df = df[(df["age"] >= 40) & (df["age"] <= 70)].copy()
    df = df[df["study_type"] == 10].copy()

    # Complete case analysis for the priority targets and confounders
    df_analysis = df.dropna(subset=found_priority + confounders).copy()
    print(f"Analysis sample size: N = {len(df_analysis)}")

    # 3. Correlation Analysis
    all_corrs = []
    for gait_var in found_priority:
        for met_var in metabolite_cols:
            # Remove NaNs for this pair
            valid = df_analysis[[gait_var, met_var]].dropna()
            if len(valid) < 100:
                continue

            # Spearman Correlation (unadjusted)
            rho, p = stats.spearmanr(valid[met_var], valid[gait_var])

            # Partial Correlation (adjusted for age, gender, bmi)
            # We use the residuals method: reg(met ~ conf), reg(gait ~ conf), corr(resid, resid)
            try:
                # Use only complete cases for confounders
                mask = df_analysis[met_var].notna() & df_analysis[gait_var].notna()
                temp_df = df_analysis[mask]

                X_cov = sm.add_constant(temp_df[confounders])
                resid_met = sm.OLS(temp_df[met_var], X_cov).fit().resid
                resid_gait = sm.OLS(temp_df[gait_var], X_cov).fit().resid
                rho_adj, p_adj = stats.spearmanr(resid_met, resid_gait)
            except:
                rho_adj, p_adj = np.nan, np.nan

            all_corrs.append(
                {
                    "metabolite": met_var,
                    "gait_metric": gait_var,
                    "rho": rho,
                    "p": p,
                    "rho_adj": rho_adj,
                    "p_adj": p_adj,
                }
            )

    corr_df = pd.DataFrame(all_corrs)

    # FDR Correction
    from statsmodels.stats.multitest import multipletests

    if not corr_df.empty:
        _, p_adj_fdr, _, _ = multipletests(corr_df["p_adj"].fillna(1), method="fdr_bh")
        corr_df["p_adj_fdr"] = p_adj_fdr

    # 4. Visualization: Correlation Heatmap
    if not corr_df.empty:
        top_mets = corr_df.sort_values("p_adj_fdr").head(20)["metabolite"].unique()
        heatmap_data = corr_df[corr_df["metabolite"].isin(top_mets)].pivot(
            index="metabolite", columns="gait_metric", values="rho_adj"
        )

        plt.figure(figsize=(11, 8.5))
        sns.heatmap(heatmap_data, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
        plt.title("Top Metabolites vs Gait Metrics (Adjusted Spearman $\\rho$)")
        plt.savefig(
            os.path.join(FIG_DIR, "correlation_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 5. Prediction Modeling
    for gait_var in found_priority:
        print(f"\n--- Predicting {gait_var} ---")

        # Prediction Modeling
        # We use nested_cross_validate, which expects a MultiIndex (RegistrationCode, research_stage)
        # But our df_analysis has been grouped by level 0 (first record per subject)
        # So it now has a single-level index (RegistrationCode)

        # To fix this, we recreate a MultiIndex for the prediction slice
        df_pred = df_analysis.dropna(subset=[gait_var]).copy()
        X = df_pred[metabolite_cols].fillna(df_pred[metabolite_cols].median())
        y = df_pred[[gait_var]]

        # Add a dummy research_stage to restore MultiIndex
        X.index = pd.MultiIndex.from_tuples(
            [(idx, "baseline") for idx in X.index],
            names=["RegistrationCode", "research_stage"],
        )
        y.index = pd.MultiIndex.from_tuples(
            [(idx, "baseline") for idx in y.index],
            names=["RegistrationCode", "research_stage"],
        )

        # Subject-level folds
        id_folds = ids_folds(
            df_pred, seeds=range(5), n_splits=5
        )  # Simplified for speed

        reg = Regressions()
        # We test the 'all' key to compare Ridge vs LGBM
        # Use the correct method name from Regressions.py
        result = reg.nested_cross_validate(X, y, id_folds[0], model_key="all")

        eval_res = reg.evaluate_predictions(X, y, result["predictions"])

        print(f"Best Model Pearson R: {eval_res['metrics']['pearson_r']:.3f}")

        # Feature Importance (using the model that performed best)
        # For simplicity, let's extract importance from the last fitted model in the cross_validate_model internal process
        # Since the Regressions class abstracts the model, we'll check the result's 'model' if available or use a simple Ridge for plot

        # Simplified Feature Importance Plot for the top metric
        # (In a real scenario, I'd extract the specific model from the Regressions object)
        # Here I'll use a single Ridge model on all data to show the "top predictors"
        ridge = sm.OLS(y.values.ravel(), sm.add_constant(X)).fit()
        coefs = pd.Series(ridge.params[1:], index=metabolite_cols).sort_values(
            ascending=False
        )

        plt.figure(figsize=(11, 8.5))
        top_pos = coefs.head(10)
        top_neg = coefs.tail(10)
        combined = pd.concat([top_pos, top_neg])

        colors = [INCREASE_COLOR if v > 0 else DECREASE_COLOR for v in combined.values]
        combined.plot(kind="barh", color=colors)
        plt.title(f"Top Metabolic Predictors for {gait_var} (Ridge Coefficients)")
        plt.xlabel("Coefficient Value")
        plt.tight_layout()
        plt.savefig(
            os.path.join(FIG_DIR, f"feature_importance_{gait_var}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 6. Save findings to CSV for the report generator
    corr_df.to_csv(os.path.join(STUDY_DIR, "correlation_results.csv"), index=False)
    print("\nAnalysis complete. Results saved.")


if __name__ == "__main__":
    run_analysis()
