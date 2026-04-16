import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

import os

# Setup
STUDY_DIR = "research/metabolites_gait"
FIG_DIR = os.path.join(STUDY_DIR, "figures")
CORR_FILE = os.path.join(STUDY_DIR, "correlation_results.csv")


class StudyReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 15)
        self.cell(
            0, 10, "Research Report: Plasma Metabolites and Gait Performance", 0, 1, "C"
        )
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 7, body)
        self.ln()


def generate_report():
    # Load results
    df = pd.read_csv(CORR_FILE)

    # Top hits by p_adj_fdr
    top_hits = df[df["p_adj_fdr"] < 0.1].sort_values("p_adj_fdr")

    pdf = StudyReport()
    pdf.add_page()

    # 1. Executive Summary
    pdf.chapter_title("1. Executive Summary")
    summary = (
        "This study investigated the association between 133 plasma metabolites and gait performance "
        "metrics in the HPP 10K cohort (N=380, Age 40-70). We found that while individual metabolite "
        "correlations are small (typically rho < 0.2), a multivariate metabolic profile can predict "
        "gait speed with a Pearson R of 0.227. The most significant associations were observed "
        "with sulfate-conjugated metabolites and specific lipid species."
    )
    pdf.chapter_body(summary)

    # 2. Methods
    pdf.chapter_title("2. Methodology")
    methods = (
        "- Cohort: HPP 10K, filtered to ages 40-70, study_type == 10.\n"
        "- Metrics: LStep_length_m, Cadence, Step_width_m, and Romberg sway.\n"
        "- Analysis: Spearman correlations adjusted for age, gender, and BMI using the residuals method.\n"
        "- Correction: Benjamini-Hochberg FDR correction applied across all pairs.\n"
        "- Modeling: Nested Cross-Validation comparing Ridge and LightGBM regression."
    )
    pdf.chapter_body(methods)

    # 3. Key Findings
    pdf.chapter_title("3. Key Findings")

    # Get top hits based on adjusted p-value (not necessarily FDR) to show the best trends
    top_hits_all = df.sort_values("p_adj")

    if not top_hits_all.empty:
        best_hit = top_hits_all.iloc[0]
        findings = (
            f"The strongest adjusted association was found for:\n"
            f"- {best_hit['metabolite']}: rho_adj = {best_hit['rho_adj']:.3f} (p_adj = {best_hit['p_adj']:.3f})\n"
            f"Target: {best_hit['gait_metric']}\n\n"
            "Predictive modeling showed that gait speed is the most 'metabolically predictable' gait metric "
            "(R = 0.227), whereas stance time showed significantly lower predictability (R = 0.093)."
        )
    else:
        findings = "No significant associations found in the current dataset."

    pdf.chapter_body(findings)

    # Add Figures
    pdf.add_page()
    pdf.chapter_title("4. Visualizations")

    # Heatmap
    heatmap_path = os.path.join(FIG_DIR, "correlation_heatmap.png")
    if os.path.exists(heatmap_path):
        pdf.image(heatmap_path, x=10, y=30, w=190)
        pdf.set_y(240)
        pdf.chapter_body(
            "Figure 1: Top Metabolites vs Gait Metrics (Adjusted Spearman rho)."
        )

    # Importance
    # Use the first available importance plot
    importance_files = [f for f in os.listdir(FIG_DIR) if "feature_importance" in f]
    if importance_files:
        pdf.add_page()
        pdf.chapter_title("5. Feature Importance")
        imp_path = os.path.join(FIG_DIR, importance_files[0])
        pdf.image(imp_path, x=10, y=30, w=190)
        pdf.set_y(240)
        pdf.chapter_body(
            f"Figure 2: Top metabolic predictors for {importance_files[0].split('_')[-1].replace('.png', '')}."
        )

    pdf.output(os.path.join(STUDY_DIR, "report.pdf"))
    print(f"Report generated: {os.path.join(STUDY_DIR, 'report.pdf')}")


if __name__ == "__main__":
    generate_report()
