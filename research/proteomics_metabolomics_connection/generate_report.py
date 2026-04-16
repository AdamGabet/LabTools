import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from body_system_loader.load_feature_df import BODY_SYSTEMS

# High-quality styling
plt.style.use("seaborn-v0_8-whitegrid")
INCREASE_COLOR = "#E74C3C"
DECREASE_COLOR = "#3498DB"
NEUTRAL_COLOR = "#2C3E50"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "font.family": "sans-serif",
        "figure.dpi": 150,
    }
)


def add_image_page(pdf, image_path, title=None):
    fig = plt.figure(figsize=(11, 8.5))
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis("off")
    if title:
        plt.title(title, fontsize=18, fontweight="bold", pad=20)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def add_text_page(pdf, content):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.05, 0.95, content, ha="left", va="top", fontsize=12, family="monospace")
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close()


def run_full_pipeline():
    # 1. Load Data
    met_path = os.path.join(BODY_SYSTEMS, "metabolites_annotated.csv")
    prot_path = os.path.join(BODY_SYSTEMS, "proteomics.csv")
    age_path = os.path.join(BODY_SYSTEMS, "Age_Gender_BMI.csv")

    df_met = pd.read_csv(met_path, index_col=[0, 1])
    df_prot = pd.read_csv(prot_path, index_col=[0, 1])
    df_age = pd.read_csv(age_path, index_col=[0, 1])

    age_mask = (df_age["age"] >= 40) & (df_age["age"] <= 70)
    valid_ids = df_age.index[age_mask].get_level_values(1).unique()

    targets = ["SIRT1", "SIRT2", "TNF", "IL6"]
    targets = [t for t in targets if t in df_prot.columns]

    df = df_prot[targets].join(df_met, how="inner")
    df = df[df.index.get_level_values(1).isin(valid_ids)]

    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # --- FIGURE 1: The Hero Bar Chart (Polished) ---
    sirt2_corrs = df.corr(method="spearman")["SIRT2"].drop(
        labels=targets, errors="ignore"
    )
    top_pos = sirt2_corrs.sort_values(ascending=False).head(10)
    top_neg = sirt2_corrs.sort_values(ascending=False).tail(10)
    all_top = pd.concat([top_pos, top_neg])

    plt.figure(figsize=(11, 8.5))
    colors = [INCREASE_COLOR if v > 0 else DECREASE_COLOR for v in all_top.values]
    bars = plt.barh(
        all_top.index, all_top.values, color=colors, edgecolor="black", alpha=0.8
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.title(
        "SIRT2 Metabolic Signature: Top Associated Metabolites",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Spearman Correlation (rho)", fontweight="bold")
    plt.ylabel("Metabolite", fontweight="bold")

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {width:.3f}",
            va="center",
            ha="left" if width > 0 else "right",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "main_finding.png"), dpi=300)
    plt.close()

    # --- FIGURE 2: Detailed Correlation Matrix (SIRT2 vs Others) ---
    plt.figure(figsize=(8, 8))
    corr_matrix = df[["SIRT2", "SIRT1", "TNF", "IL6"]].corr(method="spearman")
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="RdBu_r",
        center=0,
        fmt=".2f",
        linewidths=1,
        square=True,
    )
    plt.title("Inter-Protein Correlation Matrix", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "protein_matrix.png"), dpi=300)
    plt.close()

    # --- FIGURE 3: Scatter Plots with Regression Line ---
    top_metabolites = top_pos.index[:2].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    for i, met in enumerate(top_metabolites):
        sns.regplot(
            x=df["SIRT2"],
            y=df[met],
            ax=axes[i],
            scatter_kws={"alpha": 0.2, "color": NEUTRAL_COLOR},
            line_kws={"color": "red", "lw": 2},
        )
        rho = df[["SIRT2", met]].corr(method="spearman").iloc[0, 1]
        axes[i].set_title(
            f"SIRT2 vs {met}\n(rho = {rho:.3f})", fontsize=14, fontweight="bold"
        )
        axes[i].set_xlabel("SIRT2 Expression", fontweight="bold")
        axes[i].set_ylabel("Metabolite Level", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "supporting_scatters.png"), dpi=300)
    plt.close()

    # --- PDF Construction ---
    pdf_path = os.path.join(output_dir, "report.pdf")
    with PdfPages(pdf_path) as pdf:
        methods_text = f"""
        METHODS
        ═══════════════════════════════════════
        
        Study Design
        • Cross-sectional multi-omics integration (Proteomics + Metabolomics)
        • Sample size: N = {len(df)} (Filtered for Age 40-70)
        • Dataset: LabTools HPP 10K
        
        Statistical Analysis
        • Model: Spearman Rank Correlation (non-parametric)
        • Feature Selection: Top 10 positive and negative correlations for SIRT2
        • Interpretation: rho > 0 (Positive Association), rho < 0 (Negative Association)
        
        Inclusion Criteria
        • Subjects aged 40-70 years
        • Complete case analysis (dropna) for SIRT2 and associated metabolites
        """
        add_text_page(pdf, methods_text)
        add_image_page(
            pdf,
            os.path.join(figures_dir, "main_finding.png"),
            title="SIRT2 Metabolic Fingerprint",
        )
        add_image_page(
            pdf,
            os.path.join(figures_dir, "protein_matrix.png"),
            title="SIRT2 vs Inflammation Markers",
        )
        add_image_page(
            pdf,
            os.path.join(figures_dir, "supporting_scatters.png"),
            title="Detailed Protein-Metabolite Coupling",
        )

    print(f"High-quality PDF Report generated at: {pdf_path}")


if __name__ == "__main__":
    run_full_pipeline()
