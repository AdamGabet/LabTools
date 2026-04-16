import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from body_system_loader.load_feature_df import load_body_system_df

# Setup styling
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]


def main():
    target_metabolite = "urate"

    try:
        df_metabs = load_body_system_df("metabolites_annotated")
        demo_df = load_body_system_df("Age_Gender_BMI")
        df = df_metabs[[target_metabolite]].join(demo_df[["age"]], how="inner")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Filter for age 40-70 and remove NaNs
    df_filtered = df[(df["age"] >= 40) & (df["age"] <= 70)].dropna().copy()

    # Create clean age bins
    age_bins = [40, 45, 50, 55, 60, 65, 70]
    labels = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69"]
    df_filtered["age_group"] = pd.cut(
        df_filtered["age"], bins=age_bins, labels=labels, right=False
    )

    # Sort categories to ensure the plot axis is ordered
    df_filtered["age_group"] = pd.Categorical(
        df_filtered["age_group"], categories=labels, ordered=True
    )

    pdf_path = "research/quick_looks/metabolite_age_binned.pdf"
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(12, 7))

        # Violin plot shows the density distribution, Boxplot shows the quartiles
        sns.violinplot(
            data=df_filtered,
            x="age_group",
            y=target_metabolite,
            palette="muted",
            inner="quartile",
            alpha=0.7,
        )

        # Overlay a boxplot for precision
        sns.boxplot(
            data=df_filtered,
            x="age_group",
            y=target_metabolite,
            width=0.15,
            color="white",
            linewidth=2,
            showfliers=False,
        )

        plt.title(
            f"Distribution of {target_metabolite} across Age Groups",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Age Group", fontsize=13)
        plt.ylabel(f"{target_metabolite} Value", fontsize=13)

        # Clean up the layout
        plt.tight_layout()

        pdf.savefig()
        plt.close()

    print(f"PDF saved to: {pdf_path}")
    print(f"URL: http://127.0.0.1:8766/quick_looks/metabolite_age_binned.pdf")


if __name__ == "__main__":
    main()
