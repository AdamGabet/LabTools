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
    # Use a representative metabolite (Urate is usually interesting for age)
    target_metabolite = "urate"

    try:
        # Load metabolites and demographics
        df_metabs = load_body_system_df("metabolites_annotated")
        demo_df = load_body_system_df("Age_Gender_BMI")

        # Join for age
        df = df_metabs[[target_metabolite]].join(demo_df[["age"]], how="inner")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Filter for age 40-70
    df_filtered = df[(df["age"] >= 40) & (df["age"] <= 70)].dropna().copy()

    # PDF Generation
    pdf_path = "research/quick_looks/metabolite_age_trend.pdf"
    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(10, 6))

        # Use a regression plot to show the trend clearly
        sns.regplot(
            data=df_filtered,
            x="age",
            y=target_metabolite,
            scatter_kws={"alpha": 0.3, "color": PALETTE[1]},
            line_kws={"color": PALETTE[0], "lw": 3},
        )

        plt.title(f"Age Trend for {target_metabolite}", fontsize=15)
        plt.xlabel("Age", fontsize=12)
        plt.ylabel(f"{target_metabolite} (Value)", fontsize=12)

        pdf.savefig()
        plt.close()

    print(f"PDF saved to: {pdf_path}")
    print(f"URL: http://127.0.0.1:8766/quick_looks/metabolite_age_trend.pdf")


if __name__ == "__main__":
    main()
