import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from body_system_loader.load_feature_df import load_body_system_df
from body_system_loader.biomarker_browser import BiomarkerBrowser

# Setup styling
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]


def main():
    # 1. Find metabolite features
    bb = BiomarkerBrowser()
    metabolite_features = bb.search_columns("metabolites")  # Fixed method name

    # Actually, load the full body system "metabolites_annotated" to see what we have
    try:
        df = load_body_system_df("metabolites_annotated")
    except Exception as e:
        print(f"Error loading 'metabolites_annotated' system: {e}")
        return

    print(f"Loaded metabolites dataframe. Shape: {df.shape}")

    # Basic summary
    print("\n--- Data Summary ---")
    print(f"Number of subjects: {df.index.get_level_values(0).nunique()}")
    print(f"Number of features: {df.shape[1]}")

    # Age and BMI distributions (assuming they are in the df or need to be joined)
    # Usually, load_body_system_df handles common demographics if available,
    # but let's check columns.
    cols = df.columns.tolist()

    # If age/bmi not in 'metabolites', we might need Age_Gender_BMI system
    if "age" not in cols or "bmi" not in cols:
        print("Age or BMI missing from metabolites df, loading Age_Gender_BMI...")
        demo_df = load_body_system_df("Age_Gender_BMI")
        # Join on index (RegistrationCode, research_stage)
        df = df.join(demo_df[["age", "bmi"]], how="inner")

    # Filter for age 40-70 as per rules
    df_filtered = df[(df["age"] >= 40) & (df["age"] <= 70)].copy()

    # Binning age for distribution check
    age_bins = [40, 45, 50, 55, 60, 65, 70]
    df_filtered["age_bin"] = pd.cut(df_filtered["age"], bins=age_bins, right=False)

    print("\n--- Age Distribution (40-70) ---")
    print(df_filtered.groupby("age_bin", observed=True).size())

    # PDF Generation
    pdf_path = "research/quick_looks/metabolites_dist_quicklook.pdf"
    with PdfPages(pdf_path) as pdf:
        # Plot 1: Age Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df_filtered["age"], bins=20, kde=True, color=PALETTE[0])
        plt.title("Age Distribution (Filtered 40-70)")
        plt.xlabel("Age")
        plt.ylabel("Count")
        pdf.savefig()
        plt.close()

        # Plot 2: BMI Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df_filtered["bmi"], bins=20, kde=True, color=PALETTE[1])
        plt.title("BMI Distribution (Filtered 40-70)")
        plt.xlabel("BMI")
        plt.ylabel("Count")
        pdf.savefig()
        plt.close()

        # Plot 3: Age vs BMI Scatter
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_filtered, x="age", y="bmi", alpha=0.5, color=PALETTE[2])
        plt.title("Age vs BMI")
        plt.xlabel("Age")
        plt.ylabel("BMI")
        pdf.savefig()
        plt.close()

    print(f"\nPDF saved to: {pdf_path}")
    print(f"URL: http://127.0.0.1:8766/quick_looks/metabolites_dist_quicklook.pdf")


if __name__ == "__main__":
    main()
