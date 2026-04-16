import pandas as pd

df = pd.read_csv(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/10K_Trajectories/subject_test_date_overlap.csv"
)
print("Modalities and their subject counts (at least 1 date):")
for col in df.columns:
    if col.startswith("n_dates_"):
        mod = col.replace("n_dates_", "")
        cnt = (df[col] > 0).sum()
        print(f"  {mod}: {cnt}")
