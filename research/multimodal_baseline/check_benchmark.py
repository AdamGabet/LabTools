import pandas as pd

benchmark = pd.read_csv(
    "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/benchmark_subjects.csv"
)
print("Current benchmark subjects:", len(benchmark))
print("Columns:", benchmark.columns.tolist()[:10])

# Check unique subjects
unique_subjects = benchmark["RegistrationCode"].unique()
print(f"\nUnique subjects: {len(unique_subjects)}")
print("Sample:", unique_subjects[:5].tolist())

# Load the overlap file
overlap = pd.read_csv(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/10K_Trajectories/subject_test_date_overlap.csv"
)

# Check what's available for benchmark subjects
benchmark_codes = unique_subjects.tolist()
sample_check = [
    c for c in benchmark_codes[:5] if c in overlap["registration_code"].values
]
print("\n--- Current benchmark subject modality counts ---")
for code in sample_check:
    row = overlap[overlap["registration_code"] == code].iloc[0]
    print(
        f"{code}: CGM={row['n_dates_cgm']}, DEXA={row['n_dates_dexa']}, Retina={row['n_dates_retina']}, Metab={row['n_dates_metabolites']}, Sleep={row['n_dates_sleep']}, ECG={row['n_dates_ecg']}"
    )
