import pandas as pd

# Current benchmark subjects
benchmark = pd.read_csv(
    "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/benchmark_subjects.csv"
)
current_codes = set(benchmark["RegistrationCode"].unique())
print(f"Current benchmark: {len(current_codes)} subjects")

# Load the overlap file
overlap = pd.read_csv(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/10K_Trajectories/subject_test_date_overlap.csv"
)

# Current criteria: 4 modalities + multi-visit dexa+metab
has_cgm = overlap["n_dates_cgm"] > 0
has_dexa = overlap["n_dates_dexa"] > 0
has_retina = overlap["n_dates_retina"] > 0
has_metab = overlap["n_dates_metabolites"] > 0
has_blood = overlap["n_dates_blood_test"] > 0

# Eligible pool
eligible = overlap[
    has_cgm
    & has_dexa
    & has_retina
    & has_metab
    & has_blood
    & (overlap["n_dates_dexa"] > 1)
    & (overlap["n_dates_metabolites"] > 1)
]
eligible_codes = set(eligible["registration_code"])
print(f"Eligible pool (4 mods + multi-visit dexa+metab): {len(eligible_codes)}")

# Not in current benchmark
new_candidates = eligible_codes - current_codes
print(f"New subjects available for expansion: {len(new_candidates)}")

# Check which additional modalities these new candidates have
new_df = overlap[overlap["registration_code"].isin(new_candidates)]

print("\n--- Additional modality availability in new candidates ---")
for mod in ["sleep", "ecg", "gait", "microbiome", "nightingale", "proteomics"]:
    col = f"n_dates_{mod}"
    if col in new_df.columns:
        has_mod = (new_df[col] > 0).sum()
        print(f"{mod}: {has_mod} subjects")

# Also check: subjects with ALL 7 core modalities
has_sleep = new_df["n_dates_sleep"] > 0
has_ecg = new_df["n_dates_ecg"] > 0
has_gait = new_df["n_dates_gait"] > 0

all_seven = new_df[
    has_cgm
    & has_dexa
    & has_retina
    & has_metab
    & has_blood
    & has_sleep
    & has_ecg
    & (new_df["n_dates_dexa"] > 1)
    & (new_df["n_dates_metabolites"] > 1)
]
print(f"\nNew candidates with ALL 7 core modalities: {len(all_seven)}")

# Sample new candidates with many modalities
print("\n--- Sample new candidates with 6+ modalities ---")
new_df["total_mods"] = (
    (new_df["n_dates_cgm"] > 0).astype(int)
    + (new_df["n_dates_dexa"] > 0).astype(int)
    + (new_df["n_dates_retina"] > 0).astype(int)
    + (new_df["n_dates_metabolites"] > 0).astype(int)
    + (new_df["n_dates_sleep"] > 0).astype(int)
    + (new_df["n_dates_ecg"] > 0).astype(int)
    + (new_df["n_dates_blood_test"] > 0).astype(int)
)

top_candidates = new_df.nlargest(20, "total_mods")[
    [
        "registration_code",
        "n_dates_cgm",
        "n_dates_dexa",
        "n_dates_retina",
        "n_dates_metabolites",
        "n_dates_sleep",
        "n_dates_ecg",
        "n_dates_blood_test",
        "total_mods",
    ]
]
print(top_candidates.to_string())
