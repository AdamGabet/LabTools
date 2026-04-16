import pandas as pd

df = pd.read_csv(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/10K_Trajectories/subject_test_date_overlap.csv"
)

print(f"Total subjects: {len(df)}")

# Current benchmark coverage
has_cgm = df["n_dates_cgm"] > 0
has_dexa = df["n_dates_dexa"] > 0
has_retina = df["n_dates_retina"] > 0
has_metab = df["n_dates_metabolites"] > 0

all_four = df[has_cgm & has_dexa & has_retina & has_metab]
print(f"Subjects with all 4 modalities: {len(all_four)}")

# Multi-visit counts
multi_cgm = (df["n_dates_cgm"] > 1).sum()
multi_dexa = (df["n_dates_dexa"] > 1).sum()
multi_retina = (df["n_dates_retina"] > 1).sum()
multi_metab = (df["n_dates_metabolites"] > 1).sum()

print(f"Multi-visit CGM: {multi_cgm}")
print(f"Multi-visit DEXA: {multi_dexa}")
print(f"Multi-visit Retina: {multi_retina}")
print(f"Multi-visit Metabolites: {multi_metab}")

# Current cohort criteria
eligible = df[
    has_cgm
    & has_dexa
    & has_retina
    & has_metab
    & (df["n_dates_dexa"] > 1)
    & (df["n_dates_metabolites"] > 1)
]
print(f"\nEligible (4 mods + multi-visit dexa+metab): {len(eligible)}")

# Also check blood_test (for KPIs) - it's always available
has_blood = df["n_dates_blood_test"] > 0
print(f"Subjects with blood_test: {has_blood.sum()}")

# New potential: sleep + ECG + gait as additional modalities
has_sleep = df["n_dates_sleep"] > 0
has_ecg = df["n_dates_ecg"] > 0
has_gait = df["n_dates_gait"] > 0

sleep_plus = df[has_sleep & has_cgm & has_dexa & has_retina & has_metab & has_blood]
print(f"\nWith SLEEP added (5+ mods): {len(sleep_plus)}")

ecg_plus = df[has_ecg & has_cgm & has_dexa & has_retina & has_metab & has_blood]
print(f"With ECG added (5+ mods): {len(ecg_plus)}")

gait_plus = df[has_gait & has_cgm & has_dexa & has_retina & has_metab & has_blood]
print(f"With GAIT added (5+ mods): {len(gait_plus)}")

# Subjects with ALL 7 modalities
all_seven = df[
    has_sleep & has_ecg & has_cgm & has_dexa & has_retina & has_metab & has_blood
]
print(f"\nAll 7 core modalities: {len(all_seven)}")

# Save some examples to file
eligible[
    [
        "registration_code",
        "n_dates_cgm",
        "n_dates_dexa",
        "n_dates_retina",
        "n_dates_metabolites",
        "n_dates_blood_test",
    ]
].head(20).to_csv(
    "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/eligible_subjects.csv",
    index=False,
)
print("\nSaved eligible_subjects.csv")
