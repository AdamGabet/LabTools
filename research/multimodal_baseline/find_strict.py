import pandas as pd

# Load the overlap file
overlap = pd.read_csv(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/10K_Trajectories/subject_test_date_overlap.csv"
)

# Current benchmark subjects
benchmark = pd.read_csv(
    "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/benchmark_subjects.csv"
)
current_codes = set(benchmark["RegistrationCode"].unique())

# Criteria: 2+ visits in metabolomics, cgm, retina (the longitudinal modalities)
# And 1+ visit in dexa and blood_test (for DEXA features and KPIs)

multi_metab = overlap["n_dates_metabolites"] >= 2
multi_cgm = overlap["n_dates_cgm"] >= 2
multi_retina = overlap["n_dates_retina"] >= 2
has_dexa = overlap["n_dates_dexa"] >= 1
has_blood = overlap["n_dates_blood_test"] >= 1

eligible = overlap[multi_metab & multi_cgm & multi_retina & has_dexa & has_blood]
eligible_codes = set(eligible["registration_code"])

new_candidates = eligible_codes - current_codes

print(f"Current benchmark: {len(current_codes)}")
print(f"Eligible with 2+ visits in metab, cgm, retina: {len(eligible_codes)}")
print(f"New candidates: {len(new_candidates)}")

# Sample
sample = eligible[eligible["registration_code"].isin(new_candidates)].head(20)
print("\n--- Sample new candidates ---")
print(
    sample[
        [
            "registration_code",
            "n_dates_metabolites",
            "n_dates_cgm",
            "n_dates_retina",
            "n_dates_dexa",
            "n_dates_blood_test",
        ]
    ].to_string()
)
