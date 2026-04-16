import pandas as pd

df = pd.read_csv(
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/10K_Trajectories/subject_test_date_overlap.csv"
)

benchmark = pd.read_csv(
    "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/benchmark_subjects.csv"
)
current_codes = set(benchmark["RegistrationCode"].unique())

pictures = ["retina", "dexa", "ultrasound"]
timeseries = ["cgm", "sleep", "ecg", "gait", "voice"]
tabulars = [
    "metabolites",
    "blood_test",
    "microbiome",
    "nightingale",
    "proteomics",
    "abi",
]

for times in timeseries:
    print(f"\n=== Top 3 combos with {times.upper()} ===")
    results = []
    for pic in pictures:
        for tab in tabulars:
            has_pic = df[f"n_dates_{pic}"] >= 2
            has_times = df[f"n_dates_{times}"] >= 2
            has_tab = df[f"n_dates_{tab}"] >= 2

            eligible = df[has_pic & has_times & has_tab]
            count = len(eligible)
            new_cnt = len(set(eligible["registration_code"]) - current_codes)
            results.append((pic, tab, count, new_cnt))

    results.sort(key=lambda x: -x[2])
    for i, (pic, tab, count, new_cnt) in enumerate(results[:3]):
        print(f"  {i + 1}. {pic} + {times} + {tab}: {count} ({new_cnt} new)")
