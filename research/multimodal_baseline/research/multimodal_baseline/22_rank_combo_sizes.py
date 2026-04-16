"""
Rank 4/5/6-modality combinations that include at least one timeseries,
one picture, and one tabular modality.
Primary metric: subjects with 3+ visits in all selected modalities.
Secondary metric: subjects with 2+ visits in all selected modalities.
"""

import pandas as pd
from itertools import combinations


CSV = "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/subject_test_date_overlap.csv"
df = pd.read_csv(CSV)

timeseries = ["sleep", "ecg", "cgm", "voice", "gait"]
pictures = ["retina", "dexa", "ultrasound"]
tabular = [
    "metabolites",
    "blood_test",
    "microbiome",
    "nightingale",
    "mental",
    "proteomics",
    "abi",
]

all_mods = timeseries + pictures + tabular


def count_combo(combo, threshold):
    mask = pd.Series(True, index=df.index)
    for mod in combo:
        mask &= df[f"n_dates_{mod}"] >= threshold
    return int(mask.sum())


def has_all_types(combo):
    s = set(combo)
    return (
        bool(s & set(timeseries)) and bool(s & set(pictures)) and bool(s & set(tabular))
    )


for size in [4, 5, 6]:
    rows = []
    for combo in combinations(all_mods, size):
        if not has_all_types(combo):
            continue
        n3 = count_combo(combo, 3)
        n2 = count_combo(combo, 2)
        if n3 == 0 and n2 == 0:
            continue
        rows.append((combo, n3, n2))

    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    print("=" * 90)
    print(f"TOP {size}-MODALITY COMBOS (>=1 TS + >=1 picture + >=1 tabular)")
    print("=" * 90)
    for combo, n3, n2 in rows[:20]:
        print(f"{', '.join(combo):80s}  3+={n3:5d}  2+={n2:5d}")
    print()
