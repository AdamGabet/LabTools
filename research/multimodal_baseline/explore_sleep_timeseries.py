"""
Explore the sleep timeseries data (preprocessed .pt files).
"""

import os

SLEEP_DIR = "/net/mraid20/ifs/wisdom/segal_lab/jafar/SleepFM/ssl_sleep/preprocessed_data_SR_125Hz_gold"

files = sorted(os.listdir(SLEEP_DIR))
print(f"Total files: {len(files)}")

# Parse filenames to get subject, visit, segment
from collections import defaultdict
import torch

subjects = defaultdict(lambda: defaultdict(list))

for f in files:
    if f.startswith("preprocessed_") and f.endswith(".pt"):
        # Format: preprocessed_{subject}__{visit}__{segment}.pt
        parts = f.replace("preprocessed_", "").replace(".pt", "").split("__")
        if len(parts) >= 2:
            subject = parts[0]
            visit = parts[1] if len(parts) > 1 else "unknown"
            subjects[subject][visit].append(f)

print(f"\nUnique subjects: {len(subjects)}")

# Count subjects per visit
visit_counts = defaultdict(set)
for subj, visits in subjects.items():
    for visit, files in visits.items():
        visit_counts[visit].add(subj)

print("\nSubjects per visit:")
for visit, subj_set in sorted(visit_counts.items()):
    print(f"  {visit}: {len(subj_set)} subjects")

# Check multi-visit subjects
multi_visit = {s: len(v) for s, v in subjects.items() if len(v) >= 2}
print(f"\nSubjects with 2+ visits: {len(multi_visit)}")
print(f"Subjects with 3+ visits: {sum(1 for v in multi_visit.values() if v >= 3)}")

# Check tensor shapes
print("\n" + "=" * 80)
print("SAMPLE TENSOR SHAPES")
print("=" * 80)

for subj in list(subjects.keys())[:3]:
    for visit in list(subjects[subj].keys())[:2]:
        f = subjects[subj][visit][0]
        full = os.path.join(SLEEP_DIR, f)
        data = torch.load(full)
        print(f"{subj} / {visit} / {f}")
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        if len(data.shape) > 0:
            print(f"  Duration (125Hz): {data.shape[-1] / 125 / 60:.1f} min")
        break
