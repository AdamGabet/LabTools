"""
Check for additional timeseries signals we might be missing.
"""

import sys

sys.path.insert(0, "/home/adamgab/PycharmProjects/LabTools")
import pandas as pd
import os

print("=" * 80)
print("PART 1: NANOSE (wearable?)")
print("=" * 80)

NANOSE = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/nanose"
for d in sorted(os.listdir(NANOSE)):
    full = os.path.join(NANOSE, d)
    if os.path.isdir(full):
        files = os.listdir(full)
        print(f"  {d}: {len(files)} files")
        # Check file types
        for f in files[:5]:
            print(f"    {f}")

print()
print("=" * 80)
print("PART 2: APPLE WATCH / THIRD PARTY DATA")
print("=" * 80)

THIRD = (
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/third-party"
)
for d in sorted(os.listdir(THIRD)):
    full = os.path.join(THIRD, d)
    if os.path.isdir(full):
        files = os.listdir(full)
        print(f"  {d}: {len(files)} files")
        for f in files[:3]:
            print(f"    {f}")

print()
print("=" * 80)
print("PART 3: CBC (complete blood count - could be timeseries?)")
print("=" * 80)

CBC = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/cbc"
files = os.listdir(CBC)
print(f"  Top-level files: {len(files)}")
for f in files[:5]:
    full = os.path.join(CBC, f)
    if os.path.isdir(full):
        sub = os.listdir(full)
        print(f"  {f}/: {len(sub)} files")
        for s in sub[:3]:
            print(f"    {s}")
    else:
        print(f"  {f}")

print()
print("=" * 80)
print("PART 4: SAMSUNG PROJECT (DXA + population data)")
print("=" * 80)

SAMSUNG = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/samsung-project-export"
for d in sorted(os.listdir(SAMSUNG)):
    full = os.path.join(SAMSUNG, d)
    if os.path.isdir(full):
        files = os.listdir(full)
        print(f"  {d}: {len(files)} files")
        for f in files[:3]:
            print(f"    {f}")

print()
print("=" * 80)
print("PART 5: GAIT - SKELETON DATA STRUCTURE")
print("=" * 80)

GAIT = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/gait"
for d in sorted(os.listdir(GAIT)):
    full = os.path.join(GAIT, d)
    if os.path.isdir(full):
        files = os.listdir(full)
        print(f"  {d}: {len(files)} files")
        for f in files[:3]:
            print(f"    {f}")

print()
print("=" * 80)
print("PART 6: ABI - CSV DATA")
print("=" * 80)

ABI = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/abi"
for d in sorted(os.listdir(ABI)):
    full = os.path.join(ABI, d)
    if os.path.isdir(full):
        files = os.listdir(full)
        print(f"  {d}: {len(files)} files")
        for f in files[:5]:
            print(f"    {f}")

print()
print("=" * 80)
print("PART 7: CGM - CAN WE EXTRACT MORE VISITS FROM RAW FILES?")
print("=" * 80)

# Each CGM file is a ~14 day recording. Some subjects may have multiple files.
CGM_DIR = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/cgm"
# Count files per subject
subject_files = {}
for f in os.listdir(CGM_DIR):
    if f.endswith(".txt") and "_" in f:
        subj_id = f.split("_")[0]
        if subj_id.isdigit():
            reg = "10K_" + subj_id
            subject_files.setdefault(reg, []).append(f)

multi_file_subjects = {k: v for k, v in subject_files.items() if len(v) >= 2}
print(f"Total CGM subjects: {len(subject_files)}")
print(
    f"Subjects with 1 CGM file: {sum(1 for v in subject_files.values() if len(v) == 1)}"
)
print(
    f"Subjects with 2 CGM files: {sum(1 for v in subject_files.values() if len(v) == 2)}"
)
print(
    f"Subjects with 3+ CGM files: {sum(1 for v in subject_files.values() if len(v) >= 3)}"
)

# Show subjects with multiple files
print("\nSample subjects with multiple CGM files:")
for subj, files in list(multi_file_subjects.items())[:10]:
    print(f"  {subj}: {len(files)} files")
    for f in files:
        print(f"    {f}")

print()
print("=" * 80)
print("PART 8: SUMMARY - ALL TIMESERIES MODALITIES RANKED")
print("=" * 80)

CSV = "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/subject_test_date_overlap.csv"
df = pd.read_csv(CSV)

timeseries = ["cgm", "sleep", "ecg", "voice", "gait"]
print(
    f"{'Modality':15s} | {'Any':>6s} | {'1+':>6s} | {'2+':>6s} | {'3+':>6s} | {'4+':>6s} | {'5+':>6s} | {'Max':>3s}"
)
print("-" * 75)
for mod in timeseries:
    col = f"n_dates_{mod}"
    vals = [
        (df[col] > 0).sum(),
        (df[col] >= 1).sum(),
        (df[col] >= 2).sum(),
        (df[col] >= 3).sum(),
        (df[col] >= 4).sum(),
        (df[col] >= 5).sum(),
        df[col].max(),
    ]
    print(f"{mod:15s} | {' | '.join(f'{v:6d}' for v in vals)}")

print()
print("=" * 80)
print("PART 9: BEST COMBOS INCLUDING TIMESERIES FOR 3+ VISITS")
print("=" * 80)

from itertools import combinations

# Key timeseries: sleep (best), voice, ecg, cgm (worst)
# Key tabular: blood_test, microbiome, mental, nightingale, metabolites
# Key image: retina, ultrasound, dexa

all_mods = [m for m in df.columns if m.startswith("n_dates_")]
all_mods = [m.replace("n_dates_", "") for m in all_mods]

# Focus on combos with at least 2 timeseries
timeseries_mods = ["sleep", "voice", "ecg", "cgm", "gait"]
non_timeseries = [m for m in all_mods if m not in timeseries_mods]

print("\n--- 2x timeseries + 2x non-timeseries (3+ visits) ---")
best_combos = []
for ts_combo in combinations(timeseries_mods, 2):
    for nt_combo in combinations(non_timeseries, 2):
        combo = ts_combo + nt_combo
        mask = pd.Series(True, index=df.index)
        for mod in combo:
            mask &= df[f"n_dates_{mod}"] >= 3
        count = mask.sum()
        if count > 0:
            best_combos.append((combo, count))

best_combos.sort(key=lambda x: x[1], reverse=True)
for combo, count in best_combos[:15]:
    ts = [m for m in combo if m in timeseries_mods]
    nt = [m for m in combo if m not in timeseries_mods]
    print(f"  TS={ts} + NT={nt}: {count:5d}")

print("\n--- 1x timeseries + 3x non-timeseries (3+ visits) ---")
best_combos_1ts = []
for ts in timeseries_mods:
    for nt_combo in combinations(non_timeseries, 3):
        combo = (ts,) + nt_combo
        mask = pd.Series(True, index=df.index)
        for mod in combo:
            mask &= df[f"n_dates_{mod}"] >= 3
        count = mask.sum()
        if count > 0:
            best_combos_1ts.append((combo, count))

best_combos_1ts.sort(key=lambda x: x[1], reverse=True)
for combo, count in best_combos_1ts[:15]:
    ts = [m for m in combo if m in timeseries_mods]
    nt = [m for m in combo if m not in timeseries_mods]
    print(f"  TS={ts} + NT={nt}: {count:5d}")
