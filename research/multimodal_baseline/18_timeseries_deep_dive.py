"""
Deep dive on timeseries modalities and DEXA visit structure.
"""

import sys

sys.path.insert(0, "/home/adamgab/PycharmProjects/LabTools")
import pandas as pd
import os

CSV = "/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/subject_test_date_overlap.csv"
df = pd.read_csv(CSV)

print("=" * 80)
print("PART 1: DEXA VISIT DISTRIBUTION")
print("=" * 80)

dexa_dates = df["n_dates_dexa"].value_counts().sort_index()
print("DEXA visit distribution:")
for n, count in dexa_dates.items():
    print(f"  {n} visits: {count}")

# Show actual date patterns for subjects with 3+ DEXA visits
dexa_3plus = df[df["n_dates_dexa"] >= 3][
    ["registration_code", "n_dates_dexa", "date_dexa"]
]
print(f"\nSubjects with 3+ DEXA visits: {len(dexa_3plus)}")
print("\nSample date patterns:")
for _, row in dexa_3plus.head(20).iterrows():
    print(
        f"  {row['registration_code']}: {row['n_dates_dexa']} visits | {row['date_dexa']}"
    )

print()
print("=" * 80)
print("PART 2: ALL TIMESERIES MODALITIES - DETAILED VISIT COUNTS")
print("=" * 80)

# Timeseries: cgm, sleep, ecg, voice, gait
# Also check: retina (has many dates per subject due to multiple images), ultrasound
timeseries = ["cgm", "sleep", "ecg", "voice", "gait"]
# Also check these for timeseries potential
image_based = ["retina", "ultrasound", "dexa"]

for mod in timeseries:
    col = f"n_dates_{mod}"
    date_col = f"date_{mod}"
    print(f"\n--- {mod.upper()} ---")
    dist = df[col].value_counts().sort_index()
    for n, count in dist.items():
        print(f"  {n} visits: {count}")

    # Show max dates subject
    max_row = df.loc[df[col].idxmax()]
    print(f"  Max dates per subject: {max_row[col]}")
    print(f"  Sample dates: {max_row[date_col][:200]}")

print()
print("=" * 80)
print("PART 3: RETINA - WHY SO MANY DATES? (multiple images per visit?)")
print("=" * 80)

retina_3plus = df[df["n_dates_retina"] >= 3]
print(f"Subjects with 3+ retina dates: {len(retina_3plus)}")

# Check if these are truly separate visits or multiple images per visit
sample = retina_3plus[["registration_code", "n_dates_retina", "date_retina"]].head(10)
for _, row in sample.iterrows():
    dates = str(row["date_retina"]).split(";")
    print(f"  {row['registration_code']}: {row['n_dates_retina']} dates")
    print(f"    First 10: {dates[:10]}")

print()
print("=" * 80)
print("PART 4: ARE THERE HIDDEN TIMESERIES SIGNALS WE'RE MISSING?")
print("=" * 80)

# Check raw data directories for additional timeseries
RAW_BASE = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files"

print(f"\nRaw data directories in {RAW_BASE}:")
for d in sorted(os.listdir(RAW_BASE)):
    full = os.path.join(RAW_BASE, d)
    if os.path.isdir(full):
        # Count files and check extensions
        files = os.listdir(full)
        extensions = {}
        for f in files[:500]:
            ext = os.path.splitext(f)[1]
            extensions[ext] = extensions.get(ext, 0) + 1
        print(
            f"  {d:25s}: {len(files):6d} files | top exts: {dict(sorted(extensions.items(), key=lambda x: -x[1])[:5])}"
        )

        # Check subdirectories
        subs = [s for s in files if os.path.isdir(os.path.join(full, s))]
        if subs:
            print(f"    subdirs: {subs[:10]}")

print()
print("=" * 80)
print("PART 5: CGM DEEP DIVE - CAN WE SPLIT INTO MULTIPLE VISITS?")
print("=" * 80)

# CGM has very few "visits" but each CGM recording is a long timeseries
# Maybe the issue is how visits are counted - perhaps each subject has 1 long CGM recording
cgm_1visit = df[df["n_dates_cgm"] == 1]
cgm_2visit = df[df["n_dates_cgm"] == 2]
cgm_3plus = df[df["n_dates_cgm"] >= 3]

print(
    f"CGM: 1 visit = {len(cgm_1visit)}, 2 visits = {len(cgm_2visit)}, 3+ visits = {len(cgm_3plus)}"
)

# Check path hints to understand file structure
cgm_paths = df[df["n_dates_cgm"] > 0]["path_hint_cgm"].dropna().head(10)
print("\nCGM path hints:")
for p in cgm_paths:
    print(f"  {p}")

print()
print("=" * 80)
print("PART 6: SLEEP DEEP DIVE - VISIT STRUCTURE")
print("=" * 80)

sleep_paths = df[df["n_dates_sleep"] > 0]["path_hint_sleep"].dropna().head(10)
print("Sleep path hints:")
for p in sleep_paths:
    print(f"  {p}")

# Check sleep directory structure
SLEEP_DIR = (
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/itamar"
)
print(f"\nSleep directory structure ({SLEEP_DIR}):")
for d in sorted(os.listdir(SLEEP_DIR))[:15]:
    full = os.path.join(SLEEP_DIR, d)
    if os.path.isdir(full):
        files = os.listdir(full)
        print(
            f"  {d}: {len(files)} files | extensions: {dict(pd.Series([os.path.splitext(f)[1] for f in files[:200]]).value_counts().head(5))}"
        )
    elif d != "zzp":
        print(f"  {d} (file)")

print()
print("=" * 80)
print("PART 7: ECG DEEP DIVE")
print("=" * 80)

ecg_paths = df[df["n_dates_ecg"] > 0]["path_hint_ecg"].dropna().head(10)
print("ECG path hints:")
for p in ecg_paths:
    print(f"  {p}")

print()
print("=" * 80)
print("PART 8: GAIT DEEP DIVE")
print("=" * 80)

gait_paths = df[df["n_dates_gait"] > 0]["path_hint_gait"].dropna().head(10)
print("Gait path hints:")
for p in gait_paths:
    print(f"  {p}")

print()
print("=" * 80)
print("PART 9: VOICE DEEP DIVE")
print("=" * 80)

voice_paths = df[df["n_dates_voice"] > 0]["path_hint_voice"].dropna().head(10)
print("Voice path hints:")
for p in voice_paths:
    print(f"  {p}")
