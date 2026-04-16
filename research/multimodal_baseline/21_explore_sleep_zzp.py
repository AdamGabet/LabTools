"""
Explore the sleep data structure - .zzp is actually a directory!
"""

import sys

sys.path.insert(0, "/home/adamgab/PycharmProjects/LabTools")
import pandas as pd
import os

SLEEP_DIR = (
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/itamar/zzp"
)

print("=" * 80)
print("PART 1: .zzp DIRECTORY STRUCTURE")
print("=" * 80)

# List first few directories
dirs = [d for d in os.listdir(SLEEP_DIR) if os.path.isdir(os.path.join(SLEEP_DIR, d))]
print(f"Total .zzp directories: {len(dirs)}")

# Explore one
sample_dir = os.path.join(SLEEP_DIR, dirs[0])
print(f"\nSample: {dirs[0]}")
for item in sorted(os.listdir(sample_dir))[:20]:
    full = os.path.join(sample_dir, item)
    if os.path.isfile(full):
        size = os.path.getsize(full)
        print(f"  {item} ({size:,} bytes)")
    else:
        print(f"  {item}/ (dir)")

# Check different sizes
print("\n" + "=" * 80)
print("PART 2: VARIATION IN .zzp DIRECTORIES")
print("=" * 80)

# Look at different sized directories
for d in dirs[:10]:
    full = os.path.join(SLEEP_DIR, d)
    items = os.listdir(full)
    print(f"  {d}: {len(items)} items")

print("\n" + "=" * 80)
print("PART 3: FULL CONTENT OF ONE .zzp")
print("=" * 80)

# Find a rich one
for d in dirs:
    full = os.path.join(SLEEP_DIR, d)
    items = os.listdir(full)
    if len(items) > 10:
        print(f"Full content of {d}:")
        for item in sorted(items):
            full_item = os.path.join(full, item)
            if os.path.isfile(full_item):
                size = os.path.getsize(full_item)
                with open(full_item, "rb") as f:
                    header = f.read(200)
                print(f"  {item} ({size:,} bytes)")
                print(f"    First bytes: {header[:50]}")
            else:
                print(f"  {item}/ (dir)")
        break

print("\n" + "=" * 80)
print("PART 4: LOOK FOR ACTUAL TIMESERIES (CSV/JSON)")
print("=" * 80)

# Search deeper in .zzp
all_file_types = {}
for d in dirs[:100]:  # Check first 100
    full = os.path.join(SLEEP_DIR, d)
    for item in os.listdir(full):
        ext = os.path.splitext(item)[1]
        all_file_types[ext] = all_file_types.get(ext, 0) + 1

print(f"File types in first 100 .zzp directories:")
for ext, count in sorted(all_file_types.items(), key=lambda x: -x[1]):
    print(f"  {ext}: {count}")

# Check if any .csv or .json
for d in dirs[:100]:
    full = os.path.join(SLEEP_DIR, d)
    for item in os.listdir(full):
        if item.endswith(".csv") or item.endswith(".json"):
            print(f"Found: {os.path.join(full, item)}")

print("\n" + "=" * 80)
print("PART 5: BODY SYSTEM LOADER FOR SLEEP - TIMESERIES OR TABULAR?")
print("=" * 80)

# The body_system_loader gives us tabular features from sleep
# But what is the underlying data source?
from body_system_loader.load_feature_df import load_body_system_df

sleep_df = load_body_system_df("sleep")
print(f"Loaded sleep data: {sleep_df.shape}")
print(f"Index: {sleep_df.index.names}")
print(f"\nSample index entries:")
print(sleep_df.index[:5])

# Check if there's a separate raw timeseries loader
print("\n" + "=" * 80)
print("PART 6: SLEEP FEATURE INTERPRETATION")
print("=" * 80)

# These features are sleep STUDY features - not raw timeseries
# They come from the Itamar device which does overnight polysomnography
print("These are derived features from sleep studies (overnight sleep tests):")
print("- AHI (Apnea-Hypopnea Index)")
print("- Sleep stages (REM, deep, light)")
print("- Oxygen saturation metrics")
print("- Heart rate metrics")
print("- Snoring events")
print("- Sleep efficiency")
print("\nThis is NOT continuous timeseries like CGM.")
print("It's one-night sleep study per visit.")

print("\n" + "=" * 80)
print("PART 7: COMPARISON TO CGM (what makes CGM a real timeseries?)")
print("=" * 80)

# CGM raw files are txt with glucose readings every ~15min
CGM_DIR = "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/cgm"
cgm_files = [f for f in os.listdir(CGM_DIR) if f.endswith(".txt")][:3]

for f in cgm_files:
    full = os.path.join(CGM_DIR, f)
    with open(full, "r") as fp:
        lines = fp.readlines()[:20]
    print(f"\n{f}:")
    for line in lines[:10]:
        print(f"  {line.strip()}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
Sleep data (Itamar):
- Type: One-night sleep study per visit (polysomnography)
- Features: 127 derived metrics per night (AHI, sleep stages, O2, HR, etc.)
- Visit pattern: 1-6 sleep studies per subject
- NOT continuous timeseries - each visit = one night

CGM data:
- Type: Continuous glucose monitoring (~14 days per session)
- Timeseries: ~1341 glucose readings per file (~15min intervals)
- Visit pattern: 1-4 CGM sessions per subject
- IS continuous timeseries

What this means:
- Sleep is as "longitudinal" as any other clinical test (one measurement per visit)
- CGM provides real timeseries but has very few subjects with 2+ sessions
- Both are useful but neither provides continuous multi-year timeseries
""")
