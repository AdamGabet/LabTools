"""
Explore the sleep (Itamar) data structure - is it a proper timeseries?
"""

import sys

sys.path.insert(0, "/home/adamgab/PycharmProjects/LabTools")
import pandas as pd
import os

# Check what body system loader gives us for sleep
from body_system_loader.load_feature_df import load_body_system_df

print("=" * 80)
print("PART 1: SLEEP TABULAR FEATURES (from body_system_loader)")
print("=" * 80)

try:
    sleep_df = load_body_system_df("sleep")
    print(f"Shape: {sleep_df.shape}")
    print(f"\nColumns ({len(sleep_df.columns)}):")
    for c in sleep_df.columns:
        print(f"  {c}")
    print(f"\nSample row:")
    print(sleep_df.iloc[0])
    print(
        f"\nVisits: {sleep_df.index.get_level_values('research_stage').value_counts().to_dict()}"
    )
    print(f"Subjects: {sleep_df.index.get_level_values(0).nunique()}")
except Exception as e:
    print(f"Error loading sleep: {e}")

print()
print("=" * 80)
print("PART 2: RAW SLEEP FILES (.zzp)")
print("=" * 80)

SLEEP_DIR = (
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/itamar/zzp"
)

# List some files
files = os.listdir(SLEEP_DIR)
print(f"Total files: {len(files)}")
print(f"\nFirst 20 files:")
for f in sorted(files)[:20]:
    full = os.path.join(SLEEP_DIR, f)
    size = os.path.getsize(full)
    print(f"  {f} ({size:,} bytes)")

print()
print("=" * 80)
print("PART 3: SLEEP PDF REPORTS")
print("=" * 80)

PDF_DIR = (
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/itamar/pdf"
)
pdf_files = os.listdir(PDF_DIR)
print(f"Total PDF files: {len(pdf_files)}")
print(f"\nFirst 10:")
for f in sorted(pdf_files)[:10]:
    full = os.path.join(PDF_DIR, f)
    size = os.path.getsize(full)
    print(f"  {f} ({size:,} bytes)")

print()
print("=" * 80)
print("PART 4: CAN WE READ A .zzp FILE?")
print("=" * 80)

# .zzp files are Itamar's proprietary format - let's check if they're actually zip archives
sample_file = os.path.join(SLEEP_DIR, sorted(files)[0])
print(f"Sample file: {sample_file}")

with open(sample_file, "rb") as f:
    header = f.read(100)
    print(f"First 100 bytes (hex): {header[:50].hex()}")
    print(f"First 100 bytes (ascii): {header[:50]}")

# Check if it's a zip file
if header[:2] == b"PK":
    print("\nThis is a ZIP archive!")
    import zipfile

    with zipfile.ZipFile(sample_file, "r") as zf:
        print(f"Contents: {zf.namelist()[:20]}")
else:
    print("\nNot a ZIP archive. Checking magic bytes...")
    # Check common formats
    if header[:4] == b"\x89PNG":
        print("PNG image")
    elif header[:2] == b"\x1f\x8b":
        print("GZIP archive")
    elif header[:4] == b"HDF":
        print("HDF5 file")

print()
print("=" * 80)
print("PART 5: SLEEP FEATURES - WHAT DO THEY MEAN?")
print("=" * 80)

if "sleep_df" in dir():
    # Look at feature names to understand what's captured
    print("\nFeature categories:")
    for c in sorted(sleep_df.columns):
        parts = c.split("__") if "__" in c else [c]
        prefix = parts[0]
        print(
            f"  {prefix}: {[col for col in sleep_df.columns if col.startswith(prefix)]}"
        )
        break  # Just show first category

    # Show all unique prefixes
    prefixes = set()
    for c in sleep_df.columns:
        prefix = c.split("__")[0] if "__" in c else c
        prefixes.add(prefix)

    print(f"\nAll feature prefixes ({len(prefixes)}):")
    for p in sorted(prefixes):
        cols = [c for c in sleep_df.columns if c.startswith(p)]
        print(f"  {p}: {len(cols)} features")

print()
print("=" * 80)
print("PART 6: IS THERE A RAW TIMESERIES EXPORT?")
print("=" * 80)

# Check if there's any CSV or timeseries export from the sleep device
RAW_BASE = (
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/itamar"
)
for d in sorted(os.listdir(RAW_BASE)):
    full = os.path.join(RAW_BASE, d)
    if os.path.isdir(full):
        files = os.listdir(full)
        # Check for CSV, JSON, or other structured data
        csv_files = [f for f in files if f.endswith(".csv")]
        json_files = [f for f in files if f.endswith(".json")]
        txt_files = [f for f in files if f.endswith(".txt")]
        if csv_files or json_files or txt_files:
            print(
                f"  {d}: csv={len(csv_files)}, json={len(json_files)}, txt={len(txt_files)}"
            )
            if csv_files:
                print(f"    Sample CSV: {csv_files[0]}")

# Also check if there's a separate directory with extracted timeseries
print("\nSearching for sleep-related timeseries in broader data directories...")
SEARCH_DIRS = [
    "/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses",
]

for base in SEARCH_DIRS:
    if os.path.exists(base):
        for root, dirs, files in os.walk(base):
            # Look for sleep-related directories
            if any(
                "sleep" in d.lower() or "itamar" in d.lower() or "zzp" in d.lower()
                for d in dirs + [os.path.basename(root)]
            ):
                print(f"  Found: {root}")
                for d in dirs[:5]:
                    print(f"    subdir: {d}")
                for f in files[:5]:
                    print(f"    file: {f}")
