"""
Step 2: Check raw data files for CGM and DEXA to verify actual measurements exist.
"""
import os
import sys
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
import pandas as pd
from collections import defaultdict

OUT = '/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/raw_data_check.txt'
lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

# ── CGM RAW DATA ───────────────────────────────────────────────────────────
CGM_DIR = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/cgm'
log('=== CGM RAW DATA ===')
log(f'Path: {CGM_DIR}')
log(f'Exists: {os.path.exists(CGM_DIR)}')

if os.path.exists(CGM_DIR):
    files = os.listdir(CGM_DIR)
    log(f'Total files: {len(files)}')
    log(f'Sample files: {files[:5]}')
    # Check a sample file
    sample = os.path.join(CGM_DIR, files[0])
    log(f'\nSample file: {files[0]}')
    log(f'Size: {os.path.getsize(sample)} bytes')
    try:
        df = pd.read_csv(sample, nrows=5)
        log(f'Columns: {df.columns.tolist()}')
        log(f'Head:\n{df.head(3).to_string()}')
    except Exception as e:
        log(f'Read error: {e}')
        # Try parquet
        try:
            df = pd.read_parquet(sample)
            log(f'Parquet columns: {df.columns.tolist()}')
        except Exception as e2:
            log(f'Parquet error: {e2}')

    # Count by extension
    exts = defaultdict(int)
    for f in files:
        exts[os.path.splitext(f)[1]] += 1
    log(f'File extensions: {dict(exts)}')

log()

# ── DEXA RAW DATA ──────────────────────────────────────────────────────────
DEXA_DIR = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/dxa'
log('=== DEXA RAW DATA ===')
log(f'Path: {DEXA_DIR}')
log(f'Exists: {os.path.exists(DEXA_DIR)}')

if os.path.exists(DEXA_DIR):
    entries = os.listdir(DEXA_DIR)
    log(f'Total entries: {len(entries)}')
    log(f'Sample entries: {entries[:5]}')

    # Check if entries are dirs or files
    first = os.path.join(DEXA_DIR, entries[0])
    log(f'\nFirst entry is dir: {os.path.isdir(first)}')

    if os.path.isdir(first):
        sub = os.listdir(first)
        log(f'  Sub-files: {sub[:5]}')
        if sub:
            sample_file = os.path.join(first, sub[0])
            log(f'  Sample file size: {os.path.getsize(sample_file)} bytes')
            try:
                df = pd.read_csv(sample_file, nrows=5)
                log(f'  Columns: {df.columns.tolist()}')
                log(f'  Head:\n{df.head(3).to_string()}')
            except Exception as e:
                log(f'  CSV error: {e}')
    else:
        # Files directly
        exts = defaultdict(int)
        for f in entries:
            exts[os.path.splitext(f)[1]] += 1
        log(f'File extensions: {dict(exts)}')
        sample = os.path.join(DEXA_DIR, entries[0])
        log(f'Sample file size: {os.path.getsize(sample)} bytes')
        try:
            df = pd.read_csv(sample, nrows=5)
            log(f'Columns: {df.columns.tolist()}')
            log(f'Head:\n{df.head(3).to_string()}')
        except Exception as e:
            log(f'CSV error: {e}')

    # Count how many entries have non-empty data
    log('\nChecking for empty DEXA entries...')
    empty = 0
    nonempty = 0
    for entry in entries[:200]:  # sample 200
        p = os.path.join(DEXA_DIR, entry)
        if os.path.isdir(p):
            sub = os.listdir(p)
            if not sub:
                empty += 1
            else:
                # check any file has content
                has_content = any(os.path.getsize(os.path.join(p, f)) > 100 for f in sub)
                if has_content:
                    nonempty += 1
                else:
                    empty += 1
        else:
            if os.path.getsize(p) > 100:
                nonempty += 1
            else:
                empty += 1
    log(f'  Of first 200: {nonempty} non-empty, {empty} empty/tiny')

    # Parse subject-visit structure from folder names
    log('\nParsing subject-visit structure...')
    records = []
    for entry in entries:
        # Try to parse RegistrationCode and date/stage from folder name
        parts = entry.replace('10K_', '').split('_')
        records.append({'raw_name': entry, 'parts': parts})
    log(f'Sample parsed: {records[:3]}')

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
log(f'\nSaved to {OUT}')
