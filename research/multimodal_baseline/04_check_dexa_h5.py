"""
Step 4: Check the dxa_dataset.h5 file and actual numpy file contents.
"""
import os, sys
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
import h5py
import numpy as np
import pandas as pd

DXA_ROOT = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/dxa'
OUT = '/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/dexa_h5_check.txt'
lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

# ── Check dxa_dataset.h5 ───────────────────────────────────────────────────
h5_path = os.path.join(DXA_ROOT, 'dxa_dataset.h5')
log(f'=== dxa_dataset.h5 ===')
log(f'Size: {os.path.getsize(h5_path) / 1e9:.2f} GB')

with h5py.File(h5_path, 'r') as f:
    keys = list(f.keys())
    log(f'Total entries: {len(keys)}')
    log(f'Sample keys: {keys[:5]}')

    # Inspect first entry
    first = f[keys[0]]
    if hasattr(first, 'keys'):
        log(f'\nFirst key: {keys[0]}')
        log(f'Sub-keys: {list(first.keys())}')
        for sk in list(first.keys())[:5]:
            item = first[sk]
            if hasattr(item, 'shape'):
                log(f'  {sk}: shape={item.shape}, dtype={item.dtype}')
            else:
                log(f'  {sk}: {list(item.keys())[:5]}')
    else:
        log(f'First entry shape: {first.shape}, dtype: {first.dtype}')

    # Parse subject-visit structure
    log('\n=== Parsing subject-visit keys ===')
    records = []
    for k in keys:
        parts = k.split('_')
        reg = '_'.join(parts[:2])   # 10K_XXXXXXXXXX
        stage = '_'.join(parts[2:]) # baseline / 02_00_visit etc
        records.append({'RegistrationCode': reg, 'research_stage': stage})
    df = pd.DataFrame(records)
    log(f'Visit distribution: {df["research_stage"].value_counts().to_dict()}')
    log(f'Unique subjects: {df["RegistrationCode"].nunique()}')
    multi = df.groupby("RegistrationCode").size()
    log(f'Subjects with 2+ visits: {(multi >= 2).sum()}')
    log(f'Subjects with 3+ visits: {(multi >= 3).sum()}')

# ── Check numpy folder structure more carefully ────────────────────────────
log('\n=== Checking actual numpy file content ===')
numpy_dir = os.path.join(DXA_ROOT, 'numpy')
subj_dirs = [d for d in os.listdir(numpy_dir) if d.isdigit()]

# Find a subject with actual files
for subj in subj_dirs[:20]:
    sp = os.path.join(numpy_dir, subj)
    for visit in os.listdir(sp):
        vp = os.path.join(sp, visit)
        if os.path.isdir(vp):
            for date_d in os.listdir(vp):
                dp = os.path.join(vp, date_d)
                if os.path.isdir(dp):
                    files = os.listdir(dp)
                    for fn in files:
                        fp = os.path.join(dp, fn)
                        sz = os.path.getsize(fp)
                        if sz > 1000:
                            log(f'Found large file: {subj}/{visit}/{date_d}/{fn} ({sz} bytes)')
                            if fn.endswith('.npy'):
                                arr = np.load(fp, allow_pickle=True)
                                log(f'  shape: {arr.shape}, dtype: {arr.dtype}')
                            break
                elif os.path.isfile(dp):
                    sz = os.path.getsize(dp)
                    if sz > 1000:
                        log(f'Found large file: {subj}/{visit}/{date_d} ({sz} bytes)')
                        if date_d.endswith('.npy'):
                            arr = np.load(dp, allow_pickle=True)
                            log(f'  shape: {arr.shape}, dtype: {arr.dtype}')

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
log(f'\nSaved to {OUT}')
