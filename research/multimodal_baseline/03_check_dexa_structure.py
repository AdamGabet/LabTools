"""
Step 3: Explore DEXA numpy directory structure in detail.
"""
import os
import sys
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
import numpy as np
import pandas as pd

DXA_ROOT = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/dxa'
OUT = '/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/dexa_structure.txt'
lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

# Top-level dirs
log('=== DXA top-level dirs ===')
top = [d for d in os.listdir(DXA_ROOT) if not d.startswith('.') and not d.startswith('._')]
log(f'Dirs: {top}')

# Explore numpy dir
numpy_dir = os.path.join(DXA_ROOT, 'numpy')
log(f'\n=== numpy dir ===')
numpy_entries = [d for d in os.listdir(numpy_dir) if not d.startswith('.')]
log(f'Total entries: {len(numpy_entries)}')
log(f'Sample: {numpy_entries[:5]}')

# Check first subject dir
first_subj = os.path.join(numpy_dir, numpy_entries[1])  # skip logs
log(f'\nFirst subject dir: {numpy_entries[1]}')
if os.path.isdir(first_subj):
    visits = os.listdir(first_subj)
    log(f'  Visits/files: {visits}')
    for v in visits[:3]:
        vpath = os.path.join(first_subj, v)
        if os.path.isdir(vpath):
            vfiles = os.listdir(vpath)
            log(f'  {v}/ -> {vfiles[:5]}')
            for vf in vfiles[:2]:
                fp = os.path.join(vpath, vf)
                log(f'    {vf}: {os.path.getsize(fp)} bytes')
                if vf.endswith('.npy'):
                    try:
                        arr = np.load(fp, allow_pickle=True)
                        log(f'    shape: {arr.shape}, dtype: {arr.dtype}')
                        if arr.ndim <= 2 and arr.size < 100:
                            log(f'    data: {arr}')
                    except Exception as e:
                        log(f'    load error: {e}')
        else:
            log(f'  {v}: {os.path.getsize(vpath)} bytes')

# Count subjects and check coverage
log('\n=== Subject coverage ===')
subj_dirs = [d for d in numpy_entries if d != 'logs' and os.path.isdir(os.path.join(numpy_dir, d))]
log(f'Total subjects in numpy: {len(subj_dirs)}')

# Check how many are 10K subjects (format: 10 digits)
tenk = [d for d in subj_dirs if len(d) == 10 and d.isdigit()]
log(f'10-digit IDs (10K format): {len(tenk)}')

# Check visit counts
visit_counts = {}
for subj in subj_dirs[:500]:
    sp = os.path.join(numpy_dir, subj)
    if os.path.isdir(sp):
        visits = [v for v in os.listdir(sp) if not v.startswith('.')]
        visit_counts[subj] = len(visits)

from collections import Counter
log(f'Visit count distribution (sample 500): {Counter(visit_counts.values())}')

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
log(f'\nSaved to {OUT}')
