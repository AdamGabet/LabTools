"""
Step 7: Rebuild overlap analysis using actual signal data (raw files).
If proteomics has poor multi-visit coverage, suggest better body systems.
"""
import os, sys, h5py
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
import pandas as pd
from collections import Counter
from body_system_loader.load_feature_df import load_body_system_df

OUT = '/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/signal_overlap_alternatives.txt'
lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

# ── ACTUAL SIGNAL DATA ────────────────────────────────────────────────────
log('=' * 60)
log('ACTUAL SIGNAL COVERAGE (raw data files)')
log('=' * 60)

# CGM raw txt files → subjects
CGM_DIR = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/cgm'
cgm_raw_subjects = set()
for f in os.listdir(CGM_DIR):
    if f.endswith('.txt') and '_' in f:
        subj_id = f.split('_')[0]
        if subj_id.isdigit():
            cgm_raw_subjects.add('10K_' + subj_id)
log(f'\nCGM raw txt files: {len(cgm_raw_subjects)} subjects')

# DEXA H5
DEXA_H5 = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/dxa/dxa_dataset.h5'
dexa_records = []
with h5py.File(DEXA_H5, 'r') as f:
    for k in f.keys():
        parts = k.split('_')
        reg = '_'.join(parts[:2])
        stage = '_'.join(parts[2:])
        dexa_records.append({'RegistrationCode': reg, 'research_stage': stage})
dexa_df = pd.DataFrame(dexa_records)
dexa_idx = set(zip(dexa_df['RegistrationCode'], dexa_df['research_stage']))
dexa_subjects = set(dexa_df['RegistrationCode'])
log(f'DEXA H5 images: {len(dexa_subjects)} subjects, {len(dexa_df)} subject-visits')
log(f'  Visits: {dexa_df["research_stage"].value_counts().to_dict()}')

# Retina JEPA embeddings
jepa_ret = pd.read_csv(
    '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/DeepMultiJepa/embeddings/y0wh39vh/retina_embeddings.csv',
    usecols=['RegistrationCode', 'research_stage']
)
retina_idx = set(zip(jepa_ret['RegistrationCode'], jepa_ret['research_stage']))
retina_subjects = set(jepa_ret['RegistrationCode'])
log(f'Retina JEPA embeddings: {len(retina_subjects)} subjects')
log(f'  Visits: {jepa_ret["research_stage"].value_counts().to_dict()}')

# Core 3-modality overlap (CGM + DEXA + Retina)
log('\n--- Core 3 modalities (CGM raw + DEXA H5 + Retina JEPA) ---')
core3 = cgm_raw_subjects & dexa_subjects & retina_subjects
log(f'Subjects with all 3: {len(core3)}')

# Multi-visit for core 3
def multivis_subjects(idx, min_visits=2):
    subs = {}
    for r, s in idx:
        subs.setdefault(r, []).append(s)
    return {r: v for r, v in subs.items() if len(v) >= min_visits}

dexa_multi = multivis_subjects(dexa_idx)
core3_multi = set(dexa_multi.keys()) & cgm_raw_subjects & retina_subjects
log(f'Multi-visit DEXA + CGM + Retina: {len(core3_multi)}')

# ── CANDIDATE 4TH MODALITIES ──────────────────────────────────────────────
log('\n' + '=' * 60)
log('4TH MODALITY CANDIDATES (multi-visit coverage)')
log('=' * 60)

candidates = [
    'proteomics',
    'nightingale',
    'metabolites_annotated',
    'blood_lipids',
    'blood_tests_lipids',
    'hematopoietic',
    'immune_system',
    'microbiome',
    'rna',
    'renal_function',
    'liver',
    'cardiovascular_system',
    'sleep',
    'gait',
]

results = []
for cand in candidates:
    try:
        df = load_body_system_df(cand)
        idx = set(df.index)
        subjects = set(r for r, v in idx)
        # Per-visit
        per_visit = {}
        for stage in ['baseline', '02_00_visit', '04_00_visit']:
            s_stage = {r for r, v in idx if v == stage}
            per_visit[stage] = len(s_stage)

        # Multi-visit within this modality
        multi = multivis_subjects(idx)

        # Overlap with core3 (any visit)
        overlap_core3 = subjects & core3
        # Multi-visit overlap: both DEXA multi-visit AND this modality multi-visit
        overlap_multi = set(multi.keys()) & set(dexa_multi.keys()) & cgm_raw_subjects & retina_subjects

        results.append({
            'modality': cand,
            'total_subjects': len(subjects),
            'baseline': per_visit.get('baseline', 0),
            '02_00_visit': per_visit.get('02_00_visit', 0),
            '04_00_visit': per_visit.get('04_00_visit', 0),
            'multi_visit_subjects': len(multi),
            'overlap_with_core3': len(overlap_core3),
            'multi_visit_overlap_core3': len(overlap_multi),
            'n_features': len(df.columns),
        })
        log(f'\n{cand}:')
        log(f'  Total: {len(subjects)} subj | BL={per_visit.get("baseline",0)} '
            f'| 02={per_visit.get("02_00_visit",0)} | 04={per_visit.get("04_00_visit",0)}')
        log(f'  Multi-visit: {len(multi)} | Overlap with CGM+DEXA+Retina: {len(overlap_core3)}')
        log(f'  Multi-visit overlap (CGM+DEXA+Retina+this): {len(overlap_multi)}')
        log(f'  N features: {len(df.columns)}')
    except Exception as e:
        log(f'\n{cand}: ERROR - {e}')

# Summary table
log('\n' + '=' * 60)
log('SUMMARY TABLE (sorted by multi-visit overlap with core 3)')
log('=' * 60)
results_df = pd.DataFrame(results).sort_values('multi_visit_overlap_core3', ascending=False)
log(results_df[['modality', 'total_subjects', 'baseline', '02_00_visit',
                 'multi_visit_subjects', 'overlap_with_core3', 'multi_visit_overlap_core3']].to_string(index=False))

results_df.to_csv('/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/modality_coverage.csv', index=False)

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
log(f'\nSaved to {OUT}')
