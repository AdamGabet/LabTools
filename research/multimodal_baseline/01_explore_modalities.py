"""
Step 1: Explore available modalities, columns, and subject overlap.
Outputs: exploration_results.txt
"""
import sys
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')

import os
import pandas as pd
import h5py
from body_system_loader.load_feature_df import load_body_system_df

OUT = '/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/exploration_results.txt'
lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

# ── CGM ────────────────────────────────────────────────────────────────────
cgm = load_body_system_df('glycemic_status')
log('=== CGM (glycemic_status) ===')
log(f'Columns: {cgm.columns.tolist()}')
log(f'Visits: {cgm.index.get_level_values("research_stage").value_counts().to_dict()}')
log(f'Subjects: {cgm.index.get_level_values(0).nunique()}')
log()

# ── PROTEOMICS ─────────────────────────────────────────────────────────────
prot = load_body_system_df('proteomics')
log('=== PROTEOMICS ===')
log(f'Total columns: {len(prot.columns)}')
log(f'Visits: {prot.index.get_level_values("research_stage").value_counts().to_dict()}')
log(f'Subjects: {prot.index.get_level_values(0).nunique()}')
inflammation = [c for c in prot.columns if c in ['IL6','TNF','CRP','CXCL10','CXCL8','IL10','GDF15','LGALS3','MMP9']]
kidney       = [c for c in prot.columns if c in ['CST3','CST5','UMOD','LCN2','FGF23','KLOTHO']]
log(f'Inflammation markers present: {inflammation}')
log(f'Kidney/aging markers present: {kidney}')
log()

# ── DEXA ───────────────────────────────────────────────────────────────────
bc = load_body_system_df('body_composition')
bd = load_body_system_df('bone_density')
log('=== DEXA - body_composition ===')
visc = [c for c in bc.columns if 'vat' in c.lower() or 'visceral' in c.lower()]
log(f'Visceral cols: {visc}')
log(f'Visits: {bc.index.get_level_values("research_stage").value_counts().to_dict()}')
log()
log('=== DEXA - bone_density ===')
bmd_cols = [c for c in bd.columns if c in ['body_total_bmd','femur_neck_mean_bmd','spine_l1_l4_bmd']]
log(f'BMD cols present: {bmd_cols}')
log(f'Visits: {bd.index.get_level_values("research_stage").value_counts().to_dict()}')
log()

# ── CARDIOVASCULAR (for blood pressure as retina proxy) ───────────────────
cardio = load_body_system_df('cardiovascular_system')
bp_cols = [c for c in cardio.columns if 'blood_pressure' in c and ('systolic' in c or 'diastolic' in c)]
imt_cols = [c for c in cardio.columns if 'intima' in c]
log('=== CARDIOVASCULAR (retina-adjacent KPIs) ===')
log(f'Blood pressure cols: {bp_cols}')
log(f'IMT cols: {imt_cols}')
log(f'Visits: {cardio.index.get_level_values("research_stage").value_counts().to_dict()}')
log()

# ── RETINA ─────────────────────────────────────────────────────────────────
h5_path = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/eyes/retina_dataset.h5'
retina_records = []
with h5py.File(h5_path, 'r') as f:
    first_key = list(f.keys())[0]
    first_entry = f[first_key]
    subkeys = list(first_entry.keys())
    log(f'=== RETINA (raw H5 images) ===')
    log(f'H5 entry sub-keys (per subject-visit): {subkeys}')
    log(f'OD shape: {first_entry["OD"].shape}')
    for k in f.keys():
        parts = k.split('_')
        reg = '_'.join(parts[:2])
        stage = '_'.join(parts[2:])
        retina_records.append({'RegistrationCode': reg, 'research_stage': stage})
retina_df = pd.DataFrame(retina_records)
log(f'Visits: {retina_df["research_stage"].value_counts().to_dict()}')
log(f'Unique subjects: {retina_df["RegistrationCode"].nunique()}')
log()

# ── JEPA RETINA EMBEDDINGS ─────────────────────────────────────────────────
jepa_retina = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/DeepMultiJepa/embeddings/y0wh39vh/retina_embeddings.csv'
ret_emb = pd.read_csv(jepa_retina)
log('=== JEPA RETINA EMBEDDINGS ===')
log(f'Shape: {ret_emb.shape}')
log(f'Visits: {ret_emb["research_stage"].value_counts().to_dict()}')
log(f'Embedding dims: {len([c for c in ret_emb.columns if c not in ["RegistrationCode","research_stage"]])}')
log()

# ── OVERLAP ANALYSIS ───────────────────────────────────────────────────────
log('=== MODALITY OVERLAP ===')
cgm_idx   = set(cgm.index)
dexa_idx  = set(bc.index)
prot_idx  = set(prot.index)
retina_idx = set(zip(retina_df['RegistrationCode'], retina_df['research_stage']))
cardio_idx = set(cardio.index)

for stage in ['baseline', '02_00_visit', '04_00_visit']:
    s = lambda idx: {r for r, v in idx if v == stage}
    overlap4 = s(cgm_idx) & s(dexa_idx) & s(prot_idx) & s(retina_idx)
    overlap5 = overlap4 & s(cardio_idx)
    log(f'{stage}: CGM={len(s(cgm_idx))}, DEXA={len(s(dexa_idx))}, Prot={len(s(prot_idx))}, '
        f'Retina={len(s(retina_idx))}, Cardio={len(s(cardio_idx))}, All4={len(overlap4)}, All5={len(overlap5)}')

# Multi-visit analysis
all4_idx = cgm_idx & dexa_idx & prot_idx & retina_idx
all4_subjects = {}
for reg, stage in all4_idx:
    all4_subjects.setdefault(reg, []).append(stage)
multi_visit = {r: v for r, v in all4_subjects.items() if len(v) >= 2}
log(f'\nSubjects with 2+ visits having CGM+DEXA+Prot+Retina: {len(multi_visit)}')

# Relax retina requirement: CGM+DEXA+Prot at 2+ visits + retina at baseline
cgm_dexa_prot = cgm_idx & dexa_idx & prot_idx
cdp_subjects = {}
for reg, stage in cgm_dexa_prot:
    cdp_subjects.setdefault(reg, []).append(stage)
multi_cdp = {r: v for r, v in cdp_subjects.items() if len(v) >= 2}
log(f'Subjects with 2+ visits having CGM+DEXA+Prot (no retina req): {len(multi_cdp)}')
# of those, how many also have retina at baseline?
retina_baseline = {r for r, v in retina_idx if v == 'baseline'}
multi_cdp_retina = {r: v for r, v in multi_cdp.items() if r in retina_baseline}
log(f'  ...also having retina at baseline: {len(multi_cdp_retina)}')

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
log(f'\nSaved to {OUT}')
