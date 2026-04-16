"""
Step 5: Check overlap between subjects with actual scan images vs tabular features.
Raw modality inputs: CGM txt files, DEXA H5 images, Retina H5 images, Proteomics CSV
KPI targets: come from body_composition, bone_density, glycemic_status, proteomics CSVs
"""
import os, sys, h5py
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
import pandas as pd
from body_system_loader.load_feature_df import load_body_system_df

OUT = '/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/overlap_with_images.txt'
lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

# ── Parse CGM subject-visits from raw files ────────────────────────────────
CGM_DIR = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/cgm'
log('=== CGM raw files ===')
cgm_files = [f for f in os.listdir(CGM_DIR) if f.endswith('.txt')]
# filename format: {10-digit-id}_{YYYYMMDD}.txt
cgm_raw_subjects = set()
for f in cgm_files:
    parts = f.replace('.txt', '').split('_')
    if len(parts) >= 1 and parts[0].isdigit():
        cgm_raw_subjects.add('10K_' + parts[0])
log(f'Subjects with raw CGM files: {len(cgm_raw_subjects)}')

# ── Parse DEXA H5 subject-visits ───────────────────────────────────────────
DEXA_H5 = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/dxa/dxa_dataset.h5'
log('\n=== DEXA H5 images ===')
dexa_h5_records = []
with h5py.File(DEXA_H5, 'r') as f:
    for k in f.keys():
        parts = k.split('_')
        reg = '_'.join(parts[:2])
        stage = '_'.join(parts[2:])
        dexa_h5_records.append({'RegistrationCode': reg, 'research_stage': stage})
dexa_h5_df = pd.DataFrame(dexa_h5_records)
dexa_h5_idx = set(zip(dexa_h5_df['RegistrationCode'], dexa_h5_df['research_stage']))
log(f'Subjects: {dexa_h5_df["RegistrationCode"].nunique()}, rows: {len(dexa_h5_df)}')
log(f'Visits: {dexa_h5_df["research_stage"].value_counts().to_dict()}')

# ── Parse Retina H5 subject-visits ────────────────────────────────────────
RETINA_H5 = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/eyes/retina_dataset.h5'
log('\n=== Retina H5 images ===')
retina_records = []
with h5py.File(RETINA_H5, 'r') as f:
    for k in f.keys():
        parts = k.split('_')
        reg = '_'.join(parts[:2])
        stage = '_'.join(parts[2:])
        retina_records.append({'RegistrationCode': reg, 'research_stage': stage})
retina_df = pd.DataFrame(retina_records)
retina_idx = set(zip(retina_df['RegistrationCode'], retina_df['research_stage']))
log(f'Subjects: {retina_df["RegistrationCode"].nunique()}, rows: {len(retina_df)}')

# ── Load tabular KPI sources ───────────────────────────────────────────────
log('\n=== Tabular KPI sources ===')
cgm_tab = load_body_system_df('glycemic_status')
bc_tab  = load_body_system_df('body_composition')
bd_tab  = load_body_system_df('bone_density')
prot    = load_body_system_df('proteomics')
cardio  = load_body_system_df('cardiovascular_system')

cgm_idx   = set(cgm_tab.index)
bc_idx    = set(bc_tab.index)
bd_idx    = set(bd_tab.index)
prot_idx  = set(prot.index)
cardio_idx= set(cardio.index)

log(f'CGM tabular: {len(cgm_idx)} rows')
log(f'Body composition: {len(bc_idx)} rows')
log(f'Bone density: {len(bd_idx)} rows')
log(f'Proteomics: {len(prot_idx)} rows')

# ── Overlap: subjects with raw IMAGE data + tabular KPIs ─────────────────
log('\n=== 4-modality overlap (images as input + tabular KPIs) ===')
# For each visit, need: CGM file + DEXA H5 image + Retina H5 + Proteomics tabular
# (Using proteomics tabular as both input features and KPI source)
for stage in ['baseline', '02_00_visit', '04_00_visit']:
    s = lambda idx: {r for r, v in idx if v == stage}
    cgm_s   = s(cgm_idx)    # tabular CGM KPIs available
    dexa_s  = s(dexa_h5_idx) # DEXA scan image available
    retina_s= s(retina_idx)  # retina image available
    prot_s  = s(prot_idx)    # proteomics available
    bc_s    = s(bc_idx)      # body composition KPIs
    cardio_s= s(cardio_idx)  # cardiovascular KPIs

    all4_img = cgm_s & dexa_s & retina_s & prot_s
    all4_kpi = cgm_s & bc_s & retina_s & prot_s  # DEXA KPI (body_comp) instead of image
    log(f'\n  {stage}:')
    log(f'    CGM-tab={len(cgm_s)}, DEXA-img={len(dexa_s)}, Retina-img={len(retina_s)}, Prot={len(prot_s)}, BodyComp-KPI={len(bc_s)}')
    log(f'    All4 with DEXA image: {len(all4_img)}')
    log(f'    All4 with BodyComp KPI (not DEXA image req): {len(all4_kpi)}')
    log(f'    All5 (img + cardio): {len(all4_img & cardio_s)}')

# ── Multi-visit: subjects with DEXA image at 2+ visits ────────────────────
log('\n=== DEXA image multi-visit subjects ===')
dexa_h5_df['reg'] = dexa_h5_df['RegistrationCode']
multi_dexa = dexa_h5_df.groupby('reg').size()
log(f'2+ visits: {(multi_dexa >= 2).sum()}')
log(f'3+ visits: {(multi_dexa >= 3).sum()}')

# Among multi-visit DEXA, how many have CGM + Prot + Retina at any visit?
multi_dexa_subjects = set(multi_dexa[multi_dexa >= 2].index)
cgm_subjects = set(r for r, v in cgm_idx)
prot_subjects = set(r for r, v in prot_idx)
retina_subjects = set(r for r, v in retina_idx)

log(f'Multi-visit DEXA + CGM + Prot + Retina at any visit: {len(multi_dexa_subjects & cgm_subjects & prot_subjects & retina_subjects)}')
log(f'Multi-visit DEXA + CGM + Prot (no retina): {len(multi_dexa_subjects & cgm_subjects & prot_subjects)}')

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
log(f'\nSaved to {OUT}')
