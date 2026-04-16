"""
Step 6: Final cohort check — determine feasible subject pool for benchmark.
Using JEPA retina embeddings (not raw H5) since they cover more subjects.
"""
import os, sys, h5py
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
import pandas as pd
from body_system_loader.load_feature_df import load_body_system_df

OUT = '/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/final_cohort_check.txt'
lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

# Load all modality indices
cgm_tab   = load_body_system_df('glycemic_status')
bc_tab    = load_body_system_df('body_composition')
bd_tab    = load_body_system_df('bone_density')
prot      = load_body_system_df('proteomics')
cardio    = load_body_system_df('cardiovascular_system')
agb       = load_body_system_df('Age_Gender_BMI')

# JEPA retina embeddings
jepa_ret = pd.read_csv(
    '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/DeepMultiJepa/embeddings/y0wh39vh/retina_embeddings.csv',
    usecols=['RegistrationCode', 'research_stage']
)
retina_idx = set(zip(jepa_ret['RegistrationCode'], jepa_ret['research_stage']))

cgm_idx   = set(cgm_tab.index)
bc_idx    = set(bc_tab.index)
prot_idx  = set(prot.index)
cardio_idx= set(cardio.index)
agb_idx   = set(agb.index)

# Subjects with all 4 tabular modalities at any single visit
log('=== Per-visit coverage (JEPA retina embeddings) ===')
for stage in ['baseline', '02_00_visit', '04_00_visit']:
    s = lambda idx: {r for r, v in idx if v == stage}
    all4 = s(cgm_idx) & s(bc_idx) & s(prot_idx) & s(retina_idx)
    all5 = all4 & s(cardio_idx)
    log(f'{stage}: CGM={len(s(cgm_idx))}, DEXA={len(s(bc_idx))}, Prot={len(s(prot_idx))}, '
        f'Retina(JEPA)={len(s(retina_idx))}, All4={len(all4)}, +Cardio={len(all5)}')

# Multi-visit: subjects with CGM+DEXA+Prot at 2+ visits + retina at any visit
log('\n=== Multi-visit cohort ===')
cdp = cgm_idx & bc_idx & prot_idx
cdp_subjects = {}
for reg, stage in cdp:
    cdp_subjects.setdefault(reg, []).append(stage)
multi_cdp = {r: sorted(v) for r, v in cdp_subjects.items() if len(v) >= 2}
log(f'CGM+DEXA+Prot at 2+ visits: {len(multi_cdp)}')

retina_subjects_any = {r for r, v in retina_idx}
multi_cdp_retina = {r: v for r, v in multi_cdp.items() if r in retina_subjects_any}
log(f'  + retina at any visit: {len(multi_cdp_retina)}')

cardio_subjects_any = {r for r, v in cardio_idx}
multi_cdp_retina_cardio = {r: v for r, v in multi_cdp_retina.items() if r in cardio_subjects_any}
log(f'  + cardio: {len(multi_cdp_retina_cardio)}')

# Visit distribution for these subjects
from collections import Counter
visit_counts = Counter(len(v) for v in multi_cdp_retina.values())
log(f'Visit count distribution: {dict(visit_counts)}')

# Check which visits they have
has_baseline = sum(1 for v in multi_cdp_retina.values() if 'baseline' in v)
has_02 = sum(1 for v in multi_cdp_retina.values() if '02_00_visit' in v)
has_04 = sum(1 for v in multi_cdp_retina.values() if '04_00_visit' in v)
log(f'Has baseline: {has_baseline}, has 02: {has_02}, has 04: {has_04}')

# Age/Gender/BMI of this cohort
log('\n=== Cohort demographics ===')
cohort_subjects = list(multi_cdp_retina.keys())
agb_cohort = agb.loc[agb.index.get_level_values(0).isin(cohort_subjects)]
agb_cohort = agb_cohort[~agb_cohort.index.get_level_values(0).duplicated(keep='first')]
log(f'Demographics available for {len(agb_cohort)} subjects')

if 'Age' in agb_cohort.columns and len(agb_cohort) > 0:
    log(f'Age: mean={agb_cohort["Age"].mean():.1f}, std={agb_cohort["Age"].std():.1f}, '
        f'min={agb_cohort["Age"].min():.0f}, max={agb_cohort["Age"].max():.0f}')
if 'gender' in agb_cohort.columns:
    log(f'Gender: {agb_cohort["gender"].value_counts().to_dict()}')
if 'BMI' in agb_cohort.columns:
    log(f'BMI: mean={agb_cohort["BMI"].mean():.1f}, std={agb_cohort["BMI"].std():.1f}')

log(f'\nAll Age_Gender_BMI columns: {agb.columns.tolist()}')

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
log(f'\nSaved to {OUT}')
