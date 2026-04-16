"""
Step 8: Define KPIs for each modality and inspect metabolites for KPI candidates.
Modalities: CGM (raw txt), DEXA (H5 images), Retina (H5 images), Metabolites (tabular)
"""
import sys, os, h5py
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
import pandas as pd
import numpy as np
from body_system_loader.load_feature_df import load_body_system_df

OUT = '/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline/kpi_candidates.txt'
lines = []
def log(msg=''):
    print(msg)
    lines.append(str(msg))

# ── CGM KPIs (from glycemic_status tabular) ───────────────────────────────
log('=== CGM KPIs ===')
cgm = load_body_system_df('glycemic_status')
for col in ['iglu_gmi', 'iglu_in_range_70_180', 'bt__hba1c']:
    if col in cgm.columns:
        s = cgm[col].dropna()
        log(f'  {col}: n={len(s)}, mean={s.mean():.3f}, std={s.std():.3f}, nan%={cgm[col].isna().mean()*100:.1f}%')

# ── DEXA KPIs (body_composition + bone_density) ───────────────────────────
log('\n=== DEXA KPIs ===')
bc = load_body_system_df('body_composition')
bd = load_body_system_df('bone_density')
for df_name, df, cols in [
    ('body_composition', bc, ['total_scan_vat_mass', 'total_scan_vat_area']),
    ('bone_density', bd, ['body_total_bmd', 'femur_neck_mean_bmd', 'spine_l1_l4_bmd']),
]:
    for col in cols:
        if col in df.columns:
            s = df[col].dropna()
            log(f'  {col} ({df_name}): n={len(s)}, mean={s.mean():.3f}, std={s.std():.3f}, nan%={df[col].isna().mean()*100:.1f}%')

# ── Retina KPIs — need clinical proxies since no pre-computed AVR/RNFL ────
log('\n=== Retina KPI candidates (cardiovascular proxies) ===')
cardio = load_body_system_df('cardiovascular_system')
for col in ['sitting_blood_pressure_systolic', 'lying_blood_pressure_systolic',
            'intima_media_th_mm_1_intima_media_thickness', 'intima_media_th_mm_2_intima_media_thickness']:
    if col in cardio.columns:
        s = cardio[col].dropna()
        log(f'  {col}: n={len(s)}, mean={s.mean():.3f}, std={s.std():.3f}, nan%={cardio[col].isna().mean()*100:.1f}%')

# ── Metabolites KPI candidates ─────────────────────────────────────────────
log('\n=== Metabolites KPI candidates ===')
met = load_body_system_df('metabolites_annotated')
log(f'Total metabolites: {len(met.columns)}, total rows: {len(met)}')
log(f'Visits: {met.index.get_level_values("research_stage").value_counts().to_dict()}')

# Compute stats for all metabolites and find most complete, most variable ones
stats = []
for col in met.columns:
    s = met[col].dropna()
    if len(s) > 1000:  # needs decent coverage
        stats.append({
            'metabolite': col,
            'n': len(s),
            'nan_pct': met[col].isna().mean() * 100,
            'cv': s.std() / abs(s.mean()) if s.mean() != 0 else 0,
            'mean': s.mean(),
        })

stats_df = pd.DataFrame(stats).sort_values('cv', ascending=False)
log(f'\nTop 20 most variable metabolites (n>1000):')
log(stats_df.head(20)[['metabolite', 'n', 'nan_pct', 'cv', 'mean']].to_string(index=False))

# Known clinically meaningful metabolites to search for
known = ['urate', 'tryptophan', 'kynurenine', 'glutamine', 'glucose', 'lactate',
         'citrate', 'succinate', 'phenylalanine', 'tyrosine', 'histidine', 'alanine',
         'leucine', 'isoleucine', 'valine', 'serine', 'threonine', 'methionine',
         'homocysteine', 'cysteine', 'arginine', 'proline', 'ornithine',
         'indole', 'serotonin', 'dopamine', 'cortisol', 'bile', 'acylcarnitine']
log(f'\nKnown clinical metabolites found:')
for term in known:
    hits = [c for c in met.columns if term.lower() in c.lower()]
    if hits:
        for h in hits[:2]:
            s = met[h].dropna()
            if len(s) > 500:
                log(f'  {h}: n={len(s)}, nan%={met[h].isna().mean()*100:.1f}%')

# ── Final multi-visit overlap with metabolites ────────────────────────────
log('\n=== Final multi-visit cohort size ===')
met_idx = set(met.index)
CGM_DIR = '/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/cgm'
cgm_subjects = set()
for f in os.listdir(CGM_DIR):
    if f.endswith('.txt') and '_' in f:
        sid = f.split('_')[0]
        if sid.isdigit():
            cgm_subjects.add('10K_' + sid)

dexa_records = []
with h5py.File('/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/dxa/dxa_dataset.h5', 'r') as f:
    for k in f.keys():
        parts = k.split('_')
        dexa_records.append({'RegistrationCode': '_'.join(parts[:2]), 'research_stage': '_'.join(parts[2:])})
dexa_df = pd.DataFrame(dexa_records)
dexa_idx = set(zip(dexa_df['RegistrationCode'], dexa_df['research_stage']))

retina_records = []
with h5py.File('/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/eyes/retina_dataset.h5', 'r') as f:
    for k in f.keys():
        parts = k.split('_')
        retina_records.append({'RegistrationCode': '_'.join(parts[:2]), 'research_stage': '_'.join(parts[2:])})
retina_df = pd.DataFrame(retina_records)
retina_idx = set(zip(retina_df['RegistrationCode'], retina_df['research_stage']))
retina_subjects = set(retina_df['RegistrationCode'])

def multi_subjects(idx, min_v=2):
    d = {}
    for r, s in idx:
        d.setdefault(r, []).append(s)
    return {r: v for r, v in d.items() if len(v) >= min_v}

dexa_multi = multi_subjects(dexa_idx)
met_multi  = multi_subjects(met_idx)

# Multi-visit overlap: DEXA multi + met multi + CGM + retina any
overlap = set(dexa_multi) & set(met_multi) & cgm_subjects & retina_subjects
log(f'Subjects with multi-visit DEXA + multi-visit Metabolites + CGM + Retina: {len(overlap)}')

# Also check: DEXA multi + metabolites any visit + CGM + retina any
met_subjects = set(r for r, v in met_idx)
overlap2 = set(dexa_multi) & met_subjects & cgm_subjects & retina_subjects
log(f'Subjects with multi-visit DEXA + Metabolites(any) + CGM + Retina: {len(overlap2)}')

# Visit details for overlap2
visit_detail = {}
for reg in overlap2:
    visits = dexa_multi.get(reg, [])
    visit_detail[reg] = sorted(visits)
vc = pd.Series([len(v) for v in visit_detail.values()]).value_counts()
log(f'Visit count distribution: {vc.to_dict()}')
log(f'Has baseline: {sum(1 for v in visit_detail.values() if "baseline" in v)}')
log(f'Has 02_00_visit: {sum(1 for v in visit_detail.values() if "02_00_visit" in v)}')

with open(OUT, 'w') as f:
    f.write('\n'.join(lines))
log(f'\nSaved to {OUT}')
