"""
Build multimodal benchmark dataset.

4 Modalities:
  - CGM         : glycemic_status tabular (48 features)
  - DEXA        : body_composition + bone_density (tabular from DEXA scans)
  - Retina proxy: cardiovascular_system (BP, IMT, ECG — retinal-vascular proxies)
  - Metabolites : metabolites_annotated (133 features)

8 KPIs (2 per modality):
  CGM        : bt__hba1c, bt__glucose
  DEXA       : total_scan_vat_mass, body_total_bmd
  Retina     : sitting_blood_pressure_systolic, intima_media_th_mm_1_intima_media_thickness
  Metabolites: urate, Bilirubin

Eligibility: ALL 8 KPIs must be non-null at baseline AND at 02_00_visit (hard requirement).
             Also: retina any visit, multi-visit DEXA, multi-visit metabolites, CGM any visit.
             Only visit rows with all 8 KPIs non-null are included in the output dataset.

120 subjects stratified by age (40-50 / 50-60 / 60-70) × gender (0/1) → 20 per cell.
Subject-level train/test split (80/20) stratified by age+gender.
"""
import sys, os, h5py
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
import pandas as pd
import numpy as np
from body_system_loader.load_feature_df import load_body_system_df

OUTDIR = '/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline'
SEED   = 42
rng    = np.random.default_rng(SEED)

# ── KPI definitions ───────────────────────────────────────────────────────
KPIS = {
    'cgm':       ['bt__hba1c', 'bt__glucose'],
    'dexa':      ['total_scan_vat_mass', 'body_total_bmd'],
    'retina':    ['sitting_blood_pressure_systolic',
                  'intima_media_th_mm_1_intima_media_thickness'],
    'metabolites': ['urate', 'Bilirubin_Lumirubin_(4E 15E)-Bilirubin_(4E 15Z)-Bilirubin'],
}
ALL_KPIS = [kpi for kpis in KPIS.values() for kpi in kpis]

print('Loading body systems...')
cgm_df    = load_body_system_df('glycemic_status')
bc_df     = load_body_system_df('body_composition')
bd_df     = load_body_system_df('bone_density')
cardio_df = load_body_system_df('cardiovascular_system')
met_df    = load_body_system_df('metabolites_annotated')
agb_df    = load_body_system_df('Age_Gender_BMI')

# ── Identify eligible subjects (multi-visit DEXA, multi-visit metabolites,
#    CGM any visit, retina any visit) ─────────────────────────────────────
print('Identifying eligible subjects...')

def idx_to_subjects(df, min_visits=1):
    counts = df.index.get_level_values(0).value_counts()
    return set(counts[counts >= min_visits].index)

def subject_visits(df):
    d = {}
    for reg, stage in df.index:
        d.setdefault(reg, []).append(stage)
    return d

# Raw retina subject list from H5
retina_records = []
with h5py.File('/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/eyes/retina_dataset.h5', 'r') as f:
    for k in f.keys():
        parts = k.split('_')
        retina_records.append({'RegistrationCode': '_'.join(parts[:2]), 'research_stage': '_'.join(parts[2:])})
retina_subjects = set(pd.DataFrame(retina_records)['RegistrationCode'])

dexa_multi    = idx_to_subjects(bc_df, min_visits=2)   # 2+ DEXA visits
met_multi     = idx_to_subjects(met_df, min_visits=2)  # 2+ metabolite visits
cgm_any       = idx_to_subjects(cgm_df, min_visits=1)

pool = dexa_multi & met_multi & cgm_any & retina_subjects
print(f'Eligible pool (modality coverage): {len(pool)} subjects')

# ── Hard filter: ALL 8 KPIs must be non-null at baseline AND 02_00_visit ─
print('Filtering for full KPI coverage at baseline and 02_00_visit...')

def get_visit_kpis(visit):
    def _xs(df, cols):
        if visit in df.index.get_level_values(1).unique():
            return df.xs(visit, level='research_stage')[cols]
        return pd.DataFrame(columns=cols)
    cgm_v    = _xs(cgm_df,    ['bt__hba1c', 'bt__glucose'])
    bc_v     = _xs(bc_df,     ['total_scan_vat_mass'])
    bd_v     = _xs(bd_df,     ['body_total_bmd'])
    cardio_v = _xs(cardio_df, ['sitting_blood_pressure_systolic',
                                'intima_media_th_mm_1_intima_media_thickness'])
    met_v    = _xs(met_df,    ['urate', 'Bilirubin_Lumirubin_(4E 15E)-Bilirubin_(4E 15Z)-Bilirubin'])
    merged = cgm_v.join(bc_v, how='outer').join(bd_v, how='outer') \
                  .join(cardio_v, how='outer').join(met_v, how='outer')
    return set(merged[merged[ALL_KPIS].notna().all(axis=1)].index)

kpis_at_baseline = get_visit_kpis('baseline')
kpis_at_v2       = get_visit_kpis('02_00_visit')
print(f'Subjects with all 8 KPIs at baseline: {len(kpis_at_baseline)}')
print(f'Subjects with all 8 KPIs at 02_00_visit: {len(kpis_at_v2)}')
print(f'Subjects with all 8 KPIs at both: {len(kpis_at_baseline & kpis_at_v2)}')

pool = pool & kpis_at_baseline & kpis_at_v2
print(f'Eligible pool after KPI filter: {len(pool)} subjects')

# ── Attach demographics and stratify ─────────────────────────────────────
print('Attaching demographics...')
agb_base = agb_df.xs('baseline', level='research_stage', drop_level=True) \
    if 'baseline' in agb_df.index.get_level_values(1).unique() \
    else agb_df.groupby(level=0).first()

pool_agb = agb_base.loc[agb_base.index.isin(pool)].copy()
pool_agb = pool_agb.dropna(subset=['age', 'gender', 'bmi'])
print(f'Pool with demographics: {len(pool_agb)} subjects')

# Age bins 40-70 (per CLAUDE.md guideline)
pool_agb = pool_agb[(pool_agb['age'] >= 40) & (pool_agb['age'] <= 70)]
print(f'Pool age 40-70: {len(pool_agb)} subjects')

pool_agb['age_bin'] = pd.cut(pool_agb['age'], bins=[40, 50, 60, 70],
                              labels=['40-50', '50-60', '60-70'], right=False)
pool_agb['gender_bin'] = pool_agb['gender'].map({0.0: 'F', 1.0: 'M', 0: 'F', 1: 'M'})
pool_agb = pool_agb.dropna(subset=['age_bin', 'gender_bin'])

print('\nPool by age × gender:')
print(pool_agb.groupby(['age_bin', 'gender_bin'], observed=True).size().unstack())

# ── Stratified sampling: 20 per cell (3 age × 2 gender = 6 cells) ────────
print('\nSelecting 120 subjects (20 per age×gender cell)...')
selected = []
for age_bin in ['40-50', '50-60', '60-70']:
    for gender_bin in ['F', 'M']:
        cell = pool_agb[(pool_agb['age_bin'] == age_bin) & (pool_agb['gender_bin'] == gender_bin)]
        n_available = len(cell)
        n_select = min(20, n_available)
        chosen = cell.sample(n=n_select, random_state=SEED)
        print(f'  {age_bin} {gender_bin}: {n_available} available, selected {n_select}')
        selected.append(chosen)

selected_df = pd.concat(selected)
print(f'\nTotal selected: {len(selected_df)} subjects')
print(f'Age: {selected_df["age"].mean():.1f} ± {selected_df["age"].std():.1f}')
print(f'BMI: {selected_df["bmi"].mean():.1f} ± {selected_df["bmi"].std():.1f}')
print(f'Gender: {selected_df["gender_bin"].value_counts().to_dict()}')

# ── Train/test split at subject level (stratified, 80/20) ────────────────
print('\nAssigning train/test split...')
selected_df['split'] = 'train'
for age_bin in ['40-50', '50-60', '60-70']:
    for gender_bin in ['F', 'M']:
        cell_idx = selected_df[(selected_df['age_bin'] == age_bin) &
                                (selected_df['gender_bin'] == gender_bin)].index
        n_test = max(1, round(len(cell_idx) * 0.2))
        test_chosen = rng.choice(cell_idx, size=n_test, replace=False)
        selected_df.loc[test_chosen, 'split'] = 'test'

print(f'Train: {(selected_df["split"]=="train").sum()}, Test: {(selected_df["split"]=="test").sum()}')

subject_ids = set(selected_df.index)
id_to_split = selected_df['split'].to_dict()

# ── Build per-subject-visit rows ──────────────────────────────────────────
print('\nBuilding subject-visit rows...')
# Collect all available visits for selected subjects across all modalities
cgm_sv    = subject_visits(cgm_df)
bc_sv     = subject_visits(bc_df)
met_sv    = subject_visits(met_df)
cardio_sv = subject_visits(cardio_df)

rows = []
for subj in subject_ids:
    # Collect all visits where at least one modality is present
    all_visits = set(cgm_sv.get(subj, [])) | set(bc_sv.get(subj, [])) | \
                 set(met_sv.get(subj, [])) | set(cardio_sv.get(subj, []))
    for visit in sorted(all_visits):
        rows.append({'RegistrationCode': subj, 'research_stage': visit})

idx_df = pd.DataFrame(rows).set_index(['RegistrationCode', 'research_stage'])
print(f'Total subject-visit rows: {len(idx_df)}')
print(f'Visit distribution: {idx_df.index.get_level_values(1).value_counts().to_dict()}')

# ── Merge all columns in one shot via concat ──────────────────────────────
print('\nMerging all columns via concat...')
kpi_set  = set(ALL_KPIS)
seen     = set(ALL_KPIS)  # track cols already assigned to avoid duplicates

sources  = [cgm_df, bc_df, bd_df, cardio_df, met_df]
parts    = [idx_df]

# KPIs first (explicit)
parts.append(cgm_df[['bt__hba1c', 'bt__glucose']])
parts.append(bc_df[['total_scan_vat_mass']])
parts.append(bd_df[['body_total_bmd']])
parts.append(cardio_df[['sitting_blood_pressure_systolic',
                         'intima_media_th_mm_1_intima_media_thickness']])
parts.append(met_df[['urate', 'Bilirubin_Lumirubin_(4E 15E)-Bilirubin_(4E 15Z)-Bilirubin']])

# Feature cols (non-KPI, no duplicates across sources)
for src_df in sources:
    feat_cols = [c for c in src_df.columns if c not in seen]
    seen.update(feat_cols)
    if feat_cols:
        parts.append(src_df[feat_cols])

# Left-join all parts onto the subject-visit index
rows_df = parts[0]
for part in parts[1:]:
    new_cols = [c for c in part.columns if c not in rows_df.columns]
    if new_cols:
        rows_df = rows_df.join(part[new_cols], how='left')
rows_df = rows_df.copy()  # defragment

# ── Drop visit rows missing any KPI ──────────────────────────────────────
before = len(rows_df)
rows_df = rows_df[rows_df[ALL_KPIS].notna().all(axis=1)]
print(f'Dropped {before - len(rows_df)} rows with missing KPIs → {len(rows_df)} rows remain')
print(f'Visit distribution after KPI filter: {rows_df.index.get_level_values(1).value_counts().to_dict()}')

# ── Add demographics + split + modality availability flags ────────────────
print('Adding metadata...')
rows_df = rows_df.join(agb_df[['age', 'gender', 'bmi']], how='left')

# Split assignment
rows_df['split'] = rows_df.index.get_level_values(0).map(id_to_split)

# Modality availability flags
rows_df['has_cgm']        = ~rows_df['bt__hba1c'].isna()
rows_df['has_dexa']       = ~rows_df['total_scan_vat_mass'].isna()
rows_df['has_retina']     = rows_df.index.get_level_values(0).isin(retina_subjects)
rows_df['has_metabolites']= ~rows_df['urate'].isna()

# ── Compute number of visits per modality per subject ────────────────────
for mod, flag in [('cgm','has_cgm'), ('dexa','has_dexa'), ('metabolites','has_metabolites')]:
    n_visits = rows_df[rows_df[flag]].groupby(level=0).size().rename(f'n_visits_{mod}')
    rows_df = rows_df.join(n_visits, how='left')

# ── Save main CSV ─────────────────────────────────────────────────────────
out_csv = os.path.join(OUTDIR, 'benchmark_subjects.csv')
rows_df = rows_df.reset_index()
rows_df.to_csv(out_csv, index=False)
print(f'\nSaved: {out_csv}')
print(f'Shape: {rows_df.shape}')
print(f'\nKPI coverage:')
for kpi in ALL_KPIS:
    n = rows_df[kpi].notna().sum()
    pct = n / len(rows_df) * 100
    print(f'  {kpi}: {n}/{len(rows_df)} rows ({pct:.0f}%)')

print('\nModality flags:')
for flag in ['has_cgm', 'has_dexa', 'has_retina', 'has_metabolites']:
    n = rows_df[flag].sum()
    print(f'  {flag}: {n} rows')

print('\nVisit distribution by modality:')
for mod in ['cgm', 'dexa', 'metabolites']:
    col = f'n_visits_{mod}'
    if col in rows_df.columns:
        vc = rows_df.drop_duplicates('RegistrationCode')[col].value_counts().sort_index()
        print(f'  {mod}: {vc.to_dict()}')

# Save a compact KPI-only CSV too
kpi_cols = ['RegistrationCode', 'research_stage', 'split', 'age', 'gender', 'bmi',
            'has_cgm', 'has_dexa', 'has_retina', 'has_metabolites'] + ALL_KPIS
kpi_csv = os.path.join(OUTDIR, 'benchmark_kpis_only.csv')
rows_df[kpi_cols].to_csv(kpi_csv, index=False)
print(f'\nKPI-only CSV: {kpi_csv}')
