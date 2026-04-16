"""
Find the most longitudinally stable metabolites in metabolites_annotated
to replace urate + Tryptophan as KPIs.

Criteria:
  1. High baseline → 02_00_visit Pearson r (longitudinal stability)
  2. Good coverage at 02_00_visit (>= current urate/Tryptophan threshold)
  3. Clinical relevance (human interpretable)
  4. Still allows 120-subject eligible pool (all 8 KPIs at both visits)
"""
import sys
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
import pandas as pd
import numpy as np
import h5py
from body_system_loader.load_feature_df import load_body_system_df

OUTDIR = '/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline'

print('Loading metabolites_annotated...')
met_df = load_body_system_df('metabolites_annotated')
print(f'Shape: {met_df.shape}')
print(f'Visits: {met_df.index.get_level_values(1).value_counts().to_dict()}')

# Get baseline and 02_00_visit rows
base = met_df.xs('baseline',    level='research_stage')
v2   = met_df.xs('02_00_visit', level='research_stage')
common = list(set(base.index) & set(v2.index))
print(f'Subjects with both visits: {len(common)}')

b_sub = base.loc[common]
v2_sub = v2.loc[common]

# Compute longitudinal stability for every metabolite
print('Computing stability (Pearson r) for all metabolites...')
stab = []
for col in met_df.columns:
    n_base = base[col].notna().sum()
    n_v2   = v2[col].notna().sum()
    # only consider metabolites with decent v2 coverage
    if n_v2 < 800:
        continue
    r = b_sub[col].corr(v2_sub[col])
    if pd.isna(r):
        continue
    stab.append({'metabolite': col, 'r_stability': round(r, 4),
                 'n_baseline': n_base, 'n_v2': n_v2})

stab_df = pd.DataFrame(stab).sort_values('r_stability', ascending=False)
print(f'\n{len(stab_df)} metabolites with n_v2 >= 800')

print('\n=== Top 30 most stable metabolites ===')
print(stab_df.head(30).to_string(index=False))

stab_df.to_csv(f'{OUTDIR}/metabolite_stability.csv', index=False)
print(f'\nSaved: {OUTDIR}/metabolite_stability.csv')

# ── Now check: if we swap urate+Tryptophan for top-2, does pool hold? ────
print('\n=== Eligibility check for top KPI candidates ===')
cgm_df    = load_body_system_df('glycemic_status')
bc_df     = load_body_system_df('body_composition')
bd_df     = load_body_system_df('bone_density')
cardio_df = load_body_system_df('cardiovascular_system')
agb_df    = load_body_system_df('Age_Gender_BMI')

retina_records = []
with h5py.File('/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Data/10K/aws_lab_files/eyes/retina_dataset.h5', 'r') as f:
    for k in f.keys():
        parts = k.split('_')
        retina_records.append({'RegistrationCode': '_'.join(parts[:2]),
                               'research_stage': '_'.join(parts[2:])})
retina_subjects = set(pd.DataFrame(retina_records)['RegistrationCode'])

def idx_to_subjects(df, min_visits=1):
    counts = df.index.get_level_values(0).value_counts()
    return set(counts[counts >= min_visits].index)

dexa_multi = idx_to_subjects(bc_df, min_visits=2)
met_multi  = idx_to_subjects(met_df, min_visits=2)
cgm_any    = idx_to_subjects(cgm_df, min_visits=1)
pool_base  = dexa_multi & met_multi & cgm_any & retina_subjects

OTHER_KPIS = ['bt__hba1c', 'bt__glucose', 'total_scan_vat_mass', 'body_total_bmd',
              'sitting_blood_pressure_systolic',
              'intima_media_th_mm_1_intima_media_thickness']

def eligible_pool_size(met_kpi1, met_kpi2):
    def get_complete(visit):
        def _xs(df, cols):
            avail = [c for c in cols if c in df.columns]
            if visit not in df.index.get_level_values(1).unique():
                return pd.DataFrame(columns=avail)
            return df.xs(visit, level='research_stage')[avail]
        cgm_v    = _xs(cgm_df,    ['bt__hba1c', 'bt__glucose'])
        bc_v     = _xs(bc_df,     ['total_scan_vat_mass'])
        bd_v     = _xs(bd_df,     ['body_total_bmd'])
        cardio_v = _xs(cardio_df, ['sitting_blood_pressure_systolic',
                                    'intima_media_th_mm_1_intima_media_thickness'])
        met_v    = _xs(met_df,    [met_kpi1, met_kpi2])
        all_kpis = OTHER_KPIS + [met_kpi1, met_kpi2]
        merged = cgm_v.join(bc_v, how='outer').join(bd_v, how='outer') \
                      .join(cardio_v, how='outer').join(met_v, how='outer')
        present = [k for k in all_kpis if k in merged.columns]
        return set(merged[merged[present].notna().all(axis=1)].index)

    s_base = get_complete('baseline')
    s_v2   = get_complete('02_00_visit')
    both   = s_base & s_v2
    pool   = pool_base & both

    agb_base = agb_df.groupby(level=0).first()
    pool_agb = agb_base.loc[agb_base.index.isin(pool)].copy()
    pool_agb = pool_agb.dropna(subset=['age', 'gender', 'bmi'])
    pool_agb = pool_agb[(pool_agb['age'] >= 40) & (pool_agb['age'] <= 70)]
    pool_agb['age_bin'] = pd.cut(pool_agb['age'], bins=[40, 50, 60, 70],
                                  labels=['40-50', '50-60', '60-70'], right=False)
    pool_agb['gender_bin'] = pool_agb['gender'].map({0.0: 'F', 1.0: 'M', 0: 'F', 1: 'M'})
    pool_agb = pool_agb.dropna(subset=['age_bin', 'gender_bin'])
    min_cell = pool_agb.groupby(['age_bin', 'gender_bin'], observed=True).size().min() \
               if len(pool_agb) > 0 else 0
    return len(pool), len(pool_agb), int(min_cell)

# Check current KPIs
pool_n, age_n, min_cell = eligible_pool_size('urate', 'Tryptophan')
print(f'\n  Current (urate + Tryptophan): pool={pool_n}, age-filtered={age_n}, min_cell={min_cell}')

# Check top candidates from stability analysis
top_candidates = stab_df.head(20)['metabolite'].tolist()
print('\n  Testing top 20 stable metabolites as KPI pairs (paired with most stable):')
top1 = stab_df.iloc[0]['metabolite']
print(f'\n  Anchoring on #{1}: {top1} (r={stab_df.iloc[0]["r_stability"]:.3f})')
candidate_results = []
for i, row in stab_df.head(20).iterrows():
    met2 = row['metabolite']
    if met2 == top1:
        continue
    pool_n, age_n, min_cell = eligible_pool_size(top1, met2)
    candidate_results.append({'kpi1': top1, 'kpi2': met2,
                               'r1': stab_df.iloc[0]['r_stability'],
                               'r2': row['r_stability'],
                               'pool': pool_n, 'age_filtered': age_n,
                               'min_cell': min_cell, 'viable': min_cell >= 20})
    marker = '✓' if min_cell >= 20 else '✗'
    print(f'    {marker} {top1} + {met2}: pool={pool_n}, age_n={age_n}, min_cell={min_cell}')

cand_df = pd.DataFrame(candidate_results)
viable = cand_df[cand_df['viable']]
print(f'\n  Viable pairs (min_cell >= 20): {len(viable)}')
if len(viable) > 0:
    print(viable[['kpi1', 'kpi2', 'r1', 'r2', 'pool', 'age_filtered', 'min_cell']].to_string(index=False))

cand_df.to_csv(f'{OUTDIR}/metabolite_kpi_candidates.csv', index=False)
print(f'\nSaved: {OUTDIR}/metabolite_kpi_candidates.csv')
