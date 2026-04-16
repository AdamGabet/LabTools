"""
Exploration: Can we replace metabolites_annotated with proteomics or nightingale?

Key constraint: all 8 KPIs must be non-null at baseline AND 02_00_visit.
CGM blood tests (bt__hba1c, bt__glucose) are only ~22-26% coverage at 02_00_visit,
so any 4th modality with sparse 02_00_visit data will fail the eligibility filter.

Run this script to reproduce the exploration findings.
"""
import sys
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
import pandas as pd
import numpy as np
import h5py
from body_system_loader.load_feature_df import load_body_system_df

OUTDIR = '/home/adamgab/PycharmProjects/LabTools/research/multimodal_baseline'

print('Loading core modalities...')
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
cgm_any    = idx_to_subjects(cgm_df, min_visits=1)
core3      = dexa_multi & cgm_any & retina_subjects

# ── Summary of candidate 4th modalities ──────────────────────────────────
print('\n=== 4th modality candidate comparison ===')
results = []
for system, kpis in [
    ('metabolites_annotated', ['urate', 'Tryptophan']),
    ('proteomics',            ['IL6', 'FGF21']),
    ('nightingale',           ['HDL_C', 'GlycA']),
]:
    try:
        df = load_body_system_df(system)
        mod_multi = idx_to_subjects(df, min_visits=2)
        pool = core3 & mod_multi

        # KPI coverage at each visit
        def get_complete(visit, kpis):
            def _xs(src, cols):
                avail = [c for c in cols if c in src.columns]
                if visit in src.index.get_level_values(1).unique():
                    return src.xs(visit, level='research_stage')[avail]
                return pd.DataFrame(columns=avail)
            cgm_v    = _xs(cgm_df,    ['bt__hba1c', 'bt__glucose'])
            bc_v     = _xs(bc_df,     ['total_scan_vat_mass'])
            bd_v     = _xs(bd_df,     ['body_total_bmd'])
            cardio_v = _xs(cardio_df, ['sitting_blood_pressure_systolic',
                                        'intima_media_th_mm_1_intima_media_thickness'])
            mod_v    = _xs(df,        kpis)
            all_kpis = ['bt__hba1c', 'bt__glucose', 'total_scan_vat_mass', 'body_total_bmd',
                        'sitting_blood_pressure_systolic',
                        'intima_media_th_mm_1_intima_media_thickness'] + [k for k in kpis if k in df.columns]
            merged = cgm_v.join(bc_v, how='outer').join(bd_v, how='outer') \
                          .join(cardio_v, how='outer').join(mod_v, how='outer')
            present = [k for k in all_kpis if k in merged.columns]
            return set(merged[merged[present].notna().all(axis=1)].index)

        s_base = get_complete('baseline', kpis)
        s_v2   = get_complete('02_00_visit', kpis)
        both   = s_base & s_v2
        pool_both = pool & both

        # Age-filtered
        agb_base = agb_df.groupby(level=0).first()
        pool_agb = agb_base.loc[agb_base.index.isin(pool_both)].copy()
        pool_agb = pool_agb.dropna(subset=['age', 'gender', 'bmi'])
        pool_agb = pool_agb[(pool_agb['age'] >= 40) & (pool_agb['age'] <= 70)]
        pool_agb['age_bin'] = pd.cut(pool_agb['age'], bins=[40, 50, 60, 70],
                                      labels=['40-50', '50-60', '60-70'], right=False)
        pool_agb['gender_bin'] = pool_agb['gender'].map({0.0: 'F', 1.0: 'M', 0: 'F', 1: 'M'})
        pool_agb = pool_agb.dropna(subset=['age_bin', 'gender_bin'])
        min_cell = pool_agb.groupby(['age_bin', 'gender_bin'], observed=True).size().min() if len(pool_agb) > 0 else 0

        results.append({
            'system': system, 'kpis': '+'.join(kpis),
            'pool_modality': len(pool), 'complete_both_visits': len(both),
            'eligible_pool': len(pool_both), 'age_filtered': len(pool_agb),
            'min_cell': min_cell, 'feasible_120': min_cell >= 20
        })
        print(f'\n{system} ({"+".join(kpis)}):')
        print(f'  Modality pool: {len(pool)}')
        print(f'  Complete at both visits: {len(both)}')
        print(f'  Eligible pool: {len(pool_both)} | age-filtered: {len(pool_agb)} | min cell: {min_cell}')
    except Exception as e:
        print(f'\n{system}: ERROR — {e}')

res_df = pd.DataFrame(results)
print('\n=== Summary ===')
print(res_df.to_string(index=False))
res_df.to_csv(f'{OUTDIR}/modality_candidate_comparison.csv', index=False)
print(f'\nSaved: {OUTDIR}/modality_candidate_comparison.csv')
