#!/usr/bin/env python
"""Create 5 prevalence-by-age plots: one per disease group, top 5 conditions by n (n > 200)."""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
from body_system_loader.load_feature_df import load_body_system_df

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

MIN_CASES = 200  # n > 200
TOP_N = 10
AGE_MIN, AGE_MAX = 40, 70
# 3-year age bins, 40–70 inclusive
AGE_BINS = list(range(40, 71, 3))  # 40,43,...,69; add 71 so [67,71) includes 70
if AGE_BINS[-1] != 71:
    AGE_BINS.append(71)

# Exact groups from age_condition_analysis.py
DISEASE_GROUPS = {
    'Cardiovascular': [
        'Hypertension', 'Ischemic Heart Disease', 'Atrial Fibrillation',
        'Heart valve disease', 'AV. Conduction Disorder', 'Myocarditis',
        'Atherosclerotic'
    ],
    'Cancer': ['Breast Cancer', 'Melanoma', 'Lymphoma'],
    'Respiratory': ['Asthma', 'COPD'],
    'Metabolic/Endocrine': [
        'Diabetes', 'Prediabetes', 'Hyperlipidemia', 'Hypercholesterolaemia',
        'Obesity', 'Fatty Liver Disease', 'Hashimoto', 'Goiter',
        'Hyperparathyroidism', 'Thyroid Adenoma'
    ],
    'Neurodegenerative/Mental': [
        'Depression', 'Anxiety', 'PTSD', 'Insomnia',
        'Migraine', 'Headache'
    ],
    'Musculoskeletal': [
        'Osteoarthritis', 'Back Pain', 'Fibromyalgia', 'Fractures',
        'Gout', 'Meniscus Tears'
    ],
    'Gastrointestinal': [
        'IBD', 'IBS', 'Celiac', 'Peptic Ulcer Disease', 'Gallstone Disease',
        'Fatty Liver Disease', 'Haemorrhoids', 'Anal Fissure'
    ],
    'Autoimmune/Inflammatory': [
        'Psoriasis', 'Vitiligo', 'Hashimoto', 'Allergy', 'Atopic Dermatitis',
        'Uveitis', 'FMF'
    ],
    'Eye/Ear/Sensory': [
        'Glaucoma', 'Hearing loss', 'Tinnitus', 'Retinal detachment',
        'Ocular Hypertension'
    ],
    'Renal/Urological': [
        'Renal Stones', 'Urinary Tract Stones', 'Urinary tract infection'
    ]
}

# Display name for plot titles/filenames (Mental instead of Neurodegenerative/Mental)
GROUP_DISPLAY_NAMES = {
    'Neurodegenerative/Mental': 'Mental',
}


def wilson_ci(n_pos, n_total, z=1.96):
    """95% Wilson score interval for proportion (in [0,1])."""
    p = n_pos / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return center - margin, center + margin


def calculate_prevalence_by_age(df, condition):
    """Prevalence per age bin with 95% CI (Wilson)."""
    d = df.copy()
    d['age_bin'] = pd.cut(d['age'], bins=AGE_BINS, right=False)
    stats = d.groupby('age_bin', observed=True).agg({condition: ['sum', 'count']})
    stats.columns = ['n_positive', 'n_total']
    stats['prevalence'] = stats['n_positive'] / stats['n_total']
    ci = stats.apply(lambda r: wilson_ci(r['n_positive'], r['n_total']), axis=1)
    stats['ci_lower'] = [x[0] for x in ci]
    stats['ci_upper'] = [x[1] for x in ci]
    stats['age_midpoint'] = [(i.left + i.right) / 2 for i in stats.index]
    return stats.reset_index()


def main():
    print('Loading data...')
    age_df = load_body_system_df('Age_Gender_BMI', specific_columns=['age', 'gender'])
    mc_df = load_body_system_df('medical_conditions')
    df = pd.merge(age_df, mc_df, left_index=True, right_index=True, how='inner')
    df = df[(df['age'] >= AGE_MIN) & (df['age'] <= AGE_MAX)]
    all_conditions = [c for c in df.columns if c not in ['age', 'gender']]

    # Per-condition case counts (n > MIN_CASES)
    n_pos = df[all_conditions].sum()

    # For each group: top 5 conditions by n with n > MIN_CASES
    group_to_plot_conditions = {}
    for group in DISEASE_GROUPS:
        group_conditions = [c for c in DISEASE_GROUPS[group] if c in all_conditions]
        with_enough = [c for c in group_conditions if n_pos.get(c, 0) > MIN_CASES]
        top5 = sorted(with_enough, key=lambda c: n_pos.get(c, 0), reverse=True)[:TOP_N]
        if top5:
            group_to_plot_conditions[group] = top5

    # 5 plots: Cardiovascular, Metabolic/Endocrine, Mental, Musculoskeletal, Autoimmune
    groups_to_plot = [
        'Cardiovascular', 'Metabolic/Endocrine', 'Neurodegenerative/Mental',
        'Musculoskeletal', 'Autoimmune/Inflammatory'
    ]
    groups_to_plot = [g for g in groups_to_plot if g in group_to_plot_conditions]
    if len(groups_to_plot) < 5:
        print(f'Warning: only {len(groups_to_plot)} of 5 groups have conditions with n>{MIN_CASES}')

    colors = plt.cm.tab10(np.linspace(0, 1, TOP_N))
    for group in groups_to_plot:
        conditions = group_to_plot_conditions[group]
        display_name = GROUP_DISPLAY_NAMES.get(group, group)
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, condition in enumerate(conditions):
            prev = calculate_prevalence_by_age(df, condition)
            n = int(n_pos[condition])
            label = f'{condition} (n={n})'
            eps = 0.01  # for log scale
            y = prev['prevalence'] * 100 + eps
            ax.plot(prev['age_midpoint'], y, marker='o', linewidth=2,
                    markersize=5, color=colors[i], label=label)
        ax.set_xlabel('Age (years)', fontsize=11)
        ax.set_ylabel('Prevalence ', fontsize=11)
        ax.set_title(f'{display_name}: prevalence by Age {AGE_MIN}-{AGE_MAX} (top {len(conditions)}, n>{MIN_CASES})', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(AGE_MIN, AGE_MAX)
        plt.tight_layout()
        safe_name = display_name.replace('/', '_')
        fname = f'top5_prevalence_{safe_name}.png'
        fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: figures/{fname}')

    print('Done.')


if __name__ == '__main__':
    main()
