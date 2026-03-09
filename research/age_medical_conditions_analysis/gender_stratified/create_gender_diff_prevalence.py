#!/usr/bin/env python
"""Simple prevalence plot for conditions with largest gender differences in age effect."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import sys

sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
from body_system_loader.load_feature_df import load_body_system_df

OUT = '/home/adamgab/PycharmProjects/LabTools/research/age_medical_conditions_analysis/gender_stratified'

# The 12 conditions from summary_gender_differences.png
CONDITIONS = [
    'Gout', 'Fatty Liver Disease', 'Sleep Apnea', 'Ischemic Heart Disease',
    'Anemia', 'Retinal detachment', 'Insomnia', 'Headache',
    'Osteoarthritis', 'Fractures', 'B12 Deficiency', 'Renal Stones',
]

print("Loading data...")
age_df = load_body_system_df('Age_Gender_BMI', specific_columns=['age', 'gender'])
mc_df = load_body_system_df('medical_conditions')
df = pd.merge(age_df, mc_df, left_index=True, right_index=True, how='inner')
df = df[(df['age'] >= 40) & (df['age'] <= 70)]
print(f"N = {len(df)} | Male: {(df['gender']==1).sum()} | Female: {(df['gender']==0).sum()}")


def calc_prevalence(df, condition, gender_val):
    """Prevalence (%) per age bin with Wilson 95% CI."""
    d = df[df['gender'] == gender_val].copy()
    bins = [40, 45, 50, 55, 60, 65, 70]
    d['bin'] = pd.cut(d['age'], bins=bins, right=False)

    stats = d.groupby('bin', observed=True).agg({condition: ['sum', 'count']})
    stats.columns = ['pos', 'n']
    if stats['n'].sum() < 50:
        return None

    # Prevalence + Wilson CI
    p = stats['pos'] / stats['n']
    n = stats['n']
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom

    stats['prevalence'] = 100 * p
    stats['ci_lower'] = 100 * np.maximum(0, center - margin)
    stats['ci_upper'] = 100 * np.minimum(1, center + margin)
    stats['ci_err'] = stats['prevalence'] - stats['ci_lower']
    stats['age_mid'] = [(i.left + i.right) / 2 for i in stats.index]
    return stats.reset_index()


# --- Summary plot: 4 rows x 3 columns, larger text ---
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
axes = axes.flatten()

for i, cond in enumerate(CONDITIONS):
    ax = axes[i]
    stats_m = calc_prevalence(df, cond, 1)
    stats_f = calc_prevalence(df, cond, 0)

    if stats_m is not None:
        ax.errorbar(stats_m['age_mid'], stats_m['prevalence'], yerr=stats_m['ci_err'],
                    fmt='o-', markersize=10, capsize=5, color='#2980B9', alpha=0.7,
                    linewidth=1.5, label='Male')
    if stats_f is not None:
        ax.errorbar(stats_f['age_mid'], stats_f['prevalence'], yerr=stats_f['ci_err'],
                    fmt='s-', markersize=10, capsize=5, color='#E74C3C', alpha=0.7,
                    linewidth=1.5, label='Female')

    ax.set_title(cond, fontsize=18, fontweight='bold')
    ax.legend(fontsize=15, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Age', fontsize=16)
    ax.set_ylabel('Prevalence (%)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(bottom=0)

fig.suptitle('Conditions with Largest Gender Differences - Prevalence\n'
             '(Blue=Male, Red=Female, Error bars=95% CI)',
             fontsize=22, fontweight='bold', y=0.995)
plt.tight_layout()
out_path = os.path.join(OUT, 'summary_gender_diff_prevalence.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
