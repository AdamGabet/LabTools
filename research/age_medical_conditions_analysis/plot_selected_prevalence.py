#!/usr/bin/env python
"""Prevalence-by-age for selected conditions (3x3). Tunnel = 95% CI band, ages 40-70 only."""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
from body_system_loader.load_feature_df import load_body_system_df

FIGURES_DIR = '/home/adamgab/PycharmProjects/LabTools/research/age_medical_conditions_analysis/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

CONDITIONS = [
    'Osteoarthritis', 'Fatty Liver Disease', 'Prediabetes', 'Hyperlipidemia',
    'Hypertension', 'Ischemic Heart Disease', 'Hearing loss', 'Depression', 'Gout',
]

AGE_MIN, AGE_MAX = 40, 70
AGE_BINS = [40, 45, 50, 55, 60, 65, 70]


def wilson_ci(n_pos, n_total, z=1.96):
    """95% Wilson score interval for proportion (in [0,1])."""
    p = n_pos / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return np.clip(center - margin, 0, 1), np.clip(center + margin, 0, 1)


def calculate_prevalence_by_age(df, condition):
    """Prevalence by age bin with 95% CI (tunnel)."""
    d = df.copy()
    d['age_bin'] = pd.cut(d['age'], bins=AGE_BINS, right=False)
    prev = d.groupby('age_bin', observed=True).agg({condition: ['sum', 'count']})
    prev.columns = ['n_positive', 'n_total']
    prev['prevalence'] = prev['n_positive'] / prev['n_total']
    ci = prev.apply(lambda r: wilson_ci(r['n_positive'], r['n_total']), axis=1)
    prev['ci_lower'] = [x[0] for x in ci]
    prev['ci_upper'] = [x[1] for x in ci]
    prev['age_midpoint'] = [(i.left + i.right) / 2 for i in prev.index]
    return prev.reset_index()


def main():
    print('Loading data...')
    age_df = load_body_system_df('Age_Gender_BMI', specific_columns=['age', 'gender'])
    mc_df = load_body_system_df('medical_conditions')
    df = pd.merge(age_df, mc_df, left_index=True, right_index=True, how='inner')
    df = df[(df['age'] >= AGE_MIN) & (df['age'] <= AGE_MAX)]

    # Resolve condition names (must exist in data)
    all_cols = set(df.columns)
    to_plot = [c for c in CONDITIONS if c in all_cols]
    missing = [c for c in CONDITIONS if c not in all_cols]
    if missing:
        print(f'Warning: not in data: {missing}')
    if not to_plot:
        print('No conditions found in data.')
        return

    # 3x3 grid, prevalence tunnel (95% CI band)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    plt.rcParams['axes.labelcolor'] = '0.15'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    color = '#2980B9'

    for i, condition in enumerate(to_plot):
        ax = axes[i]
        prev = calculate_prevalence_by_age(df, condition)
        # Tunnel: fill between CI bounds (not 0 to prevalence)
        ax.fill_between(prev['age_midpoint'],
                        prev['ci_lower'] * 100,
                        prev['ci_upper'] * 100,
                        alpha=0.3, color=color)
        ax.plot(prev['age_midpoint'], prev['prevalence'] * 100, marker='o', linewidth=2, markersize=8, color=color)
        ax.set_title(condition, fontsize=16, fontweight='bold', color='0.15')
        ax.set_xlabel('Age (years)', fontsize=14, color='0.15')
        ax.set_ylabel('Prevalence (%)', fontsize=14, color='0.15')
        ax.tick_params(axis='both', labelsize=12, colors='0.15')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.set_xlim(AGE_MIN, AGE_MAX)

    for j in range(len(to_plot), 9):
        axes[j].set_visible(False)

    fig.suptitle(f'Prevalence by Age ({AGE_MIN}-{AGE_MAX} years)', fontsize=20, fontweight='bold', color='0.15', y=0.995)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, 'prevalence_selected_4x2.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
