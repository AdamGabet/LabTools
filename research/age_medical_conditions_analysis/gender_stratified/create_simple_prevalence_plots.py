#!/usr/bin/env python
"""Create simple gender-stratified prevalence plots - NO FITTED LINES, just raw data."""
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
os.makedirs(OUT, exist_ok=True)

MIN_N = 100

print("Loading data...")
age_df = load_body_system_df('Age_Gender_BMI', specific_columns=['age', 'gender'])
mc_df = load_body_system_df('medical_conditions')
df = pd.merge(age_df, mc_df, left_index=True, right_index=True, how='inner')
df = df[(df['age'] >= 40) & (df['age'] <= 70)]

# Gender: 0 = Female, 1 = Male
print(f"Total N = {len(df)}")
print(f"Gender distribution: {df['gender'].value_counts().to_dict()}")
df['gender_label'] = df['gender'].map({0: 'Female', 1: 'Male'})
print(f"Female: {(df['gender']==0).sum()}, Male: {(df['gender']==1).sum()}")


def calc_prevalence_by_age_gender(df, condition, gender_val):
    """Calculate prevalence (percentage) for each age bin for one gender."""
    d = df[df['gender'] == gender_val].copy()
    bins = [40, 45, 50, 55, 60, 65, 70]
    d['bin'] = pd.cut(d['age'], bins=bins, right=False)

    stats = d.groupby('bin', observed=True).agg({condition: ['sum', 'count']})
    stats.columns = ['pos', 'n']

    # Skip if too few cases
    if stats['n'].sum() < 50:
        return None

    # Calculate prevalence as percentage
    stats['prevalence'] = 100 * stats['pos'] / stats['n']
    
    # Wilson score confidence interval for binomial proportion
    p = stats['pos'] / stats['n']
    n = stats['n']
    z = 1.96  # 95% CI
    
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n)) / denominator
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
    
    stats['ci_lower'] = 100 * np.maximum(0, center - margin)
    stats['ci_upper'] = 100 * np.minimum(1, center + margin)
    stats['ci_error'] = stats['prevalence'] - stats['ci_lower']  # symmetric for plotting
    
    stats['age_mid'] = [(i.left + i.right) / 2 for i in stats.index]

    return stats.reset_index()


def plot_simple_prevalence(df, condition, save_path):
    """Create simple prevalence plot - NO FITTED LINES."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate for each gender
    stats_m = calc_prevalence_by_age_gender(df, condition, 1)  # Male
    stats_f = calc_prevalence_by_age_gender(df, condition, 0)  # Female

    # Plot Male (Blue) - points with error bars and connecting lines
    if stats_m is not None:
        n_male = int(stats_m['pos'].sum())
        ax.errorbar(stats_m['age_mid'], stats_m['prevalence'], yerr=stats_m['ci_error'],
                    fmt='o-', markersize=10, capsize=5, capthick=2,
                    color='#2980B9', ecolor='#2980B9', alpha=0.8, linewidth=2,
                    label=f"Male (n={n_male})")

    # Plot Female (Red) - points with error bars and connecting lines
    if stats_f is not None:
        n_female = int(stats_f['pos'].sum())
        ax.errorbar(stats_f['age_mid'], stats_f['prevalence'], yerr=stats_f['ci_error'],
                    fmt='s-', markersize=10, capsize=5, capthick=2,
                    color='#E74C3C', ecolor='#E74C3C', alpha=0.8, linewidth=2,
                    label=f"Female (n={n_female})")

    ax.set_xlabel('Age (years)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Prevalence (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'{condition}\nPrevalence by Age, Stratified by Gender', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_ylim(bottom=0)
    
    # Set x-axis ticks
    ax.set_xticks([40, 45, 50, 55, 60, 65, 70])
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return stats_m, stats_f


# Get conditions with enough cases in both genders
print("\nIdentifying conditions with n>=100 in at least one gender...")
condition_cols = [c for c in df.columns if c not in ['age', 'gender', 'gender_label']]

valid_conditions = []
for cond in condition_cols:
    n_male = df[df['gender'] == 1][cond].sum()
    n_female = df[df['gender'] == 0][cond].sum()
    n_total = n_male + n_female
    if n_total >= MIN_N and (n_male >= 20 or n_female >= 20):
        valid_conditions.append({
            'condition': cond,
            'n_male': int(n_male),
            'n_female': int(n_female),
            'n_total': int(n_total)
        })

valid_df = pd.DataFrame(valid_conditions).sort_values('n_total', ascending=False)
print(f"Conditions to plot: {len(valid_df)}")

# Generate plots
print("\nGenerating simple prevalence plots (no fitted lines)...")
results = []

for _, row in valid_df.iterrows():
    cond = row['condition']
    fname = f"simple_prev_{cond.replace(' ', '_').replace('/', '_')}.png"
    fpath = os.path.join(OUT, fname)

    try:
        stats_m, stats_f = plot_simple_prevalence(df, cond, fpath)
        print(f"  Saved: {fname}")

        results.append({
            'condition': cond,
            'n_male': row['n_male'],
            'n_female': row['n_female']
        })
    except Exception as e:
        print(f"  Error with {cond}: {e}")

# Create summary plot
print("\nCreating summary comparison plot...")

# Select top conditions by total prevalence
top_conditions = valid_df.nlargest(12, 'n_total')

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for i, (_, row) in enumerate(top_conditions.iterrows()):
    if i >= 12:
        break
    cond = row['condition']
    ax = axes[i]

    stats_m = calc_prevalence_by_age_gender(df, cond, 1)
    stats_f = calc_prevalence_by_age_gender(df, cond, 0)

    if stats_m is not None:
        ax.errorbar(stats_m['age_mid'], stats_m['prevalence'], yerr=stats_m['ci_error'],
                    fmt='o-', markersize=7, capsize=4, color='#2980B9', alpha=0.7, linewidth=1.5,
                    label='Male')

    if stats_f is not None:
        ax.errorbar(stats_f['age_mid'], stats_f['prevalence'], yerr=stats_f['ci_error'],
                    fmt='s-', markersize=7, capsize=4, color='#E74C3C', alpha=0.7, linewidth=1.5,
                    label='Female')

    ax.set_title(cond, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Age', fontsize=10)
    ax.set_ylabel('Prevalence (%)', fontsize=10)
    ax.set_ylim(bottom=0)

fig.suptitle('Top Conditions by Prevalence - Gender Stratified\n(Blue=Male, Red=Female, Error bars=95% CI)',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'summary_simple_prevalence.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: summary_simple_prevalence.png")

print("\n" + "="*50)
print("DONE!")
print("="*50)
print(f"Output folder: {OUT}")
print(f"Individual plots: {len(results)} conditions")
print(f"Summary plot: summary_simple_prevalence.png")
