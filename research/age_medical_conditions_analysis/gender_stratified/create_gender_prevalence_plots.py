#!/usr/bin/env python
"""Create gender-stratified prevalence plots (actual percentages)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
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


def calc_regression_by_gender(df, condition, gender_val):
    """Calculate logistic regression for one gender."""
    d = df[df['gender'] == gender_val].copy()
    valid = d[[condition, 'age']].dropna()

    if len(valid) < 50 or valid[condition].sum() < 10:
        return {'coef': np.nan, 'p': np.nan, 'n_pos': 0}

    X = sm.add_constant(valid['age'].values.reshape(-1, 1))
    try:
        m = sm.Logit(valid[condition].values, X).fit(disp=0, maxiter=100)
        return {
            'coef': m.params[1],
            'se': m.bse[1],
            'p': m.pvalues[1],
            'n_pos': int(valid[condition].sum()),
            'n': len(valid),
            'intercept': m.params[0]
        }
    except:
        return {'coef': np.nan, 'p': np.nan, 'n_pos': 0}


def plot_gender_stratified_prevalence(df, condition, save_path):
    """Create gender-stratified prevalence plot."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Calculate for each gender
    stats_m = calc_prevalence_by_age_gender(df, condition, 1)  # Male
    stats_f = calc_prevalence_by_age_gender(df, condition, 0)  # Female
    reg_m = calc_regression_by_gender(df, condition, 1)
    reg_f = calc_regression_by_gender(df, condition, 0)

    x_fit = np.linspace(40, 70, 100)

    # Plot Male (Blue)
    if stats_m is not None and not np.isnan(reg_m['coef']):
        # Points with error bars
        ax.errorbar(stats_m['age_mid'], stats_m['prevalence'], yerr=stats_m['ci_error'],
                    fmt='o', markersize=8, capsize=4, capthick=1.5,
                    color='#2980B9', ecolor='#2980B9', alpha=0.7)

        # Fitted line (convert from log odds to probability)
        log_odds_fit_m = reg_m['intercept'] + reg_m['coef'] * x_fit
        prob_fit_m = 100 / (1 + np.exp(-log_odds_fit_m))

        p_str = f"p={reg_m['p']:.2e}" if reg_m['p'] < 0.001 else f"p={reg_m['p']:.3f}"
        label_m = f"Male (n={reg_m['n_pos']}, {p_str})"
        ax.plot(x_fit, prob_fit_m, '--', color='#2980B9', linewidth=2.5, label=label_m)

    # Plot Female (Red)
    if stats_f is not None and not np.isnan(reg_f['coef']):
        # Points with error bars
        ax.errorbar(stats_f['age_mid'], stats_f['prevalence'], yerr=stats_f['ci_error'],
                    fmt='s', markersize=8, capsize=4, capthick=1.5,
                    color='#E74C3C', ecolor='#E74C3C', alpha=0.7)

        # Fitted line (convert from log odds to probability)
        log_odds_fit_f = reg_f['intercept'] + reg_f['coef'] * x_fit
        prob_fit_f = 100 / (1 + np.exp(-log_odds_fit_f))

        p_str = f"p={reg_f['p']:.2e}" if reg_f['p'] < 0.001 else f"p={reg_f['p']:.3f}"
        label_f = f"Female (n={reg_f['n_pos']}, {p_str})"
        ax.plot(x_fit, prob_fit_f, '--', color='#E74C3C', linewidth=2.5, label=label_f)

    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Prevalence (%)', fontsize=12)
    ax.set_title(f'{condition}\nPrevalence by Age, Stratified by Gender', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return reg_m, reg_f


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
print("\nGenerating gender-stratified prevalence plots...")
results = []

for _, row in valid_df.iterrows():
    cond = row['condition']
    fname = f"prev_gender_{cond.replace(' ', '_').replace('/', '_')}.png"
    fpath = os.path.join(OUT, fname)

    try:
        reg_m, reg_f = plot_gender_stratified_prevalence(df, cond, fpath)
        print(f"  Saved: {fname}")

        results.append({
            'condition': cond,
            'n_male': row['n_male'],
            'n_female': row['n_female'],
            'coef_male': reg_m['coef'],
            'p_male': reg_m['p'],
            'coef_female': reg_f['coef'],
            'p_female': reg_f['p'],
            'OR10_male': np.exp(reg_m['coef'] * 10) if not np.isnan(reg_m['coef']) else np.nan,
            'OR10_female': np.exp(reg_f['coef'] * 10) if not np.isnan(reg_f['coef']) else np.nan
        })
    except Exception as e:
        print(f"  Error with {cond}: {e}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT, 'gender_prevalence_results.csv'), index=False)
print(f"\nSaved results: gender_prevalence_results.csv")

# Create summary plot
print("\nCreating summary comparison plot...")

# Find conditions where gender effect differs most
results_df['coef_diff'] = abs(results_df['coef_male'] - results_df['coef_female'])
top_diff = results_df.dropna().nlargest(12, 'coef_diff')

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for i, (_, row) in enumerate(top_diff.iterrows()):
    if i >= 12:
        break
    cond = row['condition']
    ax = axes[i]

    stats_m = calc_prevalence_by_age_gender(df, cond, 1)
    stats_f = calc_prevalence_by_age_gender(df, cond, 0)

    x_fit = np.linspace(40, 70, 100)

    if stats_m is not None and not np.isnan(row['coef_male']):
        ax.errorbar(stats_m['age_mid'], stats_m['prevalence'], yerr=stats_m['ci_error'],
                    fmt='o', markersize=6, capsize=3, color='#2980B9', alpha=0.6)
        log_odds_m = reg_m['intercept'] + row['coef_male'] * x_fit
        prob_m = 100 / (1 + np.exp(-log_odds_m))
        p_m = f"p={row['p_male']:.1e}" if row['p_male'] < 0.01 else f"p={row['p_male']:.2f}"
        ax.plot(x_fit, prob_m, '--', color='#2980B9', lw=2, label=f"M ({p_m})")

    if stats_f is not None and not np.isnan(row['coef_female']):
        ax.errorbar(stats_f['age_mid'], stats_f['prevalence'], yerr=stats_f['ci_error'],
                    fmt='s', markersize=6, capsize=3, color='#E74C3C', alpha=0.6)
        log_odds_f = reg_f['intercept'] + row['coef_female'] * x_fit
        prob_f = 100 / (1 + np.exp(-log_odds_f))
        p_f = f"p={row['p_female']:.1e}" if row['p_female'] < 0.01 else f"p={row['p_female']:.2f}"
        ax.plot(x_fit, prob_f, '--', color='#E74C3C', lw=2, label=f"F ({p_f})")

    ax.set_title(cond, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Age', fontsize=9)
    ax.set_ylabel('Prevalence (%)', fontsize=9)
    ax.set_ylim(bottom=0)

fig.suptitle('Conditions with Largest Gender Differences in Age Effect\n(Blue=Male, Red=Female, dashed=fitted line)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'summary_gender_prevalence_differences.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: summary_gender_prevalence_differences.png")

print("\n" + "="*50)
print("DONE!")
print("="*50)
print(f"Output folder: {OUT}")
print(f"Individual plots: {len(results)} conditions")
print(f"Summary plot: summary_gender_prevalence_differences.png")
print(f"Results CSV: gender_prevalence_results.csv")
