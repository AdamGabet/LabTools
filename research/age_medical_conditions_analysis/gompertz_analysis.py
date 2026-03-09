#!/usr/bin/env python
"""
Gompertz CDF fitting for age-medical conditions prevalence.

Instead of fitting noisy log-odds slopes, we fit the Gompertz cumulative
distribution directly to prevalence data P(t), then derive clean incidence
curves I(t) as the analytic derivative.

Model:
    P(t) = phi * (1 - exp(-(h0/alpha) * (exp(alpha*(t - t0)) - 1)))

Parameters:
    phi   - susceptibility ceiling (max fraction that will ever get the disease)
    alpha - aging rate (exponential growth of hazard, typically 0.03-0.08)
    h0    - baseline hazard at reference age t0

Incidence (derivative):
    I(t) = phi * h0 * exp(alpha*(t-t0)) * exp(-(h0/alpha)*(exp(alpha*(t-t0)) - 1))
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')
import os
import sys

sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
from body_system_loader.load_feature_df import load_body_system_df

OUT = '/home/adamgab/PycharmProjects/LabTools/research/age_medical_conditions_analysis/gompertz'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

T0 = 40  # reference age (left edge of our window)

# Use 2-year bins for more data points
AGE_BINS = list(range(40, 72, 2))  # [40, 42, 44, ..., 70]


# ── Gompertz model ──────────────────────────────────────────────────────

def gompertz_prevalence(t, phi, alpha, h0):
    """Gompertz CDF: probability of having condition by age t."""
    dt = t - T0
    exponent = -(h0 / alpha) * (np.exp(alpha * dt) - 1)
    return phi * (1.0 - np.exp(exponent))


def gompertz_incidence(t, phi, alpha, h0):
    """Derivative of Gompertz CDF: instantaneous incidence at age t."""
    dt = t - T0
    hazard = h0 * np.exp(alpha * dt)
    survival = np.exp(-(h0 / alpha) * (np.exp(alpha * dt) - 1))
    return phi * hazard * survival


def gompertz_log_incidence(t, phi, alpha, h0):
    """Log of incidence - the Gompertz law predicts this is linear then drops."""
    inc = gompertz_incidence(t, phi, alpha, h0)
    return np.log(np.maximum(inc, 1e-12))


# ── Prevalence calculation ──────────────────────────────────────────────

def calc_prevalence(df, condition, bins=None):
    """Calculate prevalence in each age bin."""
    if bins is None:
        bins = AGE_BINS
    d = df.copy()
    d['bin'] = pd.cut(d['age'], bins=bins, right=False)
    stats = d.groupby('bin', observed=True).agg({condition: ['sum', 'count']})
    stats.columns = ['pos', 'n']
    stats['prev'] = stats['pos'] / stats['n']
    stats['se'] = np.sqrt(stats['prev'] * (1 - stats['prev']) / stats['n'])
    stats['mid'] = [(i.left + i.right) / 2 for i in stats.index]
    stats['disease_free'] = 1.0 - stats['prev']
    return stats.reset_index()


# ── Fitting ─────────────────────────────────────────────────────────────

def fit_gompertz(ages, prevalences, weights=None):
    """Fit Gompertz CDF to prevalence data.

    Returns dict with fitted parameters and quality metrics, or None on failure.
    """
    # Initial guesses
    phi0 = min(max(prevalences.max() * 1.5, 0.01), 1.0)
    alpha0 = 0.05
    h0_0 = 0.001

    bounds_lo = [0.001, 0.005, 1e-8]
    bounds_hi = [1.0, 0.30, 0.5]

    try:
        popt, pcov = curve_fit(
            gompertz_prevalence, ages, prevalences,
            p0=[phi0, alpha0, h0_0],
            bounds=(bounds_lo, bounds_hi),
            sigma=weights,
            maxfev=10000,
        )
        phi, alpha, h0 = popt

        # Goodness of fit
        fitted = gompertz_prevalence(ages, *popt)
        ss_res = np.sum((prevalences - fitted) ** 2)
        ss_tot = np.sum((prevalences - prevalences.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Standard errors from covariance
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan] * 3

        return {
            'phi': phi, 'alpha': alpha, 'h0': h0,
            'phi_se': perr[0], 'alpha_se': perr[1], 'h0_se': perr[2],
            'r2': r2,
        }
    except (RuntimeError, ValueError):
        return None


# ── Plotting ────────────────────────────────────────────────────────────

def plot_gompertz_fit(df, condition, fit_params, prev_stats, ax_prev, ax_inc):
    """Plot prevalence fit + derived incidence on two axes."""
    phi, alpha, h0 = fit_params['phi'], fit_params['alpha'], fit_params['h0']
    t_fine = np.linspace(40, 70, 200)

    # ── Left panel: Prevalence ──
    # Raw data
    ax_prev.errorbar(
        prev_stats['mid'], prev_stats['prev'] * 100,
        yerr=1.96 * prev_stats['se'] * 100,
        fmt='o', markersize=7, capsize=4, color='#2E86AB',
        ecolor='#2E86AB', alpha=0.8, label='Observed',
    )
    # Fitted curve
    fitted_prev = gompertz_prevalence(t_fine, phi, alpha, h0) * 100
    ax_prev.plot(t_fine, fitted_prev, '-', color='#E74C3C', lw=2.5,
                 label=f'Gompertz fit (R²={fit_params["r2"]:.3f})')

    ax_prev.set_xlabel('Age (years)')
    ax_prev.set_ylabel('Prevalence (%)')
    ax_prev.set_title(f'{condition}\nPrevalence P(t)')
    ax_prev.legend(fontsize=9)
    ax_prev.grid(True, alpha=0.3)

    # Annotate parameters
    txt = (f'φ={phi:.3f} (ceiling={phi*100:.1f}%)\n'
           f'α={alpha:.4f} (aging rate)\n'
           f'h₀={h0:.2e} (baseline hazard)')
    ax_prev.text(0.03, 0.97, txt, transform=ax_prev.transAxes,
                 fontsize=8, va='top', ha='left',
                 bbox=dict(boxstyle='round', fc='wheat', alpha=0.8))

    # ── Right panel: Incidence ──
    incidence = gompertz_incidence(t_fine, phi, alpha, h0) * 1000  # per 1000
    ax_inc.plot(t_fine, incidence, '-', color='#8E44AD', lw=2.5)
    ax_inc.fill_between(t_fine, 0, incidence, alpha=0.15, color='#8E44AD')

    # Also plot empirical incidence (ΔP/Δt) from raw data
    if len(prev_stats) > 1:
        dp = np.diff(prev_stats['prev'].values)
        dt = np.diff(prev_stats['mid'].values)
        empirical_inc = dp / dt * 1000
        mid_points = (prev_stats['mid'].values[:-1] + prev_stats['mid'].values[1:]) / 2
        # Only plot positive incidence
        mask = empirical_inc > 0
        if mask.any():
            ax_inc.scatter(mid_points[mask], empirical_inc[mask],
                           color='#2E86AB', s=50, zorder=5, alpha=0.7,
                           label='Empirical ΔP/Δt')

    # Mark peak incidence age
    peak_idx = np.argmax(incidence)
    peak_age = t_fine[peak_idx]
    peak_val = incidence[peak_idx]
    ax_inc.axvline(peak_age, color='gray', ls='--', alpha=0.5)
    ax_inc.annotate(f'Peak: age {peak_age:.0f}',
                    xy=(peak_age, peak_val), xytext=(10, 10),
                    textcoords='offset points', fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray'))

    ax_inc.set_xlabel('Age (years)')
    ax_inc.set_ylabel('Incidence (per 1000/year)')
    ax_inc.set_title(f'{condition}\nDerived Incidence I(t)')
    ax_inc.legend(fontsize=9)
    ax_inc.grid(True, alpha=0.3)


def plot_gompertz_gender(df, condition, save_path):
    """Create gender-stratified Gompertz fit: 2x2 grid (prev + inc per gender)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t_fine = np.linspace(40, 70, 200)

    gender_map = {1: ('Male', '#2980B9', axes[0, :]),
                  0: ('Female', '#E74C3C', axes[1, :])}
    fit_results = {}

    for gender_val, (label, color, ax_row) in gender_map.items():
        d = df[df['gender'] == gender_val]
        prev = calc_prevalence(d, condition)

        if prev['pos'].sum() < 20 or len(prev[prev['pos'] > 0]) < 3:
            ax_row[0].text(0.5, 0.5, f'{label}: insufficient data',
                           transform=ax_row[0].transAxes, ha='center')
            ax_row[1].text(0.5, 0.5, f'{label}: insufficient data',
                           transform=ax_row[1].transAxes, ha='center')
            continue

        weights = 1.0 / np.maximum(prev['se'].values, 1e-6)
        fit = fit_gompertz(prev['mid'].values, prev['prev'].values, weights)

        if fit is None:
            ax_row[0].text(0.5, 0.5, f'{label}: fit failed',
                           transform=ax_row[0].transAxes, ha='center')
            ax_row[1].text(0.5, 0.5, f'{label}: fit failed',
                           transform=ax_row[1].transAxes, ha='center')
            continue

        fit_results[label] = fit
        plot_gompertz_fit(d, f'{condition} ({label})', fit, prev,
                          ax_row[0], ax_row[1])

    fig.suptitle(f'{condition} — Gompertz Model by Gender', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fit_results


# ── Main ────────────────────────────────────────────────────────────────

print("=" * 60)
print("GOMPERTZ CDF FIT — Age-Medical Conditions")
print("=" * 60)

# Load data
print("\nLoading data...")
age_df = load_body_system_df('Age_Gender_BMI', specific_columns=['age', 'gender'])
mc_df = load_body_system_df('medical_conditions')
df = pd.merge(age_df, mc_df, left_index=True, right_index=True, how='inner')
df = df[(df['age'] >= 40) & (df['age'] <= 70)]
print(f"N = {len(df)}")
print(f"Male: {(df['gender']==1).sum()}, Female: {(df['gender']==0).sum()}")

# Get conditions with enough cases
condition_cols = [c for c in df.columns if c not in ['age', 'gender']]
print(f"Total condition columns: {len(condition_cols)}")

# ── Fit all conditions ──────────────────────────────────────────────────

print("\nFitting Gompertz model to each condition...")
all_results = []

for cond in condition_cols:
    n_pos = df[cond].sum()
    if n_pos < 50:
        continue

    prev = calc_prevalence(df, cond)
    if len(prev[prev['pos'] > 0]) < 3:
        continue

    weights = 1.0 / np.maximum(prev['se'].values, 1e-6)
    fit = fit_gompertz(prev['mid'].values, prev['prev'].values, weights)

    if fit is None:
        continue

    # Derive peak incidence age
    t_fine = np.linspace(40, 70, 1000)
    inc = gompertz_incidence(t_fine, fit['phi'], fit['alpha'], fit['h0'])
    peak_age = t_fine[np.argmax(inc)]

    all_results.append({
        'condition': cond,
        'n_pos': int(n_pos),
        'n': len(df[cond].dropna()),
        'phi': fit['phi'],
        'alpha': fit['alpha'],
        'h0': fit['h0'],
        'r2': fit['r2'],
        'peak_incidence_age': peak_age,
        'max_prevalence_pct': fit['phi'] * 100,
    })

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('r2', ascending=False)
results_df.to_csv(os.path.join(OUT, 'gompertz_results.csv'), index=False)

print(f"\nSuccessfully fitted: {len(results_df)} conditions")
print(f"Good fits (R² > 0.8): {(results_df['r2'] > 0.8).sum()}")
print(f"Moderate fits (0.5 < R² ≤ 0.8): {((results_df['r2'] > 0.5) & (results_df['r2'] <= 0.8)).sum()}")

print("\nTop fits by R²:")
for _, r in results_df.head(15).iterrows():
    print(f"  {r['condition']:30s} R²={r['r2']:.3f}  φ={r['phi']:.3f}  "
          f"α={r['alpha']:.4f}  peak={r['peak_incidence_age']:.0f}y")


# ── Individual condition plots (prevalence + incidence) ─────────────────

print("\nCreating individual Gompertz fit plots...")
good_fits = results_df[results_df['r2'] > 0.5].head(30)

for _, row in good_fits.iterrows():
    cond = row['condition']
    prev = calc_prevalence(df, cond)
    weights = 1.0 / np.maximum(prev['se'].values, 1e-6)
    fit = fit_gompertz(prev['mid'].values, prev['prev'].values, weights)

    if fit is None:
        continue

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_gompertz_fit(df, cond, fit, prev, ax1, ax2)
    plt.tight_layout()
    fname = f"gompertz_{cond.replace(' ', '_').replace('/', '_')}.png"
    fig.savefig(os.path.join(FIG, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  {fname}")


# ── Summary grid: top 12 conditions ────────────────────────────────────

print("\nCreating summary grids...")
top12 = results_df[results_df['r2'] > 0.5].nlargest(12, 'n_pos')

fig, axes = plt.subplots(3, 4, figsize=(22, 15))
axes = axes.flatten()

for i, (_, row) in enumerate(top12.iterrows()):
    cond = row['condition']
    ax = axes[i]
    t_fine = np.linspace(40, 70, 200)

    prev = calc_prevalence(df, cond)
    ax.errorbar(prev['mid'], prev['prev'] * 100, yerr=1.96 * prev['se'] * 100,
                fmt='o', markersize=5, capsize=3, color='#2E86AB', alpha=0.7)

    fitted = gompertz_prevalence(t_fine, row['phi'], row['alpha'], row['h0']) * 100
    ax.plot(t_fine, fitted, '-', color='#E74C3C', lw=2)

    ax.set_title(f"{cond}\nR²={row['r2']:.3f}, φ={row['phi']:.2f}", fontsize=9)
    ax.set_xlabel('Age', fontsize=8)
    ax.set_ylabel('Prevalence (%)', fontsize=8)
    ax.grid(True, alpha=0.3)

for i in range(len(top12), len(axes)):
    axes[i].set_visible(False)

fig.suptitle('Gompertz CDF Fits — Top 12 Conditions by Sample Size\n'
             '(Blue dots = observed, Red line = Gompertz fit)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'summary_prevalence_fits.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  summary_prevalence_fits.png")

# ── Summary grid: derived incidence curves ──────────────────────────────

fig, axes = plt.subplots(3, 4, figsize=(22, 15))
axes = axes.flatten()

for i, (_, row) in enumerate(top12.iterrows()):
    cond = row['condition']
    ax = axes[i]
    t_fine = np.linspace(40, 70, 200)

    inc = gompertz_incidence(t_fine, row['phi'], row['alpha'], row['h0']) * 1000
    ax.plot(t_fine, inc, '-', color='#8E44AD', lw=2)
    ax.fill_between(t_fine, 0, inc, alpha=0.15, color='#8E44AD')

    peak_idx = np.argmax(inc)
    ax.axvline(t_fine[peak_idx], color='gray', ls='--', alpha=0.5)

    ax.set_title(f"{cond}\npeak={t_fine[peak_idx]:.0f}y, α={row['alpha']:.3f}",
                 fontsize=9)
    ax.set_xlabel('Age', fontsize=8)
    ax.set_ylabel('Inc. (per 1000/y)', fontsize=8)
    ax.grid(True, alpha=0.3)

for i in range(len(top12), len(axes)):
    axes[i].set_visible(False)

fig.suptitle('Derived Incidence Curves from Gompertz Model — Top 12 Conditions\n'
             '(Analytic derivative of fitted prevalence)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'summary_incidence_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  summary_incidence_curves.png")


# ── Gender-stratified fits for top conditions ───────────────────────────

print("\nCreating gender-stratified Gompertz plots...")
gender_results = []
top_gender = results_df[results_df['r2'] > 0.5].nlargest(20, 'n_pos')

for _, row in top_gender.iterrows():
    cond = row['condition']
    fname = f"gender_gompertz_{cond.replace(' ', '_').replace('/', '_')}.png"
    fpath = os.path.join(FIG, fname)

    fits = plot_gompertz_gender(df, cond, fpath)
    print(f"  {fname}")

    for label, fit in fits.items():
        t_fine = np.linspace(40, 70, 1000)
        inc = gompertz_incidence(t_fine, fit['phi'], fit['alpha'], fit['h0'])
        gender_results.append({
            'condition': cond,
            'gender': label,
            'phi': fit['phi'],
            'alpha': fit['alpha'],
            'h0': fit['h0'],
            'r2': fit['r2'],
            'peak_incidence_age': t_fine[np.argmax(inc)],
        })

gender_df = pd.DataFrame(gender_results)
gender_df.to_csv(os.path.join(OUT, 'gompertz_gender_results.csv'), index=False)


# ── Overlay: Male vs Female incidence for top conditions ────────────────

print("\nCreating male vs female incidence overlay...")
overlay_conditions = gender_df.groupby('condition').filter(
    lambda x: len(x) == 2  # both genders fitted
)['condition'].unique()[:12]

fig, axes = plt.subplots(3, 4, figsize=(22, 15))
axes = axes.flatten()

for i, cond in enumerate(overlay_conditions):
    if i >= 12:
        break
    ax = axes[i]
    t_fine = np.linspace(40, 70, 200)

    cond_fits = gender_df[gender_df['condition'] == cond]
    for _, gfit in cond_fits.iterrows():
        color = '#2980B9' if gfit['gender'] == 'Male' else '#E74C3C'
        inc = gompertz_incidence(t_fine, gfit['phi'], gfit['alpha'], gfit['h0']) * 1000
        ax.plot(t_fine, inc, '-', color=color, lw=2,
                label=f"{gfit['gender']} (α={gfit['alpha']:.3f})")
        ax.fill_between(t_fine, 0, inc, alpha=0.1, color=color)

    ax.set_title(cond, fontsize=10, fontweight='bold')
    ax.set_xlabel('Age', fontsize=8)
    ax.set_ylabel('Inc. (per 1000/y)', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for i in range(len(overlay_conditions), len(axes)):
    axes[i].set_visible(False)

fig.suptitle('Male vs Female Incidence Curves (Gompertz Model)\n'
             'Blue=Male, Red=Female',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'gender_incidence_overlay.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  gender_incidence_overlay.png")


# ── Parameter comparison plot ───────────────────────────────────────────

print("\nCreating parameter comparison plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

good = results_df[results_df['r2'] > 0.5].copy()

# Alpha distribution
axes[0].hist(good['alpha'], bins=20, color='#3498DB', alpha=0.7, edgecolor='black')
axes[0].axvline(0.03, color='red', ls='--', label='α=0.03 (slow aging)')
axes[0].axvline(0.08, color='red', ls=':', label='α=0.08 (fast aging)')
axes[0].set_xlabel('α (aging rate)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Aging Rate α')
axes[0].legend(fontsize=9)

# Phi vs alpha scatter
sc = axes[1].scatter(good['alpha'], good['phi'], c=good['peak_incidence_age'],
                     cmap='viridis', s=60, alpha=0.7, edgecolor='black', lw=0.5)
plt.colorbar(sc, ax=axes[1], label='Peak incidence age')
axes[1].set_xlabel('α (aging rate)')
axes[1].set_ylabel('φ (susceptibility)')
axes[1].set_title('Aging Rate vs Susceptibility')

# Peak incidence age
axes[2].hist(good['peak_incidence_age'], bins=15, color='#8E44AD', alpha=0.7,
             edgecolor='black')
axes[2].set_xlabel('Peak incidence age')
axes[2].set_ylabel('Count')
axes[2].set_title('When Do Conditions Peak?')

fig.suptitle('Gompertz Model Parameters Across Conditions',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(FIG, 'parameter_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  parameter_distributions.png")


# ── Print summary ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print(f"\nFitted {len(results_df)} conditions total")
print(f"Good fits (R² > 0.8): {(results_df['r2'] > 0.8).sum()}")

print("\nFastest aging (highest α):")
for _, r in results_df.nlargest(10, 'alpha').iterrows():
    print(f"  {r['condition']:30s}  α={r['alpha']:.4f}  peak={r['peak_incidence_age']:.0f}y  "
          f"ceiling={r['phi']*100:.1f}%")

print("\nSlowest aging (lowest α, still good fit):")
for _, r in results_df[results_df['r2'] > 0.5].nsmallest(10, 'alpha').iterrows():
    print(f"  {r['condition']:30s}  α={r['alpha']:.4f}  peak={r['peak_incidence_age']:.0f}y  "
          f"ceiling={r['phi']*100:.1f}%")

print("\nHighest susceptibility (φ):")
for _, r in results_df.nlargest(10, 'phi').iterrows():
    print(f"  {r['condition']:30s}  φ={r['phi']:.3f} ({r['phi']*100:.1f}% ceiling)  "
          f"α={r['alpha']:.4f}")

print(f"\nOutput: {OUT}")
print(f"Figures: {FIG}")
print("Done!")
